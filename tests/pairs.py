"""The paired oracle: an R expression and its Python counterpart, registered together.

``tools/oracle/`` writes fixtures and ``tests/`` reads them, and nothing between
the two says which R expression a given Python function is supposed to equal.
That pairing lives only in a contributor's head: ``cpp("pnbd_nocov_LL_ind")``
appears in a 475-line R script, :func:`clvtools.pnbd.aggregate.log_likelihood_ind`
is asserted in a test module, and no artefact relates them. Drop a column from a
generator, or point an assertion at the neighbouring column, and the suite stays
green.

A *pair* is that missing artefact. It carries the R snippet and the Python
callable in one declaration, the inputs both are evaluated at, and the tolerance
they must agree to::

    @pair(
        id="pnbd.nocov.LL_ind",
        spec="M-03",
        r='cpp("pnbd_nocov_LL_ind")(vLogparams = log(p), vX = x, vT_x = tx, vT_cal = Tc)',
        ...
    )
    def _(p):
        return aggregate.log_likelihood_ind(X, TX, TC, *p)

The R side is a string because R is the foreign language; the Python side stays
a real function, so ruff and the complexity gate still see it. Note what becomes
visible by writing them adjacently: ``log(p)`` on one side and natural-scale
parameters on the other is the log-scale convention that CLAUDE.md lists first
among the traps that have already cost time. It is now readable as a pair rather
than asserted in two files.

Two runners consume one registry:

**replay**, the default, needs no R. The Python side is evaluated and compared
to values recorded under ``tests/fixtures/pairs/``. This is the guarantee the
committed fixtures already give -- except that the recording is generated *from
the registry*, so a pair and its expectation cannot drift apart.

**live** (``pytest --oracle-live``) evaluates the R snippets in a real R session
and compares three ways: Python against live R, which is the differential test;
the recording against live R, which is the staleness gate that nothing in this
repository currently has; and each prelude's input vectors against the committed
copies the Python side reads, so that "both implementations were fed the same
numbers" is checked rather than assumed.

Re-record with ``tools/oracle/record_pairs.py``. Recording is deliberately not a
pytest mode: a suite that can rewrite its own expectations is one bad flag away
from proving nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
PAIRS = Path(__file__).resolve().parent / "fixtures" / "pairs"
RUNNER = ROOT / "tools" / "oracle" / "run_pairs.R"
RLIB = ROOT / ".Rlib"

#: Joins a pair's id to an input's name to key one evaluation. ``@`` cannot
#: appear in either, so the key splits back unambiguously.
SEP = "@"


@dataclass(frozen=True)
class Pair:
    """One quantity, expressed in both languages.

    :param id: dotted name, unique across the registry
    :param family: which recording file this pair belongs to
    :param prelude: the named R environment the snippet is evaluated in
    :param spec: the ``docs/spec.md`` item this pins
    :param r: an R expression, evaluated with ``p`` bound to the input vector
    :param py: the Python side, called with the same vector
    :param inputs: name -> parameter vector, on the *natural* scale
    :param tol: the largest relative error the two may differ by
    :param undefined_at: inputs where the quantity does not exist
    """

    id: str
    family: str
    prelude: str
    spec: str
    r: str
    py: Callable[[Sequence[float]], Any]
    inputs: Mapping[str, Sequence[float]]
    tol: float
    undefined_at: frozenset[str] = frozenset()

    def defined_at(self, input_name: str) -> bool:
        """Whether the two sides are expected to agree numerically here."""
        return input_name not in self.undefined_at


_REGISTRY: dict[str, Pair] = {}


def pair(
    *,
    id: str,  # noqa: A002 - `id` is the field name; the builtin is not wanted here
    family: str,
    prelude: str,
    spec: str,
    r: str,
    inputs: Mapping[str, Sequence[float]],
    tol: float,
    undefined_at: Sequence[str] = (),
) -> Callable[[Callable[[Sequence[float]], Any]], Callable[[Sequence[float]], Any]]:
    """Register the decorated function as the Python side of ``r``.

    The function is returned unchanged, so a pair is still an ordinary callable
    that can be tested or debugged on its own.

    ``undefined_at`` names the inputs where the quantity does not exist -- the
    Pareto/NBD's CET divides by ``s - 1``, so ``s = 1`` has no value to compare.
    Naming them does not skip them. The two implementations disagree about how
    to say "undefined": CLVTools returns ``NaN``, this package raises, and the
    pair asserts *both* halves of that, which is a stronger statement than
    leaving the input out of the grid.
    """

    def register(fn: Callable[[Sequence[float]], Any]) -> Callable[[Sequence[float]], Any]:
        if id in _REGISTRY:
            raise ValueError(f"duplicate pair id: {id}")
        unknown = set(undefined_at) - set(inputs)
        if unknown:
            raise ValueError(
                f"{id}: undefined_at names inputs that do not exist: {unknown}"
            )
        _REGISTRY[id] = Pair(
            id=id, family=family, prelude=prelude, spec=spec,
            r=r, py=fn, inputs=inputs, tol=tol,
            undefined_at=frozenset(undefined_at),
        )
        return fn

    return register


def registry() -> tuple[Pair, ...]:
    """Every registered pair, in declaration order."""
    return tuple(_REGISTRY.values())


def cases() -> tuple[tuple[Pair, str], ...]:
    """Every ``(pair, input name)`` the runners evaluate."""
    return tuple((p, name) for p in registry() for name in p.inputs)


def key(pair_id: str, input_name: str) -> str:
    """The recording key for one evaluation.

    >>> key("pnbd.nocov.LL_ind", "mle")
    'pnbd.nocov.LL_ind@mle'
    """
    return f"{pair_id}{SEP}{input_name}"


def max_rel_error(got: Any, want: Any) -> float:
    """Largest elementwise error, relative where the values are large.

    The denominator is ``max(1, |want|)`` rather than ``|want|``: these are
    log-likelihoods and probabilities side by side, and a quantity that is
    legitimately near zero would otherwise report a vast relative error for an
    absolute one of 1e-16.

    >>> max_rel_error([1.0, 2.0], [1.0, 2.0])
    0.0
    >>> round(max_rel_error([100.0], [100.001]), 8)
    1e-05
    """
    a = np.atleast_1d(np.asarray(got, dtype=float))
    b = np.atleast_1d(np.asarray(want, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: python {a.shape}, oracle {b.shape}")
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


# -- The R side ---------------------------------------------------------------


def spec_source(pairs: Sequence[Pair]) -> str:
    """The R source ``run_pairs.R`` reads: the inputs, then one entry per case.

    Generating R rather than JSON keeps a JSON parser out of the runner -- the
    generators in ``tools/oracle/`` hand-roll a writer and have never needed a
    reader -- and makes a malformed snippet fail at ``source()`` with R's own
    parse error pointing at the line, rather than somewhere inside an eval.
    """
    wanted = {p.id for p in pairs}
    lines = ["# Generated by tests/pairs.py. Do not edit.", "CASES <- list("]
    entries = [
        f'  list(id = "{p.id}", input = "{name}", prelude = "{p.prelude}",\n'
        f"       p = c({', '.join(repr(float(v)) for v in p.inputs[name])}),\n"
        f"       expr = quote({p.r}))"
        for p, name in cases()
        if p.id in wanted
    ]
    lines.append(",\n".join(entries))
    lines.append(")")
    return "\n".join(lines) + "\n"


def run_r(pairs: Sequence[Pair], tmp: Path) -> dict[str, Any]:
    """Evaluate every snippet in one R session, and return what it wrote.

    One process for the whole registry, not one per case: R's startup dominates
    an expression that takes microseconds, and the prelude fits a model.
    """
    spec = tmp / "spec.R"
    out = tmp / "r_side.json"
    spec.write_text(spec_source(pairs), encoding="utf-8")
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is not on PATH; see tools/setup_oracle.sh")
    result = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        [rscript, str(RUNNER), str(spec), str(out)],
        capture_output=True, text=True, cwd=ROOT,
        env={"R_LIBS": str(RLIB), "PATH": _path()},
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"the R side failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(out.read_text(encoding="utf-8"))


def _path() -> str:
    """``PATH`` for the R subprocess, taken from this process."""
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


def r_is_available() -> bool:
    """Whether a live run is possible here: ``Rscript`` plus CLVTools in ``.Rlib``."""
    return shutil.which("Rscript") is not None and (RLIB / "CLVTools").is_dir()


# -- The recordings -----------------------------------------------------------


def recording(family: str) -> dict[str, Any]:
    """The committed values for one family."""
    return json.loads((PAIRS / f"{family}.json").read_text(encoding="utf-8"))


def prelude_inputs(name: str) -> dict[str, NDArray[np.float64]]:
    """The committed copy of what a prelude feeds both implementations."""
    raw = json.loads((PAIRS / f"_prelude_{name}.json").read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=float) for k, v in raw["inputs"].items()}
