#!/usr/bin/env python
"""Where the time goes, as a report that can be pasted into docs/performance.md.

``tools/benchmark.py`` reports wall-clock in the shape of the paper's Appendix
B. This is the complementary question: for the four paths ``docs/performance.md``
covers -- building and describing a data set, ``fit_pnbd``, ``fit_pnbd_staticcov``
and **one** evaluation of the time-varying covariate likelihood -- which
functions account for the time, and how often are they called.

    uv run tools/profile.py                          # all four, markdown
    uv run tools/profile.py --paths pnbd dyncov      # a subset
    uv run tools/profile.py --top 20                 # deeper tables

The output is markdown so it can go straight into ``docs/performance.md``, and
it is built to be **diffed between versions**. That is why the tables carry call
counts and shares of ``tottime`` rather than seconds: a call count is a property
of the code and moves only when the code does, while seconds move with the
machine, the interpreter and the weather. The one seconds figure per path sits
on its own line above the table, where a diff can ignore it.

Two runs of unchanged code give **identical call counts** for any function
that appears in both. What is not stable is the table's membership: the rows
are selected by share, so near the ``--top`` cut a row can drop out entirely
and another take its place, which in a diff reads as a function appearing from
nowhere. Measured across two runs here, that happened to three of the twenty
pandas internals in the ``summary()`` path and to none of this package's own
functions, which are not close enough to the cut to move.

So read a diff by the counts, and read a function appearing or disappearing as
noise unless its count is large: a share that moved by a tenth means nothing,
one that moved by ten points means something, and a call count that moved at
all means the code did.

Nothing here is asserted. This is a report, not a gate: it is never imported by
a test and never runs in CI. The invariants that *are* gated -- vectorisation,
evaluations per fit, cost per customer -- are counted deterministically in
``tests/test_performance.py``, which looks at no clock at all.

The fit itself of the time-varying model is 13.5 minutes and is not run here;
one likelihood evaluation is, at CLVTools' own fitted parameters.
"""

from __future__ import annotations

import argparse
import platform
import pstats
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import scipy

from clvtools import (
    ClvData,
    ClvDataDynCov,
    ClvDataStaticCov,
    load_apparel_dyn_cov,
    load_apparel_static_cov,
    load_apparel_trans,
    load_cdnow,
)
from clvtools.pnbd import fit_pnbd, fit_pnbd_staticcov
from clvtools.pnbd.dyncov import log_likelihood as dyncov_log_likelihood

ROOT = Path(__file__).resolve().parent.parent

#: The apparel cohort of S6.2.1: 104 weeks of estimation, 600 customers.
ESTIMATION_WEEKS = 104

#: CLVTools' own fitted time-varying parameters, as docs/paper.md records them.
#: The likelihood there is -5752.9367; this evaluates it once.
DYNCOV_FIT = {
    "r": 1.977706,
    "alpha": 115.177940,
    "s": 2.012683,
    "beta": 158.181797,
    "gamma_life": [-2.482678, -0.512544, 0.505730],
    "gamma_trans": [0.718314, 0.264898, 0.613721],
}


def _import_cprofile() -> ModuleType:
    """Import the standard library's ``cProfile`` past this file's own name.

    ``cProfile`` starts with ``import profile``, and running a script puts its
    directory first on ``sys.path`` -- so from ``uv run tools/profile.py`` the
    name ``profile`` resolves to *this* module, which is then executed a second
    time in the middle of ``cProfile``'s own import. Confirmed to fail with
    ``AttributeError: partially initialized module 'cProfile' has no attribute
    'Profile'``.

    Dropping the script's directory for the duration of the import is what makes
    ``tools/profile.py`` -- the name ``docs/backlog.md`` item 8 asks for -- safe.
    ``pstats`` needs no such care; it never imports ``profile``.
    """
    here = Path(__file__).resolve().parent
    shadowing = [p for p in sys.path if p and Path(p).resolve() == here]
    for entry in shadowing:
        sys.path.remove(entry)
    try:
        import cProfile
    finally:
        sys.path[:0] = shadowing
    return cProfile


cprofile = _import_cprofile()


# ---------------------------------------------------------------------------
# The paths, each set up once and then run three times: warm, timed, profiled.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfiledPath:
    """One profiled path.

    ``unit`` names the function whose call count is the natural denominator for
    the path -- the likelihood, for a fit. Its count turns raw call counts into
    calls *per likelihood evaluation*, which is the figure that stays put when
    the optimiser takes a different number of steps.
    """

    name: str
    title: str
    setup: Callable[[], tuple[Callable[[], object], str]]
    unit: str | None = None
    unit_label: str = "evaluation"


def _apparel() -> ClvData:
    return ClvData(load_apparel_trans(), estimation_split=ESTIMATION_WEEKS)


def _setup_summary() -> tuple[Callable[[], object], str]:
    """S6.1: build a ``ClvData`` over the CDNOW log and describe it."""
    transactions = load_cdnow()
    customers = transactions["Id"].nunique()

    def call() -> object:
        return ClvData(transactions).summary()

    return call, f"{customers:,} customers; construction and `summary()` together"


def _setup_pnbd() -> tuple[Callable[[], object], str]:
    """S6.2.1: the plain Pareto/NBD on the apparel cohort."""
    cbs = _apparel().customer_summary()
    x, t_x, T = (cbs[column].to_numpy(dtype=float) for column in ("x", "t_x", "T"))

    def call() -> object:
        return fit_pnbd(x, t_x, T, hessian=False)

    return call, f"{len(x):,} customers, no Hessian"


def _setup_pnbd_staticcov() -> tuple[Callable[[], object], str]:
    """S6.4.1: two time-invariant covariates on both processes."""
    names = ["Gender", "Channel"]
    data = ClvDataStaticCov(
        _apparel(), load_apparel_static_cov(),
        names_cov_life=names, names_cov_trans=names,
    )

    def call() -> object:
        return fit_pnbd_staticcov(data, hessian=True)

    return call, (
        f"{len(data.customer_summary()):,} customers, with Hessian; the "
        "evaluation count includes the differencing the Hessian costs, which "
        "is why it exceeds the optimiser's own"
    )


def _setup_dyncov() -> tuple[Callable[[], object], str]:
    """S6.4.2: **one** evaluation of the time-varying covariate likelihood.

    Never the fit. That is 1,870 evaluations and 13.5 minutes, and it has a
    home already: the ``dyncov_fit`` marker, run nightly.

    Building the walks is setup rather than part of the path -- it happens once
    per fit, not once per evaluation -- so it is timed and reported separately.
    """
    names = ["High.Season", "Gender", "Channel"]
    dynamic = ClvDataDynCov(
        _apparel(), load_apparel_dyn_cov(),
        names_cov_life=names, names_cov_trans=names,
    )
    started = time.perf_counter()
    walks = dynamic.walks()
    built = time.perf_counter() - started

    def call() -> object:
        return dyncov_log_likelihood(walks, **DYNCOV_FIT)

    return call, (
        f"{walks.n_customers:,} customers, one evaluation at CLVTools' fitted "
        f"parameters; `build_walks` is setup, {built:.3f} s once per fit"
    )


PATHS = (
    ProfiledPath("summary", "`ClvData` + `summary()`", _setup_summary,
                 unit_label="call"),
    ProfiledPath("pnbd", "`fit_pnbd`", _setup_pnbd,
                 unit="clvtools/pnbd/aggregate.py:log_likelihood",
                 unit_label="likelihood evaluation"),
    ProfiledPath("pnbd-staticcov", "`fit_pnbd_staticcov`", _setup_pnbd_staticcov,
                 unit="clvtools/pnbd/staticcov.py:log_likelihood",
                 unit_label="likelihood evaluation"),
    ProfiledPath("dyncov", "dyncov `log_likelihood`", _setup_dyncov),
)


# ---------------------------------------------------------------------------
# Turning a profile into a table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One function in a profile: what it is, how often, and how much."""

    label: str
    calls: int
    tottime: float


def _label(filename: str, funcname: str) -> str:
    """A short, machine-independent name for a profiled function.

    Absolute paths and line numbers both defeat a diff -- the first differs
    between machines, the second between edits that changed nothing about the
    profile. What survives is the module's path within its distribution and the
    function's name.

    >>> _label(str(ROOT / "src" / "clvtools" / "special.py"), "hyp2f1_ratio")
    'clvtools/special.py:hyp2f1_ratio'
    >>> _label("~", "<built-in method scipy.special._ufuncs.hyp2f1>")
    'scipy.special._ufuncs.hyp2f1'
    >>> _label("<string>", "__init__")
    '<string>:__init__'
    """
    if not filename or filename == "~":
        return funcname.strip("<>").removeprefix("built-in method ")
    if filename.startswith("<"):
        return f"{filename}:{funcname}"
    path = Path(filename)
    parts = path.parts
    for anchor in ("site-packages", "src"):
        if anchor in parts:
            tail = parts[len(parts) - parts[::-1].index(anchor):]
            return f"{'/'.join(tail)}:{funcname}"
    if path.is_relative_to(ROOT):
        return f"{path.relative_to(ROOT)}:{funcname}"
    return f"{path.name}:{funcname}"


def _rows(stats: pstats.Stats) -> tuple[list[Row], float]:
    """Every function in ``stats``, hottest first, with the total ``tottime``.

    Entries sharing a label are summed: dropping line numbers can put two
    generator expressions from the same module under one name, and one row is a
    truer report of that than two identical ones a diff cannot tell apart.

    Rows are ordered by the share **as printed** -- rounded to a tenth of a
    percent -- and then by name. Ordering by the raw ``tottime`` would put two
    rows showing ``0.7%`` in whichever order this particular run measured them,
    so a diff between two runs of identical code shows a swap that means
    nothing. Rounding first makes the order a function of what the reader sees.
    """
    calls_by_label: dict[str, int] = {}
    tottime_by_label: dict[str, float] = {}
    # `pstats` keys are (filename, lineno, funcname); the values are
    # (primitive calls, total calls, tottime, cumtime, callers).
    for (filename, _, funcname), (_, calls, tottime, _, _) in stats.stats.items():
        if "_lsprof" in funcname:
            continue
        label = _label(filename, funcname)
        calls_by_label[label] = calls_by_label.get(label, 0) + int(calls)
        tottime_by_label[label] = tottime_by_label.get(label, 0.0) + float(tottime)
    rows = [
        Row(label, calls_by_label[label], tottime_by_label[label])
        for label in calls_by_label
    ]
    total = sum(row.tottime for row in rows)
    return sorted(rows, key=lambda row: (-_share(row.tottime, total), row.label)), total


def _share(tottime: float, total: float) -> float:
    """``tottime`` as a percentage of ``total``, to the tenth printed.

    >>> _share(0.5, 2.0), _share(1.0, 0.0)
    (25.0, 0.0)
    """
    return round(100 * tottime / total, 1) if total else 0.0


def _per_unit(calls: int, units: int) -> str:
    """``calls`` per unit, exact when it divides and one decimal when it does not.

    >>> _per_unit(580, 290), _per_unit(330, 165), _per_unit(777, 165)
    ('2', '2', '4.7')
    """
    if calls % units == 0:
        return f"{calls // units:,}"
    return f"{calls / units:,.1f}"


def _table(rows: list[Row], total: float, top: int, units: int, label: str) -> str:
    """The hottest ``top`` functions as a markdown table."""
    per_unit = units != 1
    head = ["Function", "Calls"] + ([f"Per {label}"] if per_unit else []) + ["tottime"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows[:top]:
        cells = [f"`{row.label}`", f"{row.calls:,}"]
        if per_unit:
            cells.append(_per_unit(row.calls, units))
        cells.append(f"{_share(row.tottime, total):.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _units(rows: list[Row], unit: str | None) -> int:
    """How many times the path's denominator function ran; 1 if it has none."""
    if unit is None:
        return 1
    for row in rows:
        if row.label == unit:
            return row.calls
    # A renamed or inlined likelihood should say so rather than silently
    # reporting per-call figures as if they were per-evaluation.
    print(f"<!-- warning: {unit} never ran; counts below are totals -->\n")
    return 1


def _wall(call: Callable[[], object], repeats: int) -> float:
    """Median unprofiled seconds, as ``tools/benchmark.py`` reports them."""
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def _header() -> str:
    """Which machine and which library versions produced the numbers below."""
    when = datetime.now(tz=UTC).date()
    return "\n".join([
        f"Generated by `uv run tools/profile.py` on {when}.",
        "",
        (f"{platform.platform(terse=True)}, {platform.machine()}, "
         f"{platform.python_implementation()} {platform.python_version()}, "
         f"numpy {np.__version__}, scipy {scipy.__version__}, "
         f"pandas {pd.__version__}."),
        "",
        "`tottime` is the share of the profiled run's total self-time. Call",
        "counts and shares are what to diff between versions; the seconds on",
        "each path's own line move with the machine. cProfile charges every",
        "Python-level call, so a profiled run is slower than an unprofiled one",
        "-- by half on the vectorised paths and by several times on the dyncov",
        "likelihood, which is exactly the finding.",
    ])


def _report(path: ProfiledPath, top: int, repeats: int) -> None:
    """Profile one path and print its section."""
    call, note = path.setup()
    call()  # warm: first-touch caching is not what this measures
    wall = _wall(call, repeats)

    profiler = cprofile.Profile()
    profiler.enable()
    call()
    profiler.disable()
    rows, total = _rows(pstats.Stats(profiler))
    units = _units(rows, path.unit)

    print(f"### {path.title}\n")
    print(f"{note}.\n")
    counted = f"{units:,} {path.unit_label}s" if units != 1 else f"1 {path.unit_label}"
    print(f"{counted}, {wall:.3f} s unprofiled (median of {repeats}), "
          f"{total:.3f} s profiled.\n")
    print(_table(rows, total, top, units, path.unit_label) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", default=[p.name for p in PATHS],
                        choices=[p.name for p in PATHS],
                        help="which paths to profile (default: all)")
    parser.add_argument("--top", type=int, default=12,
                        help="functions per table (default: 12)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="unprofiled runs to take the median of (default: 3)")
    args = parser.parse_args()

    print("## Profile\n")
    print(_header() + "\n")
    for path in PATHS:
        if path.name in args.paths:
            _report(path, args.top, args.repeats)


if __name__ == "__main__":
    main()
