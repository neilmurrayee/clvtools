"""Static analysis, run as part of the ordinary suite.

The point of putting the analysers here rather than in a separate ``lint``
command is that there is then only one way to be green. ``uv run pytest`` is
the gate; a change that tangles a function or outgrows a module fails it in the
same breath as a change that gets an equation wrong.

What is enforced, and why each limit is where it is:

``ruff``
    The rule selection and the design limits live in ``pyproject.toml``, next
    to the reasoning for each. The thresholds were measured against this
    codebase rather than taken from defaults, so each sits just above what the
    code needs and trips on a regression.

design limits, measured
    Those thresholds only mean something next to what the code actually
    scores, and that pairing used to live in a comment. It had gone stale in
    two places out of four -- 42 statements recorded against an actual 49, and
    5 returns against an actual 6 -- because nothing re-ran it.
    :class:`TestDesignLimits` re-measures all five from ruff on every run, so
    the recorded numbers are either true or the suite is red.

``ty``
    ``src/clvtools/py.typed`` tells every downstream type checker that the
    annotations in this package are meant to be relied on. That is a promise,
    and this is what keeps it true. Only ``src/`` is checked, because only
    ``src/`` is what ``py.typed`` covers; the three rules that are off, and
    why, are recorded in ``pyproject.toml`` beside the ruff ignores.

annotation coverage
    ``ty`` checks that the annotations which exist are consistent, and
    :meth:`TestTy.test_the_shipped_annotations_resolve` that they evaluate.
    Neither notices a parameter with no annotation at all -- and a bare
    parameter is the one thing ``py.typed`` cannot excuse, because a consumer
    reading the signature gets nothing back. :class:`TestAnnotations` is the
    half that was missing.

module length
    Counted in *code* lines -- docstrings, comments and blanks excluded.
    Roughly 37% of ``src/`` is docstring, deliberately: the docstrings carry the
    paper. A raw line count would measure how well a module is documented and
    call the best-documented ones the worst, which is exactly backwards.

Three functions carry a ``noqa``. Two are for the argument-count limit. Both are the
paper's equations written out -- the GGom/NBD's covariate likelihood, which is
the one family with five model parameters, and the dyncov ``F2`` term, which
runs per customer per likelihood evaluation and so cannot afford a wrapper
object. The third is ``_validate.finished``, whose ``TypeVar`` ruff would
rather see written with PEP 695 type parameters -- the one spelling that does
not resolve under ``get_type_hints()`` on every 3.12 this package supports.
Each says so at the site.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import json
import pkgutil
import re
import subprocess
import sys
import tokenize
import tomllib
import typing
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

import clvtools

pytestmark = pytest.mark.quality

ROOT = Path(__file__).resolve().parent.parent

#: Everything that is ours. ``docs/`` holds the executable case study.
TARGETS = ("src", "tests", "tools", "docs")

#: No module carries more than this many code lines. Which module is currently
#: largest, and by how much it clears the limit, is deliberately not written
#: down here: the previous note named ``pnbd/dyncov.py`` at 455 and
#: ``tests/test_predict.py`` at 553, and by the time anyone checked, the real
#: figures were 527 and 677. :meth:`TestSize.test_the_limit_still_binds` and
#: :meth:`TestSize.test_the_largest_module_keeps_its_distance` compute both
#: ends on every run and name the file in the failure.
MAX_CODE_LINES = 700

#: How much room the largest module must leave under :data:`MAX_CODE_LINES`.
#:
#: ``test_families.py`` reached 697 against the 700 and was split, on the
#: grounds that a gate three lines from tripping is one the next commit trips
#: for no reason. That was a judgement made once, in prose, and then not
#: applied again: ``test_diagnostics.py`` was sitting at 677 when this margin
#: was added, and nothing said so. Splitting it -- the bootstrap half became
#: ``test_bootstrap.py``, which is a different subject anyway -- left
#: ``tests/test_pnbd_dyncov.py`` largest at 658.
MIN_HEADROOM = 25


def code_lines(path: Path) -> int:
    """Lines of actual code: no docstrings, no comments, no blanks.

    >>> code_lines(ROOT / "src" / "clvtools" / "py.typed")
    0
    """
    source = path.read_text(encoding="utf-8")
    total = len(source.splitlines())

    documented = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ) and ast.get_docstring(node, clean=False) is not None:
            expression = node.body[0]
            documented.update(range(expression.lineno, expression.end_lineno + 1))

    comments = sum(
        1
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    )
    blanks = sum(1 for line in source.splitlines() if not line.strip())
    return total - len(documented) - comments - blanks


def _modules() -> list[ModuleType]:
    """Every module in the package, imported.

    >>> "clvtools.pnbd.dyncov" in {m.__name__ for m in _modules()}
    True
    """
    found = [clvtools]
    for info in pkgutil.walk_packages(clvtools.__path__, f"{clvtools.__name__}."):
        found.append(importlib.import_module(info.name))
    return found


def python_files() -> list[Path]:
    """Every Python file the gate covers."""
    return sorted(
        path
        for target in TARGETS
        for path in (ROOT / target).rglob("*.py")
        if "__pycache__" not in path.parts
    )


class TestRuff:
    """Lint, complexity, and the design limits of ``[tool.ruff.lint.pylint]``."""

    def test_reports_nothing(self):
        """``ruff check`` is clean across everything we wrote."""
        result = subprocess.run(  # noqa: S603 - a fixed argv, no shell
            [sys.executable, "-m", "ruff", "check", "--no-cache", *TARGETS],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "ruff reported findings:\n\n"
            f"{result.stdout}{result.stderr}\n"
            "Run `uv run ruff check --fix src tests tools docs` for the "
            "mechanical ones."
        )


class TestTy:
    """The annotations ``py.typed`` promises are usable."""

    def test_reports_nothing(self):
        """``ty check src`` is clean."""
        result = subprocess.run(
            [sys.executable, "-m", "ty", "check", "src"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "ty reported findings:\n\n"
            f"{result.stdout}{result.stderr}\n"
            "Fix the annotation rather than adding a rule to "
            "[tool.ty.rules] -- those three are stub noise, and the notes "
            "beside them say how much of it there is."
        )

    def test_the_shipped_annotations_resolve(self):
        """``typing.get_type_hints()`` works on every public signature.

        A checker reads annotations lazily; ``get_type_hints()`` evaluates
        them, which is what a downstream consumer generating docs or
        validating arguments will do. An annotation naming something imported
        only inside the function body -- which is how the covariate fits and
        ``build_walks`` used to break their import cycles -- passes the first
        and raises ``NameError`` on the second.
        """
        unresolved = {}
        for module in _modules():
            for name, member in vars(module).items():
                if not (inspect.isfunction(member) or inspect.isclass(member)):
                    continue
                if getattr(member, "__module__", None) != module.__name__:
                    continue
                try:
                    typing.get_type_hints(member)
                except Exception as error:
                    unresolved[f"{module.__name__}.{name}"] = (
                        f"{type(error).__name__}: {error}"
                    )
        assert not unresolved, (
            f"these public names carry annotations that do not resolve: "
            f"{unresolved}. Import the name for real rather than inside the "
            "function; see the notes in bgnbd.py and pnbd/dyncov.py."
        )


def public_signatures() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every public module-level function and method under ``src/``, by name.

    Nested functions are excluded: a closure inside a fit is not part of the
    surface ``py.typed`` describes, and annotating one documents nothing a
    caller can reach. Everything else -- module-level functions, methods,
    properties -- is reachable from outside and is held to the same bar.
    """
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def walk(node: ast.AST, module: str, in_function: bool, cls: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if not in_function and not child.name.startswith("_"):
                    qualified = (
                        f"{module}.{cls}.{child.name}" if cls
                        else f"{module}.{child.name}"
                    )
                    found[qualified] = child
                walk(child, module, in_function=True, cls=cls)
            elif isinstance(child, ast.ClassDef):
                walk(child, module, in_function=in_function, cls=child.name)
            else:
                walk(child, module, in_function=in_function, cls=cls)

    for path in sorted((ROOT / "src").rglob("*.py")):
        module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        walk(
            ast.parse(path.read_text()),
            module.removesuffix(".__init__"),
            in_function=False,
            cls=None,
        )
    return found


def unannotated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The parameters, and the return, that carry no annotation.

    ``self`` and ``cls`` are not parameters a caller passes. ``*args`` and
    ``**kwargs`` are excluded too: the entry points that take them forward
    them straight to a family's own fit, where the accepted keywords differ
    by family, and the prose in the docstring says more than a type could.
    """
    args = node.args
    named = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    missing = [
        a.arg for a in named
        if a.annotation is None and a.arg not in ("self", "cls")
    ]
    if node.returns is None:
        missing.append("return")
    return missing


class TestAnnotations:
    """``py.typed`` promises a signature says something. This is that promise."""

    #: The public signatures allowed to carry an incomplete annotation, each
    #: with the reason. One entry, and it is structural rather than an
    #: oversight: see the comment at the site, which is longer than this note.
    UNANNOTATED: ClassVar[dict[str, str]] = {
        "clvtools.data.ClvDataDynCov.walks": (
            "returns `DyncovWalks`, whose module imports `ClvData` from "
            "`data.py` at module scope so that `build_walks`' own annotation "
            "resolves. Naming the type here closes that import cycle."
        ),
    }

    def test_every_public_signature_is_annotated(self):
        """No public parameter or return is left bare.

        ``ty`` is satisfied by an absent annotation -- there is nothing for it
        to contradict -- and so is ``get_type_hints()``, which returns the
        annotations that are there without noticing the ones that are not.
        This is the check that the annotation exists at all, which is what a
        consumer reading the signature actually depends on.
        """
        gaps = {
            name: missing
            for name, node in public_signatures().items()
            if name not in self.UNANNOTATED and (missing := unannotated(node))
        }
        assert not gaps, (
            f"these public signatures are missing annotations: {gaps}. "
            "`py.typed` says the annotations in this package can be relied "
            "on, so add the type rather than the name to "
            "TestAnnotations.UNANNOTATED -- that list is for import cycles "
            "that cannot be broken, and it has one entry."
        )

    def test_the_exemptions_are_all_still_needed(self):
        """An exemption for a signature that is now annotated is a stale one.

        The same shape as :meth:`TestSize.test_the_limit_still_binds`: a list
        nothing checks stops describing the code and starts excusing it. If
        the cycle behind an entry is ever broken, this fails and the entry
        goes rather than lingering as folklore.
        """
        signatures = public_signatures()
        unknown = sorted(set(self.UNANNOTATED) - set(signatures))
        assert not unknown, (
            f"UNANNOTATED names signatures that no longer exist: {unknown}"
        )
        needless = sorted(
            name for name in self.UNANNOTATED if not unannotated(signatures[name])
        )
        assert not needless, (
            f"these are fully annotated now and need no exemption: {needless}. "
            "Delete the entry."
        )


#: Every design limit in ``pyproject.toml``, by the ruff rule that enforces it:
#: the config key to set, and the worst value the code currently scores.
#:
#: Setting a limit to zero makes ruff report every function together with its
#: real value -- ``Too many statements (49 > 0)`` -- so one pass measures all
#: five. :class:`TestDesignLimits` does exactly that and checks these numbers,
#: which is why they can be trusted in a way the comment they replace could
#: not: two of its four figures were wrong when this was written.
DESIGN_LIMITS = {
    "C901": ("lint.mccabe.max-complexity", 8),
    "PLR0911": ("lint.pylint.max-returns", 6),
    "PLR0912": ("lint.pylint.max-branches", 8),
    "PLR0913": ("lint.pylint.max-args", 12),
    "PLR0915": ("lint.pylint.max-statements", 40),
}


def measure_design_limits() -> dict[str, tuple[int, str]]:
    """The worst value each design rule scores anywhere we wrote code.

    One ruff pass with every limit set to zero, which turns each rule into a
    report of what the code actually measures rather than a pass/fail. Returns
    the worst value per rule and where it is, so a failure can point at the
    function rather than leaving the reader to go looking.

    ``noqa`` is honoured rather than ignored: a function that is explicitly
    exempt is not what the limit governs, and counting it would misreport the
    headroom the limit has.
    """
    zeroed = [
        arg
        for _, (key, _) in sorted(DESIGN_LIMITS.items())
        for arg in ("--config", f"{key}=0")
    ]
    result = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        [
            sys.executable, "-m", "ruff", "check", "--no-cache",
            "--output-format=json", "--select", ",".join(sorted(DESIGN_LIMITS)),
            *zeroed, *TARGETS,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    worst: dict[str, tuple[int, str]] = {}
    for item in json.loads(result.stdout or "[]"):
        found = re.search(r"\((\d+) > 0\)", item["message"])
        code = item.get("code")
        if found is None or code not in DESIGN_LIMITS:
            continue
        value = int(found.group(1))
        if value > worst.get(code, (0, ""))[0]:
            where = Path(item["filename"]).relative_to(ROOT)
            worst[code] = (value, f"{where}:{item['location']['row']}")
    return worst


class TestDesignLimits:
    """What the code scores against each limit, measured rather than recalled.

    ``pyproject.toml`` carries a threshold per design rule and a sentence
    saying what the code measured when the threshold was chosen. The threshold
    is enforced; the sentence was not, and drifted -- it claimed 42 statements
    against an actual 49, and 5 returns against an actual 6, so the two figures
    that had gone tight were the two that read as comfortable. These tests are
    that sentence, executed.
    """

    def test_the_measured_worst_cases_are_current(self):
        """:data:`DESIGN_LIMITS` says what the code scores. It has to be right.

        A number here that no longer matches is not a small documentation
        problem: these are what say whether a limit still has room, and a
        stale one hides a limit that has quietly gone tight.
        """
        measured = measure_design_limits()
        drifted = {
            rule: f"recorded {recorded}, measured {measured[rule][0]} "
            f"at {measured[rule][1]}"
            for rule, (_, recorded) in DESIGN_LIMITS.items()
            if rule in measured and measured[rule][0] != recorded
        }
        assert not drifted, (
            f"DESIGN_LIMITS is out of date: {drifted}. Update the numbers -- "
            "the diff is the record of what grew, which is the point of "
            "keeping them here rather than in a comment."
        )

    def test_no_rule_scores_above_its_configured_limit(self):
        """The measured worst cases sit under the thresholds that enforce them.

        ``TestRuff`` already fails if one does. This says the same thing in
        terms of the numbers, so that a limit lowered below what the code
        scores fails with the value and the function that made it impossible
        rather than with a list of findings.
        """
        configured = configured_limits()
        over = {
            rule: f"{value} at {where} against a limit of {configured[rule]}"
            for rule, (value, where) in measure_design_limits().items()
            if rule in configured and value > configured[rule]
        }
        assert not over, f"these exceed their configured limit: {over}"


def configured_limits() -> dict[str, int]:
    """The design limits as ``pyproject.toml`` actually sets them.

    Read rather than restated, so that this file and the configuration cannot
    disagree about what is being enforced.
    """
    with (ROOT / "pyproject.toml").open("rb") as handle:
        lint = tomllib.load(handle)["tool"]["ruff"]["lint"]
    out = {}
    for rule, (key, _) in DESIGN_LIMITS.items():
        section, name = key.removeprefix("lint.").split(".")
        out[rule] = lint[section][name]
    return out


class TestSize:
    """The limits ruff has no rule for."""

    def test_no_module_outgrows_the_limit(self):
        """No module carries more than :data:`MAX_CODE_LINES` lines of code."""
        oversized = {
            str(path.relative_to(ROOT)): count
            for path in python_files()
            if (count := code_lines(path)) > MAX_CODE_LINES
        }
        assert not oversized, (
            f"these modules exceed {MAX_CODE_LINES} code lines "
            f"(docstrings and comments excluded): {oversized}. "
            "Split one out rather than raising the limit."
        )

    def test_the_limit_still_binds(self):
        """A limit far above the code is not a limit.

        If the largest module drops well below the cap, the cap has stopped
        measuring anything and should come down to meet it.
        """
        largest, where = max(
            (code_lines(path), path.relative_to(ROOT)) for path in python_files()
        )
        assert largest > MAX_CODE_LINES * 0.75, (
            f"the largest module is {where} at {largest} code lines against a "
            f"{MAX_CODE_LINES} limit; lower MAX_CODE_LINES to keep it binding."
        )

    def test_the_largest_module_keeps_its_distance(self):
        """A module close enough to the cap is one the next commit trips.

        The rule that split ``test_families.py`` at 697, applied by something
        other than whoever happens to look. It was stated once in a comment and
        then not applied again -- ``test_diagnostics.py`` reached 677 and sat
        there, because a comment cannot notice anything. :data:`MIN_HEADROOM`
        is that judgement as a number, and this is what enforces it.

        Split the module rather than shrinking the margin; the point of the
        margin is that arriving here means the module has two subjects in it,
        which is what the split will show.
        """
        largest, where = max(
            (code_lines(path), path.relative_to(ROOT)) for path in python_files()
        )
        assert largest <= MAX_CODE_LINES - MIN_HEADROOM, (
            f"{where} is {largest} code lines, leaving "
            f"{MAX_CODE_LINES - largest} under the {MAX_CODE_LINES} limit, "
            f"and {MIN_HEADROOM} is the least this gate allows. Split it."
        )


class TestTheToolsRun:
    """Finding 14: ``tools/benchmark.py`` had been raising on every invocation.

    ``fit_static_covariates``' optimiser arguments moved into a
    ``SearchSettings`` when the covariate fits were unified, and the benchmark
    still passed them loose. Nothing noticed: ``ty`` checks ``src/`` only,
    ``tools/`` is not in ``testpaths``, and no test imported it -- while the
    README documents running it. Twenty customers and one period keep this a
    smoke test rather than a benchmark.
    """

    def test_benchmark_runs(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "tools/benchmark.py", "--sizes", "20",
             "--periods", "13", "--repeats", "1"],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "customers" in result.stdout

    def test_profile_runs(self):
        """``tools/profile.py`` has doctests that never run for the same
        reason, and shadows the standard library's ``profile`` -- the trap its
        own docstring describes. Importing it is the cheap half of that."""
        import subprocess
        import sys

        # Registered in sys.modules before executing, because doctest looks a
        # module up by name -- and because profile.py shadows the standard
        # library's `profile`, which is the trap its own docstring describes.
        result = subprocess.run(
            [sys.executable, "-c", (
                "import doctest, importlib.util, pathlib, sys\n"
                "spec = importlib.util.spec_from_file_location("
                "'clv_profile_tool', pathlib.Path('tools/profile.py'))\n"
                "m = importlib.util.module_from_spec(spec)\n"
                "sys.modules['clv_profile_tool'] = m\n"
                "spec.loader.exec_module(m)\n"
                "sys.exit(doctest.testmod(m, verbose=False).failed)\n"
            )],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


class TestTheSlowFitStaysDeselected:
    """Finding 15: a caller's own ``-m`` used to replace the deselection.

    ``-m 'not dyncov_fit'`` was in ``addopts``, and pytest does not compose two
    ``-m`` expressions -- the later one wins. So ``pytest -m "not slow"``, which
    reads as "everything quick", collected the ten-minute time-varying fit.
    """

    def test_a_users_marker_expression_does_not_reselect_it(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "not slow",
             "tests/test_pnbd_dyncov.py", "--collect-only", "-q"],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        assert "test_reaches_at_least_the_oracles_optimum" not in result.stdout

    def test_but_asking_for_it_by_name_still_works(self):
        """Which is what ``.github/workflows/dyncov.yml`` runs."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "dyncov_fit",
             "--collect-only"],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        assert "test_reaches_at_least_the_oracles_optimum" in result.stdout
        assert "1/" in result.stdout, result.stdout[-500:]


class TestImportingTheePackageIsCheap:
    """``scipy.stats`` cost 78% of the import.

    It is wanted by three expressions in :mod:`clvtools.inference` -- two normal
    tails and one chi-squared -- and by nothing in a fit, a prediction or a
    diagnostic. Imported at module scope it was 0.55 s of a 0.70 s
    ``import clvtools``; deferred, the import is ~0.44 s.

    Asserted as *absence from* ``sys.modules`` rather than as a wall clock,
    which is the rule the rest of this suite follows: the saving is a property
    of what gets imported, and only the seconds move with the machine.
    """

    @staticmethod
    def _in_fresh_interpreter(body: str) -> str:
        import subprocess
        import sys

        result = subprocess.run(  # noqa: S603 - a fixed argv, no shell
            [sys.executable, "-c", body],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
        return result.stdout.strip()

    def test_scipy_stats_is_not_imported_by_import_clvtools(self):
        out = self._in_fresh_interpreter(
            "import sys, clvtools; print('scipy.stats' in sys.modules)"
        )
        assert out == "False", "scipy.stats is being imported at module scope again"

    def test_nor_by_fitting(self):
        """The path a script that only fits and predicts actually takes."""
        out = self._in_fresh_interpreter(
            "import sys, warnings, numpy as np, clvtools\n"
            "warnings.simplefilter('ignore')\n"
            "clvtools.pnbd.fit_pnbd(np.array([1.,0.,3.]), np.array([2.,0.,4.]),\n"
            "                       np.array([6.,6.,6.]), hessian=False)\n"
            "print('scipy.stats' in sys.modules)"
        )
        assert out == "False"

    def test_but_a_p_value_still_gets_it(self):
        """The deferral has to be a deferral, not a removal."""
        out = self._in_fresh_interpreter(
            "import sys, warnings, numpy as np, clvtools\n"
            "warnings.simplefilter('ignore')\n"
            "f = clvtools.pnbd.fit_pnbd(np.array([1.,0.,3.]), np.array([2.,0.,4.]),\n"
            "                           np.array([6.,6.,6.]), hessian=True)\n"
            "f.summary()\n"
            "print('scipy.stats' in sys.modules)"
        )
        assert out == "True"
