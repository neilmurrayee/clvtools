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

``ty``
    ``src/clvtools/py.typed`` tells every downstream type checker that the
    annotations in this package are meant to be relied on. That is a promise,
    and this is what keeps it true. Only ``src/`` is checked, because only
    ``src/`` is what ``py.typed`` covers; the three rules that are off, and
    why, are recorded in ``pyproject.toml`` beside the ruff ignores.

module length
    Counted in *code* lines -- docstrings, comments and blanks excluded.
    Roughly 37% of ``src/`` is docstring, deliberately: the docstrings carry the
    paper. A raw line count would measure how well a module is documented and
    call the best-documented ones the worst, which is exactly backwards.

Two functions carry a ``noqa`` for the argument-count limit. Both are the
paper's equations written out -- the GGom/NBD's covariate likelihood, which is
the one family with five model parameters, and the dyncov ``F2`` term, which
runs per customer per likelihood evaluation and so cannot afford a wrapper
object. Each says so at the site.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import pkgutil
import subprocess
import sys
import tokenize
import typing
from pathlib import Path
from types import ModuleType

import pytest

import clvtools

pytestmark = pytest.mark.quality

ROOT = Path(__file__).resolve().parent.parent

#: Everything that is ours. ``docs/`` holds the executable case study.
TARGETS = ("src", "tests", "tools", "docs")

#: Measured: the largest module under ``src/`` is ``pnbd/dyncov.py`` at 455
#: code lines, and the largest anywhere is ``tests/test_predict.py`` at 553.
#: The limit catches anything that runs away from there.
#:
#: ``test_families.py`` reached 697 against this 700 and was split; a gate
#: three lines from tripping is one the next commit trips for no reason.
#: :meth:`TestSize.test_the_limit_still_binds` is the other half of that --
#: re-measure this note when a module is split, and bring the limit down if
#: the largest module has dropped away from it.
MAX_CODE_LINES = 700


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
        largest = max(code_lines(path) for path in python_files())
        assert largest > MAX_CODE_LINES * 0.75, (
            f"the largest module is {largest} code lines against a "
            f"{MAX_CODE_LINES} limit; lower MAX_CODE_LINES to keep it binding."
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
    """Backlog item 27, finding 20: ``scipy.stats`` cost 78% of the import.

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
