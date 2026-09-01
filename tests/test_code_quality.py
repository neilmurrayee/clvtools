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
import io
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

pytestmark = pytest.mark.quality

ROOT = Path(__file__).resolve().parent.parent

#: Everything that is ours. ``docs/`` holds the executable case study.
TARGETS = ("src", "tests", "tools", "docs")

#: Measured: the largest module is ``pnbd/dyncov.py`` at 655 code lines, and
#: the largest test module is ``test_families.py`` at 662. The limit leaves a
#: little headroom over both and catches anything that runs away from there.
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
