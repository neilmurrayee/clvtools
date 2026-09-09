# CLAUDE.md

## What this is

A from-scratch Python port of the R package **CLVTools 0.12.1**, following
Meierer, Bachmann, Näf, Schilter & Algesheimer, *"Estimating Individual Customer
Lifetime Values with R: The CLVTools Package"* (JSS submission 5634). The paper is
[arXiv:2602.09845](https://arxiv.org/abs/2602.09845); its LaTeX source and PDF
are **not committed** — arXiv's licence is non-exclusive distribution, which
covers arXiv's redistribution and not ours — and `.gitignore` carries the two
commands that fetch them to `arXiv-2602.09845v1/jss5634.tex` and
`2602.09845v1.pdf`, the paths every section reference here assumes.

The port is **section by section**: each module implements a numbered section of
the paper, its docstrings quote that section, and its examples are doctests that
`pytest` executes. `README.md` has the full status table, the verified-against-
the-paper numbers, and the findings log — read it before starting real work.

## Layout

```
src/clvtools/       the package. numpy + scipy + pandas only.
tests/              pytest suite; tests/fixtures/ holds committed oracle output
tests/paper_values.py   every number printed in the paper, in one place
tests/rdoc_values.py    every number printed in the R package's own documentation
examples/quickstart.ipynb  the twenty-cell tour; executed by pytest via nbmake,
                    outputs stripped, rebuilt by tools/build_quickstart.py
docs/paper.md       §6 case study as an executable doctest document
docs/vignette.md    the R package's walkthrough + advanced-techniques vignettes
docs/performance.md where the time goes, and why nothing asserts a wall clock
docs/spec.md        what a correct port must do, derived from the sources alone —
                    every one of its 222 items has a test; cite the id in yours
CHANGELOG.md        the short version: what installing a given version gets you
.github/workflows/  ci.yml on every push and PR; dyncov.yml nightly, for the fit
                    the ordinary gate deselects
tools/oracle/*.R    fixture generators — the only thing that needs R
tools/setup_oracle.sh   installs CLVTools into ./.Rlib (never the system library)
tools/benchmark.py  run times in the shape of the paper's Appendix B
tools/profile.py    cProfile summaries of the standard paths, as markdown
src/clvtools/data/  CLVTools' bundled datasets as CSV — inside the package,
                    so they ship in the wheel
```

Module ↔ paper mapping lives in `src/clvtools/__init__.py` and the README table.
There is no backlog. Seven audit rounds against `docs/spec.md` are closed and
all 222 of its items have a test behind them, so `docs/spec.md` is the one
remaining reference: it states what a correct port must do, derived from the
sources without reading this code, which is what makes it worth checking a
change against. **How** each claim is pinned is a `grep -rn 'F-12' tests/` away,
because every test names the spec item it covers — which is why the audit tables
that used to say the same thing were deleted rather than maintained. The tests
cite the spec, so the tests are the map. Findings and deliberate divergences
live in the README.

## Commands

```bash
uv run pytest                  # 2,083 tests inc. doctests in src/ and docs/; ~5:10 on an M-series
uv run pytest -m paper         # 22 numbers printed in the paper
uv run pytest -m rdoc          # 22 numbers printed in the R package's docs
uv run pytest -m literature    # 22 numbers published in the CLV literature
uv run pytest -m oracle        # 689 checks against the R oracle (247 fixtures + 435 pairs)
uv run pytest -m pair          # 435 paired R/Python checks, replayed from recordings
R_LIBS=.Rlib uv run pytest -m pair --oracle-live   # ...and diffed against live R
uv run pytest -m slow          # 202 full-dataset MLE fits
uv run pytest -m dyncov_fit    # the time-varying covariate MLE; ~10 min, deselected by default
uv run pytest --cov=clvtools --cov-report=term-missing
uv run pytest docs/paper.md    # the paper's case study alone
uv run pytest docs/vignette.md # the R package's own walkthrough
uv run pytest -m quality       # the static-analysis gate alone (runs by default)
uv run pytest -m performance   # the operation-count gates alone (runs by default; ~1s)
```

Static analysis is part of the suite, not a separate step:

```bash
uv run ruff check src tests tools docs          # what `-m quality` shells out to
uv run ruff check --fix src tests tools docs    # the mechanical ones
uv run radon cc src -s -n C                     # informational complexity report
```

`uv run` handles the environment; there is no separate install step. The
`dyncov_fit` marker is deselected by `tests/conftest.py`'s
`pytest_collection_modifyitems` — not by `addopts`, which is where it used to
live — so the default run excludes only that, and a `-m` of your own *composes*
with the deselection instead of replacing it. Ask for the fit back with
`-m dyncov_fit` or `--run-dyncov-fit`; both routes are asserted in
`tests/test_code_quality.py`.

**The suite needs no R.** R is needed only to re-baseline fixtures:

```bash
./tools/setup_oracle.sh
R_LIBS=.Rlib Rscript tools/extract_data.R                    # datasets -> data/
R_LIBS=.Rlib Rscript tools/oracle/generate_fixtures.R        # -> tests/fixtures/
R_LIBS=.Rlib Rscript tools/oracle/generate_family_fixtures.R
R_LIBS=.Rlib Rscript tools/oracle/generate_interface_fixtures.R  # summary, plots, generics
R_LIBS=.Rlib Rscript tools/oracle/generate_cdnow_fixtures.R       # the CDNOW fit, pmf, frequencies
R_LIBS=.Rlib Rscript tools/oracle/generate_time_fixtures.R       # S5's calendar arithmetic
R_LIBS=.Rlib Rscript tools/oracle/generate_dyncov_fixtures.R     # slow: fits dyncov twice
R_LIBS=.Rlib uv run python tools/oracle/record_pairs.py          # the paired oracle
```

Pipe an R generator to a file or to `tail`, never to `head`: closing the pipe
early leaves the R process wedged rather than killing it.

## How work is validated

The discipline that makes this port trustworthy, in order of strength:

1. **The paired oracle** (`tests/pairs.py`), which is fixtures plus the one
   thing they cannot carry: the *statement* that a given R expression and a
   given Python function are the same quantity. A pair holds both, the inputs,
   and the tolerance, in one declaration. `pytest -m pair` replays it against
   committed recordings with no R; `--oracle-live` evaluates the R in a live
   session and diffs three ways -- Python against R, the recordings against R
   (the staleness gate), and the input vectors both sides were fed. Prefer a
   pair over a new fixture column for anything expression-level: drop a
   `tests/pairs_*.py` module in place -- they are discovered, not imported by
   name -- and run `tools/oracle/record_pairs.py`. Every per-customer entry
   point CLVTools exposes is now paired, dyncov included; a new one means a new
   prelude in `tools/oracle/run_pairs.R`, and the dyncov one is worth reading
   first because it is the only one that has to re-parameterise a fitted
   object (`at()`, and note why it refreshes `@LL.data`).
2. **Oracle fixtures, expression by expression.** The generators call CLVTools'
   *internal* per-customer Rcpp entry points and dump every model expression at
   several parameter vectors — including points off the optimum and both arms of
   the `α ≥ β` branch. That makes each equation testable before an optimiser
   exists. A single total agreeing can hide two errors cancelling; thirty columns
   agreeing at two parameter vectors cannot. Prefer a new fixture column over a
   hand-computed constant.
3. **Published numbers.** `tests/paper_values.py` (`-m paper`) and
   `tests/rdoc_values.py` (`-m rdoc`). The paper is not the only place
   CLVTools prints results: its vignettes print a constrained covariate table,
   a regularized one and an `lrtest()` that the paper never does, and `?pmf`
   prints a PMF table with the empirical frequencies beside it. Treat a number
   printed in the R documentation as an oracle of the same standing.
   Keep estimation and evaluation apart: given the *published* parameters, every
   expression should match to 1e-9..1e-14; where this package's own optimiser
   runs, the last digits move (the Pareto/NBD ridge shifts 3e-5 for 1e-10 of
   log-likelihood), so assert accordingly.
4. **Internal cross-checks.** Mix the individual-level expressions numerically
   and require the marginalised closed form (see `tests/test_pnbd_individual.py`).
   Check the nesting the paper asserts — zero covariate effects recover the plain
   model, `m = 0` recovers independence.
5. **Doctests.** Everything in `src/` and `docs/paper.md` runs, so no printed
   number can drift from what the code returns.

100% **line and branch** coverage of `src/` is the standing bar. Branch
coverage was off until it was switched on and immediately found four arms line
coverage had called covered — the same lines run whichever way a condition
goes, so an `if` whose false arm nothing takes still reads as 100%. `branch =
true` lives in `[tool.coverage.run]`; don't land an uncovered line or an
untaken arm.

### Mutation testing: works, with cosmic-ray and not with mutmut

Coverage says a line ran, not that a test would have noticed it being wrong.
`cosmic-ray.toml` carries the configuration and the procedure; it is **not** a
gate and not part of `pytest` — one module against its own tests is about
twenty minutes, and a survivor is a question rather than a failure.

Run it in a **git worktree with its own `uv sync`**, never in place: cosmic-ray
edits the file on disk and reverts it after each mutant, so an interrupted run
leaves mutated source behind, and the worktree's own editable install is what
makes the mutation visible to the tests at all.

`timeunit.py` scored **87.5%** — 585 mutants, 512 killed, 73 survived. Reading
the survivors by hand is the whole job, because most cannot be killed by any
test:

- 13 mutate a *type annotation*, which `from __future__ import annotations`
  never evaluates;
- most of the `_Calendar.elapsed` cluster changes a value the following
  `while` loop corrects anyway — the estimate "can overshoot but never
  undershoot", as the comment there says;
- `year + (month == 12)` in `_anniversary`'s overflow branch cannot fire at
  all, because December has 31 days and so never overflows.

Three were real, and are now tests: a 31st rolling into **September or
November** (the spelling `(month % 12) | 1` gives the right answer for months
2, 4 and 6 and the wrong one for 9 and 11, so `2005-08-31 + 1 month` returned
2005-09-01 with the suite green), and `_Fixed`'s `frozen=True` and `repr=False`,
neither of which anything held it to.

**mutmut 3.7 does not work here — do not reach for it.** It reported 69
survivors on the same module; three sampled by hand were all false, including
one that raises `TypeError` at import and errors four test modules. All sixteen
mutants of one function were reported survived with none killed, which is a
function whose mutants never execute rather than a weak suite.
`use_git_change_detection`, `track_dependencies` and `PYTHONPATH` were each
ruled out and the cause was not found. The lesson generalises past the tool:
**sample survivors by hand before believing any aggregate.** That is what
separated a real finding from a fabricated one here, in both directions.

## House style

- **Docstrings carry the paper.** Section number, the paper's own words in
  quotes, the equation in `.. math::`, then a worked doctest. Match the density
  of the surrounding modules — they are unusually documented on purpose.
- **Static analysis is a gate, not advice.** `tests/test_code_quality.py` runs
  `ruff`, `ty`, an annotation-coverage gate and a module-size limit inside the
  ordinary `pytest` run, so there is one way to be green. The thresholds in
  `pyproject.toml` were measured against this code, not taken from defaults:
  mccabe 10, 50 statements, 12 branches, 12 arguments, 700 *code* lines per
  module (docstrings excluded — `src/` is 37% docstring on purpose, and a raw
  line count would punish that). Prefer splitting a function to raising a
  limit; where the paper's own signature is
  the reason, a `noqa` with the reason at the site is the escape hatch, and
  there are three — two for the argument count, one for `UP047` on
  `_validate.finished`, whose `TypeVar` ruff would rather see written with PEP
  695 type parameters that do not resolve under `get_type_hints()` on every
  3.12 this package supports.
- **Measure the code; don't write the measurement down.** What the code scores
  against each limit lives in `DESIGN_LIMITS` in `tests/test_code_quality.py`,
  and `TestDesignLimits` re-derives all five from ruff on every run. This
  replaced a comment in `pyproject.toml`, two of whose four figures were wrong
  by the time anyone checked — it claimed 42 statements against an actual 49
  and 5 returns against an actual 6, so the two limits that had gone tight read
  as the roomy ones. `MIN_HEADROOM` is the same idea for module size: the rule
  that split `test_families.py` at 697, enforced instead of recalled. Any
  number about this codebase belongs in something that runs.
- **A public signature is annotated, or it is exempt with a reason.**
  `py.typed` promises the annotations can be relied on, and `ty` alone does not
  keep that promise: an absent annotation has nothing to contradict, so it
  passes. `TestAnnotations` requires every public parameter and return under
  `src/` to carry a type, and its `UNANNOTATED` list — one entry, for an import
  cycle that genuinely cannot be broken — is itself checked for staleness.
- **Deviations get a test, not a comment.** Where the paper misprints an
  equation or CLVTools stops at a worse optimum, that is pinned by a test and
  recorded in the README's Findings section. Add to both.
- **Dependencies stay at numpy, scipy, pandas.** matplotlib is a `plot` extra
  used only by `diagnostics.render()`; nothing in `src/` may import it at module
  scope. R never enters `src/`. The first two are gated by
  `TestWhatSrcIsAllowedToImport`, which reads the dependency list out of
  `pyproject.toml` rather than repeating it — they were prose until a review
  asked what enforced them, and the answer was nothing.
- **All fits search over log-parameters** — same convention as CLVTools' C++
  entry points. Shared optimiser setup is `clvtools._optimize.options_for`;
  shared static-covariate machinery is `clvtools._staticcov`; the generics every
  fit exposes (`vcov`, `confint`, `summary`, `standard_errors`) come from the
  `clvtools.inference.Fitted` mixin, so a new family gets them by inheriting it
  and providing `names`, `__iter__` and `hessian`.
- **Two entry points, `latent_attrition()` and `spending()`**, dispatch on the
  data object's type. New estimators should be reachable from them as well as
  directly.
- Commit messages are a subject line plus prose explaining *why*, what was
  verified, and to what tolerance. Match the existing log.

## Traps that have already cost time

Oracle conventions (asserted by the generators before they write anything):
model parameters go in on the **log scale**, covariate parameters natural;
`*_LL_sum` and `gg_LL` return the **negated** sum; static-covariate arguments
are ordered **life-then-trans**; `pnbd_nocov_expectation` transposes the middle
pair relative to every sibling. Getting one wrong yields fixtures that are
plausible and wrong.

- SciPy's Nelder-Mead builds a microscopic simplex at an all-zeros log-parameter
  start and converges "successfully" at a far-off local optimum. `options_for`
  fixes this; use it rather than calling `scipy.optimize.minimize` bare.
- `continuous.discount.factor` in CLVTools is an unscaled annual rate; use
  `clvtools.predict.discount_factor` for the per-period value.
- Regularized fits penalise the **mean** log-likelihood, not the sum, so
  `log_likelihood` is ~-9.7 rather than ~-5821; compare with
  `unpenalised_log_likelihood`.
- Regularized fits run from both a cold and a warm start and keep the better —
  neither is universally right (necessary on Pareto/NBD, harmful on BG/NBD).
- `Id` is a **string** everywhere. Reading it as an integer silently reorders
  rows relative to the oracle; `tests/conftest.py:fixture_csv` enforces this.
- The dyncov fit is minutes, not seconds. Test its likelihood *and its
  prediction* against fixtures at fixed parameters; reserve `-m dyncov_fit` for
  the fit itself.
- Hessians are differenced with a **relative** step of 1e-4
  (`inference.numerical_hessian`). Smaller loses to cancellation; absolute steps
  send near-zero parameters like the GGom/NBD's `b` negative.
- A covariate Hessian must be taken over the parameters actually estimated: an
  equality constraint reports one coefficient where the unconstrained fit
  reports two.

## Adding a model or feature

1. Read the paper section; find the matching CLVTools entry point.
2. Extend the relevant `tools/oracle/generate_*.R` to dump per-customer values
   at ≥2 parameter vectors, with a `check()` against a public generic
   (`logLik()`, `coef()`, `predict()`) so a sign or ordering slip cannot ship.
   Regenerate, commit the fixture.
3. Write the equation-level tests against that fixture first.
4. Implement, with the paper's words and a doctest in the docstring.
5. Add the fit, then the `-m paper` check against published values if any exist.
6. Update `docs/paper.md`, the README table, and the Findings list if anything
   surprising turned up.
