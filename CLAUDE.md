# CLAUDE.md

## What this is

A from-scratch Python port of the R package **CLVTools 0.12.1**, following
Meierer, Bachmann, Näf, Schilter & Algesheimer, *"Estimating Individual Customer
Lifetime Values with R: The CLVTools Package"* (JSS submission 5634). The paper's
LaTeX source is in `arXiv-2602.09845v1/jss5634.tex`; the PDF is `2602.09845v1.pdf`.

The port is **section by section**: each module implements a numbered section of
the paper, its docstrings quote that section, and its examples are doctests that
`pytest` executes. `README.md` has the full status table, the verified-against-
the-paper numbers, and the findings log — read it before starting real work.

## Layout

```
src/clvtools/       the package. numpy + scipy + pandas only.
tests/              pytest suite; tests/fixtures/ holds committed oracle output
tests/paper_values.py   every number printed in the paper, in one place
docs/paper.md       §6 case study as an executable doctest document
docs/audit.md       gaps against the paper and the R package, as a task list
tools/oracle/*.R    fixture generators — the only thing that needs R
tools/setup_oracle.sh   installs CLVTools into ./.Rlib (never the system library)
tools/benchmark.py  run times in the shape of the paper's Appendix B
data/               CLVTools' bundled datasets as CSV
```

Module ↔ paper mapping lives in `src/clvtools/__init__.py` and the README table.

## Commands

```bash
uv run pytest                  # 808 tests inc. doctests in src/ and docs/; ~3:05 on an M-series
uv run pytest -m paper         # 24 published-number checks
uv run pytest -m oracle        # 221 checks against R CLVTools fixtures
uv run pytest -m slow          # 114 full-dataset MLE fits
uv run pytest -m dyncov_fit    # the time-varying covariate MLE; ~17 min, deselected by default
uv run pytest --cov=clvtools --cov-report=term-missing
uv run pytest docs/paper.md    # the case study alone
```

`uv run` handles the environment; there is no separate install step. `-m
'not dyncov_fit'` is already in `addopts`, so the default run excludes only that.

**The suite needs no R.** R is needed only to re-baseline fixtures:

```bash
./tools/setup_oracle.sh
R_LIBS=.Rlib Rscript tools/extract_data.R                    # datasets -> data/
R_LIBS=.Rlib Rscript tools/oracle/generate_fixtures.R        # -> tests/fixtures/
R_LIBS=.Rlib Rscript tools/oracle/generate_family_fixtures.R
R_LIBS=.Rlib Rscript tools/oracle/generate_interface_fixtures.R  # summary, plots, generics
R_LIBS=.Rlib Rscript tools/oracle/generate_dyncov_fixtures.R     # slow: fits dyncov twice
```

Pipe an R generator to a file or to `tail`, never to `head`: closing the pipe
early leaves the R process wedged rather than killing it.

## How work is validated

The discipline that makes this port trustworthy, in order of strength:

1. **Oracle fixtures, expression by expression.** The generators call CLVTools'
   *internal* per-customer Rcpp entry points and dump every model expression at
   several parameter vectors — including points off the optimum and both arms of
   the `α ≥ β` branch. That makes each equation testable before an optimiser
   exists. A single total agreeing can hide two errors cancelling; thirty columns
   agreeing at two parameter vectors cannot. Prefer a new fixture column over a
   hand-computed constant.
2. **Published numbers.** `tests/paper_values.py`, checked under `-m paper`.
   Keep estimation and evaluation apart: given the *published* parameters, every
   expression should match to 1e-9..1e-14; where this package's own optimiser
   runs, the last digits move (the Pareto/NBD ridge shifts 3e-5 for 1e-10 of
   log-likelihood), so assert accordingly.
3. **Internal cross-checks.** Mix the individual-level expressions numerically
   and require the marginalised closed form (see `tests/test_pnbd_individual.py`).
   Check the nesting the paper asserts — zero covariate effects recover the plain
   model, `m = 0` recovers independence.
4. **Doctests.** Everything in `src/` and `docs/paper.md` runs, so no printed
   number can drift from what the code returns.

100% line coverage of `src/` is the standing bar; don't land uncovered lines.

## House style

- **Docstrings carry the paper.** Section number, the paper's own words in
  quotes, the equation in `.. math::`, then a worked doctest. Match the density
  of the surrounding modules — they are unusually documented on purpose.
- **Deviations get a test, not a comment.** Where the paper misprints an
  equation or CLVTools stops at a worse optimum, that is pinned by a test and
  recorded in the README's Findings section. Add to both.
- **Dependencies stay at numpy, scipy, pandas.** matplotlib is a `plot` extra
  used only by `diagnostics.render()`; nothing in `src/` may import it at module
  scope. R never enters `src/`.
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
