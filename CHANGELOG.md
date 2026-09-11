# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project does not
yet use semantic versioning, because it has not had a release to be
semantic about.

the backlog carries the working queue and the paper/R-package audit the two audit
rounds against the paper and the R package. This file is the short version: what
someone installing a given version gets.

## Unreleased

Everything below is on `main` and in no released artifact. `0.1.0` has been
built but never published. Whether to publish it is a decision for the
maintainer, not a task: the package is complete against the paper and the R
package, and what a release would add is a name on PyPI and a support
commitment.

### Added

- The public API in full: `latent_attrition()` and `spending()`, the four model
  families, `predict()` with prospective customers, `vcov`/`confint`/`summary`/
  `fitted`/`lrtest`, the descriptive and model diagnostics, bootstrap intervals,
  and calendar time units. The README's table maps each to a section of the
  paper.
- Continuous integration on 3.12 and 3.13, and a nightly job for the
  time-varying covariate MLE.
- **A paired oracle** (`tests/pairs.py`). An R expression and the Python
  function that must equal it are now declared together, with the inputs and
  the tolerance, instead of living in a generator and a test module with
  nothing relating them. `pytest -m pair` replays the pairing against committed
  recordings and needs no R; `--oracle-live` evaluates the R snippets in a live
  session and diffs Python against R, the recordings against R, and the inputs
  both sides were fed. The second of those is a gate this project did not have:
  a fixture edited by hand, or left behind by a CLVTools upgrade, was
  previously green. A weekly workflow runs it. The Pareto/NBD without
  covariates was the first family on it; the covariate arms of all three
  families followed, which is where the mechanism earned itself. Of the 76
  per-customer entry points CLVTools exposes, 38 are called by no fixture
  generator, and the largest block of those was static-covariate machinery --
  every GGompertz/NBD covariate expression, and all of the BG/NBD's beyond
  three scale transforms, had no equation-level check at all. Fifteen now do.
  The no-covariate arms of the other families and the time-varying covariate
  machinery followed, so every per-customer expression CLVTools exposes is
  paired: 44 pairs, 139 evaluations, 435 checks.
- A referee for `scipy.special.hyperu`. The paired oracle found it 90x slower
  for `1 < s < 2`; following that up against an independent evaluation of the
  integral representation showed it is also *wrong* there, by up to 6.1e-07.
  Nothing in `src/` changed -- neither candidate is better everywhere, and the
  standard `DERT` evaluates three orders below the bad region -- but the fact
  is now pinned rather than assumed away.
- `ConvergenceWarning`: every fit now says when it did not converge, and when a
  Hessian cannot be trusted.

### Fixed

- **The built wheel carried no datasets.** `DATA_DIR` resolved to the repository
  root, which does not exist under `site-packages`, so `load_apparel_trans()`
  raised `FileNotFoundError` on any installed copy — the README's first usage
  line. The data now lives inside the package.
- **Convergence tolerances asked for better than machine precision.**
  `ftol = 1e-16` is `factr = 0.45`; on x86-64 Linux the line search failed and
  the same optimum came back with `converged = False`.
- **The GGom/NBD lost its death term for heavy buyers**, returning `PAlive`
  exactly 1.0 and `CET` `NaN` from about `x = 140` — around `x = 105` on daily
  data.
- **A recency a hair above `T` collapsed a whole fit** to its start values with
  `-inf` and no exception.
- **NaN prices and NaN covariates travelled** into the likelihoods and came back
  as plausible numbers.
- **The bootstrap** lost every draw to one failure, ignored `seed` whenever a
  sampler was given, and spent 0.965 s a draw rebuilding data that now takes
  0.016 s.
- `tools/benchmark.py`, which the README documents and which had been raising
  `TypeError` on every invocation.
- **A heavy buyer's time-varying likelihood silently became the alive-only
  one.** Every term of `F2` underflows past about `x = 160`, which selected the
  branch for a customer who is certainly dead with no signal: at `x = 200` the
  answer was wrong by 225 log-units and `PAlive` came back as exactly 1.0 where
  the truth is 1.6e-98. `F2` is now carried as a log magnitude and a sign
  throughout. No oracle fixture could see this — CLVTools underflows in the same
  place, and the apparel cohort's largest `x` is 21 — so the check is the
  nesting §3.3 asserts against the closed-form Pareto/NBD.
- **A fit that reported success on a wrong answer.** On hourly data the
  Pareto/NBD stopped 223 log-units short of the optimum, at a degenerate
  `s = 0.0011`, with `converged = True`; the GGompertz/NBD raised instead. The
  cause is that a start value of 1 is a claim about the time unit, and these
  likelihoods are exactly invariant to that unit while the optimiser is not. The
  default start now scales with the data.
- **The GGompertz/NBD's survival term cancelled.** Formed as a difference of two
  logs, it was wrong by 2.2e-10 relative at the parameters CDNOW's published fit
  lands on, where the `log1p`/`expm1` form is exact to 2.2e-16.
- **A diverged fit predicted in silence**, returning a table whose `PAlive`,
  `CET` and `DERT` were entirely `NaN`; a covariate named twice built a
  rank-deficient design and reported two coefficients for one covariate; and the
  time-varying covariate series was never checked for duplicates, gaps or `NA`.
- **Four defects the spec-derived audit found that no test here could see**
  (the spec audit): standard errors `sqrt(600) = 24.5` times too large at
  a zero penalty; a dyncov bootstrap that resampled but refitted *without* the
  covariates, because `ClvDataDynCov` subclasses `ClvData` and the branch never
  fired; a `NaN` prediction horizon accepted and returned as a `NaN`
  prediction; and a mistyped covariate name silently dropped, so a scenario
  built on a typo answered from the covariates that were recognised.

### Added (continued)

- `confint(parm=...)`, `diagnostics.pmf_table()` (the per-customer PMF frame),
  formula support for bare calls (`log(Gender + 2)`) and interactions
  (`Gender * Channel`, `Gender : Channel`), and single-logical validation on
  every fit's `hessian`.
- **Complete type annotations on the public API**, and a gate that keeps them
  that way. `py.typed` already promised the annotations could be relied on, but
  nothing checked that a given parameter had one: `ty` and `get_type_hints()`
  are both satisfied by an annotation that is simply absent. Twenty-four public
  signatures were carrying a bare parameter or return — among them every
  entry point in `pnbd/dyncov_predict.py`, whose `data` and `params` were the
  only untyped arguments in signatures where everything else was typed.
  `latent_attrition()` now declares that it returns a `Fitted`, `render()` and
  `_validate.finished()` describe what they accept with protocols rather than
  reaching for an optional or a private dependency, and `diagnostics` exports
  `Expectation` and `Pmf` for the two callables its frames are built from.
  One signature stays exempt, with the reason recorded at the site and in the
  gate: `ClvDataDynCov.walks()` returns a type whose module must import
  `ClvData` from `data.py`, and naming it would close that import cycle.

### Changed

- **`confint` now refuses `level = 0`**, which the code always rejected and no
  test asserted. It came out of a mutation run on `inference.py` that is
  otherwise a cautionary tale: scored against its own test file the module
  reported 82.7% with 122 survivors, including `aic`'s entire formula and
  `summary`'s z and p columns. Almost none was real — AIC is asserted in six
  places across the family test files, the summary table is printed with its
  `Pr(>|z|)` column as a doctest in `docs/vignette.md`, and the `parm` bound is
  held by a `slow` test the selection excluded. One survivor of the 122 was
  genuine. A mutation score is a property of the module *and* the tests it is
  run against, and `inference.py` is verified almost entirely from outside
  itself; CLAUDE.md now says to check a selection can kill a known-fatal mutant
  before trusting anything it reports.
- **`ggomnbd.py` mutation-tested: 89.2% raw, 91.8% measured**, and its `CET`'s
  overflow branch turned out never to have been evaluated. The function forms
  its denominator directly where `P` is representable and in logs where
  `exp(log P)` overflows — the branch that exists to stop the answer becoming
  `NaN`. Reaching it needs a customer with thousands of transactions, which no
  fixture holds and no fit produces, so twenty mutations of its single line
  survived the whole suite, among them a sign flip on `2 log(bs)` worth a factor
  of `e^58`. `tests/test_ggomnbd_numerics.py` pins it without a reference
  implementation or a new fixture: the two branches compute the same quantity,
  so `log CET` cannot have a kink where control passes between them, and over an
  even grid of `x` straddling the crossover the second difference is 0.003
  against the 29 that sign flip would add. Also: `fit_ggomnbd` rejects a start
  value of zero, which the guard always did and only the *negative* case tested.
- **The BG/NBD's and GGom/NBD's Hessians are now tested at all.** The fixtures
  carry standard errors for the Pareto/NBD and for the GGom/NBD *with*
  covariates, and nothing for either plain fit — so those two families returned
  a Hessian no test ever looked at. Inverting `if hessian:`, so that asking for
  one skips it, survived the entire suite in both; so did flipping the sign of
  the objective the BG/NBD's is differenced from, which negates the matrix and
  turns every standard error into `nan` while `converged` still reads `True`.
  `TestAFitsHessianIsUsable` pins the properties rather than the numbers, so it
  needs no new oracle fixture: symmetric, correctly shaped, positive definite,
  and standard errors finite and positive. The GGom/NBD is exempted from the
  last two deliberately — it fits `b` to 3e-06 and `beta` to 1e-04, where the
  likelihood is flat enough that the matrix is genuinely indefinite and two of
  its standard errors are `nan`; what is asserted there is the warning that
  says so, which is finding 9's subject. Five mutants across four families were
  confirmed killed by it.
- **`pnbd/individual.py` mutation-tested: 96.8%**, and 24 of its 26 survivors
  were the same guard. `poisson_pmf` and `nbd_pmf` both special-case
  `t == 0 and x == 0`, where the log form is `0 * log 0` and undefined. The
  Poisson had one test there at `x = 0`; the NBD had none at all. Together that
  left the guard almost entirely unpinned — replacing the returned `1.0` with
  `0.0` survived the whole suite. Both now assert that no elapsed time means
  `P(X=0) = 1` and every other count is impossible, and the counts above zero
  are what pin the condition rather than the value. Two smaller ones came with
  it: `gamma_pdf_lambda`'s exponent `shape - 1` is numerically equal to
  `shape % 1` for every shape in `[1, 2)`, and `r = 1.449` is the only shape the
  density was ever checked at, so the densities now cover a shape above 2 and
  one below 1; and `_require_positive` had the same zero-only gap already found
  in `gg.py`.
- **`pnbd/aggregate.py` mutation-tested: 96.7%**, the highest of the four and
  the source of the most interesting finding, because a *redundant* path can
  hide a broken one. `pmf` computes `b1 - b2` and falls back to `_series_tail`
  when `(b1 - b2) / b1` drops below `_CANCELLATION_LIMIT`. Corrupting `b2`'s
  exponent drives that ratio to about **-20** — below the limit like any
  severely cancelled value — so the series ran instead and returned the right
  answer, and every mutation of `b1` and `b2` survived. A negative ratio is not
  cancellation: `b2` truncates a series of positive terms summing to `b1`, so
  the difference is positive by construction. The new test forbids the fallback
  at well-conditioned parameters and demands the same answer, which pins the
  primary path; four separately verified mutants die on it. Narrowing
  `cancelled` to `(0, 1]` in `src/` would be a behaviour change to code the
  oracle agrees with, so that is recorded rather than made.
- **Mutation testing extended to `special.py` and `gg.py`.** 95.6% and 91.4%
  respectively. Five more real gaps, and every one the same shape — a guard
  tested only far from its own boundary. `_hyp2f1_series` rejects `z` outside
  `(0, 1)`, and at exactly `z = 1` the term count divides by zero, so relaxing
  the test to `z <= 1.0` turned a documented `nan` into an `OverflowError` with
  the suite green. `expected_mean_spending`'s existence condition was checked at
  `-0.5`, comfortably inside the rejected region and comfortably enough that
  four wrong thresholds reject it too; `q = 1, x = 0` puts it exactly on zero,
  where the formula's own denominator vanishes. `_require_positive` had only
  ever been handed zero, the one non-positive value `> 0` and `!= 0` both
  reject, so writing it the second way would have admitted every negative
  parameter. And `frozen=True` with the Hessian kept out of the repr is a
  convention across **eight** fitted params classes that nothing held any of
  them to — now a test that discovers the classes rather than listing them, so
  a new family is covered the day it is written.
- **Mutation testing, and the three real gaps it found in `timeunit.py`.**
  Coverage says a line ran, not that a test would have noticed it being wrong.
  `timeunit.py` scores 87.5% — 585 mutants, 512 killed. Most survivors cannot be
  killed by any test: 13 mutate a type annotation that `from __future__ import
  annotations` never evaluates, and much of the `_Calendar.elapsed` cluster
  changes an estimate the following loop corrects anyway. Three were real. A
  31st rolling into **September or November** was unpinned — the wrong spelling
  `(month % 12) | 1` happens to be right for months 2, 4 and 6, so
  `2005-08-31 + 1 month` could return 2005-09-01 instead of 2005-10-01 with the
  whole suite green — and `_Fixed`'s `frozen=True` and `repr=False` were held to
  by nothing, though the units are shared module-level singletons. All three now
  have tests, each verified to kill the mutant that found it. `cosmic-ray.toml`
  carries the configuration; it is not a gate, and it is run in a worktree.
- **Branch coverage turned on, and the four arms it found closed.** The suite
  had reported 100% coverage for as long as anyone had looked, but there was no
  `[tool.coverage]` section, so it was 100% of *lines*. Switching branches on
  dropped it to 99%: a float `Id` column with nothing whole in it, a GGom/NBD
  fit run from a caller's own start, a Nelder-Mead polish that finds nothing
  better than the gradient result, and the PMF tail series running out of terms
  rather than breaking early. All four are arms where the same lines execute
  either way, which is exactly what line coverage cannot distinguish. Each now
  has a test, and 100% of both is the standing bar.
- **`src/`'s allowed imports are gated.** "Dependencies stay at numpy, scipy,
  pandas" and "nothing in `src/` may import matplotlib at module scope" were
  rules in CLAUDE.md that nothing enforced, alongside an existing test gating
  an import rule of exactly the same shape. `TestWhatSrcIsAllowedToImport`
  reads the dependency list out of `pyproject.toml` rather than repeating it,
  so declaring a dependency without allowing it here cannot pass.
- **The design limits are measured rather than described.** `pyproject.toml`
  carried a sentence saying what the code scored against each limit — "mccabe
  8, 42 statements, 8 branches, 5 returns" — and two of the four figures were
  wrong: statements had reached 49 and returns 6, so the two limits that had
  quietly gone tight were the two the note called roomy. The numbers now live
  in `DESIGN_LIMITS`, and `TestDesignLimits` re-derives all five from ruff in a
  single pass on every run; a drifted figure fails the suite and names the
  function responsible. The same treatment for module size: `MIN_HEADROOM`
  makes the rule that split `test_families.py` at 697 into something enforced
  rather than remembered, and the note that used to name the largest modules —
  wrong by 72 and 124 lines when it was checked — is gone in favour of tests
  that compute both ends and name the file.
- **`tests/test_bootstrap.py` split out of `tests/test_diagnostics.py`**, which
  had reached 677 code lines against the 700-line limit. The two halves shared
  a data fixture and nothing else: the diagnostic frames of S6.2.2 and S6.2.4
  are checked against CLVTools' `plot(..., plot = FALSE)` row for row, and
  S6.3.3's bootstrap cannot be, because it is random. 89 tests before, 89
  after; the largest module is now `tests/test_pnbd_dyncov.py` at 658.
- **`new_customer_expectation` split.** It was the most-stressed function in
  the package: 49 statements against a limit of 50, and at the complexity
  ceiling too. The multiplier frame it built twice -- once per process, the two
  calls differing only in which coefficients and which noun went in -- is now
  `_new_customer_multipliers`, which is also where the two guards belong, since
  a missing column and a repeated date are both things the later merge cannot
  diagnose. 40 statements now, and off the complexity list; the statement limit
  went from one line of headroom to ten.
- **The shared `data` fixture moved to `tests/conftest.py`.** Six modules had
  built the same S6.2 data object privately -- three from `apparel_trans` and
  three by calling `load_apparel_trans()`, which re-reads the CSV to produce a
  frame equal to the one the session fixture already holds. The plain
  counterpart of `static_data`, extracted for the same reason and at the same
  scope. All six already called it `data`, so consolidating was a deletion.
- A regularized fit's standard errors are differenced on the penalised objective
  that was optimised rather than the unpenalised sum, and warn that they are
  ridge standard errors dominated by the penalty. CLVTools' own answer here is
  not followable — see the README's findings.
- Printed values in the documentation elide their last digits where a fitted
  quantity is not portable between platforms; comparisons against the paper and
  the oracle use tolerances, which is where the precision lives.
