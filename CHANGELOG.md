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
built but never published — see backlog item 15, which is a decision
rather than a task.

### Added

- The public API in full: `latent_attrition()` and `spending()`, the four model
  families, `predict()` with prospective customers, `vcov`/`confint`/`summary`/
  `fitted`/`lrtest`, the descriptive and model diagnostics, bootstrap intervals,
  and calendar time units. The README's table maps each to a section of the
  paper.
- Continuous integration on 3.12 and 3.13, and a nightly job for the
  time-varying covariate MLE.
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

### Changed

- A regularized fit's standard errors are differenced on the penalised objective
  that was optimised rather than the unpenalised sum, and warn that they are
  ridge standard errors dominated by the penalty. CLVTools' own answer here is
  not followable — see the README's findings.
- Printed values in the documentation elide their last digits where a fitted
  quantity is not portable between platforms; comparisons against the paper and
  the oracle use tolerances, which is where the precision lives.
