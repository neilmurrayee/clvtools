# clvtools

A from-scratch, test-driven Python implementation of the **CLVTools** R package,
following Meierer, Bachmann, Näf, Schilter & Algesheimer, *"Estimating
Individual Customer Lifetime Values with R: The CLVTools Package"* (Journal of
Statistical Software, submission 5634).

**[📄 The paper's case study, executable →](docs/paper.md)** &nbsp;·&nbsp; **[📘 The R package's own walkthrough →](docs/vignette.md)**

Every section of the paper maps to a module; every module's docstrings carry the
paper's equations, its own words, and worked examples that **pytest executes**.
The numbers in the documentation cannot drift from what the code returns.

## What it implements

| Paper | Model | Module |
|---|---|---|
| §3.2, App. A | Pareto/NBD | `clvtools.pnbd` |
| §3.3 | Time-invariant covariates | `clvtools.pnbd.staticcov` |
| §3.3 | Time-varying covariates | `clvtools.pnbd.dyncov` |
| §3.4 | Sarmanov process correlation | `clvtools.pnbd.correlation` |
| §3.4 | L2 regularization, equality constraints | `clvtools.pnbd.staticcov` |
| §3.5 | Gamma-Gamma spending | `clvtools.gg` |
| Table 4 | BG/NBD | `clvtools.bgnbd` |
| Table 4 | GGom/NBD | `clvtools.ggomnbd` |
| §6.1 | `clvdata()` — the data layer | `clvtools.data` |
| §6.1.2 | `summary()`, `as.data.frame()` and Table 3's five descriptive plots | `clvtools.data`, `clvtools.diagnostics` |
| Table 2 | `latentAttrition()` and `spending()` — the formula interface | `clvtools.estimate` |
| Table 2 | `vcov()`, `confint()`, `summary()`, `fitted()`, `lrtest()` | `clvtools.inference` |
| §6.3 | `predict()` — PAlive, CET, DERT, CLV, for all three families | `clvtools.predict` |
| §6.3.4 | `newcustomer()` — prospective customers, with or without covariates | `clvtools.predict` |
| §6.4.2 | Prediction with time-varying covariates | `clvtools.pnbd.dyncov_predict` |
| §6.2.2, §6.2.4 | Tracking, PMF and spending diagnostics | `clvtools.diagnostics` |
| §6.3.3 | Bootstrap confidence intervals | `clvtools.bootstrap` |
| §5 | Time units, including calendar months and years | `clvtools.timeunit` |

Time-invariant covariates, equality constraints and regularization are
available for all three latent attrition families, as Table 4 marks them.
Time-varying covariates and process correlation are Pareto/NBD only, likewise.

## Installation

Not on PyPI. From the repository:

```bash
pip install git+https://github.com/neilmurrayee/clvtools
pip install "clvtools[plot] @ git+https://github.com/neilmurrayee/clvtools"   # + matplotlib
```

Python 3.12 or newer, and NumPy, SciPy and pandas come with it. The five
datasets ship inside the package, so `load_cdnow()` works on a fresh install —
which is worth saying because for a while it did not, and that is recorded in
the findings below.

For development, `uv sync` and then `uv run pytest`; there is no separate
install step and the suite needs no R.

## Usage

```python
import clvtools
from clvtools import (
    ClvData, latent_attrition, load_apparel_trans, predict, spending,
)
from clvtools.predict import discount_factor

data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)

pnbd = latent_attrition(family=clvtools.pnbd, data=data)
gg = spending(family=clvtools.gg, data=data)

# discount_factor() turns an annual rate into a per-period one. Without it,
# CLVTools' default discounts at log(1.1) *per week* on weekly data, and the
# total DERT here is 92.92 against 2642.78 -- see the findings below.
predict(data, pnbd, gg, continuous_discount_factor=discount_factor(0.10))
pnbd.summary()            # estimates, standard errors, z- and p-values
data.summary()            # §6.1.2's descriptive table
```

The two entry points are the paper's own; `latent_attrition` takes S6.4's
formula (`~ Gender + Channel | Gender + Channel`) and picks the estimator from
the data object's type. The per-family `fit_*` functions underneath take
`(x, t_x, T)` directly.

Bring your own data as a frame of `Id`, `Date` and optionally `Price`.

## The public API

Twenty-four names, and what each is for. Everything else is an implementation
detail that may move.

| Name | What it is |
|---|---|
| `ClvData` | A transaction log with an estimation/holdout split — §6.1's `clvdata()` |
| `ClvDataStaticCov`, `ClvDataDynCov` | The same, with time-invariant or time-varying covariates |
| `latent_attrition`, `spending` | §6.2's two entry points; they dispatch on the data object's type |
| `pnbd`, `bgnbd`, `ggomnbd`, `gg` | The four model families, as modules — pass one as `family=` |
| `predict` | §6.3's table: PAlive, CET, DERT, predicted CLV, and the actuals |
| `newcustomer`, `newcustomer_static`, `newcustomer_dynamic` | §6.3.4's prospective customer, with or without covariates |
| `newcustomer_spending` | The same for average order value |
| `discount_factor` | Turns an annual rate into the per-period one `predict` wants |
| `inference` | `vcov`, `confint`, `summary`, `fitted`, and `likelihood_ratio_test` |
| `likelihood_ratio_test` | §6.5.3's `lrtest()`, re-exported for convenience |
| `diagnostics` | §6.2.2's tracking and PMF data, §6.1.2's five descriptive frames, and `render()` |
| `bootstrap` | §6.3.3's resampling: `bootstrap_apply`, `bootstrap_data`, `confidence_intervals` |
| `load_apparel_trans`, `load_apparel_static_cov`, `load_apparel_dyn_cov`, `load_apparel_dyn_cov_future`, `load_cdnow` | The five bundled datasets |

The per-family `fit_*` functions underneath — `clvtools.pnbd.fit_pnbd` and its
siblings — take `(x, t_x, T)` directly, for when the data layer is in the way.

## The oracle

The R package CLVTools is the reference implementation. Rather than testing only
against the ~40 numbers the paper prints, the generators in `tools/oracle/` call
CLVTools' internal per-customer entry points to dump expectations for **every**
model expression, at several parameter vectors — including points deliberately
off the optimum, both arms of the `α ≥ β` branch in the appendix likelihood, and
all thirty intermediate quantities of the time-varying covariate likelihood.

That makes each equation testable on its own, before an optimiser exists to find
its maximum. A single total agreeing can hide two errors cancelling; thirty
columns agreeing at two parameter vectors cannot.

Fixtures are committed, so **the test suite needs no R**. R is needed only to
re-baseline them:

```bash
./tools/setup_oracle.sh                                   # installs into ./.Rlib only
R_LIBS=.Rlib Rscript tools/extract_data.R                 # datasets  -> data/
R_LIBS=.Rlib Rscript tools/oracle/generate_fixtures.R     # -> tests/fixtures/
R_LIBS=.Rlib Rscript tools/oracle/generate_family_fixtures.R
R_LIBS=.Rlib Rscript tools/oracle/generate_interface_fixtures.R  # summary, plots, generics
R_LIBS=.Rlib Rscript tools/oracle/generate_dyncov_fixtures.R     # slow: fits dyncov twice
```

`setup_oracle.sh` never touches your system R library. CRAN's macOS binaries lag
the newest R release, so it falls back through recent series; the 4.5 build
installs and runs correctly under R 4.6.

The generators assert their own conventions before writing anything — the C++
entry points take **log-scale** model parameters, return the **negated** sum, and
order static-covariate arguments life-then-trans, while `pnbd_nocov_expectation`
transposes the middle pair relative to every sibling. Each fixture family is
checked against a public generic (`logLik()`, `coef()`, `predict()`) so a sign or
ordering slip cannot ship as a plausible-looking expectation.

## Verified against the paper

| Quantity | Paper | This implementation |
|---|---|---|
| Pareto/NBD `r, α, s, β` | 1.4490, 48.6361, 0.5613, 46.8844 | to 1e-3 — the last digit is platform-dependent, below |
| Pareto/NBD log-likelihood | — | −5848.0978 (oracle: −5848.0978) |
| Mean purchase / attrition rate | 0.030 / 0.012 | 0.030 / 0.012 |
| Gamma-Gamma `p, q, γ` | 3.099, 5.654, 56.504 | identical |
| Gamma-Gamma log-likelihood | — | −1670.663 |
| `predict()` table, §6.3.2 | 18 values | all to 1e-6 |
| Holdout `mae.cet` / `rmse.cet` | 2.039532 / 3.329395 | 2.0396 / 3.3294 |
| New-customer transactions / spend | 2.218635 / 39.1372 | identical |
| Static-cov coefficients, §6.4.1 | −0.6430, 0.7907, 0.2859, 0.6241 | −0.64, 0.79, 0.29, 0.62 |
| Static-cov log-likelihood / AIC | −5821.0627 / 11658.1254 | −5821.0627 / 11658.13 |
| Tracking plot data, §6.2.2 | 626 rows × 2 series | to 6e-11 |
| PMF plot data, §6.2.2 | 22 bins | to 3e-12 |
| Descriptive table, §6.1.2 | 39 cells | every one |
| Descriptive plot data, §6.1.2 | 5 frames, 3,900 rows | to 1e-12 |
| Standard errors, §6.4.1 | 8 values | to 1e-3 |
| Likelihood ratio test, §6.5.3 | χ² on a constraint | to 1e-3 against the oracle |
| Dyncov prediction table, §6.4.2 | 18 values | oracle to 1e-12, *not* the paper — see below |
| Time arithmetic, §5 | 840 spans, 280 additions | to 5e-15 |

Given the *published* parameters rather than its own fit, every expression
matches the oracle to between 1e-9 and 1e-14. Where this package's own optimiser
is used, the last digits move: the Pareto/NBD likelihood traces a long flat
ridge, and moving 3e-5 along it changes the log-likelihood by under 1e-10. The
tests keep estimation and evaluation apart for exactly this reason.

## Findings

Working through the paper turned up several things worth recording. Each is
covered by a test rather than left in a comment.

**Three transcription errors in the printed equations.** Eq. (14) writes the
per-transaction spending density with $z_i^{r-1}$, using the Pareto/NBD's $r$
where the shape $p$ belongs. Eq. (17)'s integral writes $\nu{q-1}$ for
$\nu^{q-1}$. And eq. (17)'s result drops the exponent $px$ from its final
factor — as printed it is not a density, and its log differs from what CLVTools
maximises by 3.45 at a representative point.

**A stray $\mu$ in Appendix A.** The integrand's second term is printed as
$\frac{\lambda^{x+1}\mu}{\lambda+\mu}e^{-(\lambda+\mu)T}$, where eq. (10) in the
body has no $\mu$. Integrating the appendix version does not reproduce the
closed form the appendix then states; eq. (10) does.

**CLVTools computes AIC and BIC of a regularized fit from the penalised mean
log-likelihood.** Its advanced-techniques vignette prints `AIC 35.4626` and
`BIC 70.6380` for a model whose log-likelihood is −5833.33; those are exactly
`2k − 2L` and `k·ln(n) − 2L` for `k = 8`, `n = 600` and `L = −9.7313`, the
penalised *mean* objective its `logLik()` returns. The same model without
regularization is printed at `AIC 11658.1254`. An information criterion
computed on a per-customer mean is not comparable with one computed on a sum,
so those two numbers cannot be compared with each other — which is the one
thing an AIC is for. This package reports `11682.6547` from the unpenalised
sum. Both relationships are asserted in
`TestRegularizationAgainstTheVignette`, so the deviation cannot drift into
being an accident.

**The R documentation prints z-values for parameters it says have none.** The
same vignette's coefficient tables give `r`, `alpha`, `s` and `beta` z- and
p-values, which contradicts §6.4.1 — a null of zero "lies outside the
admissible parameter space" — and CLVTools' own `?pnbd`, which states the
indicators "are set to NA on purpose". This package follows the paper and
reports `NaN`.

**Several examples in the R documentation are stale.**
`?predict.clv.fitted.transactions` prints two prediction end dates,
`2010-11-28` and `2016-12-17`, that are unreachable from the code beside them:
the comment says "the 37 weeks fitting period" while the call passes
`estimation.split = 52`, and ten weeks past either estimation end on
`apparelTrans` is 2006-03-12 or 2011-02-28. Both `?summary.clv.data` examples
name customers — `"1219"`, and `"1000"` — that are not in `apparelTrans`, whose
ids run 1..600; CLVTools answers with a table of `Inf`, `-Inf` and `NaN` and a
warning. `ClvData.summary(ids=...)` raises instead. None of these printed
values are used as oracles here, and `docs/audit.md` records why.

**Three defects that static analysis found, once it was turned on.** A
non-raw edit had written literal control characters into `pnbd/dyncov.py`'s
`PAlive` docstring, so the LaTeX for that equation read `\x0crac{\x07lpha_0}`
rather than `\frac{\alpha_0}` — invisible in a diff, and wrong in the rendered
maths. Separately, `build_walks` computed the transactional covariate grid and
then discarded it: every walk's interval indices came from the *lifetime* grid
and were used to slice both covariate matrices, so a transactional series on a
different grid, or one too short for the walks it had to cover, would have been
sliced silently into misalignment and returned a wrong but finite likelihood.
Both cases now raise, and both are covered by
`TestWalkConstructionValidation`. The unused variable that pointed at them was
the linter's `RUF059`.

**Two places this implementation reaches a better optimum than CLVTools 0.12.1.**
Its *correlated* Pareto/NBD fit attains −5850.82 against −5848.10 for its own
uncorrelated fit — impossible at a true optimum, since `m = 0` nests it — because
`m` is pinned on its lower Sarmanov bound. And its time-varying covariate fit is
0.31 log-likelihood units below this one. In both cases the two implementations
agree about the *likelihood function* to nine or more significant figures at
fixed parameters; only the optimisation differs.

**The GGom/NBD is the Pareto/NBD in disguise on this data.** Its fitted `b` is
8.1e-07. That is *not* simply the `b → 0` limit, which describes an immortal
customer; since $\beta - 1 + e^{bT} \approx \beta + bT$, the Pareto/NBD is
recovered along $\beta = b\beta_P$. The fitted parameters sit on that path —
`beta / b` is 46.72 against the Pareto/NBD's `beta` of 46.8844 — so the fifth
parameter buys nothing and AIC charges for it.

**`F2.2` in the time-varying likelihood is structurally zero.** Its two
hypergeometrics are evaluated at identical arguments, because the auxiliary walk
spans exactly $t_x$ to $T$ by construction.

**The paper's §6.4.2 table cannot be reproduced by CLVTools 0.12.1 either.**
Its time-varying covariate prediction prints `PAlive = 0.0139206` for customer
1; CLVTools 0.12.1 predicts 0.0107292 from its own fit, and this package
reproduces *that* to 1e-12. The likelihood agrees to nine significant figures
at fixed parameters, so what moved is where the optimiser stops, not what it is
optimising. `tests/test_pnbd_dyncov_predict.py` pins both the agreement and the
gap.

**CLVTools predicts from a *correlated* Pareto/NBD with the uncorrelated
expressions.** The Sarmanov correlation enters estimation only: `PAlive` and
`CET` are the plain ones evaluated at the fitted `(r, α, s, β)`. Checked
against its internal per-customer entry points, the difference is exactly zero,
so the correlation reaches a prediction only through the parameter estimates.

**Standard errors under an equality constraint were misaligned here.** The
covariate Hessian was taken over the full *unconstrained* parameter vector,
which is one longer than the vector actually estimated, so `life.Channel` was
being handed `life.Gender`'s standard error. Differencing over the reported
parameters instead reproduces CLVTools' seven values.

**A Hessian step of 1e-5 was too small.** These log-likelihoods are around
−5800 while their second differences are around 1e-2, so the central difference
cancels four significant figures before dividing. At 1e-5 the standard errors
were wrong in the fourth digit; at a *relative* 1e-4 they agree with
`numDeriv`'s to about 1e-4. Relative rather than absolute also keeps the
GGom/NBD's `b = 8.1e-07` from being stepped negative, where the likelihood does
not exist.

**`continuous.discount.factor` defaults to an unscaled annual rate.** CLVTools
uses `log(1.1)` per *period* regardless of the time unit; §6.3.2 is explicit that
scaling is the caller's job. On weekly data the raw default discounts 52 times
too fast. `discount_factor()` does the scaling.

**A regularized fit's standard errors are almost entirely the penalty's.**
They are now differenced on the objective that was actually minimised — eq.
(13)'s penalised *mean* — rather than on the unpenalised sum, so that the
estimates and their standard errors describe the same function. The consequence
is worth knowing before trusting them. Dividing the likelihood by `n` and
leaving the penalty unscaled makes the implied prior 600 times stronger than the
printed equation reads, so the curvature a covariate coefficient sees is
`2·lambda` plus a per-customer term of order 1e-2. At `lambda = 10` the four
apparel coefficients come out at 0.2231, 0.2152, 0.2227 and 0.2228 against a
prior-only `1/sqrt(2·lambda) = 0.2236` — the data has moved them by under 4%.
Two of the four are *larger* than the unregularized fit's 0.1041 and 0.1049,
which no account of shrinkage explains and the arithmetic does. At
`lambda = 0.1` the data is visible again, 5% to 27% below the prior-only value.
`TestRegularizedStandardErrorsUseThePenalisedObjective` pins both ends, and
`standard_errors()` now warns on a regularized fit saying exactly this — which
neither CLVTools nor the paper does.

**Asked for the first time, CLVTools' regularized `vcov` cannot be followed.**
Nothing had ever requested one, so the port had nothing to check against. It
gives, on the apparel cohort at `lambda = 10`, four covariate variances that are
*identical to twelve significant figures* (0.007580647473) while their
off-diagonals differ — which no curvature computed from data can be — and
standard errors that are **not monotone in the penalty**: 0.1303, 0.0871,
0.0913, 0.0853 at `lambda` = 1, 10, 40, 100. Both properties are asserted in R
by `tools/oracle/generate_interface_fixtures.R` before it writes the fixture, so
the evidence travels with the number. This package therefore deviates
deliberately here, as it already does for the regularized AIC and BIC: its own
value, 0.2231, is `1/sqrt(2·lambda)` to within 4% and moves with `lambda`;
CLVTools' 0.0871 corresponds to no `lambda` and does not. The two agree on the
estimates — the log-likelihood matches to 1e-3 and the model parameters to 1% —
so the disagreement is confined to the standard errors, and
`TestTheRegularizedVcovDisagreementIsPinned` keeps it measured.

**Regularization penalises the mean, not the sum.** Eq. (13) shows the penalty
applied to the summed likelihood; the implementation divides by `n`, so
`logLik()` on a regularized fit reports about −9.7 rather than −5821.
`unpenalised_log_likelihood` is provided for anything comparable across models.

**The BG/NBD's beta parameters are barely identified with covariates.** `a_i`
and `b_i` are scaled by the *same* `exp(γ'x)`, so the data pins their ratio far
better than their size. CLVTools stops at `a + b ≈ 38,600`; a derivative-free
polish climbs to about 2.5 million for a gain of 3e-4 in the log-likelihood.
Neither is wrong — the direction is nearly flat.

**Warm-starting a regularized fit is not universally right.** It is necessary on
the Pareto/NBD, where a cold start converges in a clearly worse basin. It is
harmful on the BG/NBD, whose unpenalised optimum sits far out on that same
ridge while the penalised one is back near `a + b = 10`. Regularized fits now
run from both starting points and keep the better.

**lubridate returns `NA` for a leap day plus n years.** So CLVTools cannot
express an estimation split in years from a 29 February start. Its own
`time_length` is more forgiving and treats the anniversary as 1 March, which is
the convention taken here in both directions, so `add` and `elapsed` stay
mutually inverse.

**The convergence tolerances were tighter than double precision, and only
macOS forgave it.** `_optimize` set `ftol = 1e-16` for L-BFGS-B. SciPy turns
that into `factr = ftol / eps = 0.45`, a demand for a relative reduction better
than machine epsilon, which no line search can report satisfying; `gtol = 1e-14`
was unreachable for the same reason, since the gradient of an objective of order
5e3 cannot be resolved below about 1e-9. The only exits left were an impossible
test and a failed line search. On macOS/ARM the reduction reached exactly zero
and the fits reported success; on x86-64 Linux the line search failed first and
the *same optimum, to twelve significant figures*, came back with
`converged = False`. The first CI run found it — a Gamma-Gamma fit on CDNOW at
`p = 7.4875, q = 3.5829, gamma = 12.2457` in both places, `success` differing.
The tolerances are now `ftol = 1e-14` (`factr = 45`, still 200,000x tighter than
SciPy's default) and `gtol = 1e-10`, and every published number is unchanged.

**The last printed digit of a Pareto/NBD estimate is not portable.** The same
ridge that moves the estimate 3e-5 for 1e-10 of log-likelihood moves it about
1e-4 to 2e-3 between macOS/ARM and x86-64 Linux, depending on the tolerance the
search stops at — `beta` lands on 46.8837 here and 46.8815 there, and the
z-value of a covariate coefficient on −2.1721 against −2.1722. Every doctest
that printed a fitted quantity was asserting one of those digits. They now elide
the last one (`46.88...`, `-22.9...`), so what is printed is what both platforms
agree on and the elision marks where agreement stops; quantities computed at
*published* parameters, like `gg.py`'s 39.1372, stay exact, because a fixed
input is reproducible by construction.

One precision rule now covers the suite: an estimate is compared to no better
than 1e-3 relative, a log-likelihood tightly (the two platforms agree to 9e-10
on −5848), "at least as good as the oracle" carries 1e-6 of slack, and no test
asserts a printed digit of an estimate. It took four red CI runs to arrive at
it, because a failing doctest aborts its file and hides every failure after it.

**A partly covered period gets no observed count.** The tracking plot's grid
runs one period past the data so the last period is shown whole; CLVTools
reports `NA` for its observed value rather than the fraction it has, and so does
this.

Two defects in this package's own code were caught the same way. The `hyp2f1`
fallback summed its series in a Python loop, costing over a second per call as
`z → 1`, which made a degenerate fit appear to hang. And SciPy's Nelder-Mead
builds a microscopic initial simplex at an all-zeros log-parameter start, from
which it reported successful convergence on the Gamma-Gamma at a local optimum
34 log-likelihood units below the published one. Both have regression tests.

## Testing

```bash
uv run pytest                  # 906 tests, including doctests in src/ and docs/
uv run pytest -m paper         # 24 numbers printed in the paper
uv run pytest -m rdoc          # 22 numbers printed in the R package's docs
uv run pytest -m oracle        # 231 checks against R CLVTools fixtures
uv run pytest -m slow          # 138 full-dataset MLE fits
uv run pytest -m dyncov_fit    # the time-varying covariate MLE; ~10 minutes
uv run pytest --cov=clvtools --cov-report=term-missing
uv run pytest -m quality       # lint, complexity and size gates (run by default)
uv run pytest -m performance   # vectorisation and O(n) gates (run by default)
```

100% line coverage of `src/`. The time-varying covariate fit is deselected by
default; everything else runs in about three minutes.

Static analysis is one of the tests rather than a separate command, so there is
a single way to be green. `ruff` runs over `src/`, `tests/`, `tools/` and
`docs/` with complexity and design limits — mccabe 10, 50 statements, 12
branches, 12 arguments — and a module-size limit counted in *code* lines,
docstrings excluded: 37% of `src/` is docstring by design, and a raw line count
would rank the best-documented modules worst. Every threshold was measured
against this codebase rather than taken from a default, so each sits just above
what the code needs and trips on a regression.

Efficiency is gated the same way, and without a clock. `tests/test_performance.py`
counts *operations*: `hyp2f1_ratio` runs twice per likelihood evaluation over all
*n* customers rather than *n* times on scalars, its scalar series fallback stays
cold (0 of 252,000 elements in a `fit_pnbd` on the apparel data), the tightened
`ftol` is shown to buy a better optimum than SciPy's default rather than merely
more evaluations -- a comparison made within whatever platform is running it,
after the first CI run showed the old fixed band of 200-400 was a statement
about one libm -- the work per customer is
identical at 1,178 and 2,357 customers -- O(*n*) by construction rather than by
stopwatch -- and the time-varying covariate likelihood stays batched over
covariate intervals, four hypergeometric dispatches per customer rather than
one per interval, with both branches of the hypergeometric exercised. Wall-clock stays in `tools/benchmark.py` and the question of *where*
the time goes in `tools/profile.py`, both reported and neither asserted;
`docs/performance.md` explains why a timing assertion would be the first flaky
gate here, and carries the profiles that `tools/profile.py` regenerates.

The suite is layered: published numbers, agreement with the reference
implementation expression by expression, internal cross-checks between
independently derived equations (mixing the individual-level expressions
numerically to reproduce the marginalised ones), nesting relationships the paper
asserts (zero covariate effects recover the standard model; `m = 0` recovers
independence), and every worked example in the docs.

"Published" means the R package's own documentation as well as the paper. Its
vignettes print a constrained covariate table, a regularized one and an
`lrtest()` that the paper never prints, and `?pmf` prints a fitted PMF table
with the empirical frequencies beside it — on CDNOW, which the man pages use
throughout where the paper uses the apparel data alone. Those are checked under
`-m rdoc` and collected in `tests/rdoc_values.py`; `docs/vignette.md` walks the
same ground as an executable document.

`docs/audit.md` records what was compared against the paper and the R package,
and what each gap turned into.

## Run times

Appendix B benchmarks CLVTools across sample sizes, on data simulated from
`r = 1, alpha = 0.5, s = 1, beta = 0.5` with L-BFGS-B and no Hessian.
`tools/benchmark.py` does the same here:

```bash
uv run python tools/benchmark.py --sizes 1000 10000 100000
```

| Customers | Weeks | No covariate | Time-invariant | CLVTools (Table 5) |
|---|---|---|---|---|
| 1,000 | 52 | 0.07 s | 0.32 s | 0.19 s / 0.26 s |
| 10,000 | 52 | 0.32 s | 1.45 s | 0.32 s / 0.57 s |
| 100,000 | 52 | 6.32 s | 8.91 s | 0.62 s / 1.34 s |

Comparable at ten thousand customers and several times slower at a hundred
thousand, on a different machine from the paper's. The likelihood is vectorised
over customers; what grows is the number of NumPy passes per evaluation against
CLVTools' single C++ sweep.

Where that time goes is a second question, and `tools/profile.py` answers it —
a cProfile summary of `summary()`, `fit_pnbd`, `fit_pnbd_staticcov` and one
evaluation of the time-varying likelihood, emitted as markdown with call counts
and shares rather than seconds so two versions diff cleanly:

```bash
uv run tools/profile.py
```

`docs/performance.md` is its output plus the reading of it. The vectorised
models sit at the floor of what SciPy costs. The time-varying covariate
likelihood did not -- it spent 0.328 s of interpreter overhead on 600,000 calls
for one number, because each of a customer's ~66 covariate intervals took its
own scalar trip through the hypergeometric. Batching those intervals into array
work made an evaluation 3.3-5.1x faster, with 27 of the 30 oracle intermediates
bit-identical and the rest moving in the sixteenth significant digit. The fit
fell by only 1.33x, 13:27 to 10:07, and that gap is the more interesting
result: two thirds of a fit is spent at parameters where 84% of the time is
inside `scipy.special.hyp2f1` and there is no interpreter overhead left to
remove. `docs/performance.md` has the deciles and the argument.

## Dependencies

NumPy, SciPy and pandas — nothing else. The R package is a test-time oracle
invoked out-of-process; nothing in `src/` depends on it.

`clvtools.diagnostics.render()` will draw the diagnostic frames with matplotlib
if you install the `plot` extra, but the frames those functions return are
useful on their own, so it stays out of the core.

## Data

`src/clvtools/data/` holds the datasets bundled with CLVTools 0.12.1, exported
to CSV. They live inside the package rather than beside it so that an installed
wheel carries them:
`apparelTrans` (3,187 transactions from 600 customers, one acquisition cohort
whose first purchase was 2005-01-02), `apparelStaticCov`, `apparelDynCov`,
`apparelDynCovFuture` — the covariate series continued into the prediction
window, which §6.4.2 needs — and `cdnow`.

The paper is [arXiv:2602.09845](https://arxiv.org/abs/2602.09845). Its source
and PDF are not redistributed here — arXiv's non-exclusive licence covers
arXiv's distribution, not ours — and `.gitignore` carries the two commands that
fetch them to the paths the docstrings cite.

## Licence

GPL-3.0-only. See [`LICENSE`](LICENSE).

The code here is written from scratch against the paper rather than translated
from CLVTools: R never enters `src/`, and the oracle is invoked out-of-process
by the fixture generators. The licence nonetheless matches CLVTools 0.12.1's
own, because the package redistributes the five datasets CLVTools bundles.
`cdnow` is the CDNOW cohort of Fader and Hardie (*Interfaces*, 2001) and
predates it; the three `apparel*` tables come from CLVTools itself.
