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

## Usage

```python
import clvtools
from clvtools import ClvData, latent_attrition, load_apparel_trans, predict, spending

data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)

pnbd = latent_attrition(family=clvtools.pnbd, data=data)
gg = spending(family=clvtools.gg, data=data)

predict(data, pnbd, gg)   # PAlive, CET, DERT, predicted.CLV, and the actuals
pnbd.summary()            # estimates, standard errors, z- and p-values
data.summary()            # §6.1.2's descriptive table
```

The two entry points are the paper's own; `latent_attrition` takes S6.4's
formula (`~ Gender + Channel | Gender + Channel`) and picks the estimator from
the data object's type. The per-family `fit_*` functions underneath take
`(x, t_x, T)` directly.

Bring your own data as a frame of `Id`, `Date` and optionally `Price`.

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
| Pareto/NBD `r, α, s, β` | 1.4490, 48.6361, 0.5613, 46.8844 | 1.449, 48.635, 0.561, 46.884 |
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
uv run pytest                  # 888 tests, including doctests in src/ and docs/
uv run pytest -m paper         # 24 numbers printed in the paper
uv run pytest -m rdoc          # 22 numbers printed in the R package's docs
uv run pytest -m oracle        # 229 checks against R CLVTools fixtures
uv run pytest -m slow          # 138 full-dataset MLE fits
uv run pytest -m dyncov_fit    # the time-varying covariate MLE; ~17 minutes
uv run pytest --cov=clvtools --cov-report=term-missing
uv run pytest -m quality       # lint, complexity and size gates (run by default)
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

## Dependencies

NumPy, SciPy and pandas — nothing else. The R package is a test-time oracle
invoked out-of-process; nothing in `src/` depends on it.

`clvtools.diagnostics.render()` will draw the diagnostic frames with matplotlib
if you install the `plot` extra, but the frames those functions return are
useful on their own, so it stays out of the core.

## Data

`data/` holds the datasets bundled with CLVTools 0.12.1, exported to CSV:
`apparelTrans` (3,187 transactions from 600 customers, one acquisition cohort
whose first purchase was 2005-01-02), `apparelStaticCov`, `apparelDynCov`,
`apparelDynCovFuture` — the covariate series continued into the prediction
window, which §6.4.2 needs — and `cdnow`.

The paper's LaTeX source and PDF are in `arXiv-2602.09845v1/` and
`2602.09845v1.pdf`.
