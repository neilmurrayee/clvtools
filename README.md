# clvtools

A from-scratch, test-driven Python implementation of the **CLVTools** R package,
following Meierer, Bachmann, Näf, Schilter & Algesheimer, *"Estimating
Individual Customer Lifetime Values with R: The CLVTools Package"* (Journal of
Statistical Software, submission 5634).

**[🚀 Quick start notebook →](examples/quickstart.ipynb)** &nbsp;·&nbsp; **[📄 The paper's case study, executable →](docs/paper.md)** &nbsp;·&nbsp; **[📘 The R package's own walkthrough →](docs/vignette.md)**

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

### Deliberately not ported

Six things an audit comparing this against CLVTools' `NAMESPACE` would flag,
each a decision or a non-gap rather than an omission — recorded here because an
audit cannot otherwise tell the three apart.

* **The BG/BB model** (`bgbb()`) — **not a gap: CLVTools has not implemented it
  either.** The paper states that BG/BB "is not currently included in CLVTools,
  which currently focuses on continuous-time probabilistic models with
  closed-form marginal likelihoods", and 0.12.1 bears that out. It exports
  `bgbb` and registers three method signatures — plain, static-covariate,
  dynamic-covariate — whose man page is titled "BG/BB models - Work In
  Progress" and reads "Not yet implemented… No value is returned"; calling it
  raises `This model has not yet been implemented!`. This README said the
  opposite until 2026-09-03, on the evidence that `args(bgbb)` returns a full
  fitting signature. It does; the body is a `stop()`. Reading a signature as
  behaviour is the same mistake as the two in the findings below.
* **`as.clv.data()`**, R's coercion generic. `ClvData(...)` is the only
  constructor, and Python has no dispatch-on-coercion idiom that would make a
  second spelling of it worth having.
* **`predict(newdata = ...)` as a keyword.** Applying a fit to another set of
  customers is supported — the data object is `predict()`'s first positional
  argument, so it is simply `predict(other_data, params)` — but there is no
  separately named `newdata` parameter to pass the fitting data back in.
* **`predict.spending = TRUE`.** R will fit a Gamma-Gamma of its own when
  asked for spending columns; here `spending_params` takes a fitted model or
  nothing, so the spending fit is always the caller's and always visible.
* **A specification-carrying bootstrap.** `clv.bootstrapped.apply` re-fits with
  the original model's own arguments, which is what makes "did it silently drop
  `use.cor`?" a question worth testing. Here `apply` receives the resampled data
  and does its own fitting, so no specification is held anywhere to be dropped.
* **A named-parameter likelihood accessor.** Every likelihood function here
  takes its parameters positionally, in the paper's own order.

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

Then [`examples/quickstart.ipynb`](examples/quickstart.ipynb) walks the whole
thing in seventeen code cells — data, descriptive plots, both fits, prediction,
diagnostics, covariates and bootstrap intervals — in about five seconds. It is
committed with its outputs stripped and **executed by the test suite**, so it
cannot drift from what the code does; run it with `uv run jupyter lab
examples/quickstart.ipynb`, which needs the `plot` extra for the figures.

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
R_LIBS=.Rlib Rscript tools/oracle/generate_cdnow_fixtures.R      # the CDNOW fit, pmf, frequencies
R_LIBS=.Rlib Rscript tools/oracle/generate_time_fixtures.R       # S5's calendar arithmetic
R_LIBS=.Rlib Rscript tools/oracle/generate_dyncov_fixtures.R     # slow: fits dyncov twice
```

Every committed fixture is reachable from that list. Two were not until
2026-09-03: `time_elapsed.csv` and `time_add_periods.csv` came from CLVTools'
`clv.time` classes by a script that was never committed with them, and the
generator above was reconstructed from the grid they describe — it reproduces
both **byte for byte**.

`setup_oracle.sh` never touches your system R library. CRAN's macOS binaries lag
the newest R release, so it falls back through recent series; the 4.5 build
installs and runs correctly under R 4.6.

The generators assert their own conventions before writing anything — the C++
entry points take **log-scale** model parameters, return the **negated** sum, and
order static-covariate arguments life-then-trans, while `pnbd_nocov_expectation`
transposes the middle pair relative to every sibling. Each fixture family is
checked against a public generic (`logLik()`, `coef()`, `predict()`) so a sign or
ordering slip cannot ship as a plausible-looking expectation.

Two oracle classes stand outside R. The papers the models come from —
Fader, Hardie & Lee (2005) for the Pareto/NBD and the BG/NBD, Fader & Hardie
(2013) for the Gamma-Gamma — publish estimates and log-likelihoods on CDNOW at
the `1997-09-30` split, and `-m literature` reproduces all five, each compared
to half a unit in the last decimal the source printed. And a class of claim
that determines its own answer needs no oracle at all: with covariate effects
set to zero the covariate models *are* the plain ones, a spending model's
transaction count *is* the Pareto/NBD's, and a bootstrap that draws every
customer once *is* the original data. `tests/test_invariants.py` asserts those
bit for bit, which is stronger than any fixture comparison can be.

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

**A start value of 1 is a claim about the time unit, and on hourly data it is
four orders of magnitude wrong.** CLVTools starts every parameter at 1, and this
port did too. On weekly data the Pareto/NBD's $\alpha$ is about 49, so the
search starts four e-folds out and gets there; on the *same data read hourly* it
is about 8,171, and L-BFGS-B **stops 223 log-units short of the optimum, at a
degenerate $s = 0.0011$, and reports `converged = True`**. The GGompertz/NBD was
louder and raised; the BG/NBD, with one mis-scaled coordinate rather than three,
was fine. Spec item `F-12` asks that fits work on hourly data, and one of the
three did.

The likelihoods are *exactly* invariant to the unit — scale $t_x$ and $T$ by $c$
and the log-likelihood moves by $-(\sum_i x_i)\log c$, verified to the twelfth
decimal — so the optimum is knowable in advance and the failure is the
optimiser's alone. The fix puts the default start's scale parameters at the
average observation window instead of at 1, which is 1 exactly when CLVTools'
convention was already right; a start the caller gives is left alone.
Normalising the *data* instead was tried and is more principled, but it divides
the objective's magnitude by that same Jacobian, and the absolute `gtol` then
returns the weekly fit as `ABNORMAL` on the identical optimum. A better start
also subsumes part of what the tight `ftol` was doing: from the scaled start,
SciPy's own loose `ftol` reaches the same optimum in fewer evaluations, where
from all-ones it lands 1.5e-5 worse.

**Three published GGompertz/NBD `(b, β)` pairs, four orders of magnitude apart,
all with the same likelihood.** On CDNOW, CLVTools reports `(1.1e-6, 1.3e-5)`,
Bemmaor & Glady (2012) `(2.0e-4, 2.6e-3)`, and this port `(1.19e-4, 1.39e-3)` —
`b` spanning a factor of 180 — while the log-likelihood moves in the fourth
decimal. The survival term is $(\beta/(\beta - 1 + e^{bT}))^{s}$, and for
$bT \ll 1$ that is the Pareto/NBD's own survival with $\beta_P = \beta/b$: **the
identified quantity is the ratio**, and all three agree on it within 11% and
with the nested Pareto/NBD's $\beta = 11.67$. Neither published pair reproduces
its own published likelihood, because rounding to two significant figures is a
5% move along this direction. So the tests assert the ratio and the likelihood
and not the coordinates, and `s` is asserted only as a spread — it tilts along
the same ridge, moving 0.001 for 9e-7 of log-likelihood.

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

**The built wheel carried no datasets, and no test in a checkout could see
it.** `DATA_DIR` resolved `__file__/../../../data`, which is the repository root
in a source tree and a directory that does not exist under `site-packages`, so
`load_apparel_trans()` — the README's own first usage line — raised
`FileNotFoundError` on any installed copy. Every test here passed throughout,
because a checkout always has the files whether or not they are packaged. The
data now lives inside the package, and
`tests/test_data.py::TestTheDatasetsShipWithThePackage` asserts both halves of
what that costs to prevent: `DATA_DIR` is inside the package directory, and
every CSV is reachable through `importlib.resources` the way an installed
package reaches it.

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

**The GGom/NBD's `CET` is the post-erratum one, not the original MATLAB's.**
CLVTools' own suite records in a comment that this expression changed after
Adler's erratum (issue #206) and no longer agrees with the code the model was
first published with. This package follows the erratum, as CLVTools does, so a
reader comparing against the original paper will find a disagreement that is
neither implementation's mistake. The value was pinned to 1e-6 and the *reason*
was written down nowhere — the fourth case round 5 found of a deliberate choice
recorded in a test but not in these findings, which is half of this repository's
own rule. Spec M-09.

**The GGom/NBD is the Pareto/NBD in disguise on this data.** Its fitted `b` is
8.1e-07. That is *not* simply the `b → 0` limit, which describes an immortal
customer; since $\beta - 1 + e^{bT} \approx \beta + bT$, the Pareto/NBD is
recovered along $\beta = b\beta_P$. The fitted parameters sit on that path —
`beta / b` is 46.72 against the Pareto/NBD's `beta` of 46.8844 — so the fifth
parameter buys nothing and AIC charges for it.

**`F2.2` in the time-varying likelihood is structurally zero.** Its two
hypergeometrics are evaluated at identical arguments, because the auxiliary walk
spans exactly $t_x$ to $T$ by construction.

**A heavy buyer's time-varying likelihood silently became the alive-only one.**
Every term of $F_2$ carries $\alpha^{-(r+s+x)}$, so past $x = 160$ to $190$
depending on the walk all of them are below float64 and $F_2$ was exactly
zero — which
selected the $\log F_0 + \log F_3$ branch, the likelihood of a customer who is
certainly dead, with no signal. $F_3$ underflows at the same rate, so the ratio
being discarded is $O(1)$: at $x = 200$ the answer was wrong by **225
log-units** and `PAlive` was reported as **exactly 1.0** where the truth is
1.6e-98. CLVTools arranges the arithmetic the same way and underflows in the
same place, so its fixtures agree with the broken version by construction, and
the apparel cohort's largest $x$ is 21 — no fixture reaches the regime at all.
$F_2$ is now combined as a log magnitude and a sign throughout, and the check
is the nesting §3.3 asserts: with zero coefficients this model *is* the
standard Pareto/NBD, whose likelihood is closed-form at any $x$, and the two
agree to 1e-12 out to $x = 400$. The intermediates table keeps the value form,
so a term below float64 still prints as zero — as CLVTools' does — and only
the likelihood is formed from the logs.

**The PMF's closed form was quietly wrong long before it was visibly wrong.**
`pmf` computes the die-inside-the-window term as `b1 - b2`, and those two are
nearly equal: at `alpha = 500, beta = 1, s = 1.5, T = 52` the share of `b1`
surviving the subtraction falls from 3.6e-8 at `k = 10` to 5.5e-16 at `k = 18`,
so **sixteen leading digits cancel** — the whole of float64 — and twenty-three
by `k = 25`. `np.log` of what was left returned `NaN` from `k = 18`, which is
where anyone would have noticed. The interesting part is what happened before
that: measured against a 50-digit evaluation of the same closed form, the
relative error was 2.4e-8 at `k = 10`, 6.5e-5 at `k = 14` and **1.0e-3 at
`k = 16`** — wrong in the third decimal, finite, and silent. `b2` turns out to
be the first `k+1` terms of a convergent series whose full sum is `b1`, so
`b1 - b2` is the *tail* of that series, and a tail of positive terms has
nothing to cancel: summed directly it is right to 1e-12 or better everywhere,
including where the subtraction returned `NaN`. The subtraction is kept where
little cancels, which is every published table — at the paper's own parameters
the surviving share stays above 2.5e-3 out to `k = 20`, so `-m paper` and
`-m rdoc` are unmoved. CLVTools arranges the arithmetic the same way and
cancels in the same place, so no fixture could see any of it; the reference is
`mpmath` at 50 digits.

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

**A categorical covariate could not be selected by its own name.** S6.4 turns
one into k-1 dummies, so a column `Region` with three levels becomes `Region_b`
and `Region_c` — and the requested names were checked against the *encoded*
frame, so `names_cov_life=["Region"]` reported "covariates not in the data" of a
column plainly in the data. Nothing caught it because every covariate in the
apparel cohort is 0/1 **numeric** and keeps its name through the encoding, and
every test here uses that cohort. A name now expands to its dummies, a
single-level column says it carries no information rather than that it is
absent, and a genuinely missing one still says so. Spec C-09.

**A fractional `estimation_split` is honoured and not warned about.**
`estimation_split=37.5` ends the estimation period at 12:00 on its day; R warns
about partial periods. It is the same choice recorded above for a fractional
`prediction_end`, and for the same reason — this package carries partial periods
elsewhere — so it is listed rather than fixed. Spec T-05 and T-10.

**Three input-checking places where R is stricter, and one where it is not.**
`start_cov` is a **single scalar** applied to every covariate coefficient, where
CLVTools takes a named vector with one entry per covariate — so five of the
seven failure modes its own tests cover (an unnamed entry, a duplicate, a name
that is not a covariate, a covariate left out) cannot arise here at all. The
same goes for `time_unit`, which defaults to `"week"` where R requires it, and
for `plot`'s `label`, `other.models` and `annotate.ids`, which have no
counterpart because `diagnostics` returns frames and leaves rendering to the
caller. None of those are gaps to close; each is a smaller surface with fewer
ways to be wrong, and they are listed so an audit against R's input checks can
tell a decision from an omission. Spec V-02, V-07 and V-08.

**Where a non-finite argument used to be blamed on the model.** `nan <= 0` is
`False`, so a `NaN` in `start`, in `start_cov` or in `reg_lambdas` passed every
positivity check and reached the objective — which then reported *"the objective
is not finite at the point the search started"*, a statement about the model or
the data for a fault in the argument. All three now name themselves where they
are given. Spec V-01, V-02 and X-14.

**The two time-varying covariate series may run to different dates.** R
requires that "if one covariate's data is longer than the other's, all data must
be that long"; here they may differ. The weakening is deliberate and is safe
only because the questions equal length would have answered are asked directly
instead, at the three points that matter: the *lifetime* grid must reach the
estimation end, since every walk's interval indices are derived from it and then
used to slice both matrices; the *transaction* grid must cover the walks it is
sliced for; and the prediction horizon must be reachable from whichever series
runs shortest. Three checks where they bite rather than one blanket rule at the
door. Spec C-08.

**Two more formula spellings, one of which needed no `I()` to begin with.**
`FI-06` asks for `~ I(Gender + 1) | log(Gender + 2)` — note that only the first
term is wrapped. In R, `I()` protects *operators* from the formula grammar; a
bare call is not formula syntax and needs no protection at all. This package
supported `I(...)` and refused `log(Gender + 2)` with "covariates not in the
data", which reads as a typo in a term that is not one. Both now evaluate, and
the coefficient carries the term exactly as written — R deparses and respaces,
nothing here reformats. `FI-07`'s interactions are new too: `*` gives main
effects and their product, `:` the product alone, and the product is named
`Gender.Channel`, which is how `make.names` renders R's `Gender:Channel`. A real
column wins over any reading of its name, so a covariate someone called
`log(spend)` still selects.

**Two formula spellings R has that this does not, and one it shares.**
S6.4's formula is `~ life | trans`, and three constructs inside it needed
deciding. `.` expands to every covariate the data carries, *including* beside
other terms — `~ . | . + I(Gender + 1)` selects all of them on the attrition
side and all plus the transformation on the transaction side; it used to read
the `.` as a literal column name and go looking for a covariate called `"."`.
`.` also takes exclusions, `~ . - Gender | .`, which arrived as the single term
`". - Gender"` for the same reason. Both now work. **`constraint()` does not**:
R names tied covariates inside the formula, this package takes them as
`names_cov_constr=['Gender']`, and the formula now says so rather than reading
`constraint(Gender)` as a column name. The capability is identical, the
spelling is not, and `TestRsConstraintSyntaxIsRefusedNotMisread` pins the
refusal alongside the argument that replaces it. Spec FI-04 and FI-15.

**A formula's narrowed data shares its frames; R's copies.** `with_covariates`
hands the narrowed object the same transaction and covariate frames, where
CLVTools' formula interface copies so that the result shares no storage with
the input. The boundary that matters is guarded: `ClvData.__init__` copies the
caller's frame, so nothing a caller holds is reachable from a fitted object,
and what is shared is one of this package's objects with a narrowed descendant
of itself. Copying 187,800 covariate rows on every formula call to prevent an
in-place mutation of an internal attribute is not a trade worth making. Spec
FI-09, pinned in both directions by
`TestNarrowingKeepsTheClassAndSharesTheFrames`.

**Covariate names are kept verbatim where R mangles them.** CLVTools passes
column names through `make.names()`, so a covariate called `my var!` becomes
`my.var.` and the coefficient is reported under a name the caller never wrote.
This package keeps it: `ClvDataStaticCov` accepts it, the formula interface
parses `~ my var! | my var!`, and `summary()` reports `life.my var!`. Renaming
someone's column silently is the kind of helpfulness that costs an afternoon
when the coefficient you are looking for is not there. Recorded rather than
matched — spec C-05.

**Timezone-aware dates are refused, having previously been half-supported.**
A numeric estimation split built a usable object whose spans came from
`total_seconds()`, so a daylight-saving transition inside the observation window
moved recency by an hour; a date split raised pandas' own "Cannot compare
tz-naive and tz-aware timestamps" from three frames down. R never faces this —
its `Date` carries no zone — so there is no oracle and this is a decision.
Refusing is the safe one: dropping the zone silently would move a late-evening
transaction to the previous day. The error names both conversions, and a test
checks that the route it recommends actually works.

**A fractional `prediction_end` means what it says here, and is truncated in
R.** `prediction.end = 14.4` gives CLVTools a 14-period window and a warning —
"may not indicate partial periods. Digits after the decimal point are cut off"
— while this package predicts 14.4 periods, ending two days later. Both are
defensible and they are not the same: CLVTools' grid is whole periods, and this
one already carries partial periods elsewhere (the tracking plot's last period
is partial by construction). The capability is kept and the divergence recorded
rather than silently inherited, because code moving from R gets a different
window with no warning. Spec T-22.

**Zero-length horizons are answers, not errors.** `predict(prediction_end=0)`
and `newcustomer(0)` both raised where CLVTools returns a zero-length window
with `CET = 0` and the value `1` respectively — the latter being §6.3.4's "+1 to
account for all transactions that a prospective customer will make, including
the first one", so over zero periods they make exactly that one. Both now match,
checked against CLVTools 0.12.1. Negative horizons are still refused. Spec
PR-05, NC-02.

**The discount factor's range was wrong in both directions, and a test pinned
it that way.** CLVTools admits `[0, 1)`; this admitted `(0, inf)`, so `0` was
refused where R returns the undiscounted expectation (`Inf`, correctly — a
customer who may never die has unbounded residual value) and `100` was accepted
silently, returning a number for a per-period discount rate of 10,000%. Every
boundary was checked against CLVTools 0.12.1 rather than reasoned about, and it
errors at 1.0 with "needs to be in the interval [0,1)". The test covering this
asserted that *zero is rejected* — our divergence rather than the claim — which
is why nothing caught it.

**`continuous.discount.factor` defaults to an unscaled annual rate.** CLVTools
uses `log(1.1)` per *period* regardless of the time unit; §6.3.2 is explicit that
scaling is the caller's job. On weekly data the raw default discounts 52 times
too fast. `discount_factor()` does the scaling.

**`reg_lambdas=(0, 0)` was not the same fit as no regularization.** A zero
weight still selected eq. (13)'s *mean* objective, so the likelihood was divided
by `n` before the penalty of zero was added to it. The estimates were
unaffected — scaling an objective does not move its optimum — but the Hessian
was 1/600 of the unregularized one, and every standard error came back
`sqrt(600) = 24.5` times too large: `r` at 8.47 against 0.346. Nothing caught it
because the test for this claim compared a single log-likelihood at `abs=1e-4`,
while CLVTools' own suite asserts that `lambda = 0` reproduces the coefficient
vector *and* the summary table. A weight that contributes nothing should not
change which objective is minimised, so it no longer does, and the test now
compares both.

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
uv run pytest                  # 1,244 tests, including doctests in src/ and docs/
uv run pytest -m paper         # 22 numbers printed in the paper
uv run pytest -m rdoc          # 22 numbers printed in the R package's docs
uv run pytest -m literature    # 14 numbers published in the CLV literature
uv run pytest -m oracle        # 249 checks against R CLVTools fixtures
uv run pytest -m slow          # 157 full-dataset MLE fits
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
