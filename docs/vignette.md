# The CLVTools walkthrough, in Python

The R package ships four vignettes of its own, and they are not the paper. This
document follows two of them:

* `doc/CLVTools.Rmd` — the walkthrough: load, inspect, fit, predict, plot, then
  covariates and spending.
* `doc/CLVTools_advanced_techniques.pdf` — regularization, correlation,
  equality constraints and the likelihood ratio test.

Every code block is a doctest that `pytest` executes, so the numbers shown are
what the code actually returns.

```bash
uv run pytest docs/vignette.md
```

[`docs/paper.md`](paper.md) is the companion document for §6 of the paper. The
two overlap deliberately — both fit a Pareto/NBD to the same apparel data — but
the vignettes reach places the paper never does. The advanced-techniques
vignette prints a **constrained** covariate table, a **regularized** one and an
`lrtest()`, none of which appear in the paper; those printed values are checked
against this implementation under `-m rdoc`, and the numbers below are the same
ones. `?pmf` and most of the man pages work on CDNOW rather than on the apparel
data, which is why [`tests/test_cdnow.py`](../tests/test_cdnow.py) exists.

## Contents

| Topic | Source | Module |
|---|---|---|
| [The data](#the-data) | walkthrough | `clvtools.data` |
| [Fitting](#fitting-a-model) | walkthrough | `clvtools.estimate` |
| [What a fit reports](#what-a-fit-reports) | walkthrough | `clvtools.inference` |
| [Predicting](#predicting) | walkthrough | `clvtools.predict` |
| [Describing the data](#describing-the-data) | walkthrough | `clvtools.diagnostics` |
| [Contextual factors](#contextual-factors) | walkthrough | `clvtools.pnbd.staticcov` |
| [Equality constraints](#equality-constraints) | advanced | `clvtools._staticcov` |
| [Regularization](#regularization) | advanced | `clvtools._staticcov` |
| [Spending](#spending) | walkthrough | `clvtools.gg` |

---

## The data

> "The `apparelTrans` dataset […] consists of 600 customers who purchased for
> the first time from this business on the day of 2005-01-02."

`clvdata()` is spelled `ClvData`. The walkthrough splits at 104 weeks.

```python
>>> from clvtools import ClvData, load_apparel_trans
>>> transactions = load_apparel_trans()
>>> data = ClvData(transactions, time_unit="week", estimation_split=104)
>>> data
ClvData(600 customers, 3183 transactions, weeks, estimation 2005-01-02..2006-12-31, holdout to 2010-12-20)

```

3,187 records become 3,183 transactions: purchases by the same customer on the
same day are one transaction, which is what the models are defined on.

`summary()` returns the descriptive table as a frame rather than printing it,
so any cell can be read out:

```python
>>> table = data.summary()
>>> rows = ["Number of customers", "Total # Transactions",
...         "Percentage of zero repeaters"]
>>> print(table.loc[rows].to_string())
                             Estimation Holdout   Total
Number of customers                None    None   600.0
Total # Transactions             1866.0  1317.0  3183.0
Percentage of zero repeaters       35.5    None    None

```

`summary()` also takes customer ids, as `?summary.clv.data` does — the same
table restricted to the customers named:

```python
>>> print(data.summary(ids=["1", "10", "100"]).loc[rows].to_string())
                             Estimation Holdout Total
Number of customers                None    None   3.0
Total # Transactions               11.0    10.0  21.0
Percentage of zero repeaters  33.333333    None  None

```

## Fitting a model

`latentAttrition()` is `latent_attrition()`, and dispatches on the data object
exactly as the R generic does. With no covariates, no formula is given.

```python
>>> from clvtools import latent_attrition, pnbd as pnbd_family
>>> fit = latent_attrition(family=pnbd_family, data=data)
>>> {name: round(value, 4) for name, value in fit.coefficients.items()}
{'r': 1.4489, 'alpha': 48.63..., 's': 0.56..., 'beta': 46.88...}

```

## What a fit reports

Every generic the vignette calls — `summary()`, `coef()`, `confint()`,
`logLik()`, `AIC()`, `BIC()`, `vcov()` — is available on the fit.

```python
>>> print(fit.summary().round(4).to_string())
        Estimate  Std. Error  z-val  Pr(>|z|)
r  1.44...  0.24...  NaN  NaN
alpha  48.63...  7.48...  NaN  NaN
s  0.56...  0.27...  NaN  NaN
beta  46.88...  35.6...  NaN  NaN

```

The estimates are shown with their last digit elided. It is not portable: the
Pareto/NBD's ridge puts `beta` at 46.8837 on macOS/ARM and 46.8830 on x86-64
Linux for the same data and the same code, and a doctest that asserts that digit
asserts a fact about a C library. See the README's findings; `-m paper` and
`-m rdoc` compare the same quantities with tolerances, which is where the
precision lives.

The z- and p-values are `NaN` on purpose. §6.4.1: these four parameters are
"constrained to be strictly positive", so "a null hypothesis of $\theta = 0$
lies outside the admissible parameter space". Covariate coefficients, which are
unconstrained, do carry them — see below.

```python
>>> [round(float(v), 4) for v in (fit.log_likelihood, fit.aic, fit.bic)]
[-5848.0978, 11704.1957, 11721.7834]
>>> print(fit.confint().round(3).to_string())
        2.5 %   97.5 %
r  0.97...  1.92...
alpha  33.95...  63.31...
s  0.03...  1.09...
beta  -22.9...  116.6...

```

The interval on `beta` runs below zero, which a rate cannot be. It is a Wald
interval on a parameter whose standard error is most of its own size, and the
vignette's own advice applies: with `beta` this weakly identified, prefer the
bootstrap intervals of §6.3.3.

## Predicting

`predict()` needs a spending model as well if it is to produce a CLV.

```python
>>> from clvtools import spending, predict, gg as gg_family
>>> gg = spending(family=gg_family, data=data)
>>> [round(v, 4) for v in gg]
[3.099, 5.6537, 56.50...]
>>> columns = ["PAlive", "CET", "DERT", "predicted.mean.spending", "predicted.CLV"]
>>> print(predict(data, fit, gg)[columns].head(3).round(4).to_string())
     PAlive     CET    DERT  predicted.mean.spending  predicted.CLV
Id                                                                 
1  0.94...  7.32...  0.46...  88.64...  41.45...
10  0.98...  3.51...  0.22...  41.21...  9.25...
100  0.27...  0.41...  0.02...  37.62...  1.00...

```

Omitting the spending model omits its columns, which is `predict.spending =
FALSE`.

## Describing the data

The walkthrough plots before it fits anything. Each plot here is a frame; the
drawing is `clvtools.diagnostics.render()`, which needs the `plot` extra.

```python
>>> from clvtools.diagnostics import frequency_data, interpurchase_time_data
>>> print(frequency_data(data).head(4).to_string(index=False))
num.transactions  num.customers
               0            213
               1            116
               2             82
               3             63

```

213 zero repeaters is the 35.5% the summary reports. Interpurchase times exist
only for customers who came back, so the frame is shorter than the cohort:

```python
>>> gaps = interpurchase_time_data(data)
>>> len(gaps), round(float(gaps["mean.interpurchase.time"].mean()), 4)
(387, 24.8229)

```

## Contextual factors

`SetStaticCovariates()` is `ClvDataStaticCov`. With covariates present, the
formula names them, attrition process first.

```python
>>> from clvtools import ClvDataStaticCov, load_apparel_static_cov
>>> static = ClvDataStaticCov(
...     data, load_apparel_static_cov(),
...     names_cov_life=["Gender", "Channel"],
...     names_cov_trans=["Gender", "Channel"],
... )
>>> covariate_fit = latent_attrition(
...     formula="~ . | .", family=pnbd_family, data=static)
>>> print(covariate_fit.summary().round(4).to_string())
        Estimate  Std. Error  z-val  Pr(>|z|)
r  1.83...  0.34...  NaN  NaN
alpha  92.96...  16.97...  NaN  NaN
s  0.59...  0.26...  NaN  NaN
beta  49.51...  36.14...  NaN  NaN
life.Gender  -0.64...  0.29...  -2.17...  0.02...
life.Channel  0.78...  0.30...  2.58...  0.00...
trans.Gender  0.28...  0.10...  2.74...  0.00...
trans.Channel  0.62...  0.10...  5.94...  0.00...

```

This is the table §6.4.1 prints, and the covariate rows now carry z- and
p-values. A formula term wrapped in `I(...)` is an expression rather than a
column name, as in R:

```python
>>> derived = static.with_covariates(["Gender"], ["I(log(Channel + 2))"])
>>> derived.names_cov_trans
['I(log(Channel + 2))']

```

## Equality constraints

> "it is possible to test whether the parameter value of the covariate Gender
> is the same for both processes."

`names_cov_constr` forces one coefficient across both processes — eq. (14).
The constrained covariate is then reported once, as `constr.Gender`:

```python
>>> constrained = latent_attrition(
...     formula="~ . | .", family=pnbd_family, data=static,
...     names_cov_constr=["Gender"])
>>> print(constrained.summary().round(4).to_string())
        Estimate  Std. Error  z-val  Pr(>|z|)
r  1.79...  0.33...  NaN  NaN
alpha  94.73...  17.22...  NaN  NaN
s  0.42...  0.14...  NaN  NaN
beta  59.06...  34.5...  NaN  NaN
life.Channel  1.02...  0.35...  2.88...  0.00...
trans.Channel  0.63...  0.10...  5.99...  0.00...
constr.Gender  0.32...  0.10...  3.05...  0.00...

```

Seven parameters where the unconstrained fit had eight, so the two are nested
and a likelihood ratio test decides between them:

```python
>>> from clvtools.inference import likelihood_ratio_test
>>> test = likelihood_ratio_test(constrained, covariate_fit)
>>> test
LikelihoodRatioTest(df=1, chisq=10.94, p=0.0009396)
>>> test.n_parameters_restricted, test.n_parameters_unrestricted
(7, 8)

```

The vignette's conclusion: the constraint "significantly worsened the model
fit", so the effect of `Gender` genuinely differs between the two processes.
Every figure above matches what the vignette prints — including the p-value.

## Regularization

> "The larger $\lambda_{reg}$, the stronger the effect of the regularization
> while a value of 0 results in no regularization."

Weights are given per process. The vignette uses `trans = 0.1, life = 0.2`;
this package takes them in the order `(life, trans)`.

```python
>>> regularized = latent_attrition(
...     formula="~ . | .", family=pnbd_family, data=static,
...     reg_lambdas=(0.2, 0.1), hessian=False)
>>> {n: round(v, 5) for n, v in regularized.coefficients.items()}
{'r': 1.73..., 'alpha': 69.79..., 's': 0.53..., 'beta': 39.73..., 'life.Gender': -0.04..., 'life.Channel': 0.02..., 'trans.Gender': 0.17..., 'trans.Channel': 0.23...}

```

The lifetime process carried the heavier weight and both of its coefficients
have been shrunk to near zero, while the transaction coefficients survive.

**Two traps here, both worth stating plainly.** With a penalty applied,
`log_likelihood` is the penalised *mean* objective, not a log-likelihood — the
vignette prints `LL -9.7313` for this fit. The unpenalised value is what
compares across models:

```python
>>> round(regularized.log_likelihood, 4)
-9.7313
>>> round(regularized.unpenalised_log_likelihood, 4)
-5833.3274

```

And CLVTools computes AIC and BIC from that penalised mean, printing `AIC
35.4626` for this model against `AIC 11658.1254` for the same model
unregularized. Two information criteria on different scales cannot be compared
with each other, which is the one thing an information criterion is for, so
this package computes them from the unpenalised sum instead:

```python
>>> round(regularized.aic, 4)
11682.6547

```

That is a deliberate deviation from CLVTools; it is pinned by a test and
recorded in the README's findings.

## Spending

The walkthrough re-splits the data at 40 weeks for the spending model.

```python
>>> data_40 = ClvData(transactions, time_unit="week", estimation_split=40)
>>> gg_40 = spending(family=gg_family, data=data_40)
>>> print(gg_40.summary().round(4).to_string())
        Estimate  Std. Error  z-val  Pr(>|z|)
p  3.67...  0.76...  NaN  NaN
q  4.62...  0.80...  NaN  NaN
gamma  36.11...  13.83...  NaN  NaN

```

§6.2.3: CLVTools "by default does not use the first transaction when estimating
a spending model because in many cases this transaction has been found to be
atypical for future purchases". `remove_first_transaction=False` keeps it, which
is what a prediction counting the initial purchase needs.

```python
>>> kept = spending(family=gg_family, data=data_40,
...                 remove_first_transaction=False, hessian=False)
>>> all(v > 0 for v in kept) and kept.converged
True

```
