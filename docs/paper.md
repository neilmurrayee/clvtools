# Estimating Individual Customer Lifetime Values with R: The CLVTools Package

Markus Meierer, Patrick Bachmann, Jeffrey Näf, Patrik Schilter, René Algesheimer
*Journal of Statistical Software*, submission 5634

---

This is §6 of the paper — the worked case study — run in Python instead of R.
Every code block below is a doctest that `pytest` executes, so the numbers shown
are what the code actually returns.

```bash
uv run pytest docs/paper.md
```

The equations themselves live next to the functions that implement them; each
module's docstrings carry the paper's section, the quoted justification, and its
own worked examples. Start at `clvtools.pnbd.individual` for the model and
`clvtools.data` for the data layer.

Source: [`arXiv-2602.09845v1/jss5634.tex`](../arXiv-2602.09845v1/jss5634.tex).

## Contents

| § | Topic | Module |
|---|---|---|
| [6.1](#61-data-preparation) | Data preparation | `clvtools.data` |
| [6.2](#62-model-estimation) | Model estimation | `clvtools.pnbd`, `clvtools.gg` |
| [6.3](#63-predicting-customer-lifetime-value) | Predicting CLV | `clvtools.predict` |
| [6.4](#64-covariates) | Covariates | `clvtools.pnbd.staticcov`, `.dyncov` |
| [—](#diagnostics) | Diagnostics and intervals | `clvtools.diagnostics`, `.bootstrap` |
| [6.5](#65-advanced-modelling-techniques) | Advanced techniques | `clvtools.pnbd.correlation` |
| [—](#the-other-two-families) | BG/NBD and GGom/NBD | `clvtools.bgnbd`, `clvtools.ggomnbd` |

### Notation

| Symbol | Meaning |
|---|---|
| $x$ | repeat transactions observed in the estimation period |
| $t_x$ | time of the last repeat transaction |
| $T$ | length of the estimation period for this customer |
| $\lambda, \mu$ | that customer's purchase and attrition rates |
| $r, \alpha$ | gamma shape and rate governing $\lambda$ |
| $s, \beta$ | gamma shape and rate governing $\mu$ |
| $p, q, \gamma$ | Gamma-Gamma spending parameters |

---

## 6.1 Data preparation

> "**CLVTools** requires customers' purchase history as input data […] Every
> transaction record consists of a purchase date and a customer identifier.
> Optionally, the value of the transaction may be included."

The `apparelTrans` dataset is one acquisition cohort: 600 customers who all
first purchased on 2005-01-02.

```python
>>> from clvtools import ClvData, load_apparel_trans
>>> transactions = load_apparel_trans()
>>> len(transactions), transactions["Id"].nunique()
(3187, 600)

```

`ClvData` is the counterpart of `clvdata()`. Splitting at 104 weeks puts the
estimation period end at 2006-12-31 and leaves the rest as holdout.

```python
>>> data = ClvData(transactions, time_unit="week", estimation_split=104)
>>> data.estimation_end.date(), data.data_end.date()
(datetime.date(2006, 12, 31), datetime.date(2010, 12, 20))
>>> data.has_holdout
True

```

Everything the latent attrition models see is $(x, t_x, T)$:

```python
>>> summary = data.customer_summary()
>>> print(summary.head(3).to_string(index=False))
 Id  x       t_x     T date_first_transaction
  1  6 93.285714 104.0             2005-01-02
 10  2 99.571429 104.0             2005-01-02
100  0  0.000000 104.0             2005-01-02

```

The spending model sees only a count and a mean. §6.2.3: "**CLVTools** by
default does not use the first transaction when estimating a spending model
because in many cases this transaction has been found to be atypical."

```python
>>> spending = data.spending_summary()
>>> print(spending.head(3).to_string(index=False))
 Id  x  Spending
  1  6   101.415
 10  2    43.900
100  0     0.000

```

## 6.2 Model estimation

### Pareto/NBD

The paper prints `r = 1.4490, alpha = 48.6361, s = 0.5613, beta = 46.8844`.

```python
>>> from clvtools.pnbd import fit_pnbd
>>> pnbd = fit_pnbd(summary["x"], summary["t_x"], summary["T"], hessian=False)
>>> [round(v, 3) for v in pnbd]
[1.449, 48.635, 0.561, 46.884]
>>> round(pnbd.log_likelihood, 4)
-5848.0978

```

§6.2.1 reads the two average rates straight off: "we observe an average purchase
rate of $r/\alpha = 0.030$ transactions and an average attrition rate of
$s/\beta = 0.012$."

```python
>>> round(pnbd.mean_purchase_rate, 3), round(pnbd.mean_attrition_rate, 3)
(0.03, 0.012)

```

### Gamma-Gamma

The paper prints `p = 3.099, q = 5.654, gamma = 56.504`.

```python
>>> from clvtools.gg import fit_gg
>>> gg = fit_gg(spending["x"], spending["Spending"])
>>> [round(v, 3) for v in gg]
[3.099, 5.654, 56.504]
>>> round(gg.log_likelihood, 3)
-1670.663

```

## 6.3 Predicting customer lifetime value

### Evaluating on holdout data

> "Latent attrition models predict customers' future number of orders, while
> spending models infer the average value per order for each customer. Their
> predictions therefore need to be combined."

```python
>>> from clvtools.predict import predict
>>> holdout = predict(data, pnbd, gg)
>>> list(holdout.columns)
['period.first', 'period.last', 'period.length', 'actual.x',
 'actual.period.spending', 'PAlive', 'CET', 'DERT',
 'predicted.mean.spending', 'predicted.period.spending', 'predicted.CLV']

```

§6.3.1's evaluation. The paper prints `mae.cet = 2.039532` and
`rmse.cet = 3.329395`; CLVTools 0.12.1 itself gives 2.03962 and 3.329425 on the
same data, so the fourth decimal moved between releases.

```python
>>> import numpy as np
>>> error = holdout["CET"] - holdout["actual.x"]
>>> round(float(np.abs(error).mean()), 4)
2.0396
>>> round(float(np.sqrt((error ** 2).mean())), 4)
3.3294

```

### Final predictions without a holdout period

> "To obtain the final, most accurate estimates, we make predictions with a
> model that is estimated on all available purchasing data."

95 weeks ahead, discounting at a 7.5% annual rate. §6.3.2 is explicit that the
scaling by time unit is the caller's job, which `discount_factor` does.

```python
>>> from clvtools.predict import discount_factor
>>> full = ClvData(transactions, time_unit="week")
>>> full_summary, full_spending = full.customer_summary(), full.spending_summary()
>>> pnbd_full = fit_pnbd(full_summary["x"], full_summary["t_x"],
...                      full_summary["T"], hessian=False)
>>> gg_full = fit_gg(full_spending["x"], full_spending["Spending"])
>>> final = predict(full, pnbd_full, gg_full, prediction_end=95,
...                 continuous_discount_factor=discount_factor(0.075))
>>> final["period.first"].iloc[0].date(), final["period.last"].iloc[0].date()
(datetime.date(2010, 12, 21), datetime.date(2012, 10, 15))

```

The paper prints, for customer 1, `PAlive = 0.007191623`, `CET = 0.01300226`,
`DERT = 0.06200625` and `predicted.CLV = 4.823691`:

```python
>>> row = final.loc["1"]
>>> [round(float(row[c]), 4) for c in ("PAlive", "CET", "DERT")]
[0.0072, 0.013, 0.062]
>>> round(float(row["predicted.CLV"]), 2)
4.82

```

The last digits move because the parameters come from this package's optimiser,
which stops at a slightly different point on the Pareto/NBD's flat likelihood
ridge. Given the published parameters instead, every column reproduces to 1e-12
— `tests/test_predict.py` separates the two.

### Prospective customers

> "the unconditional expectation […] gives the expected number of orders for a
> customer for which no information is available. […] We add +1 to the
> unconditional expectation to account for all transactions that a prospective
> customer will make, including the first one."

The paper prints 2.218635 transactions and 39.1372 per order.

```python
>>> from clvtools.gg import expected_mean_spending
>>> from clvtools.pnbd import expectation
>>> transactions_first_year = 1 + float(expectation(52.0, **pnbd_full.as_dict()))
>>> round(transactions_first_year, 4)
2.2186

```

§6.3.4: "the spending model should be fitted on all orders, including the
initial purchases of each customer."

```python
>>> gg_with_first = fit_gg(
...     *full.spending_summary(remove_first_transaction=False)[["x", "Spending"]].T.to_numpy())
>>> per_order = float(expected_mean_spending(0, 0.0, **gg_with_first.as_dict()))
>>> round(per_order, 4)
39.1372
>>> round(transactions_first_year * per_order, 3)
86.832

```

The paper prints 86.83115 for that product; the difference is in the fourth
decimal of `transactions_first_year`, which inherits the flat-ridge difference
in the Pareto/NBD fit.

```python

```

### Diagnostics

> "The key diagnostics for a latent attrition model are two plots: (1) the
> tracking plot and (2) the probability mass function (PMF) plot."

Each returns the data a plot would be drawn from, in the long form CLVTools'
`plot(..., plot = FALSE)` produces. Rendering is separate, and optional.

```python
>>> from clvtools import diagnostics
>>> from clvtools.pnbd import expectation, pmf
>>> tracking = diagnostics.tracking_data(
...     data, lambda t: expectation(t, **pnbd.as_dict()), model_name="Pareto/NBD")
>>> list(tracking.columns)
['period.until', 'variable', 'value']

```

The model series opens at zero — §6.2.2: "this fact gives the plot its
characteristic shape" — and slopes gently down as customers drop out:

```python
>>> model = tracking[tracking["variable"] == "Pareto/NBD"]["value"].to_numpy()
>>> bool(model[0] == 0.0 and np.all(np.diff(model[1:]) < 0))
True

```

The observed count for the final, partly covered period is left missing rather
than reported low:

```python
>>> observed = tracking[tracking["variable"] == "Actual"]["value"]
>>> int(observed.isna().sum())
1

```

The PMF plot compares customer counts by repeat-transaction count. §6.2.2: "the
results illustrate that the model fits the data well".

```python
>>> bins = diagnostics.pmf_data(
...     data, lambda k, T: pmf(k, T, **pnbd.as_dict()), model_name="Pareto/NBD")
>>> wide = bins.pivot(index="num.transactions", columns="variable", values="value")
>>> int(wide.loc["0", "Actual"])
213
>>> bool(abs(wide["Actual"] - wide["Pareto/NBD"]).max() < 30)
True

```

`diagnostics.render(frame)` will draw any of these with matplotlib, which is an
optional extra (`clvtools[plot]`).

### Confidence intervals

> "Customers, together with their entire purchasing history, are sampled with
> replacement. A new model is estimated on the sampled data with the same
> specification and optimization options as the given fitted model."

Two details of §6.3.3 shape this. A customer drawn twice counts as two
customers, and the estimation and holdout periods are pinned — otherwise
resampling would move them, since "the end of the data is determined by the last
order".

```python
>>> from clvtools import bootstrap
>>> resampled = bootstrap.bootstrap_data(data, ["1", "1", "10"])
>>> len(resampled.customer_summary())
3
>>> bool(resampled.estimation_end == data.estimation_end)
True

```

```python
>>> def refit(sample):
...     cbs, spend = sample.customer_summary(), sample.spending_summary()
...     return predict(
...         sample,
...         fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False),
...         fit_gg(spend["x"], spend["Spending"]))
>>> intervals = bootstrap.predict_intervals(
...     data, refit, num_boots=5, level=0.9, columns=["CET"], seed=1)
>>> list(intervals.columns)
['CET.CI.5', 'CET.CI.95']
>>> bool((intervals["CET.CI.5"] <= intervals["CET.CI.95"]).all())
True

```

What these cover is worth keeping in view. §6.3.3: "bootstrapping only accounts
for uncertainty in model parameters (epistemic uncertainty), and not sampling
variability in the actual outcomes (aleatoric uncertainty)."

## 6.4 Covariates

### Time-invariant

> "female customers have a significantly higher purchase rate
> (`trans.Gender = 0.2859`) […] customers acquired offline, coded as 1, purchase
> more (`trans.Channel = 0.6241`) but drop out more quickly
> (`life.Channel = 0.7907`)."

```python
>>> from clvtools import ClvDataStaticCov, load_apparel_static_cov
>>> from clvtools.pnbd.staticcov import fit_pnbd_staticcov
>>> static = ClvDataStaticCov(
...     data, load_apparel_static_cov(),
...     names_cov_life=["Gender", "Channel"],
...     names_cov_trans=["Gender", "Channel"])
>>> covariate_fit = fit_pnbd_staticcov(static)
>>> round(covariate_fit.log_likelihood, 4)
-5821.0627
>>> {k: round(float(v), 2)
...  for k, v in covariate_fit.coefficients().items() if "." in k}
{'life.Gender': -0.64, 'life.Channel': 0.79,
 'trans.Gender': 0.29, 'trans.Channel': 0.62}

```

§6.4.1 leaves the four base parameters without z- or p-values, because "a null
hypothesis of $\theta = 0$ lies outside the admissible parameter space":

```python
>>> table = covariate_fit.summary()
>>> bool(table.loc[["r", "alpha", "s", "beta"], "z-val"].isna().all())
True
>>> (table.loc["trans.Channel", "Pr(>|z|)"] < 0.001).item()
True

```

### Time-varying

> "the model estimation with time-varying covariates is computationally much
> more demanding than the previously detailed alternatives."

The likelihood is built from *walks*: for each span of time, the sequence of
$\exp(\gamma'x_k)$ over the covariate intervals it crosses.

```python
>>> from clvtools import ClvDataDynCov, load_apparel_dyn_cov
>>> names = ["High.Season", "Gender", "Channel"]
>>> dynamic = ClvDataDynCov(data, load_apparel_dyn_cov(),
...                         names_cov_life=names, names_cov_trans=names)
>>> walks = dynamic.walks()
>>> walks.n_customers, walks.n_cov_life
(600, 3)

```

At CLVTools' own fitted parameters the likelihood agrees exactly:

```python
>>> from clvtools.pnbd.dyncov import log_likelihood as dyncov_log_likelihood
>>> round(dyncov_log_likelihood(
...     walks, r=1.977706, alpha=115.177940, s=2.012683, beta=158.181797,
...     gamma_life=[-2.482678, -0.512544, 0.505730],
...     gamma_trans=[0.718314, 0.264898, 0.613721]), 4)
-5752.9367

```

Fitting takes about 17 minutes, so it is marked `dyncov_fit` and deselected by
default:

```bash
uv run pytest -m dyncov_fit
```

## 6.5 Advanced modelling techniques

### Correlated processes

§3.4 uses a Sarmanov distribution to relax independence between $\lambda$ and
$\mu$, giving eq. (12) — the correlated likelihood as four evaluations of the
uncorrelated one. At $m = 0$ it must reproduce the uncorrelated model exactly:

```python
>>> from clvtools.pnbd import log_likelihood
>>> from clvtools.pnbd.correlation import correlated_log_likelihood
>>> args = (summary["x"], summary["t_x"], summary["T"])
>>> params = dict(r=1.4490, alpha=48.6361, s=0.5613, beta=46.8844)
>>> plain = log_likelihood(*args, **params)
>>> correlated = correlated_log_likelihood(*args, **params, m=0.0)
>>> bool(np.isclose(plain, correlated))
True

```

§3.4 warns that $m$ "must not be directly interpreted as a correlation
coefficient"; eq. (13) converts it:

```python
>>> from clvtools.pnbd.correlation import correlation_coefficient
>>> round(correlation_coefficient(1.0, **params), 6)
0.000364

```

### Equality constraints

§6.5.3 forces one covariate to take the same coefficient in both processes, then
uses a likelihood-ratio test to ask whether that costs anything.

```python
>>> constrained = fit_pnbd_staticcov(
...     static, names_cov_constr=["Gender"], hessian=False)
>>> constrained.names
['r', 'alpha', 's', 'beta', 'life.Channel', 'trans.Channel', 'constr.Gender']
>>> from scipy import stats
>>> statistic = 2 * (covariate_fit.log_likelihood - constrained.log_likelihood)
>>> bool(stats.chi2.sf(statistic, df=1) < 0.05)
True

```

The constraint is rejected: gender does act differently on the two processes.

### Regularization

§6.5.1: "The larger this regularization weight, the stronger the effect of the
regularization."

```python
>>> light = fit_pnbd_staticcov(static, reg_lambdas=(0.01, 0.01), hessian=False)
>>> heavy = fit_pnbd_staticcov(static, reg_lambdas=(100.0, 100.0), hessian=False)
>>> def size(fit):
...     return float(np.sum(fit.gamma_life ** 2) + np.sum(fit.gamma_trans ** 2))
>>> bool(size(heavy) < size(light))
True

```

Note that with a penalty applied, `log_likelihood` holds the penalised *mean*
objective — matching what CLVTools' `logLik()` returns — so anything comparable
across models should use `unpenalised_log_likelihood`:

```python
>>> bool(-20 < light.log_likelihood < 0)
True
>>> bool(light.unpenalised_log_likelihood < -5000)
True

```

## The other two families

Table 3 lists two alternatives that share the transaction process and differ in
how attrition is modelled.

```python
>>> from clvtools import bgnbd, ggomnbd

```

The BG/NBD lets a customer drop out only immediately *after* a transaction, so
one with no repeat purchase is alive with certainty — where the Pareto/NBD gives
them about 0.28:

```python
>>> bg = dict(r=0.6073, alpha=20.9567, a=1.2755, b=8.8608)
>>> float(bgnbd.probability_alive(0, 0.0, 104.0, **bg))
1.0
>>> from clvtools.pnbd import probability_alive
>>> round(float(probability_alive(0, 0.0, 104.0, **params)), 4)
0.2784

```

The GGom/NBD's fitted `b` is 8.1e-07 and its log-likelihood matches the
Pareto/NBD's to three decimals. That is not because `b → 0` gives the
Pareto/NBD — there it describes an immortal customer — but because $\beta$
shrinks with it. Since $\beta - 1 + e^{bT} \approx \beta + bT$, the Pareto/NBD is
recovered along $\beta = b\,\beta_P$:

```python
>>> errors = [
...     abs(ggomnbd.log_likelihood(*args, r=1.4490, alpha=48.6361, b=b,
...                                s=0.5613, beta=b * 46.8844) - plain)
...     for b in (1e-3, 1e-5, 1e-7)]
>>> bool(errors[0] > errors[1] > errors[2])
True

```

And the fitted parameters sit on that path — `beta / b` is 46.72 against the
Pareto/NBD's 46.8844.

## Implementation notes

Three transcription errors in the printed equations are worked around, each
noted at the function concerned and each with a test:

- **eq. (14)** writes the per-transaction spending density with $z_i^{r-1}$,
  using the Pareto/NBD's $r$ where the shape $p$ belongs.
- **eq. (17)**'s integral writes $\nu{q-1}$ for $\nu^{q-1}$.
- **eq. (17)**'s result drops the exponent $px$ from its final factor. This one
  matters: as printed the expression is not a density, and its log differs from
  what CLVTools maximises by 3.45 at a representative point.

**Appendix A** writes the integrand's second term as
$\frac{\lambda^{x+1}\mu}{\lambda+\mu}e^{-(\lambda+\mu)T}$, with a stray $\mu$
that eq. (10) in the body does not have. Integrating the appendix version does
not reproduce the closed form the appendix then states; eq. (10) does.
`tests/test_pnbd_individual.py` checks both directions.

Two places where this implementation reaches a **better optimum** than
CLVTools 0.12.1, both documented in tests rather than smoothed over:

- the **correlated** Pareto/NBD, where CLVTools' fit is 2.72 log-likelihood
  units *below* its own uncorrelated fit — impossible at a true optimum, since
  $m = 0$ nests it — because $m$ is pinned on its lower Sarmanov bound;
- the **time-varying covariate** model, by 0.31 units.

In both cases the two implementations agree about the *likelihood function* to
nine or more significant figures at fixed parameters; what differs is where
each optimiser stops.
