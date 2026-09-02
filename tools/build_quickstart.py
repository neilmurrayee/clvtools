"""Build examples/quickstart.ipynb from cells defined here.

The notebook is committed with its outputs stripped and executed by the test
suite, so nothing in it can drift from what the code does -- the same rule the
docstrings and docs/*.md live under.
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (MD, """\
# clvtools quick start

Fitting a customer-lifetime-value model to a transaction log, end to end, on
the apparel cohort that ships with the package: 3,187 purchases from 600
customers whose first purchase was 2005-01-02.

Everything here is a section of Meierer, Bachmann, Näf, Schilter &
Algesheimer, *"Estimating Individual Customer Lifetime Values with R: The
CLVTools Package"*. The section numbers are given as we go, and
[`docs/paper.md`](../docs/paper.md) works through the whole case study.

This notebook is executed by the test suite, so the numbers below are what the
code returns rather than what it returned once.\
"""),
    (CODE, """\
import matplotlib.pyplot as plt
import numpy as np

import clvtools
from clvtools import (
    ClvData,
    ClvDataStaticCov,
    diagnostics,
    gg,
    latent_attrition,
    load_apparel_static_cov,
    load_apparel_trans,
    pnbd,
    predict,
    spending,
)
from clvtools.predict import discount_factor

plt.rcParams["figure.figsize"] = (9, 3.5)\
"""),
    (MD, """\
## 1. The data — §6.1

A transaction log is three columns: who, when, and (optionally) how much.
`ClvData` adds the one modelling decision that matters, where the estimation
period ends and the holdout begins.\
"""),
    (CODE, """\
transactions = load_apparel_trans()
transactions.head()\
"""),
    (CODE, """\
data = ClvData(transactions, time_unit="week", estimation_split=104)
data\
"""),
    (MD, """\
### What the data looks like — §6.1.2

`summary()` is the 39-cell table the paper prints: counts, spending and
interpurchase times, split across estimation, holdout and total.\
"""),
    (CODE, """\
data.summary()\
"""),
    (MD, """\
The per-customer view a latent-attrition model actually consumes is three
numbers each — frequency `x`, recency `t_x`, and the observation window `T`.\
"""),
    (CODE, """\
cbs = data.customer_summary()
cbs.head()\
"""),
    (MD, """\
### Descriptive plots — Table 3

Five frames describe the data before any model is fitted. Each returns a
`DataFrame`; `diagnostics.render()` draws it if matplotlib is installed.\
"""),
    (CODE, """\
frequency = diagnostics.frequency_data(data)
fig, axes = plt.subplots(1, 2)
axes[0].bar(frequency["num.transactions"], frequency["num.customers"])
axes[0].set(xlabel="repeat transactions", ylabel="customers",
            title="How often people buy")

gaps = diagnostics.interpurchase_time_data(data)
axes[1].hist(gaps["mean.interpurchase.time"], bins=30)
axes[1].set(xlabel="mean weeks between purchases", ylabel="customers",
            title="How long they wait")
fig.tight_layout()\
"""),
    (MD, """\
## 2. Fitting — §6.2

`latent_attrition()` estimates the transaction and attrition processes;
`spending()` estimates order value. Both dispatch on the data object, so
adding covariates later changes nothing about how they are called.\
"""),
    (CODE, """\
fit = latent_attrition(family=pnbd, data=data)
fit.summary()\
"""),
    (MD, """\
The paper reports `r = 1.4490, alpha = 48.6361, s = 0.5613, beta = 46.8844`.
The last digits differ between machines — this likelihood has a long flat
ridge, and two optimisers stop at different points on it — which is why the
tests compare with tolerances rather than by printing digits.

`z-val` and `Pr(>|z|)` are `NaN` on purpose: §6.4.1 notes that a null of zero
"lies outside the admissible parameter space" for these four.\
"""),
    (CODE, """\
spend = spending(family=gg, data=data)
spend.summary()\
"""),
    (MD, """\
## 3. Predicting — §6.3

`predict()` gives the per-customer table: the probability each customer is
still alive, their expected transactions, the discounted residual value, and
the CLV that follows.

`discount_factor()` converts an annual rate to the per-period one — without it
CLVTools' default discounts at `log(1.1)` *per week* on weekly data, which is
52 times too fast.\
"""),
    (CODE, """\
predicted = predict(data, fit, spend,
                    continuous_discount_factor=discount_factor(0.10))
predicted.head()\
"""),
    (CODE, """\
fig, axes = plt.subplots(1, 2)
axes[0].hist(predicted["PAlive"], bins=30)
axes[0].set(xlabel="P(alive)", ylabel="customers",
            title="Who is still a customer")
axes[1].scatter(predicted["CET"], predicted["actual.x"], s=8, alpha=0.4)
limit = max(predicted["CET"].max(), predicted["actual.x"].max())
axes[1].plot([0, limit], [0, limit], linewidth=1, color="black")
axes[1].set(xlabel="predicted transactions", ylabel="actual (holdout)",
            title="Predicted against observed")
fig.tight_layout()\
"""),
    (MD, """\
How good is that? The holdout period is the honest test, and the paper reports
`mae.cet = 2.039532`.\
"""),
    (CODE, """\
error = predicted["CET"] - predicted["actual.x"]
{"mae": round(float(np.abs(error).mean()), 4),
 "rmse": round(float(np.sqrt((error ** 2).mean())), 4)}\
"""),
    (MD, """\
### A customer who has not bought yet — §6.3.4

`newcustomer()` asks what a prospective customer is worth, using no history
because there is none.\
"""),
    (CODE, """\
from clvtools import newcustomer, newcustomer_spending

full = ClvData(load_apparel_trans(), time_unit="week")
fit_full = latent_attrition(family=pnbd, data=full, hessian=False)
spend_full = spending(family=gg, data=full,
                      remove_first_transaction=False, hessian=False)

transactions_year_one = predict(newcustomer(52), fit_full)
per_order = predict(newcustomer_spending(), spend_full)
{"transactions in year one": round(transactions_year_one, 4),
 "average order value": round(per_order, 4),
 "first-year value": round(transactions_year_one * per_order, 2)}\
"""),
    (MD, """\
## 4. Does the model fit? — §6.2.2

Two diagnostics. The tracking plot compares the model's expected transactions
against what happened, week by week; the PMF plot compares the distribution of
purchase counts.\
"""),
    (CODE, """\
# Both diagnostics take the model *expression* rather than the fit, so any
# family -- or a model this package does not implement -- can be plotted the
# same way.
from clvtools.pnbd import expectation, pmf

tracking = diagnostics.tracking_data(
    data, lambda t: expectation(t, **fit.as_dict()), model_name="Pareto/NBD")
wide = tracking.pivot(index="period.until", columns="variable", values="value")

fig, axes = plt.subplots(1, 2)
wide.plot(ax=axes[0], linewidth=1)
axes[0].axvline(data.estimation_end, linestyle="--", linewidth=1, color="grey")
axes[0].set(xlabel="", ylabel="transactions",
            title="Tracking (dashed: holdout begins)")

bins = diagnostics.pmf_data(
    data, lambda k, T: pmf(k, T, **fit.as_dict()), model_name="Pareto/NBD",
    max_transactions=7)
counts = bins.pivot(index="num.transactions", columns="variable", values="value")
counts.plot.bar(ax=axes[1], width=0.8)
axes[1].set(xlabel="transactions in the estimation period", ylabel="customers",
            title="PMF")
fig.tight_layout()\
"""),
    (MD, """\
## 5. Covariates — §6.4

The same two calls, with a formula. `~ Gender + Channel | Gender + Channel`
puts both covariates on the attrition process (left of the bar) and the
transaction process (right).\
"""),
    (CODE, """\
static = ClvDataStaticCov(
    data, load_apparel_static_cov(),
    names_cov_life=["Gender", "Channel"],
    names_cov_trans=["Gender", "Channel"],
)
covariate_fit = latent_attrition(
    formula="~ Gender + Channel | Gender + Channel", family=pnbd, data=static
)
covariate_fit.summary().round(4)\
"""),
    (MD, """\
The covariate coefficients *do* get z-values, because zero is an admissible
null for them: it means the covariate does not matter. Both `Channel` effects
are strongly significant here.

And the scenario question §6.3.4 poses — "region A versus region B" — becomes
answerable for a customer who does not exist yet.\
"""),
    (CODE, """\
from clvtools import newcustomer_static

def scenario(gender, channel):
    values = {"Gender": gender, "Channel": channel}
    return predict(newcustomer_static(52, values, values), covariate_fit)

{f"Gender={g}, Channel={c}": round(scenario(g, c), 3)
 for g, c in ((0, 0), (1, 0), (0, 1), (1, 1))}\
"""),
    (MD, """\
## 6. How certain is any of this? — §6.3.3

Bootstrap the customers, refit, and look at the spread. Twenty draws here to
keep the notebook quick; a real interval wants a few hundred.\
"""),
    (CODE, """\
from clvtools.bootstrap import bootstrap_apply, confidence_intervals

def refit_and_predict(resampled):
    refit = latent_attrition(family=pnbd, data=resampled, hessian=False)
    respend = spending(family=gg, data=resampled, hessian=False)
    return predict(resampled, refit, respend,
                   continuous_discount_factor=discount_factor(0.10))

draws = bootstrap_apply(data, refit_and_predict, num_boots=10, seed=42)
intervals = confidence_intervals(draws, level=0.9, columns=["PAlive", "CET"])
intervals.head()\
"""),
    (MD, """\
Each customer now has a band rather than a point. The width is the honest
answer to "how much of this is the data and how much is the fit", and it is
wide -- which is the argument for reporting it.\
"""),
    (CODE, """\
width = intervals["CET.CI.95"] - intervals["CET.CI.5"]
{"median 90% width, expected transactions": round(float(width.median()), 3),
 "widest": round(float(width.max()), 3)}\
"""),
    (MD, """\
## Where to go next

* [`docs/paper.md`](../docs/paper.md) — the paper's §6 case study, executable.
* [`docs/vignette.md`](../docs/vignette.md) — the R package's own walkthrough.
* `README.md` — the public API table, and the findings: the places where this
  package deliberately differs from CLVTools, and why.

Three families are available wherever `pnbd` appears above: `bgnbd` and
`ggomnbd` fit the same way, and `predict()` reports what each supports.\
"""),
]


def build() -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(body) if kind == MD else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out = pathlib.Path("examples/quickstart.ipynb")
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    print("wrote", build())
