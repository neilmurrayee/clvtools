r"""S6.3 - predicting customer lifetime value.

S6.3: "Recall that latent attrition models predict customers' future number of
orders, while spending models infer the average value per order for each
customer. Their predictions therefore need to be combined to obtain a measure
of future spending and of customer lifetime value."

The combined table carries, per customer:

===============================  ==============================================
``PAlive``                       probability of being alive at the estimation end
``CET``                          expected transactions over the prediction period
``DERT``                         discounted expected residual transactions
``predicted.mean.spending``      expected value of each future transaction
``predicted.period.spending``    ``CET`` x mean spending
``predicted.CLV``                ``DERT`` x mean spending
``actual.x``                     transactions actually observed, if in holdout
``actual.period.spending``       spending actually observed, if in holdout
===============================  ==============================================

S6.3: "Further, if the predictions are made no further than the end of the
holdout period, the true number of transactions ("actual.x") and true spending
("actual.total.spending") during the prediction period are reported. This allows
for a convenient evaluation with common metrics such as the root mean square
error (RMSE) and the mean absolute error (MAE)."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clvtools import timeunit
from clvtools.data import ClvData, ClvDataStaticCov
from clvtools.gg import GgParams, expected_mean_spending
from clvtools.pnbd.aggregate import (
    conditional_expected_transactions,
    discounted_expected_residual_transactions,
    probability_alive,
)
from clvtools.pnbd.fit import PnbdParams
from clvtools.pnbd.staticcov import PnbdStaticCovParams, alpha_i, beta_i

__all__ = ["DEFAULT_DISCOUNT_FACTOR", "discount_factor", "predict"]

#: CLVTools' default, :math:`\ln(1 + 0.10)`.
#:
#: .. warning::
#:    This is an *annual* rate applied without reference to the data's time
#:    unit. S6.3.2 is explicit that scaling is the caller's job: "If the
#:    ``time.unit`` chosen in ``clvdata()`` takes any value other than
#:    "yearly", the continuous rate must be scaled down by the number of its
#:    occurrences per year." With weekly data the unscaled default discounts 52
#:    times too fast. Use :func:`discount_factor` rather than this constant
#:    unless you are deliberately matching CLVTools' default.
DEFAULT_DISCOUNT_FACTOR = float(np.log(1 + 0.10))




def discount_factor(annual_rate: float, time_unit: str = "week") -> float:
    r"""The per-period continuous discount factor for a discrete annual rate.

    .. math::
        \delta_k = \frac{\ln(1 + d)}{k}

    S6.3.2: "The natural logarithm appears because continuous compounding
    models growth as :math:`e^{\delta}`; equating this to the discrete one-year
    growth factor :math:`1+d` and solving for :math:`\delta` gives
    :math:`\delta=\ln(1+d)`", and then :math:`k` is "the number of time units
    per year (e.g. :math:`k = 52` for weekly, :math:`k = 365` for daily units)".

    Examples
    --------
    S6.3.2 predicts with a 7.5% annual rate on weekly data:

    >>> import numpy as np
    >>> bool(np.isclose(discount_factor(0.075), np.log(1.075) / 52))
    True
    """
    periods = timeunit.get(time_unit).periods_per_year
    if annual_rate <= -1:
        raise ValueError("annual_rate must exceed -1")
    return float(np.log1p(annual_rate) / periods)


def _resolve_prediction_end(
    clv_data: ClvData, prediction_end: int | float | str | pd.Timestamp | None
) -> pd.Timestamp:
    """The last date the prediction covers.

    S6.3: "If there is a holdout period, the prediction is made by default until
    the end of the holdout period." Without one, S6.3.2 requires an explicit
    horizon: "the argument ``prediction.end`` must be given to specify how far
    into the future to predict."
    """
    if prediction_end is None:
        if not clv_data.has_holdout:
            raise ValueError(
                "prediction_end is required when the data has no holdout period"
            )
        return clv_data.data_end
    if isinstance(prediction_end, (int, float, np.integer, np.floating)):
        if prediction_end <= 0:
            raise ValueError("prediction_end must be a positive number of periods")
        return clv_data.time.add(clv_data.estimation_end, float(prediction_end))
    return pd.Timestamp(prediction_end)


def _actuals(clv_data: ClvData, first: pd.Timestamp, last: pd.Timestamp,
             ids: pd.Index) -> pd.DataFrame:
    """Transactions and spending observed in the prediction window."""
    window = clv_data.transactions[
        (clv_data.transactions["Date"] >= first)
        & (clv_data.transactions["Date"] <= last)
    ]
    counts = window.groupby("Id").size().reindex(ids).fillna(0).astype(int)
    spend = (
        window.groupby("Id")["Price"].sum().reindex(ids).fillna(0.0)
        if clv_data.has_spending
        else None
    )
    out = pd.DataFrame({"actual.x": counts.to_numpy()}, index=ids)
    if spend is not None:
        out["actual.period.spending"] = spend.to_numpy()
    return out



def _model_rates(clv_data: ClvData, params) -> dict:
    r"""``(r, alpha, s, beta)`` for the expressions, per customer if needed.

    With covariates each customer has their own :math:`\alpha_i, \beta_i`
    (S3.3), built from the design matrices on the data object. The rest of the
    prediction is then identical, which is the point of the extension.
    """
    if isinstance(params, PnbdStaticCovParams):
        if not isinstance(clv_data, ClvDataStaticCov):
            raise ValueError(
                "a covariate model needs covariate data: build the data with "
                "ClvDataStaticCov before predicting"
            )
        return {
            "r": params.r,
            "alpha": alpha_i(
                params.alpha, params.gamma_trans,
                clv_data.design_trans(params.names_cov_trans),
            ),
            "s": params.s,
            "beta": beta_i(
                params.beta, params.gamma_life,
                clv_data.design_life(params.names_cov_life),
            ),
        }
    return params.as_dict()


def predict(
    clv_data: ClvData,
    params: PnbdParams,
    spending_params: GgParams | None = None,
    prediction_end: int | float | str | pd.Timestamp | None = None,
    continuous_discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
) -> pd.DataFrame:
    r"""The combined prediction table of S6.3. Cf. ``predict()``.

    Parameters
    ----------
    clv_data
        The transaction data the model was fitted on, or -- as S6.3.1's
        ``newdata`` argument allows -- another set of customers to apply the
        fitted parameters to.
    params
        A fitted Pareto/NBD, with or without time-invariant covariates. A
        covariate model requires ``clv_data`` to be a
        :class:`~clvtools.data.ClvDataStaticCov`.
    spending_params
        A fitted Gamma-Gamma. Without it the spending columns are omitted, as
        ``predict.spending = FALSE`` does in S6.3.1.
    prediction_end
        A number of periods, or a date. Defaults to the end of the holdout
        period, and is required when there is none.
    continuous_discount_factor
        Used for ``DERT`` only. See :data:`DEFAULT_DISCOUNT_FACTOR` for the
        scaling trap, and :func:`discount_factor` to avoid it.

    Notes
    -----
    S6.3.2: "the prediction horizon (``prediction.end``) is only considered for
    metrics ``CET`` and ``predicted.period.spending``, while the value of
    ``continuous.discount.factor`` only affects ``DERT`` and
    ``predicted.CLV``. ``PAlive`` is unaffected by both parameters as it
    describes customers at the end of the estimation period."

    Examples
    --------
    S6.3.2 fits on all the data and predicts 95 weeks ahead at a 7.5% annual
    rate, printing a table whose first three rows this reproduces:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.gg import fit_gg
    >>> from clvtools.pnbd import fit_pnbd
    >>> data = ClvData(load_apparel_trans(), time_unit="week")
    >>> cbs = data.customer_summary()
    >>> spend = data.spending_summary()
    >>> pnbd = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
    >>> gg = fit_gg(spend["x"], spend["Spending"])
    >>> table = predict(data, pnbd, gg, prediction_end=95,
    ...                 continuous_discount_factor=discount_factor(0.075))
    >>> list(table.columns)
    ['period.first', 'period.last', 'period.length', 'PAlive', 'CET', 'DERT',
     'predicted.mean.spending', 'predicted.period.spending', 'predicted.CLV']
    >>> table["period.first"].iloc[0].date(), table["period.last"].iloc[0].date()
    (datetime.date(2010, 12, 21), datetime.date(2012, 10, 15))

    The paper prints ``PAlive = 0.007191623``, ``CET = 0.01300226``,
    ``DERT = 0.06200625`` and ``predicted.CLV = 4.823691`` for customer 1:

    >>> row = table.loc["1"]
    >>> [round(float(row[c]), 4) for c in ("PAlive", "CET", "DERT")]
    [0.0072, 0.013, 0.062]
    >>> round(float(row["predicted.mean.spending"]), 3)
    77.794
    >>> round(float(row["predicted.CLV"]), 2)
    4.82

    The last digits move because the parameters come from this package's own
    optimiser, which stops at a marginally different point on the Pareto/NBD's
    flat ridge (see :func:`~clvtools.pnbd.fit.fit_pnbd`). Given the published
    parameters instead, every column reproduces to 1e-13.

    The two spending columns are definitional, and hold exactly:

    >>> import numpy as np
    >>> bool(np.allclose(table["predicted.CLV"],
    ...                  table["DERT"] * table["predicted.mean.spending"]))
    True
    >>> bool(np.allclose(table["predicted.period.spending"],
    ...                  table["CET"] * table["predicted.mean.spending"]))
    True
    """
    last = _resolve_prediction_end(clv_data, prediction_end)
    if last <= clv_data.estimation_end:
        raise ValueError(
            f"the prediction period ends {last.date()}, on or before the "
            f"estimation period ends {clv_data.estimation_end.date()}"
        )
    first = clv_data.estimation_end + pd.Timedelta(days=1)
    length = clv_data.time.elapsed(clv_data.estimation_end, last)

    cbs = clv_data.customer_summary().set_index("Id")
    x, t_x, T = cbs["x"].to_numpy(), cbs["t_x"].to_numpy(), cbs["T"].to_numpy()
    model = _model_rates(clv_data, params)

    table = pd.DataFrame(
        {
            "period.first": first,
            "period.last": last,
            "period.length": length,
        },
        index=cbs.index,
    )

    # Actuals come first, matching the column order CLVTools emits.
    within_holdout = clv_data.has_holdout and last <= clv_data.data_end
    if within_holdout:
        table = table.join(_actuals(clv_data, first, last, cbs.index))

    table["PAlive"] = probability_alive(x, t_x, T, **model)
    table["CET"] = conditional_expected_transactions(x, t_x, T, length, **model)
    table["DERT"] = discounted_expected_residual_transactions(
        x, t_x, T, continuous_discount_factor, **model
    )

    if spending_params is not None:
        if not clv_data.has_spending:
            raise ValueError(
                "spending was requested but the transaction data has no Price column"
            )
        spend = clv_data.spending_summary().set_index("Id").loc[cbs.index]
        mean_spending = expected_mean_spending(
            spend["x"].to_numpy(),
            spend["Spending"].to_numpy(),
            **spending_params.as_dict(),
        )
        table["predicted.mean.spending"] = mean_spending
        # S6.3: total spending expected in the prediction period is CET times
        # mean spending; CLV is DERT times mean spending.
        table["predicted.period.spending"] = table["CET"] * mean_spending
        table["predicted.CLV"] = table["DERT"] * mean_spending

    return table
