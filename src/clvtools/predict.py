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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from clvtools import bgnbd, ggomnbd, pnbd, timeunit
from clvtools.data import ClvData, ClvDataStaticCov
from clvtools.gg import GgParams, expected_mean_spending
from clvtools.pnbd.correlation import PnbdCorrelatedParams
from clvtools.pnbd.dyncov import PnbdDynCovParams
from clvtools.pnbd.fit import PnbdParams
from clvtools.pnbd.staticcov import PnbdStaticCovParams, alpha_i, beta_i

__all__ = [
    "DEFAULT_DISCOUNT_FACTOR",
    "discount_factor",
    "newcustomer",
    "newcustomer_dynamic",
    "newcustomer_spending",
    "newcustomer_static",
    "predict",
]

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


@dataclass(frozen=True)
class NewCustomer:
    """A prospective customer with no history. Cf. ``newcustomer()``, S6.3.4.

    S6.3.4: "such predictions are for entirely hypothetical average customers
    with no order history. Therefore, none of the transaction data stored in
    the fitted model is used, nor can any order data be given."
    """

    num_periods: float


@dataclass(frozen=True)
class NewCustomerStatic(NewCustomer):
    """A prospective customer with known covariates. Cf. ``newcustomer.static()``.

    S6.3.4: "If covariates have been included in the model, it is possible to
    account for these and assess various scenarios. For example, to determine
    the difference between prospective customers in region A versus region B."
    """

    cov_life: dict[str, float]
    cov_trans: dict[str, float]


@dataclass(frozen=True)
class NewCustomerDynamic(NewCustomer):
    """A prospective customer on a covariate path. Cf. ``newcustomer.dynamic()``.

    ``cov_life`` and ``cov_trans`` are covariate series for one hypothetical
    customer: a date column and one column per covariate, covering from
    ``first_transaction`` to at least ``num_periods`` beyond it.
    """

    cov_life: pd.DataFrame
    cov_trans: pd.DataFrame
    first_transaction: pd.Timestamp
    time_unit: str = "week"


@dataclass(frozen=True)
class NewCustomerSpending:
    """A prospective customer's order value. Cf. ``newcustomer.spending()``."""


def newcustomer(num_periods: float) -> NewCustomer:
    """A prospective customer observed for ``num_periods``. See :class:`NewCustomer`."""
    if num_periods <= 0:
        raise ValueError("num_periods must be strictly positive")
    return NewCustomer(float(num_periods))


def newcustomer_static(
    num_periods: float, cov_life: dict[str, float], cov_trans: dict[str, float]
) -> NewCustomerStatic:
    """A prospective customer with time-invariant covariates.

    ``cov_life`` and ``cov_trans`` map covariate name to value, one scenario at
    a time.
    """
    if num_periods <= 0:
        raise ValueError("num_periods must be strictly positive")
    return NewCustomerStatic(float(num_periods), dict(cov_life), dict(cov_trans))


def newcustomer_dynamic(
    num_periods: float,
    cov_life: pd.DataFrame,
    cov_trans: pd.DataFrame,
    first_transaction,
    time_unit: str = "week",
) -> NewCustomerDynamic:
    """A prospective customer with time-varying covariates.

    ``time_unit`` must be the unit the model was fitted on; unlike the other
    constructors this one does arithmetic on dates, and the parameters do not
    carry the unit they were estimated in.
    """
    if num_periods <= 0:
        raise ValueError("num_periods must be strictly positive")
    return NewCustomerDynamic(
        float(num_periods), cov_life, cov_trans,
        pd.Timestamp(first_transaction), time_unit,
    )


def newcustomer_spending() -> NewCustomerSpending:
    """The average order value of a prospective customer."""
    return NewCustomerSpending()


def _predict_new_customer(spec, params) -> float:
    r"""S6.3.4's prediction for a customer who has bought nothing yet.

    For a latent attrition model this is :math:`1 + E[X(t)]` -- "We add +1 to
    the unconditional expectation to account for all transactions that a
    prospective customer will make, including the first one." For a spending
    model it is :math:`\gamma p / (q - 1)`, the unconditional mean order value.

    Examples
    --------
    S6.3.4 prints 2.218635 transactions in the first year and 39.1372 per
    order, for a total of 86.83115:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.gg import fit_gg
    >>> from clvtools.pnbd import fit_pnbd
    >>> data = ClvData(load_apparel_trans(), time_unit="week")
    >>> cbs = data.customer_summary()
    >>> fit = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
    >>> transactions = predict(newcustomer(52), fit)
    >>> round(transactions, 4)
    2.2186

    The spending model must be fitted on *all* orders here, since the count
    above includes the first one -- S6.3.4 passes
    ``remove.first.transaction = FALSE`` for exactly this reason:

    >>> spend = data.spending_summary(remove_first_transaction=False)
    >>> gg = fit_gg(spend["x"], spend["Spending"], hessian=False)
    >>> round(predict(newcustomer_spending(), gg), 4)
    39.1372
    >>> round(transactions * predict(newcustomer_spending(), gg), 2)
    86.83
    """
    if isinstance(spec, NewCustomerSpending):
        if not isinstance(params, GgParams):
            raise TypeError(
                "newcustomer_spending() needs a spending model; got "
                f"{type(params).__name__}"
            )
        return float(expected_mean_spending(np.array([0.0]), np.array([0.0]),
                                            **params.as_dict())[0])

    if isinstance(spec, NewCustomerDynamic):
        from clvtools.pnbd.dyncov_predict import new_customer_expectation

        if not isinstance(params, PnbdDynCovParams):
            raise TypeError(
                "newcustomer_dynamic() needs a time-varying covariate model; "
                f"got {type(params).__name__}"
            )
        return 1.0 + new_customer_expectation(
            params, spec.num_periods, spec.first_transaction,
            spec.cov_life, spec.cov_trans, timeunit.get(spec.time_unit),
        )

    family, _ = _family_of(params)
    if isinstance(spec, NewCustomerStatic):
        if not _has_covariates(params):
            raise TypeError(
                "newcustomer_static() needs a covariate model; got "
                f"{type(params).__name__}"
            )
        model = _new_customer_rates(params, spec)
    else:
        if _has_covariates(params):
            raise TypeError(
                "a covariate model needs covariate values: use "
                "newcustomer_static()"
            )
        model = params.as_dict()
    return 1.0 + float(np.ravel(family.expectation(spec.num_periods, **model))[0])


def _new_customer_rates(params, spec: NewCustomerStatic) -> dict:
    """One scenario's rates, from the covariate values it names.

    Scalars, not one-element arrays: this is a single hypothetical customer,
    and each family's ``expectation`` broadcasts its rates against the *times*
    it is asked about.
    """
    def scalar(value) -> float:
        return float(np.ravel(value)[0])

    def row(values: dict[str, float], names: list[str]) -> np.ndarray:
        missing = [n for n in names if n not in values]
        if missing:
            raise ValueError(f"no value given for covariates: {missing}")
        return np.array([[float(values[n]) for n in names]])

    life = row(spec.cov_life, params.names_cov_life)
    trans = row(spec.cov_trans, params.names_cov_trans)

    if isinstance(params, PnbdStaticCovParams):
        return {
            "r": params.r,
            "alpha": scalar(alpha_i(params.alpha, params.gamma_trans, trans)),
            "s": params.s,
            "beta": scalar(beta_i(params.beta, params.gamma_life, life)),
        }
    if isinstance(params, bgnbd.BgnbdStaticCovParams):
        return {
            "r": params.r,
            "alpha": scalar(
                bgnbd.alpha_i(params.alpha, params.gamma_trans, trans)
            ),
            "a": scalar(bgnbd.a_i(params.a, params.gamma_life, life)),
            "b": scalar(bgnbd.b_i(params.b, params.gamma_life, life)),
        }
    return {
        "r": params.r,
        "alpha": scalar(
            ggomnbd.alpha_i(params.alpha, params.gamma_trans, trans)
        ),
        "b": params.b,
        "s": params.s,
        "beta": scalar(ggomnbd.beta_i(params.beta, params.gamma_life, life)),
    }


def _resolve_prediction_end(
    clv_data: ClvData, prediction_end: float | str | pd.Timestamp | None
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



#: One of the paper's expressions, as a family exposes it. The model
#: parameters differ family by family -- :math:`(r, \alpha, s, \beta)` against
#: :math:`(r, \alpha, a, b)` -- and reach the call as ``**model``, so the
#: argument list is genuinely open; what is fixed is the float array out.
_Expression = Callable[..., NDArray[np.float64]]


class Family(Protocol):
    """What :func:`predict` requires of a model family.

    A "family" here is a module -- :mod:`clvtools.pnbd`, :mod:`clvtools.bgnbd`
    or :mod:`clvtools.ggomnbd` -- and this is the part of one that prediction
    touches. Table 4 gives all three a ``PAlive`` and a ``CET``; ``DERT`` is
    Pareto/NBD only and so is not required here, but carried alongside as an
    optional :data:`_Expression` (see :data:`_FAMILIES`).

    The members are read-only properties rather than plain attributes because
    a module supplies them, and a module's attributes cannot be written
    through this interface.
    """

    @property
    def expectation(self) -> _Expression: ...

    @property
    def probability_alive(self) -> _Expression: ...

    @property
    def conditional_expected_transactions(self) -> _Expression: ...


#: Which family's expressions evaluate a given parameter object, and that
#: family's ``DERT`` expression if it has one.
#:
#: Table 4 gives all three families a ``PAlive`` and a ``CET``; only the
#: Pareto/NBD has a closed form for the discounted expected residual
#: transactions, so only it can report ``DERT`` and therefore
#: ``predicted.CLV``. CLVTools omits both columns for the other two, and so
#: does this.
_FAMILIES: dict[type, tuple[Family, _Expression | None]] = {
    PnbdParams: (pnbd, pnbd.discounted_expected_residual_transactions),
    PnbdStaticCovParams: (pnbd, pnbd.discounted_expected_residual_transactions),
    PnbdCorrelatedParams: (pnbd, pnbd.discounted_expected_residual_transactions),
    bgnbd.BgnbdParams: (bgnbd, None),
    bgnbd.BgnbdStaticCovParams: (bgnbd, None),
    ggomnbd.GgomnbdParams: (ggomnbd, None),
    ggomnbd.GgomnbdStaticCovParams: (ggomnbd, None),
}


def _family_of(params) -> tuple[Family, _Expression | None]:
    """The module whose expressions evaluate ``params``, and its DERT, if any."""
    try:
        return _FAMILIES[type(params)]
    except KeyError:
        raise TypeError(
            f"no prediction expressions for {type(params).__name__}"
        ) from None


def _predict_dyncov(
    clv_data, params, spending_params, first, last, length, discount,
) -> pd.DataFrame:
    """S6.4.2's table, which names two columns differently.

    ``DECT`` rather than ``DERT``, and ``predicted.period.CLV`` rather than
    ``predicted.CLV``: with time-varying covariates the discounting runs to the
    end of the covariate series rather than to infinity, so what it produces is
    the value of that period rather than a residual lifetime value.
    """
    from clvtools.data import ClvDataDynCov
    from clvtools.pnbd.dyncov_predict import prediction_table

    if not isinstance(clv_data, ClvDataDynCov):
        raise TypeError(
            "a time-varying covariate model needs covariate data: build the "
            "data with ClvDataDynCov before predicting"
        )
    table = pd.DataFrame(
        {"period.first": first, "period.last": last, "period.length": length},
        index=clv_data.customer_summary().set_index("Id").index,
    )
    if clv_data.has_holdout and last <= clv_data.data_end:
        table = table.join(_actuals(clv_data, first, last, table.index))

    table = table.join(
        prediction_table(clv_data, params, last, length, discount)
    )
    if spending_params is not None:
        spend = clv_data.spending_summary().set_index("Id").loc[table.index]
        mean_spending = expected_mean_spending(
            spend["x"].to_numpy(), spend["Spending"].to_numpy(),
            **spending_params.as_dict(),
        )
        table["predicted.mean.spending"] = mean_spending
        table["predicted.period.spending"] = table["CET"] * mean_spending
        table["predicted.period.CLV"] = table["DECT"] * mean_spending
    return table


def _model_rates(clv_data: ClvData, params) -> dict:
    r"""``(r, alpha, s, beta)`` for the expressions, per customer if needed.

    With covariates each customer has their own :math:`\alpha_i, \beta_i`
    (S3.3), built from the design matrices on the data object. The rest of the
    prediction is then identical, which is the point of the extension.
    """
    if not _has_covariates(params):
        return params.as_dict()
    if not isinstance(clv_data, ClvDataStaticCov):
        raise TypeError(
            "a covariate model needs covariate data: build the data with "
            "ClvDataStaticCov before predicting"
        )
    life = clv_data.design_life(params.names_cov_life)
    trans = clv_data.design_trans(params.names_cov_trans)

    if isinstance(params, PnbdStaticCovParams):
        return {
            "r": params.r,
            "alpha": alpha_i(params.alpha, params.gamma_trans, trans),
            "s": params.s,
            "beta": beta_i(params.beta, params.gamma_life, life),
        }
    if isinstance(params, bgnbd.BgnbdStaticCovParams):
        # The BG/NBD scales both beta parameters by the same factor, with the
        # opposite sign to every other family's rate parameters; see a_i().
        return {
            "r": params.r,
            "alpha": bgnbd.alpha_i(params.alpha, params.gamma_trans, trans),
            "a": bgnbd.a_i(params.a, params.gamma_life, life),
            "b": bgnbd.b_i(params.b, params.gamma_life, life),
        }
    return {
        "r": params.r,
        "alpha": ggomnbd.alpha_i(params.alpha, params.gamma_trans, trans),
        "b": params.b,
        "s": params.s,
        "beta": ggomnbd.beta_i(params.beta, params.gamma_life, life),
    }


def _has_covariates(params) -> bool:
    """Whether the fit carries per-customer rates. Cf. S3.3."""
    return isinstance(
        params,
        (
            PnbdStaticCovParams,
            bgnbd.BgnbdStaticCovParams,
            ggomnbd.GgomnbdStaticCovParams,
        ),
    )


def predict(
    clv_data: ClvData | NewCustomer | NewCustomerSpending,
    params: PnbdParams,
    spending_params: GgParams | None = None,
    prediction_end: float | str | pd.Timestamp | None = None,
    continuous_discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
) -> pd.DataFrame | float:
    r"""The combined prediction table of S6.3. Cf. ``predict()``.

    Parameters
    ----------
    clv_data
        The transaction data the model was fitted on, or -- as S6.3.1's
        ``newdata`` argument allows -- another set of customers to apply the
        fitted parameters to. A :class:`NewCustomer` scenario from S6.3.4 is
        accepted here too, and returns the single number of that section
        rather than a table; see :func:`newcustomer`.
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
    if isinstance(clv_data, (NewCustomer, NewCustomerSpending)):
        return _predict_new_customer(clv_data, params)

    last = _resolve_prediction_end(clv_data, prediction_end)
    if last <= clv_data.estimation_end:
        raise ValueError(
            f"the prediction period ends {last.date()}, on or before the "
            f"estimation period ends {clv_data.estimation_end.date()}"
        )
    first = clv_data.estimation_end + pd.Timedelta(days=1)
    length = clv_data.time.elapsed(clv_data.estimation_end, last)

    if isinstance(params, PnbdDynCovParams):
        return _predict_dyncov(
            clv_data, params, spending_params, first, last, length,
            continuous_discount_factor,
        )

    family, dert = _family_of(params)
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

    table["PAlive"] = family.probability_alive(x, t_x, T, **model)
    table["CET"] = family.conditional_expected_transactions(
        x, t_x, T, length, **model
    )
    if dert is not None:
        table["DERT"] = dert(x, t_x, T, continuous_discount_factor, **model)

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
        # mean spending; CLV is DERT times mean spending, so a family without a
        # DERT has no CLV column either.
        table["predicted.period.spending"] = table["CET"] * mean_spending
        if dert is not None:
            table["predicted.CLV"] = table["DERT"] * mean_spending

    return table
