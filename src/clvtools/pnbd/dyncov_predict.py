r"""S6.4.2 - predicting with time-varying covariates.

S6.4.2: "the time-varying covariates have to be available for the entire
prediction period", because the model integrates over the covariate path a
customer would live through, not just the one already observed.

Three quantities come out of that integral. ``PAlive`` needs nothing new -- it
is a ratio of two pieces the likelihood already computes (see
:func:`~clvtools.pnbd.dyncov.probability_alive`). ``CET`` and ``DECT`` need the
covariate path *ahead* of the estimation period, summarised per period into
four running quantities:

======  ======================================================================
``A``   :math:`\exp(\gamma_{purch}'x)` in this period -- the purchase rate multiplier
``C``   :math:`\exp(\gamma_{attr}'x)` in this period -- the attrition rate multiplier
``B``   the purchase multiplier integrated from the estimation end to here
``D``   the attrition multiplier integrated from the customer's birth to here
======  ======================================================================

Each period contributes one term to a sum, and the sums differ between ``CET``
and ``DECT`` only in that ``DECT`` discounts each period and so carries a
confluent hypergeometric where ``CET`` carries a power.

``DECT`` rather than ``DERT``: with covariates there is no infinite horizon to
discount over, since the covariates are only known as far as they are given.
S6.4.2's table names the column ``DECT`` and its product with spending
``predicted.period.CLV`` for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from clvtools.pnbd.dyncov import log_likelihood_ind, probability_alive
from clvtools.special import kummer_u

__all__ = [
    "CustomerTerms",
    "abcd",
    "conditional_expected_transactions",
    "customer_terms",
    "discounted_expected_transactions",
    "new_customer_expectation",
    "prediction_table",
]


@dataclass(frozen=True)
class CustomerTerms:
    """What the likelihood already knows about each customer.

    ``PAlive``, ``CET`` and ``DECT`` all need the same three per-customer
    quantities out of the log-likelihood -- ``Bksum``, ``DkT`` and the
    probability of being alive -- and computing them means walking every
    customer's covariate path. Ten seconds a pass on the apparel data, so the
    three share one.
    """

    alive: pd.Series
    x: pd.Series
    Bksum: pd.Series
    DkT: pd.Series


def customer_terms(data, params) -> CustomerTerms:
    """Evaluate the likelihood's per-customer terms once. See :class:`CustomerTerms`."""
    walks = data.walks()
    arguments = (
        walks, params.r, params.alpha, params.s, params.beta,
        params.gamma_life, params.gamma_trans,
    )
    table = log_likelihood_ind(*arguments, intermediates=True)
    return CustomerTerms(
        alive=pd.Series(probability_alive(*arguments), index=walks.ids),
        x=pd.Series(walks.x, index=walks.ids),
        Bksum=table["Bksum"],
        DkT=table["DkT"],
    )


def _grid_floor(grid: pd.DatetimeIndex, when: pd.Timestamp) -> pd.Timestamp:
    """The last covariate period start at or before ``when``.

    The covariate dates *are* the period grid -- S6.1 has each one marking the
    start of the period it describes -- so flooring against them needs no
    assumption about which weekday a week begins on.
    """
    earlier = grid[grid <= when]
    if not len(earlier):
        raise ValueError(f"no covariate period covers {when.date()}")
    return earlier[-1]


def _require_coverage(grid: pd.DatetimeIndex, horizon: pd.Timestamp, time) -> None:
    """S6.4.2: the covariates must reach the end of the prediction window.

    The last covariate date describes the period that *starts* there, so the
    series covers one period past its final row. A horizon beyond that has no
    covariate values to integrate, and silently stopping the walk short would
    return a confident answer to a different question.
    """
    if horizon > time.add(grid[-1], 1):
        raise ValueError(
            f"the covariate series ends {grid[-1].date()} and does not reach "
            f"the prediction horizon {horizon.date()}: extend it (S6.4.2)"
        )


def _alive_covariates(data, params, upper: pd.Timestamp) -> dict[str, pd.DataFrame]:
    r""":math:`\exp(\gamma'x)` per customer per period, while the customer exists.

    A customer contributes nothing before their first transaction -- they had
    not been acquired -- so periods before the one they came alive in are
    dropped, exactly as ``pnbd_dyncov_alivecovariates`` drops them.
    """
    date_column = data.name_date_cov
    cbs = data.customer_summary().set_index("Id")
    grid = pd.DatetimeIndex(sorted(data.data_cov_life[date_column].unique()))
    born = cbs["date_first_transaction"].map(lambda d: _grid_floor(grid, d))
    first_period = _grid_floor(grid, data.estimation_start)

    out = {}
    for kind, raw, names, gamma in (
        ("life", data.data_cov_life, params.names_cov_life, params.gamma_life),
        ("trans", data.data_cov_trans, params.names_cov_trans, params.gamma_trans),
    ):
        frame = raw[
            (raw[date_column] >= first_period) & (raw[date_column] <= upper)
        ].copy()
        frame["exp_gX"] = np.exp(
            frame[list(names)].to_numpy(dtype=float) @ np.asarray(gamma, dtype=float)
        )
        frame = frame[frame[date_column] >= frame["Id"].map(born)]
        out[kind] = frame.sort_values(["Id", date_column], kind="stable")[
            ["Id", date_column, "exp_gX"]
        ].reset_index(drop=True)
    return out


def abcd(data, params, prediction_end: pd.Timestamp) -> pd.DataFrame:
    r"""The per-period :math:`A_i, B_i, C_i, D_i` of the prediction window.

    One row per customer per period from the period the estimation ends in
    through the one ``prediction_end`` falls in. ``Bbar_i`` and ``Dbar_i`` are
    the integrated multipliers, offset so that the sums in
    :func:`conditional_expected_transactions` and
    :func:`discounted_expected_transactions` can be written per period.

    ``d1`` is the fraction of a period left after the estimation period ends:
    the first prediction period is generally entered part-way through.
    """
    date_column = data.name_date_cov
    grid = pd.DatetimeIndex(sorted(data.data_cov_life[date_column].unique()))
    _require_coverage(grid, prediction_end, data.time)
    upper = _grid_floor(grid, prediction_end)
    covariates = _alive_covariates(data, params, upper)

    window_start = _grid_floor(grid, data.estimation_end)
    later = grid[grid > window_start]
    if not len(later):
        raise ValueError(
            "the covariate series does not reach past the estimation period: "
            "extend it to the prediction horizon (S6.4.2)"
        )
    d1 = data.time.elapsed(data.estimation_end, later[0])

    walks = data.walks()
    cbs = pd.DataFrame(
        {"T_cal": walks.T_cal, "d_omega": walks.d_omega}, index=walks.ids
    )

    life = covariates["life"]
    life["num_alive"] = life.groupby("Id").cumcount() + 1
    life["d_omega"] = life["Id"].map(cbs["d_omega"])

    # D integrates the attrition multiplier from the customer's birth. The
    # first period they are alive for is only partly theirs -- d_omega of it.
    first = life["num_alive"] == 1
    contribution = np.where(
        first, life["exp_gX"] * life["d_omega"], life["exp_gX"]
    )
    life["Dbar"] = pd.Series(contribution, index=life.index).groupby(
        life["Id"]
    ).cumsum()

    in_window = life[date_column] >= window_start
    life["i"] = (
        life[in_window].groupby("Id").cumcount() + 1
    ).reindex(life.index)

    table = life[in_window].copy()
    # Undo this period's own contribution and re-add it as a distance, which is
    # what the per-period terms below are written in terms of.
    offset = np.where(
        table["num_alive"] <= 1,
        -table["d_omega"],
        -table["d_omega"] - (table["num_alive"] - 2),
    )
    table["Dbar_i"] = (table["Dbar"] - table["exp_gX"]) + table["exp_gX"] * offset
    table = table.rename(columns={"exp_gX": "Ci"})

    trans = covariates["trans"].rename(columns={"exp_gX": "Ai"})
    table = table.merge(trans, on=["Id", date_column], how="left")
    if table["Ai"].isna().any():
        raise ValueError(
            "the transaction covariates do not cover every period the "
            "attrition covariates do"
        )

    table["T_cal"] = table["Id"].map(cbs["T_cal"]).to_numpy()
    table["d1"] = d1

    # B integrates the purchase multiplier from the estimation end onwards, so
    # its first period counts only the d1 of it that is left.
    contribution = np.where(table["i"] == 1, table["Ai"] * d1, table["Ai"])
    running = pd.Series(contribution, index=table.index).groupby(
        table["Id"]
    ).cumsum()
    table["Bbar_i"] = np.where(
        table["i"] > 1,
        (running - table["Ai"])
        + table["Ai"] * (-table["T_cal"] - d1 - (table["i"] - 2)),
        table["Ai"] * -table["T_cal"],
    )
    return table[
        ["Id", date_column, "i", "Ai", "Ci", "d1", "Bbar_i", "T_cal", "Dbar_i"]
    ].reset_index(drop=True)


def _reject_unit_s(s: float) -> None:
    """Refuse :math:`s = 1`, where every expression here divides by ``s - 1``.

    The same guard, and the same message, as
    :func:`clvtools.pnbd.aggregate.conditional_expected_transactions`. Finding
    10 of ``docs/review-2026-09-02.md`` noticed that only one of the two had it.
    """
    if np.isclose(s, 1.0):
        raise ValueError(
            "CET is undefined at s = 1: the expression divides by (s - 1)"
        )


def _last_period(table: pd.DataFrame) -> NDArray[np.bool_]:
    """Whether each row is its customer's final prediction period."""
    return (table["i"] == table.groupby("Id")["i"].transform("max")).to_numpy()


def conditional_expected_transactions(
    data, params, prediction_end: pd.Timestamp, periods: float,
    terms: CustomerTerms | None = None,
) -> pd.Series:
    r"""``CET`` with time-varying covariates. S6.4.2.

    The expected number of transactions in the next ``periods``, given the
    customer's history and the covariate path they will live through.

    .. math::
        \mathrm{CET} = \mathrm{PAlive} \cdot
            \frac{(r{+}x)(\beta_0{+}D_{k_T})^s}{(\mathrm{Bksum}{+}\alpha_0)(s{-}1)}
            \cdot \Big( F_2 + \sum_i S_i \Big)

    with one :math:`S_i` per period of the horizon. Each is a difference of the
    same expression at the two ends of that period, so the sum telescopes down
    the covariate path.

    Undefined at :math:`s = 1`, where the leading factor divides by
    :math:`s - 1`. :func:`clvtools.pnbd.aggregate.conditional_expected_transactions`
    has raised there since it was written; this divided anyway and returned an
    ``inf`` or a very large finite number depending on which side of 1 the
    optimiser had stopped, so the same model at the same parameters answered
    differently depending on which entry point was asked.
    """
    r, alpha, s, beta = params.r, params.alpha, params.s, params.beta
    _reject_unit_s(s)
    table = abcd(data, params, prediction_end)
    t = float(periods)

    Ai, Ci = table["Ai"].to_numpy(), table["Ci"].to_numpy()
    Bbar, Dbar = table["Bbar_i"].to_numpy(), table["Dbar_i"].to_numpy()
    T_cal, d1, i = (
        table["T_cal"].to_numpy(), table["d1"].to_numpy(), table["i"].to_numpy()
    )

    def term(upper: NDArray[np.float64]) -> NDArray[np.float64]:
        return (Ai * (upper * s + (Dbar + beta) / Ci) + Bbar * (s - 1)) / (
            Dbar + beta + Ci * upper
        ) ** s

    last = _last_period(table)
    single = table.groupby("Id")["i"].transform("max").to_numpy() < 2

    bT = T_cal + d1 + (i - 2)
    lower = np.where(i == 1, T_cal, bT)
    upper = np.where(i == 1, T_cal + d1, bT + 1.0)
    # The final period runs to the horizon, not to the next period boundary,
    # and a horizon inside the first period runs there directly.
    upper = np.where(last | single, T_cal + t, upper)
    lower = np.where(single, T_cal, lower)

    table = table.assign(S=term(lower) - term(upper))
    totals = table.groupby("Id")["S"].sum()

    terms = terms if terms is not None else customer_terms(data, params)
    F1 = ((r + terms.x) * (beta + terms.DkT) ** s) / (
        (terms.Bksum + alpha) * (s - 1)
    )

    tail = table.loc[_last_period(table)].set_index("Id")
    F2_no_sum = (
        (tail["Bbar_i"] + tail["Ai"] * (tail["T_cal"] + t)) * (s - 1)
    ) / (tail["Dbar_i"] + tail["Ci"] * (tail["T_cal"] + t) + beta) ** s

    return (terms.alive * F1 * (F2_no_sum + totals)).rename("CET")


def discounted_expected_transactions(
    data, params, prediction_end: pd.Timestamp, periods: float,
    continuous_discount_factor: float, terms: CustomerTerms | None = None,
) -> pd.Series:
    r"""``DECT`` with time-varying covariates. S6.4.2.

    ``CET`` with each period discounted back to the estimation end. The power
    of the ``CET`` sum becomes Tricomi's :math:`U(s, s, \cdot)`, which is the
    same function the standard model's ``DERT`` uses -- there over an infinite
    horizon, here period by period over a finite one.
    """
    r, alpha, s, beta = params.r, params.alpha, params.s, params.beta
    delta = float(continuous_discount_factor)
    t = float(periods)
    table = abcd(data, params, prediction_end)

    Ai, Ci = table["Ai"].to_numpy(), table["Ci"].to_numpy()
    Dbar = table["Dbar_i"].to_numpy()
    T_cal, d1, i = (
        table["T_cal"].to_numpy(), table["d1"].to_numpy(), table["i"].to_numpy()
    )
    bT = T_cal + d1 + (i - 2)

    def u_at(upper: NDArray[np.float64]) -> NDArray[np.float64]:
        return kummer_u(s, s, delta * (Ci * upper + Dbar + beta) / Ci)

    # U is the expensive part of this function -- one evaluation per customer
    # per period -- so the lower end that the middle and final periods share is
    # evaluated once.
    u_lower = u_at(bT)
    first_term = u_at(T_cal) - np.exp(-delta * d1) * u_at(T_cal + d1)
    middle_term = (
        np.exp(-delta * (d1 + i - 2)) * u_lower
        - np.exp(-delta * (d1 + i - 1)) * u_at(bT + 1.0)
    )
    last_term = (
        np.exp(-delta * (d1 + i - 2)) * u_lower
        - np.exp(-delta * t) * u_at(T_cal + t)
    )

    # A window one period long is both the first period and the last; the last
    # form wins, as it does in CLVTools, because it is the one that runs to the
    # horizon rather than to a period boundary.
    last = _last_period(table)
    S = np.where(i == 1, first_term, middle_term)
    S = np.where(last, last_term, S)
    table = table.assign(S=S * Ai / Ci**s)
    totals = table.groupby("Id")["S"].sum()

    terms = terms if terms is not None else customer_terms(data, params)
    F1 = (
        delta ** (s - 1)
        * ((r + terms.x) * (beta + terms.DkT) ** s)
        / (terms.Bksum + alpha)
    )
    return (terms.alive * F1 * totals).rename("DECT")


def prediction_table(
    data, params, prediction_end: pd.Timestamp, periods: float,
    continuous_discount_factor: float,
) -> pd.DataFrame:
    """``PAlive``, ``CET`` and ``DECT`` per customer, in CLVTools' order."""
    terms = customer_terms(data, params)
    cet = conditional_expected_transactions(
        data, params, prediction_end, periods, terms=terms
    )
    dect = discounted_expected_transactions(
        data, params, prediction_end, periods, continuous_discount_factor,
        terms=terms,
    )
    return pd.concat([terms.alive.rename("PAlive"), cet, dect], axis=1)


def new_customer_expectation(
    params,
    num_periods: float,
    first_transaction: pd.Timestamp,
    cov_life: pd.DataFrame,
    cov_trans: pd.DataFrame,
    time,
    name_date_cov: str = "Cov.Date",
) -> float:
    r"""``E[X(t)]`` for a prospective customer on a given covariate path.

    S6.3.4 points at ``newcustomer.dynamic()`` for scenario work with
    time-varying covariates: the customer has no history at all, so the whole
    expectation comes from the covariate path they are assumed to live through
    from their first transaction onwards.

    The same per-period sum as :func:`conditional_expected_transactions`, with
    the customer's own history removed: the walk starts at the first
    transaction rather than at the estimation end, and ``d_omega`` -- the part
    of the first period that falls after that transaction -- takes the place of
    ``d1``.
    """
    r, alpha, s, beta = params.r, params.alpha, params.s, params.beta
    first_transaction = pd.Timestamp(first_transaction)

    grid = pd.DatetimeIndex(sorted(cov_life[name_date_cov].unique()))
    start = _grid_floor(grid, first_transaction)
    horizon = time.add(first_transaction, float(num_periods))
    _require_coverage(grid, horizon, time)
    later = grid[grid > start]
    if not len(later):
        raise ValueError("the covariate series ends in the first period")
    d_omega = time.elapsed(first_transaction, later[0])

    def multipliers(frame: pd.DataFrame, names, gamma) -> pd.DataFrame:
        frame = frame[frame[name_date_cov] >= start].sort_values(name_date_cov)
        values = np.exp(
            frame[list(names)].to_numpy(dtype=float) @ np.asarray(gamma, float)
        )
        return pd.DataFrame(
            {name_date_cov: frame[name_date_cov].to_numpy(), "exp_gX": values}
        )

    life = multipliers(cov_life, params.names_cov_life, params.gamma_life)
    trans = multipliers(cov_trans, params.names_cov_trans, params.gamma_trans)
    table = life.rename(columns={"exp_gX": "Ci"}).merge(
        trans.rename(columns={"exp_gX": "Ai"}), on=name_date_cov, how="inner"
    )
    table["i"] = np.arange(1, len(table) + 1)

    def integrated(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative multiplier, re-expressed as a distance from the start."""
        contribution = values.copy()
        contribution[0] = values[0] * d_omega
        running = np.cumsum(contribution)
        out = (running - values) + values * (-d_omega - (table["i"] - 2))
        out.iloc[0] = 0.0
        return out.to_numpy()

    table["Bbar_i"] = integrated(table["Ai"].to_numpy())
    table["Dbar_i"] = integrated(table["Ci"].to_numpy())

    table = table[table[name_date_cov] <= horizon].reset_index(drop=True)

    A, B = table["Ai"].to_numpy(), table["Bbar_i"].to_numpy()
    C, D = table["Ci"].to_numpy(), table["Dbar_i"].to_numpy()
    i = table["i"].to_numpy()

    def g(term):
        return (A * (term * s + (beta + D) / C) + B * (s - 1)) / (
            beta + D + C * term
        ) ** s

    S = np.where(i == 1, g(0.0) - g(d_omega), g(d_omega + i - 2) - g(d_omega + i - 1))
    S[-1] = g(d_omega + i - 2)[-1] - g(float(num_periods))[-1]
    total = float(np.sum(S))

    t = float(num_periods)
    _reject_unit_s(s)
    scale = (beta**s * r) / ((s - 1) * alpha)
    a, b, c, d = A[-1], B[-1], C[-1], D[-1]
    if len(table) == 1:
        # Alive for one covariate period only: there is no earlier period for
        # the sum to telescope through, so the two ends are written out.
        inner = (
            (a * t * (s - 1)) / (beta + c * t) ** s
            + (a / c) / beta ** (s - 1)
            - (a * (t * s + beta / c)) / (beta + c * t) ** s
        )
    else:
        inner = (
            ((a * t + b) * (s - 1)) / (beta + (c * t + d)) ** s
        ) + total
    return float(scale * inner)
