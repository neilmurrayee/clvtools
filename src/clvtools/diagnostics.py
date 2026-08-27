r"""S6.2.2 and S6.2.4 - the diagnostic plots, as data.

S6.2.2: "The key diagnostics for a latent attrition model are two plots: (1) the
tracking plot and (2) the probability mass function (PMF) plot."

Each function here returns the *data* a plot would be drawn from, in the long
form CLVTools' own ``plot(..., plot = FALSE)`` produces: one row per period (or
bin) per series, with an ``Actual`` series and a model series. That is the form
the paper itself works with -- S6.3.3's bootstrap example calls
``plot(..., plot = FALSE)`` precisely to get the numbers and build a ribbon from
them.

Rendering is a separate step. :func:`render` will draw any of these frames with
matplotlib if it is installed; it is not a dependency of the package, since
nothing in the models needs it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clvtools.data import ClvData

__all__ = [
    "pmf_data",
    "render",
    "spending_density_data",
    "tracking_data",
]

#: Column name for the observed series, matching CLVTools.
ACTUAL = "Actual"


def _period_grid(data: ClvData, end: pd.Timestamp) -> pd.DatetimeIndex:
    r"""Period ends from the start of the data through ``end``.

    S6.2.2: "In line with previous literature, the first predicted date is the
    start of the data. The expected number of repeat transactions on this date
    by definition is zero and this fact gives the plot its characteristic
    shape."

    The grid runs one period *past* ``end`` when ``end`` falls mid-period, so
    the final period is shown whole rather than truncated. On the apparel data
    that carries it to 2010-12-26 against a last transaction of 2010-12-20.
    """
    points, when = [], data.estimation_start
    while when <= end:
        points.append(when)
        when = data.time.add(when, 1)
    if points and points[-1] < end:
        points.append(when)
    return pd.DatetimeIndex(points)


def tracking_data(
    data: ClvData,
    expectation,
    prediction_end: int | float | str | pd.Timestamp | None = None,
    cumulative: bool = False,
    model_name: str = "Model",
) -> pd.DataFrame:
    r"""Actual against expected repeat transactions per period.

    S6.2.2: "The plot shows the actual repeat transactions and adds an overlay
    with the repeat transaction expected by the fitted model. This is based on
    the model's unconditional expectation, i.e., the sum of all transactions
    expected by all customers in each period."

    Parameters
    ----------
    expectation
        ``t -> E[X(t)]`` for a single customer with no history, as each family's
        ``expectation`` provides. It is summed over customers by multiplying by
        the number of them, since the unconditional expectation does not depend
        on any customer's data.
    cumulative
        S6.2.2 contrasts the two: the incremental plot "gives an indication of
        how well the model captures dynamic effects such as seasonality", while
        in the cumulative one "it is easier to see how well the model tracks the
        number of repeat purchases overall".

    Returns
    -------
    A long frame of ``period.until``, ``variable``, ``value`` -- two rows per
    period, one ``Actual`` and one for the model.

    Examples
    --------
    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.pnbd import expectation
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> curve = lambda t: expectation(t, 1.4490, 48.6361, 0.5613, 46.8844)
    >>> frame = tracking_data(data, curve, model_name="Pareto/NBD")
    >>> list(frame.columns)
    ['period.until', 'variable', 'value']
    >>> sorted(frame["variable"].unique())
    ['Actual', 'Pareto/NBD']

    The model series opens at zero, which is what gives the plot its shape:

    >>> bool(frame.loc[frame["variable"] == "Pareto/NBD", "value"].iloc[0] == 0.0)
    True
    """
    end = data.data_end if prediction_end is None else (
        data.time.add(data.estimation_end, float(prediction_end))
        if isinstance(prediction_end, (int, float, np.integer, np.floating))
        else pd.Timestamp(prediction_end)
    )
    grid = _period_grid(data, end)

    # Repeat transactions only: a customer's first purchase is not one.
    transactions = data.transactions
    first = transactions.groupby("Id")["Date"].transform("min")
    repeats = transactions.loc[transactions["Date"] > first, "Date"]

    counted = (
        pd.Series(np.searchsorted(grid, repeats.to_numpy(), side="left"))
        .value_counts()
        .reindex(range(len(grid)), fill_value=0)
        .sort_index()
        .to_numpy(dtype=float)
    )

    n_customers = int(transactions["Id"].nunique())
    elapsed = np.array(
        [data.time.elapsed(data.estimation_start, when) for when in grid]
    )
    cumulative_expected = n_customers * np.asarray(
        [float(expectation(t)) for t in elapsed], dtype=float
    )

    # A period the data does not fully cover gets no observed count. Reporting
    # the transactions that happen to fall in it would understate it, since the
    # rest of the period simply has not happened yet -- so it is left missing,
    # as CLVTools leaves it.
    if len(grid) and grid[-1] > data.data_end:
        counted[-1] = np.nan

    if cumulative:
        actual_series = np.cumsum(counted)
        model_series = cumulative_expected
    else:
        actual_series = counted
        model_series = np.diff(cumulative_expected, prepend=0.0)

    return pd.concat([
        pd.DataFrame({
            "period.until": grid, "variable": model_name, "value": model_series,
        }),
        pd.DataFrame({
            "period.until": grid, "variable": ACTUAL, "value": actual_series,
        }),
    ], ignore_index=True)


def pmf_data(
    data: ClvData,
    pmf,
    max_transactions: int = 10,
    model_name: str = "Model",
) -> pd.DataFrame:
    r"""Observed against expected customer counts by repeat-transaction count.

    S6.2.2: "It shows the actual and expected number of customers who make a
    given number of repeat transactions during the estimation period. […] For
    each bin, the expected number of customers is the sum of all customers'
    individual PMF values for this number of purchases."

    The final bin is a tail: everything at or above ``max_transactions``, so the
    two series sum to the customer count.

    Parameters
    ----------
    pmf
        ``(k, T) -> P(X(T) = k)`` per customer, as each family's ``pmf``
        provides.

    Examples
    --------
    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.pnbd import pmf
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> frame = pmf_data(
    ...     data, lambda k, T: pmf(k, T, 1.4490, 48.6361, 0.5613, 46.8844),
    ...     model_name="Pareto/NBD")
    >>> list(frame.columns)
    ['num.transactions', 'variable', 'value']

    213 of the 600 customers made no repeat purchase:

    >>> observed = frame[frame["variable"] == "Actual"].set_index("num.transactions")
    >>> int(observed.loc["0", "value"])
    213

    Both series account for every customer:

    >>> [int(round(v)) for v in frame.groupby("variable")["value"].sum()]
    [600, 600]
    """
    if max_transactions < 1:
        raise ValueError("max_transactions must be at least 1")

    summary = data.customer_summary()
    T = summary["T"].to_numpy(dtype=float)
    counts = summary["x"].to_numpy(dtype=float)

    labels = [str(k) for k in range(max_transactions)] + [f"{max_transactions}+"]

    observed = [float(np.sum(counts == k)) for k in range(max_transactions)]
    observed.append(float(np.sum(counts >= max_transactions)))

    expected = [float(np.sum(pmf(k, T))) for k in range(max_transactions)]
    # The tail is what the bins below it leave over, so the series totals match.
    expected.append(float(len(T) - sum(expected)))

    return pd.concat([
        pd.DataFrame({
            "num.transactions": labels, "variable": ACTUAL, "value": observed,
        }),
        pd.DataFrame({
            "num.transactions": labels, "variable": model_name, "value": expected,
        }),
    ], ignore_index=True)


def spending_density_data(
    data: ClvData,
    params,
    grid: np.ndarray | None = None,
    model_name: str = "Gamma-Gamma",
) -> pd.DataFrame:
    r"""The spending model's density against the observed one. S6.2.4.

    S6.2.4: the plot "compares the density of each customer's observed average
    order value (i.e., the empirical distribution) to the model's distribution
    of mean transaction spending across customers."

    The model curve is eq. (17) evaluated for every customer's own transaction
    count and averaged, since each customer's density depends on how many
    purchases they made. Customers with no spending are excluded, as they are
    from estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.gg import fit_gg
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> spend = data.spending_summary()
    >>> fitted = fit_gg(spend["x"], spend["Spending"])
    >>> frame = spending_density_data(data, fitted, grid=np.linspace(1, 400, 256))
    >>> list(frame.columns)
    ['spending', 'variable', 'value']

    It is a density, so it integrates to about one over the grid:

    >>> model = frame[frame["variable"] == "Gamma-Gamma"]
    >>> mass = np.trapezoid(model["value"], model["spending"])
    >>> bool(0.9 < mass < 1.01)
    True
    """
    from clvtools.gg import mean_spending_pdf

    spending = data.spending_summary()
    active = spending[spending["x"] > 0]
    if active.empty:
        raise ValueError("no customer has both a transaction and a spend")

    if grid is None:
        grid = np.linspace(
            float(active["Spending"].min()), float(active["Spending"].max()), 256
        )
    grid = np.asarray(grid, dtype=float)

    x = active["x"].to_numpy(dtype=float)
    density = np.array([
        float(np.mean(mean_spending_pdf(value, x, **params.as_dict())))
        for value in grid
    ])

    observed = _kernel_density(active["Spending"].to_numpy(dtype=float), grid)

    return pd.concat([
        pd.DataFrame({"spending": grid, "variable": ACTUAL, "value": observed}),
        pd.DataFrame({"spending": grid, "variable": model_name, "value": density}),
    ], ignore_index=True)


def _kernel_density(sample: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A Gaussian kernel density with Silverman's bandwidth.

    The same default R's ``density()`` uses, which is what CLVTools draws the
    ``Actual`` curve with.
    """
    n = sample.size
    spread = min(
        sample.std(ddof=1),
        (np.percentile(sample, 75) - np.percentile(sample, 25)) / 1.349,
    )
    bandwidth = 0.9 * spread * n ** (-0.2)
    z = (grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))


def render(frame: pd.DataFrame, title: str | None = None, ax=None):
    """Draw one of these frames with matplotlib.

    Optional: matplotlib is not a dependency, because nothing in the models
    needs it and the frames above are useful without it.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "render() needs matplotlib, which is an optional extra: "
            "install it with `uv add --optional plot matplotlib`, or use the "
            "returned frame with any other plotting library."
        ) from error

    x_column = next(
        c for c in ("period.until", "num.transactions", "spending")
        if c in frame.columns
    )
    if ax is None:
        _, ax = plt.subplots()
    for name, group in frame.groupby("variable", sort=False):
        ax.plot(group[x_column], group["value"], label=str(name))
    ax.set_xlabel(x_column)
    ax.set_ylabel("value")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax
