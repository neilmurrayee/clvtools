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

from collections.abc import Sequence

import numpy as np
import pandas as pd

from clvtools.data import ClvData

__all__ = [
    "fitted_data",
    "frequency_data",
    "interpurchase_time_data",
    "pmf_data",
    "pmf_table",
    "render",
    "spending_data",
    "spending_density_data",
    "timings_data",
    "tracking_data",
]

#: Column name for the observed series, matching CLVTools.
ACTUAL = "Actual"

#: What CLVTools labels the observed series when no model is overlaid.
REPEAT_TRANSACTIONS = "Number of Repeat Transactions"


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
    expectation=None,
    prediction_end: float | str | pd.Timestamp | None = None,
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
        on any customer's data. ``None`` gives the descriptive tracking plot of
        S6.1.2 instead -- the observed series alone, with no model overlaid,
        which is what ``plot(clv.data, which = "tracking")`` draws.
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

    Without a model it is the descriptive plot of S6.1.2, "the total number of
    repeat transactions per period":

    >>> observed = tracking_data(data)
    >>> sorted(observed["variable"].unique())
    ['Number of Repeat Transactions']
    >>> float(observed["value"].iloc[1])
    10.0
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

    if expectation is not None:
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

    actual_series = np.cumsum(counted) if cumulative else counted

    if expectation is None:
        return pd.DataFrame({
            "period.until": grid,
            "variable": REPEAT_TRANSACTIONS,
            "value": actual_series,
        })

    model_series = (
        cumulative_expected if cumulative
        else np.diff(cumulative_expected, prepend=0.0)
    )
    # The opening value is zero "by definition"; differencing can hand back a
    # negative zero for it, which prints as -0.0 and reads as a defect.
    model_series = np.where(model_series == 0.0, 0.0, model_series)
    return pd.concat([
        pd.DataFrame({
            "period.until": grid, "variable": model_name, "value": model_series,
        }),
        pd.DataFrame({
            "period.until": grid, "variable": ACTUAL, "value": actual_series,
        }),
    ], ignore_index=True)


def pmf_table(data: ClvData, pmf, x=range(6)) -> pd.DataFrame:
    r"""Each customer's PMF at each of several counts. Cf. ``pmf()``.

    Spec `PMF-05`, `absent`: CLVTools' ``pmf()`` generic on a fitted object
    returns one row per customer and one ``pmf.x.<k>`` column per requested
    count, defaulting to ``x = 0:5``. This package had
    :func:`pmf_data`, which is a different thing -- it aggregates customers
    into bins for S6.2.2's plot, and cannot answer "what is *this* customer's
    probability of buying twice".

    ``x`` may be an integer, a float that is a whole number, or any iterable of
    them, as R's does; the column name uses the integer, so ``2.0`` and ``2``
    give the same column and asking for both is an error rather than two
    columns that silently collide.

    Examples
    --------
    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.pnbd import pmf
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> frame = pmf_table(
    ...     data, lambda k, T: pmf(k, T, 1.4490, 48.6361, 0.5613, 46.8844))
    >>> list(frame.columns)
    ['Id', 'pmf.x.0', 'pmf.x.1', 'pmf.x.2', 'pmf.x.3', 'pmf.x.4', 'pmf.x.5']
    >>> len(frame)
    600

    A single count is a single column, and R accepts a bare number for it:

    >>> list(pmf_table(data, lambda k, T: pmf(k, T, 1.449, 48.64, 0.561, 46.88),
    ...                x=0).columns)
    ['Id', 'pmf.x.0']

    Each row is a probability distribution's first few terms, so it sums to at
    most one:

    >>> bool(frame.set_index("Id").sum(axis=1).le(1.0).all())
    True
    """
    counts = [x] if isinstance(x, (int, float, np.integer, np.floating)) else list(x)
    wanted: list[int] = []
    for value in counts:
        whole = float(value)
        if whole != int(whole) or whole < 0:
            raise ValueError(
                f"pmf counts must be whole and non-negative, got {value!r}"
            )
        if int(whole) in wanted:
            raise ValueError(
                f"pmf count {int(whole)} was asked for twice, which would give "
                "one column two definitions"
            )
        wanted.append(int(whole))
    if not wanted:
        raise ValueError("pmf needs at least one count")

    summary = data.customer_summary()
    T = summary["T"].to_numpy(dtype=float)
    frame = pd.DataFrame({"Id": summary["Id"].to_numpy()})
    for k in wanted:
        frame[f"pmf.x.{k}"] = np.asarray(pmf(k, T), dtype=float)
    return frame


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


def fitted_data(data: ClvData, expectation) -> pd.DataFrame:
    """The model's expected repeat transactions per period. Cf. ``fitted()``.

    Table 2 lists ``fitted()`` among the generics every fitted model offers. It
    is the model half of the tracking plot on its own: one row per period, with
    the period's number and the expected number of repeat transactions in it.

    The doctest below prints what this implementation returns, so it pins the
    output against drift rather than against R. All 313 periods are compared
    with R's own ``fitted()`` in ``tests/test_diagnostics.py``.

    Examples
    --------
    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.pnbd import expectation
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> curve = lambda t: expectation(t, 1.4490, 48.6361, 0.5613, 46.8844)
    >>> frame = fitted_data(data, curve)
    >>> print(frame.head(3).to_string(index=False))
    period.until  period.num  expectation
      2005-01-02           1     0.000000
      2005-01-09           2    17.769779
      2005-01-16           3    17.562679
    """
    frame = tracking_data(data, expectation)
    model = frame.loc[frame["variable"] != ACTUAL]
    return pd.DataFrame({
        "period.until": model["period.until"].to_numpy(),
        "period.num": np.arange(1, len(model) + 1),
        "expectation": model["value"].to_numpy(),
    })


def frequency_data(
    data: ClvData,
    bins: range | Sequence[int] = range(10),
    count_repeat_transactions: bool = True,
    count_remaining: bool = True,
    label_remaining: str = "10+",
    sample: str = "estimation",
) -> pd.DataFrame:
    """How many customers made how many transactions. S6.1.2, Table 3.

    "The distribution of the number of transactions per customer." One row per
    bin, with the customers at or above the last bin collected into a final
    ``label_remaining`` row so the counts still sum to every customer.

    Counting *repeat* transactions is the default, matching the models: a
    customer seen once has made none. Counting every transaction instead
    (``count_repeat_transactions=False``) leaves no customer at zero, so the
    bins must then start at one.

    Examples
    --------
    S6.1.2 reports 35.5% zero repeaters, which is this frame's first bin:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> frame = frequency_data(data)
    >>> print(frame.head(3).to_string(index=False))
    num.transactions  num.customers
                   0            213
                   1            116
                   2             82
    >>> int(frame["num.customers"].sum())
    600
    """
    bins = list(bins)
    if not count_repeat_transactions and min(bins) < 1:
        raise ValueError(
            "counting all transactions leaves no customer below one: "
            "bins must be strictly positive"
        )
    frame = data._sample(sample)
    counts = frame.groupby("Id", sort=True)["Date"].size()
    if count_repeat_transactions:
        counts = counts - 1

    rows = [(str(b), int((counts == b).sum())) for b in bins]
    if count_remaining:
        rows.append((label_remaining, int((counts > max(bins)).sum())))
    return pd.DataFrame(rows, columns=["num.transactions", "num.customers"])


def interpurchase_time_data(
    data: ClvData, sample: str = "estimation"
) -> pd.DataFrame:
    """Each customer's mean time between transactions. S6.1.2, Table 3.

    "The empirical density of customer's mean time between transactions [...]
    Only data from customers with repeat transactions are shown in this graph",
    so single-transaction customers are dropped rather than counted as zero.

    Examples
    --------
    387 of the 600 customers made a repeat purchase in the estimation period:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> frame = interpurchase_time_data(data)
    >>> len(frame), round(float(frame["mean.interpurchase.time"].mean()), 3)
    (387, 24.823)
    """
    frame = data.mean_interpurchase_times(sample)
    return frame.dropna(subset=["mean.interpurchase.time"]).reset_index(drop=True)


def spending_data(
    data: ClvData, sample: str = "estimation", mean_spending: bool = True
) -> pd.DataFrame:
    """Observed spending, per customer or per transaction. S6.1.2, Table 3.

    "It either shows the empirical density of customers' average order values
    (``mean.spending = TRUE``) or the value of every transaction in the data."
    S6.1.2 draws it for both samples to check "whether the distribution remains
    stable in the estimation and the holdout period, a key assumption of the
    spending models".

    Examples
    --------
    >>> from clvtools import ClvData, load_apparel_trans
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> per_customer = spending_data(data)
    >>> per_transaction = spending_data(data, mean_spending=False)
    >>> len(per_customer), len(per_transaction)
    (600, 1866)
    >>> round(float(per_transaction["Spending"].mean()), 3)
    40.545
    """
    if not data.has_spending:
        raise ValueError("no Price column: there is no spending to plot")
    frame = data._sample(sample)
    if not mean_spending:
        return frame[["Id", "Price"]].rename(
            columns={"Price": "Spending"}
        ).reset_index(drop=True)
    mean = frame.groupby("Id", sort=True)["Price"].mean()
    return pd.DataFrame({"Id": mean.index, "Spending": mean.to_numpy()})


def timings_data(
    data: ClvData,
    ids: Sequence[str] | None = None,
    n: int = 50,
    seed: int | None = None,
) -> pd.DataFrame:
    """When each of a subset of customers transacted. S6.1.2, Table 3.

    "Each dot in a row illustrates when a transaction for a particular customer
    was recorded." The frame is the drawing itself, long: one horizontal
    ``segment`` per customer spanning their first transaction to the end of the
    data, and one ``point`` per transaction, labelled by which period it falls
    in. ``x`` values are dates and ``y`` values are the row a customer sits on,
    both as strings, exactly as CLVTools emits them.

    Customers are laid out by first transaction, ties broken by descending
    ``Id``, ten units apart -- CLVTools' own order, kept so the frames can be
    compared row for row.

    Parameters
    ----------
    ids
        Which customers to draw. Defaults to ``n`` of them at random, as
        CLVTools defaults to 50.

    Examples
    --------
    >>> from clvtools import ClvData, load_apparel_trans
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> frame = timings_data(data, ids=["1", "2", "3"])
    >>> sorted(frame["type"].unique())
    ['point_calibration', 'point_holdout', 'segment_end', 'segment_start']
    >>> print(frame.loc[frame["type"] == "segment_start"].to_string(index=False))
    Id          type variable      value
     3 segment_start        x 2005-01-02
     2 segment_start        x 2005-01-02
     1 segment_start        x 2005-01-02
     3 segment_start        y         10
     2 segment_start        y         20
     1 segment_start        y         30
    """
    everyone = data.transactions
    if ids is None:
        pool = pd.Index(sorted(everyone["Id"].unique()))
        n = min(n, len(pool))
        ids = list(
            pd.Series(pool).sample(n=n, random_state=seed, replace=False)
        )
    ids = list(dict.fromkeys(str(i) for i in ids))
    unknown = [i for i in ids if i not in set(everyone["Id"])]
    if unknown:
        raise ValueError(f"no such customers: {unknown[:3]}")

    chosen = everyone[everyone["Id"].isin(ids)]
    first = chosen.groupby("Id")["Date"].min().rename("first")
    layout = (
        first.reset_index()
        .sort_values(["first", "Id"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )
    layout["y"] = (layout.index + 1) * 10

    def melted(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
        """One block, x rows then y rows, as CLVTools' ``melt`` produces."""
        return pd.concat([
            pd.DataFrame({
                "Id": frame["Id"], "type": kind, "variable": "x",
                "value": frame["x"].astype(str),
            }),
            pd.DataFrame({
                "Id": frame["Id"], "type": kind, "variable": "y",
                "value": frame["y"].astype(str),
            }),
        ], ignore_index=True)

    def dated(dates: pd.DataFrame) -> pd.DataFrame:
        return dates.merge(layout[["Id", "y"]], on="Id").assign(
            x=lambda f: f["Date"].dt.date
        )[["Id", "x", "y"]]

    blocks = [
        melted(layout.assign(x=layout["first"].dt.date)[["Id", "x", "y"]],
               "segment_start"),
        melted(layout.assign(x=data.data_end.date())[["Id", "x", "y"]],
               "segment_end"),
        melted(dated(data.as_data_frame("estimation", ids=ids)),
               "point_calibration"),
    ]
    if data.has_holdout:
        blocks.append(
            melted(dated(data.as_data_frame("holdout", ids=ids)), "point_holdout")
        )
    return pd.concat(blocks, ignore_index=True)


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
