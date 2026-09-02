"""S6.1 - preparing and inspecting transaction data.

This is the Python counterpart of ``clvdata()``. It turns a transaction log --
"the date and monetary value of the transaction as well as a customer
identifier" (S3) -- into the sufficient statistics every model in the paper
consumes:

    x      the number of *repeat* transactions in the estimation period
    t_x    the time of the last repeat transaction, from that customer's first
    T      the length of the estimation period, from that customer's first

S6.2.1: "the required model inputs (x_i, t_x_i, T_i) can be derived from the
purchase history of the estimation period alone, and no other data about the
customers or holdout period is needed."

Every worked example below is a doctest executed by ``pytest``, and the values
shown are the ones printed in the paper.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

from clvtools import timeunit
from clvtools.timeunit import TIME_UNITS

__all__ = [
    "TIME_UNITS",
    "ClvData",
    "ClvDataDynCov",
    "ClvDataStaticCov",
    "load_apparel_dyn_cov",
    "load_apparel_dyn_cov_future",
    "load_apparel_static_cov",
    "load_apparel_trans",
    "load_cdnow",
]

#: The bundled datasets, *inside* the package. They used to live at the
#: repository root, two directories up from here, which works in a source
#: checkout and fails in an installed one: the wheel carried no CSVs at all and
#: ``load_cdnow()`` raised ``FileNotFoundError`` pointing at a path under
#: ``site-packages/`` that never existed. Verified by installing the built
#: wheel into a throwaway environment, which is the only way to see it -- every
#: test in this repo runs from the checkout, where the old path resolved.
DATA_DIR = Path(__file__).resolve().parent / "data"

#: Every unit :class:`ClvData` accepts. See :mod:`clvtools.timeunit` for the
#: calendar arithmetic the month and year units need.


def load_apparel_trans() -> pd.DataFrame:
    """The apparel retailer's transaction log used throughout S6.

    S6.1: "It consists of 600 customers who purchased for the first time from
    this business on the day of 2005-01-02."

    >>> trans = load_apparel_trans()
    >>> len(trans), trans["Id"].nunique()
    (3187, 600)
    >>> print(trans.head(3).to_string(index=False))
    Id       Date  Price
     1 2005-01-02 230.30
     1 2005-09-06  84.39
     1 2006-01-18 131.07
    """
    return pd.read_csv(
        DATA_DIR / "apparelTrans.csv", dtype={"Id": str}, parse_dates=["Date"]
    )


def load_apparel_static_cov() -> pd.DataFrame:
    """Time-invariant covariates: gender, and the channel of the first purchase.

    S6.4: "we have the gender and the sales channel that a customer used for
    the first purchase as a time-invariant covariate."

    >>> print(load_apparel_static_cov().head(3).to_string(index=False))
    Id  Gender  Channel
     1       0        0
     2       1        0
     3       1        0
    """
    return pd.read_csv(DATA_DIR / "apparelStaticCov.csv", dtype={"Id": str})


def load_apparel_dyn_cov() -> pd.DataFrame:
    """Time-varying covariates: one row per customer per week.

    S6.4: "data on seasonal patterns (``High.Season``) is available as a
    time-varying covariate. It indicates whether a week falls into the high
    season or not."

    >>> dyn = load_apparel_dyn_cov()
    >>> print(dyn.head(3).to_string(index=False))
    Id   Cov.Date  High.Season  Gender  Channel
     1 2005-01-02            0       0        0
     1 2005-01-09            0       0        0
     1 2005-01-16            0       0        0
    """
    return pd.read_csv(
        DATA_DIR / "apparelDynCov.csv", dtype={"Id": str}, parse_dates=["Cov.Date"]
    )


def load_apparel_dyn_cov_future() -> pd.DataFrame:
    """The covariate series continued past the end of the transaction data.

    S6.4.2: "the time-varying covariates have to be available for the entire
    prediction period" -- the model needs a covariate value for every period it
    predicts over, and the transaction data stops before the horizon does.
    ``apparelDynCovFuture`` carries the series on from where
    :func:`load_apparel_dyn_cov` stops, and the two are concatenated before the
    prediction, as S6.4.2 does with ``rbind()``.

    >>> future = load_apparel_dyn_cov_future()
    >>> future["Cov.Date"].min().date(), future["Cov.Date"].max().date()
    (datetime.date(2011, 1, 2), datetime.date(2012, 10, 14))
    >>> past = load_apparel_dyn_cov()
    >>> past["Cov.Date"].max() < future["Cov.Date"].min()
    True
    """
    return pd.read_csv(
        DATA_DIR / "apparelDynCovFuture.csv",
        dtype={"Id": str},
        parse_dates=["Cov.Date"],
    )


def load_cdnow() -> pd.DataFrame:
    """The CDNOW transaction log bundled with CLVTools."""
    return pd.read_csv(
        DATA_DIR / "cdnow.csv", dtype={"Id": str}, parse_dates=["Date"]
    )


def _identified(df: pd.DataFrame) -> pd.DataFrame:
    """``Id`` as a string and ``Date`` as a timestamp, or an explanation.

    A row that does not say who or when is not a transaction. Both used to
    travel silently: an NA ``Id`` became the string ``"None"`` and an NA
    ``Date`` was dropped by ``to_datetime``, so transactions could leave the
    data between a caller's frame and the model with nothing said -- five of
    them, in the case that found this. Finding A4 of ``docs/spec-audit.md``.

    >>> import pandas as pd
    >>> _identified(pd.DataFrame({"Id": [None], "Date": ["2005-01-02"]}))
    Traceback (most recent call last):
        ...
    ValueError: 1 transaction has no Id; drop or repair those rows before modelling them
    """
    for column in ("Id", "Date"):
        missing = df[column].isna()
        if missing.any():
            count = int(missing.sum())
            raise ValueError(
                f"{count} transaction{'s have' if count > 1 else ' has'} no "
                f"{column}; drop or repair those rows before modelling them"
            )

    df["Id"] = df["Id"].astype(str)
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    unparsed = parsed.isna()
    if unparsed.any():
        raise ValueError(
            f"{int(unparsed.sum())} transaction dates could not be parsed, "
            f"e.g. {list(df['Date'][unparsed][:3])}"
        )
    df["Date"] = parsed
    return df


class ClvData:
    """A transaction log with an estimation/holdout split. Cf. ``clvdata()``.

    Parameters
    ----------
    transactions
        One row per purchase, with columns ``Id``, ``Date`` and optionally
        ``Price``. S6.1: "Every transaction record consists of a purchase date
        and a customer identifier. Optionally, the value of the transaction may
        be included."
    time_unit
        The unit in which all time spans are measured; one of ``TIME_UNITS``.
        ``"month"`` and ``"year"`` count calendar anniversaries rather than
        dividing by a fixed number of days -- see :mod:`clvtools.timeunit`.
    estimation_split
        Length of the estimation period in ``time_unit`` units, or a date, or
        ``None`` for no holdout period.
    data_end
        A fictional end of the data beyond the last purchase record. S6.1:
        "useful, for example, when the last purchase record was on 2000-12-29
        but customers were actually observed until 2000-12-31."

    Examples
    --------
    S6.1 splits the apparel data at 104 weeks:

    >>> clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> clv.estimation_start.date(), clv.estimation_end.date()
    (datetime.date(2005, 1, 2), datetime.date(2006, 12, 31))
    >>> clv.has_holdout
    True

    The model inputs for the first three customers:

    >>> print(clv.customer_summary().head(3).to_string(index=False))
    Id  x       t_x     T date_first_transaction
     1  6 93.285714 104.0             2005-01-02
    10  2 99.571429 104.0             2005-01-02
    100  0  0.000000 104.0             2005-01-02
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        time_unit: str = "week",
        estimation_split: float | str | pd.Timestamp | None = None,
        data_end: str | pd.Timestamp | None = None,
        name_id: str = "Id",
        name_date: str = "Date",
        name_price: str | None = "Price",
    ) -> None:
        self.time = timeunit.get(time_unit)
        self.time_unit = time_unit

        if not isinstance(transactions, pd.DataFrame):
            raise TypeError(
                "transactions must be a pandas DataFrame with Id and Date "
                f"columns, not {type(transactions).__name__}"
            )
        if transactions.empty:
            raise ValueError("transaction data is empty: there is nothing to model")

        cols = {name_id: "Id", name_date: "Date"}
        has_price = name_price is not None and name_price in transactions.columns
        if has_price:
            cols[name_price] = "Price"
        elif name_price not in (None, "Price"):
            # An explicit column that is not there is a typo, not a decision to
            # model without spending. Naming it costs one line; not naming it
            # costs a silent switch to a transaction-only model, which surfaces
            # much later as `no Price column: spending cannot be modelled`.
            raise ValueError(
                f"name_price={name_price!r} is not a column of the transaction "
                f"data; pass name_price=None to model transactions only"
            )

        missing = [c for c in cols if c not in transactions.columns]
        if missing:
            raise ValueError(f"transaction data is missing columns: {missing}")

        df = _identified(transactions[list(cols)].rename(columns=cols).copy())
        if not has_price:
            df["Price"] = np.nan
        elif not np.isfinite(df["Price"].to_numpy(dtype=float)).all():
            # A customer whose prices are all NaN is counted in ``x`` and
            # dropped from the mean, and ``fillna(0.0)`` then records
            # ``Spending = 0``. The Gamma-Gamma silently excludes that row and
            # ``predict()`` reports the population mean for the customer --
            # 7.72 against 88.65 with real prices, in the review's example.
            # Finding 6 of ``docs/review-2026-09-02.md``.
            bad = df.loc[~np.isfinite(df["Price"].to_numpy(dtype=float)), "Id"]
            raise ValueError(
                f"Price is not finite for {len(bad)} transaction"
                f"{'s' if len(bad) > 1 else ''} "
                f"(customers e.g. {sorted(set(bad))[:3]}); "
                "drop or impute those rows, or pass name_price=None"
            )
        self.has_spending = has_price

        self.transactions = self._aggregate_to_day(df)

        self.estimation_start = self.transactions["Date"].min()
        last_record = self.transactions["Date"].max()
        self.data_end = (
            last_record if data_end is None else pd.Timestamp(data_end)
        )
        if self.data_end < last_record:
            raise ValueError(
                f"data_end {self.data_end.date()} precedes the last purchase "
                f"record {last_record.date()}"
            )

        self.estimation_end = self._resolve_split(estimation_split)
        self.has_holdout = self.estimation_end < self.data_end
        self.holdout_end = self.data_end if self.has_holdout else None

    # -- construction helpers -------------------------------------------------

    def _aggregate_to_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse the log so at most one record remains per customer-day.

        S6.1: "For any customer-day combination, multiple purchases are combined
        into a single record whose transaction count equals one and whose
        monetary value equals the sum of that day's spending. [...] Often
        purchases on the same day should not be regarded as separate,
        independent events [...] The Poisson process explicitly assumes that the
        events are independent."
        """
        floor = "h" if self.time_unit == "hour" else "D"  # S6.1
        df = df.copy()
        df["Date"] = df["Date"].dt.floor(floor)
        return (
            df.groupby(["Id", "Date"], as_index=False, sort=True)["Price"]
            .sum(min_count=1)
            .sort_values(["Id", "Date"], kind="stable")
            .reset_index(drop=True)
        )

    def _resolve_split(
        self, split: float | str | pd.Timestamp | None
    ) -> pd.Timestamp:
        """Turn ``estimation_split`` into the date the estimation period ends.

        S6.1: "We use ``estimation.split`` to set the estimation period to 104
        periods. This means, 104 weeks since the first purchase record.
        Alternatively, a date can be provided."
        """
        if split is None:
            return self.data_end
        if isinstance(split, (int, float, np.integer, np.floating)):
            end = self.time.add(self.estimation_start, float(split))
        else:
            end = pd.Timestamp(split)
        if end > self.data_end:
            raise ValueError(
                f"estimation period ends {end.date()}, after the data ends "
                f"{self.data_end.date()}"
            )
        if end <= self.estimation_start:
            raise ValueError("estimation period must be longer than zero")
        return end

    def _elapsed(self, start: pd.Series, end) -> pd.Series:
        """A span between dates, expressed in ``time_unit`` units.

        Delegated to the unit because a calendar unit cannot be a division:
        see :mod:`clvtools.timeunit`.
        """
        if not hasattr(end, "__len__"):
            end = pd.Series([end] * len(start), index=start.index)
        return pd.Series(
            [self.time.elapsed(a, b) for a, b in zip(start, end, strict=True)],
            index=start.index,
        )

    # -- model inputs ---------------------------------------------------------

    def customer_summary(self) -> pd.DataFrame:
        """``(x, t_x, T)`` per customer over the estimation period.

        Returns a frame with columns ``Id``, ``x``, ``t_x``, ``T`` and
        ``date_first_transaction``.

        ``x`` counts *repeat* transactions, so a customer seen once has
        ``x = 0``. S3: "Most applications of probabilistic models for customer
        base analysis focus on modeling repeat transactions that occur after the
        first transaction of a customer."

        Examples
        --------
        The apparel cohort is a single acquisition cohort, so ``T`` is the same
        104 weeks for every customer:

        >>> clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        >>> summary = clv.customer_summary()
        >>> len(summary), summary["T"].unique()
        (600, array([104.]))

        With no holdout period the window runs to the last purchase record,
        2010-12-20, which is 311.14 weeks after the cohort's first purchase:

        >>> full = ClvData(load_apparel_trans(), time_unit="week", estimation_split=None)
        >>> round(float(full.customer_summary()["T"].iloc[0]), 6)
        311.142857
        """
        est = self.transactions[self.transactions["Date"] <= self.estimation_end]
        grouped = est.groupby("Id", sort=True)["Date"]

        first = grouped.min()
        # The last *repeat* transaction: for a customer seen once this is the
        # first transaction itself, which the t_x definition then zeroes out.
        last = grouped.max()
        n = grouped.size()

        summary = pd.DataFrame({
            "Id": first.index,
            "x": (n - 1).to_numpy(),
            "t_x": self._elapsed(first, last).to_numpy(),
            "T": self._elapsed(first, self.estimation_end).to_numpy(),
            "date_first_transaction": first.to_numpy(),
        })
        # S7 of Fader et al., carried over here: "If x_i = 0, t_x_i = 0."
        summary.loc[summary["x"] == 0, "t_x"] = 0.0
        return summary.reset_index(drop=True)

    def spending_summary(self, remove_first_transaction: bool = True) -> pd.DataFrame:
        """Average spend per transaction, per customer. Cf. the Gamma-Gamma input.

        S6.2.3: "CLVTools by default does not use the first transaction when
        estimating a spending model because in many cases this transaction has
        been found to be atypical for future purchases. As a consequence,
        customers with a single purchase are ignored during model estimation."

        Returns ``Id``, ``x`` (transactions counted) and ``Spending``
        (their mean value, eq. 13: ``z_bar = sum(z_i) / x``). Customers
        contributing nothing are kept with ``x = 0`` and ``Spending = 0`` so the
        frame stays aligned with :meth:`customer_summary`.

        Examples
        --------
        >>> clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        >>> print(clv.spending_summary().head(3).to_string(index=False))
        Id  x  Spending
         1  6   101.415
        10  2    43.900
        100  0     0.000
        """
        if not self.has_spending:
            raise ValueError(
                "no Price column: spending cannot be modelled without it"
            )
        est = self.transactions[self.transactions["Date"] <= self.estimation_end].copy()

        if remove_first_transaction:
            first_idx = est.groupby("Id")["Date"].idxmin()
            est = est.drop(index=first_idx)

        agg = est.groupby("Id", sort=True)["Price"].agg(["size", "mean"])
        all_ids = pd.Index(
            sorted(self.transactions["Id"].unique()), name="Id"
        )
        agg = agg.reindex(all_ids).fillna(0.0)

        return pd.DataFrame({
            "Id": agg.index,
            "x": agg["size"].astype(int).to_numpy(),
            "Spending": agg["mean"].to_numpy(),
        }).reset_index(drop=True)

    # -- inspecting the data, S6.1.2 ------------------------------------------

    @property
    def holdout_start(self) -> pd.Timestamp | None:
        """The first timepoint of the holdout period, or ``None`` without one.

        One time step past the estimation period, which is a day -- or an hour
        on hourly data, the one unit finer than a day that S6.1 allows.
        """
        if not self.has_holdout:
            return None
        step = pd.Timedelta(hours=1) if self.time_unit == "hour" else pd.Timedelta(days=1)
        return self.estimation_end + step

    def nobs(self) -> int:
        """The number of customers. Cf. ``nobs()`` on a data object."""
        return int(self.transactions["Id"].nunique())

    def _resolve_ids(self, ids: Sequence[str] | str) -> list[str]:
        """Customer ids as a list, rejecting any this data has never seen.

        CLVTools filters leniently: ``summary(clv, ids = "1219")`` on data
        whose ids run 1..600 returns a table of ``Inf``, ``-Inf`` and ``NaN``
        with a warning rather than an error, and both examples in
        ``?summary.clv.data`` do exactly that. A summary of no customers is
        not an answer to any question worth asking, so this raises.
        """
        wanted = [ids] if isinstance(ids, str) else [str(i) for i in ids]
        known = set(self.transactions["Id"])
        missing = [i for i in wanted if i not in known]
        if missing:
            raise ValueError(
                f"no transactions for {len(missing)} of the ids given, "
                f"e.g. {missing[:3]}"
            )
        return wanted

    def as_data_frame(
        self, sample: str = "full", ids: Sequence[str] | str | None = None
    ) -> pd.DataFrame:
        """The transaction log itself. Cf. ``as.data.frame()`` in S6.1.2.

        S6.1.2 takes the data out three ways: ``sample = "full"``,
        ``sample = "estimation"`` and ``ids = "1"``. ``sample`` defaults to
        ``"full"``, as it does there -- unlike the descriptive plots, which
        default to the estimation period.

        The frame is post-aggregation -- at most one row per customer-day, per
        S6.1 -- so it is shorter than the log that was passed in.

        R's ``subset()`` has no Python counterpart here because pandas already
        has one: ``clv.as_data_frame(sample="holdout").query('Price >= 50')``
        is ``subset(clv, Price >= 50, sample = "holdout")``.

        One trap in that idiom, on dates. pandas does not coerce a bare string
        on the right of ``==``, so a date equality written inline matches
        *nothing* rather than raising -- while the range comparisons beside it
        coerce perfectly well. Bind the timestamp, or use a mask.

        Examples
        --------
        >>> clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        >>> len(clv.as_data_frame()), len(clv.as_data_frame(sample="estimation"))
        (3183, 1866)
        >>> len(clv.as_data_frame(ids="1"))
        7

        The date trap, and the two spellings that avoid it:

        >>> frame = ClvData(load_cdnow(), time_unit="week").as_data_frame()
        >>> len(frame.query('Date == "1997-02-16"'))          # silently empty
        0
        >>> when = pd.Timestamp("1997-02-16")
        >>> len(frame.query("Date == @when")), len(frame[frame["Date"] == when])
        (44, 44)
        """
        frame = self._sample(sample)
        if ids is not None:
            frame = frame[frame["Id"].isin(self._resolve_ids(ids))]
        return frame.reset_index(drop=True)

    def _sample(self, sample: str) -> pd.DataFrame:
        """The transactions of one period. ``sample`` is estimation/holdout/full."""
        if sample == "full":
            return self.transactions
        if sample == "estimation":
            return self.transactions[self.transactions["Date"] <= self.estimation_end]
        if sample == "holdout":
            if not self.has_holdout:
                raise ValueError("the data has no holdout period")
            return self.transactions[self.transactions["Date"] >= self.holdout_start]
        raise ValueError(
            f"sample must be one of estimation, holdout, full; got {sample!r}"
        )

    def mean_interpurchase_times(
        self, sample: str = "estimation", ids: Sequence[str] | str | None = None
    ) -> pd.DataFrame:
        """Each customer's mean time between transactions, in ``time_unit``.

        S6.1.2: "the empirical density of customers' mean time between
        transactions, after aggregating purchases of the same customer on the
        same date. [...] Only data from customers with repeat transactions are
        shown in this graph."

        Customers with a single transaction in the sample have no interpurchase
        time at all and are returned as ``NaN`` rather than dropped, which is
        what lets :meth:`summary` average over them with the same expression
        CLVTools uses.
        """
        frame = self._sample(sample).sort_values(["Id", "Date"], kind="stable")
        if ids is not None:
            frame = frame[frame["Id"].isin(self._resolve_ids(ids))]
        gaps: dict[str, float] = {}
        for customer, dates in frame.groupby("Id", sort=True)["Date"]:
            when = dates.to_numpy()
            if len(when) < 2:
                gaps[customer] = np.nan
                continue
            spans = [
                self.time.elapsed(pd.Timestamp(a), pd.Timestamp(b))
                for a, b in pairwise(when)
            ]
            gaps[customer] = float(np.mean(spans))
        return pd.DataFrame(
            {"Id": list(gaps), "mean.interpurchase.time": list(gaps.values())}
        )

    def summary(self, ids: Sequence[str] | str | None = None) -> pd.DataFrame:
        """The descriptive statistics of S6.1.2. Cf. ``summary()``.

        One row per statistic and one column per sample -- ``Estimation``,
        ``Holdout`` (only when there is one) and ``Total`` -- holding the values
        themselves rather than the three-decimal strings CLVTools prints:
        timestamps for the four date rows, floats for the rest, and ``None``
        where a statistic does not apply to that sample.

        Examples
        --------
        The paper prints this table for the apparel data. 35.5% of customers
        are zero repeaters, and the estimation period holds 1,866 of the 3,183
        transactions that remain after same-day aggregation:

        >>> clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        >>> table = clv.summary()
        >>> table.loc["Percentage of zero repeaters", "Estimation"]
        35.5
        >>> [int(table.loc["Total # Transactions", c]) for c in table.columns]
        [1866, 1317, 3183]
        >>> round(table.loc["Mean Interpurchase time", "Estimation"], 3)
        24.823

        Spending statistics need a ``Price`` column; without one those rows are
        absent rather than empty.
        """
        samples = ["Estimation"] + (["Holdout"] if self.has_holdout else []) + ["Total"]
        return pd.DataFrame(
            {name: self._descriptives(name, ids) for name in samples}
        )

    def _descriptives(
        self, sample: str, ids: Sequence[str] | str | None = None
    ) -> pd.Series:
        """One column of :meth:`summary`, in CLVTools' own row order."""
        which = {"Estimation": "estimation", "Holdout": "holdout"}.get(sample, "full")
        frame = self._sample(which)
        if ids is not None:
            frame = frame[frame["Id"].isin(self._resolve_ids(ids))]
        per_customer = frame.groupby("Id", sort=True)["Date"].size()
        gaps = self.mean_interpurchase_times(which, ids)["mean.interpurchase.time"]

        start, end = {
            "Estimation": (self.estimation_start, self.estimation_end),
            "Holdout": (self.holdout_start, self.data_end),
        }.get(sample, (self.estimation_start, self.data_end))

        values: dict[str, object] = {
            "Period Start": start,
            "Period End": end,
            # Only the total counts customers: the other two samples are
            # windows on the same cohort, so the count would not mean anything
            # different. CLVTools prints "-" there.
            "Number of customers": (
                float(frame["Id"].nunique()) if sample == "Total" else None
            ),
            "First Transaction in period": frame["Date"].min(),
            "Last Transaction in period": frame["Date"].max(),
            "Total # Transactions": float(len(frame)),
            "Mean # Transactions per cust": float(per_customer.mean()),
            "(SD) # Transactions": float(per_customer.std(ddof=1)),
        }
        if self.has_spending:
            values["Mean Spending per Transaction"] = float(frame["Price"].mean())
            values["(SD) Spending"] = float(frame["Price"].std(ddof=1))
            values["Total Spending"] = float(frame["Price"].sum())
        # A zero repeater is a customer who never came back, which only the
        # estimation period can establish.
        is_estimation = sample == "Estimation"
        values["Total # zero repeaters"] = (
            float((per_customer == 1).sum()) if is_estimation else None
        )
        values["Percentage of zero repeaters"] = (
            float((per_customer == 1).mean() * 100) if is_estimation else None
        )
        values["Mean Interpurchase time"] = float(gaps.mean(skipna=True))
        values["(SD) Interpurchase time"] = float(gaps.std(ddof=1, skipna=True))
        return pd.Series(values, dtype=object)

    def __repr__(self) -> str:
        span = "no holdout" if not self.has_holdout else (
            f"holdout to {self.data_end.date()}"
        )
        return (
            f"ClvData({self.transactions['Id'].nunique()} customers, "
            f"{len(self.transactions)} transactions, {self.time_unit}s, "
            f"estimation {self.estimation_start.date()}"
            f"..{self.estimation_end.date()}, {span})"
        )


#: A formula term wrapping an expression to evaluate, as R's ``I()`` does.
_TRANSFORMED_TERM = re.compile(r"^I\s*\((?P<expression>.+)\)$", re.DOTALL)


class ClvDataStaticCov(ClvData):
    """Transaction data with time-invariant covariates. Cf. ``SetStaticCovariates()``.

    S6.4: "The arguments ``data.cov.life`` and ``data.cov.trans`` are the
    ``data.frame`` or ``data.table`` that contain the covariate data for the
    attrition and the transaction process, respectively. If a covariate can
    affect both processes it has to be added in both arguments."

    The two processes need not share covariates, so the design matrices are
    built and stored separately.

    Examples
    --------
    S6.4 adds gender and acquisition channel to both processes:

    >>> clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> static = ClvDataStaticCov(
    ...     clv, load_apparel_static_cov(),
    ...     names_cov_life=["Gender", "Channel"],
    ...     names_cov_trans=["Gender", "Channel"],
    ... )
    >>> static.names_cov_life
    ['Gender', 'Channel']
    >>> static.design_life().shape
    (600, 2)
    """

    def __init__(
        self,
        clv_data: ClvData,
        data_cov_life: pd.DataFrame,
        data_cov_trans: pd.DataFrame | None = None,
        names_cov_life: list[str] | None = None,
        names_cov_trans: list[str] | None = None,
        name_id: str = "Id",
    ) -> None:
        # Copy the parent's state rather than re-deriving it, so the covariate
        # object and the plain one always agree on (x, t_x, T).
        self.__dict__.update(clv_data.__dict__)

        if data_cov_trans is None:
            data_cov_trans = data_cov_life

        customers = pd.Index(sorted(self.transactions["Id"].unique()), name="Id")
        self._cov_life = self._prepare(data_cov_life, name_id, customers, "life")
        self._cov_trans = self._prepare(data_cov_trans, name_id, customers, "trans")

        self.names_cov_life = list(
            names_cov_life
            if names_cov_life is not None
            else self._cov_life.columns
        )
        self.names_cov_trans = list(
            names_cov_trans
            if names_cov_trans is not None
            else self._cov_trans.columns
        )
        for names, frame, which in (
            (self.names_cov_life, self._cov_life, "life"),
            (self.names_cov_trans, self._cov_trans, "trans"),
        ):
            missing = [n for n in names if n not in frame.columns]
            if missing:
                raise ValueError(f"{which} covariates not in the data: {missing}")

        self.customers = customers

    @staticmethod
    def _prepare(
        frame: pd.DataFrame, name_id: str, customers: pd.Index, which: str
    ) -> pd.DataFrame:
        """One row per customer, in the order the customer summary uses.

        S6.4: "Categorical data (``factor`` and ``character``) is turned into
        k-1 dummy variables."
        """
        if name_id not in frame.columns:
            raise ValueError(f"{which} covariate data has no {name_id!r} column")
        out = frame.copy()
        out[name_id] = out[name_id].astype(str)
        out = out.set_index(name_id)

        missing = customers.difference(out.index)
        if len(missing):
            raise ValueError(
                f"{which} covariate data is missing {len(missing)} customers, "
                f"e.g. {list(missing[:3])}"
            )
        # A repeated id makes ``.loc`` return more rows than there are
        # customers -- 601 for 600 in the review's example -- and the mismatch
        # then surfaces as a broadcast error deep inside the likelihood, which
        # names neither the customer nor the frame. Finding 12.
        duplicated = out.index[out.index.duplicated()].unique()
        if len(duplicated):
            raise ValueError(
                f"{which} covariate data has {len(duplicated)} duplicated "
                f"customer id{'s' if len(duplicated) > 1 else ''}, "
                f"e.g. {list(duplicated[:3])}: one row per customer is required"
            )
        out = out.loc[customers]

        # Everything that is not already a number becomes dummies. Selected by
        # dtype predicate rather than ``select_dtypes(include=["object", ...])``
        # because pandas 3 gives string columns their own ``str`` dtype, which
        # that call stopped matching; datetime columns stay out of it either way.
        categorical = [
            name
            for name, dtype in out.dtypes.items()
            if is_object_dtype(dtype)
            or is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or is_bool_dtype(dtype)
        ]
        if categorical:
            out = pd.get_dummies(out, columns=categorical, drop_first=True)
        out = out.astype(float)

        # A NaN here is not a modelling choice, it is a hole. Left alone it
        # travels: ``exp(gamma' x)`` is NaN, every customer's likelihood
        # contribution is NaN, the objective is ``-inf`` everywhere and the
        # covariate "fit" comes back at its start values with no exception and
        # several hundred RuntimeWarnings. Named here, with the ids, it is a
        # data-cleaning problem the caller can act on -- finding 6 of
        # ``docs/review-2026-09-02.md``.
        bad = out.index[~np.isfinite(out.to_numpy()).all(axis=1)]
        if len(bad):
            raise ValueError(
                f"{which} covariate data is not finite for {len(bad)} "
                f"customer{'s' if len(bad) > 1 else ''}, e.g. {list(bad[:3])}"
            )
        return out

    def with_covariates(
        self, names_life: list[str] | None = None,
        names_trans: list[str] | None = None,
    ) -> ClvDataStaticCov:
        """The same data restricted to the covariates a formula names.

        S6.4's formula selects from the covariates the data object carries;
        this applies that selection without re-preparing the design matrices.
        ``None`` on either side keeps what is already there.

        A term wrapped in ``I(...)`` is an expression to evaluate rather than a
        column to select, as in R -- ``?latentAttrition`` fits
        ``~ Gender | I(log(Channel + 2))``. The derived column is named after
        the term exactly as it was written, so the coefficient carries its own
        definition. R names it by deparsing, which respaces the expression to
        ``I(log(Channel + 2))``; nothing here reformats it, so write the term
        the way you want the coefficient labelled.

        Examples
        --------
        >>> data = ClvDataStaticCov(
        ...     ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
        ...     load_apparel_static_cov(),
        ... )
        >>> derived = data.with_covariates(["Gender"], ["I(log(Channel + 2))"])
        >>> derived.names_cov_trans
        ['I(log(Channel + 2))']
        >>> sorted({float(v) for v in derived.design_trans().ravel().round(6)})
        [0.693147, 1.098612]
        """
        other = copy.copy(self)
        for attribute, source, wanted in (
            ("names_cov_life", "_cov_life", names_life),
            ("names_cov_trans", "_cov_trans", names_trans),
        ):
            if wanted is None:
                continue
            resolved, frame = self._evaluate_terms(getattr(self, source), wanted)
            missing = [n for n in resolved if n not in frame.columns]
            if missing:
                raise ValueError(f"covariates not in the data: {missing}")
            setattr(other, source, frame)
            setattr(other, attribute, resolved)
        return other

    @staticmethod
    def _evaluate_terms(
        frame: pd.DataFrame, terms: Sequence[str]
    ) -> tuple[list[str], pd.DataFrame]:
        """Turn every ``I(...)`` term into a column, leaving plain names alone.

        The expression is handed to :meth:`pandas.DataFrame.eval` rather than
        to :func:`eval`, so it reaches the covariate columns and arithmetic and
        not the interpreter.
        """
        resolved, derived = [], {}
        for term in terms:
            name = str(term).strip()
            match = _TRANSFORMED_TERM.match(name)
            resolved.append(name)
            if match is None or name in frame.columns:
                continue
            expression = match.group("expression")
            try:
                values = frame.eval(expression)
            except Exception as error:
                raise ValueError(
                    f"cannot evaluate the covariate expression {expression!r}: {error}"
                ) from error
            derived[name] = np.asarray(values, dtype=float)
        if not derived:
            return resolved, frame
        frame = frame.copy()
        for name, values in derived.items():
            frame[name] = values
        return resolved, frame

    def design_life(self, names: list[str] | None = None) -> np.ndarray:
        r"""The attrition process's covariate matrix, customers in summary order."""
        return self._cov_life[names or self.names_cov_life].to_numpy(dtype=float)

    def design_trans(self, names: list[str] | None = None) -> np.ndarray:
        r"""The transaction process's covariate matrix."""
        return self._cov_trans[names or self.names_cov_trans].to_numpy(dtype=float)


class ClvDataDynCov(ClvData):
    """Transaction data with time-varying covariates. Cf. ``SetDynamicCovariates()``.

    S6.4: "Data for time-varying covariates require a time series of covariate
    values for every customer. In other words, if a time-varying covariate is
    included and the analysis is done based on weekly data, the covariate value
    can change every week. Thus, a value has to be specified for every customer
    every week."

    The covariate date marks the *start* of the period it describes, so the
    intervals are :math:`[\\text{Cov.Date}_k, \\text{Cov.Date}_{k+1})`.

    Examples
    --------
    S6.4 adds seasonality alongside the two time-invariant covariates:

    >>> clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> dynamic = ClvDataDynCov(
    ...     clv, load_apparel_dyn_cov(),
    ...     names_cov_life=["High.Season", "Gender", "Channel"],
    ...     names_cov_trans=["High.Season", "Gender", "Channel"])
    >>> walks = dynamic.walks()
    >>> walks.n_customers, walks.n_cov_life
    (600, 3)

    ``Gender`` and ``Channel`` do not actually vary; S6.4 explains that they are
    repeated anyway because "the data structure of time-invariant covariates
    [...] needs to be aligned with the structure of time-varying covariates".
    """

    def __init__(
        self,
        clv_data: ClvData,
        data_cov_life: pd.DataFrame,
        data_cov_trans: pd.DataFrame | None = None,
        names_cov_life: list[str] | None = None,
        names_cov_trans: list[str] | None = None,
        name_date_cov: str = "Cov.Date",
    ) -> None:
        self.__dict__.update(clv_data.__dict__)
        self.data_cov_life = data_cov_life
        self.data_cov_trans = (
            data_cov_life if data_cov_trans is None else data_cov_trans
        )
        self.names_cov_life = names_cov_life
        self.names_cov_trans = names_cov_trans
        self.name_date_cov = name_date_cov

    def with_covariates(
        self, names_life: list[str] | None = None,
        names_trans: list[str] | None = None,
    ) -> ClvDataDynCov:
        """The same data restricted to the covariates a formula names.

        The walks are rebuilt on demand, since which covariates are in the
        model changes them.
        """
        other = copy.copy(self)
        other.__dict__.pop("_walks", None)
        available = set(self.data_cov_life.columns) | set(
            self.data_cov_trans.columns
        )
        for attribute, wanted in (
            ("names_cov_life", names_life), ("names_cov_trans", names_trans),
        ):
            if wanted is None:
                continue
            missing = [n for n in wanted if n not in available]
            if missing:
                raise ValueError(f"covariates not in the data: {missing}")
            setattr(other, attribute, list(wanted))
        return other

    def walks(self):
        """Build the walk structures the likelihood consumes.

        Cached on first use: assembling them touches every covariate row for
        every customer, and the result does not depend on the parameters.
        """
        if not hasattr(self, "_walks"):
            from clvtools.pnbd.dyncov import build_walks

            self._walks = build_walks(
                self,
                self.data_cov_life,
                self.data_cov_trans,
                names_cov_life=self.names_cov_life,
                names_cov_trans=self.names_cov_trans,
                name_date_cov=self.name_date_cov,
            )
        return self._walks
