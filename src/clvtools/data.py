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

from pathlib import Path

import numpy as np
import pandas as pd

from clvtools import timeunit
from clvtools.timeunit import TIME_UNITS

__all__ = [
    "ClvData",
    "ClvDataDynCov",
    "ClvDataStaticCov",
    "TIME_UNITS",
    "load_apparel_dyn_cov",
    "load_apparel_static_cov",
    "load_apparel_trans",
    "load_cdnow",
]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

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


def load_cdnow() -> pd.DataFrame:
    """The CDNOW transaction log bundled with CLVTools."""
    return pd.read_csv(
        DATA_DIR / "cdnow.csv", dtype={"Id": str}, parse_dates=["Date"]
    )


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
        estimation_split: int | float | str | pd.Timestamp | None = None,
        data_end: str | pd.Timestamp | None = None,
        name_id: str = "Id",
        name_date: str = "Date",
        name_price: str | None = "Price",
    ) -> None:
        self.time = timeunit.get(time_unit)
        self.time_unit = time_unit

        cols = {name_id: "Id", name_date: "Date"}
        has_price = name_price is not None and name_price in transactions.columns
        if has_price:
            cols[name_price] = "Price"

        missing = [c for c in cols if c not in transactions.columns]
        if missing:
            raise ValueError(f"transaction data is missing columns: {missing}")

        df = transactions[list(cols)].rename(columns=cols).copy()
        df["Id"] = df["Id"].astype(str)
        df["Date"] = pd.to_datetime(df["Date"])
        if not has_price:
            df["Price"] = np.nan
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
        out = (
            df.groupby(["Id", "Date"], as_index=False, sort=True)["Price"]
            .sum(min_count=1)
            .sort_values(["Id", "Date"], kind="stable")
            .reset_index(drop=True)
        )
        return out

    def _resolve_split(
        self, split: int | float | str | pd.Timestamp | None
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
        end = pd.Series([end] * len(start), index=start.index) if not hasattr(end, "__len__") else end
        return pd.Series(
            [self.time.elapsed(a, b) for a, b in zip(start, end)], index=start.index
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
        out = out.loc[customers]

        categorical = out.select_dtypes(include=["object", "category", "bool"]).columns
        if len(categorical):
            out = pd.get_dummies(out, columns=list(categorical), drop_first=True)
        return out.astype(float)

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
