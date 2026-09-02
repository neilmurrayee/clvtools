"""S6.1 - the data layer: a transaction log reduced to model inputs.

Every model in the paper consumes the same three numbers per customer:

    x      the number of *repeat* transactions in the estimation period
    t_x    the time of the last repeat transaction, from that customer's first
    T      the length of the estimation period, from that customer's first

S6.1: "Recall that the required model inputs (x_i, t_x_i, T_i) can be derived
from the purchase history of the estimation period alone."

Two details of S6.1 that the arithmetic depends on:

  * for weekly units the log is first collapsed to the day - "For any
    customer-day combination, multiple purchases are combined into a single
    record whose transaction count equals one and whose monetary value equals
    the sum of that day's spending" - because the Poisson process assumes
    independent events;
  * consequently time is measured in whole days, so with ``time.unit="week"``
    spans can only fall on steps of 1/7.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from paper_values import (
    COHORT_START,
    DATA_END,
    ESTIMATION_END,
    ESTIMATION_WEEKS,
    N_CUSTOMERS,
    N_TRANSACTIONS,
)

from clvtools.data import ClvData


class TestTheDatasetsShipWithThePackage:
    """The wheel has to carry the CSVs, and no test in a checkout can see it.

    ``load_cdnow()`` resolved ``__file__/../../../data``, which is the
    repository root in a source tree and a directory that does not exist under
    ``site-packages``. Every test here passed while the built wheel raised
    ``FileNotFoundError`` on its first call, because the checkout always has
    the files whether or not they are packaged. The two checks below are what
    that costs to prevent: the data has to be *inside* the package, and it has
    to be reachable the way an installed package reaches it.
    """

    def test_the_data_lives_inside_the_package(self):
        import clvtools.data as module

        package = pathlib.Path(module.__file__).resolve().parent
        assert module.DATA_DIR.is_relative_to(package), (
            f"DATA_DIR is {module.DATA_DIR}, outside the package at {package}. "
            "Anything outside it is absent from the wheel."
        )

    def test_every_dataset_is_a_package_resource(self):
        """Reached through ``importlib.resources``, as an installed one is."""
        from importlib import resources

        root = resources.files("clvtools") / "data"
        for name in (
            "apparelTrans", "apparelStaticCov", "apparelDynCov",
            "apparelDynCovFuture", "cdnow",
        ):
            assert (root / f"{name}.csv").is_file(), f"{name}.csv is not packaged"

class TestApparelTrans:
    """The dataset itself, as described in S6.1."""

    def test_shape_matches_paper(self, apparel_trans):
        assert len(apparel_trans) == N_TRANSACTIONS
        assert apparel_trans["Id"].nunique() == N_CUSTOMERS

    def test_is_a_single_acquisition_cohort(self, apparel_trans):
        """"600 customers who purchased for the first time [...] 2005-01-02"."""
        first = apparel_trans.groupby("Id")["Date"].min()
        assert (first == pd.Timestamp(COHORT_START)).all()

    def test_head_matches_paper(self, apparel_trans):
        head = apparel_trans.head(3)
        assert list(head["Id"]) == ["1", "1", "1"]
        assert list(head["Date"].dt.strftime("%Y-%m-%d")) == [
            "2005-01-02", "2005-09-06", "2006-01-18",
        ]
        assert list(head["Price"]) == pytest.approx([230.30, 84.39, 131.07])


class TestEstimationPeriod:
    """``estimation.split = 104`` -- 104 weeks from the first purchase record."""

    def test_estimation_end_matches_paper(self, apparel_trans):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        assert clv.estimation_end == pd.Timestamp(ESTIMATION_END)

    def test_estimation_start_is_first_transaction_in_data(self, apparel_trans):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        assert clv.estimation_start == pd.Timestamp(COHORT_START)

    def test_holdout_runs_to_the_last_transaction(self, apparel_trans):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        assert clv.holdout_end == pd.Timestamp(DATA_END)

    def test_no_split_means_no_holdout(self, apparel_trans):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=None)
        assert clv.has_holdout is False
        assert clv.estimation_end == pd.Timestamp(DATA_END)


class TestCustomerSummary:
    """``(x, t_x, T)`` per customer, against the oracle's CBS."""

    def test_matches_oracle_over_estimation_period(self, apparel_trans, cbs_estimation):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        got = clv.customer_summary().set_index("Id").sort_index()
        want = cbs_estimation.set_index("Id").sort_index()

        assert list(got.index) == list(want.index)
        np.testing.assert_array_equal(got["x"], want["x"])
        np.testing.assert_allclose(got["t_x"], want["t.x"], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(got["T"], want["T.cal"], rtol=1e-12, atol=1e-9)

    def test_matches_oracle_over_full_period(self, apparel_trans, cbs_full):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=None)
        got = clv.customer_summary().set_index("Id").sort_index()
        want = cbs_full.set_index("Id").sort_index()

        np.testing.assert_array_equal(got["x"], want["x"])
        np.testing.assert_allclose(got["t_x"], want["t.x"], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(got["T"], want["T.cal"], rtol=1e-12, atol=1e-9)

    def test_first_transaction_date_matches_oracle(self, apparel_trans, cbs_estimation):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        got = clv.customer_summary().set_index("Id").sort_index()
        want = cbs_estimation.set_index("Id").sort_index()
        assert (
            got["date_first_transaction"].dt.strftime("%Y-%m-%d")
            == want["date.first.actual.trans"]
        ).all()

    def test_x_counts_repeat_transactions_only(self, apparel_trans):
        """x excludes the first transaction, so a one-purchase customer has x=0."""
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        summary = clv.customer_summary()
        assert (summary["x"] >= 0).all()
        assert (summary.loc[summary["x"] == 0, "t_x"] == 0).all()

    def test_recency_never_exceeds_the_observation_window(self, apparel_trans):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        summary = clv.customer_summary()
        assert (summary["t_x"] <= summary["T"] + 1e-9).all()

    def test_times_fall_on_whole_days(self, apparel_trans):
        """S6.1: with weekly units, spans can only fall on steps of 1/7."""
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        days = clv.customer_summary()["t_x"] * 7.0
        np.testing.assert_allclose(days, np.round(days), atol=1e-9)


class TestDayLevelAggregation:
    """S6.1: same-day purchases count once, and their spending is summed."""

    def test_same_day_purchases_collapse_to_one_transaction(self):
        log = pd.DataFrame({
            "Id": ["a", "a", "a"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-08"]),
            "Price": [10.0, 20.0, 5.0],
        })
        clv = ClvData(log, time_unit="week", estimation_split=None)
        row = clv.customer_summary().iloc[0]
        assert row["x"] == 1
        assert row["t_x"] == pytest.approx(1.0)

    def test_same_day_spending_is_summed(self):
        log = pd.DataFrame({
            "Id": ["a", "a", "a"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-08"]),
            "Price": [10.0, 20.0, 5.0],
        })
        clv = ClvData(log, time_unit="week", estimation_split=None)
        assert clv.transactions["Price"].tolist() == pytest.approx([10.0, 25.0])


class TestSpendingSummary:
    """S6.2.3 - the input to the Gamma-Gamma model."""

    def test_matches_oracle(self, apparel_trans):
        from conftest import fixture_csv

        clv = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_WEEKS)
        got = clv.spending_summary().set_index("Id").sort_index()
        want = fixture_csv("cbs_spending_estimation").set_index("Id").sort_index()

        assert list(got.index) == list(want.index)
        np.testing.assert_array_equal(got["x"], want["x"])
        np.testing.assert_allclose(got["Spending"], want["Spending"], rtol=1e-10)

    def test_keeping_first_transaction_matches_oracle(self, apparel_trans):
        from conftest import fixture_csv

        clv = ClvData(apparel_trans, time_unit="week", estimation_split=None)
        got = (
            clv.spending_summary(remove_first_transaction=False)
            .set_index("Id").sort_index()
        )
        want = fixture_csv("cbs_spending_full_with_first").set_index("Id").sort_index()
        np.testing.assert_array_equal(got["x"], want["x"])
        np.testing.assert_allclose(got["Spending"], want["Spending"], rtol=1e-10)

    def test_single_purchase_customers_drop_out_by_default(self):
        """S6.2.3: "customers with a single purchase are ignored"."""
        log = pd.DataFrame({
            "Id": ["once", "twice", "twice"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"]),
            "Price": [50.0, 10.0, 30.0],
        })
        clv = ClvData(log, time_unit="week", estimation_split=None)
        summary = clv.spending_summary().set_index("Id")
        assert summary.loc["once", "x"] == 0
        assert summary.loc["twice", "x"] == 1
        assert summary.loc["twice", "Spending"] == pytest.approx(30.0)

    def test_averages_over_transactions_not_days(self):
        """Eq. (13): z_bar = sum(z_i) / x, over the day-collapsed log."""
        log = pd.DataFrame({
            "Id": ["a"] * 4,
            "Date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01"]
            ),
            "Price": [5.0, 20.0, 10.0, 60.0],
        })
        clv = ClvData(log, time_unit="week", estimation_split=None)
        row = clv.spending_summary().iloc[0]
        # The two February purchases collapse to one worth 30.
        assert row["x"] == 2
        assert row["Spending"] == pytest.approx((30.0 + 60.0) / 2)

    def test_requires_a_price_column(self, apparel_trans):
        clv = ClvData(apparel_trans[["Id", "Date"]], time_unit="week")
        with pytest.raises(ValueError, match="no Price column"):
            clv.spending_summary()


class TestInputValidation:
    def test_rejects_unknown_time_unit(self, apparel_trans):
        with pytest.raises(ValueError, match="time_unit must be one of"):
            ClvData(apparel_trans, time_unit="fortnight")

    def test_rejects_missing_columns(self):
        with pytest.raises(ValueError, match="missing columns"):
            ClvData(pd.DataFrame({"customer": ["a"]}), name_id="Id")

    def test_rejects_data_end_before_last_transaction(self, apparel_trans):
        with pytest.raises(ValueError, match="precedes the last purchase"):
            ClvData(apparel_trans, time_unit="week", data_end="2009-01-01")

    def test_rejects_split_beyond_the_data(self, apparel_trans):
        with pytest.raises(ValueError, match="after the data ends"):
            ClvData(apparel_trans, time_unit="week", estimation_split=500)

    def test_rejects_empty_estimation_period(self, apparel_trans):
        with pytest.raises(ValueError, match="longer than zero"):
            ClvData(apparel_trans, time_unit="week", estimation_split=0)


class TestSplitSpecification:
    """S6.1: "Alternatively, a date can be provided"."""

    def test_date_split_equals_duration_split(self, apparel_trans):
        by_count = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        by_date = ClvData(apparel_trans, time_unit="week", estimation_split=ESTIMATION_END)
        assert by_count.estimation_end == by_date.estimation_end
        pd.testing.assert_frame_equal(
            by_count.customer_summary(), by_date.customer_summary()
        )

    def test_data_end_extends_the_window_past_the_last_record(self, apparel_trans):
        """S6.1: a fictional end of data beyond the last purchase record."""
        clv = ClvData(apparel_trans, time_unit="week", data_end="2010-12-31")
        assert clv.data_end == pd.Timestamp("2010-12-31")
        # 11 days further out than the 2010-12-20 last record.
        extra = clv.customer_summary()["T"].iloc[0] - (2178 / 7)
        assert extra == pytest.approx(11 / 7)

    def test_repr_is_informative(self, apparel_trans):
        clv = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        text = repr(clv)
        assert "600 customers" in text
        assert "2006-12-31" in text


class TestOtherDatasets:
    def test_cdnow_loads(self):
        from clvtools.data import load_cdnow

        cdnow = load_cdnow()
        assert len(cdnow) == 6696
        assert {"Id", "Date"} <= set(cdnow.columns)
