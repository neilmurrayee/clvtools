r"""S6.1.2 - inspecting the transaction data.

The descriptive layer: ``summary()``, ``as.data.frame()`` and the five
descriptive plots of Table 3. None of it involves a model, so every expectation
here is either a number the paper prints or a frame CLVTools produces through
its public ``plot(..., plot = FALSE)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clvtools import ClvData
from clvtools.diagnostics import (
    frequency_data,
    interpurchase_time_data,
    spending_data,
    timings_data,
    tracking_data,
)
from conftest import fixture_csv, fixture_json
from paper_values import (
    DESCRIPTIVES,
    N_CUSTOMERS,
    N_TRANSACTIONS,
    N_TRANSACTIONS_AGGREGATED,
    ZERO_REPEATERS,
)


@pytest.fixture(scope="module")
def data(apparel_trans) -> ClvData:
    return ClvData(apparel_trans, time_unit="week", estimation_split=104)


def _oracle_value(text: str):
    """One cell of the oracle's summary table: a date, a number, or nothing."""
    text = text.strip()
    if text in ("NA", "", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return pd.Timestamp(text)


class TestSummary:
    """The descriptive table of S6.1.2."""

    @pytest.mark.paper
    def test_matches_every_printed_cell(self, data):
        table = data.summary()
        for name, printed in DESCRIPTIVES.items():
            for column, want in zip(("Estimation", "Holdout", "Total"), printed):
                got = table.loc[name, column]
                if want is None:
                    assert got is None, f"{name}/{column} should not apply"
                elif isinstance(want, str):
                    assert got == pd.Timestamp(want), f"{name}/{column}"
                else:
                    assert got == pytest.approx(want, abs=5e-4), f"{name}/{column}"

    @pytest.mark.oracle
    def test_matches_the_oracle_at_full_precision(self, data):
        want = fixture_csv("descriptives_summary")
        table = data.summary()
        assert len(want) == len(table)
        for name, row in zip(table.index, want.itertuples()):
            for column, cell in zip(
                ("Estimation", "Holdout", "Total"),
                (row.Estimation, row.Holdout, row.Total),
            ):
                expected = _oracle_value(str(cell))
                got = table.loc[name, column]
                if expected is None:
                    assert got is None, f"{name}/{column}"
                elif isinstance(expected, pd.Timestamp):
                    assert got == expected, f"{name}/{column}"
                else:
                    assert got == pytest.approx(expected, rel=1e-12), f"{name}/{column}"

    def test_row_order_follows_the_paper(self, data):
        assert list(data.summary().index) == list(DESCRIPTIVES)

    def test_zero_repeaters_are_the_frequency_plots_first_bin(self, data):
        table = data.summary()
        assert table.loc["Total # zero repeaters", "Estimation"] == ZERO_REPEATERS
        assert (
            frequency_data(data)["num.customers"].iloc[0] == ZERO_REPEATERS
        )

    def test_same_day_purchases_are_one_transaction(self, data, apparel_trans):
        """S6.1: 3,187 records become 3,183 transactions."""
        assert len(apparel_trans) == N_TRANSACTIONS
        assert (
            data.summary().loc["Total # Transactions", "Total"]
            == N_TRANSACTIONS_AGGREGATED
        )

    def test_without_a_holdout_there_is_no_holdout_column(self, apparel_trans):
        full = ClvData(apparel_trans, time_unit="week")
        assert list(full.summary().columns) == ["Estimation", "Total"]

    def test_without_spending_the_spending_rows_are_absent(self, apparel_trans):
        bare = ClvData(apparel_trans[["Id", "Date"]], time_unit="week",
                       estimation_split=104)
        index = bare.summary().index
        assert "Total Spending" not in index
        assert "Mean Interpurchase time" in index

    def test_the_three_sd_rows_are_distinguishable(self, data):
        """CLVTools names them all "(SD)", padded with spaces to differ."""
        index = list(data.summary().index)
        assert len(set(index)) == len(index)


class TestAsDataFrame:
    """``as.data.frame()`` and the samples of S6.1.2."""

    @pytest.mark.oracle
    def test_sample_row_counts_match(self, data):
        want = fixture_json("data_samples")
        assert len(data.as_data_frame()) == want["n.transactions.total"]
        assert (
            len(data.as_data_frame(sample="estimation"))
            == want["n.transactions.estimation"]
        )
        assert (
            len(data.as_data_frame(sample="holdout")) == want["n.transactions.holdout"]
        )
        assert data.nobs() == want["nobs"]

    def test_full_is_the_default(self, data):
        """Unlike the plots, which default to the estimation period."""
        assert data.as_data_frame().equals(data.as_data_frame(sample="full"))

    def test_the_samples_partition_the_data(self, data):
        parts = len(data.as_data_frame("estimation")) + len(
            data.as_data_frame("holdout")
        )
        assert parts == len(data.as_data_frame("full"))

    @pytest.mark.oracle
    def test_ids_select_one_customer(self, data):
        one = data.as_data_frame(ids="1")
        assert set(one["Id"]) == {"1"}
        assert len(one) == fixture_json("data_samples")["n.ids.1"]

    @pytest.mark.oracle
    def test_ids_accept_a_sequence(self, data):
        """S6.1.2's ``subset(clv.apparel, Id=="7"|Id=="9", sample = "holdout")``."""
        two = data.as_data_frame(sample="holdout", ids=["7", "9"])
        assert set(two["Id"]) == {"7", "9"}
        assert len(two) == fixture_json("data_samples")["n.ids.7.9.holdout"]

    def test_rejects_an_unknown_sample(self, data):
        with pytest.raises(ValueError, match="sample must be one of"):
            data.as_data_frame(sample="calibration")

    def test_rejects_a_holdout_that_does_not_exist(self, apparel_trans):
        full = ClvData(apparel_trans, time_unit="week")
        with pytest.raises(ValueError, match="no holdout period"):
            full.as_data_frame(sample="holdout")

    @pytest.mark.oracle
    def test_pandas_query_stands_in_for_subset(self, data):
        """S6.1.2's ``subset(clv.apparel, Price >= 50 & Price <= 100)``.

        Its default sample is the full data, like ``as.data.frame()`` and
        unlike the descriptive plots.
        """
        want = fixture_json("data_samples")
        expression = "Price >= 50 & Price <= 100"
        assert len(data.as_data_frame().query(expression)) == want["n.price.50.to.100"]
        assert (
            len(data.as_data_frame("estimation").query(expression))
            == want["n.price.50.to.100.estimation"]
        )


class TestDescriptivePlots:
    """Table 3's five frames, against CLVTools' own ``plot(plot = FALSE)``."""

    @pytest.mark.oracle
    @pytest.mark.parametrize("cumulative,name", [
        (False, "plot_data_tracking"),
        (True, "plot_data_tracking_cumulative"),
    ])
    def test_tracking_without_a_model(self, data, cumulative, name):
        want = fixture_csv(name)
        got = tracking_data(data, cumulative=cumulative)
        assert list(got["variable"].unique()) == ["Number of Repeat Transactions"]
        assert len(got) == len(want)
        np.testing.assert_array_equal(
            got["period.until"].dt.date.astype(str).to_numpy(),
            want["period.until"].to_numpy(),
        )
        np.testing.assert_allclose(
            got["value"].to_numpy(), want["value"].to_numpy(), rtol=1e-12
        )

    @pytest.mark.oracle
    @pytest.mark.parametrize("kwargs,name", [
        ({}, "plot_data_frequency"),
        ({"count_remaining": False}, "plot_data_frequency_no_remaining"),
        (
            {"bins": range(5), "label_remaining": "5+"},
            "plot_data_frequency_five_bins",
        ),
    ])
    def test_frequency(self, data, kwargs, name):
        want = fixture_csv(name)
        got = frequency_data(data, **kwargs)
        np.testing.assert_array_equal(
            got["num.transactions"].to_numpy(),
            want["num.transactions"].astype(str).to_numpy(),
        )
        np.testing.assert_array_equal(
            got["num.customers"].to_numpy(), want["num.customers"].to_numpy()
        )

    def test_frequency_bins_must_start_at_one_when_counting_all(self, data):
        with pytest.raises(ValueError, match="strictly positive"):
            frequency_data(data, count_repeat_transactions=False)

    def test_frequency_counting_all_transactions_shifts_by_one(self, data):
        repeats = frequency_data(data)
        every = frequency_data(
            data, bins=range(1, 11), count_repeat_transactions=False,
            label_remaining="11+",
        )
        np.testing.assert_array_equal(
            repeats["num.customers"].to_numpy(), every["num.customers"].to_numpy()
        )

    @pytest.mark.oracle
    @pytest.mark.parametrize("sample,name", [
        ("estimation", "plot_data_interpurchasetime"),
        ("holdout", "plot_data_interpurchasetime_holdout"),
    ])
    def test_interpurchase_time(self, data, sample, name):
        want = fixture_csv(name).sort_values("Id").reset_index(drop=True)
        got = interpurchase_time_data(data, sample=sample).sort_values(
            "Id"
        ).reset_index(drop=True)
        np.testing.assert_array_equal(got["Id"].to_numpy(), want["Id"].to_numpy())
        np.testing.assert_allclose(
            got["mean.interpurchase.time"].to_numpy(),
            want["mean.interpurchase.time"].to_numpy(),
            rtol=1e-12,
        )

    def test_interpurchase_time_drops_zero_repeaters(self, data):
        kept = interpurchase_time_data(data)
        assert len(kept) == N_CUSTOMERS - ZERO_REPEATERS
        assert kept["mean.interpurchase.time"].notna().all()

    @pytest.mark.oracle
    @pytest.mark.parametrize("kwargs,name", [
        ({}, "plot_data_spending_mean"),
        ({"sample": "holdout"}, "plot_data_spending_mean_holdout"),
        ({"mean_spending": False}, "plot_data_spending_transactions"),
    ])
    def test_spending(self, data, kwargs, name):
        want = fixture_csv(name)
        got = spending_data(data, **kwargs)
        assert len(got) == len(want)
        np.testing.assert_allclose(
            np.sort(got["Spending"].to_numpy()),
            np.sort(want["Spending"].to_numpy()),
            rtol=1e-12,
        )

    def test_spending_needs_prices(self, apparel_trans):
        bare = ClvData(apparel_trans[["Id", "Date"]], time_unit="week",
                       estimation_split=104)
        with pytest.raises(ValueError, match="no Price column"):
            spending_data(bare)

    @pytest.mark.oracle
    def test_timings(self, data):
        want = fixture_csv("plot_data_timings").astype(str)
        got = timings_data(data, ids=["1", "2", "3"])
        assert len(got) == len(want)
        for column in ("Id", "type", "variable", "value"):
            np.testing.assert_array_equal(
                got[column].to_numpy().astype(str), want[column].to_numpy()
            )

    def test_timings_lays_customers_ten_apart(self, data):
        frame = timings_data(data, ids=["1", "2", "3"])
        rows = frame.loc[
            (frame["type"] == "segment_start") & (frame["variable"] == "y")
        ]
        assert list(rows["value"]) == ["10", "20", "30"]

    def test_timings_samples_when_given_no_ids(self, data):
        frame = timings_data(data, n=5, seed=1)
        assert frame.loc[frame["type"] == "segment_start", "Id"].nunique() == 5

    def test_timings_rejects_unknown_customers(self, data):
        with pytest.raises(ValueError, match="no such customers"):
            timings_data(data, ids=["nobody"])

    def test_timings_without_holdout_has_no_holdout_points(self, apparel_trans):
        full = ClvData(apparel_trans, time_unit="week")
        frame = timings_data(full, ids=["1"])
        assert "point_holdout" not in set(frame["type"])
