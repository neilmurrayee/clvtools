r"""S6.2.2, S6.2.4 and S6.3.3 - diagnostics and bootstrap intervals.

The plot fixtures come from CLVTools' own ``plot(..., plot = FALSE)``, which is
the same escape hatch S6.3.3's bootstrap example uses to get at the numbers. So
the tracking and PMF frames are held to the reference row for row.

The spending density has no ``plot = FALSE``; its curve is
``clv.model.probability.density``, taken from the generic directly.

The bootstrap has no fixture at all -- it is random, and the paper's own example
seeds R's generator, which has no Python equivalent. It is checked against the
properties S6.3.3 states instead: the periods are preserved, a customer drawn
twice counts twice, and the intervals bracket the point estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv, fixture_json

from clvtools import ClvData, diagnostics, load_apparel_trans
from clvtools.gg import GgParams, fit_gg
from clvtools.pnbd import expectation, pmf

#: CLVTools' own fitted values, so the comparison is not blurred by rounding.
FITTED = fixture_json("pnbd_nocov_fit")["coefficients"]
MODEL_NAME = "Pareto/NBD Standard"


# -- tracking, S6.2.2 ---------------------------------------------------------


@pytest.mark.oracle
class TestTrackingData:
    @pytest.mark.parametrize(
        "cumulative,fixture",
        [(False, "plot_tracking_incremental"), (True, "plot_tracking_cumulative")],
    )
    def test_matches_the_oracle(self, data, cumulative, fixture):
        got = diagnostics.tracking_data(
            data, lambda t: expectation(t, **FITTED),
            cumulative=cumulative, model_name=MODEL_NAME,
        )
        want = fixture_csv(fixture, parse_dates=["period.until"])

        merged = got.merge(
            want, on=["period.until", "variable"],
            suffixes=("_got", "_want"), how="outer", indicator=True,
        )
        assert (merged["_merge"] == "both").all(), "period grids differ"
        # The final period is only partly covered, so its observed count is
        # missing on both sides; see the test below.
        np.testing.assert_array_equal(
            merged["value_got"].isna(), merged["value_want"].isna()
        )
        present = merged["value_want"].notna()
        np.testing.assert_allclose(
            merged.loc[present, "value_got"], merged.loc[present, "value_want"],
            rtol=1e-9, atol=1e-9,
        )

    def test_a_partly_covered_period_reports_no_observed_count(self, data):
        r"""The last period runs past the data, so counting it would understate.

        The apparel data ends 2010-12-20 and the final period runs to
        2010-12-26. Six days of it simply have not happened, so CLVTools leaves
        the observed value missing rather than reporting the four days it does
        have, and this does the same. The model series is unaffected -- an
        expectation is defined for any horizon.
        """
        got = diagnostics.tracking_data(
            data, lambda t: expectation(t, **FITTED), model_name=MODEL_NAME
        )
        last = got[got["period.until"] == pd.Timestamp("2010-12-26")]
        actual = last[last["variable"] == diagnostics.ACTUAL]["value"]
        model = last[last["variable"] == MODEL_NAME]["value"]
        assert actual.isna().all()
        assert np.isfinite(model).all()

    def test_the_grid_runs_one_period_past_the_last_transaction(self, data):
        r"""So the final, partly observed period is shown whole.

        The apparel data ends 2010-12-20; the grid carries to 2010-12-26.
        """
        got = diagnostics.tracking_data(data, lambda t: expectation(t, **FITTED))
        periods = got["period.until"].drop_duplicates().sort_values()
        assert periods.iloc[0] == pd.Timestamp("2005-01-02")
        assert periods.iloc[-1] == pd.Timestamp("2010-12-26")
        assert periods.iloc[-1] > data.data_end


class TestTrackingProperties:
    def test_the_model_series_opens_at_zero(self, data):
        r"""S6.2.2: "The expected number of repeat transactions on this date by
        definition is zero and this fact gives the plot its characteristic
        shape"."""
        got = diagnostics.tracking_data(
            data, lambda t: expectation(t, **FITTED), model_name=MODEL_NAME
        )
        model = got[got["variable"] == MODEL_NAME].reset_index(drop=True)
        assert model.loc[0, "value"] == 0.0

    def test_the_actual_series_counts_repeat_transactions_only(self, data):
        """A first purchase is not a repeat purchase.

        The series covers every repeat transaction except the one falling in
        the final, partly covered period, whose count is deliberately missing.
        """
        got = diagnostics.tracking_data(data, lambda t: expectation(t, **FITTED))
        actual = got[got["variable"] == diagnostics.ACTUAL]

        transactions = data.transactions
        first = transactions.groupby("Id")["Date"].transform("min")
        repeats = transactions.loc[transactions["Date"] > first, "Date"]
        in_final = int((repeats > pd.Timestamp("2010-12-19")).sum())

        assert in_final == 1
        assert actual["value"].sum() + in_final == float(len(repeats))
        assert actual["value"].isna().sum() == 1

    def test_the_cumulative_series_is_the_running_total(self, data):
        curve = lambda t: expectation(t, **FITTED)  # noqa: E731
        incremental = diagnostics.tracking_data(data, curve, model_name=MODEL_NAME)
        cumulative = diagnostics.tracking_data(
            data, curve, cumulative=True, model_name=MODEL_NAME
        )
        for series in (diagnostics.ACTUAL, MODEL_NAME):
            step = incremental.loc[incremental["variable"] == series, "value"]
            total = cumulative.loc[cumulative["variable"] == series, "value"]
            np.testing.assert_allclose(
                np.cumsum(step.to_numpy()), total.to_numpy(),
                rtol=1e-9, atol=1e-9, equal_nan=True,
            )

    def test_the_model_line_slopes_gently_down(self, data):
        r"""S6.2.2: "The slightly downward sloping line shows how the model
        expects fewer purchases over time as more customers stop doing business
        with the firm"."""
        got = diagnostics.tracking_data(
            data, lambda t: expectation(t, **FITTED), model_name=MODEL_NAME
        )
        model = got.loc[got["variable"] == MODEL_NAME, "value"].to_numpy()
        # Skipping the opening zero, every step is smaller than the last.
        assert np.all(np.diff(model[1:]) < 0)

    def test_a_horizon_can_be_given_as_periods_or_a_date(self, data):
        curve = lambda t: expectation(t, **FITTED)  # noqa: E731
        by_count = diagnostics.tracking_data(data, curve, prediction_end=52)
        by_date = diagnostics.tracking_data(
            data, curve, prediction_end="2007-12-30"
        )
        assert len(by_count) == len(by_date)
        assert by_count["period.until"].max() == by_date["period.until"].max()

    def test_it_works_with_another_family(self, data):
        """Any ``t -> E[X(t)]`` will do; nothing here is Pareto/NBD specific."""
        from clvtools.bgnbd import expectation as bgnbd_expectation

        got = diagnostics.tracking_data(
            data,
            lambda t: bgnbd_expectation(t, 0.6073, 20.9567, 1.2755, 8.8608),
            model_name="BG/NBD",
        )
        assert sorted(got["variable"].unique()) == ["Actual", "BG/NBD"]
        assert np.isfinite(got.loc[got["variable"] == "BG/NBD", "value"]).all()


# -- PMF, S6.2.2 --------------------------------------------------------------


@pytest.mark.oracle
class TestPmfData:
    def test_matches_the_oracle(self, data):
        got = diagnostics.pmf_data(
            data, lambda k, T: pmf(k, T, **FITTED), model_name=MODEL_NAME
        )
        want = fixture_csv("plot_pmf")
        want["num.transactions"] = want["num.transactions"].astype(str)

        merged = got.merge(
            want, on=["num.transactions", "variable"],
            suffixes=("_got", "_want"), how="outer", indicator=True,
        )
        assert (merged["_merge"] == "both").all()
        np.testing.assert_allclose(
            merged["value_got"], merged["value_want"], rtol=1e-9, atol=1e-9
        )

    def test_the_last_bin_is_a_tail(self, data):
        got = diagnostics.pmf_data(
            data, lambda k, T: pmf(k, T, **FITTED), model_name=MODEL_NAME
        )
        assert got["num.transactions"].iloc[-1] == "10+"


@pytest.mark.oracle
class TestFittedData:
    """``fitted()``, against R's own -- finding B3 of the 2026-09 spec audit.

    ``fitted_data`` had one doctest and no test file, and the doctest's printed
    values came from this implementation rather than from R, so it could not
    fail if the function were wrong -- only if it changed. ``fitted_pnbd.csv``
    was generated by ``tools/oracle/generate_interface_fixtures.R`` at the same
    time and never read by anything; it is R's ``fitted(est.pnbd)`` in full,
    313 periods of it.
    """

    def test_matches_the_oracle(self, data):
        got = diagnostics.fitted_data(data, lambda t: expectation(t, **FITTED))
        want = fixture_csv("fitted_pnbd", parse_dates=["period.until"])

        assert len(got) == len(want) == 313
        np.testing.assert_array_equal(
            got["period.until"].to_numpy(), want["period.until"].to_numpy()
        )
        np.testing.assert_array_equal(got["period.num"], want["period.num"])
        np.testing.assert_allclose(
            got["expectation"].to_numpy(dtype=float),
            want["expectation"].to_numpy(dtype=float),
            rtol=1e-10, atol=1e-10,
        )

    def test_it_is_the_model_half_of_the_tracking_plot(self, data):
        """The claim the docstring makes, which is what makes one oracle do
        for both."""
        curve = lambda t: expectation(t, **FITTED)  # noqa: E731
        tracking = diagnostics.tracking_data(data, curve, model_name=MODEL_NAME)
        model = tracking.loc[tracking["variable"] == MODEL_NAME]
        got = diagnostics.fitted_data(data, curve)

        np.testing.assert_array_equal(
            got["expectation"].to_numpy(), model["value"].to_numpy()
        )
        np.testing.assert_array_equal(got["period.num"], np.arange(1, 314))


class TestPmfProperties:
    def test_both_series_account_for_every_customer(self, data):
        got = diagnostics.pmf_data(data, lambda k, T: pmf(k, T, **FITTED))
        for _, group in got.groupby("variable"):
            assert group["value"].sum() == pytest.approx(600.0)

    def test_the_observed_counts_are_the_histogram(self, data):
        got = diagnostics.pmf_data(data, lambda k, T: pmf(k, T, **FITTED))
        observed = got[got["variable"] == diagnostics.ACTUAL].set_index(
            "num.transactions"
        )
        counts = data.customer_summary()["x"]
        for k in range(10):
            assert observed.loc[str(k), "value"] == float((counts == k).sum())

    def test_the_model_tracks_the_histogram_closely(self, data):
        r"""S6.2.2: "the results illustrate that the model fits the data well"."""
        got = diagnostics.pmf_data(
            data, lambda k, T: pmf(k, T, **FITTED), model_name=MODEL_NAME
        )
        wide = got.pivot(
            index="num.transactions", columns="variable", values="value"
        )
        assert np.abs(wide[diagnostics.ACTUAL] - wide[MODEL_NAME]).max() < 30

    def test_the_number_of_bins_is_configurable(self, data):
        got = diagnostics.pmf_data(
            data, lambda k, T: pmf(k, T, **FITTED), max_transactions=4
        )
        labels = got["num.transactions"].unique().tolist()
        assert labels == ["0", "1", "2", "3", "4+"]

    def test_rejects_a_meaningless_bin_count(self, data):
        with pytest.raises(ValueError, match="at least 1"):
            diagnostics.pmf_data(data, lambda k, T: pmf(k, T, **FITTED), 0)


# -- spending density, S6.2.4 -------------------------------------------------


@pytest.mark.oracle
class TestSpendingDensity:
    def test_the_model_curve_matches_the_oracle(self, data):
        want = fixture_csv("plot_spending_density")
        # The oracle's own estimates, so the comparison is not blurred by the
        # last digits of a refit.
        published = fixture_json("gg_fit")["coefficients"]
        fitted = GgParams(
            **published, log_likelihood=float("nan"),
            converged=True, n_customers=600,
        )
        got = diagnostics.spending_density_data(
            data, fitted, grid=want["spending"].to_numpy(dtype=float)
        )
        model = got[got["variable"] == "Gamma-Gamma"]
        np.testing.assert_allclose(
            model["value"], want["density"], rtol=1e-6, atol=1e-12
        )


class TestSpendingDensityProperties:
    @staticmethod
    @pytest.fixture(scope="class")
    def fitted(data):
        spend = data.spending_summary()
        return fit_gg(spend["x"], spend["Spending"])

    def test_the_model_curve_is_a_density(self, data, fitted):
        grid = np.linspace(0.5, 1500, 600)
        got = diagnostics.spending_density_data(data, fitted, grid=grid)
        model = got[got["variable"] == "Gamma-Gamma"]
        mass = np.trapezoid(model["value"], model["spending"])
        assert mass == pytest.approx(1.0, abs=0.02)

    def test_both_series_share_the_grid(self, data, fitted):
        grid = np.linspace(1, 400, 64)
        got = diagnostics.spending_density_data(data, fitted, grid=grid)
        for _, group in got.groupby("variable"):
            np.testing.assert_allclose(group["spending"], grid)

    def test_the_default_grid_spans_the_observed_range(self, data, fitted):
        got = diagnostics.spending_density_data(data, fitted)
        spend = data.spending_summary()
        active = spend[spend["x"] > 0]["Spending"]
        assert got["spending"].min() == pytest.approx(active.min())
        assert got["spending"].max() == pytest.approx(active.max())

    def test_the_two_curves_broadly_agree(self, data, fitted):
        r"""S6.2.4: "the plot shows that the spending model fits the data in the
        estimation period reasonably well"."""
        grid = np.linspace(1, 400, 256)
        got = diagnostics.spending_density_data(data, fitted, grid=grid)
        wide = got.pivot(index="spending", columns="variable", values="value")
        # Both integrate to about 1, so a bounded gap is a real statement.
        assert np.abs(wide["Actual"] - wide["Gamma-Gamma"]).max() < 0.01

    def test_rejects_data_with_no_spending(self):
        transactions = load_apparel_trans()[["Id", "Date"]]
        empty = ClvData(transactions, time_unit="week", estimation_split=104)
        with pytest.raises(ValueError, match="no Price column"):
            diagnostics.spending_density_data(empty, None)


class TestRendering:
    r"""``render`` is a convenience; matplotlib is an optional extra.

    It is in the dev dependencies so these run, but nothing in ``src/`` imports
    it and the frames above are useful without it.
    """

    @pytest.fixture(autouse=True)
    def _headless(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

    @pytest.mark.parametrize(
        "frame_name", ["tracking", "pmf", "spending"]
    )
    def test_it_draws_every_frame_shape(self, data, frame_name):
        """Each diagnostic keys its x-axis on a different column."""
        if frame_name == "tracking":
            frame = diagnostics.tracking_data(
                data, lambda t: expectation(t, **FITTED)
            )
            expected_axis = "period.until"
        elif frame_name == "pmf":
            frame = diagnostics.pmf_data(data, lambda k, T: pmf(k, T, **FITTED))
            expected_axis = "num.transactions"
        else:
            spend = data.spending_summary()
            frame = diagnostics.spending_density_data(
                data, fit_gg(spend["x"], spend["Spending"]),
                grid=np.linspace(1, 400, 32),
            )
            expected_axis = "spending"

        ax = diagnostics.render(frame, title=frame_name)
        assert ax.get_title() == frame_name
        assert ax.get_xlabel() == expected_axis
        assert len(ax.get_lines()) == frame["variable"].nunique()

    def test_it_draws_onto_a_given_axis(self, data):
        import matplotlib.pyplot as plt

        frame = diagnostics.pmf_data(data, lambda k, T: pmf(k, T, **FITTED))
        _, ax = plt.subplots()
        assert diagnostics.render(frame, ax=ax) is ax

    def test_the_error_names_the_extra_when_matplotlib_is_absent(self, data):
        """The message a user without the extra would see."""
        import builtins

        frame = diagnostics.pmf_data(data, lambda k, T: pmf(k, T, **FITTED))
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("no matplotlib")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked
        try:
            with pytest.raises(ImportError, match="needs matplotlib"):
                diagnostics.render(frame)
        finally:
            builtins.__import__ = real_import


class TestSpendingDensityGuards:
    def test_rejects_data_where_nobody_has_both(self):
        r"""After dropping first transactions, a one-purchase cohort has none."""
        transactions = load_apparel_trans()
        once = transactions.groupby("Id", as_index=False).first()
        only_first = ClvData(once, time_unit="week")
        spend = only_first.spending_summary()
        assert (spend["x"] == 0).all()
        with pytest.raises(ValueError, match="no customer has both"):
            diagnostics.spending_density_data(only_first, None)




class TestThePerCustomerPmfTable:
    """Spec `PMF-05`, `absent`: the generic this package did not have.

    CLVTools' ``pmf()`` on a fitted object gives one row per customer and one
    ``pmf.x.<k>`` column per requested count, defaulting to ``0:5``. What
    existed here was :func:`~clvtools.diagnostics.pmf_data`, which aggregates
    customers into bins for S6.2.2's plot -- a different question, and one that
    cannot answer "what is *this* customer's probability of buying twice".
    """

    @staticmethod
    def _pmf(k, T):
        from clvtools.pnbd import pmf

        return pmf(k, T, 1.4490, 48.6361, 0.5613, 46.8844)

    @pytest.fixture(scope="class")
    def data(self, apparel_trans):
        from clvtools import ClvData

        return ClvData(apparel_trans, time_unit="week", estimation_split=104)

    def test_the_default_is_zero_through_five(self, data):
        from clvtools.diagnostics import pmf_table

        frame = pmf_table(data, self._pmf)
        assert list(frame.columns) == ["Id"] + [f"pmf.x.{k}" for k in range(6)]
        assert len(frame) == 600

    def test_the_ids_are_the_data_s_own(self, data):
        from clvtools.diagnostics import pmf_table

        got = pmf_table(data, self._pmf)["Id"]
        np.testing.assert_array_equal(got, data.customer_summary()["Id"])

    @pytest.mark.parametrize("x", [0, 0.0, [0], np.int64(0)])
    def test_a_single_count_is_accepted_bare_and_boxed(self, data, x):
        """R takes an integer, a numeric, or a length-one vector alike."""
        from clvtools.diagnostics import pmf_table

        assert list(pmf_table(data, self._pmf, x=x).columns) == ["Id", "pmf.x.0"]

    def test_the_columns_agree_with_the_family_s_own_pmf(self, data):
        """Selection, not recomputation -- the table adds no arithmetic."""
        from clvtools.diagnostics import pmf_table

        frame = pmf_table(data, self._pmf, x=[0, 3])
        T = data.customer_summary()["T"].to_numpy(dtype=float)
        np.testing.assert_allclose(frame["pmf.x.3"], self._pmf(3, T), rtol=0)

    def test_each_row_is_the_start_of_a_distribution(self, data):
        """Six terms of a PMF, so every row sums to at most one."""
        from clvtools.diagnostics import pmf_table

        totals = pmf_table(data, self._pmf).set_index("Id").sum(axis=1)
        assert (totals <= 1.0).all()
        assert (totals > 0.0).all()

    def test_the_same_count_twice_is_refused(self, data):
        """``2`` and ``2.0`` name one column, so asking for both is a mistake."""
        from clvtools.diagnostics import pmf_table

        with pytest.raises(ValueError, match="asked for twice"):
            pmf_table(data, self._pmf, x=[2, 2.0])

    @pytest.mark.parametrize("bad", [1.5, -1])
    def test_a_count_that_is_not_a_whole_non_negative_number_is_refused(
        self, data, bad
    ):
        from clvtools.diagnostics import pmf_table

        with pytest.raises(ValueError, match="whole and non-negative"):
            pmf_table(data, self._pmf, x=[bad])

    def test_no_counts_at_all_is_refused(self, data):
        from clvtools.diagnostics import pmf_table

        with pytest.raises(ValueError, match="at least one count"):
            pmf_table(data, self._pmf, x=[])


class TestTrackingPastTheLastTransaction:
    """Spec `S-12`, `absent !`: a divergence, recorded rather than matched.

    CLVTools' tracking plot reports ``NA`` for every period between the last
    transaction and ``data.end``, with no warning. This package reports
    ``0.0``. Both are defensible and they answer different questions: R's
    ``NA`` says "the log ends here and I cannot tell you"; the ``0.0`` here
    says "you told me the observation window runs to ``data_end``, and in these
    weeks nobody bought". Since ``data_end`` is an argument the caller supplies
    -- it is not inferred from the log -- the second reading is the one that
    matches what was asked for, and an ``NA`` would discard a real observation.

    Pinned rather than silently inherited, because a plot moving from R gets a
    line at zero where it used to get a gap.
    """

    @pytest.fixture(scope="class")
    def extended(self, apparel_trans):
        from clvtools import ClvData

        few = apparel_trans[
            apparel_trans["Id"].isin(apparel_trans["Id"].unique()[:20])
        ]
        return ClvData(
            few, time_unit="week", estimation_split=104,
            data_end=pd.Timestamp(few["Date"].max()) + pd.Timedelta(weeks=8),
        )

    @pytest.fixture(scope="class")
    def tracked(self, extended):
        from clvtools.diagnostics import tracking_data
        from clvtools.pnbd import expectation, fit_pnbd

        cbs = extended.customer_summary()
        fit = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
        return tracking_data(
            extended,
            lambda periods: expectation(
                periods, fit.r, fit.alpha, fit.s, fit.beta
            ),
        )

    def test_the_trailing_periods_are_zero_and_not_missing(self, tracked):
        actual = tracked[tracked["variable"] == "Actual"]["value"]
        assert actual.notna().all()
        assert (actual.tail(6) == 0.0).all()

    def test_and_the_expected_series_keeps_rising_through_them(self, tracked):
        """Which is why the zero is informative: the two series diverge.

        An ``NA`` would hide the eight weeks in which the model predicted
        transactions and none happened -- the part of a tracking plot a reader
        most wants to see.
        """
        expected = tracked[tracked["variable"] != "Actual"]["value"]
        assert expected.notna().all()
        assert (expected.tail(6) > 0).all()

    def test_the_two_series_still_cover_the_same_periods(self, tracked):
        counts = set(tracked.groupby("variable").size())
        assert len(counts) == 1
