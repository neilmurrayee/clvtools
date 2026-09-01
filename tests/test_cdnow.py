r"""The CDNOW data, which the R package's documentation uses throughout.

The paper works on ``apparelTrans`` from beginning to end. CLVTools' own
documentation does not: ``?clvdata``, ``?pmf``, ``?subset.clv.data``,
``?as.data.frame.clv.data``, ``?spending`` and ``?clv.bootstrapped.apply`` all
reach for ``cdnow``. It is therefore a second, independent dataset for
machinery this suite otherwise exercises on one -- 2,357 customers rather than
600, no covariates, and a fifth as many transactions per customer.

``?pmf`` is the reason this module exists. It prints a Pareto/NBD PMF table for
``x = 0..10`` *and* the empirical frequencies beside it, which makes it the
only place in either the paper or the package documentation where the fitted
PMF is published at all.

Three tolerances are used, and the difference between them is the point:

* the empirical frequencies are **exact** -- they are counts, and no model or
  optimiser stands between the data and them;
* the fitted PMF is checked against the oracle fixture at 1e-4 and against the
  *printed* table at 1e-4 as well, because the man page's numbers were produced
  by an older fit: it prints 0.616514 at ``x = 0`` where CLVTools 0.12.1 now
  gives 0.616551 and this package gives 0.616533. All three sit within 4e-5 of
  each other, which is the Pareto/NBD ridge and not an error in any of them.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv, fixture_json
from rdoc_values import (
    CDNOW_ESTIMATION_WEEKS,
    CDNOW_FREQUENCIES,
    CDNOW_N_CUSTOMERS,
    CDNOW_N_TRANSACTIONS,
    CDNOW_PMF,
)

from clvtools import ClvData, load_cdnow
from clvtools.diagnostics import REPEAT_TRANSACTIONS
from clvtools.pnbd import fit_pnbd, pmf


@pytest.fixture(scope="module")
def cdnow() -> pd.DataFrame:
    return load_cdnow()


@pytest.fixture(scope="module")
def clv(cdnow) -> ClvData:
    """``clvdata(cdnow, time.unit="w", estimation.split=37)``, as ``?pmf``."""
    return ClvData(cdnow, time_unit="week", estimation_split=CDNOW_ESTIMATION_WEEKS)


@pytest.fixture(scope="module")
def cbs(clv) -> pd.DataFrame:
    return clv.customer_summary()


@pytest.fixture(scope="module")
def fitted(cbs):
    return fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=True)


class TestTheDataItself:
    def test_shape(self, cdnow):
        assert len(cdnow) == CDNOW_N_TRANSACTIONS
        assert cdnow["Id"].nunique() == CDNOW_N_CUSTOMERS

    def test_carries_spending_and_quantity(self, cdnow):
        """CDNOW has a ``CDs`` column the apparel data has no counterpart for."""
        assert {"Id", "Date", "CDs", "Price"} <= set(cdnow.columns)

    def test_ids_are_strings(self, cdnow):
        """As everywhere: reading `Id` as an integer reorders against the oracle."""
        assert cdnow["Id"].map(type).eq(str).all()


@pytest.mark.rdoc
class TestEmpiricalFrequencies:
    """The right-hand column of ``?pmf``'s table.

    Printed there as "actual percentage of x", over 2,357 customers. No model
    is involved, so these are exact -- and between them they check
    :func:`~clvtools.data.load_cdnow`, the day-level aggregation of S6.1 and
    :meth:`~clvtools.data.ClvData.customer_summary` in a single assertion.
    """

    def test_customer_count(self, cbs):
        assert len(cbs) == CDNOW_N_CUSTOMERS

    def test_repeat_transaction_counts_are_exact(self, cbs):
        counts = cbs["x"].astype(int).value_counts()
        for x, want in enumerate(CDNOW_FREQUENCIES):
            assert int(counts.get(x, 0)) == want, f"x = {x}"

    @pytest.mark.oracle
    def test_counts_match_the_oracles_own_cbs(self, cbs):
        want = fixture_json("cdnow_frequencies")
        assert len(cbs) == want["n.customers"]
        counts = cbs["x"].astype(int).value_counts()
        for x, expected in zip(want["x"], want["count"], strict=True):
            assert int(counts.get(x, 0)) == expected, f"x = {x}"

    def test_the_zero_repeaters_dominate(self, cbs):
        """1432 of 2357 -- 61% -- made no repeat purchase in 37 weeks."""
        share = CDNOW_FREQUENCIES[0] / CDNOW_N_CUSTOMERS
        assert share == pytest.approx(0.6075519, abs=1e-6)
        assert int((cbs["x"] == 0).sum()) == CDNOW_FREQUENCIES[0]


@pytest.mark.slow
class TestTheFit:
    """``pnbd(clvdata(cdnow, ...))``, against the oracle."""

    @pytest.mark.oracle
    def test_coefficients_match(self, fitted):
        want = fixture_json("cdnow_pnbd_fit")["coefficients"]
        for name, value in want.items():
            assert fitted.coefficients[name] == pytest.approx(value, rel=1e-3), name

    @pytest.mark.oracle
    def test_log_likelihood_matches(self, fitted):
        want = fixture_json("cdnow_pnbd_fit")
        assert fitted.log_likelihood == pytest.approx(want["logLik"], abs=1e-4)

    @pytest.mark.oracle
    def test_information_criteria_match(self, fitted):
        want = fixture_json("cdnow_pnbd_fit")
        assert fitted.n_customers == want["nobs"]
        assert fitted.aic == pytest.approx(want["AIC"], abs=1e-3)
        assert fitted.bic == pytest.approx(want["BIC"], abs=1e-3)

    @pytest.mark.oracle
    def test_standard_errors_match(self, fitted):
        want = fixture_json("cdnow_pnbd_fit")["standard.errors"]
        errors = fitted.standard_errors()
        for name, value in want.items():
            assert errors[name] == pytest.approx(value, rel=5e-3), name

    def test_the_estimates_are_the_canonical_cdnow_ones(self, fitted):
        """A second dataset with a very different shape from the apparel one:
        ``alpha`` near 10 rather than near 49, and ``r`` below 1."""
        assert fitted.r == pytest.approx(0.5455, abs=1e-3)
        assert fitted.alpha == pytest.approx(10.2819, abs=1e-2)


@pytest.mark.slow
@pytest.mark.rdoc
class TestProbabilityMassFunction:
    """``pmf(pnbd.cdnow, x=0:10)`` -- the table printed in ``?pmf``."""

    @staticmethod
    @pytest.fixture(scope="class")
    def means(cbs, fitted):
        T = cbs["T"].to_numpy(dtype=float)
        return [
            float(np.mean(pmf(x, T, fitted.r, fitted.alpha, fitted.s, fitted.beta)))
            for x in range(len(CDNOW_PMF))
        ]

    def test_matches_the_printed_table(self, means):
        for x, want in enumerate(CDNOW_PMF):
            assert means[x] == pytest.approx(want, abs=1e-4), f"x = {x}"

    @pytest.mark.oracle
    def test_matches_the_oracle_at_full_precision(self, means):
        want = fixture_json("cdnow_pmf_means")["mean.pmf"]
        for x, value in enumerate(want):
            assert means[x] == pytest.approx(value, abs=1e-4), f"x = {x}"

    @pytest.mark.oracle
    def test_per_customer_values_match_the_oracle(self, cbs, fitted):
        """The column means could agree while individual customers do not."""
        want = fixture_csv("cdnow_pmf").set_index("Id")
        T = cbs.set_index("Id").loc[want.index, "T"].to_numpy(dtype=float)
        for x in (0, 1, 5, 10):
            got = pmf(x, T, fitted.r, fitted.alpha, fitted.s, fitted.beta)
            np.testing.assert_allclose(
                got, want[f"pmf.x.{x}"].to_numpy(), atol=1e-4, err_msg=f"x = {x}"
            )

    def test_the_model_understates_the_repeat_buyers(self, means):
        """The comparison ``?pmf`` sets up by printing both columns.

        At ``x = 0`` the model puts more mass than the data shows (0.6165
        against 0.6076) and at ``x = 1`` and ``x = 2`` less. The man page
        prints the two columns side by side precisely so this is visible.
        """
        actual = [c / CDNOW_N_CUSTOMERS for c in CDNOW_FREQUENCIES]
        assert means[0] > actual[0]
        assert means[1] < actual[1]
        assert means[2] < actual[2]

    def test_the_mass_is_a_decreasing_distribution(self, means):
        assert all(a > b for a, b in pairwise(means))
        assert sum(means) < 1.0


class TestClvDataConstructions:
    """The four ``clvdata()`` calls of ``?clvdata``, all on ``cdnow``."""

    def test_no_split_leaves_no_holdout(self, cdnow):
        clv = ClvData(cdnow, time_unit="week")
        assert clv.holdout_start is None
        assert clv.estimation_end == clv.data_end

    def test_split_in_periods(self, clv):
        """37 weeks from the first transaction, 1997-01-01."""
        assert clv.estimation_end == pd.Timestamp("1997-09-17")
        assert clv.holdout_start == pd.Timestamp("1997-09-18")

    def test_split_as_a_date(self, cdnow):
        clv = ClvData(cdnow, time_unit="week", estimation_split="1997-10-15")
        assert clv.estimation_end == pd.Timestamp("1997-10-15")

    def test_extending_data_end_moves_only_the_holdout(self, cdnow):
        """``?clvdata``: "this only moves the holdout period and has no effect
        on the estimation"."""
        plain = ClvData(cdnow, time_unit="week", estimation_split="1997-10-15")
        extended = ClvData(
            cdnow, time_unit="week", estimation_split="1997-10-15",
            data_end="1998-12-31",
        )
        assert extended.estimation_end == plain.estimation_end
        assert extended.holdout_start == plain.holdout_start
        assert extended.data_end == pd.Timestamp("1998-12-31")
        assert extended.data_end > plain.data_end

    def test_extending_data_end_leaves_the_estimation_summary_alone(self, cdnow):
        """The invariant that matters: no estimation-period statistic moves."""
        plain = ClvData(cdnow, time_unit="week", estimation_split="1997-10-15")
        extended = ClvData(
            cdnow, time_unit="week", estimation_split="1997-10-15",
            data_end="1998-12-31",
        )
        pd.testing.assert_frame_equal(
            plain.customer_summary(), extended.customer_summary()
        )


class TestSubsetting:
    """``?subset.clv.data``, in the terms this package offers instead.

    :meth:`~clvtools.data.ClvData.as_data_frame` documents that R's
    ``subset()`` has no counterpart here because pandas already has one. These
    turn that claim into tests: every expression in the man page's example is
    written out and checked against the filter it is supposed to be equal to.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def clv_subset(cdnow):
        """The man page's own object: ``estimation.split = "1997-09-30"``."""
        return ClvData(cdnow, time_unit="week", estimation_split="1997-09-30")

    def test_one_customer(self, clv_subset):
        """``subset(clv.cdnow, Id == "1")``."""
        got = clv_subset.as_data_frame(ids="1")
        assert set(got["Id"]) == {"1"}
        assert len(got) == len(
            clv_subset.as_data_frame().query('Id == "1"')
        )

    def test_one_customer_per_sample(self, clv_subset):
        """``subset(clv.cdnow, Id == "111", sample = "estimation")`` and
        the same with ``sample = "holdout"``; together they are the whole."""
        estimation = clv_subset.as_data_frame(sample="estimation", ids="111")
        holdout = clv_subset.as_data_frame(sample="holdout", ids="111")
        whole = clv_subset.as_data_frame(ids="111")
        assert len(estimation) + len(holdout) == len(whole)
        assert (estimation["Date"] <= clv_subset.estimation_end).all()
        assert (holdout["Date"] > clv_subset.estimation_end).all()

    def test_several_customers(self, clv_subset):
        """``subset(clv.cdnow, Id %in% c("1", "2", "999"))``."""
        got = clv_subset.as_data_frame(ids=["1", "2", "999"])
        assert set(got["Id"]) == {"1", "2", "999"}

    def test_one_date(self, clv_subset):
        """``subset(clv.cdnow, Date == "1997-02-16")``."""
        when = pd.Timestamp("1997-02-16")
        frame = clv_subset.as_data_frame()
        got = frame[frame["Date"] == when]
        assert len(got) == 44
        assert (got["Date"] == when).all()
        # The `query()` spelling of the same thing, which needs the timestamp
        # bound rather than written inline -- see the test below.
        assert len(frame.query("Date == @when")) == 44

    def test_a_bare_date_string_in_query_matches_nothing(self, clv_subset):
        """The trap in the idiom :meth:`as_data_frame` recommends.

        ``query('Date == "1997-02-16"')`` returns an *empty* frame rather than
        raising: pandas 3 does not coerce the string to a timestamp for `==`.
        Range comparisons on the same column do coerce, which is what makes
        this worth pinning -- the two spellings look alike and only one of them
        is silently wrong. `as_data_frame`'s docstring says so; this fails if
        pandas ever changes its mind.
        """
        frame = clv_subset.as_data_frame()
        assert frame.query('Date == "1997-02-16"').empty
        assert not frame.query('"1997-02-01" <= Date <= "1997-02-16"').empty

    def test_a_date_range(self, clv_subset):
        """``subset(clv.cdnow, Date >= "1997-02-01" & Date <= "1997-02-16")``,
        which the man page also writes with data.table's ``between()``."""
        frame = clv_subset.as_data_frame()
        got = frame.query('"1997-02-01" <= Date <= "1997-02-16"')
        assert not got.empty
        assert got["Date"].min() >= pd.Timestamp("1997-02-01")
        assert got["Date"].max() <= pd.Timestamp("1997-02-16")
        assert len(got) >= int((frame["Date"] == pd.Timestamp("1997-02-16")).sum())

    def test_a_price_range(self, clv_subset):
        """``subset(clv.cdnow, Price >= 50 & Price <= 100)``."""
        got = clv_subset.as_data_frame().query("50 <= Price <= 100")
        assert not got.empty
        assert got["Price"].between(50, 100).all()

    def test_selecting_a_column(self, clv_subset):
        """``subset(clv.cdnow, Date == "1997-02-16", "Id")``."""
        when = pd.Timestamp("1997-02-16")
        frame = clv_subset.as_data_frame()
        got = frame[frame["Date"] == when][["Id"]]
        assert list(got.columns) == ["Id"]
        assert len(got) == 44

    def test_the_samples_partition_the_whole(self, clv_subset):
        """Not in the man page, but the property its examples rely on."""
        whole = clv_subset.as_data_frame()
        estimation = clv_subset.as_data_frame(sample="estimation")
        holdout = clv_subset.as_data_frame(sample="holdout")
        assert len(estimation) + len(holdout) == len(whole)


class TestTheDescriptiveLayerOnASecondDataset:
    """``?plot.clv.data`` and ``?spending`` demonstrate these on ``cdnow``.

    The variants themselves are already covered on the apparel data; what is
    new here is the data. Every descriptive in S6.1.2 has only ever run against
    one 600-customer cohort with a 104-week split. CDNOW is four times larger,
    split at 37 weeks, and 61% of its customers never come back -- a much
    heavier zero-repeater tail than apparel's 35.5%.
    """

    def test_frequency_bins_account_for_every_customer(self, clv):
        from clvtools.diagnostics import frequency_data

        frame = frequency_data(clv)
        assert int(frame["num.customers"].sum()) == CDNOW_N_CUSTOMERS
        assert int(frame["num.customers"].iloc[0]) == CDNOW_FREQUENCIES[0]

    def test_frequency_with_the_documented_bins(self, clv):
        """``trans.bins = 0:15, label.remaining = "16+"``."""
        from clvtools.diagnostics import frequency_data

        frame = frequency_data(clv, bins=range(16), label_remaining="16+")
        assert len(frame) == 17
        assert frame["num.transactions"].iloc[-1] == "16+"
        assert int(frame["num.customers"].sum()) == CDNOW_N_CUSTOMERS

    def test_interpurchase_times_drop_the_zero_repeaters(self, clv):
        from clvtools.diagnostics import interpurchase_time_data

        frame = interpurchase_time_data(clv)
        assert len(frame) == CDNOW_N_CUSTOMERS - CDNOW_FREQUENCIES[0]
        assert frame["mean.interpurchase.time"].notna().all()

    def test_spending_per_transaction_and_per_customer_differ(self, clv):
        """``which = "spending", mean.spending = TRUE`` and ``FALSE``."""
        from clvtools.diagnostics import spending_data

        per_customer = spending_data(clv, mean_spending=True)
        per_transaction = spending_data(clv, mean_spending=False)
        assert len(per_transaction) > len(per_customer)
        assert per_customer["Spending"].mean() > 0

    def test_timings_samples_the_customers(self, clv):
        """``which = "timings", ids = 25``."""
        from clvtools.diagnostics import timings_data

        frame = timings_data(clv, n=25, seed=1)
        assert frame["Id"].nunique() == 25

    def test_the_tracking_grid_starts_at_the_first_transaction(self, clv):
        """S6.2.2: the first predicted date is the start of the data, where
        the expected number of repeat transactions is zero by definition."""
        from clvtools.diagnostics import tracking_data

        frame = tracking_data(clv)
        observed = frame[frame["variable"] == REPEAT_TRANSACTIONS]
        assert observed["period.until"].iloc[0] == clv.estimation_start
        assert observed["value"].iloc[0] == 0
        # Summed over the estimation grid, the plotted series is the CBS --
        # a cross-check between the tracking plot and the model's own inputs.
        # The whole series runs past that, into the holdout, and its final
        # period is only partly covered and so reports no count at all.
        upto = observed[observed["period.until"] <= clv.estimation_end]
        assert upto["value"].sum() == float(clv.customer_summary()["x"].sum())
        assert pd.isna(observed["value"].iloc[-1])

    def test_the_summary_totals_agree_with_the_data(self, clv, cdnow):
        table = clv.summary()
        assert table.loc["Number of customers", "Total"] == CDNOW_N_CUSTOMERS
        assert table.loc["Total # Transactions", "Total"] == len(
            clv.as_data_frame()
        )
        assert table.loc["Percentage of zero repeaters", "Estimation"] == (
            pytest.approx(100 * CDNOW_FREQUENCIES[0] / CDNOW_N_CUSTOMERS, abs=1e-9)
        )


@pytest.mark.slow
class TestSpendingOnCdnow:
    """``spending(family = gg, data = clv.cdnow)`` -- ``?spending``.

    The man page fits the Gamma/Gamma on ``cdnow`` both ways round, with and
    without the first transaction. S6.2.3's default drops it; the alternative
    exists because a prediction that counts the initial purchase needs a
    spending estimate that counted it too.
    """

    def test_both_conventions_estimate(self, clv):
        from clvtools import gg, spending

        kept = spending(family=gg, data=clv, remove_first_transaction=False,
                        hessian=False)
        dropped = spending(family=gg, data=clv, hessian=False)
        for fit in (kept, dropped):
            assert all(v > 0 for v in fit)
            assert fit.converged

    def test_dropping_the_first_transaction_empties_the_zero_repeaters(self, clv):
        """Which is what the two conventions actually differ by.

        Both frames carry every customer. Removing the first transaction
        leaves a customer who bought once with nothing left to average, and
        the number of those is exactly the zero-repeater count `?pmf` prints.
        """
        kept = clv.spending_summary(remove_first_transaction=False)
        dropped = clv.spending_summary(remove_first_transaction=True)
        assert len(kept) == len(dropped) == CDNOW_N_CUSTOMERS
        assert int((dropped["x"] == 0).sum()) == CDNOW_FREQUENCIES[0]
        assert int((kept["x"] == 0).sum()) == 0
        assert (dropped.loc[dropped["x"] == 0, "Spending"] == 0).all()
