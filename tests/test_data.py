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

from clvtools.data import (
    ClvData,
    ClvDataStaticCov,
    load_apparel_static_cov,
    load_apparel_trans,
)


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


class TestBadInputIsLoud:
    """Findings 6 and 12 of ``docs/review-2026-09-02.md``.

    Each of these used to be accepted and to come back as a plausible number
    much later, which is the failure mode the review calls silent-wrong. The
    reproductions are in the messages: what the value *was* before it raised.
    """

    def test_non_finite_prices_are_rejected(self):
        """A customer whose prices are all NaN was recorded as Spending = 0.

        Counted in ``x``, dropped from the mean, and then ``fillna(0.0)``.
        The Gamma-Gamma silently excluded the row and ``predict()`` reported
        the population mean for that customer.
        """
        trans = load_apparel_trans().copy()
        victim = trans["Id"].iloc[0]
        trans.loc[trans["Id"] == victim, "Price"] = np.nan
        with pytest.raises(ValueError, match="Price is not finite"):
            ClvData(trans, estimation_split=104)

    def test_a_misspelled_price_column_raises(self):
        """It used to disable spending silently, surfacing much later."""
        with pytest.raises(ValueError, match="name_price='Prise'"):
            ClvData(load_apparel_trans(), estimation_split=104, name_price="Prise")

    def test_name_price_none_still_means_no_spending(self):
        """The escape hatch the error above points at."""
        data = ClvData(load_apparel_trans(), estimation_split=104, name_price=None)
        assert not data.has_spending

    def test_non_finite_covariates_are_rejected(self):
        """One NaN gave a fit at the start values with -inf and no exception."""
        cov = load_apparel_static_cov().copy()
        cov.loc[0, "Gender"] = np.nan
        data = ClvData(load_apparel_trans(), estimation_split=104)
        with pytest.raises(ValueError, match="not finite for 1 customer"):
            ClvDataStaticCov(
                data, cov,
                names_cov_life=["Gender", "Channel"],
                names_cov_trans=["Gender", "Channel"],
            )

    def test_duplicated_covariate_rows_are_rejected(self):
        """601 design rows for 600 customers, then a broadcast error deep in
        the likelihood that named neither the customer nor the frame."""
        cov = load_apparel_static_cov()
        data = ClvData(load_apparel_trans(), estimation_split=104)
        with pytest.raises(ValueError, match="duplicated customer id"):
            ClvDataStaticCov(
                data, pd.concat([cov, cov.head(1)], ignore_index=True),
                names_cov_life=["Gender"], names_cov_trans=["Gender"],
            )


class TestTheSharedValidatorAndResultHelper:
    """``clvtools._validate``: the two halves of the review's findings 5 and 7."""

    def test_non_finite_history_is_rejected(self):
        """NaN in x, t_x or T makes every likelihood evaluation NaN."""
        from clvtools._validate import customer_history

        with pytest.raises(ValueError, match="must all be finite"):
            customer_history(
                np.array([1.0, np.nan]),
                np.array([10.0, 5.0]),
                np.array([104.0, 104.0]),
            )

    def test_a_non_finite_objective_is_not_a_fit(self):
        """The other half of finding 5: it used to be returned as estimates.

        When the objective is infinite everywhere, the optimiser hands back
        the point it started from. Dressed in a result object that is
        indistinguishable from a fit except for a flag, which is how
        ``r = alpha = s = beta = 1`` reached a caller as an estimate.
        """
        from types import SimpleNamespace

        from clvtools._validate import finished

        stuck = SimpleNamespace(success=False, fun=np.inf, message="ABNORMAL")
        with pytest.raises(ValueError, match="not finite at the point"):
            finished(stuck, "Pareto/NBD")

    def test_a_polish_that_wins_without_converging_is_reported(self):
        """Finding 7's ambiguous flag, in the one shape that cannot be fitted
        into existence: the gradient search converges, the Nelder-Mead polish
        finds a better point but hits its own 20,000-evaluation cap, and
        ``converged`` then describes the polish rather than the search. The
        objective and the candidates are stubs because what is under test is
        the reporting, not the arithmetic.
        """
        from scipy import optimize

        import clvtools._staticcov as staticcov
        from clvtools._validate import ConvergenceWarning

        calls = []

        def fake_minimize(objective, x0, **kwargs):
            calls.append(kwargs.get("method"))
            if kwargs.get("method") == "Nelder-Mead":
                return optimize.OptimizeResult(
                    x=np.zeros(2), fun=1.0, success=False,
                    message="Maximum number of function evaluations has been exceeded.",
                )
            return optimize.OptimizeResult(
                x=np.zeros(2), fun=2.0, success=True, message="CONVERGENCE",
            )

        settings = staticcov.SearchSettings()
        monkeypatched = pytest.MonkeyPatch()
        monkeypatched.setattr(staticcov.optimize, "minimize", fake_minimize)
        try:
            with pytest.warns(ConvergenceWarning, match="polish improved"):
                result = staticcov._search(
                    lambda p: float(np.sum(p)), [np.zeros(2)], settings
                )
        finally:
            monkeypatched.undo()

        assert result.fun == 1.0          # the better point is what is reported
        assert "Nelder-Mead" in calls


class TestATransactionMustSayWhoAndWhen:
    """Finding A4 of ``docs/spec-audit.md``: four ways in, all of them quiet.

    The R suite is the only source that says what must happen on bad input, and
    it is not installed with the package, so none of this had ever been
    checked. Each reproduction is in the assertion: what the data *became*
    before it raised.
    """

    def test_a_missing_id_is_rejected(self):
        """It became the string ``"None"`` and modelled as a customer."""
        trans = load_apparel_trans().copy()
        trans.loc[0, "Id"] = None
        with pytest.raises(ValueError, match="1 transaction has no Id"):
            ClvData(trans, time_unit="week", estimation_split=104)

    def test_missing_dates_are_rejected_with_a_count(self):
        """``to_datetime`` dropped them: 3,187 rows in, 3,182 out, silently."""
        trans = load_apparel_trans().copy()
        trans.loc[[0, 1, 2], "Date"] = None
        with pytest.raises(ValueError, match="3 transactions have no Date"):
            ClvData(trans, time_unit="week", estimation_split=104)

    def test_an_unparseable_date_is_rejected(self):
        """Distinct from a missing one, and it says which values failed."""
        trans = load_apparel_trans().copy()
        trans["Date"] = trans["Date"].astype(str)
        trans.loc[0, "Date"] = "the fourth of never"
        with pytest.raises(ValueError, match="could not be parsed"):
            ClvData(trans, time_unit="week", estimation_split=104)

    def test_an_empty_frame_is_rejected(self):
        """It was accepted, and failed later somewhere less obvious."""
        with pytest.raises(ValueError, match="nothing to model"):
            ClvData(pd.DataFrame(columns=["Id", "Date", "Price"]), time_unit="week")

    def test_something_that_is_not_a_frame_is_rejected(self):
        """A list of dicts gave ``AttributeError`` from inside pandas."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            ClvData([{"Id": "1", "Date": "2005-01-02"}], time_unit="week")

    def test_timezone_aware_dates_are_refused_rather_than_half_supported(self):
        """Finding A6: it worked one way and raised pandas' error the other.

        A numeric estimation split built a usable object whose spans came from
        ``total_seconds()``, so a daylight-saving transition inside the window
        moved recency by an hour; a date or string split raised "Cannot
        compare tz-naive and tz-aware timestamps" from inside pandas. R has no
        such case -- its ``Date`` carries no zone -- so this is a decision
        rather than a divergence, and the decision is to refuse, because
        dropping the zone silently would move a late-evening transaction to
        the previous day.
        """
        trans = load_apparel_trans().copy()
        trans["Date"] = pd.to_datetime(trans["Date"]).dt.tz_localize("Europe/Berlin")
        with pytest.raises(ValueError, match="timezone-aware"):
            ClvData(trans, time_unit="week", estimation_split=104)

    def test_and_the_route_the_message_gives_works(self):
        """An error that names a fix should have that fix work."""
        trans = load_apparel_trans().copy()
        trans["Date"] = pd.to_datetime(trans["Date"]).dt.tz_localize("Europe/Berlin")
        trans["Date"] = trans["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
        converted = ClvData(trans, time_unit="week", estimation_split=104)
        plain = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        assert converted.nobs() == plain.nobs()


class TestTheColumnRenameActuallyRenames:
    """Finding B4: it was only ever exercised as the identity.

    The three uses in the suite were the default passed explicitly
    (``name_id="Id"``), a misspelling checked for raising, and ``None``. So the
    mapping in ``ClvData.__init__`` never renamed anything, and a mapping that
    pointed at the wrong column -- or silently dropped one -- would have passed
    every test in this repository. Spec D-14.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def renamed():
        return load_apparel_trans().rename(
            columns={"Id": "customer", "Date": "when", "Price": "amount"}
        )

    def test_a_full_rename_gives_the_same_data(self, renamed):
        plain = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        got = ClvData(
            renamed, time_unit="week", estimation_split=104,
            name_id="customer", name_date="when", name_price="amount",
        )
        assert got.nobs() == plain.nobs()
        assert got.has_spending
        pd.testing.assert_frame_equal(
            got.transactions.reset_index(drop=True),
            plain.transactions.reset_index(drop=True),
        )

    def test_the_summary_is_identical_too(self, renamed):
        """Not just the frame: everything derived from it."""
        plain = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        got = ClvData(
            renamed, time_unit="week", estimation_split=104,
            name_id="customer", name_date="when", name_price="amount",
        )
        pd.testing.assert_frame_equal(got.summary(), plain.summary())

    def test_naming_a_column_that_is_not_there_raises(self, renamed):
        """The mapping is checked, rather than quietly producing nothing."""
        with pytest.raises(ValueError, match="missing columns"):
            ClvData(renamed, time_unit="week", name_id="Id", name_date="when")


class TestReprNamesTheClassInHand:
    """Backlog item 27, finding 19: every subclass printed `ClvData(`.

    A literal class name in `__repr__` is inherited, so the covariate objects
    identified themselves as the base class -- in the one line most likely to
    be pasted into a bug report to say which object was held.
    """

    def test_each_class_names_itself(self, static_data):
        from clvtools import ClvData

        plain = ClvData(
            static_data.transactions, time_unit="week", estimation_split=104
        )
        assert repr(plain).startswith("ClvData(")
        assert repr(static_data).startswith("ClvDataStaticCov(")

    @pytest.mark.parametrize(
        "fragment", ["600 customers", "transactions", "weeks", "estimation"]
    )
    def test_the_rest_of_the_line_is_unchanged(self, static_data, fragment):
        """Only the prefix moved; the contents are still the contents."""
        assert fragment in repr(static_data)


class TestIdsMustBeStrings:
    """Backlog item 27, finding 20: `ids=1` gave "'int' object is not iterable".

    `Id` is a string everywhere in this package -- `tests/conftest.py` enforces
    it on every fixture -- so an integer id is a common slip with a one
    character fix, and the message now names it instead of reporting that an
    `int` cannot be iterated.
    """

    @pytest.mark.parametrize("method", ["as_data_frame", "summary"])
    def test_an_integer_id_says_what_to_do(self, static_data, method):
        with pytest.raises(ValueError, match="ids are strings here"):
            getattr(static_data, method)(ids=1)

    def test_a_string_id_still_works(self, static_data):
        assert len(static_data.as_data_frame(ids="1")) == 7

    def test_and_so_does_a_sequence_of_them(self, static_data):
        both = static_data.as_data_frame(ids=["1", "10"])
        assert set(both["Id"]) == {"1", "10"}


@pytest.mark.oracle
class TestTheFutureCovariatesMatchWhatRExported:
    """Backlog item 25: ``dyncov_future_covariates.json`` had no reader.

    S6.4.2 needs the covariate path *ahead* of the estimation period, and
    ``apparelDynCovFuture`` is where it comes from. Its shape was asserted
    nowhere -- the frame was loaded and used, so a truncated or duplicated
    export would have shown up as a prediction that disagreed with the oracle
    rather than as a dataset that was the wrong size, which is a much harder
    thing to read.
    """

    @pytest.fixture(scope="class")
    def want(self):
        from conftest import fixture_json

        return fixture_json("dyncov_future_covariates")

    def test_the_row_counts_are_rs(self, want):
        from clvtools import load_apparel_dyn_cov, load_apparel_dyn_cov_future

        assert len(load_apparel_dyn_cov()) == want["n.rows.past"]
        assert len(load_apparel_dyn_cov_future()) == want["n.rows.future"]

    def test_and_so_is_the_window_they_cover(self, want):
        """The future frame has to start where prediction does, not before."""
        from clvtools import load_apparel_dyn_cov_future

        dates = load_apparel_dyn_cov_future()["Cov.Date"]
        assert str(dates.min().date()) == want["first.future.date"]
        assert str(dates.max().date()) == want["last.future.date"]


class TestTheTwoSamplesPartitionTheLogExactly:
    """Spec T-01, `weak`: "the hour unit uses 1 hour, not 1 second".

    True, and it turns out not to matter -- which is the finding rather than a
    fix. `T-01` asks for an epsilon of one day on date-based units and one
    *second* on datetime-based ones. :attr:`ClvData.holdout_start` steps by one
    hour on hourly data and one day otherwise, so the constant genuinely
    differs from the spec's.

    It cannot be observed. ``_aggregate_to_day`` floors every transaction to
    the time unit before anything else looks at it -- ``"h"`` on hourly data,
    ``"D"`` otherwise, which is S6.1's own rule -- so no transaction can land
    strictly between the estimation end and one whole unit later. A finer
    epsilon would select the same rows.

    What is asserted here is the property the epsilon exists to guarantee: the
    two samples **partition** the log, with nothing dropped between them and
    nothing counted twice. That holds whatever the constant is, and would fail
    for a coarser one. Backlog item 34, round 5.
    """

    @staticmethod
    def _log(unit: str) -> pd.DataFrame:
        """Six purchases one *unit* apart, plus one half a unit off the grid.

        Stepped by the unit itself rather than by a day: with ``estimation_split
        = 3`` a day-stepped weekly log ends before its own split, which is what
        the first draft of this did.
        """
        from clvtools import timeunit

        period = timeunit.get(unit)
        start = pd.Timestamp("2005-01-02")
        rows = [
            {"Id": customer, "Date": period.add(start, n)}
            for customer in ("a", "b")
            for n in range(6)
        ]
        # Half a unit past the estimation end: the gap a coarse epsilon opens.
        rows.append({"Id": "a", "Date": period.add(start, 3.5)})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("unit", ["hour", "day", "week"])
    def test_nothing_falls_between_the_samples(self, unit):
        data = ClvData(self._log(unit), time_unit=unit, estimation_split=3)
        full = data.as_data_frame(sample="full")
        estimation = data.as_data_frame(sample="estimation")
        holdout = data.as_data_frame(sample="holdout")
        assert len(estimation) + len(holdout) == len(full)

    @pytest.mark.parametrize("unit", ["hour", "day", "week"])
    def test_and_nothing_is_counted_twice(self, unit):
        data = ClvData(self._log(unit), time_unit=unit, estimation_split=3)
        estimation = data.as_data_frame(sample="estimation")
        holdout = data.as_data_frame(sample="holdout")
        assert estimation["Date"].max() < holdout["Date"].min()

    def test_a_sub_unit_transaction_is_floored_rather_than_lost(self):
        """Which is why the epsilon's size is unobservable.

        The extra purchase 30 minutes past an hour boundary merges into that
        hour rather than sitting between the samples.
        """
        data = ClvData(self._log("hour"), time_unit="hour", estimation_split=3)
        stamps = data.as_data_frame()["Date"]
        assert (stamps.dt.minute == 0).all()
        assert (stamps.dt.second == 0).all()


class TestDataEndIsRefusedWhenItWouldDiscardTransactions:
    """Spec T-11, `weak`: tested "only without ``data_end``".

    `T-11` asks that ``data.end`` fail when it precedes ``estimation.split``.
    It does, through a **stricter** rule than the spec states: any ``data_end``
    before the last purchase is refused, and since the split always sits at or
    before the last purchase, a ``data_end`` before the split is refused too.

    Worth pinning as the stricter rule rather than the spec's, because that is
    what the code promises and the two are not the same statement. Backlog item
    34, round 5.
    """

    def test_before_the_estimation_split_is_refused(self, apparel_trans):
        with pytest.raises(ValueError, match="precedes the last purchase"):
            ClvData(
                apparel_trans, time_unit="week", estimation_split=104,
                data_end="2005-06-01",
            )

    def test_and_so_is_any_data_end_that_would_discard_a_purchase(
        self, apparel_trans
    ):
        """The stricter half: after the split, still before the last purchase."""
        with pytest.raises(ValueError, match="precedes the last purchase"):
            ClvData(
                apparel_trans, time_unit="week", estimation_split=104,
                data_end="2007-06-01",
            )
