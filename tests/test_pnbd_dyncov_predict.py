r"""S6.4.2 - prediction with time-varying covariates.

Like the likelihood it builds on, none of this is written out in the paper: the
expressions come from CLVTools, so the tests are held to CLVTools at its own
fitted parameters. Three layers, so a failure localises to one of them:

  * the ``ABCD`` table, the per-period covariate summaries both ``CET`` and
    ``DECT`` sum over, compared row for row for a handful of customers;
  * ``PAlive``, ``CET`` and ``DECT`` for all 600, against ``predict()``;
  * the paper's own S6.4.2 scenario -- fitted on the full data, with the
    covariate series extended into the prediction window.

Fitting is never involved: every parameter here is read from a fixture.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv, fixture_json
from paper_values import DYNCOV_FUTURE_PAPER

from clvtools import (
    ClvData,
    ClvDataDynCov,
    load_apparel_dyn_cov,
    load_apparel_dyn_cov_future,
    newcustomer_dynamic,
    predict,
    timeunit,
)
from clvtools.gg import GgParams
from clvtools.pnbd.dyncov import PnbdDynCovParams
from clvtools.pnbd.dyncov_predict import abcd, new_customer_expectation

NAMES = ["High.Season", "Gender", "Channel"]


def _params(fixture: str) -> PnbdDynCovParams:
    """CLVTools' own fitted parameters, in the dataclass ``predict`` expects."""
    fitted = fixture_json(fixture)
    c = fitted["coefficients"]
    return PnbdDynCovParams(
        r=c["r"], alpha=c["alpha"], s=c["s"], beta=c["beta"],
        gamma_life=np.array([c[f"life.{n}"] for n in NAMES]),
        gamma_trans=np.array([c[f"trans.{n}"] for n in NAMES]),
        names_cov_life=NAMES, names_cov_trans=NAMES,
        log_likelihood=fitted["logLik"], converged=True, n_customers=600,
    )


@pytest.fixture(scope="module")
def holdout_data(apparel_trans) -> ClvDataDynCov:
    return ClvDataDynCov(
        ClvData(apparel_trans, time_unit="week", estimation_split=104),
        load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
    )


@pytest.fixture(scope="module")
def future_data(apparel_trans) -> ClvDataDynCov:
    """The full data, with the covariate series continued past its end.

    S6.4.2 builds exactly this: ``rbind(apparelDynCov, apparelDynCovFuture)``,
    because "the time-varying covariates have to be available for the entire
    prediction period".
    """
    covariates = pd.concat(
        [load_apparel_dyn_cov(), load_apparel_dyn_cov_future()],
        ignore_index=True,
    )
    return ClvDataDynCov(
        ClvData(apparel_trans, time_unit="week"),
        covariates, names_cov_life=NAMES, names_cov_trans=NAMES,
    )


@pytest.mark.oracle
class TestAbcdTable:
    """The per-period covariate summaries CET and DECT are built from."""

    @staticmethod
    @pytest.fixture(scope="class")
    def built(apparel_trans):
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        settings = fixture_json("dyncov_predict_holdout_settings")
        table = abcd(data, _params("dyncov_fit"), pd.Timestamp(settings["prediction.end"]))
        return table[table["Id"].isin(settings["sample.ids"])].reset_index(drop=True)

    @pytest.mark.parametrize("column,oracle", [
        ("i", "i"), ("Ai", "Ai"), ("Ci", "Ci"), ("d1", "d1"),
        ("Bbar_i", "Bbar_i"), ("T_cal", "T.cal"), ("Dbar_i", "Dbar_i"),
    ])
    def test_column_matches(self, built, column, oracle):
        want = fixture_csv("dyncov_abcd_sample")
        assert len(built) == len(want)
        np.testing.assert_allclose(
            built[column].to_numpy(dtype=float),
            want[oracle].to_numpy(dtype=float),
            rtol=1e-10, atol=1e-10,
        )

    def test_the_window_starts_at_the_estimation_period_end(self, built):
        first = built.groupby("Id")["Cov.Date"].min().unique()
        assert list(first) == [pd.Timestamp("2006-12-31")]

    def test_the_first_period_is_a_whole_one_here(self, built):
        """``d1`` is what remains of the period after the estimation ends.

        The apparel split lands on a covariate boundary, so a whole period
        remains and ``d1`` is 1. It is not 1 in general, which
        :class:`TestAbcdOverEveryCustomer` is where it is checked.
        """
        assert set(built["d1"]) == {1.0}


class TestAbcdOverEveryCustomer:
    """DY-06, and the ``d1`` the apparel split cannot exercise.

    The oracle comparison above runs over the eight ids the fixture holds, and
    on a split that lands exactly on a covariate boundary -- so ``d1`` is 1 for
    every row of it and comparing that column against the oracle discriminates
    nothing. Findings B5 and B7 of ``docs/spec-audit.md``. Both claims below
    are determined by their own inputs, so neither needs the oracle.
    """

    @staticmethod
    def table(apparel_trans, split: str) -> pd.DataFrame:
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=split),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        return abcd(data, _params("dyncov_fit"), pd.Timestamp("2007-06-30"))

    def test_every_customer_shares_the_window(self, apparel_trans):
        """DY-06: "``i`` is an integer with the same maximum for every
        customer, and all customers start and end on the same date"."""
        table = self.table(apparel_trans, "2006-12-31")
        assert table["Id"].nunique() == 600

        per_customer = table.groupby("Id")
        assert set(table["i"]) == set(range(1, 27))
        np.testing.assert_array_equal(per_customer["i"].max().unique(), [26])
        np.testing.assert_array_equal(per_customer["i"].min().unique(), [1])
        assert list(per_customer["Cov.Date"].min().unique()) == [
            pd.Timestamp("2006-12-31")
        ]
        assert list(per_customer["Cov.Date"].max().unique()) == [
            pd.Timestamp("2007-06-24")
        ]

    def test_a_split_inside_a_period_leaves_a_fraction_of_it(self, apparel_trans):
        """The apparel covariate grid is weekly and starts on a Sunday, so a
        Wednesday split leaves four of the seven days: ``d1 = 4/7``. Nothing
        else about the window moves -- the grid is the covariates', not the
        split's."""
        table = self.table(apparel_trans, "2007-01-03")
        assert table["d1"].to_numpy() == pytest.approx(4 / 7)
        assert list(table.groupby("Id")["Cov.Date"].min().unique()) == [
            pd.Timestamp("2006-12-31")
        ]


@pytest.mark.oracle
class TestPredictionAgainstTheOracle:
    """PAlive, CET and DECT for every customer, over the holdout window."""

    @staticmethod
    @pytest.fixture(scope="class")
    def predicted(apparel_trans):
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        settings = fixture_json("dyncov_predict_holdout_settings")
        return predict(
            data, _params("dyncov_fit"),
            continuous_discount_factor=settings["continuous.discount.factor"],
        )

    @pytest.mark.parametrize("column,rtol", [
        ("PAlive", 1e-11), ("CET", 1e-11), ("DECT", 1e-8),
    ])
    def test_column_matches(self, predicted, column, rtol):
        want = fixture_csv("dyncov_predict_holdout").set_index("Id")
        np.testing.assert_allclose(
            predicted[column].to_numpy(), want[column].to_numpy(), rtol=rtol
        )

    def test_the_actuals_come_through(self, predicted):
        want = fixture_csv("dyncov_predict_holdout").set_index("Id")
        np.testing.assert_array_equal(
            predicted["actual.x"].to_numpy(), want["actual.x"].to_numpy()
        )

    def test_column_names_follow_the_covariate_model(self, predicted):
        """S6.4.2 reports DECT, not DERT: the horizon here is finite."""
        assert "DECT" in predicted.columns
        assert "DERT" not in predicted.columns

    def test_palive_is_a_probability(self, predicted):
        """To floating point: one customer comes out at 1 + 1.4e-14.

        CLVTools' own value for that customer is over one by the same amount,
        so this is the arithmetic rather than a difference between the two.

        What is asserted is the *size* of the excess rather than how many
        customers show one. Counting equality with the oracle asks whether a
        value 1.4e-14 above one lands on the same side of one on both
        platforms, which is not something either implementation determines --
        the precision rule at the top of ``test_pnbd_fit.py``, applied here.
        """
        assert predicted["PAlive"].between(0, 1 + 1e-12).all()
        want = fixture_csv("dyncov_predict_holdout").set_index("Id")
        assert (want["PAlive"] - 1).max() < 1e-12
        excess = (predicted["PAlive"] - 1).clip(lower=0)
        assert excess.max() < 1e-12

    def test_dect_is_below_cet(self, predicted):
        """Discounting can only reduce a positive count."""
        assert (predicted["DECT"] <= predicted["CET"]).all()


@pytest.mark.oracle
class TestThePapersScenario:
    """S6.4.2's own prediction: full data, covariates extended, 95 weeks."""

    @staticmethod
    @pytest.fixture(scope="class")
    def predicted(apparel_trans):
        covariates = pd.concat(
            [load_apparel_dyn_cov(), load_apparel_dyn_cov_future()],
            ignore_index=True,
        )
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week"),
            covariates, names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        fitted = fixture_json("dyncov_fit_full")
        spending = GgParams(
            **fitted["spending.coefficients"], log_likelihood=float("nan"),
            converged=True, n_customers=600,
        )
        return predict(
            data, _params("dyncov_fit_full"), spending,
            prediction_end=fitted["prediction.periods"],
            continuous_discount_factor=fitted["continuous.discount.factor"],
        )

    @pytest.mark.parametrize("column", [
        "PAlive", "CET", "DECT", "predicted.mean.spending",
        "predicted.period.spending", "predicted.period.CLV",
    ])
    def test_column_matches(self, predicted, column):
        want = fixture_csv("dyncov_predict_future").set_index("Id")
        np.testing.assert_allclose(
            predicted[column].to_numpy(), want[column].to_numpy(), rtol=1e-9
        )

    def test_the_window_is_the_papers(self, predicted):
        assert predicted["period.first"].iloc[0] == pd.Timestamp("2010-12-21")
        assert predicted["period.last"].iloc[0] == pd.Timestamp("2012-10-15")
        assert predicted["period.length"].iloc[0] == 95

    def test_the_paper_printed_a_different_fit(self, predicted):
        """S6.4.2's own numbers are not reachable from CLVTools 0.12.1.

        The paper prints PAlive = 0.0139206 for customer 1 where CLVTools
        0.12.1 predicts 0.0107292 from its own fit, and this package reproduces
        0.12.1 to 1e-12 (above). The likelihood is not in dispute -- it agrees
        to nine significant figures at fixed parameters -- so what differs is
        where each optimiser stopped. Asserting the gap keeps it from being
        mistaken for a defect here later.
        """
        for customer, printed in DYNCOV_FUTURE_PAPER.items():
            for column, value in printed.items():
                assert predicted.loc[customer, column] != pytest.approx(
                    value, rel=1e-3
                ), f"{customer}/{column} now matches the paper"

    def test_spending_columns_are_definitional(self, predicted):
        np.testing.assert_allclose(
            predicted["predicted.period.CLV"],
            predicted["DECT"] * predicted["predicted.mean.spending"],
        )
        np.testing.assert_allclose(
            predicted["predicted.period.spending"],
            predicted["CET"] * predicted["predicted.mean.spending"],
        )

    def test_prediction_needs_covariates_for_the_whole_window(self, apparel_trans):
        """Without the extension, the series stops at the estimation end."""
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week"),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        with pytest.raises(ValueError, match="does not reach the prediction"):
            predict(data, _params("dyncov_fit_full"), prediction_end=95)

    def test_a_dyncov_model_needs_dyncov_data(self, apparel_trans):
        with pytest.raises(TypeError, match="ClvDataDynCov"):
            predict(
                ClvData(apparel_trans, time_unit="week"),
                _params("dyncov_fit_full"), prediction_end=95,
            )


@pytest.mark.oracle
class TestProspectiveCustomerOnACovariatePath:
    """``newcustomer.dynamic()``, which S6.3.4 points at for scenario work."""

    @staticmethod
    def _covariates(want) -> pd.DataFrame:
        return pd.DataFrame({
            "Cov.Date": pd.to_datetime(want["cov.dates"]),
            "High.Season": want["High.Season"],
            "Gender": want["Gender"],
            "Channel": want["Channel"],
        })

    def test_matches_the_oracle(self):
        want = fixture_json("dyncov_newcustomer")
        covariates = self._covariates(want)
        got = predict(
            newcustomer_dynamic(
                want["num.periods"], covariates, covariates,
                want["first.transaction"],
            ),
            _params("dyncov_fit_full"),
        )
        assert got == pytest.approx(want["expected.num.transactions"], rel=1e-12)

    def test_the_first_transaction_counts(self):
        """S6.3.4 adds one for the purchase that makes them a customer."""
        want = fixture_json("dyncov_newcustomer")
        covariates = self._covariates(want)
        expectation = new_customer_expectation(
            _params("dyncov_fit_full"), want["num.periods"],
            pd.Timestamp(want["first.transaction"]), covariates, covariates,
            timeunit.get("week"),
        )
        assert expectation == pytest.approx(
            want["expected.num.transactions"] - 1.0, rel=1e-12
        )

    def test_a_longer_horizon_expects_more(self):
        want = fixture_json("dyncov_newcustomer")
        covariates = self._covariates(want)
        params = _params("dyncov_fit_full")
        short, long = (
            predict(
                newcustomer_dynamic(
                    periods, covariates, covariates, want["first.transaction"]
                ),
                params,
            )
            for periods in (5, 10)
        )
        assert long > short

    def test_it_needs_a_time_varying_model(self):
        want = fixture_json("dyncov_newcustomer")
        covariates = self._covariates(want)
        from clvtools.pnbd.fit import PnbdParams

        plain = PnbdParams(
            r=1.0, alpha=1.0, s=1.0, beta=1.0, log_likelihood=float("nan"),
            converged=True, n_customers=1,
        )
        with pytest.raises(TypeError, match="time-varying covariate model"):
            predict(
                newcustomer_dynamic(
                    10, covariates, covariates, want["first.transaction"]
                ),
                plain,
            )

    def test_a_horizon_inside_the_first_period(self):
        """The other branch: no earlier period for the sum to telescope through."""
        want = fixture_json("dyncov_newcustomer_single_period")
        covariates = self._covariates(fixture_json("dyncov_newcustomer"))
        got = predict(
            newcustomer_dynamic(
                want["num.periods"], covariates, covariates,
                want["first.transaction"],
            ),
            _params("dyncov_fit_full"),
        )
        assert got == pytest.approx(want["expected.num.transactions"], rel=1e-12)

    def test_a_date_outside_the_covariate_series_is_refused(self):
        want = fixture_json("dyncov_newcustomer")
        covariates = self._covariates(want)
        with pytest.raises(ValueError, match="no covariate period covers"):
            predict(
                newcustomer_dynamic(
                    2, covariates, covariates, "2009-01-04"
                ),
                _params("dyncov_fit_full"),
            )

    def test_the_series_must_reach_the_horizon(self):
        want = fixture_json("dyncov_newcustomer")
        covariates = self._covariates(want).head(3)
        with pytest.raises(ValueError, match="does not reach the prediction"):
            predict(
                newcustomer_dynamic(
                    52, covariates, covariates, want["first.transaction"]
                ),
                _params("dyncov_fit_full"),
            )


class TestGuardsOnTheCovariateSeries:
    """The window has to be covered on both processes, and reach past the fit."""

    @staticmethod
    def _truncated(apparel_trans, upper: str) -> ClvDataDynCov:
        covariates = load_apparel_dyn_cov()
        return ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            covariates[covariates["Cov.Date"] <= pd.Timestamp(upper)],
            names_cov_life=NAMES, names_cov_trans=NAMES,
        )

    def test_a_series_stopping_at_the_estimation_end(self, apparel_trans):
        """Nothing to predict over: every covariate period is already spent."""
        data = self._truncated(apparel_trans, "2006-12-31")
        with pytest.raises(ValueError, match="does not reach past"):
            predict(
                data, _params("dyncov_fit"), prediction_end="2007-01-05",
            )

    def test_the_two_processes_must_cover_the_same_periods(self, apparel_trans):
        covariates = load_apparel_dyn_cov()
        short = covariates[covariates["Cov.Date"] <= pd.Timestamp("2008-01-06")]
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            data_cov_life=covariates, data_cov_trans=short,
            names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        with pytest.raises(ValueError, match="do not cover every period"):
            predict(data, _params("dyncov_fit"), prediction_end="2010-12-20")

    def test_a_prospective_customer_needs_a_second_period(self):
        want = fixture_json("dyncov_newcustomer")
        one_row = pd.DataFrame({
            "Cov.Date": [pd.Timestamp("2010-12-19")],
            "High.Season": [1], "Gender": [0], "Channel": [1],
        })
        with pytest.raises(ValueError, match="ends in the first period"):
            predict(
                newcustomer_dynamic(
                    0.1, one_row, one_row, want["first.transaction"]
                ),
                _params("dyncov_fit_full"),
            )

    @pytest.mark.parametrize("periods", [-1, -0.5])
    def test_a_negative_horizon_is_refused(self, periods):
        """Zero is admitted now, as R admits it -- finding A1. Negative is
        not a horizon at all."""
        want = fixture_json("dyncov_newcustomer")
        covariates = pd.DataFrame({
            "Cov.Date": pd.to_datetime(want["cov.dates"]),
            "High.Season": 1, "Gender": 0, "Channel": 1,
        })
        with pytest.raises(ValueError, match="must not be negative"):
            newcustomer_dynamic(
                periods, covariates, covariates, want["first.transaction"]
            )


class TestStaticCovariatesSuppliedAsDynamic:
    r"""DY-07: the cleanest cross-check of the dyncov machinery, needing no R.

    If a covariate never changes, a time-varying model must reduce to the
    time-invariant one, and the walk quantities say so exactly. Supplying
    constant covariates as dynamic data, CLVTools' own suite asserts three
    things of the CET input table
    (``test_correctness_pnbd_dyncov.R:208``, asserted at line 224):

    * :math:`A_i` and :math:`C_i` equal the static covariate values;
    * :math:`\bar{D}_i = 0`;
    * :math:`\bar{B}_i = -T_{cal} A_i`.

    None of it needs a fixture: the input is constructed and the output is
    determined by it. Finding D1 of ``docs/spec-audit.md``, and the reason it
    matters is that the dyncov path is otherwise checked only against oracle
    tables that CLVTools produced with the same arrangement of the arithmetic.
    """

    NAMES: ClassVar[list[str]] = ["Gender", "Channel"]

    @staticmethod
    @pytest.fixture(scope="class")
    def constant_covariates():
        """One row per customer per week, never changing within a customer."""
        from clvtools import load_apparel_static_cov

        static = load_apparel_static_cov().set_index("Id")
        weeks = pd.date_range("2005-01-02", "2013-01-06", freq="7D")
        frames = []
        for customer, row in static.iterrows():
            frames.append(pd.DataFrame({
                "Id": customer,
                "Cov.Date": weeks,
                "Gender": float(row["Gender"]),
                "Channel": float(row["Channel"]),
            }))
        return static, pd.concat(frames, ignore_index=True)

    @staticmethod
    @pytest.fixture(scope="class")
    def table(apparel_trans, constant_covariates):
        _, covariates = constant_covariates
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            covariates,
            names_cov_life=TestStaticCovariatesSuppliedAsDynamic.NAMES,
            names_cov_trans=TestStaticCovariatesSuppliedAsDynamic.NAMES,
        )
        params = _params("dyncov_fit")
        # The fixture's parameters carry three covariates; this data has two.
        two = replace(
            params,
            gamma_life=params.gamma_life[:2],
            gamma_trans=params.gamma_trans[:2],
            names_cov_life=TestStaticCovariatesSuppliedAsDynamic.NAMES,
            names_cov_trans=TestStaticCovariatesSuppliedAsDynamic.NAMES,
        )
        table = abcd(data, two, pd.Timestamp("2007-12-29"))
        # Asserted here because three of the checks below are `assert_allclose`
        # against constants, and every one of them passes on an empty frame:
        # the first draft of this used a prediction end *before* the
        # estimation end and looked green.
        assert len(table) > 0
        assert table["Id"].nunique() == 600
        return data, two, table

    def test_the_multipliers_are_the_static_values(self, table, constant_covariates):
        r""":math:`A_i = e^{\gamma' x}` with the customer's own covariates."""
        static, _ = constant_covariates
        _, params, got = table
        design = np.column_stack([
            static.loc[got["Id"], name].to_numpy(dtype=float)
            for name in self.NAMES
        ])
        np.testing.assert_allclose(
            got["Ai"].to_numpy(),
            np.exp(design @ params.gamma_trans),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            got["Ci"].to_numpy(),
            np.exp(design @ params.gamma_life),
            rtol=1e-12,
        )

    def test_the_multipliers_never_change_within_a_customer(self, table):
        """Which is the premise: a constant covariate is a constant multiplier."""
        _, _, got = table
        for column in ("Ai", "Ci"):
            spread = got.groupby("Id")[column].agg(lambda s: s.max() - s.min())
            assert spread.max() < 1e-12, column

    def test_the_integrated_lifetime_multiplier_is_zero(self, table):
        r""":math:`\bar{D}_i = 0` when nothing varies."""
        _, _, got = table
        np.testing.assert_allclose(got["Dbar_i"].to_numpy(), 0.0, atol=1e-9)

    def test_the_integrated_transaction_multiplier_is_minus_t_cal(self, table):
        r""":math:`\bar{B}_i = -T_{cal} A_i`, the offset the sums are written
        around."""
        data, _, got = table
        T_cal = data.customer_summary().set_index("Id")["T"]
        np.testing.assert_allclose(
            got["Bbar_i"].to_numpy(),
            -T_cal.loc[got["Id"]].to_numpy() * got["Ai"].to_numpy(),
            rtol=1e-9,
        )


class TestTheDyncovCetRefusesUnitS:
    """Backlog item 29, from finding 10: only one of the two CETs guarded s = 1.

    Every expression in the time-varying ``CET`` divides by ``s - 1``, and
    :func:`clvtools.pnbd.aggregate.conditional_expected_transactions` has raised
    at ``s = 1`` since it was written. This one divided anyway, so the same
    model at the same parameters answered differently depending on which entry
    point was asked -- ``inf``, or a very large finite number, according to
    which side of 1 the optimiser had stopped on.
    """

    def test_the_two_entry_points_now_agree_that_it_is_undefined(self):
        from clvtools.pnbd import aggregate
        from clvtools.pnbd.dyncov_predict import _reject_unit_s

        with pytest.raises(ValueError, match="s = 1"):
            _reject_unit_s(1.0)
        with pytest.raises(ValueError, match="s = 1"):
            aggregate.conditional_expected_transactions(
                1, 2.0, 52.0, 26.0, r=1.0, alpha=10.0, s=1.0, beta=10.0
            )

    def test_the_message_is_the_same_one(self):
        """Two spellings of the same refusal would be its own small trap."""
        from clvtools.pnbd import aggregate
        from clvtools.pnbd.dyncov_predict import _reject_unit_s

        errors = []
        for call in (
            lambda: _reject_unit_s(1.0),
            lambda: aggregate.conditional_expected_transactions(
                1, 2.0, 52.0, 26.0, r=1.0, alpha=10.0, s=1.0, beta=10.0
            ),
        ):
            with pytest.raises(ValueError, match="s = 1") as excinfo:
                call()
            errors.append(str(excinfo.value))
        assert errors[0] == errors[1]

    @pytest.mark.parametrize("s", [0.5, 0.999, 1.001, 1.5, 2.0])
    def test_and_it_does_not_fire_away_from_one(self, s):
        """`np.isclose` decides `near`, so both shoulders are checked."""
        from clvtools.pnbd.dyncov_predict import _reject_unit_s

        _reject_unit_s(s)


@pytest.mark.oracle
class TestDyncovPredictionOnNewData:
    """Spec DY-24, `weak`: "2 of 5" of its runability claims reached.

    Five claims: predicting the original data, a sample of it, further ahead
    than the fitting data allows, plotting further ahead, and with two periods
    or fewer -- CLVTools' issue #128. The first two were covered; the short
    horizons and the over-long one were not, and short horizons are where an
    off-by-one in the covariate grid shows itself.

    None turned up a defect. Backlog item 34, round 5.
    """

    @pytest.fixture(scope="class")
    def fitted(self, apparel_trans):
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        return data, _params("dyncov_fit")

    @pytest.mark.parametrize("periods", [1, 2])
    def test_two_periods_or_fewer_predicts(self, fitted, periods):
        """Issue #128's case, which the apparel tests never reached."""
        data, params = fitted
        table = predict(data, params, prediction_end=periods)
        assert len(table) == 600
        assert table["period.length"].iloc[0] == pytest.approx(float(periods))
        assert np.isfinite(table["CET"]).all()
        assert (table["CET"] > 0).all()

    def test_a_longer_horizon_predicts_more(self, fitted):
        """One period against two: the extra period has to show up in CET."""
        data, params = fitted
        one = predict(data, params, prediction_end=1)
        two = predict(data, params, prediction_end=2)
        assert (two["CET"] > one["CET"]).all()

    def test_predicting_a_sample_of_the_customers_works(self, apparel_trans):
        """DY-24's `newdata` claim, on a subset rather than the whole cohort."""
        sample_ids = sorted(apparel_trans["Id"].unique())[:20]
        sample = apparel_trans[apparel_trans["Id"].isin(sample_ids)]
        data = ClvDataDynCov(
            ClvData(sample, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        table = predict(data, _params("dyncov_fit"), prediction_end=4)
        assert len(table) == 20
        assert np.isfinite(table["PAlive"]).all()

    def test_but_further_ahead_than_the_covariates_reach_is_refused(self, fitted):
        """Not "predicting further ahead than the fitting data allows" silently.

        The covariate series ends 2010-12-26; a horizon past one period beyond
        that has nothing to integrate, and `_require_coverage` says so instead
        of stopping the walk short and returning a confident answer to a
        different question.
        """
        data, params = fitted
        with pytest.raises(ValueError, match="does not reach the prediction"):
            predict(data, params, prediction_end=400)


class TestTheAbcdTableUnderZeroAndSharedCoefficients:
    """Spec DY-02, DY-04 and DY-05 — round 6's `absent` rows, and two of them
    are claims this suite should *not* try to satisfy.

    `DY-02` holds and is worth having: with every covariate zero, ``exp(gamma'x)``
    is 1, so ``Ai`` and ``Ci`` are 1 exactly. That is the multiplier's identity,
    checked bit for bit because ``exp(0)`` is 1 to the bit.

    `DY-04` and `DY-05` do **not** hold as the audit states them, and the reason
    is structural rather than a defect. ``Bbar_i`` and ``Dbar_i`` integrate the
    same multiplier over **different spans**: ``Bbar`` runs from the estimation
    end, offset by ``-T_cal - d1 - (i - 2)``, and ``Dbar`` from the customer's
    birth, offset by ``-d_omega``. With identical covariate data and identical
    coefficients ``Ai == Ci`` bit for bit -- and ``Bbar_i`` is -140 where
    ``Dbar_i`` is +17, because they are not the same integral.

    Both columns are already compared against CLVTools' own
    ``pnbd_dyncov_ABCD`` at ``rtol=1e-10`` by
    :class:`TestAbcdAgainstTheOracle` above, and they pass -- so the asymmetry
    is CLVTools' too, and a test written to the audit's one-line reading would
    fail against the oracle it is meant to agree with. `docs/spec.md` already
    warns about exactly this pair: "two tables in the R file share column names
    and disagree at ``i = 1`` ... check bodies, not titles". Recorded rather
    than chased. Backlog item 36, round 6.
    """

    @pytest.fixture(scope="class")
    def zeroed(self, apparel_trans):
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        params = _params("dyncov_fit")
        zero = replace(
            params,
            gamma_life=np.zeros(len(NAMES)), gamma_trans=np.zeros(len(NAMES)),
        )
        return abcd(data, zero, pd.Timestamp("2007-06-30"))

    def test_zero_covariates_make_both_multipliers_exactly_one(self, zeroed):
        """DY-02: `exp(0)` is 1 to the bit, so this needs no tolerance."""
        assert np.array_equal(zeroed["Ai"].to_numpy(), np.ones(len(zeroed)))
        assert np.array_equal(zeroed["Ci"].to_numpy(), np.ones(len(zeroed)))

    def test_the_integrated_columns_are_not_the_same_integral(
        self, apparel_trans
    ):
        """DY-04 and DY-05, pinned as the asymmetry they are.

        Written so that a future reader who reaches for the audit's reading
        finds the reason it is wrong, rather than a failing test.
        """
        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        shared = np.array([0.4, -0.2, 0.3])
        params = replace(
            _params("dyncov_fit"), gamma_life=shared, gamma_trans=shared
        )
        table = abcd(data, params, pd.Timestamp("2007-06-30"))

        # Identical data and identical coefficients: the multipliers agree...
        assert np.array_equal(table["Ai"].to_numpy(), table["Ci"].to_numpy())
        # ...and their integrals do not, because the spans differ.
        assert not np.allclose(table["Bbar_i"], table["Dbar_i"])

    def test_and_the_first_period_is_zero_only_where_the_covariates_are(
        self, zeroed, apparel_trans
    ):
        """`DY-04` says `Bbar_i = Dbar_i = 0` at `i = 1`. Neither half is general.

        Under **zero** covariates `Dbar_i` is 0 at `i = 1` and `Bbar_i` is not.
        Under non-zero ones neither is -- which the first draft of the test
        above asserted, having carried the zero-covariate result across to a
        regime it does not hold in.
        """
        first_zeroed = zeroed[zeroed["i"] == 1]
        assert (first_zeroed["Dbar_i"] == 0).all()
        assert not (first_zeroed["Bbar_i"] == 0).all()

        data = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=NAMES, names_cov_trans=NAMES,
        )
        shared = np.array([0.4, -0.2, 0.3])
        table = abcd(
            data,
            replace(_params("dyncov_fit"), gamma_life=shared, gamma_trans=shared),
            pd.Timestamp("2007-06-30"),
        )
        first = table[table["i"] == 1]
        assert not (first["Dbar_i"] == 0).any()


class TestExtraCovariatePeriodsDoNotMoveTheLikelihood:
    """Spec DY-09, `absent`: "LL never evaluated on both a split and an unsplit
    build".

    The claim is that the likelihood is the same whether or not there are *more
    covariates than required*. It has to be: the walks stop at the estimation
    end, so a series running years past it describes periods no walk reaches.
    A likelihood that moved would mean the walks were reading past their own
    end -- silently, and only on data whose covariates happen to run long,
    which the apparel cohort's do.

    Bit-exact, since this is the same arithmetic over the same intervals.
    """

    def test_trimming_the_series_to_the_estimation_end_changes_nothing(
        self, apparel_trans
    ):
        from clvtools.pnbd.dyncov import log_likelihood

        full = load_apparel_dyn_cov()
        trimmed = full[full["Cov.Date"] <= pd.Timestamp("2007-01-31")]
        model = {"r": 1.9777, "alpha": 115.178, "s": 2.0127, "beta": 158.182}
        zero = [0.0, 0.0, 0.0]

        values = []
        for covariates in (full, trimmed):
            data = ClvDataDynCov(
                ClvData(apparel_trans, time_unit="week", estimation_split=104),
                covariates, names_cov_life=NAMES, names_cov_trans=NAMES,
            )
            values.append(log_likelihood(
                data.walks(), **model, gamma_life=zero, gamma_trans=zero
            ))
        assert values[0] == values[1]


class TestZeroCoefficientsAndZeroWindowsInPredict:
    """Spec `DY-11` and `DY-12`, both `absent`.

    `DY-11` is the nesting `X-04` asserts for the *static* covariate path,
    taken through the time-varying one, which shares none of its code: with
    every gamma at zero the covariates cannot move anything, so the answer must
    be the plain Pareto/NBD's. `DY-12` is `A1`'s zero-length window, which was
    fixed for the plain and static paths and never checked here. Backlog item
    36, round 6.
    """

    ZERO: ClassVar = np.zeros(3)

    @pytest.fixture(scope="class")
    def params(self):
        return PnbdDynCovParams(
            r=1.4490, alpha=48.6361, s=0.5613, beta=46.8844,
            gamma_life=self.ZERO, gamma_trans=self.ZERO,
            names_cov_life=NAMES, names_cov_trans=NAMES,
            log_likelihood=float("nan"), converged=True, n_customers=600,
        )

    @pytest.fixture(scope="class")
    def plain(self):
        from clvtools.pnbd.fit import PnbdParams

        return PnbdParams(
            r=1.4490, alpha=48.6361, s=0.5613, beta=46.8844,
            log_likelihood=float("nan"), converged=True, n_customers=600,
        )

    @pytest.fixture(scope="class")
    def both(self, holdout_data, apparel_trans, params, plain):
        without = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        return (
            predict(holdout_data, params, prediction_end=13),
            predict(without, plain, prediction_end=13),
        )

    @pytest.mark.parametrize("column", ["PAlive", "CET"])
    def test_zero_coefficients_recover_the_plain_prediction(self, both, column):
        """To 1e-13, over all 600 customers. Measured: 2.1e-14 at worst."""
        dyncov, plain = both
        np.testing.assert_allclose(
            dyncov[column].to_numpy(), plain[column].to_numpy(), atol=1e-12
        )

    def test_but_dect_is_not_dert_and_should_not_be(self, both):
        """The one column `DY-11`'s "after renaming" cannot mean by value.

        ``DERT`` discounts over an infinite horizon; ``DECT`` discounts each
        period of a *finite* one, because with covariates there is no infinite
        horizon to discount over -- the covariates are known only as far as they
        are given. The two are different quantities wearing similar names, and
        at gamma = 0 over thirteen weeks they differ by 0.387 at most. Asserting
        that they *disagree* is the useful half: a change that made ``DECT``
        silently fall back to the closed form would pass every other test here.
        """
        dyncov, plain = both
        gap = np.abs(dyncov["DECT"].to_numpy() - plain["DERT"].to_numpy())
        assert gap.max() > 0.1
        assert (dyncov["DECT"].to_numpy() < plain["DERT"].to_numpy()).all()

    def test_a_zero_length_window_gives_no_expected_transactions(
        self, holdout_data, params
    ):
        """`DY-12`, and the third path to answer rather than raise."""
        table = predict(holdout_data, params, prediction_end=0)
        assert (table["CET"] == 0).all()
        assert (table["period.length"] == 0).all()


class TestAStaticSeriesMakesTheFirstTransactionIrrelevant:
    """Spec `NC-07`, `absent`: what `first_transaction` is *for*.

    A prospective customer's covariate series is indexed from their first
    transaction, so moving that date slides the window over the series and the
    prediction moves with it. Unless the series is flat -- then every window
    sees the same values and the date cannot matter. That is the claim, and it
    is the one that says ``first_transaction`` selects a window rather than
    entering the arithmetic in its own right: a path that mixed the date into
    a coefficient would pass every varying-covariate test and fail this one.
    Backlog item 36, round 6.
    """

    @pytest.fixture(scope="class")
    def flat(self):
        """The apparel series with every covariate pinned to one value."""
        frame = load_apparel_dyn_cov().copy()
        for name in NAMES:
            frame[name] = 1.0
        return frame

    @pytest.fixture(scope="class")
    def params(self):
        return PnbdDynCovParams(
            r=1.4490, alpha=48.6361, s=0.5613, beta=46.8844,
            gamma_life=np.array([0.3, -0.2, 0.1]),
            gamma_trans=np.array([0.1, 0.2, -0.3]),
            names_cov_life=NAMES, names_cov_trans=NAMES,
            log_likelihood=float("nan"), converged=True, n_customers=600,
        )

    def test_the_prediction_does_not_move(self, flat, params):
        """Bit for bit across three dates fourteen months apart.

        Asserted as equality between them rather than against a constant: the
        claim is that the date makes no difference, and a constant would also
        pin the parameters, which are not what this is about.
        """
        one = flat[flat["Id"] == flat["Id"].iloc[0]]
        answers = [
            float(
                predict(
                    newcustomer_dynamic(7.89, one, one, first_transaction=date),
                    params,
                )
            )
            for date in ("2005-01-03", "2005-06-06", "2006-03-06")
        ]
        assert len(set(answers)) == 1
        assert answers[0] == pytest.approx(1.2228, abs=5e-4)

    def test_and_a_varying_series_does_move_it(self, params):
        """The control: without this the test above passes on a broken path.

        If ``first_transaction`` were ignored outright rather than made
        irrelevant by a flat series, the assertion above would still hold. So
        the same three dates over the *real* series must give more than one
        answer -- **two**, not three: over the 7.89 weeks this scenario spans,
        2005-01-03 and 2006-03-06 both see ``High.Season`` at zero throughout,
        while 2005-06-06 opens on a high-season week. Two identical answers out
        of three is the covariate agreeing with itself, not the date being
        ignored, and asserting three would have been asserting a coincidence.
        """
        frame = load_apparel_dyn_cov()
        one = frame[frame["Id"] == frame["Id"].iloc[0]]
        answers = {
            float(
                predict(
                    newcustomer_dynamic(7.89, one, one, first_transaction=date),
                    params,
                )
            )
            for date in ("2005-01-03", "2005-06-06", "2006-03-06")
        }
        assert len(answers) == 2
