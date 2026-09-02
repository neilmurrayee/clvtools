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
        remains and ``d1`` is 1. It is not 1 in general.
        """
        assert set(built["d1"]) == {1.0}


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
