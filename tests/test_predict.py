r"""S6.3 - the combined CLV prediction.

Two things are separated deliberately.

:class:`TestAgainstOracleWithPublishedParameters` feeds the oracle's own fitted
parameters into :func:`~clvtools.predict.predict` and requires the whole table
back to 1e-12. That isolates the prediction arithmetic from the estimation:
if it passes, every column is assembled exactly as CLVTools assembles it.

:class:`TestEndToEnd` runs the full pipeline -- fit, then predict -- and allows
1e-3. The looser bound is not slack in the prediction; it is the Pareto/NBD's
flat ridge, where this package's optimiser stops a few parts in 1e5 away from
``optimx``'s. That difference propagates into every predicted column and
nothing can remove it short of matching stopping rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv, fixture_json
from paper_values import (
    DISCOUNT_RATE_ANNUAL,
    HOLDOUT_ERRORS,
    NEWCUSTOMER_PERIODS,
    NEWCUSTOMER_SPENDING,
    NEWCUSTOMER_TOTAL,
    NEWCUSTOMER_TRANSACTIONS,
    PREDICT_FULL_HEAD,
    PREDICTION_PERIOD_FIRST,
    PREDICTION_PERIOD_LAST,
    PREDICTION_WEEKS,
)

from clvtools import ClvData, ClvDataStaticCov, load_apparel_trans
from clvtools.gg import GgParams, fit_gg
from clvtools.pnbd.fit import PnbdParams, fit_pnbd
from clvtools.predict import (
    DEFAULT_DISCOUNT_FACTOR,
    discount_factor,
    newcustomer,
    newcustomer_spending,
    newcustomer_static,
    predict,
)

DELTA = discount_factor(DISCOUNT_RATE_ANNUAL)

PREDICTED = [
    "PAlive", "CET", "DERT",
    "predicted.mean.spending", "predicted.period.spending", "predicted.CLV",
]


def _oracle_params(pnbd_fixture: str, gg_fixture: str):
    """Wrap the oracle's coefficients in the dataclasses ``predict`` expects."""
    pnbd = PnbdParams(
        **fixture_json(pnbd_fixture)["coefficients"],
        log_likelihood=float("nan"), converged=True, n_customers=600,
    )
    gg = GgParams(
        **fixture_json(gg_fixture)["coefficients"],
        log_likelihood=float("nan"), converged=True, n_customers=600,
    )
    return pnbd, gg


@pytest.fixture(scope="module")
def transactions():
    return load_apparel_trans()


@pytest.mark.oracle
class TestAgainstOracleWithPublishedParameters:
    """The prediction arithmetic, with estimation held fixed."""

    def test_holdout_table_matches(self, transactions):
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        got = predict(data, pnbd, gg)
        want = fixture_csv("predict_holdout").set_index("Id").loc[got.index]

        for column in PREDICTED:
            np.testing.assert_allclose(
                got[column], want[column], rtol=1e-12, err_msg=column
            )

    def test_full_table_matches(self, transactions):
        data = ClvData(transactions, time_unit="week")
        pnbd, gg = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        got = predict(data, pnbd, gg, prediction_end=95, continuous_discount_factor=DELTA)
        want = fixture_csv("predict_full").set_index("Id").loc[got.index]

        for column in PREDICTED:
            np.testing.assert_allclose(
                got[column], want[column], rtol=1e-11, err_msg=column
            )

    def test_actuals_match_exactly(self, transactions):
        """Counted from the transaction log, so these carry no model error."""
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        got = predict(data, pnbd, gg)
        want = fixture_csv("predict_holdout").set_index("Id").loc[got.index]

        np.testing.assert_array_equal(got["actual.x"], want["actual.x"])
        np.testing.assert_allclose(
            got["actual.period.spending"], want["actual.period.spending"], rtol=1e-12
        )

    def test_period_columns_match(self, transactions):
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        got = predict(data, pnbd, gg)
        want = fixture_csv("predict_holdout").set_index("Id").loc[got.index]

        assert (
            got["period.first"].dt.strftime("%Y-%m-%d") == want["period.first"]
        ).all()
        assert (
            got["period.last"].dt.strftime("%Y-%m-%d") == want["period.last"]
        ).all()
        np.testing.assert_allclose(
            got["period.length"], want["period.length"], rtol=1e-12
        )


@pytest.mark.paper
class TestAgainstThePaper:
    def test_prediction_window_matches(self, transactions):
        """S6.3.2 prints ``2010-12-21`` to ``2012-10-15``, 95 periods."""
        data = ClvData(transactions, time_unit="week")
        pnbd, gg = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        got = predict(data, pnbd, gg, prediction_end=PREDICTION_WEEKS,
                      continuous_discount_factor=DELTA)
        assert got["period.first"].iloc[0] == pd.Timestamp(PREDICTION_PERIOD_FIRST)
        assert got["period.last"].iloc[0] == pd.Timestamp(PREDICTION_PERIOD_LAST)
        assert got["period.length"].iloc[0] == PREDICTION_WEEKS

    def test_published_table_rows_match(self, transactions):
        """The ``head(dt.pred.full, 3)`` block of S6.3.2, all 18 values."""
        data = ClvData(transactions, time_unit="week")
        pnbd, gg = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        got = predict(data, pnbd, gg, prediction_end=PREDICTION_WEEKS,
                      continuous_discount_factor=DELTA)

        for customer, expected in PREDICT_FULL_HEAD.items():
            for column, value in expected.items():
                assert got.loc[customer, column] == pytest.approx(value, rel=1e-6), (
                    f"customer {customer}, {column}"
                )

    def test_holdout_error_metrics_match(self, transactions):
        r"""S6.3.1's evaluation block.

        The paper prints ``mae.cet = 2.039532`` and ``rmse.cet = 3.329395``.
        CLVTools 0.12.1 itself gives 2.03962 and 3.329425 on the same data --
        the fourth decimal moved between the version used for the paper and the
        current release. The comparison here is against the oracle's own numbers
        with the paper's as a sanity bound.
        """
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        got = predict(data, pnbd, gg)

        error = got["CET"] - got["actual.x"]
        mae = float(np.abs(error).mean())
        rmse = float(np.sqrt((error**2).mean()))
        assert mae == pytest.approx(HOLDOUT_ERRORS["mae.cet"], rel=1e-4)
        assert rmse == pytest.approx(HOLDOUT_ERRORS["rmse.cet"], rel=1e-4)

        spend_error = (
            got["predicted.period.spending"] - got["actual.period.spending"]
        )
        assert float(np.abs(spend_error).mean()) == pytest.approx(
            HOLDOUT_ERRORS["mae.total.spending"], rel=1e-4
        )
        assert float(np.sqrt((spend_error**2).mean())) == pytest.approx(
            HOLDOUT_ERRORS["rmse.total.spending"], rel=1e-3
        )


@pytest.mark.slow
class TestEndToEnd:
    """Fit and predict, with nothing taken from the oracle."""

    def test_full_pipeline_reproduces_the_published_table(self, transactions):
        data = ClvData(transactions, time_unit="week")
        cbs, spend = data.customer_summary(), data.spending_summary()
        pnbd = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
        gg = fit_gg(spend["x"], spend["Spending"])
        got = predict(data, pnbd, gg, prediction_end=PREDICTION_WEEKS,
                      continuous_discount_factor=DELTA)

        for customer, expected in PREDICT_FULL_HEAD.items():
            for column, value in expected.items():
                assert got.loc[customer, column] == pytest.approx(value, rel=1e-3), (
                    f"customer {customer}, {column}"
                )


class TestIdentities:
    """Relationships S6.3 states in words, which must hold exactly."""

    @staticmethod
    @pytest.fixture(scope="class")
    def table(transactions):
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        return predict(data, pnbd, gg)

    def test_clv_is_dert_times_mean_spending(self, table):
        """S6.3: "CLV [...] is calculated by multiplying DERT with the average
        spending per transaction"."""
        np.testing.assert_allclose(
            table["predicted.CLV"],
            table["DERT"] * table["predicted.mean.spending"],
            rtol=1e-15,
        )

    def test_period_spending_is_cet_times_mean_spending(self, table):
        """S6.3: "the total spending expected in the prediction period [...] is
        calculated by multiplying CET with the average spending per
        transaction"."""
        np.testing.assert_allclose(
            table["predicted.period.spending"],
            table["CET"] * table["predicted.mean.spending"],
            rtol=1e-15,
        )

    def test_palive_is_a_probability(self, table):
        assert ((table["PAlive"] >= 0) & (table["PAlive"] <= 1)).all()

    def test_every_prediction_is_non_negative(self, table):
        for column in PREDICTED:
            assert (table[column] >= 0).all(), column


class TestParameterEffects:
    r"""S6.3.2 on which argument moves which column."""

    @staticmethod
    @pytest.fixture(scope="class")
    def setup(transactions):
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        return (data, *_oracle_params("pnbd_nocov_fit", "gg_fit"))

    def test_palive_ignores_both_arguments(self, setup):
        """"PAlive is unaffected by both parameters as it describes customers
        at the end of the estimation period"."""
        data, pnbd, gg = setup
        base = predict(data, pnbd, gg, prediction_end=10)
        other = predict(data, pnbd, gg, prediction_end=200,
                        continuous_discount_factor=0.5)
        np.testing.assert_allclose(base["PAlive"], other["PAlive"], rtol=1e-15)

    def test_the_horizon_moves_cet_but_not_dert(self, setup):
        """"the prediction horizon is only considered for metrics CET and
        predicted.period.spending"."""
        data, pnbd, gg = setup
        short = predict(data, pnbd, gg, prediction_end=10)
        long = predict(data, pnbd, gg, prediction_end=200)
        assert (long["CET"] > short["CET"]).all()
        np.testing.assert_allclose(short["DERT"], long["DERT"], rtol=1e-15)

    def test_the_discount_factor_moves_dert_but_not_cet(self, setup):
        """"the value of continuous.discount.factor only affects DERT and
        predicted.CLV"."""
        data, pnbd, gg = setup
        light = predict(data, pnbd, gg, prediction_end=52,
                        continuous_discount_factor=discount_factor(0.02))
        heavy = predict(data, pnbd, gg, prediction_end=52,
                        continuous_discount_factor=discount_factor(0.30))
        assert (heavy["DERT"] < light["DERT"]).all()
        np.testing.assert_allclose(light["CET"], heavy["CET"], rtol=1e-15)


class TestSpendingIsOptional:
    def test_omitting_the_spending_model_omits_its_columns(self, transactions):
        """``predict.spending = FALSE`` in S6.3.1."""
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, _ = _oracle_params("pnbd_nocov_fit", "gg_fit")
        got = predict(data, pnbd)
        assert "predicted.CLV" not in got.columns
        assert "predicted.mean.spending" not in got.columns
        assert {"PAlive", "CET", "DERT"} <= set(got.columns)

    def test_spending_requires_a_price_column(self, transactions):
        data = ClvData(
            transactions[["Id", "Date"]], time_unit="week", estimation_split=104
        )
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        with pytest.raises(ValueError, match="no Price column"):
            predict(data, pnbd, gg)


class TestDiscountFactor:
    def test_matches_the_papers_formula(self):
        r"""S6.3.2: :math:`\delta_{52} = \ln(1.075)/52`."""
        assert discount_factor(0.075) == pytest.approx(np.log(1.075) / 52)
        assert discount_factor(0.10, "day") == pytest.approx(np.log(1.10) / 365)

    def test_clvtools_default_is_an_unscaled_annual_rate(self):
        r"""The trap S6.3.2 warns about, recorded so it cannot be forgotten.

        CLVTools defaults to :math:`\ln(1.1)`, applied per *period*. With weekly
        data that is 52 times heavier than a 10% annual rate.
        """
        assert pytest.approx(np.log(1.1)) == DEFAULT_DISCOUNT_FACTOR
        assert 50 * discount_factor(0.10) < DEFAULT_DISCOUNT_FACTOR

    def test_rejects_an_unknown_time_unit(self):
        with pytest.raises(ValueError, match="time_unit must be one of"):
            discount_factor(0.075, "fortnight")

    def test_rejects_an_impossible_rate(self):
        with pytest.raises(ValueError, match="must exceed -1"):
            discount_factor(-1.5)


class TestPredictionEnd:
    @staticmethod
    @pytest.fixture(scope="class")
    def full(transactions):
        data = ClvData(transactions, time_unit="week")
        return (data, *_oracle_params("pnbd_nocov_fit_full", "gg_fit_full"))

    def test_a_date_and_a_period_count_agree(self, full):
        data, pnbd, gg = full
        by_count = predict(data, pnbd, gg, prediction_end=95)
        by_date = predict(data, pnbd, gg, prediction_end=PREDICTION_PERIOD_LAST)
        np.testing.assert_allclose(by_count["CET"], by_date["CET"], rtol=1e-15)

    def test_is_required_without_a_holdout_period(self, full):
        """S6.3.2: "the argument prediction.end must be given"."""
        data, pnbd, gg = full
        with pytest.raises(ValueError, match="prediction_end is required"):
            predict(data, pnbd, gg)

    def test_defaults_to_the_holdout_end(self, transactions):
        """S6.3: "the prediction is made by default until the end of the
        holdout period"."""
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        got = predict(data, pnbd, gg)
        assert got["period.last"].iloc[0] == data.data_end

    def test_a_zero_length_window_predicts_zero_rather_than_raising(self, full):
        """CLVTools returns a zero-length window with ``CET = 0``; this raised.

        Predicting over no time is a well-defined question with the answer
        zero, and R gives it: ``period.length = 0``, ``CET = 0``, the window
        ending on the estimation end. Finding A1 of ``docs/spec-audit.md``,
        spec PR-05. Checked against CLVTools 0.12.1 rather than assumed.
        """
        data, pnbd, gg = full
        got = predict(data, pnbd, gg, prediction_end=0)
        assert len(got) == 600
        assert float(got["period.length"].iloc[0]) == 0.0
        assert float(got["CET"].sum()) == 0.0
        assert got["period.last"].iloc[0] == data.estimation_end

    def test_a_negative_horizon_is_still_refused(self, full):
        data, pnbd, gg = full
        with pytest.raises(ValueError, match="negative number of periods"):
            predict(data, pnbd, gg, prediction_end=-1)

    def test_rejects_a_window_that_ends_before_it_starts(self, transactions):
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        with pytest.raises(ValueError, match="before the estimation period"):
            predict(data, pnbd, gg, prediction_end="2005-06-01")

    def test_predicting_past_the_holdout_drops_the_actuals(self, transactions):
        """S6.3: actuals are reported only "if the predictions are made no
        further than the end of the holdout period"."""
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        within = predict(data, pnbd, gg, prediction_end=52)
        beyond = predict(data, pnbd, gg, prediction_end=400)
        assert "actual.x" in within.columns
        assert "actual.x" not in beyond.columns


class TestNewData:
    def test_parameters_can_be_applied_to_other_customers(self, transactions):
        """S6.3.1's ``newdata``: "to use the parameters of the estimated models
        to make predictions for a different set of customers"."""
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        subset = transactions[transactions["Id"].isin(["1", "10", "100"])]
        data = ClvData(subset, time_unit="week", estimation_split=104)
        got = predict(data, pnbd, gg)

        assert list(got.index) == ["1", "10", "100"]
        full = ClvData(transactions, time_unit="week", estimation_split=104)
        want = predict(full, pnbd, gg).loc[["1", "10", "100"]]
        np.testing.assert_allclose(got["PAlive"], want["PAlive"], rtol=1e-12)


@pytest.mark.oracle
class TestOtherFamilies:
    """Table 4's other two families, which report no DERT.

    CLVTools returns ``CET`` and ``PAlive`` for the BG/NBD and the GGom/NBD but
    neither ``DERT`` nor ``predicted.CLV``: neither model has a closed form for
    the discounted expected residual transactions. Predicting at the oracle's
    own coefficients keeps estimation out of the comparison.
    """

    @staticmethod
    def _predicted(transactions, family, name):
        from clvtools.gg import GgParams

        data = ClvData(transactions, time_unit="week", estimation_split=104)
        coefficients = fixture_json(f"predict_{name}_coefficients")
        params = getattr(family, f"{name.capitalize()}Params")(
            **coefficients, log_likelihood=float("nan"), converged=True,
            n_customers=600,
        )
        spending = GgParams(
            **fixture_json("gg_fit")["coefficients"],
            log_likelihood=float("nan"), converged=True, n_customers=600,
        )
        return predict(data, params, spending)

    # The GGom/NBD's CET is an integral this package evaluates numerically and
    # CLVTools evaluates in C++, which is where its looser tolerance comes from.
    @pytest.mark.parametrize("name,rtol", [("bgnbd", 1e-9), ("ggomnbd", 1e-6)])
    def test_table_matches(self, transactions, name, rtol):
        import clvtools

        family = getattr(clvtools, name)
        got = self._predicted(transactions, family, name)
        want = fixture_csv(f"predict_{name}").set_index("Id")
        assert list(got.index) == list(want.index)
        for column in ("PAlive", "CET", "predicted.mean.spending",
                       "predicted.period.spending", "actual.x"):
            np.testing.assert_allclose(
                got[column].to_numpy(), want[column].to_numpy(), rtol=rtol,
                err_msg=column,
            )

    @pytest.mark.parametrize("name", ["bgnbd", "ggomnbd"])
    def test_no_discounted_columns(self, transactions, name):
        import clvtools

        got = self._predicted(transactions, getattr(clvtools, name), name)
        assert "DERT" not in got.columns
        assert "predicted.CLV" not in got.columns
        # The same columns as CLVTools, though not in its order: it puts the
        # spending columns before CET and PAlive for these two families and
        # after them for the Pareto/NBD. One order for all three reads better.
        want = fixture_csv(f"predict_{name}")
        assert set(got.reset_index().columns) == set(want.columns)

    def test_an_unknown_model_is_refused(self, transactions):
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        with pytest.raises(TypeError, match="no prediction expressions"):
            predict(data, object())


@pytest.mark.oracle
class TestCorrelatedPredictsLikeTheIndependentModel:
    """The Sarmanov correlation enters estimation only, not prediction.

    CLVTools predicts from a correlated fit with the *plain* ``PAlive`` and
    ``CET``, at the fitted (r, alpha, s, beta). Checked in R against the
    internal per-customer entry points, the difference is exactly zero -- not
    small -- so ``PnbdCorrelatedParams.as_dict()`` dropping ``m`` for the
    prediction expressions is right rather than an approximation.
    """

    @staticmethod
    def _correlated():
        from clvtools.pnbd.correlation import PnbdCorrelatedParams

        want = fixture_json("correlated_predict")
        coefficients = want["coefficients"]
        return want, PnbdCorrelatedParams(
            r=coefficients["r"], alpha=coefficients["alpha"],
            s=coefficients["s"], beta=coefficients["beta"],
            m=0.0,  # replaced below; prediction never reads it
            log_likelihood=want["logLik"], converged=True, n_customers=600,
        )

    def test_matches_the_oracles_correlated_prediction(self, transactions):
        want, params = self._correlated()
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        table = predict(data, params)
        np.testing.assert_allclose(
            table["PAlive"].to_numpy()[:5], np.array(want["palive.head"]),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            table["CET"].to_numpy()[:5], np.array(want["cet.head"]), rtol=1e-9,
        )

    def test_the_correlation_parameter_does_not_enter(self, transactions):
        """Two fits differing only in ``m`` predict identically."""
        from clvtools.pnbd.correlation import PnbdCorrelatedParams

        _, params = self._correlated()
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        other = PnbdCorrelatedParams(
            r=params.r, alpha=params.alpha, s=params.s, beta=params.beta,
            m=0.5, log_likelihood=params.log_likelihood, converged=True,
            n_customers=600,
        )
        pd.testing.assert_frame_equal(predict(data, params), predict(data, other))


@pytest.mark.oracle
class TestProspectiveCustomers:
    """S6.3.4's ``newcustomer()`` family."""

    @staticmethod
    def _static_fit():
        from clvtools.pnbd.staticcov import PnbdStaticCovParams

        want = fixture_json("newcustomer_static")
        c = want["coefficients"]
        return want, PnbdStaticCovParams(
            r=c["r"], alpha=c["alpha"], s=c["s"], beta=c["beta"],
            gamma_life=np.array([c["life.Gender"], c["life.Channel"]]),
            gamma_trans=np.array([c["trans.Gender"], c["trans.Channel"]]),
            names_cov_life=["Gender", "Channel"],
            names_cov_trans=["Gender", "Channel"],
            names_cov_constr=[],
            log_likelihood=float("nan"),
            unpenalised_log_likelihood=None,
            converged=True, n_customers=600,
        )

    def test_without_covariates_matches_the_oracle(self, transactions):
        want = fixture_json("newcustomer_static")
        pnbd, _ = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        got = predict(newcustomer(want["num.periods"]), pnbd)
        assert got == pytest.approx(want["nocov.transactions"], rel=1e-9)

    @pytest.mark.parametrize("gender,channel", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_each_covariate_scenario_matches_the_oracle(self, gender, channel):
        want, params = self._static_fit()
        key = f"gender{gender}.channel{channel}"
        covariates = {"Gender": gender, "Channel": channel}
        got = predict(
            newcustomer_static(want["num.periods"], covariates, covariates),
            params,
        )
        assert got == pytest.approx(want[key], rel=1e-9)

    def test_covariates_separate_the_scenarios(self):
        """S6.3.4's "region A versus region B" comparison.

        This used to read the four values out of the *fixture* and assert that
        they differ from each other -- a statement about CLVTools' output, true
        no matter what this package computed, and it would have passed with
        ``predict`` deleted. Finding B1 of ``docs/spec-audit.md``. It now
        predicts the four scenarios here and asserts they are distinct, which
        is what "the covariates separate the scenarios" means.

        The test above already checks each against the oracle one at a time;
        what this adds is that the *spread* survives, which is the property
        S6.3.4 asks a covariate model for.
        """
        want, params = self._static_fit()
        got = [
            predict(
                newcustomer_static(
                    want["num.periods"],
                    {"Gender": g, "Channel": c},
                    {"Gender": g, "Channel": c},
                ),
                params,
            )
            for g in (0, 1) for c in (0, 1)
        ]
        assert len(set(got)) == 4, got
        # Not merely distinct: far enough apart to be a difference a reader
        # would act on. Measured, the closest pair is 0.0376 transactions over
        # the horizon apart and the widest 0.55, so 0.03 is a floor under the
        # measurement rather than a guess at one.
        assert min(abs(a - b) for a in got for b in got if a != b) > 0.03

    def test_a_covariate_model_refuses_a_plain_new_customer(self):
        _, params = self._static_fit()
        with pytest.raises(TypeError, match="newcustomer_static"):
            predict(newcustomer(52), params)

    def test_a_plain_model_refuses_covariate_values(self, transactions):
        pnbd, _ = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        with pytest.raises(TypeError, match="covariate model"):
            predict(newcustomer_static(52, {"Gender": 0}, {"Gender": 0}), pnbd)

    def test_missing_covariate_values_are_named(self):
        _, params = self._static_fit()
        with pytest.raises(ValueError, match="Channel"):
            predict(newcustomer_static(52, {"Gender": 0}, {"Gender": 0}), params)

    def test_spending_needs_a_spending_model(self, transactions):
        pnbd, _ = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        with pytest.raises(TypeError, match="spending model"):
            predict(newcustomer_spending(), pnbd)

    def test_a_zero_horizon_is_the_one_purchase_that_defines_them(self):
        """R returns 1 for ``newcustomer(0)``; this raised. Spec NC-02.

        S6.3.4 adds one "to account for all transactions that a prospective
        customer will make, including the first one", so over zero periods a
        prospective customer makes exactly that one and no more. A well-defined
        limit rather than an error.
        """
        pnbd, _ = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        assert predict(newcustomer(0), pnbd) == pytest.approx(1.0)

    @pytest.mark.parametrize("periods", [-1, -0.5])
    def test_a_negative_horizon_is_still_refused(self, periods):
        with pytest.raises(ValueError, match="must not be negative"):
            newcustomer(periods)
        with pytest.raises(ValueError, match="must not be negative"):
            newcustomer_static(periods, {}, {})

    def test_a_horizon_that_is_not_a_number_says_so(self):
        """Spec NC-13: "``num.periods`` must be numeric and ``>= 0``".

        A string reached ``<`` directly and raised Python's own "'<' not
        supported between instances of 'str' and 'int'", which names neither
        the argument nor the requirement. CLVTools 0.12.1, asked directly,
        answers "num.periods has to be numeric!" -- so ``"52"`` is refused
        rather than coerced.
        """
        with pytest.raises(TypeError, match="num_periods must be a number"):
            newcustomer("52")
        with pytest.raises(TypeError, match="num_periods must be a number"):
            newcustomer_static(None, {}, {})

    def test_a_nan_horizon_is_refused_rather_than_propagated(self):
        """The other half of NC-13, and the worse half: ``nan < 0`` is
        ``False``, so a ``NaN`` passed the negativity check and became a
        ``NaN`` prediction several frames away from its cause. R gives ``NA``
        the same answer it gives a string."""
        with pytest.raises(ValueError, match="got NaN"):
            newcustomer(float("nan"))

    def test_a_covariate_this_fit_does_not_carry_is_refused(self):
        """NC-13's "covariate data must have the right format".

        An unknown name used to be dropped, so a typo returned a plausible
        number computed from the covariates that *were* recognised. CLVTools
        0.12.1 refuses it -- "The Lifetime covariate data has to contain
        exactly the following columns: Gender, Channel!" -- and *exactly* is
        the operative word: both directions are errors there. They have
        different messages here because they are different mistakes.
        """
        _, params = self._static_fit()
        scenario = {"Gender": 0, "Channel": 1, "Gendre": 1}
        with pytest.raises(ValueError, match="not covariates of this fit"):
            predict(newcustomer_static(52, scenario, scenario), params)

    @pytest.mark.paper
    def test_reproduces_the_printed_totals(self, transactions):
        """S6.3.4: 2.218635 transactions, 39.1372 per order, 86.83115 total."""
        pnbd, _ = _oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        gg = GgParams(
            **fixture_json("gg_fit_full_with_first")["coefficients"],
            log_likelihood=float("nan"), converged=True, n_customers=600,
        )
        transactions_predicted = predict(
            newcustomer(NEWCUSTOMER_PERIODS), pnbd
        )
        spending = predict(newcustomer_spending(), gg)
        assert transactions_predicted == pytest.approx(
            NEWCUSTOMER_TRANSACTIONS, abs=5e-7
        )
        assert spending == pytest.approx(NEWCUSTOMER_SPENDING, abs=5e-5)
        assert transactions_predicted * spending == pytest.approx(
            NEWCUSTOMER_TOTAL, abs=5e-5
        )


@pytest.mark.oracle
class TestCovariatePredictionForTheOtherFamilies:
    """Table 4 gives time-invariant covariates to all three families.

    Each family builds its per-customer rates differently -- the BG/NBD scales
    both of its beta parameters by ``exp(+gamma'x)`` where every other family's
    rate parameter takes a negative exponent -- so each one's covariate
    prediction is checked separately.
    """

    @staticmethod
    def _fitted(name, coefficients):
        import numpy as np

        import clvtools

        family = getattr(clvtools, name)
        cls = getattr(family, f"{name.capitalize()}StaticCovParams")
        model = {
            "bgnbd": ["r", "alpha", "a", "b"],
            "ggomnbd": ["r", "alpha", "b", "s", "beta"],
        }[name]
        from clvtools._staticcov import StaticCovResult

        covariates = StaticCovResult(
            model=np.array([coefficients[n] for n in model]),
            gamma_life=np.array(
                [coefficients["life.Gender"], coefficients["life.Channel"]]
            ),
            gamma_trans=np.array(
                [coefficients["trans.Gender"], coefficients["trans.Channel"]]
            ),
            names_cov_life=["Gender", "Channel"],
            names_cov_trans=["Gender", "Channel"],
            names_cov_constr=[],
            log_likelihood=float("nan"),
            unpenalised_log_likelihood=float("nan"),
            converged=True, n_customers=600,
        )
        return cls(
            **{n: coefficients[n] for n in model}, covariates=covariates
        )

    @pytest.mark.parametrize("name,rtol", [("bgnbd", 1e-9), ("ggomnbd", 1e-6)])
    def test_table_matches(self, transactions, apparel_static_cov, name, rtol):
        from clvtools.gg import GgParams

        data = ClvDataStaticCov(
            ClvData(transactions, time_unit="week", estimation_split=104),
            apparel_static_cov,
            names_cov_life=["Gender", "Channel"],
            names_cov_trans=["Gender", "Channel"],
        )
        params = self._fitted(
            name, fixture_json(f"predict_{name}_staticcov_coefficients")
        )
        spending = GgParams(
            **fixture_json("gg_fit")["coefficients"],
            log_likelihood=float("nan"), converged=True, n_customers=600,
        )
        got = predict(data, params, spending)
        want = fixture_csv(f"predict_{name}_staticcov").set_index("Id")
        for column in ("PAlive", "CET", "predicted.period.spending"):
            np.testing.assert_allclose(
                got[column].to_numpy(), want[column].to_numpy(), rtol=rtol,
                err_msg=column,
            )

    def test_covariate_data_is_required(self, transactions):
        params = self._fitted(
            "bgnbd", fixture_json("predict_bgnbd_staticcov_coefficients")
        )
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        with pytest.raises(TypeError, match="needs covariate data"):
            predict(data, params)

    @pytest.mark.parametrize("name", ["bgnbd", "ggomnbd"])
    def test_a_prospective_customer_scenario_matches(self, name):
        """``newcustomer.static()`` on the other two families, S6.3.4."""
        want = fixture_json(f"newcustomer_static_{name}")
        params = self._fitted(name, want["coefficients"])
        for key, (gender, channel) in {
            "gender0.channel0": (0, 0), "gender1.channel1": (1, 1),
        }.items():
            covariates = {"Gender": gender, "Channel": channel}
            got = predict(
                newcustomer_static(want["num.periods"], covariates, covariates),
                params,
            )
            assert got == pytest.approx(want[key], rel=1e-9), key


class TestArgumentsThatCannotBeUsedAreRejected:
    """Finding 12: ``predict()`` used to ignore two of its four arguments.

    ``predict(newcustomer(52), fit, gg, prediction_end=99)`` returned a
    transaction count, silently dropping both the spending model and the
    horizon -- neither of which has anywhere to go for a scenario that returns
    a single number.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def fits(transactions):
        from clvtools import ClvData, gg, latent_attrition, pnbd, spending

        data = ClvData(transactions, time_unit="week", estimation_split=104)
        return (
            latent_attrition(family=pnbd, data=data, hessian=False),
            spending(family=gg, data=data, hessian=False),
        )

    def test_a_spending_model_with_a_prospective_customer_raises(self, fits):
        from clvtools import newcustomer, predict

        fit, gg_fit = fits
        with pytest.raises(ValueError, match="spending_params"):
            predict(newcustomer(52), fit, gg_fit)

    def test_a_prediction_end_with_a_prospective_customer_raises(self, fits):
        from clvtools import newcustomer, predict

        with pytest.raises(ValueError, match="prediction_end"):
            predict(newcustomer(52), fits[0], prediction_end=99)

    def test_the_scenario_itself_still_works(self, fits):
        from clvtools import newcustomer, predict

        assert predict(newcustomer(52), fits[0]) > 0


class TestTheDiscountFactorRange:
    """A3: the range was ``(0, inf)`` where CLVTools admits ``[0, 1)``.

    Zero was refused and 100 was accepted, returning a number for a per-period
    discount rate of 10,000%. The parameter carries CLVTools' exact semantics
    -- ``DEFAULT_DISCOUNT_FACTOR`` is ``log(1.1)`` -- so its range transfers
    with it. Spec PR-11.

    Every boundary here was checked against R rather than reasoned about, and
    the odd one is why: **CLVTools accepts zero and returns ``Inf``**, which is
    right -- with no discounting the residual value of a customer who may never
    die does not converge -- and is not what I would have chosen unprompted. It
    errors at 1.0 with "needs to be in the interval [0,1)".
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def fits(transactions):
        from clvtools import ClvData, gg, latent_attrition, pnbd, spending

        data = ClvData(transactions, time_unit="week", estimation_split=104)
        return data, (
            latent_attrition(family=pnbd, data=data, hessian=False),
            spending(family=gg, data=data, hessian=False),
        )

    @pytest.mark.parametrize("factor", [1.0, 1.5, 100.0, -0.1])
    def test_outside_the_interval_is_refused(self, fits, factor):
        data, (fit, spend) = fits
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            predict(data, fit, spend, continuous_discount_factor=factor)

    def test_zero_is_accepted_and_diverges_as_it_does_in_r(self, fits):
        """R: ``sum(DERT) = Inf``. Undiscounted residual value of an
        immortal customer is unbounded, and saying so beats refusing."""
        data, (fit, spend) = fits
        got = predict(data, fit, spend, continuous_discount_factor=0.0)
        assert np.isinf(got["DERT"]).all()

    @pytest.mark.oracle
    def test_a_middling_factor_matches_r(self, fits):
        """``sum(DERT) = 18.2282`` at 0.5, from CLVTools 0.12.1."""
        data, (fit, spend) = fits
        got = predict(data, fit, spend, continuous_discount_factor=0.5)
        assert float(got["DERT"].sum()) == pytest.approx(18.2282, abs=5e-4)


class TestAFractionalPredictionEndIsNotTruncated:
    """Spec T-22, and a divergence the README recorded without a test.

    ``prediction.end = 14.4`` gives CLVTools a **14**-period window and a
    warning -- "may not indicate partial periods. Digits after the decimal
    point are cut off" -- while this package predicts 14.4 periods, ending two
    days later. Both are defensible and they are not the same, and the README's
    findings say so.

    What was missing is the other half of this repository's own rule: *"Where
    the paper misprints an equation or CLVTools stops at a worse optimum, that
    is pinned by a test and recorded in the README's Findings section. Add to
    both."* It was in the README and nowhere else, so nothing would have
    noticed the behaviour reverting to R's. Backlog item 34, round 5.
    """

    @pytest.fixture(scope="class")
    def fitted(self, apparel_trans):
        from clvtools.pnbd import fit_pnbd

        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        cbs = data.customer_summary()
        return data, fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)

    def test_the_window_keeps_its_fraction(self, fitted):
        data, params = fitted
        table = predict(data, params, prediction_end=14.4)
        assert table["period.length"].iloc[0] == pytest.approx(14.4)

    def test_which_is_not_what_truncating_to_14_would_give(self, fitted):
        """The divergence itself: R's answer is reachable, and different."""
        data, params = fitted
        fractional = predict(data, params, prediction_end=14.4)
        truncated = predict(data, params, prediction_end=14)
        assert truncated["period.length"].iloc[0] == pytest.approx(14.0)
        assert (
            fractional["period.last"].iloc[0] > truncated["period.last"].iloc[0]
        )
        # Two days and change, which is 0.4 of a week -- so the fraction is
        # carried into the date rather than rounded anywhere along the way.
        delta = fractional["period.last"].iloc[0] - truncated["period.last"].iloc[0]
        assert delta.total_seconds() / 86400 == pytest.approx(0.4 * 7, abs=1e-6)

    def test_and_the_prediction_moves_with_it(self, fitted):
        """A longer window has to predict more transactions, or the extra
        0.4 of a period is being carried in the dates and dropped in the maths.
        """
        data, params = fitted
        fractional = predict(data, params, prediction_end=14.4)
        truncated = predict(data, params, prediction_end=14)
        assert (fractional["CET"] > truncated["CET"]).all()


class TestDataEndAndShortHorizons:
    """Spec T-15 and T-19, both `weak` and both merely untested.

    `T-15` gives exact dates: with ``estimation.split = None`` and
    ``data.end = "1998-07-15"``, ``predict(prediction.end = "1998-07-30")``
    returns ``period.first == "1998-07-16"`` and ``period.last ==
    "1998-07-30"``. The audit noted the port already emits both and pinned
    neither -- so the +1 day rule was carried by a paper example alone, in a
    configuration the paper never uses.

    `T-19` asks that the period table hold for ``prediction.end`` as a number
    *and* as a date, over a single period and over two. One-period horizons are
    where an off-by-one in the grid shows up, and nothing reached them.

    Neither turned up a defect. Backlog item 34, round 5.
    """

    @pytest.fixture(scope="class")
    def cdnow_fit(self):
        from clvtools import load_cdnow
        from clvtools.pnbd import fit_pnbd

        data = ClvData(load_cdnow(), time_unit="week", estimation_split=39)
        cbs = data.customer_summary()
        return data, fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)

    def test_data_end_moves_the_prediction_window(self):
        """T-15, to the day and against the spec's own two dates."""
        from clvtools import load_cdnow
        from clvtools.pnbd import fit_pnbd

        data = ClvData(load_cdnow(), time_unit="week", data_end="1998-07-15")
        assert not data.has_holdout
        cbs = data.customer_summary()
        params = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
        table = predict(data, params, prediction_end="1998-07-30")
        assert table["period.first"].iloc[0] == pd.Timestamp("1998-07-16")
        assert table["period.last"].iloc[0] == pd.Timestamp("1998-07-30")

    @pytest.mark.parametrize("periods", [1, 2])
    def test_a_one_or_two_period_horizon_is_that_many_periods(
        self, cdnow_fit, periods
    ):
        """T-19's numeric form, at the horizons an off-by-one would show in."""
        data, params = cdnow_fit
        table = predict(data, params, prediction_end=periods)
        assert table["period.length"].iloc[0] == pytest.approx(float(periods))
        span = table["period.last"].iloc[0] - table["period.first"].iloc[0]
        # Inclusive of both ends, so `n` periods span `7n - 1` days weekly.
        assert span == pd.Timedelta(days=7 * periods - 1)

    def test_and_a_date_gives_the_same_window_as_the_count_it_equals(
        self, cdnow_fit
    ):
        """T-19's other half: the two spellings agree where they should.

        A date two whole periods past the estimation end has to produce the
        same window as ``prediction_end=2``, or one of the two forms is
        counting from somewhere else.
        """
        data, params = cdnow_fit
        by_count = predict(data, params, prediction_end=2)
        two_weeks_on = data.estimation_end + pd.Timedelta(days=14)
        by_date = predict(data, params, prediction_end=two_weeks_on)
        assert by_date["period.first"].iloc[0] == by_count["period.first"].iloc[0]
        assert by_date["period.last"].iloc[0] == by_count["period.last"].iloc[0]
        assert by_date["period.length"].iloc[0] == pytest.approx(
            by_count["period.length"].iloc[0]
        )


class TestPredictArgumentsAndCovariateNames:
    """Spec PR-02, PR-13 and PR-15.

    `PR-13`'s scenario half was closed by item 27 -- a covariate the fit does
    not carry now raises by name. The **data** half was not: applying a fit to
    covariate data whose columns are named differently surfaced as
    ``KeyError: "['Gender'] not in index"``, pandas' words for a question about
    two objects it has never seen together.

    `PR-15` asks that `prediction.end` be single, non-`NA` and of an allowed
    type. A `NaN` reached `int()` and came back as "cannot convert float NaN to
    integer"; a list reached `pd.Timestamp` and came back as "Cannot convert
    input [[1, 2]]". Both describe a conversion rather than the argument -- the
    same shape as `V-01`'s start value, which is the fourth instance this round.

    Backlog item 34, round 5.
    """

    @pytest.fixture(scope="class")
    def fitted(self, apparel_trans):
        from clvtools.pnbd import fit_pnbd

        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        cbs = data.customer_summary()
        return data, fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)

    def test_a_non_finite_horizon_names_itself(self, fitted):
        data, params = fitted
        with pytest.raises(ValueError, match="must be a finite number"):
            predict(data, params, prediction_end=float("nan"))

    @pytest.mark.parametrize("bad", [[1, 2], {"a": 1}, (1, 2)])
    def test_and_so_does_one_that_is_not_a_single_value(self, fitted, bad):
        data, params = fitted
        with pytest.raises(TypeError, match="a single date"):
            predict(data, params, prediction_end=bad)

    def test_the_horizons_that_were_always_fine_still_are(self, fitted):
        data, params = fitted
        assert len(predict(data, params, prediction_end=10)) == 600
        assert len(predict(data, params, prediction_end="2007-01-15")) == 600

    def test_covariate_data_named_differently_from_the_fit_says_so(
        self, apparel_trans
    ):
        """PR-13's data half: not `KeyError: "['Gender'] not in index"`."""
        from clvtools import ClvDataStaticCov, load_apparel_static_cov

        params = self._static_params()
        renamed = load_apparel_static_cov().rename(columns={"Gender": "Sex"})
        data = ClvDataStaticCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            renamed, names_cov_life=["Sex", "Channel"],
            names_cov_trans=["Sex", "Channel"],
        )
        with pytest.raises(ValueError, match="no lifetime covariate 'Gender'"):
            predict(data, params, prediction_end=10)

    def test_and_the_message_lists_what_the_data_does_carry(self, apparel_trans):
        from clvtools import ClvDataStaticCov, load_apparel_static_cov

        params = self._static_params()
        renamed = load_apparel_static_cov().rename(columns={"Gender": "Sex"})
        data = ClvDataStaticCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            renamed, names_cov_life=["Sex", "Channel"],
            names_cov_trans=["Sex", "Channel"],
        )
        with pytest.raises(ValueError, match="it carries Sex, Channel"):
            predict(data, params, prediction_end=10)

    @staticmethod
    def _static_params():
        from clvtools.pnbd.staticcov import PnbdStaticCovParams

        want = fixture_json("newcustomer_static")["coefficients"]
        return PnbdStaticCovParams(
            r=want["r"], alpha=want["alpha"], s=want["s"], beta=want["beta"],
            gamma_life=np.array([want["life.Gender"], want["life.Channel"]]),
            gamma_trans=np.array([want["trans.Gender"], want["trans.Channel"]]),
            names_cov_life=["Gender", "Channel"],
            names_cov_trans=["Gender", "Channel"],
            names_cov_constr=[], log_likelihood=float("nan"),
            unpenalised_log_likelihood=None, converged=True, n_customers=600,
        )

    def test_the_discount_factor_and_bootstrap_defaults(self):
        """PR-02. The `log(1.1)` default was pinned; the other two were not."""
        import inspect

        from clvtools import bootstrap

        assert inspect.signature(
            bootstrap.confidence_intervals
        ).parameters["level"].default == 0.9
        assert inspect.signature(
            bootstrap.bootstrap_apply
        ).parameters["num_boots"].default == 100
