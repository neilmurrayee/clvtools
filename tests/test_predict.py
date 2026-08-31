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
    NEWCUSTOMER_PERIODS,
    NEWCUSTOMER_SPENDING,
    NEWCUSTOMER_TOTAL,
    NEWCUSTOMER_TRANSACTIONS,
    HOLDOUT_ERRORS,
    PREDICTION_PERIOD_FIRST,
    PREDICTION_PERIOD_LAST,
    PREDICTION_WEEKS,
    PREDICT_FULL_HEAD,
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
        assert DEFAULT_DISCOUNT_FACTOR == pytest.approx(np.log(1.1))
        assert DEFAULT_DISCOUNT_FACTOR > 50 * discount_factor(0.10)

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

    def test_rejects_a_non_positive_horizon(self, full):
        data, pnbd, gg = full
        with pytest.raises(ValueError, match="positive number of periods"):
            predict(data, pnbd, gg, prediction_end=0)

    def test_rejects_a_window_that_ends_before_it_starts(self, transactions):
        data = ClvData(transactions, time_unit="week", estimation_split=104)
        pnbd, gg = _oracle_params("pnbd_nocov_fit", "gg_fit")
        with pytest.raises(ValueError, match="on or before the estimation period"):
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
        """S6.3.4's "region A versus region B" comparison."""
        want, _ = self._static_fit()
        scenarios = [want[f"gender{g}.channel{c}"]
                     for g in (0, 1) for c in (0, 1)]
        assert len(set(scenarios)) == 4

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

    @pytest.mark.parametrize("periods", [0, -1])
    def test_the_horizon_must_be_positive(self, periods):
        with pytest.raises(ValueError, match="strictly positive"):
            newcustomer(periods)
        with pytest.raises(ValueError, match="strictly positive"):
            newcustomer_static(periods, {}, {})

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
        with pytest.raises(ValueError, match="needs covariate data"):
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
