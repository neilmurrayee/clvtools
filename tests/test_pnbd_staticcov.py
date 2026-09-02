r"""S3.3 and S6.4.1 - the Pareto/NBD with time-invariant covariates.

The extension is checked three ways:

  * against the oracle's per-customer :math:`\alpha_i`, :math:`\beta_i`,
    likelihood, PAlive, CET and DERT at two parameter vectors;
  * against S6.4.1's printed coefficient table, standard errors and z-values;
  * against the model *without* covariates, which S3.3 says is nested inside
    it: "With covariate effects set to zero, we arrive at the standard model."

The nesting test is the one that would catch a sign error in
:func:`~clvtools.pnbd.staticcov.alpha_i`, which no amount of agreement at the
fitted optimum would reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv, fixture_json
from paper_values import (
    PNBD_STATIC_AIC,
    PNBD_STATIC_BIC,
    PNBD_STATIC_LL,
    PNBD_STATIC_MLE,
    PNBD_STATIC_SE,
    PNBD_STATIC_Z,
)

from clvtools import (
    ClvData,
    ClvDataStaticCov,
    load_apparel_static_cov,
    load_apparel_trans,
)
from clvtools.pnbd import log_likelihood_ind
from clvtools.pnbd.aggregate import (
    conditional_expected_transactions,
    discounted_expected_residual_transactions,
    probability_alive,
)
from clvtools.pnbd.staticcov import (
    alpha_i,
    beta_i,
    fit_pnbd_staticcov,
    log_likelihood,
    log_likelihood_staticcov_ind,
)

NAMES = ["Gender", "Channel"]
GRID = fixture_json("pnbd_staticcov_grid")
CASES = list(GRID["params"])
DELTA = float(np.log(1.075) / 52)


@pytest.fixture(scope="module")
def data():
    return ClvDataStaticCov(
        ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
        load_apparel_static_cov(),
        names_cov_life=NAMES,
        names_cov_trans=NAMES,
    )


@pytest.fixture(scope="module")
def inputs(data):
    cbs = data.customer_summary().set_index("Id")
    return (
        cbs.index,
        cbs["x"].to_numpy(),
        cbs["t_x"].to_numpy(),
        cbs["T"].to_numpy(),
        data.design_life(),
        data.design_trans(),
    )


@pytest.fixture(scope="module")
def fitted(data):
    return fit_pnbd_staticcov(data)


def _params(case):
    p = GRID["params"][case]
    return (
        p["model"],
        np.array(list(p["life"].values())),
        np.array(list(p["trans"].values())),
    )


class TestDesignMatrices:
    """S6.4: the covariate frame lined up one row per customer."""

    @pytest.mark.oracle
    @pytest.mark.parametrize("which", ["life", "trans"])
    def test_matches_the_oracles_design(self, data, inputs, which):
        ids = inputs[0]
        want = (
            fixture_csv(f"staticcov_design_{which}")
            .set_index("Id").loc[ids, NAMES].to_numpy(dtype=float)
        )
        got = getattr(data, f"design_{which}")()
        np.testing.assert_array_equal(got, want)

    def test_rows_follow_the_customer_summary_order(self, data, inputs):
        assert data.design_life().shape == (len(inputs[0]), 2)
        assert list(data.customers) == list(inputs[0])

    def test_categorical_columns_become_k_minus_one_dummies(self):
        """S6.4: "Categorical data is turned into k-1 dummy variables"."""
        trans = load_apparel_trans()
        ids = sorted(trans["Id"].unique())
        covariates = pd.DataFrame({
            "Id": ids,
            "Region": ["north", "south", "east"] * (len(ids) // 3),
        })
        data = ClvDataStaticCov(
            ClvData(trans, time_unit="week", estimation_split=104), covariates
        )
        # Three levels give two columns, not three.
        assert sorted(data.names_cov_life) == ["Region_north", "Region_south"]
        assert data.design_life().shape[1] == 2

    def test_rejects_covariates_missing_a_customer(self):
        trans = load_apparel_trans()
        partial = load_apparel_static_cov().iloc[:100]
        with pytest.raises(ValueError, match="missing 500 customers"):
            ClvDataStaticCov(
                ClvData(trans, time_unit="week", estimation_split=104), partial
            )

    def test_rejects_an_unknown_covariate_name(self, ):
        trans = load_apparel_trans()
        with pytest.raises(ValueError, match="life covariates not in the data"):
            ClvDataStaticCov(
                ClvData(trans, time_unit="week", estimation_split=104),
                load_apparel_static_cov(),
                names_cov_life=["Nonexistent"],
            )

    def test_rejects_covariate_data_without_an_id_column(self):
        trans = load_apparel_trans()
        with pytest.raises(ValueError, match="no 'Id' column"):
            ClvDataStaticCov(
                ClvData(trans, time_unit="week", estimation_split=104),
                pd.DataFrame({"Gender": [0, 1]}),
            )

    def test_the_two_processes_can_take_different_covariates(self, ):
        """S6.4: "The covariates for the transaction process and attrition
        process may differ"."""
        data = ClvDataStaticCov(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_static_cov(),
            names_cov_life=["Gender"],
            names_cov_trans=["Gender", "Channel"],
        )
        assert data.design_life().shape[1] == 1
        assert data.design_trans().shape[1] == 2


@pytest.mark.oracle
class TestAgainstOracle:
    @pytest.mark.parametrize("case", CASES)
    def test_per_customer_rates(self, inputs, case):
        ids, _, _, _, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]

        np.testing.assert_allclose(
            alpha_i(model["alpha"], g_trans, cov_trans), want["alpha.i"], rtol=1e-12
        )
        np.testing.assert_allclose(
            beta_i(model["beta"], g_life, cov_life), want["beta.i"], rtol=1e-12
        )

    @pytest.mark.parametrize("case", CASES)
    def test_individual_log_likelihood(self, inputs, case):
        ids, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]

        got = log_likelihood_staticcov_ind(
            x, t_x, T, model["r"], model["alpha"], model["s"], model["beta"],
            g_life, g_trans, cov_life, cov_trans,
        )
        np.testing.assert_allclose(got, want["LL.ind"], rtol=1e-11)

    @pytest.mark.parametrize("case", CASES)
    def test_sample_log_likelihood(self, inputs, case):
        _, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        got = log_likelihood(
            x, t_x, T, model["r"], model["alpha"], model["s"], model["beta"],
            g_life, g_trans, cov_life, cov_trans,
        )
        assert got == pytest.approx(GRID["LL.sum"][case], rel=1e-11)

    def test_weights_repeat_rows(self, inputs):
        """Row multiplicities, as the CBS compression uses."""
        _, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params("mle")
        args = (model["r"], model["alpha"], model["s"], model["beta"],
                g_life, g_trans, cov_life[:2], cov_trans[:2])
        each = log_likelihood_staticcov_ind(x[:2], t_x[:2], T[:2], *args)
        weighted = log_likelihood(
            x[:2], t_x[:2], T[:2], *args, weights=[2.0, 3.0]
        )
        assert weighted == pytest.approx(2 * each[0] + 3 * each[1])

    @pytest.mark.parametrize("case", CASES)
    def test_probability_alive(self, inputs, case):
        ids, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]
        got = probability_alive(
            x, t_x, T,
            r=model["r"], alpha=alpha_i(model["alpha"], g_trans, cov_trans),
            s=model["s"], beta=beta_i(model["beta"], g_life, cov_life),
        )
        np.testing.assert_allclose(got, want["PAlive"], rtol=1e-10)

    @pytest.mark.parametrize(
        "case", [c for c in CASES if GRID["params"][c]["model"]["s"] != 1.0]
    )
    def test_cet_and_dert(self, inputs, case):
        ids, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]
        rates = {
            "r": model["r"], "alpha": alpha_i(model["alpha"], g_trans, cov_trans),
            "s": model["s"], "beta": beta_i(model["beta"], g_life, cov_life),
        }
        np.testing.assert_allclose(
            conditional_expected_transactions(x, t_x, T, 52.0, **rates),
            want["CET"], rtol=1e-10,
        )
        np.testing.assert_allclose(
            discounted_expected_residual_transactions(x, t_x, T, DELTA, **rates),
            want["DERT"], rtol=1e-10,
        )

    def test_fit_reaches_the_oracles_optimum(self, fitted):
        want = fixture_json("pnbd_staticcov_fit")
        assert fitted.log_likelihood == pytest.approx(want["logLik"], abs=1e-4)
        assert fitted.log_likelihood >= want["logLik"] - 1e-6


class TestNesting:
    r"""S3.3: "With covariate effects set to zero, we arrive at the standard
    model." A sign error in the rate builders would break this and nothing
    else."""

    def test_zero_coefficients_reproduce_the_plain_likelihood(self, inputs):
        _, x, t_x, T, cov_life, cov_trans = inputs
        plain = log_likelihood_ind(x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844)
        nested = log_likelihood_staticcov_ind(
            x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844,
            gamma_life=np.zeros(2), gamma_trans=np.zeros(2),
            cov_life=cov_life, cov_trans=cov_trans,
        )
        np.testing.assert_allclose(nested, plain, rtol=1e-15)

    def test_all_zero_covariates_reproduce_the_plain_likelihood(self, inputs):
        """Non-zero coefficients on an all-zero design must also collapse."""
        _, x, t_x, T, _, _ = inputs
        zeros = np.zeros((x.size, 2))
        plain = log_likelihood_ind(x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844)
        nested = log_likelihood_staticcov_ind(
            x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844,
            gamma_life=[0.5, -0.3], gamma_trans=[-0.2, 0.9],
            cov_life=zeros, cov_trans=zeros,
        )
        np.testing.assert_allclose(nested, plain, rtol=1e-15)

    def test_the_covariate_fit_beats_the_plain_one(self, fitted):
        """More parameters, nested model: the likelihood cannot be lower."""
        plain = fixture_json("pnbd_nocov_fit")["logLik"]
        assert fitted.log_likelihood > plain

    def test_a_positive_transaction_coefficient_lowers_alpha(self):
        r"""S6.4.1 reads ``trans.Gender = 0.2859`` as a higher purchase rate;
        since :math:`\alpha` is a rate parameter that means a lower
        :math:`\alpha_i`."""
        covariates = np.array([[0.0], [1.0]])
        got = alpha_i(50.0, [0.2859], covariates)
        assert got[1] < got[0]

    def test_a_positive_lifetime_coefficient_lowers_beta(self):
        """``life.Channel = 0.7907``: offline customers "drop out more
        quickly", a higher attrition rate, so a lower beta."""
        covariates = np.array([[0.0], [1.0]])
        got = beta_i(50.0, [0.7907], covariates)
        assert got[1] < got[0]


@pytest.mark.slow
@pytest.mark.paper
class TestAgainstThePaper:
    def test_log_likelihood_matches(self, fitted):
        assert fitted.log_likelihood == pytest.approx(PNBD_STATIC_LL, abs=1e-3)

    def test_aic_and_bic_match(self, fitted):
        assert fitted.aic == pytest.approx(PNBD_STATIC_AIC, abs=1e-2)
        assert fitted.bic == pytest.approx(PNBD_STATIC_BIC, abs=1e-2)

    def test_covariate_coefficients_match(self, fitted):
        """The four the paper interprets, to the precision it prints them."""
        got = fitted.coefficients
        for name in ("life.Gender", "life.Channel", "trans.Gender", "trans.Channel"):
            assert got[name] == pytest.approx(PNBD_STATIC_MLE[name], abs=2e-3), name

    def test_base_parameters_match_approximately(self, fitted):
        """Looser, because the eight-parameter likelihood ridge is broader.

        The attained log-likelihood is checked separately and exceeds the
        published one, so this bound is about where the optimiser stopped, not
        about the model.
        """
        got = fitted.coefficients
        for name in ("r", "alpha", "s", "beta"):
            assert got[name] == pytest.approx(PNBD_STATIC_MLE[name], rel=5e-3), name

    def test_standard_errors_match(self, fitted):
        got = fitted.standard_errors()
        for name in ("life.Gender", "life.Channel", "trans.Gender", "trans.Channel"):
            assert got[name] == pytest.approx(PNBD_STATIC_SE[name], rel=5e-2), name

    def test_z_values_match(self, fitted):
        """S6.4.1's ``z-val`` column, and the significance it reports."""
        table = fitted.summary()
        for name, want in PNBD_STATIC_Z.items():
            assert table.loc[name, "z-val"] == pytest.approx(want, rel=5e-2), name
        # All four are flagged significant at 5% in the printed table.
        for name in PNBD_STATIC_Z:
            assert table.loc[name, "Pr(>|z|)"] < 0.05

    def test_base_parameters_have_no_z_value(self, fitted):
        r"""S6.4.1: "a null hypothesis of :math:`\theta = 0` lies outside the
        admissible parameter space", so those cells are blank."""
        table = fitted.summary()
        for name in ("r", "alpha", "s", "beta"):
            assert np.isnan(table.loc[name, "z-val"])
            assert np.isnan(table.loc[name, "Pr(>|z|)"])


@pytest.mark.slow
class TestFitting:
    def test_converges(self, fitted):
        """S6.4.1 reports ``KKT 1 TRUE`` and ``KKT 2 TRUE``."""
        assert fitted.converged

    def test_nelder_mead_agrees(self, data, fitted):
        got = fit_pnbd_staticcov(data, method="Nelder-Mead", hessian=False)
        assert got.log_likelihood == pytest.approx(fitted.log_likelihood, abs=1e-4)

    def test_is_reached_from_different_starting_points(self, data, fitted):
        got = fit_pnbd_staticcov(
            data, start=(0.5, 30.0, 1.5, 80.0), start_cov=-0.2, hessian=False
        )
        assert got.log_likelihood == pytest.approx(fitted.log_likelihood, abs=1e-4)

    def test_parameter_count_includes_the_covariates(self, fitted):
        assert fitted.n_parameters == 8

    def test_information_criteria_use_the_plain_likelihood_when_unpenalised(
        self, fitted
    ):
        """Without a penalty the two likelihoods coincide."""
        assert fitted.unpenalised_log_likelihood == pytest.approx(
            fitted.log_likelihood
        )
        assert fitted.aic == pytest.approx(16 - 2 * fitted.log_likelihood)

    def test_a_params_object_without_an_unpenalised_value_falls_back(self):
        """Hand-built parameters, as when replaying a published fit."""
        from clvtools.pnbd.staticcov import PnbdStaticCovParams

        params = PnbdStaticCovParams(
            r=1.8378, alpha=92.9123, s=0.5920, beta=49.6227,
            gamma_life=np.array([-0.6430, 0.7907]),
            gamma_trans=np.array([0.2859, 0.6241]),
            names_cov_life=NAMES, names_cov_trans=NAMES,
            log_likelihood=-5821.0627, converged=True, n_customers=600,
        )
        assert params.unpenalised_log_likelihood is None
        assert params.aic == pytest.approx(16 - 2 * -5821.0627)
        assert params.bic == pytest.approx(
            8 * np.log(600) - 2 * -5821.0627
        )

    def test_names_follow_clvtools_convention(self, fitted):
        assert fitted.names == [
            "r", "alpha", "s", "beta",
            "life.Gender", "life.Channel", "trans.Gender", "trans.Channel",
        ]

    def test_rejects_bad_start_values(self, data):
        with pytest.raises(ValueError, match="4 model parameters"):
            fit_pnbd_staticcov(data, start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            fit_pnbd_staticcov(data, start=(1.0, -1.0, 1.0, 1.0))

    def test_standard_errors_require_a_hessian(self, data):
        got = fit_pnbd_staticcov(data, hessian=False)
        with pytest.raises(ValueError, match="hessian=True"):
            got.standard_errors()


class TestRateBuilderValidation:
    def test_rejects_a_covariate_parameter_mismatch(self):
        with pytest.raises(ValueError, match="2 transaction covariates but 1"):
            alpha_i(50.0, [0.1], np.zeros((3, 2)))
        with pytest.raises(ValueError, match="2 attrition covariates but 3"):
            beta_i(50.0, [0.1, 0.2, 0.3], np.zeros((3, 2)))


@pytest.mark.slow
class TestPrediction:
    def test_predict_uses_per_customer_rates(self, data, fitted):
        from clvtools.predict import predict

        table = predict(data, fitted, prediction_end=52)
        assert len(table) == 600
        assert ((table["PAlive"] >= 0) & (table["PAlive"] <= 1)).all()
        # Customers differ in covariates, so their predictions must differ even
        # at identical (x, t_x, T).
        cbs = data.customer_summary().set_index("Id")
        zero_purchase = cbs.index[cbs["x"] == 0]
        assert table.loc[zero_purchase, "PAlive"].nunique() > 1  # noqa: PD101

    def test_predict_rejects_data_without_covariates(self, fitted):
        from clvtools.predict import predict

        plain = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        with pytest.raises(TypeError, match="covariate model needs covariate data"):
            predict(plain, fitted, prediction_end=52)
