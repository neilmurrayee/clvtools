r"""Table 4's two alternative latent attrition models.

S6.2.1: "As an alternative to the Pareto/NBD model, CLVTools features the
Beta-Geometric/NBD model and the Gamma-Gompertz/NBD model. To use these models,
set the parameter ``family`` to either ``bgnbd`` or to ``ggomnbd``."

Neither model's equations appear in the paper -- S3.2 gives references instead
-- so, as with the time-varying covariates, correctness rests on the reference
implementation. Each expression is checked at three parameter vectors.

Beyond that, the three families are compared against each other on the same
data, which is the check the paper's Table 4 invites: they share a transaction
process and differ only in how attrition is modelled, so where one nests inside
another the fits must agree.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_csv, fixture_json

from clvtools import bgnbd, ggomnbd
from clvtools.pnbd import log_likelihood as pnbd_log_likelihood

BG_GRID = fixture_json("bgnbd_nocov_grid")
GG_GRID = fixture_json("ggomnbd_nocov_grid")
PNBD_LL = -5848.097827


@pytest.fixture(scope="module")
def cbs():
    return fixture_csv("bgnbd_cbs")


@pytest.fixture(scope="module")
def xtt(cbs):
    return (
        cbs["x"].to_numpy(dtype=float),
        cbs["t.x"].to_numpy(dtype=float),
        cbs["T.cal"].to_numpy(dtype=float),
    )


def _aligned(name: str, cbs):
    return fixture_csv(name).set_index("Id").loc[cbs["Id"]]


# -- BG/NBD -------------------------------------------------------------------


@pytest.mark.oracle
class TestBgnbdAgainstOracle:
    @pytest.mark.parametrize("case", list(BG_GRID["params"]))
    def test_individual_log_likelihood(self, cbs, xtt, case):
        want = _aligned(f"bgnbd_nocov_{case}", cbs)
        got = bgnbd.log_likelihood_ind(*xtt, **BG_GRID["params"][case])
        np.testing.assert_allclose(got, want["LL.ind"], rtol=1e-11)

    @pytest.mark.parametrize("case", list(BG_GRID["params"]))
    def test_sample_log_likelihood(self, xtt, case):
        got = bgnbd.log_likelihood(*xtt, **BG_GRID["params"][case])
        assert got == pytest.approx(BG_GRID["LL.sum"][case], rel=1e-11)

    @pytest.mark.parametrize("case", list(BG_GRID["params"]))
    def test_probability_alive(self, cbs, xtt, case):
        want = _aligned(f"bgnbd_nocov_{case}", cbs)
        got = bgnbd.probability_alive(*xtt, **BG_GRID["params"][case])
        np.testing.assert_allclose(got, want["PAlive"], rtol=1e-11)

    @pytest.mark.parametrize("case", list(BG_GRID["params"]))
    def test_conditional_expected_transactions(self, cbs, xtt, case):
        want = _aligned(f"bgnbd_nocov_{case}", cbs)
        got = bgnbd.conditional_expected_transactions(
            *xtt, BG_GRID["CET.horizon"], **BG_GRID["params"][case]
        )
        np.testing.assert_allclose(got, want["CET"], rtol=1e-11)

    def test_pmf(self, cbs, xtt):
        want = _aligned("bgnbd_nocov_pmf_mle", cbs)
        T = xtt[2]
        for k in range(11):
            np.testing.assert_allclose(
                bgnbd.pmf(k, T, **BG_GRID["params"]["mle"]),
                want[f"pmf.{k}"], rtol=1e-11, err_msg=f"k={k}",
            )

    def test_expectation(self):
        want = fixture_csv("bgnbd_nocov_expectation_mle")
        got = bgnbd.expectation(
            want["t"].to_numpy(dtype=float), **BG_GRID["params"]["mle"]
        )
        np.testing.assert_allclose(got, want["expectation"], rtol=1e-11)

    def test_static_covariate_rates(self, cbs):
        r"""``alpha_i``, ``a_i`` and ``b_i``, whose signs differ from each other."""
        from clvtools import load_apparel_static_cov

        params = fixture_json("bgnbd_staticcov_fit")["coefficients"]
        want = _aligned("bgnbd_staticcov_mle", cbs)
        design = (
            load_apparel_static_cov().set_index("Id")
            .loc[cbs["Id"], ["Gender", "Channel"]].to_numpy(dtype=float)
        )
        g_life = np.array([params["life.Gender"], params["life.Channel"]])
        g_trans = np.array([params["trans.Gender"], params["trans.Channel"]])

        np.testing.assert_allclose(
            bgnbd.alpha_i(params["alpha"], g_trans, design), want["alpha.i"], rtol=1e-12
        )
        np.testing.assert_allclose(
            bgnbd.a_i(params["a"], g_life, design), want["a.i"], rtol=1e-12
        )
        np.testing.assert_allclose(
            bgnbd.b_i(params["b"], g_life, design), want["b.i"], rtol=1e-12
        )

    def test_static_covariate_log_likelihood(self, cbs, xtt):
        from clvtools import load_apparel_static_cov

        params = fixture_json("bgnbd_staticcov_fit")["coefficients"]
        want = _aligned("bgnbd_staticcov_mle", cbs)
        design = (
            load_apparel_static_cov().set_index("Id")
            .loc[cbs["Id"], ["Gender", "Channel"]].to_numpy(dtype=float)
        )
        g_life = np.array([params["life.Gender"], params["life.Channel"]])
        g_trans = np.array([params["trans.Gender"], params["trans.Channel"]])

        got = bgnbd.log_likelihood_ind(
            *xtt, r=params["r"],
            alpha=bgnbd.alpha_i(params["alpha"], g_trans, design),
            a=bgnbd.a_i(params["a"], g_life, design),
            b=bgnbd.b_i(params["b"], g_life, design),
        )
        np.testing.assert_allclose(got, want["LL.ind"], rtol=1e-10)

    @pytest.mark.slow
    def test_fit_matches(self, xtt):
        want = fixture_json("bgnbd_fit")
        got = bgnbd.fit_bgnbd(*xtt)
        assert got.converged
        assert got.log_likelihood == pytest.approx(want["logLik"], abs=1e-5)
        assert got.log_likelihood >= want["logLik"] - 1e-9
        for name, value in want["coefficients"].items():
            assert getattr(got, name) == pytest.approx(value, rel=1e-3), name
        assert got.aic == pytest.approx(want["AIC"], abs=1e-4)
        assert got.bic == pytest.approx(want["BIC"], abs=1e-4)


class TestBgnbdProperties:
    def test_a_zero_repeater_is_alive_with_certainty(self):
        r"""Dropout only follows a transaction, so with none there is no risk.

        This is the sharpest behavioural difference from the Pareto/NBD, where
        a customer with no repeat purchase has ``PAlive`` around 0.28 on this
        data because they could have died at any moment.
        """
        p = BG_GRID["params"]["mle"]
        assert float(bgnbd.probability_alive(0, 0.0, 104.0, **p)) == 1.0

    def test_probability_alive_lies_in_the_unit_interval(self, xtt):
        p = bgnbd.probability_alive(*xtt, **BG_GRID["params"]["mle"])
        assert np.all((p >= 0) & (p <= 1))

    def test_pmf_sums_to_one(self):
        total = sum(
            float(bgnbd.pmf(k, 104.0, **BG_GRID["params"]["mle"]))
            for k in range(600)
        )
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_expectation_starts_at_zero_and_increases(self):
        p = BG_GRID["params"]["mle"]
        assert float(bgnbd.expectation(0.0, **p)) == pytest.approx(0.0, abs=1e-15)
        t = np.linspace(0, 300, 100)
        assert np.all(np.diff(bgnbd.expectation(t, **p)) > 0)

    def test_cet_is_zero_over_a_zero_horizon(self, xtt):
        got = bgnbd.conditional_expected_transactions(
            *xtt, 0.0, **BG_GRID["params"]["mle"]
        )
        np.testing.assert_allclose(got, 0.0, atol=1e-12)

    def test_scaling_a_and_b_together_leaves_the_beta_mean_alone(self):
        r"""Which is what makes the ``a_i``/``b_i`` sign convention coherent.

        Both are scaled by the same ``exp(gamma'x)``, so the dropout
        probability's mean ``a/(a+b)`` is unchanged and only its dispersion
        moves.
        """
        a, b = 1.2755, 8.8608
        design = np.array([[1.0]])
        gamma = [0.6]
        scaled_a = float(bgnbd.a_i(a, gamma, design)[0])
        scaled_b = float(bgnbd.b_i(b, gamma, design)[0])
        assert scaled_a / (scaled_a + scaled_b) == pytest.approx(a / (a + b))
        assert scaled_a > a
        assert scaled_b > b

    def test_cet_and_expectation_reject_a_equal_to_one(self, xtt):
        with pytest.raises(ValueError, match="undefined at a = 1"):
            bgnbd.conditional_expected_transactions(
                *xtt, 52.0, r=0.6, alpha=20.0, a=1.0, b=8.0
            )
        with pytest.raises(ValueError, match="undefined at a = 1"):
            bgnbd.expectation(52.0, r=0.6, alpha=20.0, a=1.0, b=8.0)

    def test_pmf_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            bgnbd.pmf(-1, 104.0, **BG_GRID["params"]["mle"])

    def test_weights_repeat_rows(self, xtt):
        x, t_x, T = xtt
        each = bgnbd.log_likelihood_ind(x[:2], t_x[:2], T[:2], **BG_GRID["params"]["mle"])
        weighted = bgnbd.log_likelihood(
            x[:2], t_x[:2], T[:2], **BG_GRID["params"]["mle"], weights=[2.0, 3.0]
        )
        assert weighted == pytest.approx(2 * each[0] + 3 * each[1])

    def test_covariate_parameter_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="2 transaction covariates but 1"):
            bgnbd.alpha_i(20.0, [0.1], np.zeros((3, 2)))
        with pytest.raises(ValueError, match="2 attrition covariates but 3"):
            bgnbd.a_i(1.2, [0.1, 0.2, 0.3], np.zeros((3, 2)))


class TestBgnbdFitValidation:
    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            bgnbd.fit_bgnbd([0, 1], [0.0], [104.0, 104.0])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="no customers"):
            bgnbd.fit_bgnbd([], [], [])

    def test_rejects_negative_frequency(self):
        with pytest.raises(ValueError, match="non-negative"):
            bgnbd.fit_bgnbd([-1, 2], [0.0, 30.0], [104.0, 104.0])

    def test_rejects_recency_beyond_the_window(self):
        with pytest.raises(ValueError, match="cannot exceed T"):
            bgnbd.fit_bgnbd([1, 2], [200.0, 30.0], [104.0, 104.0])

    def test_rejects_bad_start_values(self):
        with pytest.raises(ValueError, match="four values"):
            bgnbd.fit_bgnbd([0, 2], [0.0, 30.0], [104.0, 104.0], start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            bgnbd.fit_bgnbd(
                [0, 2], [0.0, 30.0], [104.0, 104.0], start=(1.0, -1.0, 1.0, 1.0)
            )

    def test_params_iterate_and_report_criteria(self):
        params = bgnbd.BgnbdParams(
            r=0.6, alpha=21.0, a=1.3, b=8.9,
            log_likelihood=-5857.0, converged=True, n_customers=600,
        )
        assert list(params) == [0.6, 21.0, 1.3, 8.9]
        assert params.as_dict() == {"r": 0.6, "alpha": 21.0, "a": 1.3, "b": 8.9}
        assert params.n_parameters == 4
        assert params.aic == pytest.approx(8 - 2 * -5857.0)
        assert params.bic == pytest.approx(4 * np.log(600) - 2 * -5857.0)


# -- GGom/NBD -----------------------------------------------------------------


@pytest.mark.oracle
class TestGgomnbdAgainstOracle:
    @pytest.mark.parametrize("case", list(GG_GRID["params"]))
    def test_individual_log_likelihood(self, cbs, xtt, case):
        want = _aligned(f"ggomnbd_nocov_{case}", cbs)
        got = ggomnbd.log_likelihood_ind(*xtt, **GG_GRID["params"][case])
        np.testing.assert_allclose(got, want["LL.ind"], rtol=1e-9)

    @pytest.mark.parametrize("case", list(GG_GRID["params"]))
    def test_sample_log_likelihood(self, xtt, case):
        got = ggomnbd.log_likelihood(*xtt, **GG_GRID["params"][case])
        assert got == pytest.approx(GG_GRID["LL.sum"][case], rel=1e-9)

    @pytest.mark.parametrize("case", list(GG_GRID["params"]))
    def test_probability_alive(self, cbs, xtt, case):
        want = _aligned(f"ggomnbd_nocov_{case}", cbs)
        got = ggomnbd.probability_alive(*xtt, **GG_GRID["params"][case])
        np.testing.assert_allclose(got, want["PAlive"], rtol=1e-8)

    @pytest.mark.parametrize("case", list(GG_GRID["params"]))
    def test_conditional_expected_transactions(self, cbs, xtt, case):
        r"""Loosest tolerance in the suite, and legitimately so.

        ``CET`` layers a numerical integral inside another, and at the fitted
        ``b = 8.1e-07`` the integrand is very nearly singular. Both sides are
        quadrature, so they agree to their own accuracy, not to machine
        precision.
        """
        want = _aligned(f"ggomnbd_nocov_{case}", cbs)
        got = ggomnbd.conditional_expected_transactions(
            *xtt, GG_GRID["CET.horizon"], **GG_GRID["params"][case]
        )
        np.testing.assert_allclose(got, want["CET"], rtol=1e-6)

    def test_pmf(self, cbs, xtt):
        want = _aligned("ggomnbd_nocov_pmf_mle", cbs)
        for k in range(7):
            np.testing.assert_allclose(
                ggomnbd.pmf(k, xtt[2], **GG_GRID["params"]["mle"]),
                want[f"pmf.{k}"], rtol=1e-9, err_msg=f"k={k}",
            )

    def test_expectation(self):
        want = fixture_csv("ggomnbd_nocov_expectation_mle")
        got = ggomnbd.expectation(
            want["t"].to_numpy(dtype=float), **GG_GRID["params"]["mle"]
        )
        np.testing.assert_allclose(got, want["expectation"], rtol=1e-9)


class TestGgomnbdProperties:
    def test_probability_alive_lies_in_the_unit_interval(self, xtt):
        p = ggomnbd.probability_alive(*xtt, **GG_GRID["params"]["mle"])
        assert np.all((p >= 0) & (p <= 1))

    def test_expectation_starts_at_zero_and_increases(self):
        p = GG_GRID["params"]["mle"]
        assert float(ggomnbd.expectation(0.0, **p)) == pytest.approx(0.0, abs=1e-15)
        t = np.linspace(0.5, 200, 40)
        assert np.all(np.diff(ggomnbd.expectation(t, **p)) > 0)

    def test_pmf_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            ggomnbd.pmf(-1, 104.0, **GG_GRID["params"]["mle"])

    def test_weights_repeat_rows(self, xtt):
        x, t_x, T = xtt
        each = ggomnbd.log_likelihood_ind(
            x[:2], t_x[:2], T[:2], **GG_GRID["params"]["mle"]
        )
        weighted = ggomnbd.log_likelihood(
            x[:2], t_x[:2], T[:2], **GG_GRID["params"]["mle"], weights=[2.0, 3.0]
        )
        assert weighted == pytest.approx(2 * each[0] + 3 * each[1])

    def test_covariate_rates_use_the_rate_convention(self):
        r"""Both :math:`\alpha_i` and :math:`\beta_i` take ``exp(-gamma'x)``.

        Unlike the BG/NBD's :math:`a_i, b_i`, which take a positive sign.
        """
        design = np.array([[0.0], [1.0]])
        assert bool(ggomnbd.alpha_i(50.0, [0.5], design)[1] < 50.0)
        assert bool(ggomnbd.beta_i(50.0, [0.5], design)[1] < 50.0)

    def test_covariate_parameter_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="2 transaction covariates but 1"):
            ggomnbd.alpha_i(20.0, [0.1], np.zeros((3, 2)))
        with pytest.raises(ValueError, match="2 attrition covariates but 3"):
            ggomnbd.beta_i(1.2, [0.1, 0.2, 0.3], np.zeros((3, 2)))

    def test_fit_validation(self):
        with pytest.raises(ValueError, match="same shape"):
            ggomnbd.fit_ggomnbd([0, 1], [0.0], [104.0, 104.0])
        with pytest.raises(ValueError, match="no customers"):
            ggomnbd.fit_ggomnbd([], [], [])
        with pytest.raises(ValueError, match="non-negative"):
            ggomnbd.fit_ggomnbd([-1, 2], [0.0, 30.0], [104.0, 104.0])
        with pytest.raises(ValueError, match="cannot exceed T"):
            ggomnbd.fit_ggomnbd([1, 2], [200.0, 30.0], [104.0, 104.0])
        with pytest.raises(ValueError, match="five values"):
            ggomnbd.fit_ggomnbd([0, 2], [0.0, 30.0], [104.0, 104.0], start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            ggomnbd.fit_ggomnbd(
                [0, 2], [0.0, 30.0], [104.0, 104.0], start=(1.0, -1.0, 1.0, 1.0, 1.0)
            )

    def test_params_iterate_and_report_criteria(self):
        params = ggomnbd.GgomnbdParams(
            r=1.45, alpha=48.6, b=1e-6, s=0.56, beta=4e-5,
            log_likelihood=-5848.1, converged=True, n_customers=600,
        )
        assert list(params) == [1.45, 48.6, 1e-6, 0.56, 4e-5]
        assert params.as_dict() == {
            "r": 1.45, "alpha": 48.6, "b": 1e-6, "s": 0.56, "beta": 4e-5
        }
        assert params.n_parameters == 5
        assert params.aic == pytest.approx(10 - 2 * -5848.1)
        assert params.bic == pytest.approx(5 * np.log(600) - 2 * -5848.1)

    @pytest.mark.slow
    def test_a_short_fit_runs(self, xtt):
        """The full fit needs an integral per customer per evaluation."""
        got = ggomnbd.fit_ggomnbd(*xtt, options={"maxiter": 1, "maxfun": 8})
        assert np.isfinite(got.log_likelihood)
        assert got.n_parameters == 5


# -- the three families compared ----------------------------------------------


class TestFamiliesCompared:
    r"""Table 4's three models, on one dataset.

    They share a transaction process -- Poisson with gamma heterogeneity -- and
    differ only in attrition: exponential/gamma, geometric/beta, or
    Gompertz/gamma.
    """

    @pytest.mark.paper
    def test_the_ggomnbd_collapses_onto_the_pareto_nbd(self):
        r"""Its fitted ``b`` is 8.1e-07, and the fit is the Pareto/NBD's.

        The limit is not simply :math:`b \to 0`. The GGom/NBD's lifetime term
        is :math:`s\log\frac{\beta}{\beta - 1 + e^{bT}}`, which at
        :math:`b = 0` collapses to :math:`\log 1 = 0` -- an immortal customer,
        not an exponential one. The Pareto/NBD is reached only if
        :math:`\beta` shrinks with :math:`b`, because

        .. math::
            \beta - 1 + e^{bT} \approx \beta + bT

        for small :math:`b` and :math:`\beta`, so the term matches the
        Pareto/NBD's :math:`s\log\frac{\beta_P}{\beta_P + T}` exactly when
        :math:`\beta = b\,\beta_P`.

        The fitted parameters bear that out: ``beta / b`` is 46.72 against the
        Pareto/NBD's ``beta`` of 46.8844, and the two ``r``, ``alpha`` and ``s``
        agree to three decimals. The GGom/NBD has found the same model in a
        boundary parameterisation, and pays an extra parameter for it.
        """
        fitted = fixture_json("ggomnbd_fit")
        coefficients = fitted["coefficients"]

        assert coefficients["b"] < 1e-5
        assert coefficients["beta"] / coefficients["b"] == pytest.approx(
            46.8844, rel=1e-2
        )
        assert coefficients["r"] == pytest.approx(1.4490, rel=1e-3)
        assert coefficients["alpha"] == pytest.approx(48.6361, rel=1e-3)
        assert coefficients["s"] == pytest.approx(0.5613, rel=1e-2)

        assert fitted["logLik"] == pytest.approx(PNBD_LL, abs=1e-3)
        # The fifth parameter buys nothing, and AIC charges for it.
        assert fitted["AIC"] > 2 * 4 - 2 * PNBD_LL

    def test_the_ggomnbd_likelihood_converges_on_the_pareto_nbd(self, xtt):
        r"""Along :math:`\beta = b\,\beta_P`, checked directly.

        The error falls by roughly a factor of ten for each factor of ten in
        :math:`b`, until quadrature accuracy floors it around 3e-07.
        """
        r, alpha, s, beta_p = 1.4490, 48.6361, 0.5613, 46.8844
        want = pnbd_log_likelihood(*xtt, r, alpha, s, beta_p)

        errors = [
            abs(
                ggomnbd.log_likelihood(
                    *xtt, r=r, alpha=alpha, b=b, s=s, beta=b * beta_p
                )
                - want
            )
            for b in (1e-3, 1e-5, 1e-7)
        ]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 1e-4

    def test_b_alone_going_to_zero_does_not_give_the_pareto_nbd(self, xtt):
        """Holding beta fixed makes customers immortal instead."""
        r, alpha, s, beta_p = 1.4490, 48.6361, 0.5613, 46.8844
        want = pnbd_log_likelihood(*xtt, r, alpha, s, beta_p)
        got = ggomnbd.log_likelihood(
            *xtt, r=r, alpha=alpha, b=1e-9, s=s, beta=beta_p
        )
        assert abs(got - want) > 100

    @pytest.mark.paper
    def test_the_bgnbd_fits_worse_than_the_pareto_nbd_here(self):
        """Same parameter count, lower likelihood: 9 log-likelihood units."""
        fitted = fixture_json("bgnbd_fit")
        assert fitted["logLik"] < PNBD_LL
        assert PNBD_LL - fitted["logLik"] == pytest.approx(8.9, abs=0.5)

    def test_zero_repeaters_are_treated_differently_by_each_family(self, xtt):
        r"""The clearest way the attrition assumptions differ.

        Under the BG/NBD dropout can only follow a transaction, so a customer
        with none is alive with certainty. Under the Pareto/NBD they may have
        died at any point, and on this data their ``PAlive`` is about 0.28.
        """
        from clvtools.pnbd import probability_alive as pnbd_palive

        bg = float(bgnbd.probability_alive(0, 0.0, 104.0, **BG_GRID["params"]["mle"]))
        pn = float(pnbd_palive(0, 0.0, 104.0, 1.4490, 48.6361, 0.5613, 46.8844))
        assert bg == 1.0
        assert 0.2 < pn < 0.4

    def test_every_family_agrees_on_the_customer_summary(self, cbs):
        """All three consume the same ``(x, t_x, T)``, per S6.2.1."""
        from clvtools import ClvData, load_apparel_trans

        summary = (
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
            .customer_summary().set_index("Id").loc[cbs["Id"]]
        )
        np.testing.assert_array_equal(summary["x"], cbs["x"])
        np.testing.assert_allclose(summary["T"], cbs["T.cal"])


# -- covariates for the other two families ------------------------------------


@pytest.fixture(scope="module")
def static_data():
    from clvtools import (
        ClvData,
        ClvDataStaticCov,
        load_apparel_static_cov,
        load_apparel_trans,
    )

    return ClvDataStaticCov(
        ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
        load_apparel_static_cov(),
        names_cov_life=["Gender", "Channel"],
        names_cov_trans=["Gender", "Channel"],
    )


@pytest.mark.oracle
class TestGgomnbdStaticCovariates:
    r"""Table 4 marks the GGom/NBD as taking time-invariant covariates."""

    @staticmethod
    @pytest.fixture(scope="class")
    def published():
        return fixture_json("ggomnbd_staticcov_fit")["coefficients"]

    def test_rate_builders_match(self, static_data, published, cbs):
        want = _aligned("ggomnbd_staticcov_mle", cbs)
        g_life = np.array([published["life.Gender"], published["life.Channel"]])
        g_trans = np.array([published["trans.Gender"], published["trans.Channel"]])
        np.testing.assert_allclose(
            ggomnbd.alpha_i(published["alpha"], g_trans, static_data.design_trans()),
            want["alpha.i"], rtol=1e-12,
        )
        np.testing.assert_allclose(
            ggomnbd.beta_i(published["beta"], g_life, static_data.design_life()),
            want["beta.i"], rtol=1e-11,
        )

    def test_individual_log_likelihood_matches(self, static_data, published, cbs, xtt):
        want = _aligned("ggomnbd_staticcov_mle", cbs)
        g_life = np.array([published["life.Gender"], published["life.Channel"]])
        g_trans = np.array([published["trans.Gender"], published["trans.Channel"]])
        got = ggomnbd.log_likelihood_ind(
            *xtt,
            r=published["r"],
            alpha=ggomnbd.alpha_i(
                published["alpha"], g_trans, static_data.design_trans()
            ),
            b=published["b"], s=published["s"],
            beta=ggomnbd.beta_i(
                published["beta"], g_life, static_data.design_life()
            ),
        )
        np.testing.assert_allclose(got, want["LL.ind"], rtol=1e-9)

    def test_sample_log_likelihood_matches(self, static_data, published):
        want = fixture_json("ggomnbd_staticcov_fit")["logLik"]
        cbs = static_data.customer_summary()
        got = ggomnbd.log_likelihood_staticcov(
            cbs["x"], cbs["t_x"], cbs["T"],
            published["r"], published["alpha"], published["b"],
            published["s"], published["beta"],
            [published["life.Gender"], published["life.Channel"]],
            [published["trans.Gender"], published["trans.Channel"]],
            static_data.design_life(), static_data.design_trans(),
        )
        assert got == pytest.approx(want, abs=1e-6)

    def test_it_collapses_onto_the_pareto_nbd_here_too(self):
        r"""As without covariates: fitted ``b`` is negligible.

        The covariate model reaches -5821.06296 against the Pareto/NBD's
        -5821.06271 with two parameters fewer, so the Gompertz hazard is again
        buying nothing on this data.
        """
        fitted = fixture_json("ggomnbd_staticcov_fit")
        assert fitted["coefficients"]["b"] < 1e-4
        assert fitted["logLik"] == pytest.approx(-5821.062709, abs=1e-3)


@pytest.mark.slow
class TestBgnbdStaticCovariateFit:
    @staticmethod
    @pytest.fixture(scope="class")
    def fitted(static_data):
        return bgnbd.fit_bgnbd_staticcov(static_data, hessian=False)

    @pytest.mark.oracle
    def test_reaches_at_least_the_oracles_optimum(self, fitted):
        r"""One-sided, and for an unusually clear reason.

        The BG/NBD's two beta parameters are barely identified once covariates
        scale them: ``a_i`` and ``b_i`` are both multiplied by the *same*
        ``exp(gamma'x)``, so the data pins their ratio far better than their
        common size. CLVTools stops at ``a + b`` around 38,600; this
        implementation climbs to about 2.5 million for a gain of 3e-4 in the
        log-likelihood. Neither is wrong -- the direction is nearly flat.
        """
        want = fixture_json("bgnbd_staticcov_fit")["logLik"]
        assert fitted.log_likelihood >= want - 1e-6

    def test_covariates_improve_on_the_model_without_them(self, fitted):
        plain = fixture_json("bgnbd_fit")["logLik"]
        assert fitted.log_likelihood > plain
        assert fitted.n_parameters == 8

    def test_the_barely_identified_direction_is_the_common_scale(self, fitted):
        """Its ratio is pinned; its size is not."""
        published = fixture_json("bgnbd_staticcov_fit")["coefficients"]
        published_ratio = published["a"] / (published["a"] + published["b"])
        got_ratio = fitted.a / (fitted.a + fitted.b)
        assert got_ratio == pytest.approx(published_ratio, rel=5e-3)
        assert fitted.a + fitted.b > 10 * (published["a"] + published["b"])

    def test_names_follow_clvtools_convention(self, fitted):
        assert fitted.names == [
            "r", "alpha", "a", "b",
            "life.Gender", "life.Channel", "trans.Gender", "trans.Channel",
        ]

    def test_polish_is_what_climbs_the_ridge(self, static_data):
        r"""A regression guard for the derivative-free follow-up pass.

        Without it L-BFGS-B reports successful convergence with ``a + b`` around
        1,200, where the likelihood is still rising.
        """
        unpolished = bgnbd.fit_bgnbd_staticcov(
            static_data, hessian=False, polish=False
        )
        polished = bgnbd.fit_bgnbd_staticcov(static_data, hessian=False)
        assert polished.log_likelihood > unpolished.log_likelihood
        assert polished.a + polished.b > 100 * (unpolished.a + unpolished.b)

    def test_standard_errors_require_a_hessian(self, fitted):
        with pytest.raises(ValueError, match="hessian=True"):
            fitted.standard_errors()


@pytest.mark.slow
class TestFamilyConstraintsAndRegularization:
    r"""Table 4 marks both as available for all three families."""

    @pytest.mark.oracle
    def test_bgnbd_equality_constraint(self, static_data):
        want = fixture_json("bgnbd_staticcov_constrained_fit")
        got = bgnbd.fit_bgnbd_staticcov(
            static_data, names_cov_constr=["Gender"], hessian=False
        )
        assert got.names == [
            "r", "alpha", "a", "b",
            "life.Channel", "trans.Channel", "constr.Gender",
        ]
        assert got.n_parameters == 7
        assert got.log_likelihood >= want["logLik"] - 1e-3

        index_life = got.covariates.names_cov_life.index("Gender")
        index_trans = got.covariates.names_cov_trans.index("Gender")
        assert got.gamma_life[index_life] == pytest.approx(
            got.gamma_trans[index_trans]
        )

    @pytest.mark.oracle
    def test_bgnbd_regularization(self, static_data):
        want = fixture_json("bgnbd_staticcov_regularized_0_1")
        got = bgnbd.fit_bgnbd_staticcov(
            static_data, reg_lambdas=(0.1, 0.1), hessian=False
        )
        assert got.log_likelihood >= want["logLik"] - 1e-4
        # The reported value is the penalised mean, as CLVTools reports it.
        assert -20 < got.log_likelihood < 0
        assert got.unpenalised_log_likelihood < -5000

    def test_a_heavier_penalty_shrinks_the_coefficients(self, static_data):
        light = bgnbd.fit_bgnbd_staticcov(
            static_data, reg_lambdas=(0.01, 0.01), hessian=False
        )
        heavy = bgnbd.fit_bgnbd_staticcov(
            static_data, reg_lambdas=(100.0, 100.0), hessian=False
        )

        def size(fit):
            return float(
                np.sum(fit.gamma_life**2) + np.sum(fit.gamma_trans**2)
            )

        assert size(heavy) < size(light)

    def test_information_criteria_use_the_true_likelihood(self, static_data):
        got = bgnbd.fit_bgnbd_staticcov(
            static_data, reg_lambdas=(0.1, 0.1), hessian=False
        )
        assert got.aic == pytest.approx(
            2 * got.n_parameters - 2 * got.unpenalised_log_likelihood
        )
        assert got.bic == pytest.approx(
            got.n_parameters * np.log(600) - 2 * got.unpenalised_log_likelihood
        )

    def test_constraining_a_covariate_absent_from_a_process_is_rejected(self):
        from clvtools import (
            ClvData,
            ClvDataStaticCov,
            load_apparel_static_cov,
            load_apparel_trans,
        )

        data = ClvDataStaticCov(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_static_cov(),
            names_cov_life=["Gender"], names_cov_trans=["Channel"],
        )
        with pytest.raises(ValueError, match="covariate of both"):
            bgnbd.fit_bgnbd_staticcov(
                data, names_cov_constr=["Gender"], hessian=False
            )

    def test_rejects_bad_inputs(self, static_data):
        with pytest.raises(ValueError, match="4 model parameters"):
            bgnbd.fit_bgnbd_staticcov(static_data, start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            bgnbd.fit_bgnbd_staticcov(static_data, start=(1.0, -1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="two values"):
            bgnbd.fit_bgnbd_staticcov(static_data, reg_lambdas=(0.1,))
        with pytest.raises(ValueError, match="non-negative"):
            bgnbd.fit_bgnbd_staticcov(static_data, reg_lambdas=(-1.0, 0.1))


@pytest.mark.slow
class TestGgomnbdStaticCovariateFit:
    """Slow even by this model's standards: an integral per customer per step."""

    @pytest.mark.oracle
    def test_a_short_fit_runs_and_reports_its_shape(self, static_data):
        got = ggomnbd.fit_ggomnbd_staticcov(
            static_data, polish=False, options={"maxiter": 2, "maxfun": 24}
        )
        assert got.names == [
            "r", "alpha", "b", "s", "beta",
            "life.Gender", "life.Channel", "trans.Gender", "trans.Channel",
        ]
        assert got.n_parameters == 9
        assert np.isfinite(got.log_likelihood)
        assert got.n_customers == 600

    def test_constraints_reduce_the_parameter_count(self, static_data):
        got = ggomnbd.fit_ggomnbd_staticcov(
            static_data, names_cov_constr=["Gender"],
            polish=False, options={"maxiter": 2, "maxfun": 24},
        )
        assert "constr.Gender" in got.names
        assert got.n_parameters == 8


@pytest.mark.slow
class TestCovariateParamsObjects:
    """The accessors both covariate result types expose."""

    @staticmethod
    @pytest.fixture(scope="class")
    def bg(static_data):
        return bgnbd.fit_bgnbd_staticcov(static_data, polish=False)

    @staticmethod
    @pytest.fixture(scope="class")
    def gg(static_data):
        return ggomnbd.fit_ggomnbd_staticcov(
            static_data, polish=False, options={"maxiter": 2, "maxfun": 24}
        )

    def test_coefficients_align_names_with_values(self, bg, gg):
        for fit, n_model in ((bg, 4), (gg, 5)):
            coefficients = fit.coefficients
            assert list(coefficients) == fit.names
            assert len(coefficients) == n_model + 4
            assert list(coefficients.values()) == list(fit)

    def test_a_constrained_coefficient_is_reported_once(self, static_data):
        r"""And carries the value both processes share, eq. (14)."""
        fit = bgnbd.fit_bgnbd_staticcov(
            static_data, names_cov_constr=["Gender"], polish=False, hessian=False
        )
        coefficients = fit.coefficients
        assert "constr.Gender" in coefficients
        assert "life.Gender" not in coefficients
        assert "trans.Gender" not in coefficients

        index_life = fit.covariates.names_cov_life.index("Gender")
        assert coefficients["constr.Gender"] == pytest.approx(
            float(fit.gamma_life[index_life])
        )

    def test_the_delegating_properties_reach_through(self, bg, gg):
        for fit in (bg, gg):
            assert fit.n_customers == 600
            assert isinstance(fit.converged, bool)
            assert fit.log_likelihood == fit.covariates.log_likelihood
            assert (
                fit.unpenalised_log_likelihood
                == fit.covariates.unpenalised_log_likelihood
            )
            assert len(fit.gamma_life) == 2
            assert len(fit.gamma_trans) == 2
            assert fit.aic == pytest.approx(
                2 * fit.n_parameters - 2 * fit.unpenalised_log_likelihood
            )
            assert fit.bic > fit.aic

    def test_standard_errors_are_reported_per_name(self, bg):
        errors = bg.standard_errors()
        assert list(errors) == bg.names
        assert all(np.isfinite(v) for v in errors.values())

    def test_standard_errors_report_a_constrained_coefficient_once(
        self, static_data
    ):
        fit = bgnbd.fit_bgnbd_staticcov(
            static_data, names_cov_constr=["Gender"], polish=False
        )
        errors = fit.standard_errors()
        assert list(errors) == fit.names
        assert "constr.Gender" in errors

    def test_weights_are_accepted_by_the_covariate_likelihoods(self, static_data):
        cbs = static_data.customer_summary()
        args = (cbs["x"][:2], cbs["t_x"][:2], cbs["T"][:2])
        design_life = static_data.design_life()[:2]
        design_trans = static_data.design_trans()[:2]

        plain = bgnbd.log_likelihood_staticcov(
            *args, 0.6, 20.0, 1.3, 8.9, [0.1, 0.2], [0.3, 0.4],
            design_life, design_trans,
        )
        doubled = bgnbd.log_likelihood_staticcov(
            *args, 0.6, 20.0, 1.3, 8.9, [0.1, 0.2], [0.3, 0.4],
            design_life, design_trans, weights=[2.0, 2.0],
        )
        assert doubled == pytest.approx(2 * plain)

        plain = ggomnbd.log_likelihood_staticcov(
            *args, 1.4, 48.0, 1e-6, 0.56, 4e-5, [0.1, 0.2], [0.3, 0.4],
            design_life, design_trans,
        )
        doubled = ggomnbd.log_likelihood_staticcov(
            *args, 1.4, 48.0, 1e-6, 0.56, 4e-5, [0.1, 0.2], [0.3, 0.4],
            design_life, design_trans, weights=[2.0, 2.0],
        )
        assert doubled == pytest.approx(2 * plain)


class TestSharedCovariateMachineryValidation:
    def test_rejects_mismatched_shapes(self):
        from clvtools._staticcov import fit_static_covariates

        with pytest.raises(ValueError, match="same shape"):
            fit_static_covariates(
                x=[0, 1], t_x=[0.0], T=[104.0, 104.0],
                cov_life=np.zeros((2, 1)), cov_trans=np.zeros((2, 1)),
                names_cov_life=["a"], names_cov_trans=["a"],
                log_likelihood=lambda *args: 0.0,
                n_model_params=4, model_start=(1.0, 1.0, 1.0, 1.0),
            )

    def test_rejects_empty_input(self):
        from clvtools._staticcov import fit_static_covariates

        with pytest.raises(ValueError, match="no customers"):
            fit_static_covariates(
                x=[], t_x=[], T=[],
                cov_life=np.zeros((0, 1)), cov_trans=np.zeros((0, 1)),
                names_cov_life=["a"], names_cov_trans=["a"],
                log_likelihood=lambda *args: 0.0,
                n_model_params=4, model_start=(1.0, 1.0, 1.0, 1.0),
            )


class TestCovariateResultAccessors:
    """The wrappers expose what the shared generics need."""

    @staticmethod
    def _wrapped(family, model: dict, hessian):
        from clvtools._staticcov import StaticCovResult

        covariates = StaticCovResult(
            model=np.array(list(model.values())),
            gamma_life=np.array([0.1, 0.2]),
            gamma_trans=np.array([0.3, 0.4]),
            names_cov_life=["Gender", "Channel"],
            names_cov_trans=["Gender", "Channel"],
            names_cov_constr=[],
            log_likelihood=-1.0, unpenalised_log_likelihood=-1.0,
            converged=True, n_customers=600, hessian=hessian,
        )
        return family(**model, covariates=covariates)

    @pytest.mark.parametrize("family,model", [
        (bgnbd.BgnbdStaticCovParams, {"r": 1.0, "alpha": 2.0, "a": 3.0, "b": 4.0}),
        (
            ggomnbd.GgomnbdStaticCovParams,
            {"r": 1.0, "alpha": 2.0, "b": 3.0, "s": 4.0, "beta": 5.0},
        ),
    ])
    def test_the_hessian_comes_from_the_covariate_fit(self, family, model):
        size = len(model) + 4
        hessian = np.eye(size) * 4.0
        fit = self._wrapped(family, model, hessian)
        np.testing.assert_allclose(fit.hessian, hessian)
        # Curvature of 4 in every direction is a standard error of 1/2.
        assert set(np.round(list(fit.standard_errors().values()), 6)) == {0.5}
        assert fit.names_cov_life == ["Gender", "Channel"]
        assert fit.names_cov_trans == ["Gender", "Channel"]

    @pytest.mark.parametrize("family,model", [
        (bgnbd.BgnbdStaticCovParams, {"r": 1.0, "alpha": 2.0, "a": 3.0, "b": 4.0}),
        (
            ggomnbd.GgomnbdStaticCovParams,
            {"r": 1.0, "alpha": 2.0, "b": 3.0, "s": 4.0, "beta": 5.0},
        ),
    ])
    def test_without_a_hessian_they_say_so(self, family, model):
        fit = self._wrapped(family, model, None)
        with pytest.raises(ValueError, match="hessian=True"):
            fit.confint()
