r"""Table 3's two alternative latent attrition models.

S6.2.1: "As an alternative to the Pareto/NBD model, CLVTools features the
Beta-Geometric/NBD model and the Gamma-Gompertz/NBD model. To use these models,
set the parameter ``family`` to either ``bgnbd`` or to ``ggomnbd``."

Neither model's equations appear in the paper -- S3.2 gives references instead
-- so, as with the time-varying covariates, correctness rests on the reference
implementation. Each expression is checked at three parameter vectors.

Beyond that, the three families are compared against each other on the same
data, which is the check the paper's Table 3 invites: they share a transaction
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
        assert scaled_a > a and scaled_b > b

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
    r"""Table 3's three models, on one dataset.

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
