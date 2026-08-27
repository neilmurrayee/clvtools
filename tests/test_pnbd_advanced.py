r"""S3.4 and S6.5 - correlation, regularization and equality constraints.

Each of the three is nested inside or beside a model already tested, which
gives a check that does not depend on the oracle:

  * ``m = 0`` in the Sarmanov likelihood must reproduce the uncorrelated one;
  * a zero regularization weight must reproduce the unpenalised fit;
  * constraining a covariate must lower the likelihood by exactly the amount a
    likelihood-ratio test then evaluates.

The oracle contributes the harder half: agreement at *its* fitted parameters,
which pins the formulae independently of where either optimiser stops.

Two disagreements with CLVTools 0.12.1 are recorded here as tests rather than
hidden, because both are cases where this implementation reaches a better
optimum and a future reader would otherwise assume a bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from conftest import fixture_csv, fixture_json

from clvtools import ClvData, ClvDataStaticCov, load_apparel_static_cov, load_apparel_trans
from clvtools.pnbd import log_likelihood, log_likelihood_ind
from clvtools.pnbd.correlation import (
    correlated_log_likelihood,
    correlated_log_likelihood_ind,
    correlation_bounds,
    correlation_coefficient,
    fit_pnbd_correlated,
    m_from_correlation,
)
from clvtools.pnbd.staticcov import fit_pnbd_staticcov

PLAIN_MLE = dict(r=1.4490, alpha=48.6361, s=0.5613, beta=46.8844)
PLAIN_LL = -5848.097827


@pytest.fixture(scope="module")
def cbs():
    return fixture_csv("cbs_estimation")


@pytest.fixture(scope="module")
def xtt(cbs):
    return cbs["x"].to_numpy(), cbs["t.x"].to_numpy(), cbs["T.cal"].to_numpy()


@pytest.fixture(scope="module")
def static_data():
    return ClvDataStaticCov(
        ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
        load_apparel_static_cov(),
        names_cov_life=["Gender", "Channel"],
        names_cov_trans=["Gender", "Channel"],
    )


# -- correlation, S3.4 and S6.5.2 ---------------------------------------------


class TestSarmanovConstruction:
    def test_bounds_bracket_zero(self):
        """Independence is always admissible, so 0 lies inside the interval."""
        lo, hi = correlation_bounds(**PLAIN_MLE)
        assert lo < 0 < hi

    def test_bounds_match_the_published_formula(self):
        r, alpha, s, beta = (PLAIN_MLE[k] for k in ("r", "alpha", "s", "beta"))
        la = (alpha / (1 + alpha)) ** r
        lb = (beta / (1 + beta)) ** s
        lo, hi = correlation_bounds(r, alpha, s, beta)
        assert hi == pytest.approx(1 / max(la * (1 - lb), (1 - la) * lb))
        assert lo == pytest.approx(-1 / max(la * lb, (1 - la) * (1 - lb)))

    def test_m_outside_the_bounds_is_rejected(self, xtt):
        x, t_x, T = xtt
        _, hi = correlation_bounds(**PLAIN_MLE)
        with pytest.raises(ValueError, match="outside"):
            correlated_log_likelihood_ind(x, t_x, T, **PLAIN_MLE, m=hi * 2)

    def test_zero_m_recovers_the_uncorrelated_model(self, xtt):
        """Eq. (12)'s dependence term vanishes at ``m = 0``."""
        x, t_x, T = xtt
        got = correlated_log_likelihood_ind(x, t_x, T, **PLAIN_MLE, m=0.0)
        want = log_likelihood_ind(x, t_x, T, **PLAIN_MLE)
        np.testing.assert_allclose(got, want, rtol=1e-15)

    def test_a_small_m_perturbs_the_likelihood_smoothly(self, xtt):
        """Continuity at ``m = 0``, so the null is not a special case."""
        x, t_x, T = xtt
        base = correlated_log_likelihood(x, t_x, T, **PLAIN_MLE, m=0.0)
        near = correlated_log_likelihood(x, t_x, T, **PLAIN_MLE, m=1e-9)
        assert near == pytest.approx(base, abs=1e-5)

    def test_the_sign_of_m_moves_the_likelihood_in_opposite_directions(self, xtt):
        x, t_x, T = xtt
        base = correlated_log_likelihood(x, t_x, T, **PLAIN_MLE, m=0.0)
        up = correlated_log_likelihood(x, t_x, T, **PLAIN_MLE, m=0.5)
        down = correlated_log_likelihood(x, t_x, T, **PLAIN_MLE, m=-0.5)
        assert (up - base) * (down - base) < 0

    def test_weights_repeat_rows(self, xtt):
        x, t_x, T = xtt
        each = correlated_log_likelihood_ind(
            x[:2], t_x[:2], T[:2], **PLAIN_MLE, m=0.3
        )
        weighted = correlated_log_likelihood(
            x[:2], t_x[:2], T[:2], **PLAIN_MLE, m=0.3, weights=[2.0, 3.0]
        )
        assert weighted == pytest.approx(2 * each[0] + 3 * each[1])


class TestCorrelationCoefficient:
    r"""Eq. (13) -- converting :math:`m` to something interpretable."""

    def test_is_zero_when_m_is_zero(self):
        assert correlation_coefficient(0.0, **PLAIN_MLE) == 0.0

    def test_is_linear_in_m(self):
        a = correlation_coefficient(0.4, **PLAIN_MLE)
        b = correlation_coefficient(0.8, **PLAIN_MLE)
        assert b == pytest.approx(2 * a)

    def test_round_trips_through_the_inverse(self):
        for m in (-0.9, 0.0, 0.25, 3.0):
            p = correlation_coefficient(m, **PLAIN_MLE)
            assert m_from_correlation(p, **PLAIN_MLE) == pytest.approx(m)

    def test_inversion_rejects_a_degenerate_scale(self):
        """``m`` is unrecoverable where eq. (13) collapses to zero."""
        with pytest.raises(ValueError, match="identically zero"):
            m_from_correlation(0.1, r=0.0, alpha=48.0, s=0.5, beta=47.0)

    def test_is_much_smaller_than_m(self):
        r"""S3.4: "this coefficient must not be directly interpreted as a
        correlation coefficient". On this data the scaling is ~2700x."""
        assert abs(correlation_coefficient(1.0, **PLAIN_MLE)) < 1e-3


@pytest.mark.slow
class TestCorrelatedFit:
    @pytest.fixture(scope="class")
    def fitted(self, xtt):
        x, t_x, T = xtt
        return fit_pnbd_correlated(x, t_x, T)

    def test_cannot_be_worse_than_the_uncorrelated_optimum(self, fitted):
        """The uncorrelated model is nested at ``m = 0``."""
        assert fitted.log_likelihood >= PLAIN_LL

    def test_the_fitted_correlation_is_small(self, fitted):
        """S6.5.2: "adding this correlation does indeed have a limited impact"."""
        assert abs(fitted.correlation) < 0.05

    def test_reports_five_parameters(self, fitted):
        assert fitted.n_parameters == 5
        assert len(list(fitted)) == 5

    def test_aic_penalises_the_extra_parameter(self, fitted):
        assert fitted.aic == pytest.approx(10 - 2 * fitted.log_likelihood)

    def test_rejects_bad_start_values(self, xtt):
        x, t_x, T = xtt
        with pytest.raises(ValueError, match="four values"):
            fit_pnbd_correlated(x, t_x, T, start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            fit_pnbd_correlated(x, t_x, T, start=(1.0, -1.0, 1.0, 1.0))

    def test_bic_penalises_more_than_aic_on_this_sample(self, fitted):
        """600 customers, so ``log(n) > 2``."""
        assert fitted.bic > fitted.aic
        assert fitted.bic == pytest.approx(
            5 * np.log(600) - 2 * fitted.log_likelihood
        )

    def test_as_dict_feeds_the_uncorrelated_expressions(self, fitted, xtt):
        """The four base parameters, for PAlive and friends."""
        x, t_x, T = xtt
        assert set(fitted.as_dict()) == {"r", "alpha", "s", "beta"}
        assert np.isfinite(log_likelihood(x, t_x, T, **fitted.as_dict()))


@pytest.mark.oracle
class TestCorrelationAgainstOracle:
    def test_likelihood_matches_at_the_oracles_own_estimates(self, xtt):
        """The formula check, independent of where either optimiser stopped."""
        x, t_x, T = xtt
        want = fixture_json("pnbd_correlation_fit")
        c = want["coefficients"]
        model = {k: c[k] for k in ("r", "alpha", "s", "beta")}
        m = m_from_correlation(c["Cor(life,trans)"], **model)

        got = correlated_log_likelihood(x, t_x, T, **model, m=m)
        assert got == pytest.approx(want["logLik"], abs=1e-9)

    @pytest.mark.slow
    def test_this_implementation_finds_a_better_optimum(self, xtt):
        r"""CLVTools 0.12.1's correlated fit is worse than its uncorrelated one.

        That cannot happen at a true optimum: the correlated model contains the
        uncorrelated one at ``m = 0``, so its maximum is at least as high. The
        reason is visible in the fitted value -- CLVTools' ``m`` sits on the
        lower Sarmanov bound, where the search has run out of room:

        ``m = -1.05226`` against a lower bound of ``-1.05235``.

        The likelihood check above shows the two implementations agree about
        the function; this one records that they disagree about its maximum,
        and which way.
        """
        x, t_x, T = xtt
        want = fixture_json("pnbd_correlation_fit")
        c = want["coefficients"]
        model = {k: c[k] for k in ("r", "alpha", "s", "beta")}
        m = m_from_correlation(c["Cor(life,trans)"], **model)

        lower, _ = correlation_bounds(**model)
        assert m == pytest.approx(lower, rel=1e-3), "expected m pinned at its bound"
        assert want["logLik"] < PLAIN_LL, "the oracle's correlated fit is worse"

        fitted = fit_pnbd_correlated(x, t_x, T)
        assert fitted.log_likelihood > want["logLik"]
        assert fitted.log_likelihood >= PLAIN_LL


# -- equality constraints, eq. (14) and S6.5.3 --------------------------------


@pytest.mark.slow
class TestEqualityConstraints:
    @pytest.fixture(scope="class")
    def constrained(self, static_data):
        return fit_pnbd_staticcov(
            static_data, names_cov_constr=["Gender"], hessian=False
        )

    @pytest.mark.oracle
    def test_matches_the_oracle(self, constrained):
        want = fixture_json("pnbd_staticcov_constrained_fit")
        assert constrained.log_likelihood == pytest.approx(want["logLik"], abs=1e-4)
        for name, value in want["coefficients"].items():
            assert constrained.coefficients()[name] == pytest.approx(
                value, rel=1e-3
            ), name

    def test_the_shared_coefficient_appears_once(self, constrained):
        """S6.5.3: "the model output only contains a single parameter value"."""
        assert constrained.names == [
            "r", "alpha", "s", "beta",
            "life.Channel", "trans.Channel", "constr.Gender",
        ]
        assert constrained.n_parameters == 7

    def test_both_processes_receive_the_same_value(self, constrained):
        r"""Eq. (14): :math:`\gamma_{purch} \equiv \gamma_{attr}`."""
        i_life = constrained.names_cov_life.index("Gender")
        i_trans = constrained.names_cov_trans.index("Gender")
        assert constrained.gamma_life[i_life] == pytest.approx(
            constrained.gamma_trans[i_trans]
        )

    def test_the_unconstrained_covariate_is_still_free(self, constrained):
        i_life = constrained.names_cov_life.index("Channel")
        i_trans = constrained.names_cov_trans.index("Channel")
        assert constrained.gamma_life[i_life] != pytest.approx(
            constrained.gamma_trans[i_trans]
        )

    def test_constraining_cannot_raise_the_likelihood(self, constrained):
        """A constrained model is nested in the unconstrained one."""
        unconstrained = fixture_json("pnbd_staticcov_fit")["logLik"]
        assert constrained.log_likelihood < unconstrained

    def test_likelihood_ratio_test_rejects_equality_for_gender(self, constrained):
        r"""S6.5.3: "A likelihood ratio test helps to evaluate if adding an
        equality constraint changes the model fit."

        On the static-covariate model the constraint on ``Gender`` is rejected,
        the same conclusion S6.5.3 reaches on the time-varying model: "the
        results show a significant difference for gender."
        """
        unconstrained = fixture_json("pnbd_staticcov_fit")["logLik"]
        statistic = 2 * (unconstrained - constrained.log_likelihood)
        p_value = float(stats.chi2.sf(statistic, df=1))
        assert statistic > 0
        assert p_value < 0.05

    def test_rejects_constraining_a_covariate_absent_from_a_process(self):
        data = ClvDataStaticCov(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_static_cov(),
            names_cov_life=["Gender"],
            names_cov_trans=["Channel"],
        )
        with pytest.raises(ValueError, match="covariate of both"):
            fit_pnbd_staticcov(data, names_cov_constr=["Gender"], hessian=False)

    def test_constraining_every_covariate_leaves_one_each(self, static_data):
        fit = fit_pnbd_staticcov(
            static_data, names_cov_constr=["Gender", "Channel"], hessian=False
        )
        assert fit.names == [
            "r", "alpha", "s", "beta", "constr.Gender", "constr.Channel",
        ]
        assert fit.n_parameters == 6


# -- regularization, eq. (13) and S6.5.1 --------------------------------------


@pytest.mark.slow
class TestRegularization:
    @pytest.mark.oracle
    @pytest.mark.parametrize("lam,fixture", [(0.1, "0_1"), (10.0, "10")])
    def test_objective_matches_the_oracle(self, static_data, lam, fixture):
        want = fixture_json(f"pnbd_staticcov_regularized_{fixture}")
        fit = fit_pnbd_staticcov(
            static_data, reg_lambdas=(lam, lam), hessian=False
        )
        assert fit.log_likelihood == pytest.approx(want["logLik"], abs=1e-6)
        for name, value in want["coefficients"].items():
            # Absolute, because heavy regularization drives coefficients to
            # near zero where a relative bound means nothing.
            assert fit.coefficients()[name] == pytest.approx(
                value, rel=1e-2, abs=1e-2
            ), name

    def test_zero_weight_reproduces_the_unpenalised_fit(self, static_data):
        plain = fit_pnbd_staticcov(static_data, hessian=False)
        zero = fit_pnbd_staticcov(static_data, reg_lambdas=(0.0, 0.0), hessian=False)
        assert zero.unpenalised_log_likelihood == pytest.approx(
            plain.log_likelihood, abs=1e-4
        )

    def test_a_heavier_weight_shrinks_the_coefficients(self, static_data):
        """S6.5.1: "The larger this regularization weight, the stronger the
        effect of the regularization"."""
        norms = []
        for lam in (0.01, 1.0, 100.0):
            fit = fit_pnbd_staticcov(
                static_data, reg_lambdas=(lam, lam), hessian=False
            )
            norms.append(
                float(np.sum(fit.gamma_life**2) + np.sum(fit.gamma_trans**2))
            )
        assert norms[0] > norms[1] > norms[2]

    def test_a_heavier_weight_lowers_the_true_likelihood(self, static_data):
        """Shrinkage trades likelihood for stability; it cannot improve fit."""
        light = fit_pnbd_staticcov(static_data, reg_lambdas=(0.01, 0.01), hessian=False)
        heavy = fit_pnbd_staticcov(static_data, reg_lambdas=(100.0, 100.0), hessian=False)
        assert heavy.unpenalised_log_likelihood < light.unpenalised_log_likelihood

    def test_the_two_processes_can_be_weighted_separately(self, static_data):
        """``reg.lambdas = c(trans = ..., life = ...)`` takes two values."""
        fit = fit_pnbd_staticcov(
            static_data, reg_lambdas=(100.0, 0.0), hessian=False
        )
        assert np.sum(fit.gamma_life**2) < np.sum(fit.gamma_trans**2)

    def test_reported_likelihood_is_the_penalised_objective(self, static_data):
        r"""The trap: with regularization on, ``log_likelihood`` is the
        penalised *mean* objective, matching what CLVTools' ``logLik()``
        returns -- roughly -9.7 rather than roughly -5821."""
        fit = fit_pnbd_staticcov(static_data, reg_lambdas=(0.1, 0.1), hessian=False)
        assert -20 < fit.log_likelihood < 0
        assert fit.unpenalised_log_likelihood < -5000
        assert fit.log_likelihood != pytest.approx(fit.unpenalised_log_likelihood)

    def test_information_criteria_use_the_true_likelihood(self, static_data):
        """AIC and BIC would be meaningless on a penalised mean."""
        fit = fit_pnbd_staticcov(static_data, reg_lambdas=(0.1, 0.1), hessian=False)
        assert fit.aic == pytest.approx(
            2 * fit.n_parameters - 2 * fit.unpenalised_log_likelihood
        )

    def test_rejects_bad_weights(self, static_data):
        with pytest.raises(ValueError, match="two values"):
            fit_pnbd_staticcov(static_data, reg_lambdas=(0.1,), hessian=False)
        with pytest.raises(ValueError, match="non-negative"):
            fit_pnbd_staticcov(static_data, reg_lambdas=(-1.0, 0.1), hessian=False)

    def test_warm_start_avoids_the_bad_basin(self, static_data):
        r"""A regression guard for the default start under regularization.

        Dividing the likelihood by ``n`` flattens the objective in the four
        model parameters by three orders of magnitude. From an all-ones start
        L-BFGS-B then reports success at ``s = 0.069, beta = 2.6`` -- a
        distinctly worse optimum than the ``s = 0.56, beta = 47`` the problem
        wants. Warm-starting from the unregularized fit avoids it.
        """
        default = fit_pnbd_staticcov(
            static_data, reg_lambdas=(10.0, 10.0), hessian=False
        )
        cold = fit_pnbd_staticcov(
            static_data, reg_lambdas=(10.0, 10.0),
            start=(1.0, 1.0, 1.0, 1.0), start_cov=0.1, hessian=False,
        )
        assert default.log_likelihood > cold.log_likelihood
        assert default.s == pytest.approx(0.56, abs=0.05)


@pytest.mark.slow
class TestTechniquesCombine:
    def test_constraints_and_regularization_together(self, static_data):
        fit = fit_pnbd_staticcov(
            static_data,
            names_cov_constr=["Gender"],
            reg_lambdas=(0.1, 0.1),
            hessian=False,
        )
        assert "constr.Gender" in fit.names
        assert fit.n_parameters == 7
        i_life = fit.names_cov_life.index("Gender")
        i_trans = fit.names_cov_trans.index("Gender")
        assert fit.gamma_life[i_life] == pytest.approx(fit.gamma_trans[i_trans])
