r"""S3.2 - the individual-level Pareto/NBD.

These expressions have no oracle fixture: CLVTools never exposes the model
conditional on a single customer's :math:`(\lambda, \mu)`, only after
marginalising. So they are checked two ways instead --

  * against closed forms and identities that follow from the paper's own text,
  * by numerically mixing them over the two gamma distributions and asking that
    the result equals the marginalised expression in
    :mod:`clvtools.pnbd.aggregate`, which *is* pinned to the oracle.

The second is the load-bearing one. It ties the two halves of the model
together: if either the individual likelihood of eq. (10) or the closed form of
Appendix A were wrong, the integral would not reproduce the other.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from scipy import integrate, stats

from clvtools.pnbd.aggregate import log_likelihood_ind
from clvtools.pnbd.individual import (
    gamma_pdf_lambda,
    gamma_pdf_mu,
    individual_likelihood,
    lifetime_pdf,
    lifetime_pdf_mixed,
    likelihood_alive_at_T,
    likelihood_died_at,
    log_individual_likelihood,
    nbd_pmf,
    poisson_pmf,
)

MLE = {"r": 1.4490, "alpha": 48.6361, "s": 0.5613, "beta": 46.8844}


class TestLifetime:
    r"""Eq. (4): :math:`f(\omega \mid \mu) = \mu e^{-\mu\omega}`."""

    def test_is_a_proper_density(self):
        mass, _ = integrate.quad(lifetime_pdf, 0, np.inf, args=(0.3,))
        assert mass == pytest.approx(1.0, abs=1e-9)

    def test_matches_scipy_exponential(self):
        omega = np.array([0.0, 0.5, 2.0, 10.0])
        np.testing.assert_allclose(
            lifetime_pdf(omega, 0.3), stats.expon.pdf(omega, scale=1 / 0.3), rtol=1e-12
        )

    def test_mean_lifetime_is_one_over_mu(self):
        mean, _ = integrate.quad(lambda w: w * lifetime_pdf(w, 0.3), 0, np.inf)
        assert mean == pytest.approx(1 / 0.3, rel=1e-8)

    def test_rejects_nonpositive_rate(self):
        with pytest.raises(ValueError, match="mu must be strictly positive"):
            lifetime_pdf(1.0, 0.0)


class TestGammaHeterogeneity:
    r"""Eqs. (5) and (7) -- the two mixing distributions."""

    @pytest.mark.parametrize(
        "pdf,shape,rate",
        [(gamma_pdf_mu, MLE["s"], MLE["beta"]), (gamma_pdf_lambda, MLE["r"], MLE["alpha"])],
    )
    def test_integrates_to_one(self, pdf, shape, rate):
        mass, _ = integrate.quad(pdf, 0, np.inf, args=(shape, rate))
        assert mass == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize(
        "pdf,shape,rate",
        [(gamma_pdf_mu, MLE["s"], MLE["beta"]), (gamma_pdf_lambda, MLE["r"], MLE["alpha"])],
    )
    def test_the_scale_parameter_is_scipys_rate(self, pdf, shape, rate):
        """The paper calls it a scale parameter but writes ``e^{-x * rate}``."""
        v = np.array([0.001, 0.01, 0.1, 1.0])
        np.testing.assert_allclose(
            pdf(v, shape, rate), stats.gamma.pdf(v, a=shape, scale=1 / rate), rtol=1e-12
        )

    def test_mean_purchase_rate_matches_the_paper(self):
        r"""S6.2.1: "an average purchase rate of :math:`r/\alpha` = 0.030"."""
        mean, _ = integrate.quad(
            lambda x: x * gamma_pdf_lambda(x, MLE["r"], MLE["alpha"]), 0, np.inf
        )
        assert mean == pytest.approx(MLE["r"] / MLE["alpha"], rel=1e-8)
        assert round(mean, 3) == 0.030

    def test_mean_attrition_rate_matches_the_paper(self):
        r"""S6.2.1: "an average attrition rate of :math:`s/\beta` = 0.012"."""
        mean, _ = integrate.quad(
            lambda x: x * gamma_pdf_mu(x, MLE["s"], MLE["beta"]), 0, np.inf
        )
        assert mean == pytest.approx(MLE["s"] / MLE["beta"], rel=1e-8)
        assert round(mean, 3) == 0.012


class TestParetoSecondKind:
    r"""Eq. (6) -- mixing eq. (4) over eq. (5) gives a Pareto of the second kind."""

    def test_equals_the_numerical_mixture(self):
        s, beta = MLE["s"], MLE["beta"]
        for omega in (0.0, 1.0, 25.0, 200.0):
            mixed, _ = integrate.quad(
                lambda mu, omega=omega: lifetime_pdf(omega, mu) * gamma_pdf_mu(mu, s, beta),
                0, np.inf,
            )
            assert lifetime_pdf_mixed(omega, s, beta) == pytest.approx(mixed, rel=1e-7)

    def test_is_a_proper_density(self):
        mass, _ = integrate.quad(
            lifetime_pdf_mixed, 0, np.inf, args=(MLE["s"], MLE["beta"])
        )
        assert mass == pytest.approx(1.0, abs=1e-8)

    def test_matches_scipy_lomax(self):
        omega = np.array([0.0, 5.0, 50.0])
        np.testing.assert_allclose(
            lifetime_pdf_mixed(omega, MLE["s"], MLE["beta"]),
            stats.lomax.pdf(omega, c=MLE["s"], scale=MLE["beta"]),
            rtol=1e-12,
        )


class TestTransactionCounts:
    r"""Eqs. (7) and (8) -- Poisson, and the NBD it mixes to."""

    def test_poisson_matches_scipy(self):
        k = np.arange(12)
        np.testing.assert_allclose(
            poisson_pmf(k, 0.3, 20.0), stats.poisson.pmf(k, 0.3 * 20.0), rtol=1e-11
        )

    def test_poisson_is_one_at_zero_time_and_zero_count(self):
        assert poisson_pmf(0, 0.3, 0.0) == pytest.approx(1.0)

    def test_nbd_equals_the_numerical_mixture(self):
        r, alpha, t = MLE["r"], MLE["alpha"], 104.0
        for k in (0, 1, 3, 8):
            mixed, _ = integrate.quad(
                lambda lam, k=k: poisson_pmf(k, lam, t) * gamma_pdf_lambda(lam, r, alpha),
                0, np.inf,
            )
            assert nbd_pmf(k, t, r, alpha) == pytest.approx(mixed, rel=1e-7)

    def test_nbd_matches_scipy_negative_binomial(self):
        r, alpha, t = MLE["r"], MLE["alpha"], 104.0
        k = np.arange(20)
        np.testing.assert_allclose(
            nbd_pmf(k, t, r, alpha),
            stats.nbinom.pmf(k, n=r, p=alpha / (alpha + t)),
            rtol=1e-10,
        )

    def test_nbd_sums_to_one(self):
        total = np.sum(nbd_pmf(np.arange(2000), 104.0, MLE["r"], MLE["alpha"]))
        assert total == pytest.approx(1.0, abs=1e-9)


class TestConditionalLikelihoods:
    """Eqs. (8) and (9) -- alive at T, versus dead at some omega."""

    def test_the_two_expressions_differ_only_in_their_time_argument(self):
        assert likelihood_died_at(3, 7.0, 0.4) == likelihood_alive_at_T(3, 7.0, 0.4)

    def test_alive_case_is_the_exponential_survivor_times_the_rate_powers(self):
        x, T, lam = 4, 12.0, 0.25
        assert likelihood_alive_at_T(x, T, lam) == pytest.approx(
            lam**x * np.exp(-lam * T)
        )


class TestIndividualLikelihood:
    """Eq. (10) -- the two cases combined and weighted by their probabilities."""

    def test_is_the_sum_of_the_two_weighted_cases(self):
        r"""Eq. (10) is

        .. math::
            L(\lambda \mid x, T, \omega > T) P(\omega > T \mid \mu)
            + \int_{t_x}^{T} L(\lambda \mid x, \omega) f(\omega \mid \mu)\, d\omega

        which is what the closed form must equal.
        """
        x, t_x, T, lam, mu = 3, 6.0, 10.0, 0.35, 0.08

        survived = likelihood_alive_at_T(x, T, lam) * np.exp(-mu * T)
        died, _ = integrate.quad(
            lambda w: likelihood_died_at(x, w, lam) * lifetime_pdf(w, mu), t_x, T
        )
        assert individual_likelihood(x, t_x, T, lam, mu) == pytest.approx(
            survived + died, rel=1e-9
        )

    def test_collapses_to_the_alive_case_as_mu_vanishes(self):
        x, t_x, T, lam = 2, 8.0, 10.0, 0.3
        assert individual_likelihood(x, t_x, T, lam, 1e-14) == pytest.approx(
            likelihood_alive_at_T(x, T, lam), rel=1e-9
        )

    def test_log_form_agrees_with_the_direct_form(self):
        x = np.array([0, 1, 5, 20])
        t_x = np.array([0.0, 3.0, 40.0, 90.0])
        T = np.full(4, 104.0)
        np.testing.assert_allclose(
            log_individual_likelihood(x, t_x, T, 0.05, 0.01),
            np.log(individual_likelihood(x, t_x, T, 0.05, 0.01)),
            rtol=1e-12,
        )

    def test_log_form_survives_where_the_direct_form_underflows(self):
        """A long window and a high rate drive both terms below 1e-308."""
        x, t_x, T, lam, mu = 5, 900.0, 1000.0, 2.0, 1.5
        assert individual_likelihood(x, t_x, T, lam, mu) == 0.0
        got = log_individual_likelihood(x, t_x, T, lam, mu)
        assert np.isfinite(got)
        assert got < -1000

    def test_rejects_nonpositive_rates(self):
        with pytest.raises(ValueError, match="positive"):
            individual_likelihood(1, 1.0, 2.0, 0.0, 0.1)


class TestMarginalisation:
    r"""The tie between S3.2 and Appendix A.

    Integrating eq. (10) over :math:`g(\lambda)g(\mu)` must reproduce the
    closed form. Since the closed form is pinned to CLVTools' own output, this
    also confirms the individual-level expression the paper states.
    """

    # The integrand is sharply peaked -- a gamma of shape ~1.4 and rate ~49,
    # multiplied by lambda^x e^{-lambda T} with T = 104 -- so quad's defaults
    # are not enough: at x = 6 they are out by 17%. These settings bring every
    # case below 1e-9, which is what makes the agreement meaningful.
    _QUAD: ClassVar[dict[str, float]] = {
        "limit": 500, "epsabs": 1e-16, "epsrel": 1e-13,
    }

    @pytest.mark.parametrize(
        "x,t_x,T",
        [(0, 0.0, 104.0), (1, 20.0, 104.0), (6, 93.285714, 104.0), (3, 104.0, 104.0)],
    )
    def test_double_integral_reproduces_the_closed_form(self, x, t_x, T):
        r, alpha, s, beta = MLE["r"], MLE["alpha"], MLE["s"], MLE["beta"]

        def inner(mu):
            f, _ = integrate.quad(
                lambda lam: individual_likelihood(x, t_x, T, lam, mu)
                * gamma_pdf_lambda(lam, r, alpha),
                0, np.inf, **self._QUAD,
            )
            return f * gamma_pdf_mu(mu, s, beta)

        marginal, _ = integrate.quad(inner, 0, np.inf, **self._QUAD)
        want = np.exp(log_likelihood_ind(x, t_x, T, r, alpha, s, beta))
        assert marginal == pytest.approx(float(want), rel=1e-8)

    def test_appendix_integrand_with_the_stray_mu_does_not_reproduce_it(self):
        r"""Appendix A prints :math:`\frac{\lambda^{x+1}\mu}{\lambda+\mu}` in the
        integrand's second term. Eq. (10) has no :math:`\mu` there. Integrating
        the appendix's version gives a different -- wrong -- answer, which is
        why :func:`individual_likelihood` follows eq. (10).
        """
        x, t_x, T = 6, 93.285714, 104.0
        r, alpha, s, beta = MLE["r"], MLE["alpha"], MLE["s"], MLE["beta"]

        def with_stray_mu(lam, mu):
            total = lam + mu
            return (
                lam**x * mu / total * np.exp(-total * t_x)
                + lam ** (x + 1) * mu / total * np.exp(-total * T)
            )

        def inner(mu):
            f, _ = integrate.quad(
                lambda lam: with_stray_mu(lam, mu) * gamma_pdf_lambda(lam, r, alpha),
                0, np.inf, **self._QUAD,
            )
            return f * gamma_pdf_mu(mu, s, beta)

        marginal, _ = integrate.quad(inner, 0, np.inf, **self._QUAD)
        want = float(np.exp(log_likelihood_ind(x, t_x, T, r, alpha, s, beta)))
        assert marginal != pytest.approx(want, rel=1e-3)
