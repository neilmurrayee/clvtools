r"""Appendix A and S6.3 - the marginalised Pareto/NBD, against the oracle.

The fixtures here were produced by calling CLVTools' own per-customer entry
points at six parameter vectors. Three of them exist to force particular
branches:

  * ``alpha.gt.beta`` and ``alpha.lt.beta`` select the two arms of the
    :math:`A_1` / :math:`A_2` split in Appendix A;
  * ``alpha.eq.beta`` hits the degenerate case where both hypergeometrics
    collapse to 1.

Every customer with no repeat purchase has :math:`t_x = T`, which is the other
special case of :func:`log_likelihood_ind`, and 260 of the 600 apparel
customers are in it -- so it is exercised throughout rather than in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import fixture_csv, fixture_json

from clvtools.pnbd.aggregate import (
    conditional_expected_transactions,
    discounted_expected_residual_transactions,
    expectation,
    likelihood_appendix,
    log_likelihood,
    log_likelihood_ind,
    pmf,
    probability_alive,
)

MLE = dict(r=1.4490, alpha=48.6361, s=0.5613, beta=46.8844)
GRID = fixture_json("pnbd_nocov_grid")
CASES = list(GRID["params"])
#: Cases where CET and DERT are defined; the expression divides by (s - 1).
CASES_S_NE_1 = [c for c in CASES if GRID["params"][c]["s"] != 1.0]


def _inputs(case: str):
    """``(x, t_x, T)`` aligned to the fixture's row order, and its parameters."""
    fixture = fixture_csv(f"pnbd_nocov_{case}")
    cbs = fixture_csv("cbs_estimation").set_index("Id").loc[fixture["Id"]]
    p = GRID["params"][case]
    return (
        cbs["x"].to_numpy(),
        cbs["t.x"].to_numpy(),
        cbs["T.cal"].to_numpy(),
        dict(r=p["r"], alpha=p["alpha"], s=p["s"], beta=p["beta"]),
        fixture,
    )


@pytest.mark.oracle
class TestAgainstOracle:
    @pytest.mark.parametrize("case", CASES)
    def test_individual_log_likelihood(self, case):
        x, t_x, T, p, want = _inputs(case)
        np.testing.assert_allclose(
            log_likelihood_ind(x, t_x, T, **p), want["LL.ind"], rtol=1e-11
        )

    @pytest.mark.parametrize("case", CASES)
    def test_sample_log_likelihood(self, case):
        x, t_x, T, p, _ = _inputs(case)
        assert log_likelihood(x, t_x, T, **p) == pytest.approx(
            GRID["LL.sum"][case], rel=1e-11
        )

    @pytest.mark.parametrize("case", CASES)
    def test_probability_alive(self, case):
        x, t_x, T, p, want = _inputs(case)
        np.testing.assert_allclose(
            probability_alive(x, t_x, T, **p), want["PAlive"], rtol=1e-10
        )

    @pytest.mark.parametrize("case", CASES_S_NE_1)
    def test_conditional_expected_transactions(self, case):
        x, t_x, T, p, want = _inputs(case)
        got = conditional_expected_transactions(
            x, t_x, T, GRID["CET.horizon.weeks"], **p
        )
        np.testing.assert_allclose(got, want["CET"], rtol=1e-10)

    @pytest.mark.parametrize("case", CASES_S_NE_1)
    def test_discounted_expected_residual_transactions(self, case):
        x, t_x, T, p, want = _inputs(case)
        got = discounted_expected_residual_transactions(
            x, t_x, T, GRID["DERT.continuous.discount.factor"], **p
        )
        np.testing.assert_allclose(got, want["DERT"], rtol=1e-10)

    def test_pmf(self):
        want = fixture_csv("pnbd_nocov_pmf_mle")
        cbs = fixture_csv("cbs_estimation").set_index("Id").loc[want["Id"]]
        T = cbs["T.cal"].to_numpy()
        for k in range(11):
            np.testing.assert_allclose(
                pmf(k, T, **MLE), want[f"pmf.{k}"], rtol=1e-10,
                err_msg=f"PMF mismatch at k={k}",
            )

    def test_expectation(self):
        want = fixture_csv("pnbd_nocov_expectation_mle")
        np.testing.assert_allclose(
            expectation(want["t"].to_numpy(), **MLE), want["expectation"], rtol=1e-11
        )


class TestAppendixForm:
    """The literal Appendix A transcription, against the stable rearrangement."""

    @pytest.mark.parametrize("case", CASES)
    def test_agrees_with_the_stable_form_for_modest_x(self, case):
        """Restricted to ``x <= 20``: beyond that the appendix form overflows."""
        x, t_x, T, p, _ = _inputs(case)
        keep = x <= 20
        direct = likelihood_appendix(x[keep], t_x[keep], T[keep], **p)
        stable = np.exp(log_likelihood_ind(x[keep], t_x[keep], T[keep], **p))
        np.testing.assert_allclose(direct, stable, rtol=1e-9)

    def test_overflows_where_the_stable_form_does_not(self):
        r"""``Gamma(r + x)`` is why the appendix form cannot be fitted with.

        The largest apparel customer has x = 21, which is fine; a customer with
        200 repeat purchases is not, and real transaction logs contain them.
        """
        big = dict(x=200, t_x=90.0, T=104.0, **MLE)
        with np.errstate(over="ignore", invalid="ignore"):
            assert not np.isfinite(likelihood_appendix(**big))
        assert np.isfinite(log_likelihood_ind(**big))


class TestLikelihoodProperties:
    def test_zero_purchase_customers_take_the_t_x_equals_T_branch(self):
        r"""With ``x = 0`` the paper sets ``t_x = 0``, but a customer whose last
        purchase coincides with the window end has ``t_x = T``, where the
        died-in-window term is exactly zero."""
        got = log_likelihood_ind(3, 104.0, 104.0, **MLE)
        # log X + log Y, computed directly from the definition.
        from scipy import special

        r, alpha, s, beta = MLE["r"], MLE["alpha"], MLE["s"], MLE["beta"]
        log_x = (
            r * np.log(alpha) + s * np.log(beta)
            - special.gammaln(r) + special.gammaln(r + 3)
        )
        log_y = -(r + 3) * np.log(alpha + 104.0) - s * np.log(beta + 104.0)
        assert float(got) == pytest.approx(log_x + log_y, rel=1e-12)

    def test_alpha_equals_beta_is_continuous_with_its_neighbourhood(self):
        """The ``alpha == beta`` branch must be the limit of the general one."""
        base = dict(x=4, t_x=60.0, T=104.0, r=1.2, s=0.8)
        at = log_likelihood_ind(alpha=50.0, beta=50.0, **base)
        near = log_likelihood_ind(alpha=50.0, beta=50.0 + 1e-7, **base)
        assert float(at) == pytest.approx(float(near), rel=1e-7)

    def test_weights_repeat_rows(self):
        x, t_x, T = np.array([0, 2]), np.array([0.0, 40.0]), np.array([104.0, 104.0])
        single = log_likelihood(x, t_x, T, **MLE)
        weighted = log_likelihood(x, t_x, T, **MLE, weights=[2.0, 3.0])
        each = log_likelihood_ind(x, t_x, T, **MLE)
        assert weighted == pytest.approx(2 * each[0] + 3 * each[1])
        assert weighted != pytest.approx(single)

    def test_likelihood_is_maximised_at_the_published_estimates(self):
        """S6.2.1's estimates must beat perturbations of themselves."""
        cbs = fixture_csv("cbs_estimation")
        x, t_x, T = cbs["x"], cbs["t.x"], cbs["T.cal"]
        best = log_likelihood(x, t_x, T, **MLE)
        for name in MLE:
            for factor in (0.9, 1.1):
                worse = dict(MLE, **{name: MLE[name] * factor})
                assert log_likelihood(x, t_x, T, **worse) < best


class TestPAliveProperties:
    def test_lies_in_the_unit_interval(self):
        cbs = fixture_csv("cbs_estimation")
        p = probability_alive(cbs["x"], cbs["t.x"], cbs["T.cal"], **MLE)
        assert np.all((p >= 0) & (p <= 1))

    def test_is_one_when_the_last_purchase_closes_the_window(self):
        """No interval in which to have died unobserved."""
        assert float(probability_alive(5, 104.0, 104.0, **MLE)) == pytest.approx(1.0)

    def test_falls_as_recency_recedes(self):
        """A longer silence since the last purchase means a lower PAlive."""
        t_x = np.array([100.0, 80.0, 50.0, 20.0])
        p = probability_alive(3, t_x, 104.0, **MLE)
        assert np.all(np.diff(p) < 0)


class TestCetProperties:
    def test_is_zero_over_a_zero_horizon(self):
        cbs = fixture_csv("cbs_estimation")
        got = conditional_expected_transactions(
            cbs["x"], cbs["t.x"], cbs["T.cal"], 0.0, **MLE
        )
        np.testing.assert_allclose(got, 0.0, atol=1e-12)

    def test_grows_with_the_horizon(self):
        for t in (1.0, 10.0, 52.0, 200.0):
            got = float(
                conditional_expected_transactions(6, 93.285714, 104.0, t, **MLE)
            )
            assert got > 0
        horizons = np.array([1.0, 10.0, 52.0, 200.0])
        values = [
            float(conditional_expected_transactions(6, 93.285714, 104.0, t, **MLE))
            for t in horizons
        ]
        assert np.all(np.diff(values) > 0)

    def test_is_rejected_at_s_equals_one(self):
        with pytest.raises(ValueError, match="undefined at s = 1"):
            conditional_expected_transactions(1, 10.0, 104.0, 52.0, 1.0, 50.0, 1.0, 50.0)


class TestDertProperties:
    def test_a_heavier_discount_gives_a_smaller_value(self):
        args = (6, 93.285714, 104.0)
        light = float(
            discounted_expected_residual_transactions(*args, np.log(1.02) / 52, **MLE)
        )
        heavy = float(
            discounted_expected_residual_transactions(*args, np.log(1.30) / 52, **MLE)
        )
        assert heavy < light

    def test_rejects_a_nonpositive_discount_factor(self):
        with pytest.raises(ValueError, match="strictly positive"):
            discounted_expected_residual_transactions(1, 10.0, 104.0, 0.0, **MLE)


class TestExpectationProperties:
    def test_starts_at_zero(self):
        r"""S6.2.2: "The expected number of repeat transactions on this date by
        definition is zero and this fact gives the plot its characteristic
        shape"."""
        assert float(expectation(0.0, **MLE)) == pytest.approx(0.0, abs=1e-15)

    def test_is_increasing(self):
        t = np.linspace(0, 300, 200)
        assert np.all(np.diff(expectation(t, **MLE)) > 0)

    def test_is_rejected_at_s_equals_one(self):
        with pytest.raises(ValueError, match="undefined at s = 1"):
            expectation(52.0, 1.0, 50.0, 1.0, 50.0)


class TestPmfProperties:
    def test_sums_to_one(self):
        total = sum(float(pmf(k, 104.0, **MLE)) for k in range(500))
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            pmf(-1, 104.0, **MLE)

    def test_expected_counts_track_the_observed_histogram(self):
        """S6.2.2: the PMF plot compares these to the actual counts."""
        cbs = fixture_csv("cbs_estimation")
        T = cbs["T.cal"].to_numpy()
        for k in range(4):
            expected = float(np.sum(pmf(k, T, **MLE)))
            observed = int((cbs["x"] == k).sum())
            # "the results illustrate that the model fits the data well"
            assert abs(expected - observed) < 0.15 * len(cbs)
