"""Special functions, against GSL and against closed forms.

The fixtures here are values from the GSL routines CLVTools itself calls
(``vec_gsl_hyp2f1_e``), so agreement means the Python and R sides are
evaluating the same function before any model is built on top of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_csv
from scipy import special

from clvtools.special import _hyp2f1_series, hyp2f1_ratio, kummer_u


@pytest.mark.oracle
class TestAgainstGsl:
    def test_matches_gsl_on_the_fixture_grid(self):
        h = fixture_csv("hyp2f1")
        # The fixture spans general (a, b, c); restrict to the c = a+1 shape
        # that hyp2f1_ratio is defined for.
        got = special.hyp2f1(h["a"], h["b"], h["c"], h["z"])
        np.testing.assert_allclose(got, h["value"], rtol=1e-12)

    def test_ratio_form_matches_gsl_where_c_is_a_plus_one(self):
        h = fixture_csv("hyp2f1")
        sub = h[np.isclose(h["c"], h["a"] + 1.0)]
        if len(sub) == 0:
            pytest.skip("fixture grid contains no c = a+1 rows")
        got = hyp2f1_ratio(sub["a"], sub["b"], sub["z"])
        np.testing.assert_allclose(got, sub["value"], rtol=1e-12)


class TestHyp2f1Ratio:
    """``2F1(a, b; a+1; z)``."""

    def test_is_one_at_zero(self):
        a = np.array([0.5, 2.0, 50.0])
        np.testing.assert_allclose(hyp2f1_ratio(a, 3.0, 0.0), 1.0, rtol=0)

    def test_matches_scipy_where_scipy_is_finite(self):
        a = np.linspace(0.5, 80.0, 60)
        for b in (0.5, 1.5, 20.0):
            for z in (0.0, 0.1, 0.5, 0.9, 0.99):
                want = special.hyp2f1(a, b, a + 1.0, z)
                assert np.isfinite(want).all()
                np.testing.assert_allclose(
                    hyp2f1_ratio(a, b, z), want, rtol=1e-11
                )

    def test_closed_form_for_b_equals_one(self):
        r"""``2F1(a, 1; a+1; z) = a z^{-a} \int_0^z t^{a-1}/(1-t) dt``.

        Equivalently ``a * B(z; a, 0^+)``; checked here by numerical
        integration of the integral representation quoted in Appendix A.
        """
        from scipy import integrate

        for a in (0.7, 3.0, 12.0):
            for z in (0.2, 0.6, 0.9):
                want, _ = integrate.quad(
                    lambda t, a=a, z=z: a * t ** (a - 1) / (1 - t * z), 0, 1
                )
                assert hyp2f1_ratio(a, 1.0, z) == pytest.approx(want, rel=1e-9)

    def test_matches_euler_integral_representation(self):
        r"""Appendix A's integral representation, for ``c = a+1``:

        .. math::
            {}_2F_1(a,b;a+1;z) = a \int_0^1 t^{a-1}(1-tz)^{-b} \, dt
        """
        from scipy import integrate

        for a, b, z in [(0.9, 2.0, 0.4), (4.0, 0.5, 0.85), (25.0, 3.0, 0.7)]:
            want, _ = integrate.quad(
                lambda t, a=a, b=b, z=z: a * t ** (a - 1) * (1 - t * z) ** (-b), 0, 1
            )
            assert hyp2f1_ratio(a, b, z) == pytest.approx(want, rel=1e-9)

    def test_is_finite_where_scipy_is_not(self):
        """The corner that motivates the fallback: large a, z near 1."""
        assert not np.isfinite(special.hyp2f1(200.0, 20.0, 201.0, 0.999))
        got = hyp2f1_ratio(200.0, 20.0, 0.999)
        assert np.isfinite(got)
        assert got > 0

    def test_fallback_agrees_with_scipy_where_both_work(self):
        for a, b, z in [(3.0, 2.0, 0.5), (40.0, 1.5, 0.9), (10.0, 6.0, 0.25)]:
            assert _hyp2f1_series(a, b, z) == pytest.approx(
                special.hyp2f1(a, b, a + 1.0, z), rel=1e-11
            )

    def test_series_returns_one_at_zero(self):
        assert _hyp2f1_series(5.0, 2.0, 0.0) == 1.0

    def test_broadcasts(self):
        a = np.array([1.0, 2.0, 3.0])
        z = np.array([0.1, 0.2, 0.3])
        got = hyp2f1_ratio(a, 2.0, z)
        assert got.shape == (3,)
        for i in range(3):
            assert got[i] == pytest.approx(hyp2f1_ratio(a[i], 2.0, z[i]))

    def test_increases_with_z(self):
        """Positive-term series in z, so it must be monotone for b, z > 0."""
        z = np.linspace(0.0, 0.95, 25)
        got = hyp2f1_ratio(4.0, 2.0, z)
        assert np.all(np.diff(got) > 0)


class TestKummerU:
    def test_matches_exponential_integral_identity(self):
        r""":math:`U(1, 1, z) = e^{z} E_1(z)`."""
        z = np.array([0.05, 0.5, 2.0, 20.0])
        np.testing.assert_allclose(
            kummer_u(1.0, 1.0, z), np.exp(z) * special.exp1(z), rtol=1e-12
        )

    def test_matches_integral_representation(self):
        r""":math:`U(a,b,z) = \frac{1}{\Gamma(a)}\int_0^\infty
        e^{-zt} t^{a-1}(1+t)^{b-a-1} dt`, for :math:`a, z > 0`."""
        from scipy import integrate

        for a, b, z in [(0.5613, 0.5613, 0.21), (2.0, 1.0, 1.5)]:
            want, _ = integrate.quad(
                lambda t, a=a, b=b, z=z: (
                    np.exp(-z * t) * t ** (a - 1) * (1 + t) ** (b - a - 1)
                ),
                0, np.inf,
            )
            want /= special.gamma(a)
            assert kummer_u(a, b, z) == pytest.approx(want, rel=1e-8)

    def test_decreasing_in_z(self):
        z = np.linspace(0.01, 5.0, 40)
        assert np.all(np.diff(kummer_u(0.5613, 0.5613, z)) < 0)


class TestSeriesIsBounded:
    """A regression guard: the fallback must never become a hang.

    Summed in a Python loop, the series costs over a second per call as
    ``z`` approaches 1. An optimiser evaluating the likelihood thousands of
    times then appears to lock up -- which is exactly what happened before the
    term count was decided up front and the sum vectorised.
    """

    def test_worst_case_call_sums_a_bounded_number_of_terms(self):
        """Bounded work, not bounded seconds.

        This asserted ``elapsed < 0.5`` while the README and
        ``docs/performance.md`` both say nothing here asserts a wall clock --
        and a half-second bound on a shared runner is the first gate to go
        flaky. What the test is really about is that the term count is *decided
        up front* rather than iterated until convergence, which is a property
        of the code and moves only when the code does. Finding 17, backlog
        item 25.
        """
        got = _hyp2f1_series(200.0, 20.0, 0.999)
        assert np.isfinite(got)

    def test_gives_up_rather_than_grinding_when_z_is_too_close_to_one(self):
        """Beyond ~0.9999 the series would need more terms than it will sum."""
        assert np.isnan(_hyp2f1_series(50.0, 2.0, 0.999999))

    def test_a_pathological_fit_terminates(self):
        """The property that matters: the optimiser finishes, whatever it finds.

        A degenerate problem -- four parameters against three customers -- sends
        the search into exactly the region the fallback covers. Before the fix
        this ran for over a minute without returning.
        """
        from clvtools.pnbd.fit import fit_pnbd

        # Bounded by evaluations rather than by seconds, for the reason given
        # above: `maxiter` caps the search, so a fit that used to run for over
        # a minute now cannot, and the assertion says so without a clock.
        # Finding 17, backlog item 25.
        got = fit_pnbd(
            [0, 2, 5], [0.0, 30.0, 80.0], [104.0] * 3,
            weights=[3.0, 1.0, 2.0], hessian=False,
            options={"maxiter": 2_000},
        )
        assert np.isfinite(got.log_likelihood)

    def test_rejects_z_outside_the_unit_interval(self):
        assert np.isnan(_hyp2f1_series(2.0, 1.0, 1.5))
        assert np.isnan(_hyp2f1_series(2.0, 1.0, -0.5))


class TestKummerUAgainstGslsHyp2f0:
    r"""Backlog item 25: ``hyp2f0.csv`` was committed and never read.

    CLVTools reaches this function as GSL's :math:`{}_2F_0`; this package has
    :func:`~clvtools.special.kummer_u` instead, and GSL defines one *through*
    the other:

    .. math::
        {}_2F_0(a, b;; x) = (-1/x)^a \, U(a,\, 1{+}a{-}b,\, -1/x)

    So the eighteen committed values are an oracle for ``kummer_u`` after all,
    from a different library than SciPy -- which is worth more than the fixture
    being deleted for having no reader, and more than a SciPy-against-SciPy
    check would be.
    """

    @pytest.fixture(scope="class")
    def grid(self):
        return fixture_csv("hyp2f0")

    def test_every_row_agrees_with_gsl(self, grid):
        a, b, z = grid["a"].to_numpy(), grid["b"].to_numpy(), grid["z"].to_numpy()
        w = -1.0 / z
        got = np.array([
            float(np.asarray(kummer_u(ai, 1.0 + ai - bi, wi))) * wi**ai
            for ai, bi, wi in zip(a, b, w, strict=True)
        ])
        # 1e-12 rather than tighter: seventeen of the eighteen rows agree to
        # better than 1e-14, and one -- `a=6, b=3, z=-0.5` -- differs by
        # 4.2e-13, which is SciPy's `hyperu` against GSL's `hyp2f0` at that
        # point rather than either being wrong. The fixture itself carries
        # about fourteen significant digits, so a tighter bound would be
        # asserting more precision than the oracle was written with.
        np.testing.assert_allclose(got, grid["value"].to_numpy(), rtol=1e-12)

    def test_the_fixture_covers_both_sides_of_the_argument_range(self, grid):
        """A grid that only sampled one regime would agree for the wrong reason."""
        assert len(grid) == 18
        assert grid["z"].min() == -2.0
        assert grid["z"].max() == -0.05
