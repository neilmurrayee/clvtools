r"""Table 2's model details, and the tests that go with them.

``vcov()``, ``confint()`` and ``summary()`` all descend from the Hessian of the
negative log-likelihood at the optimum. Two implementations that agree about
the likelihood can still disagree about its curvature if they difference it at
different points, so every comparison here is made at *CLVTools' own* fitted
coefficients rather than at this package's -- the same discipline the paper
numbers get elsewhere.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from conftest import fixture_json

from clvtools import ClvData, bgnbd, ggomnbd
from clvtools.gg import log_likelihood as gg_log_likelihood
from clvtools.inference import (
    likelihood_ratio_test,
    numerical_hessian,
)
from clvtools.pnbd import fit_pnbd, fit_pnbd_staticcov
from clvtools.pnbd.aggregate import log_likelihood as pnbd_log_likelihood


@pytest.fixture(scope="module")
def data(apparel_trans) -> ClvData:
    return ClvData(apparel_trans, time_unit="week", estimation_split=104)


def _cbs(data):
    cbs = data.customer_summary()
    return cbs["x"].to_numpy(), cbs["t_x"].to_numpy(), cbs["T"].to_numpy()


class TestCurvatureAgainstTheOracle:
    """The Hessian itself, at the oracle's parameters."""

    @pytest.mark.oracle
    def test_pareto_nbd_covariance(self, data):
        want = fixture_json("inference_pnbd")
        x, t_x, T = _cbs(data)
        at = np.array([want["coefficients"][n] for n in want["names"]])
        hessian = numerical_hessian(
            lambda v: -pnbd_log_likelihood(x, t_x, T, *v), at
        )
        got = np.linalg.inv(hessian)
        # Both sides difference the same likelihood numerically, so they agree
        # to the accuracy of that differencing rather than to machine
        # precision: numDeriv extrapolates from wide steps, this takes one
        # narrow step. See numerical_hessian for why 1e-4 is the step.
        np.testing.assert_allclose(
            got.ravel(), np.array(want["vcov"]), rtol=2e-3
        )
        np.testing.assert_allclose(
            np.sqrt(np.diag(got)), np.array(want["se"]), rtol=1e-3
        )

    @pytest.mark.oracle
    def test_gamma_gamma_covariance(self, data):
        want = fixture_json("inference_gg")
        spend = data.spending_summary()
        at = np.array([want["coefficients"][n] for n in want["names"]])
        hessian = numerical_hessian(
            lambda v: -gg_log_likelihood(
                spend["x"].to_numpy(), spend["Spending"].to_numpy(), *v
            ),
            at,
        )
        np.testing.assert_allclose(
            np.sqrt(np.diag(np.linalg.inv(hessian))),
            np.array(want["se"]),
            rtol=1e-4,
        )

    @pytest.mark.oracle
    @pytest.mark.parametrize("family,name,which", [
        (bgnbd, "bgnbd", ["r", "alpha", "a", "b"]),
        # The GGom/NBD's b and beta are excluded deliberately. Its fit on this
        # data is degenerate -- b comes out at 8.1e-07, on the path along which
        # the model collapses to the Pareto/NBD -- so the likelihood is flat in
        # both, and the curvature there is numerical noise in either
        # implementation: CLVTools reports 7.49e-04 for each, and differencing
        # at three step sizes here gives three different answers. The three
        # parameters that are identified agree.
        (ggomnbd, "ggomnbd", ["r", "alpha", "s"]),
    ])
    def test_other_families_covariance(self, data, family, name, which):
        want = fixture_json(f"inference_{name}")
        x, t_x, T = _cbs(data)
        at = np.array([want["coefficients"][n] for n in want["names"]])
        hessian = numerical_hessian(
            lambda v: -family.log_likelihood(x, t_x, T, *v), at
        )
        se = np.sqrt(np.diag(np.linalg.inv(hessian)))
        got = dict(zip(want["names"], se, strict=True))
        expected = dict(zip(want["names"], want["se"], strict=True))
        for parameter in which:
            assert got[parameter] == pytest.approx(
                expected[parameter], rel=2e-3
            ), parameter


@pytest.mark.slow
class TestGenerics:
    """What the fits themselves report."""

    @staticmethod
    @pytest.fixture(scope="class")
    def fit(apparel_trans):
        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        x, t_x, T = _cbs(data)
        return fit_pnbd(x, t_x, T)

    @pytest.mark.oracle
    def test_standard_errors_match(self, fit):
        want = fixture_json("inference_pnbd")
        got = fit.standard_errors()
        for name, expected in zip(want["names"], want["se"], strict=True):
            assert got[name] == pytest.approx(expected, rel=2e-3), name

    @pytest.mark.oracle
    def test_confint_matches(self, fit):
        want = fixture_json("inference_pnbd")
        got = fit.confint()
        np.testing.assert_allclose(
            got.iloc[:, 0].to_numpy(), np.array(want["confint.lower"]), rtol=5e-3
        )
        np.testing.assert_allclose(
            got.iloc[:, 1].to_numpy(), np.array(want["confint.upper"]), rtol=5e-3
        )

    def test_confint_is_the_wald_interval(self, fit):
        interval = fit.confint()
        errors = fit.standard_errors()
        for name in fit.names:
            half = 1.959963984540054 * errors[name]
            centre = fit.coefficients[name]
            assert interval.loc[name].iloc[0] == pytest.approx(centre - half)
            assert interval.loc[name].iloc[1] == pytest.approx(centre + half)

    def test_confint_level_widens_the_interval(self, fit):
        narrow, wide = fit.confint(0.5), fit.confint(0.99)
        assert (wide.iloc[:, 0] < narrow.iloc[:, 0]).all()
        assert (wide.iloc[:, 1] > narrow.iloc[:, 1]).all()
        assert list(narrow.columns) == ["25 %", "75 %"]

    def test_confint_rejects_an_impossible_level(self, fit):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            fit.confint(1.0)

    def test_vcov_is_symmetric_and_matches_the_errors(self, fit):
        cov = fit.vcov()
        np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T, rtol=1e-12)
        np.testing.assert_allclose(
            np.sqrt(np.diag(cov.to_numpy())),
            [fit.standard_errors()[n] for n in fit.names],
            rtol=1e-12,
        )
        assert list(cov.index) == fit.names == list(cov.columns)

    def test_summary_leaves_z_values_out_for_bounded_parameters(self, fit):
        table = fit.summary()
        assert table["z-val"].isna().all()
        assert list(table.columns) == [
            "Estimate", "Std. Error", "z-val", "Pr(>|z|)"
        ]

    def test_coefficients_are_named_in_order(self, fit):
        assert list(fit.coefficients) == fit.names
        np.testing.assert_allclose(list(fit.coefficients.values()), list(fit))

    def test_without_a_hessian_the_generics_say_so(self, apparel_trans):
        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        x, t_x, T = _cbs(data)
        bare = fit_pnbd(x, t_x, T, hessian=False)
        for call in (bare.standard_errors, bare.vcov, bare.confint, bare.summary):
            with pytest.raises(ValueError, match="hessian=True"):
                call()


@pytest.mark.slow
class TestCovariateSummary:
    """S6.4.1's coefficient table, including under an equality constraint."""

    @staticmethod
    @pytest.fixture(scope="class")
    def constrained(static_data):
        return fit_pnbd_staticcov(static_data, names_cov_constr=["Gender"])

    @pytest.mark.oracle
    def test_constrained_errors_line_up_with_their_names(self, constrained):
        """The Hessian is over the parameters actually estimated.

        A constrained covariate contributes one coefficient, not two. Taking
        curvature over the full unconstrained vector instead would produce one
        standard error too many, and every covariate name would take the
        neighbouring parameter's error.
        """
        want = fixture_json("inference_pnbd_staticcov_constrained")
        assert constrained.names == want["names"]
        got = constrained.standard_errors()
        for name, expected in zip(want["names"], want["se"], strict=True):
            assert got[name] == pytest.approx(expected, rel=5e-2), name

    def test_the_constrained_coefficient_is_reported_once(self, constrained):
        assert "constr.Gender" in constrained.names
        assert "life.Gender" not in constrained.names
        assert "trans.Gender" not in constrained.names

    @pytest.mark.oracle
    def test_covariates_carry_z_and_p_values(self, constrained):
        table = constrained.summary()
        covariates = [n for n in constrained.names if "." in n]
        assert table.loc[covariates, "z-val"].notna().all()
        assert table.loc[["r", "alpha", "s", "beta"], "z-val"].isna().all()


@pytest.mark.slow
class TestLikelihoodRatioTest:
    """S6.5.3's ``lrtest()``."""

    @staticmethod
    @pytest.fixture(scope="class")
    def fits(static_data):
        free = fit_pnbd_staticcov(static_data, hessian=False)
        tied = fit_pnbd_staticcov(
            static_data, names_cov_constr=["Gender"], hessian=False
        )
        return tied, free

    @pytest.mark.oracle
    def test_matches_the_oracle(self, fits):
        tied, free = fits
        want = fixture_json("lrtest_pnbd_staticcov")
        got = likelihood_ratio_test(tied, free)
        assert got.df == want["df"]
        assert got.n_parameters_restricted == want["n.parameters.restricted"]
        assert got.n_parameters_unrestricted == want["n.parameters.unrestricted"]
        assert got.statistic == pytest.approx(want["chisq"], rel=1e-3)
        assert got.p_value == pytest.approx(want["p.value"], rel=1e-2)

    def test_the_constraint_is_rejected(self, fits):
        """S6.5.3 concludes the two processes' Gender effects differ."""
        assert likelihood_ratio_test(*fits).p_value < 0.01

    def test_statistic_is_twice_the_likelihood_gap(self, fits):
        tied, free = fits
        got = likelihood_ratio_test(tied, free)
        assert got.statistic == pytest.approx(
            2 * (free.log_likelihood - tied.log_likelihood)
        )

    def test_rejects_models_the_wrong_way_round(self, fits):
        tied, free = fits
        with pytest.raises(ValueError, match="more parameters"):
            likelihood_ratio_test(free, tied)

    def test_repr_names_the_statistic(self, fits):
        assert "chisq" in repr(likelihood_ratio_test(*fits))


class TestEveryFamilyExposesTheSameAccessors:
    """Table 2's generics are "available for all fitted models"."""

    @staticmethod
    def _fits():
        from clvtools.bgnbd import BgnbdParams
        from clvtools.gg import GgParams
        from clvtools.ggomnbd import GgomnbdParams
        from clvtools.pnbd.correlation import PnbdCorrelatedParams
        from clvtools.pnbd.fit import PnbdParams

        shared = {"log_likelihood": -1.0, "converged": True, "n_customers": 600}
        return {
            "pnbd": (
                PnbdParams(r=1.0, alpha=2.0, s=3.0, beta=4.0, **shared),
                ["r", "alpha", "s", "beta"],
            ),
            "bgnbd": (
                BgnbdParams(r=1.0, alpha=2.0, a=3.0, b=4.0, **shared),
                ["r", "alpha", "a", "b"],
            ),
            "ggomnbd": (
                GgomnbdParams(r=1.0, alpha=2.0, b=3.0, s=4.0, beta=5.0, **shared),
                ["r", "alpha", "b", "s", "beta"],
            ),
            "gg": (
                GgParams(p=1.0, q=2.0, gamma=3.0, **shared),
                ["p", "q", "gamma"],
            ),
            "correlated": (
                PnbdCorrelatedParams(
                    r=1.0, alpha=2.0, s=3.0, beta=4.0, m=0.1, **shared
                ),
                ["r", "alpha", "s", "beta", "m"],
            ),
        }

    @pytest.mark.parametrize("family", list(_fits.__func__()))
    def test_names_match_the_estimates(self, family):
        fit, expected = self._fits()[family]
        assert fit.names == expected
        assert list(fit.coefficients) == expected
        np.testing.assert_allclose(list(fit.coefficients.values()), list(fit))

    @pytest.mark.parametrize("family", list(_fits.__func__()))
    def test_without_a_hessian_they_all_say_so(self, family):
        fit, _ = self._fits()[family]
        with pytest.raises(ValueError, match="hessian=True"):
            fit.standard_errors()


class TestTheRatioTestNeedsARealRestriction:
    """Finding 12: it accepted two non-nested models and reported a chi-square.

    A restricted model is a special case of the unrestricted one, so it cannot
    fit better. A negative statistic is therefore the observable signature of
    either non-nesting -- where the chi-square means nothing -- or an
    unrestricted fit that stopped somewhere worse. Both are worth knowing
    before reading a p-value, and neither used to say anything.
    """

    def test_the_arguments_the_wrong_way_round_raise(self):
        from clvtools.inference import likelihood_ratio_test

        restricted = SimpleNamespace(
            n_parameters=7, log_likelihood=-5821.0, n_customers=600)
        unrestricted = SimpleNamespace(
            n_parameters=8, log_likelihood=-5826.5, n_customers=600)
        with pytest.raises(ValueError, match="restricted model fits better"):
            likelihood_ratio_test(restricted, unrestricted)

    def test_two_different_samples_raise(self):
        from clvtools.inference import likelihood_ratio_test

        with pytest.raises(ValueError, match="same data"):
            likelihood_ratio_test(
                SimpleNamespace(n_parameters=7, log_likelihood=-5826.5, n_customers=600),
                SimpleNamespace(n_parameters=8, log_likelihood=-5821.0, n_customers=599),
            )


class TestAHessianThatCannotBeTrusted:
    """Finding 9: NaN standard errors used to ship with ``converged = True``.

    Both branches here are about saying so. Neither changes a number: what
    changes is that a caller reading `life.Gender = nan` beside
    `life.Channel = 0.594` is told why, instead of having to know that the
    BG/NBD's beta parameters are barely identified under covariates.
    """

    def test_an_indefinite_hessian_warns_and_names_the_flat_directions(self):
        from clvtools._validate import ConvergenceWarning
        from clvtools.inference import Fitted

        class Toy(Fitted):
            names: ClassVar[list[str]] = ["good", "flat"]
            # Not positive definite: the second direction curves the wrong way.
            hessian = np.array([[4.0, 0.0], [0.0, -1.0]])

            def __iter__(self):
                return iter([1.0, 2.0])

        with pytest.warns(ConvergenceWarning, match="not positive definite"):
            errors = Toy().standard_errors()
        assert np.isfinite(errors["good"])
        assert np.isnan(errors["flat"])

    def test_a_non_finite_hessian_warns_rather_than_raising_from_numpy(self):
        """``eigvalsh`` raises ``LinAlgError`` on a NaN matrix, which is not an
        answer. The GGom/NBD covariate fit produces one: its ``b`` is 8.1e-07
        and the surface there cannot be differenced at any step this package
        uses."""
        from clvtools._validate import ConvergenceWarning
        from clvtools.inference import Fitted

        class Toy(Fitted):
            names: ClassVar[list[str]] = ["a", "b"]
            hessian = np.array([[1.0, np.nan], [np.nan, 1.0]])

            def __iter__(self):
                return iter([1.0, 2.0])

        with pytest.warns(ConvergenceWarning, match="non-finite entries"):
            errors = Toy().standard_errors()
        assert all(np.isnan(v) for v in errors.values())
