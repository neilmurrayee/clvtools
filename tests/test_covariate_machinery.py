r"""The covariate machinery every family shares.

The time-invariant covariate fit is one implementation in
:mod:`clvtools._staticcov`, which each family reaches through its own
``fit_*_staticcov``. What is tested here is that shared part rather than any
one family's equations: S6.5.3's equality constraint of eq. (14), S6.5.1's
regularization of eq. (13), the validation both go through, and the accessors
:class:`~clvtools._staticcov.StaticCovResult` exposes.

Split out of ``test_families.py``, which had grown to three lines under the
module-size gate. What stayed there is what is specific to the BG/NBD and the
GGom/NBD; what moved here is what neither owns.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_json

from clvtools import bgnbd, ggomnbd


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
