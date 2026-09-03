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

from typing import ClassVar

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


class TestTheThreeFamiliesAnswerTheSameQuestions:
    """Backlog item 27, finding 19: the families had diverged around this code.

    The Pareto/NBD's covariate result carries its estimates as its own fields;
    the BG/NBD's and the GGom/NBD's hold a
    :class:`~clvtools._staticcov.StaticCovResult` and forward to it. That
    difference is deliberate and documented on
    :class:`~clvtools._staticcov.DelegatesToCovariates`. What was *not*
    deliberate is that the forwarding was incomplete: ``names_cov_constr`` and
    ``reg_lambdas`` sat on ``StaticCovResult`` and reached no family that used
    it, so "which covariates are tied?" and "are these ridge standard errors?"
    could be asked of one family's fit and not the other two's.

    **Built rather than fitted**, like ``TestCovariateResultAccessors`` above.
    The code under test is the forwarding, and a fit exercises the optimiser
    instead -- which the rest of this module already does. Two earlier drafts
    fitted: over the apparel cohort it took the suite from 3:45 to 9:32, and
    over twelve synthetic customers a single unregularized Pareto/NBD fit was
    still 10.6 s, because a cohort that small leaves the covariates
    unidentified and the search wanders. Neither was testing anything this
    does not.
    """

    #: Everything a caller should be able to ask any covariate fit, regardless
    #: of which family produced it. Extend this rather than adding a test.
    SHARED = (
        "names_cov_life", "names_cov_trans", "names_cov_constr", "reg_lambdas",
        "gamma_life", "gamma_trans", "log_likelihood",
        "unpenalised_log_likelihood", "n_customers", "converged", "aic", "bic",
    )

    NAMES_LIFE: ClassVar[list[str]] = ["Gender", "Channel"]
    CONSTRAINED: ClassVar[list[str]] = ["Gender"]
    LAMBDAS = (0.1, 0.2)

    @pytest.fixture(scope="class")
    def results(self):
        """One covariate result per family, carrying the same covariate part."""
        from clvtools._staticcov import StaticCovResult
        from clvtools.pnbd.staticcov import PnbdStaticCovParams

        shared = {
            "gamma_life": np.array([0.1, 0.2]),
            "gamma_trans": np.array([0.3, 0.4]),
            "names_cov_life": self.NAMES_LIFE,
            "names_cov_trans": self.NAMES_LIFE,
            "names_cov_constr": self.CONSTRAINED,
            "reg_lambdas": self.LAMBDAS,
            "log_likelihood": -1.0,
            "unpenalised_log_likelihood": -2.0,
            "converged": True,
            "n_customers": 600,
        }
        covariates = StaticCovResult(model=np.array([1.0, 2.0, 3.0, 4.0]), **shared)
        return {
            "pnbd": PnbdStaticCovParams(
                r=1.0, alpha=2.0, s=3.0, beta=4.0, **shared
            ),
            "bgnbd": bgnbd.BgnbdStaticCovParams(
                r=1.0, alpha=2.0, a=3.0, b=4.0, covariates=covariates
            ),
            "ggomnbd": ggomnbd.GgomnbdStaticCovParams(
                r=1.0, alpha=2.0, b=3.0, s=4.0, beta=5.0, covariates=covariates
            ),
        }

    @pytest.mark.parametrize("attribute", SHARED)
    def test_every_family_answers_it(self, results, attribute):
        missing = [n for n, f in results.items() if not hasattr(f, attribute)]
        assert not missing, f"{attribute} is missing from {missing}"

    @pytest.mark.parametrize("family", ["pnbd", "bgnbd", "ggomnbd"])
    def test_the_two_that_were_missing_report_what_was_asked_for(
        self, results, family
    ):
        """Not merely present -- carrying the fit's own values.

        A property forwarding to a constant would satisfy the check above while
        telling the caller nothing, so both are given distinguishable values:
        one of two covariates constrained, and two *different* penalties, which
        also catches a forward that returns them the wrong way round.
        """
        fit = results[family]
        assert fit.names_cov_constr == self.CONSTRAINED
        assert fit.reg_lambdas == self.LAMBDAS

    @pytest.mark.parametrize("family", ["pnbd", "bgnbd", "ggomnbd"])
    def test_an_unregularized_result_reports_no_penalty(self, results, family):
        """`None` rather than `(0, 0)`, which would be a real fit at zero."""
        import dataclasses

        fit = results[family]
        if family == "pnbd":
            bare = dataclasses.replace(fit, reg_lambdas=None, names_cov_constr=[])
        else:
            bare = dataclasses.replace(
                fit,
                covariates=dataclasses.replace(
                    fit.covariates, reg_lambdas=None, names_cov_constr=[]
                ),
            )
        assert bare.reg_lambdas is None
        assert bare.names_cov_constr == []


class TestAScalarRegLambdaSaysWhatItWanted:
    """Backlog item 27, finding 20: `reg_lambdas=1.0` gave a bare TypeError.

    It reached ``tuple(float(v) for v in reg_lambdas)`` and surfaced as
    "'float' object is not iterable", which names Python's difficulty rather
    than the caller's. Eq. (13) has two weights, one per process.
    """

    @pytest.mark.parametrize("value", [1.0, 0, 10])
    def test_it_names_the_pair_it_wanted(self, value):
        from clvtools._staticcov import _validated_reg_lambdas

        with pytest.raises(ValueError, match="two values"):
            _validated_reg_lambdas(value)

    def test_and_suggests_the_spelling_that_works(self):
        from clvtools._staticcov import _validated_reg_lambdas

        with pytest.raises(ValueError, match=r"\(10\.0, 10\.0\)"):
            _validated_reg_lambdas(10.0)

    @pytest.mark.parametrize("value", [(0.1, 0.2), [0.1, 0.2], None])
    def test_the_shapes_that_were_always_fine_still_are(self, value):
        from clvtools._staticcov import _validated_reg_lambdas

        got = _validated_reg_lambdas(value)
        assert got == (None if value is None else (0.1, 0.2))


class TestOptimiserOverridesAreCheckedAgainstTheMethod:
    """Backlog item 31: `options_for` merged anything the caller passed.

    SciPy's answer to a key the solver does not read is a
    ``UserWarning: Unknown solver options`` and then dropping it, so a caller
    who asked for a bound got a fit that ran without one. R errors. Finding 20
    of ``docs/review-2026-09-02.md``, spec ``V-03``.

    The accepted keys are asked of SciPy rather than listed here -- the keyword
    parameters of ``_minimize_neldermead`` and ``_minimize_lbfgsb`` *are* the
    contract, and a copy would drift from it.
    """

    @staticmethod
    def _options(method, **overrides):
        from clvtools._optimize import options_for

        return options_for(method, 100, np.zeros(3), overrides=overrides or None)

    def test_a_key_the_method_cannot_read_is_refused(self):
        with pytest.raises(ValueError, match="does not accept 'nonsense'"):
            self._options("L-BFGS-B", nonsense=1)

    @pytest.mark.parametrize("method,wrong,right", [
        ("Nelder-Mead", "maxfun", "maxfev"),
        ("L-BFGS-B", "maxfev", "maxfun"),
    ])
    def test_the_near_miss_pair_is_named(self, method, wrong, right):
        """`maxfun` and `maxfev` cap the same thing under two spellings.

        Which is exactly why passing one where the other belongs looks right,
        and why the message says which one this method wants.
        """
        with pytest.raises(ValueError, match=f"did you mean '{right}'"):
            self._options(method, **{wrong: 5})

    @pytest.mark.parametrize("method,key", [
        ("L-BFGS-B", "maxfun"), ("Nelder-Mead", "maxfev"),
        ("L-BFGS-B", "ftol"), ("Nelder-Mead", "fatol"),
    ])
    def test_and_the_right_spelling_still_gets_through(self, method, key):
        assert self._options(method, **{key: 5})[key] == 5

    def test_an_unrecognised_method_validates_nothing(self):
        """Rather than refusing a method SciPy might well accept."""
        assert self._options("Powell", anything=1)["anything"] == 1

    def test_no_overrides_is_not_an_error(self):
        assert "maxiter" in self._options("L-BFGS-B")


class TestTheCallersCapReachesThePolish:
    """Backlog item 31: the polish ran on its own budget, not the caller's.

    ``_fit_from_candidates`` polishes an L-BFGS-B result with Nelder-Mead under
    a hard-coded ``maxiter=20_000, maxfev=20_000``, and used to keep those
    whatever the caller asked for. Measured on twelve customers,
    ``fit_pnbd_staticcov(options={"maxiter": 3})`` took **10.527 s** with the
    polish and **0.005 s** without it -- a factor of 2,100 between what was
    requested and what ran.
    """

    def test_the_overrides_reach_the_polish(self):
        from clvtools._staticcov import _polish_overrides

        assert _polish_overrides({"maxiter": 3})["maxiter"] == 3

    def test_but_only_the_keys_nelder_mead_reads(self):
        """The search above is usually L-BFGS-B, whose `maxfun` this stage
        would reject -- so it is filtered rather than forwarded or translated.
        """
        from clvtools._staticcov import _polish_overrides

        got = _polish_overrides({"maxiter": 3, "maxfun": 9, "ftol": 1e-9})
        assert got == {"maxiter": 3}

    def test_and_no_overrides_leaves_the_polish_exactly_as_it_was(self):
        """The default path has to be unchanged, not merely equivalent."""
        from clvtools._staticcov import _polish_overrides

        assert _polish_overrides(None) == {}
        assert _polish_overrides({}) == {}
