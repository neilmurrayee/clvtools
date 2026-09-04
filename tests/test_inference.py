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

    def test_confint_selects_rows_by_name(self, fit):
        """Spec `I-03`, `absent`: the argument did not exist until round 6.

        R's ``confint`` takes ``parm`` as a character or an integer vector.
        Four claims, of which this port had none, and the rows it returns must
        be the same rows the full table has -- selection, not recomputation.
        """
        picked = fit.confint(parm=["alpha", "beta"])
        assert list(picked.index) == ["alpha", "beta"]
        full = fit.confint()
        np.testing.assert_array_equal(picked.to_numpy(), full.loc[picked.index])

    def test_confint_takes_a_bare_name_as_well_as_a_list(self, fit):
        assert list(fit.confint(parm="s").index) == ["s"]

    def test_confint_selects_rows_by_position(self, fit):
        """Positions are 0-based here; R's are 1-based, and this says so."""
        assert list(fit.confint(parm=[0, 3]).index) == ["r", "beta"]
        assert list(fit.confint(parm=-1).index) == ["beta"]

    def test_confint_returns_nan_for_a_name_the_fit_does_not_have(self, fit):
        """R's own behaviour, and the reason this is not simply an error.

        A row of ``NaN`` keeps the shape of the request, so a caller assembling
        a table across models gets one row per name asked for whether or not
        each model has it.
        """
        got = fit.confint(parm=["alpha", "nonesuch"])
        assert list(got.index) == ["alpha", "nonesuch"]
        assert got.loc["nonesuch"].isna().all()
        assert got.loc["alpha"].notna().all()

    def test_confint_refuses_a_position_that_is_not_there(self, fit):
        with pytest.raises(IndexError, match="outside the 4 parameters"):
            fit.confint(parm=4)

    def test_confint_refuses_a_parm_that_is_neither(self, fit):
        with pytest.raises(TypeError, match="names or positions"):
            fit.confint(parm=[1.5])

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

    @pytest.mark.oracle
    def test_the_two_log_likelihoods_behind_it_match_the_oracle(self, fits):
        """Backlog item 25: ``pnbd_staticcov_lrtest.json`` had no reader.

        ``lrtest_pnbd_staticcov.json``, read above, pins the *derived*
        quantities -- degrees of freedom, the statistic, the p-value. This
        second fixture carries the two log-likelihoods they are derived from,
        which is the stronger check: a statistic comes out right from two wrong
        fits that happen to differ by the right amount.
        """
        want = fixture_json("pnbd_staticcov_lrtest")
        tied, free = fits
        assert tied.log_likelihood == pytest.approx(
            want["logLik.constrained"], rel=1e-6
        )
        assert free.log_likelihood == pytest.approx(
            want["logLik.unconstrained"], rel=1e-6
        )
        assert tied.n_parameters == want["df.constrained"]
        assert free.n_parameters == want["df.unconstrained"]

    def test_the_constrained_fit_is_the_worse_one(self, fits):
        """Which is what makes the statistic positive at all."""
        tied, free = fits
        assert tied.log_likelihood < free.log_likelihood

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


class TestTheThreeViewsOfAFitAgreeOnItsNames:
    """Spec I-01 and I-02, `weak`: "coef<->vcov for plain pnbd only".

    Five claims in I-01 and four in I-02, and between them they say one thing:
    ``coef()``, ``vcov()`` and ``coef(summary())`` name the same parameters **in
    the same order**, with no ``NaN``. The audit found that asserted for the
    plain Pareto/NBD alone -- "no covariate fit has its ``vcov()`` index
    checked; under constraints only ``names`` is compared".

    Order is the whole claim. A covariate fit's vector runs model parameters
    then attrition then transaction coefficients, and a constrained one reports
    a tied covariate **once**, as ``constr.<name>``; a `vcov` indexed in a
    different order would give every standard error to the wrong coefficient
    while looking perfectly well formed. Backlog item 34, round 5.
    """

    @pytest.fixture(scope="class")
    def fits(self, static_data):
        from clvtools import bgnbd, ggomnbd, pnbd

        return {
            "pnbd-cov": pnbd.fit_pnbd_staticcov(static_data),
            "bgnbd-cov": bgnbd.fit_bgnbd_staticcov(static_data),
            "ggomnbd-cov": ggomnbd.fit_ggomnbd_staticcov(static_data),
            "pnbd-constrained": pnbd.fit_pnbd_staticcov(
                static_data, names_cov_constr=["Gender"]
            ),
        }

    @pytest.mark.slow
    @pytest.mark.parametrize("which", [
        "pnbd-cov", "bgnbd-cov", "ggomnbd-cov", "pnbd-constrained",
    ])
    def test_coef_vcov_and_summary_name_the_same_things_in_order(
        self, fits, which
    ):
        fit = fits[which]
        assert list(fit.coefficients) == list(fit.vcov().index)
        assert list(fit.vcov().index) == list(fit.vcov().columns)
        assert list(fit.summary().index) == list(fit.coefficients)

    @pytest.mark.slow
    @pytest.mark.parametrize("which", ["pnbd-cov", "pnbd-constrained"])
    def test_and_carry_no_nan(self, fits, which):
        fit = fits[which]
        assert not np.isnan(fit.vcov().to_numpy()).any()
        assert not np.isnan(list(fit.coefficients.values())).any()

    @pytest.mark.slow
    def test_a_constrained_fit_reports_the_tied_covariate_once(self, fits):
        """Which is what makes the ordering worth asserting separately."""
        constrained = fits["pnbd-constrained"]
        free = fits["pnbd-cov"]
        assert "constr.Gender" in constrained.names
        assert len(constrained.names) == len(free.names) - 1
        assert list(constrained.vcov().index) == constrained.names


class TestSummaryHasTheDocumentedStructure:
    """Spec I-04, `weak`: "no structural check against R, printing never
    exercised".

    ``?summary.clv.fitted`` documents a coefficient table of Estimate, Std.
    Error, z-val and Pr(>|z|). The columns were pinned; that the frame *prints*
    was not, and a table that raises on `str()` is no use in a session.
    """

    @pytest.fixture(scope="class")
    def fitted(self, cbs_estimation):
        from clvtools.pnbd import fit_pnbd

        return fit_pnbd(
            cbs_estimation["x"], cbs_estimation["t.x"], cbs_estimation["T.cal"]
        )

    def test_the_coefficient_table_has_rs_four_columns_in_order(self, fitted):
        assert list(fitted.summary().columns) == [
            "Estimate", "Std. Error", "z-val", "Pr(>|z|)"
        ]

    def test_one_row_per_estimated_parameter(self, fitted):
        assert list(fitted.summary().index) == fitted.names
        assert len(fitted.summary()) == fitted.n_parameters

    def test_and_it_prints(self, fitted):
        """Exercised rather than assumed: `str()` on a frame can raise."""
        printed = str(fitted.summary())
        assert "Estimate" in printed
        for name in fitted.names:
            assert name in printed


class TestNobsAnswersOnAFitAsWellAsOnTheData:
    """Spec I-08, `weak`: "`ClvData.nobs()` pinned; fitted objects have no
    `nobs()`".

    Correct, and an inconsistency rather than a decision: the count was
    reachable as ``fit.n_customers`` while the data spelled the same question
    ``data.nobs()``. Both now answer. Backlog item 34, round 5.
    """

    def test_a_fit_and_its_data_agree(self, cbs_estimation, apparel_trans):
        from clvtools import ClvData
        from clvtools.pnbd import fit_pnbd

        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        fit = fit_pnbd(
            cbs_estimation["x"], cbs_estimation["t.x"], cbs_estimation["T.cal"],
            hessian=False,
        )
        assert fit.nobs() == data.nobs() == 600

    def test_it_is_the_count_bic_is_computed_against(self, cbs_estimation):
        """So a weighted fit cannot report one number and score against another.

        Backlog item 27 found the time-varying fit doing exactly that.
        """
        from clvtools.pnbd import fit_pnbd

        fit = fit_pnbd(
            cbs_estimation["x"], cbs_estimation["t.x"], cbs_estimation["T.cal"],
            hessian=False,
        )
        expected = fit.n_parameters * np.log(fit.nobs()) - 2 * fit.log_likelihood
        assert fit.bic == pytest.approx(expected, rel=1e-12)


class TestTheRatioTestIsFamilyAgnostic:
    """Spec I-10, `weak`: "one model only; 'runs for all models' untested".

    R's ``lrtest()`` is a generic dispatching per model class, so "runs for all
    models" is a real question there. Here it is a plain function over two
    :class:`~clvtools.inference.Fitted` objects, reading exactly two things from
    each -- ``log_likelihood`` and ``n_parameters`` -- so it is family-agnostic
    **by construction**, which is a stronger statement than any number of fits
    could make and the same class of claim as ``tests/test_invariants.py``.

    Asserted over constructed results rather than fits: six covariate fits to
    show that a function which never looks at the family works for every family
    is minutes of optimiser time for no information. Backlog item 34, round 5.
    """

    @staticmethod
    def _result(family: str, n_covariates: int, log_likelihood: float):
        """One fitted object per family, carrying only what lrtest reads.

        Parameterised by how many *covariates* it names rather than by a target
        total: the families have different model-parameter counts -- four for
        the Pareto/NBD and BG/NBD, five for the GGom/NBD -- so a fixed total
        would mean a different covariate count per family, which is what the
        first draft of this got wrong.
        """
        import numpy as np

        from clvtools import bgnbd, ggomnbd
        from clvtools._staticcov import StaticCovResult
        from clvtools.pnbd.staticcov import PnbdStaticCovParams

        names = ["Gender", "Channel"][:n_covariates]
        shared = {
            "gamma_life": np.zeros(len(names)),
            "gamma_trans": np.zeros(len(names)),
            "names_cov_life": names, "names_cov_trans": names,
            "names_cov_constr": [], "reg_lambdas": None,
            "log_likelihood": log_likelihood,
            "unpenalised_log_likelihood": log_likelihood,
            "converged": True, "n_customers": 600,
        }
        if family == "pnbd":
            return PnbdStaticCovParams(r=1.0, alpha=2.0, s=3.0, beta=4.0, **shared)
        covariates = StaticCovResult(model=np.ones(4), **shared)
        if family == "bgnbd":
            return bgnbd.BgnbdStaticCovParams(
                r=1.0, alpha=2.0, a=3.0, b=4.0, covariates=covariates
            )
        return ggomnbd.GgomnbdStaticCovParams(
            r=1.0, alpha=2.0, b=3.0, s=4.0, beta=5.0, covariates=covariates
        )

    @pytest.mark.parametrize("family", ["pnbd", "bgnbd", "ggomnbd"])
    def test_it_runs_for_every_family(self, family):
        restricted = self._result(family, 1, -5826.0)
        unrestricted = self._result(family, 2, -5821.0)
        got = likelihood_ratio_test(restricted, unrestricted)
        assert got.statistic == pytest.approx(2 * (-5821.0 - -5826.0))
        # The degrees of freedom are the parameter difference, whatever the
        # family's own model-parameter count is. One extra *covariate* is two
        # extra parameters, since it enters both the attrition and transaction
        # processes -- which is why this asserts the difference rather than a
        # hard-coded 1, as the first draft did.
        assert got.df == unrestricted.n_parameters - restricted.n_parameters
        assert got.df == 2
        assert 0.0 < got.p_value < 1.0

    def test_and_across_families_too_since_it_never_looks_at_one(self):
        """The construction argument, made explicit.

        Nothing stops a caller comparing a BG/NBD against a Pareto/NBD -- the
        function has no way to tell. That is worth *knowing* rather than
        guarding: the models are not nested, so the answer is meaningless, and
        the guard R gets for free from dispatch is absent here.
        """
        got = likelihood_ratio_test(
            self._result("bgnbd", 1, -5826.0), self._result("pnbd", 2, -5821.0)
        )
        assert np.isfinite(got.statistic)
