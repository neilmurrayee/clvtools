r"""Table 2's entry points, ``latent_attrition()`` and ``spending()``.

The dispatch, the formula parsing and the guards; the estimates themselves are
tested where each family's fit is. What matters here is that the entry point
reaches the same fit the paper's chunk would, and refuses the combinations
Table 4 rules out.
"""

from __future__ import annotations

import numpy as np
import pytest
from paper_values import GG_MLE, PNBD_MLE, PNBD_STATIC_MLE

from clvtools import (
    ClvData,
    ClvDataDynCov,
    bgnbd,
    gg,
    ggomnbd,
    latent_attrition,
    load_apparel_dyn_cov,
    pnbd,
    spending,
)
from clvtools.estimate import parse_formula

NAMES_DYN = ["High.Season", "Gender", "Channel"]


@pytest.fixture(scope="module")
def data(apparel_trans) -> ClvData:
    return ClvData(apparel_trans, time_unit="week", estimation_split=104)


class TestFormula:
    """S6.4's ``~ life | trans``."""

    def test_splits_the_two_processes(self):
        assert parse_formula("~ Gender + Channel | Gender") == (
            ["Gender", "Channel"], ["Gender"]
        )

    def test_a_dot_means_everything(self):
        assert parse_formula("~ . | .") == (None, None)

    def test_a_dot_on_one_side_only(self):
        assert parse_formula("~ . | Gender") == (None, ["Gender"])

    def test_the_tilde_is_optional(self):
        assert parse_formula("Gender | Gender") == (["Gender"], ["Gender"])

    @pytest.mark.parametrize("formula", ["~ Gender", "~ Gender | A | B"])
    def test_both_processes_are_required(self, formula):
        with pytest.raises(ValueError, match="both processes"):
            parse_formula(formula)

    def test_an_empty_side_is_refused(self):
        with pytest.raises(ValueError, match="no covariates named"):
            parse_formula("~ | Gender")


@pytest.mark.slow
class TestTransformedTerms:
    """``I(...)`` in a formula -- ``?latentAttrition``.

    ``latentAttrition(formula = ~ Channel + Gender | I(log(Channel + 2)), ...)``
    passes an *expression* rather than a column name, and R evaluates it
    against the covariate data. The ``+`` inside it is arithmetic, so the term
    has to survive the split that separates ``Channel + Gender``.
    """

    def test_the_inner_plus_does_not_split_the_term(self):
        assert parse_formula("~ Gender | I(log(Channel + 2))") == (
            ["Gender"], ["I(log(Channel + 2))"]
        )

    def test_a_transformed_term_beside_a_plain_one(self):
        assert parse_formula("~ Channel + Gender | I(log(Channel + 2)) + Gender") == (
            ["Channel", "Gender"], ["I(log(Channel + 2))", "Gender"]
        )

    def test_nested_parentheses_survive(self):
        assert parse_formula("~ . | I(log(exp(Channel) + 2))") == (
            None, ["I(log(exp(Channel) + 2))"]
        )

    def test_the_expression_is_evaluated(self, static_data):
        """``Channel`` is 0 or 1, so ``log(Channel + 2)`` is log 2 or log 3."""
        derived = static_data.with_covariates(None, ["I(log(Channel + 2))"])
        assert derived.names_cov_trans == ["I(log(Channel + 2))"]
        column = derived.design_trans().ravel()
        assert set(np.round(column, 9)) == {
            round(float(np.log(2)), 9), round(float(np.log(3)), 9)
        }

    def test_it_is_the_transform_of_the_original_column(self, static_data):
        original = static_data.with_covariates(None, ["Channel"]).design_trans()
        derived = static_data.with_covariates(
            None, ["I(log(Channel + 2))"]
        ).design_trans()
        np.testing.assert_allclose(derived, np.log(original + 2))

    def test_the_original_column_is_left_alone(self, static_data):
        """Deriving must not disturb the frame it derives from."""
        before = static_data.design_trans(["Channel"]).copy()
        static_data.with_covariates(None, ["I(log(Channel + 2))"])
        np.testing.assert_array_equal(static_data.design_trans(["Channel"]), before)
        assert "I(log(Channel + 2))" not in static_data.names_cov_trans

    def test_the_two_processes_transform_independently(self, static_data):
        derived = static_data.with_covariates(
            ["Gender"], ["I(Channel * 2)"]
        )
        assert derived.names_cov_life == ["Gender"]
        assert derived.names_cov_trans == ["I(Channel * 2)"]
        np.testing.assert_allclose(
            derived.design_trans(),
            2 * static_data.design_trans(["Channel"]),
        )

    def test_a_transform_reaches_the_fit(self, static_data):
        """The whole point: the derived column is what gets estimated."""
        fit = latent_attrition(
            formula="~ Gender | I(log(Channel + 2))",
            family=pnbd, data=static_data, hessian=False, maxiter=60,
        )
        assert fit.names_cov_trans == ["I(log(Channel + 2))"]
        assert "trans.I(log(Channel + 2))" in fit.names

    def test_an_unevaluable_expression_is_rejected(self, static_data):
        with pytest.raises(ValueError, match="cannot evaluate"):
            static_data.with_covariates(None, ["I(log(NoSuchColumn))"])

    def test_the_expression_cannot_reach_the_interpreter(self, static_data):
        """It goes to `DataFrame.eval`, not to `eval`."""
        with pytest.raises(ValueError, match="cannot evaluate"):
            static_data.with_covariates(None, ["I(__import__('os').getcwd())"])


class TestDispatch:
    """The data object's type picks the estimator, as it does in S6.4."""

    @pytest.mark.paper
    def test_plain_data_gets_the_plain_fit(self, data):
        fit = latent_attrition(family=pnbd, data=data, hessian=False)
        for name, expected in PNBD_MLE.items():
            assert fit.coefficients[name] == pytest.approx(expected, abs=5e-3)

    def test_the_family_may_be_named(self, data):
        by_module = latent_attrition(family=pnbd, data=data, hessian=False)
        by_name = latent_attrition(family="pnbd", data=data, hessian=False)
        np.testing.assert_allclose(list(by_name), list(by_module))

    # 18.4 s for the GGom/NBD arm, which is a full-dataset MLE and belongs
    # under `slow` with the other 150-odd. It was the slowest unmarked test in
    # the suite. Finding 17, backlog item 25.
    @pytest.mark.slow
    @pytest.mark.parametrize("family", [bgnbd, ggomnbd])
    def test_the_other_families_dispatch_too(self, data, family):
        fit = latent_attrition(family=family, data=data, hessian=False)
        assert type(fit).__module__.endswith(family.__name__.rsplit(".", 1)[-1])

    @pytest.mark.paper
    def test_covariate_data_gets_the_covariate_fit(self, static_data):
        fit = latent_attrition(
            formula="~ Gender + Channel | Gender + Channel",
            family=pnbd, data=static_data, hessian=False,
        )
        # Loose on alpha and beta: S6.4.1's likelihood is flat along the same
        # ridge the plain model's is, so this optimiser stops a little away
        # from CLVTools'. tests/test_pnbd_staticcov.py pins the fit itself.
        for name, expected in PNBD_STATIC_MLE.items():
            assert fit.coefficients[name] == pytest.approx(expected, rel=5e-3)

    def test_a_formula_may_select_a_subset(self, static_data):
        fit = latent_attrition(
            formula="~ Gender | Channel", family=pnbd, data=static_data,
            hessian=False,
        )
        assert list(fit.coefficients)[4:] == ["life.Gender", "trans.Channel"]

    def test_a_dot_formula_takes_everything(self, static_data):
        every = latent_attrition(
            formula="~ . | .", family=pnbd, data=static_data, hessian=False
        )
        implicit = latent_attrition(family=pnbd, data=static_data, hessian=False)
        assert every.names == implicit.names

    def test_constraints_pass_through(self, static_data):
        fit = latent_attrition(
            formula="~ . | .", names_cov_constr=["Gender"], family=pnbd,
            data=static_data, hessian=False,
        )
        assert "constr.Gender" in fit.names

    @pytest.mark.paper
    def test_spending_reaches_the_published_estimates(self, data):
        fit = spending(family=gg, data=data, hessian=False)
        for name, expected in GG_MLE.items():
            assert fit.coefficients[name] == pytest.approx(expected, abs=5e-3)

    def test_spending_can_keep_the_first_transaction(self, data):
        """S6.3.4's ``remove.first.transaction = FALSE``."""
        without = spending(family=gg, data=data, hessian=False)
        with_first = spending(
            family=gg, data=data, remove_first_transaction=False, hessian=False
        )
        assert not np.allclose(list(without), list(with_first))

    def test_correlation_is_available_on_the_plain_pareto_nbd(self, data):
        fit = latent_attrition(family=pnbd, data=data, use_cor=True)
        assert hasattr(fit, "correlation")


class TestGuards:
    """What Table 4 marks as unavailable, refused rather than silently ignored."""

    def test_an_unknown_family_is_named(self, data):
        with pytest.raises(ValueError, match="unknown family"):
            latent_attrition(family="bgbb", data=data)

    def test_a_formula_needs_covariate_data(self, data):
        with pytest.raises(ValueError, match="no covariates"):
            latent_attrition(formula="~ Gender | Gender", family=pnbd, data=data)

    def test_time_varying_covariates_are_pareto_nbd_only(
        self, apparel_trans, data
    ):
        dynamic = ClvDataDynCov(
            data, load_apparel_dyn_cov(),
            names_cov_life=NAMES_DYN, names_cov_trans=NAMES_DYN,
        )
        with pytest.raises(ValueError, match="Pareto/NBD alone"):
            latent_attrition(family=bgnbd, data=dynamic)

    @pytest.mark.parametrize("family", [bgnbd, ggomnbd])
    def test_correlation_is_pareto_nbd_only(self, data, family):
        with pytest.raises(ValueError, match="Pareto/NBD alone"):
            latent_attrition(family=family, data=data, use_cor=True)

    def test_correlation_is_not_offered_with_covariates(self, static_data):
        with pytest.raises(ValueError, match="plain Pareto/NBD"):
            latent_attrition(family=pnbd, data=static_data, use_cor=True)

    def test_the_gamma_gamma_is_the_only_spending_model(self, data):
        with pytest.raises(ValueError, match="only spending model"):
            spending(family=pnbd, data=data)

    def test_a_formula_covariate_must_exist(self, static_data):
        with pytest.raises(ValueError, match="not in the data"):
            latent_attrition(
                formula="~ Region | Gender", family=pnbd, data=static_data
            )


class TestCovariateSelection:
    """``with_covariates``, which the formula goes through."""

    def test_static_selection_keeps_the_design_matrices(self, static_data):
        one = static_data.with_covariates(["Gender"], ["Channel"])
        assert one.names_cov_life == ["Gender"]
        assert one.names_cov_trans == ["Channel"]
        assert one.design_life().shape == (600, 1)
        # The original is untouched: a formula selects a view, not a mutation.
        assert static_data.names_cov_life == ["Gender", "Channel"]

    def test_static_selection_of_nothing_changes_nothing(self, static_data):
        same = static_data.with_covariates()
        assert same.names_cov_life == static_data.names_cov_life
        assert same.names_cov_trans == static_data.names_cov_trans

    def test_static_selection_rejects_a_stranger(self, static_data):
        with pytest.raises(ValueError, match="not in the data"):
            static_data.with_covariates(["Region"], None)

    def test_dynamic_selection_rebuilds_the_walks(self, apparel_trans, data):
        dynamic = ClvDataDynCov(
            data, load_apparel_dyn_cov(),
            names_cov_life=NAMES_DYN, names_cov_trans=NAMES_DYN,
        )
        one = dynamic.with_covariates(["High.Season"], ["High.Season"])
        assert one.names_cov_life == ["High.Season"]
        assert one.walks().n_cov_life == 1
        assert dynamic.walks().n_cov_life == 3

    def test_dynamic_selection_of_nothing_changes_nothing(self, data):
        dynamic = ClvDataDynCov(
            data, load_apparel_dyn_cov(),
            names_cov_life=NAMES_DYN, names_cov_trans=NAMES_DYN,
        )
        assert dynamic.with_covariates().names_cov_trans == NAMES_DYN

    def test_dynamic_selection_rejects_a_stranger(self, data):
        dynamic = ClvDataDynCov(
            data, load_apparel_dyn_cov(),
            names_cov_life=NAMES_DYN, names_cov_trans=NAMES_DYN,
        )
        with pytest.raises(ValueError, match="not in the data"):
            dynamic.with_covariates(None, ["Region"])


class TestTimeVaryingDispatch:
    """The time-varying branch, on five customers so it costs a moment.

    The fit itself is tested in ``tests/test_pnbd_dyncov.py`` under
    ``dyncov_fit``; what matters here is that the entry point reaches it and
    passes the formula through.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def small(apparel_trans):
        ids = sorted(apparel_trans["Id"].unique())[:5]
        covariates = load_apparel_dyn_cov()
        return ClvDataDynCov(
            ClvData(
                apparel_trans[apparel_trans["Id"].isin(ids)],
                time_unit="week", estimation_split=104,
            ),
            covariates[covariates["Id"].isin(ids)],
            names_cov_life=NAMES_DYN, names_cov_trans=NAMES_DYN,
        )

    def test_it_reaches_the_time_varying_fit(self, small):
        fit = latent_attrition(
            formula="~ High.Season | High.Season", family=pnbd, data=small,
            maxiter=1,
        )
        assert fit.names == [
            "r", "alpha", "s", "beta", "life.High.Season", "trans.High.Season"
        ]

    def test_without_a_formula_it_takes_every_covariate(self, small):
        fit = latent_attrition(family=pnbd, data=small, maxiter=1)
        assert fit.names_cov_life == NAMES_DYN

    def test_correlation_is_refused(self, small):
        with pytest.raises(ValueError, match="plain Pareto/NBD"):
            latent_attrition(family=pnbd, data=small, use_cor=True)


class TestAFormulaOnDataThatHasNoCovariates:
    """Backlog item 27, finding 20: `~ . | .` on plain data was accepted.

    The guard tested `names_life or names_trans`, which are the *parsed* names.
    `~ . | .` parses to `(None, None)` -- the "use every covariate" marker --
    so it read as "no names given" and fell through to the plain fit, which
    silently ignored the formula. Any formula is a mistake here, whether it
    names covariates or asks for all of them.
    """

    @pytest.fixture(scope="class")
    def plain(self):
        from clvtools import ClvData, load_apparel_trans

        return ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)

    @pytest.mark.parametrize("formula", ["~ . | .", "~ Gender | Channel", "~ . | Gender"])
    def test_it_is_refused(self, plain, formula):
        import clvtools

        with pytest.raises(ValueError, match="no covariates"):
            clvtools.latent_attrition(
                family=clvtools.pnbd, data=plain, formula=formula, hessian=False
            )

    def test_but_no_formula_still_fits(self, plain):
        """The guard must not fire on the ordinary call."""
        import clvtools

        fit = clvtools.latent_attrition(family=clvtools.pnbd, data=plain, hessian=False)
        assert fit.converged


class TestAnEmptyCovariateTerm:
    """Backlog item 27: `~ Gender + | Gender` fitted on Gender alone.

    The parser dropped empty terms, so a `+` with nothing after it -- what a
    half-finished edit leaves behind -- produced a smaller model than the text
    asks for and said nothing. The name that *is* there makes the old guard
    ("no covariates named") unreachable, which is why this needed its own.
    """

    @pytest.mark.parametrize("formula", [
        "~ Gender + | Gender", "~ Gender | Gender +", "~ Gender +  + Channel | Gender",
    ])
    def test_it_is_refused(self, formula):
        with pytest.raises(ValueError, match="empty covariate term"):
            parse_formula(formula)

    def test_a_side_naming_nothing_still_says_so(self):
        """The pre-existing message, which the new check must not shadow."""
        with pytest.raises(ValueError, match="no covariates named"):
            parse_formula("~  | Gender")

    def test_and_the_ordinary_formulas_are_unmoved(self):
        assert parse_formula("~ Gender + Channel | Gender") == (
            ["Gender", "Channel"], ["Gender"]
        )
        assert parse_formula("~ . | .") == (None, None)
