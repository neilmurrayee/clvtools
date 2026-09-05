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
    # the suite. Finding 17,
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


class TestTheDotExpandsBesideOtherTerms:
    """Spec FI-04's third claim: `~ . | . + I(Gender + 1)`.

    `.` means "every covariate the data carries". Alone it worked; beside
    another term it was passed through as a **literal column name**, so the
    formula went looking for a covariate called ``"."``. The audit put it
    plainly -- "`~ . | . + I(...)` unsupported -- `'.'` is looked up as a
    literal column".

    Expanding it needs the data, which :func:`~clvtools.estimate.parse_formula`
    does not see, so parsing keeps the ``"."`` term and ``_narrowed`` resolves
    it.
    """

    def test_the_dot_survives_parsing_as_a_term(self):
        assert parse_formula("~ . | . + I(Gender + 1)") == (
            None, [".", "I(Gender + 1)"]
        )

    @pytest.mark.slow
    def test_and_expands_against_the_data_when_the_fit_runs(self, static_data):
        fit = latent_attrition(
            family=pnbd, data=static_data,
            formula="~ . | . + I(Gender + 1)", hessian=False,
        )
        assert fit.names_cov_life == ["Gender", "Channel"]
        assert fit.names_cov_trans == ["Gender", "Channel", "I(Gender + 1)"]

    def test_a_covariate_named_twice_is_selected_once(self):
        """`~ . + Gender | .` asks for nothing more than `~ . | .`."""
        from clvtools.estimate import _expanded

        assert _expanded([".", "Gender"], ["Gender", "Channel"]) == [
            "Gender", "Channel"
        ]

    def test_the_datas_own_order_is_kept(self):
        """So a coefficient vector does not reorder with the formula's spelling."""
        from clvtools.estimate import _expanded

        assert _expanded([".", "Extra"], ["Gender", "Channel"]) == [
            "Gender", "Channel", "Extra"
        ]

    def test_a_bare_dot_still_means_everything(self):
        from clvtools.estimate import _expanded

        assert _expanded(None, ["Gender"]) is None


class TestTheEntryPointsRefuseWhatIsNotClvData:
    """Spec FI-13 and FI-14, first claim of each: non-`clv.data` data.

    Both entry points did fail -- with ``AttributeError: 'DataFrame' object has
    no attribute 'customer_summary'``, which names an internal method rather
    than what the caller got wrong, and reads like a bug in the library rather
    than in the call.
    """

    @pytest.fixture(scope="class")
    def frame(self):
        from clvtools import load_apparel_trans

        return load_apparel_trans()

    def test_latent_attrition_says_what_it_wanted(self, frame):
        with pytest.raises(TypeError, match="needs a ClvData, not DataFrame"):
            latent_attrition(family=pnbd, data=frame)

    def test_and_so_does_spending(self, frame):
        from clvtools import gg, spending

        with pytest.raises(TypeError, match="needs a ClvData, not DataFrame"):
            spending(family=gg, data=frame)

    def test_the_message_names_the_constructor_to_use(self, frame):
        with pytest.raises(TypeError, match=r"ClvData\(transactions"):
            latent_attrition(family=pnbd, data=frame)

    @pytest.mark.parametrize("value", [None, 42, "apparel"])
    def test_anything_else_is_refused_too(self, value):
        with pytest.raises(TypeError, match="needs a ClvData"):
            latent_attrition(family=pnbd, data=value)


class TestAFormulaHasNoLeftHandSide:
    """Spec FI-13: "a LHS parses silently and fails with the wrong message".

    ``y ~ Gender | Channel`` parsed to ``(['y ~ Gender'], ['Channel'])`` -- the
    left-hand side swallowed into the first covariate name -- and then failed
    complaining about a covariate called ``"y ~ Gender"``. It *did* fail, which
    is what the spec asks; it failed for the wrong reason and said so.
    """

    @pytest.mark.parametrize("formula", [
        "y ~ Gender | Channel", "Spending ~ . | .", "x~a|b",
    ])
    def test_it_is_refused_by_name(self, formula):
        with pytest.raises(ValueError, match="no left-hand side"):
            parse_formula(formula)

    def test_and_the_ordinary_formulas_are_unmoved(self):
        assert parse_formula("~ Gender | Channel") == (["Gender"], ["Channel"])
        assert parse_formula("  ~ . | .  ") == (None, None)

    def test_the_tilde_is_still_optional(self):
        """The rule is "nothing before the tilde", not "starts with one".

        The first draft required a leading ``~`` and broke
        ``TestFormula::test_the_tilde_is_optional``, which documents that
        ``Gender | Gender`` is a valid formula. Kept here as well, next to the
        check that nearly removed it.
        """
        assert parse_formula("Gender | Gender") == (["Gender"], ["Gender"])
        assert parse_formula(". | .") == (None, None)

    def test_an_empty_term_is_still_caught_beside_an_exclusion(self):
        """The two checks share ``_split_terms`` and must not mask each other.

        ``_expand_exclusions`` filtered empty terms out on its way past, which
        silently disabled the "a '+' with nothing after it" check added in
        backlog item 27 -- ``TestAnEmptyCovariateTerm`` caught it. Both now fire
        on a formula that trips both.
        """
        with pytest.raises(ValueError, match="empty covariate term"):
            parse_formula("~ . - Gender + | .")


class TestRemoveFirstTransactionChangesEverythingButTheIds:
    """Spec FI-11, `weak`: only "not np.allclose(...)" was asserted.

    Four claims, of which one was pinned loosely and three not at all: **every**
    coefficient differs, **every** cbs ``x`` differs, the id set is unchanged,
    and it equals the transaction data's ids. S6.2.3's reason for the option is
    that the first transaction "has been found to be atypical", so a difference
    that showed up in one coefficient and not the others would mean the option
    was reaching only part of the fit.
    """

    @pytest.fixture(scope="class")
    def fits(self, apparel_trans):
        from clvtools import gg, spending

        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        return data, (
            spending(family=gg, data=data, remove_first_transaction=False,
                     hessian=False),
            spending(family=gg, data=data, remove_first_transaction=True,
                     hessian=False),
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["p", "q", "gamma"])
    def test_every_coefficient_differs(self, fits, name):
        _data, (with_first, without) = fits
        assert getattr(with_first, name) != getattr(without, name)

    def test_every_customers_transaction_count_differs(self, apparel_trans):
        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        with_first = data.spending_summary(remove_first_transaction=False)
        without = data.spending_summary(remove_first_transaction=True)
        assert (with_first["x"].to_numpy() != without["x"].to_numpy()).all()

    def test_but_the_id_set_is_unchanged_and_is_the_logs(self, apparel_trans):
        """Dropping a transaction must not drop a *customer*."""
        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        with_first = data.spending_summary(remove_first_transaction=False)
        without = data.spending_summary(remove_first_transaction=True)
        assert set(with_first["Id"]) == set(without["Id"])
        assert set(without["Id"]) == set(apparel_trans["Id"])


class TestNarrowingKeepsTheClassAndSharesTheFrames:
    """Spec FI-08 and FI-09.

    `FI-08` -- a formula applied to dynamic covariate data returns a dynamic
    object and one applied to static data returns a static object that is *not*
    dynamic -- holds, and was untested. It matters because the time-varying fit
    has nothing to walk if narrowing quietly downcasts.

    `FI-09` is a **divergence, recorded rather than fixed**. R copies, so its
    narrowed object shares no storage with the input. Here narrowing shares the
    transaction and covariate frames, and this pins that it does.

    The risk it would carry is bounded at the only place it could bite:
    ``ClvData.__init__`` *does* copy the caller's frame, so nothing a caller
    holds is reachable from a fitted object. What is shared is one of this
    package's objects with a narrowed descendant of itself, and mutating
    ``data.transactions`` in place -- an internal attribute -- is outside the
    contract either way. Copying 187,800 covariate rows on every formula call
    to prevent that is not a trade worth making, so the sharing is deliberate.
    """

    def test_static_data_narrows_to_a_static_object(self, static_data):
        from clvtools import ClvDataDynCov, ClvDataStaticCov

        narrowed = static_data.with_covariates(["Gender"], ["Gender"])
        assert isinstance(narrowed, ClvDataStaticCov)
        assert not isinstance(narrowed, ClvDataDynCov)

    def test_dynamic_data_stays_dynamic(self, apparel_trans):
        from clvtools import ClvDataDynCov, load_apparel_dyn_cov

        names = ["High.Season", "Gender", "Channel"]
        dynamic = ClvDataDynCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(), names_cov_life=names, names_cov_trans=names,
        )
        narrowed = dynamic.with_covariates(["High.Season"], ["High.Season"])
        assert isinstance(narrowed, ClvDataDynCov)
        assert narrowed.names_cov_life == ["High.Season"]

    def test_narrowing_shares_the_transaction_frame(self, static_data):
        """FI-09's divergence, pinned so it cannot change unnoticed."""
        narrowed = static_data.with_covariates(["Gender"], ["Gender"])
        assert narrowed.transactions is static_data.transactions

    def test_but_construction_copies_the_callers_frame(self, apparel_trans):
        """Which is the boundary that actually matters.

        Nothing a caller holds is reachable from a fitted object, so the
        sharing above is between two of this package's own objects.
        """
        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        assert data.transactions is not apparel_trans


class TestTheDotTakesExclusions:
    """Spec FI-15, which names "`.` with exclusions" among its twelve claims.

    `_split_terms` splits on ``+`` only, so ``~ . - Gender | .`` arrived as the
    single term ``". - Gender"`` and was looked up as a column of that name.
    """

    def test_an_exclusion_becomes_its_own_term(self):
        assert parse_formula("~ . - Gender | .") == ([".", "-Gender"], None)

    def test_and_resolves_against_the_data(self):
        from clvtools.estimate import _expanded

        assert _expanded([".", "-Gender"], ["Gender", "Channel"]) == ["Channel"]

    def test_a_transformation_containing_a_minus_is_left_alone(self):
        """`I(...)` is an expression, not a list of terms."""
        assert parse_formula("~ . | . + I(Channel - 1)") == (
            None, [".", "I(Channel - 1)"]
        )

    def test_excluding_a_covariate_the_data_does_not_carry_says_so(self):
        from clvtools.estimate import _expanded

        with pytest.raises(ValueError, match="cannot exclude 'Nonexistent'"):
            _expanded([".", "-Nonexistent"], ["Gender", "Channel"])

    def test_and_excluding_without_a_dot_is_refused(self):
        """Subtracting from an explicit list is the same as not listing it."""
        from clvtools.estimate import _expanded

        with pytest.raises(ValueError, match="does not use '\\.'"):
            _expanded(["Gender", "-Channel"], ["Gender", "Channel"])

    @pytest.mark.slow
    def test_the_fit_selects_what_is_left(self, static_data):
        fit = latent_attrition(
            family=pnbd, data=static_data, formula="~ . - Gender | .",
            hessian=False,
        )
        assert fit.names_cov_life == ["Channel"]
        assert fit.names_cov_trans == ["Gender", "Channel"]


class TestRsConstraintSyntaxIsRefusedNotMisread:
    """Spec FI-15: "`constraint()` syntax does not exist here".

    True, and it was being read as a *column name*, so
    ``~ constraint(Gender) | Gender`` failed looking for a covariate called
    ``"constraint(Gender)"``. R names tied covariates inside the formula; here
    they are the ``names_cov_constr`` argument, which is a deliberate
    difference and now says so at the point of use.
    """

    def test_it_names_the_argument_to_use_instead(self):
        with pytest.raises(ValueError, match="names_cov_constr"):
            parse_formula("~ constraint(Gender) | Gender")

    def test_on_either_side(self):
        with pytest.raises(ValueError, match="constraint"):
            parse_formula("~ Gender | constraint(Gender)")

    @pytest.mark.slow
    def test_and_the_argument_itself_still_works(self, static_data):
        """The capability is present; only the spelling differs."""
        from clvtools.pnbd import fit_pnbd_staticcov

        fit = fit_pnbd_staticcov(
            static_data, names_cov_constr=["Gender"], hessian=False,
        )
        assert fit.names_cov_constr == ["Gender"]
        assert "constr.Gender" in fit.names


class TestTransformationsAndInteractionsInAFormula:
    """Spec `FI-06` and `FI-07`, both `absent !`.

    `FI-06` asks for `~I(Gender+1)|log(Gender+2)` -- note that only the *first*
    term is wrapped. In R, ``I()`` protects **operators** from the formula
    grammar; a bare call is not formula syntax and needs no protection. This
    port supported ``I(...)`` and refused ``log(Gender + 2)`` with "covariates
    not in the data", which reads as a typo in a term that is not one.

    `FI-07` asks for interactions and exclusions: `~Gender*Channel|.-Gender`
    giving life ``(Gender, Channel, Gender.Channel)`` and transactions
    ``(Channel)``. The exclusion half already worked -- it is one of the
    README's findings -- and the interaction half did not exist. ``*`` is main
    effects *and* their product, ``:`` the product alone, and the product is
    named with a dot, which is how ``make.names`` renders R's
    ``Gender:Channel``.
    """

    @staticmethod
    def _select(data, formula):
        """The formula path ``latent_attrition`` takes, without the fit.

        The ``.`` expansion needs the data to know its own covariates, so it
        happens after :func:`parse_formula` and not inside it -- which is why a
        probe that called ``with_covariates(parse_formula(f))`` directly saw
        `FI-07`'s exclusion fail when the documented path handles it.
        """
        from clvtools.estimate import _expanded, parse_formula

        life, trans = parse_formula(formula)
        return data.with_covariates(
            _expanded(life, data.names_cov_life),
            _expanded(trans, data.names_cov_trans),
        )

    def test_a_bare_call_is_evaluated_like_a_wrapped_one(self, static_data):
        """`FI-06`: both spellings, and the arithmetic, not just the naming."""
        narrowed = self._select(
            static_data, "~ I(Gender + 1) | log(Gender + 2)"
        )
        assert narrowed.names_cov_life == ["I(Gender + 1)"]
        assert narrowed.names_cov_trans == ["log(Gender + 2)"]

        gender = static_data.design_life(["Gender"]).ravel()
        np.testing.assert_allclose(
            narrowed.design_life().ravel(), gender + 1.0
        )
        np.testing.assert_allclose(
            narrowed.design_trans().ravel(), np.log(gender + 2.0)
        )

    def test_the_coefficient_carries_the_term_as_written(self, static_data):
        """R deparses and respaces; nothing here reformats, and that is a choice."""
        narrowed = self._select(static_data, "~ log(Gender+2) | Gender")
        assert narrowed.names_cov_life == ["log(Gender+2)"]

    def test_a_star_interaction_gives_main_effects_and_the_product(
        self, static_data
    ):
        """`FI-07`'s first half, spelled exactly as the spec spells it."""
        narrowed = self._select(static_data, "~ Gender * Channel | . - Gender")
        assert narrowed.names_cov_life == ["Gender", "Channel", "Gender.Channel"]
        assert narrowed.names_cov_trans == ["Channel"]
        assert narrowed.design_life().shape == (600, 3)
        assert narrowed.design_trans().shape == (600, 1)

    def test_the_product_column_is_the_product(self, static_data):
        design = self._select(
            static_data, "~ Gender * Channel | Gender"
        ).design_life()
        np.testing.assert_allclose(design[:, 2], design[:, 0] * design[:, 1])

    def test_a_colon_interaction_gives_the_product_alone(self, static_data):
        """Which is the whole of R's distinction between ``*`` and ``:``."""
        narrowed = self._select(static_data, "~ Gender : Channel | Gender")
        assert narrowed.names_cov_life == ["Gender.Channel"]

    def test_an_unevaluable_expression_says_which_one(self, static_data):
        with pytest.raises(ValueError, match="cannot evaluate"):
            self._select(static_data, "~ log(NoSuchColumn) | Gender")

    def test_a_column_whose_own_name_looks_like_a_call_still_selects(
        self, apparel_trans, apparel_static_cov
    ):
        """A real column wins over any interpretation of its name.

        The order matters: ``name in frame.columns`` is checked first, so a
        covariate someone called ``log(spend)`` selects rather than being
        evaluated. Contrived, and exactly the sort of thing a term-parsing
        change breaks silently.
        """
        from clvtools import ClvDataStaticCov

        cov = apparel_static_cov.rename(columns={"Gender": "log(spend)"})
        data = ClvDataStaticCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            cov,
            names_cov_life=["log(spend)"], names_cov_trans=["Channel"],
        )
        narrowed = data.with_covariates(["log(spend)"], ["Channel"])
        np.testing.assert_array_equal(
            narrowed.design_life().ravel(),
            data.design_life(["log(spend)"]).ravel(),
        )


class TestAnUnusedArgumentIsAnError:
    """Spec `V-04`: "every user-facing function errors on unused arguments".

    The spec audit recorded this as a divergence, on the grounds that
    ``**kwargs`` forwarding would surface a typo at a *private* signature.
    Re-run, it does not: all three entry points name a public one, because the
    functions ``latent_attrition`` and ``spending`` forward to are the same ones
    the README tells you to call directly. ``fit_pnbd() got an unexpected
    keyword argument 'hessain'`` is as good an answer as R's, and points at a
    signature the caller can read.

    A test rather than a README finding, for that reason: it is not a
    divergence, and the only thing wrong was the note.
    """

    @pytest.fixture(scope="class")
    def plain(self, apparel_trans):
        return ClvData(apparel_trans, time_unit="week", estimation_split=104)

    @pytest.fixture(scope="class")
    def covariates(self, apparel_trans, apparel_static_cov):
        from clvtools import ClvDataStaticCov

        return ClvDataStaticCov(
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            apparel_static_cov,
            names_cov_life=["Gender"], names_cov_trans=["Gender"],
        )

    def test_latent_attrition_names_a_public_function(self, plain):
        with pytest.raises(TypeError, match=r"fit_pnbd\(\) got an unexpected"):
            latent_attrition("pnbd", plain, nonsense=1)

    def test_a_plausible_typo_is_refused_rather_than_ignored(self, plain):
        """``hessain`` for ``hessian`` -- the failure mode that matters.

        Ignored silently, it would compute a Hessian the caller asked not to
        have, or skip one they wanted, and the table would look right either
        way.
        """
        with pytest.raises(TypeError, match="hessain"):
            latent_attrition("pnbd", plain, hessain=False)

    def test_the_covariate_path_names_its_own_entry_point(self, covariates):
        with pytest.raises(TypeError, match="fit_pnbd_staticcov"):
            latent_attrition(
                "pnbd", covariates, formula="~ Gender | Gender", nonsense=1
            )

    def test_and_so_does_the_spending_path(self, plain):
        from clvtools import spending

        with pytest.raises(TypeError, match="fit_gg"):
            spending("gg", plain, nonsense=1)
