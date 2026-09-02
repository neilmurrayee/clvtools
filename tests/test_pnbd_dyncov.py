r"""S3.3 and S6.4.2 - the Pareto/NBD with time-varying covariates.

This is the only model in the paper whose likelihood is not written out: S3.3
gives the construction and points to Bachmann et al. (2021) Appendix A.1. So
there is no printed expression to transcribe, and correctness rests entirely on
agreement with the reference implementation.

That makes the *granularity* of the comparison matter. ``pnbd_dyncov_LL_ind``
can be asked for thirty per-customer quantities rather than just the
log-likelihood, so every block is checked separately -- the walk integrals
``Bjsum``/``Bksum``, the partial sums ``B_i`` and ``D_i``, the three ``F_2``
terms, and the ``a``/``b`` arguments handed to the hypergeometrics. A single
total agreeing could hide two errors cancelling; thirty columns agreeing at two
different parameter vectors could not.

Comparisons carry an absolute floor scaled to whatever each column feeds into,
because several of these quantities are legitimately zero. ``F2.2`` is the clear
case: it is exactly 0 for 599 of the 600 customers, not by underflow but because
its two hypergeometrics are evaluated at identical arguments and cancel --
see ``test_the_last_interval_term_is_structurally_zero``. Judged by relative
error, this implementation's 1e-22 against that 0 reports a failure of 1e273.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv, fixture_json

from clvtools.pnbd.dyncov import (
    EMPTY_WALK,
    INTERMEDIATE_NAMES,
    TransactionWalk,
    Walk,
    a1sum,
    b_i,
    bjsum,
    bksum,
    d_i,
    log_likelihood,
    log_likelihood_ind,
    walk_integral,
)

GRID = fixture_json("dyncov_ll_grid")
CASES = list(GRID["params"])


def _split(case: str):
    r"""``(r, alpha, s, beta, gamma_life, gamma_trans)`` from a grid entry.

    The stored vector is ``[log r, log alpha, log s, log beta, life..., trans...]``
    -- model parameters on the log scale, covariate parameters natural.
    """
    p = np.asarray(GRID["params"][case], dtype=float)
    r, alpha, s, beta = np.exp(p[:4])
    return r, alpha, s, beta, p[4:7], p[7:10]


class TestWalk:
    """The walk abstraction itself."""

    def test_reports_its_ends(self):
        w = Walk(np.array([1.0, 2.0, 3.0]))
        assert (w.n_elem, w.first, w.last) == (3, 1.0, 3.0)

    def test_element_access_is_zero_based(self):
        w = Walk(np.array([10.0, 20.0, 30.0]))
        assert w.elem(0) == 10.0
        assert w.elem(2) == 30.0

    def test_sum_from_to_includes_both_ends(self):
        """Mirrors Armadillo's ``subvec``, which the C++ relies on."""
        w = Walk(np.array([1.0, 2.0, 4.0, 8.0]))
        assert w.sum_from_to(1, 2) == 6.0
        assert w.sum_from_to(0, 3) == 15.0

    def test_sum_middle_excludes_both_ends(self):
        w = Walk(np.array([1.0, 2.0, 4.0, 8.0]))
        assert w.sum_middle() == 6.0

    def test_sum_middle_needs_three_elements(self):
        with pytest.raises(ValueError, match="at least 3 elements"):
            Walk(np.array([1.0, 2.0])).sum_middle()

    def test_an_empty_walk_has_no_elements(self):
        """A customer with no repeat purchase has no real lifetime walk."""
        assert EMPTY_WALK.n_elem == 0


class TestWalkIntegral:
    r""":math:`\int \exp(\gamma'x)` over a walk, the core of every ``B`` term."""

    def test_a_walk_inside_one_interval_is_rate_times_length(self):
        w = TransactionWalk(np.array([2.0]), d1=0.3, tjk=0.4)
        assert walk_integral(w) == pytest.approx(2.0 * 0.4)

    def test_two_intervals_split_at_d1(self):
        w = TransactionWalk(np.array([2.0, 3.0]), d1=0.25, tjk=1.0)
        assert walk_integral(w) == pytest.approx(2.0 * 0.25 + 3.0 * 0.75)

    def test_whole_intervals_in_between_contribute_their_value(self):
        """Each full covariate period is one time unit long."""
        w = TransactionWalk(np.array([2.0, 5.0, 7.0, 3.0]), d1=0.5, tjk=3.5)
        expected = 2.0 * 0.5 + (5.0 + 7.0) + 3.0 * (3.5 - 0.5 - 2.0)
        assert walk_integral(w) == pytest.approx(expected)

    def test_constant_covariates_reduce_to_the_elapsed_time(self):
        """With every multiplier 1 the integral is just the walk's length.

        This is the sense in which the time-varying model contains the
        time-invariant one: constant covariates give back ``lambda * t``.
        """
        for n, tjk in ((1, 0.4), (2, 1.6), (5, 4.25)):
            w = TransactionWalk(np.ones(n), d1=0.6 if n > 1 else tjk, tjk=tjk)
            assert walk_integral(w) == pytest.approx(tjk)

    def test_scaling_every_multiplier_scales_the_integral(self):
        base = TransactionWalk(np.array([2.0, 5.0, 7.0]), d1=0.5, tjk=2.5)
        scaled = TransactionWalk(base.values * 3.0, d1=0.5, tjk=2.5)  # noqa: PD011
        assert walk_integral(scaled) == pytest.approx(3.0 * walk_integral(base))


class TestA1Sum:
    def test_a_zero_repeater_contributes_nothing(self):
        """No repeat transactions, so the product of covariates is empty."""
        assert a1sum([]) == 0.0

    def test_uses_the_covariate_active_at_the_transaction(self):
        """Each walk ends *at* a transaction, so its last element is the one."""
        w1 = TransactionWalk(np.array([9.0, 2.0]), d1=0.5, tjk=1.0)
        w2 = TransactionWalk(np.array([9.0, 9.0, 4.0]), d1=0.5, tjk=2.0)
        assert a1sum([w1, w2]) == pytest.approx(np.log(2.0) + np.log(4.0))


class TestBAndDTerms:
    def test_bksum_extends_bjsum_by_the_auxiliary_walk(self):
        real = [TransactionWalk(np.array([2.0]), d1=0.3, tjk=0.5)]
        aux = TransactionWalk(np.array([3.0]), d1=0.4, tjk=1.5)
        assert bksum(bjsum(real), aux) == pytest.approx(
            bjsum(real) + walk_integral(aux)
        )

    def test_bjsum_of_no_walks_is_zero(self):
        assert bjsum([]) == 0.0

    def test_b_i_at_one_uses_only_the_first_interval(self):
        aux = TransactionWalk(np.array([2.0, 3.0, 4.0]), d1=0.25, tjk=2.5)
        assert b_i(1, 10.0, aux) == pytest.approx(2.0 * -10.0)

    def test_b_i_falls_with_i(self):
        r"""``B_i`` carries the :math:`-t_x` offset that anchors the auxiliary
        walk to the last transaction, so it is negative and grows in magnitude
        as more intervals are covered."""
        aux = TransactionWalk(np.array([2.0, 3.0, 4.0, 5.0]), d1=0.25, tjk=3.5)
        values = [b_i(i, 1.0, aux) for i in range(1, 5)]
        assert all(v < 0 for v in values)
        assert all(np.diff(values) < 0)

    def test_d_i_at_one_is_zero_without_a_real_lifetime_walk(self):
        """First and last element coincide and cancel."""
        aux = Walk(np.array([2.0, 3.0]))
        assert d_i(1, EMPTY_WALK, aux, d_omega=0.4) == 0.0

    def test_d_i_treats_the_two_lifetime_walks_as_one_span(self):
        """The real walk is summed whole, whatever ``i`` is."""
        real = Walk(np.array([2.0, 3.0, 4.0]))
        aux = Walk(np.array([5.0, 6.0, 7.0]))
        contributions = [d_i(i, real, aux, d_omega=0.5) for i in (1, 2, 3)]
        # The real walk's contribution is the same in each.
        assert len({round(c, 12) for c in contributions}) == 3


@pytest.mark.oracle
class TestAgainstOracle:
    """Every intermediate quantity, at two parameter vectors."""

    @staticmethod
    @pytest.fixture(scope="class", params=CASES)
    def compared(request, dyncov_walks):
        case = request.param
        r, alpha, s, beta, g_life, g_trans = _split(case)
        got = log_likelihood_ind(
            dyncov_walks, r, alpha, s, beta, g_life, g_trans, intermediates=True
        )
        want = (
            fixture_csv(f"dyncov_ll_{case}").set_index("Id").loc[got.index]
        )
        return case, got, want

    def test_every_intermediate_column_matches(self, compared):
        case, got, want = compared
        for column in INTERMEDIATE_NAMES:
            g = got[column].to_numpy(dtype=float)
            w = want[column].to_numpy(dtype=float)

            # Where the reference has no value, neither should this.
            np.testing.assert_array_equal(
                np.isfinite(g), np.isfinite(w),
                err_msg=f"{case}/{column}: finite-ness differs",
            )

            finite = np.isfinite(w)
            if not finite.any():
                continue

            # The absolute floor is scaled to whatever the column feeds into.
            # For the three F2 components that is F2 itself, not their own
            # magnitude: F2.2 is structurally zero (see the test below), so its
            # own scale is 0 and any floor derived from it would be vacuous.
            if column.startswith("F2."):
                reference = np.abs(want["F2"].to_numpy(dtype=float)[finite])
                atol = 1e-8 * np.maximum(reference, 1e-300)
            else:
                atol = max(np.max(np.abs(w[finite])) * 1e-12, 1e-300)

            # Asserted directly rather than through assert_allclose, which
            # cannot format a per-element atol into its failure message.
            difference = np.abs(g[finite] - w[finite])
            tolerance = 1e-8 * np.abs(w[finite]) + atol
            worst = int(np.argmax(difference - tolerance))
            assert np.all(difference <= tolerance), (
                f"{case}/{column}: worst at row {worst}, "
                f"got {g[finite][worst]!r} want {w[finite][worst]!r}"
            )

    def test_the_last_interval_term_is_structurally_zero(self, compared):
        r"""``F2.2`` vanishes identically, for a reason worth recording.

        Its two hypergeometrics are evaluated at
        :math:`(\alpha_{1}, \beta_{1}) = (a_{kt}+\alpha_0, \ldots)` and
        :math:`(\alpha_{2}, \beta_{2}) = (a_{T}+\alpha_0, \ldots)`, and

        .. math::
            a_{kt} = \mathrm{Bjsum} + B_T + A_{kT}(t_x + d_T + k_T - 2),
            \qquad
            a_T = \mathrm{Bjsum} + B_T + T_{cal} A_{kT}

        are equal because the auxiliary walk spans exactly from :math:`t_x` to
        :math:`T_{cal}` by construction, so
        :math:`t_x + d_T + k_T - 2 = T_{cal}`. The same holds for the pair
        :math:`b_{kT}, b_T`. A difference of two identical terms is zero, which
        is what CLVTools returns for 599 of the 600 customers.

        This implementation reaches about 1e-22 instead of exactly 0, because
        the identity holds to within a rounding unit rather than bit-for-bit
        once the inputs have been through a file. That is six orders of
        magnitude below F2 itself, and the log-likelihood is unaffected.
        """
        _case, got, want = compared
        akt = want["akt"].to_numpy(dtype=float)
        aT = want["aT"].to_numpy(dtype=float)
        finite = np.isfinite(akt) & np.isfinite(aT)
        np.testing.assert_array_equal(akt[finite], aT[finite])

        f2 = np.abs(want["F2"].to_numpy(dtype=float))
        f2_2 = np.abs(got["F2.2"].to_numpy(dtype=float))
        both = np.isfinite(f2) & np.isfinite(f2_2) & (f2 > 0)
        assert np.all(f2_2[both] < 1e-6 * f2[both])

    def test_the_batched_middle_sum_matches_the_scalar_one(self, dyncov_walks):
        r"""The vectorised :math:`\sum_{i=2}^{k_T-1} Y_i` against the loop it replaced.

        ``_f2_middle`` builds every in-between covariate interval's term as
        arrays. The oracle comparison above already holds it to CLVTools, but
        only through ``F2.3``; this holds it against the scalar ``b_i``,
        ``d_i`` and ``_hyp_term`` directly, which is what it was derived from.
        It is also what keeps ``d_i(2, ...)`` exercised: the likelihood itself
        now calls ``d_i`` only at :math:`i = 1` and :math:`i = k_T`.

        Not asserted bit-for-bit, and the reason is worth stating.
        ``Walk.sum_from_to`` calls ``ndarray.sum``, which adds pairwise, while
        the batched version accumulates the same prefixes left to right with
        ``np.cumsum``. On walks of more than eight intervals the two agree to a
        rounding unit rather than exactly -- measured at 2.2e-15 relative over
        all 600 customers, against a 2.3e-13 disagreement with CLVTools itself.
        ``docs/performance.md`` records the measurement and why this order was
        the one kept.
        """
        from clvtools.pnbd.dyncov import _f2_middle, _hyp_term

        def scalar_middle(r, alpha_0, s, beta_0, c, dT, Bjsum):
            """`_f2_middle` as it was written before the rewrite."""
            total = 0.0
            for i in range(2, c.aux_walk_trans.n_elem):
                Ai = c.aux_walk_trans.elem(i - 1)
                Bi = b_i(i, c.t_x, c.aux_walk_trans)
                ai = Bjsum + Bi + Ai * (c.t_x + dT + (i - 2.0))
                Ci = c.aux_walk_life.elem(i - 1)
                Di = d_i(i, c.real_walk_life, c.aux_walk_life, c.d_omega)
                bi = Di + Ci * (c.t_x + dT + (i - 2.0))
                total += _hyp_term(
                    r, s, c.x,
                    ai + alpha_0, (bi + beta_0) * Ai / Ci,
                    ai + Ai + alpha_0, (bi + Ci + beta_0) * Ai / Ci,
                    Ai / Ci,
                )
            return total

        for case in CASES:
            r, alpha, s, beta, g_life, g_trans = _split(case)
            customers = dyncov_walks.customers(g_life, g_trans)
            with_real = without_real = 0
            for c in customers:
                dT = c.aux_walk_trans.d1
                Bjsum = bjsum(c.real_walks_trans)
                want = scalar_middle(r, alpha, s, beta, c, dT, Bjsum)
                got = _f2_middle(r, alpha, s, beta, c, dT, Bjsum)
                assert got == pytest.approx(want, rel=1e-12, abs=1e-300), case
                if c.aux_walk_trans.n_elem > 2:
                    if c.real_walk_life.n_elem:
                        with_real += 1
                    else:
                        without_real += 1
            # Both of `d_i`'s forms have to have been taken, or half of what
            # this test claims to compare was never compared.
            both = (
                f"{case}: {with_real} customers with a real lifetime walk and "
                f"{without_real} without; both `d_i` branches must be reached."
            )
            assert with_real > 0, both
            assert without_real > 0, both

    def test_the_log_likelihood_matches(self, compared):
        _case, got, want = compared
        np.testing.assert_allclose(got["LL"], want["LL"], rtol=1e-10)

    def test_the_sample_log_likelihood_matches(self, compared, dyncov_walks):
        case, got, _ = compared
        r, alpha, s, beta, g_life, g_trans = _split(case)
        total = log_likelihood(dyncov_walks, r, alpha, s, beta, g_life, g_trans)
        assert total == pytest.approx(GRID["LL.sum"][case], rel=1e-11)
        assert total == pytest.approx(float(got["LL"].sum()), rel=1e-12)

    @pytest.mark.paper
    def test_the_fitted_likelihood_matches(self, dyncov_walks):
        """The MLE grid point is CLVTools' own fit, so this pins ``logLik()``."""
        r, alpha, s, beta, g_life, g_trans = _split("mle")
        total = log_likelihood(dyncov_walks, r, alpha, s, beta, g_life, g_trans)
        assert total == pytest.approx(
            fixture_json("dyncov_fit")["logLik"], abs=1e-4
        )


class TestWalkAssembly:
    def test_customers_are_built_in_the_summary_order(self, dyncov_walks):
        customers = dyncov_walks.customers(np.zeros(3), np.zeros(3))
        assert len(customers) == dyncov_walks.n_customers == 600
        assert customers[0].x == dyncov_walks.x[0]

    def test_zero_repeaters_have_no_real_transaction_walks(self, dyncov_walks):
        customers = dyncov_walks.customers(np.zeros(3), np.zeros(3))
        for customer in customers:
            if customer.x == 0:
                assert customer.real_walks_trans == []
            else:
                assert len(customer.real_walks_trans) == customer.x

    def test_zero_coefficients_make_every_multiplier_one(self, dyncov_walks):
        r"""``exp(0'x) = 1``, so every walk is all ones and the walk integrals
        collapse to elapsed time."""
        customers = dyncov_walks.customers(np.zeros(3), np.zeros(3))
        for customer in customers[:20]:
            assert np.allclose(customer.aux_walk_trans.values, 1.0)
            assert np.allclose(customer.aux_walk_life.values, 1.0)
            assert walk_integral(customer.aux_walk_trans) == pytest.approx(
                customer.aux_walk_trans.tjk
            )

    def test_real_and_auxiliary_lifetime_walks_do_not_overlap(self, dyncov_walks):
        """The interval containing the last transaction belongs to the auxiliary
        walk alone, which ``d_i`` depends on."""
        info = fixture_csv("dyncov_walkinfo")
        has_real = info["real_life_from"].notna()
        # Real and auxiliary lifetime walks index different arrays, so the
        # non-overlap is structural; what must hold is that a customer with
        # repeat purchases has both.
        cbs = fixture_csv("dyncov_cbs")
        assert (cbs.loc[has_real.to_numpy(), "x"] > 0).all()

    def test_rejects_a_covariate_parameter_mismatch(self, dyncov_walks):
        with pytest.raises(ValueError, match="3 attrition covariates but 2"):
            dyncov_walks.customers(np.zeros(2), np.zeros(3))
        with pytest.raises(ValueError, match="3 transaction covariates but 4"):
            dyncov_walks.customers(np.zeros(3), np.zeros(4))

    def test_weights_repeat_rows(self, dyncov_walks):
        r, alpha, s, beta, g_life, g_trans = _split("mle")
        each = log_likelihood_ind(dyncov_walks, r, alpha, s, beta, g_life, g_trans)
        weights = np.ones(dyncov_walks.n_customers)
        weights[0] = 3.0
        weighted = log_likelihood(
            dyncov_walks, r, alpha, s, beta, g_life, g_trans, weights=weights
        )
        assert weighted == pytest.approx(float(each.sum() + 2 * each[0]))


class TestNesting:
    r"""S3.3: "The standard model and the extension for time-invariant
    covariates are nested within this model. With covariate effects set to zero,
    we arrive at the standard model."

    The apparel covariates are not constant over time, so setting the
    coefficients to zero is the only one of those two limits reachable from this
    data; it is checked here against the plain Pareto/NBD likelihood.
    """

    def test_zero_coefficients_give_the_plain_likelihood(self, dyncov_walks):
        from clvtools.pnbd import log_likelihood as plain_log_likelihood

        r, alpha, s, beta = 1.4490, 48.6361, 0.5613, 46.8844
        dynamic = log_likelihood(
            dyncov_walks, r, alpha, s, beta, np.zeros(3), np.zeros(3)
        )
        plain = plain_log_likelihood(
            dyncov_walks.x, dyncov_walks.t_x, dyncov_walks.T_cal,
            r, alpha, s, beta,
        )
        assert dynamic == pytest.approx(plain, rel=1e-8)


@pytest.mark.oracle
class TestWalkConstruction:
    r"""Building the walks from raw data, against the oracle's own tables.

    This is checked separately from the likelihood, and before it: the
    :func:`dyncov_walks` fixture loads CLVTools' walks directly, so a fault in
    construction cannot hide behind a correct likelihood or the reverse.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def built():
        from clvtools import ClvData, load_apparel_dyn_cov, load_apparel_trans
        from clvtools.pnbd.dyncov import build_walks

        return build_walks(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(),
            names_cov_life=["High.Season", "Gender", "Channel"],
            names_cov_trans=["High.Season", "Gender", "Channel"],
        )

    def test_customer_summary_matches(self, built):
        want = fixture_csv("dyncov_cbs").set_index("Id").loc[built.ids]
        np.testing.assert_array_equal(built.x, want["x"])
        np.testing.assert_array_equal(built.T_cal, want["T.cal"])
        np.testing.assert_array_equal(built.d_omega, want["d_omega"])
        # t_x is computed from whole days here and is the exact value; the
        # fixture went through CSV at 15 significant digits.
        np.testing.assert_allclose(built.t_x, want["t.x"], atol=1e-12)

    def test_d_omega_is_one_for_this_cohort(self, built):
        r"""Every first purchase falls on a covariate boundary.

        The apparel cohort's covariate grid starts on the cohort's own first
        purchase date, so ``d_omega`` takes the boundary value of 1 throughout.
        This is what pins the "``d`` is 1 on the lower boundary" rule: with the
        naive definition it would be 0 for all 600 customers.
        """
        np.testing.assert_array_equal(built.d_omega, np.ones(600))

    def test_auxiliary_walk_indices_match(self, built):
        want = fixture_csv("dyncov_walkinfo").set_index("Id").loc[built.ids]
        np.testing.assert_array_equal(
            built.walkinfo_aux_life, want[["aux_life_from", "aux_life_to"]]
        )
        np.testing.assert_array_equal(
            built.walkinfo_aux_trans[:, :2],
            want[["aux_trans_from", "aux_trans_to"]],
        )

    def test_auxiliary_transaction_walk_d1_and_tjk_match(self, built):
        want = fixture_csv("dyncov_walkinfo").set_index("Id").loc[built.ids]
        np.testing.assert_allclose(
            built.walkinfo_aux_trans[:, 2:],
            want[["aux_trans_d1", "aux_trans_tjk"]],
            atol=1e-12,
        )

    def test_real_lifetime_walk_indices_match(self, built):
        want = fixture_csv("dyncov_walkinfo").set_index("Id").loc[built.ids]
        want_values = want[["real_life_from", "real_life_to"]].to_numpy(float)
        np.testing.assert_array_equal(
            np.isfinite(built.walkinfo_real_life), np.isfinite(want_values)
        )
        present = np.isfinite(want_values)
        np.testing.assert_array_equal(
            built.walkinfo_real_life[present], want_values[present]
        )

    def test_real_transaction_walks_match(self, built):
        want = fixture_csv("dyncov_walkinfo_real_trans")[
            ["walk_from", "walk_to", "d1", "tjk"]
        ].to_numpy(float)
        assert built.walkinfo_real_trans.shape == want.shape
        np.testing.assert_allclose(built.walkinfo_real_trans, want, atol=1e-12)

    def test_covariate_matrices_match(self, built):
        for kind in ("aux_life", "real_life", "aux_trans", "real_trans"):
            want = fixture_csv(f"dyncov_covdata_{kind}").to_numpy(dtype=float)
            got = getattr(built, f"covdata_{kind}")
            np.testing.assert_array_equal(got, want, err_msg=kind)

    def test_the_likelihood_agrees_with_the_oracles_walks(self, built):
        """End to end: raw data in, CLVTools' log-likelihood out."""
        for case in CASES:
            r, alpha, s, beta, g_life, g_trans = _split(case)
            total = log_likelihood(built, r, alpha, s, beta, g_life, g_trans)
            assert total == pytest.approx(GRID["LL.sum"][case], abs=1e-8), case

    def test_day_aggregation_is_applied(self, built):
        """S6.1's collapse to the day, which the walks inherit.

        Passing a raw log instead would give a customer who bought twice in one
        day an extra transaction and an extra zero-length walk.
        """
        from clvtools import ClvData, load_apparel_trans

        data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        summary = data.customer_summary().set_index("Id").loc[built.ids]
        np.testing.assert_array_equal(built.x, summary["x"])

    def test_every_customer_has_an_auxiliary_walk(self, built):
        """It runs from the last transaction to the window end, always non-empty."""
        lengths = built.walkinfo_aux_trans[:, 1] - built.walkinfo_aux_trans[:, 0] + 1
        assert (lengths >= 1).all()

    def test_repeat_purchases_get_one_transaction_walk_each(self, built):
        counts = np.where(
            np.isfinite(built.real_trans_from),
            built.real_trans_to - built.real_trans_from + 1,
            0,
        )
        np.testing.assert_array_equal(counts, built.x)


class TestWalkConstructionValidation:
    @staticmethod
    @pytest.fixture(scope="class")
    def data():
        from clvtools import ClvData, load_apparel_trans

        return ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)

    def test_rejects_covariates_without_a_date_column(self, data):
        from clvtools.pnbd.dyncov import build_walks

        with pytest.raises(ValueError, match=r"no 'Cov\.Date' column"):
            build_walks(data, pd.DataFrame({"Id": ["1"], "Gender": [0]}))

    def test_rejects_an_unknown_covariate_name(self, data):
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        with pytest.raises(ValueError, match="missing columns"):
            build_walks(data, load_apparel_dyn_cov(), names_cov_life=["Nope"])

    def test_rejects_customers_without_covariate_data(self, data):
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        partial = load_apparel_dyn_cov()
        partial = partial[partial["Id"] != "1"]
        with pytest.raises(ValueError, match="have no covariate data"):
            build_walks(
                data, partial,
                names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
            )

    def test_rejects_two_processes_on_different_date_grids(self, data):
        """The walk indices are derived once and used to slice both matrices.

        Nothing downstream would notice a transactional grid shifted against
        the lifetime one: every walk would simply read the wrong intervals and
        the likelihood would come back wrong but finite.
        """
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        covariates = load_apparel_dyn_cov()
        shifted = covariates.copy()
        shifted["Cov.Date"] = shifted["Cov.Date"] + pd.Timedelta(days=7)
        with pytest.raises(ValueError, match="share one date grid"):
            build_walks(
                data, covariates, shifted,
                names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
            )

    def test_rejects_a_series_too_short_for_the_walks_it_must_cover(self, data):
        """A short series would slice to fewer rows than the walk spans.

        The two grids still agree everywhere they overlap, so the grid check
        passes; it is the stacking that would silently shift every later walk.
        """
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        covariates = load_apparel_dyn_cov()
        short = covariates[covariates["Cov.Date"] <= pd.Timestamp("2006-06-01")]
        with pytest.raises(ValueError, match="periods its walk spans"):
            build_walks(
                data, covariates, short,
                names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
            )

    def test_covariate_names_default_to_every_column(self, data):
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        walks = build_walks(data, load_apparel_dyn_cov())
        assert walks.n_cov_life == walks.n_cov_trans == 3


class TestDynCovDataObject:
    """``SetDynamicCovariates()``'s analogue."""

    @staticmethod
    @pytest.fixture(scope="class")
    def dynamic():
        from clvtools import (
            ClvData,
            ClvDataDynCov,
            load_apparel_dyn_cov,
            load_apparel_trans,
        )

        return ClvDataDynCov(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(),
            names_cov_life=["High.Season", "Gender", "Channel"],
            names_cov_trans=["High.Season", "Gender", "Channel"],
        )

    def test_inherits_the_transaction_data(self, dynamic):
        assert dynamic.estimation_end == pd.Timestamp("2006-12-31")
        assert dynamic.has_holdout

    def test_builds_walks_matching_the_oracle(self, dynamic):
        want = fixture_csv("dyncov_cbs").set_index("Id")
        walks = dynamic.walks()
        np.testing.assert_array_equal(walks.x, want.loc[walks.ids, "x"])

    def test_walks_are_cached(self, dynamic):
        """They cost a full sweep of the covariate data and do not depend on
        the parameters, so building them once is worth it."""
        assert dynamic.walks() is dynamic.walks()

    def test_the_transaction_process_can_take_different_covariates(self):
        from clvtools import (
            ClvData,
            ClvDataDynCov,
            load_apparel_dyn_cov,
            load_apparel_trans,
        )

        data = ClvDataDynCov(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(),
            names_cov_life=["High.Season"],
            names_cov_trans=["High.Season", "Gender", "Channel"],
        )
        walks = data.walks()
        assert (walks.n_cov_life, walks.n_cov_trans) == (1, 3)


class TestFitValidation:
    def test_rejects_bad_start_values(self, dyncov_walks):
        from clvtools.pnbd.dyncov import fit_pnbd_dyncov

        with pytest.raises(ValueError, match="four values"):
            fit_pnbd_dyncov(dyncov_walks, start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            fit_pnbd_dyncov(dyncov_walks, start=(1.0, -1.0, 1.0, 1.0))

    def test_parameters_report_their_names(self, dyncov_walks):
        from clvtools.pnbd.dyncov import PnbdDynCovParams

        names = ["High.Season", "Gender", "Channel"]
        params = PnbdDynCovParams(
            r=1.0, alpha=50.0, s=1.0, beta=60.0,
            gamma_life=np.zeros(3), gamma_trans=np.zeros(3),
            names_cov_life=names, names_cov_trans=names,
            log_likelihood=-5752.9, converged=True, n_customers=600,
        )
        assert params.names == [
            "r", "alpha", "s", "beta",
            "life.High.Season", "life.Gender", "life.Channel",
            "trans.High.Season", "trans.Gender", "trans.Channel",
        ]
        assert params.n_parameters == 10
        assert params.aic == pytest.approx(20 - 2 * -5752.9)
        assert params.bic == pytest.approx(10 * np.log(600) - 2 * -5752.9)


@pytest.mark.dyncov_fit
class TestFit:
    r"""The full MLE. Minutes, not seconds -- run with ``-m dyncov_fit``.

    S6.4 warns about this: "the model estimation with time-varying covariates
    is computationally much more demanding than the previously detailed
    alternatives." Each likelihood evaluation sweeps 600 customers and evaluates
    some 80,000 hypergeometrics, batched over each customer's covariate
    intervals since ``docs/backlog.md`` item 9.
    """

    def test_reaches_at_least_the_oracles_optimum(self, dyncov_walks):
        r"""Runs in about ten minutes, over some 1,900 likelihood evaluations.

        The assertion is one-sided. This implementation attains -5752.623
        against CLVTools' -5752.937 -- 0.31 log-likelihood units better -- and
        the parameters differ visibly, ``life.High.Season`` most of all at
        -8.12 against -2.48. With ten parameters and a likelihood this flat
        that is unsurprising, and the earlier tests establish that both
        implementations agree about the *function* to nine significant figures
        at two fixed parameter vectors. What differs is where each optimiser
        stops, so requiring equality here would be testing SciPy against
        optimx rather than testing the model.
        """
        from clvtools.pnbd.dyncov import fit_pnbd_dyncov

        names = ["High.Season", "Gender", "Channel"]
        want = fixture_json("dyncov_fit")
        fitted = fit_pnbd_dyncov(
            dyncov_walks, names_cov_life=names, names_cov_trans=names
        )
        assert fitted.converged
        assert fitted.log_likelihood >= want["logLik"] - 1e-6
        # The transaction-process coefficients, which S6.4.1 is what a reader
        # would interpret, do land in the same place.
        coefficients = fitted.coefficients
        assert coefficients["trans.High.Season"] == pytest.approx(0.7183, abs=5e-3)
        assert coefficients["trans.Gender"] == pytest.approx(0.2649, abs=1e-2)
        assert coefficients["trans.Channel"] == pytest.approx(0.6137, abs=1e-2)


class TestNumericalEdgeCases:
    """The guarded branches, driven directly rather than waited for."""

    def test_the_hypergeometric_fallback_matches_where_both_work(self):
        r"""Where SciPy converges, the limiting form should be close.

        The fallback only fires when the series fails, so it is normally
        unobserved. Forcing it on arguments SciPy *can* handle shows the two
        are the same quantity rather than one being a guess.
        """
        from scipy import special

        from clvtools.pnbd.dyncov import _hyp_alpha_ge_beta, _hyp_beta_gt_alpha

        r, s, x = 1.5, 0.8, 3.0
        a = r + s + x
        alpha_1, beta_1, alpha_2, beta_2 = 120.0, 60.0, 130.0, 62.0
        got = _hyp_alpha_ge_beta(r, s, x, alpha_1, beta_1, alpha_2, beta_2)
        direct = sum(
            sign * special.hyp2f1(a, s + 1.0, a + 1.0, 1.0 - beta / alpha) / alpha**a
            for alpha, beta, sign in
            ((alpha_1, beta_1, 1.0), (alpha_2, beta_2, -1.0))
        )
        assert got == pytest.approx(direct, rel=1e-12)

        got = _hyp_beta_gt_alpha(r, s, x, 60.0, 120.0, 62.0, 130.0)
        assert np.isfinite(got)

    def test_the_fallback_keeps_the_result_finite(self):
        """Large ``x`` with ``z`` near 1 is where SciPy gives up."""
        from scipy import special

        from clvtools.pnbd.dyncov import _hyp_alpha_ge_beta

        r, s, x = 1.5, 0.8, 400.0
        a = r + s + x
        assert not np.isfinite(special.hyp2f1(a, s + 1.0, a + 1.0, 1.0 - 1e-9))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            got = _hyp_alpha_ge_beta(r, s, x, 1e9, 1.0, 1.1e9, 1.0)
        assert not np.isnan(got)

    def test_a_non_finite_f2_gives_a_non_finite_likelihood(self, dyncov_walks):
        """Rather than a plausible number the optimiser would then trust."""
        from clvtools.pnbd.dyncov import log_likelihood_ind

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            got = log_likelihood_ind(
                dyncov_walks, 1e-300, 1e300, 1e-300, 1e300,
                np.full(3, 300.0), np.full(3, 300.0),
            )
        assert not np.isfinite(got).all()

    def test_f2_is_non_negative_on_this_data(self, dyncov_walks):
        r"""And structurally so, which is worth knowing.

        CLVTools guards against a negative ``F_2`` -- "The F2 can be
        negative/zero for some observations and log(F2) cannot be calculated"
        -- and that guard is reproduced here. But on well-formed walks it
        cannot fire. Each term is
        :math:`{}_2F_1(\cdot;z_1)/\alpha_1^{a} - {}_2F_1(\cdot;z_2)/\alpha_2^{a}`
        with :math:`\alpha_2 = \alpha_1 + A \ge \alpha_1`, so the subtracted
        term is the smaller one and the difference is non-negative; ``F2.2``
        vanishes identically. Checked at both grid points and across random
        parameter draws.
        """
        from clvtools.pnbd.dyncov import log_likelihood_ind

        rng = np.random.default_rng(0)
        draws = [_split(case) for case in CASES]
        for _ in range(5):
            model = np.exp(rng.uniform(-1, 3, 4)) * np.array([1.0, 20.0, 1.0, 20.0])
            draws.append(
                (*model, rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3))
            )

        for r, alpha, s, beta, g_life, g_trans in draws:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                table = log_likelihood_ind(
                    dyncov_walks, r, alpha, s, beta, g_life, g_trans,
                    intermediates=True,
                )
            finite = np.isfinite(table["F2"].to_numpy(dtype=float))
            assert (table["F2"].to_numpy(dtype=float)[finite] >= 0).all()

    def test_the_negative_f2_branch_computes_the_right_thing(self, dyncov_walks):
        r"""The guard is kept for parity, so it is checked in isolation.

        With ``F_1 F_2 < 0`` but ``F_1 F_2 + F_3 > 0``, the likelihood is
        ``log F_0 + log F_3 + log1p(exp(log F_1 - log F_3) F_2)``, which is the
        same quantity as ``log F_0 + log(F_1 F_2 + F_3)`` computed without the
        cancellation.
        """
        from unittest.mock import patch

        from clvtools.pnbd import dyncov

        customer = dyncov_walks.customers(np.zeros(3), np.zeros(3))[0]
        r, alpha, s, beta = 1.4490, 48.6361, 0.5613, 46.8844

        baseline = dyncov.log_likelihood_customer(r, alpha, s, beta, customer)
        # A small negative F2, well inside the range where F1*F2 + F3 > 0.
        negative = -abs(baseline["F2"]) if baseline["F2"] else -1e-30

        real_f2 = dyncov._f2

        def fake_f2(*args, **kwargs):
            _, parts = real_f2(*args, **kwargs)
            return negative, parts

        with patch.object(dyncov, "_f2", side_effect=fake_f2):
            got = dyncov.log_likelihood_customer(r, alpha, s, beta, customer)

        log_F0, log_F1, log_F3 = (
            got["log_F0"], got["log_F1"], got["log_F3"]
        )
        direct = log_F0 + np.log(np.exp(log_F1) * negative + np.exp(log_F3))
        assert got["F2"] == negative
        assert got["LL"] == pytest.approx(direct, rel=1e-9)

    def test_rejects_data_with_nothing_in_the_estimation_period(self):
        from clvtools import ClvData, load_apparel_dyn_cov, load_apparel_trans
        from clvtools.pnbd.dyncov import build_walks

        trans = load_apparel_trans()
        data = ClvData(trans, time_unit="week", estimation_split=104)
        # Move the window to before any transaction exists.
        data.estimation_end = pd.Timestamp("2004-01-01")
        with pytest.raises(ValueError, match="no transactions fall within"):
            build_walks(data, load_apparel_dyn_cov())


class TestFitMechanics:
    """The fit's plumbing, without paying for a full optimisation."""

    def test_runs_and_reports_its_work(self, dyncov_walks):
        from clvtools.pnbd.dyncov import fit_pnbd_dyncov

        names = ["High.Season", "Gender", "Channel"]
        fitted = fit_pnbd_dyncov(
            dyncov_walks, names_cov_life=names, names_cov_trans=names,
            options={"maxiter": 1, "maxfun": 12},
        )
        assert fitted.n_evaluations > 0
        assert fitted.n_customers == 600
        assert np.isfinite(fitted.log_likelihood)
        assert list(fitted.coefficients) == fitted.names

    def test_covariate_names_default_to_positional_labels(self, dyncov_walks):
        from clvtools.pnbd.dyncov import fit_pnbd_dyncov

        fitted = fit_pnbd_dyncov(
            dyncov_walks, options={"maxiter": 1, "maxfun": 12}
        )
        assert fitted.names[4:7] == ["life.life0", "life.life1", "life.life2"]

    def test_weights_are_accepted(self, dyncov_walks):
        from clvtools.pnbd.dyncov import fit_pnbd_dyncov

        weights = np.ones(dyncov_walks.n_customers)
        fitted = fit_pnbd_dyncov(
            dyncov_walks, weights=weights, options={"maxiter": 1, "maxfun": 12}
        )
        assert np.isfinite(fitted.log_likelihood)


class TestTheDyncovHessianIsOptional:
    """Finding 8: the fit had no ``hessian`` argument and no field.

    ``summary()`` therefore raised "fit with hessian=True", naming something
    that did not exist. It exists now and defaults to ``False``, alone among
    the fits, because differencing a likelihood that costs ~0.1 s per
    evaluation over 13 parameters is about 350 evaluations -- a minute of
    ``summary()`` on this model and nothing at all on any other. Eight
    customers keep that affordable here.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def small_walks():
        from clvtools import ClvData, load_apparel_dyn_cov, load_apparel_trans
        from clvtools.pnbd.dyncov import build_walks

        trans = load_apparel_trans()
        ids = sorted(trans["Id"].unique())[:8]
        cov = load_apparel_dyn_cov()
        return build_walks(
            ClvData(
                trans[trans["Id"].isin(ids)],
                time_unit="week", estimation_split=104,
            ),
            cov[cov["Id"].isin(ids)],
            names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
        )

    def test_it_is_off_by_default(self, small_walks):
        from clvtools.pnbd.dyncov import fit_pnbd_dyncov

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_pnbd_dyncov(small_walks, maxiter=1)
        assert fit.hessian is None
        with pytest.raises(ValueError, match="hessian=True"):
            fit.standard_errors()

    def test_and_the_advice_it_gives_can_now_be_followed(self, small_walks):
        from clvtools.pnbd.dyncov import fit_pnbd_dyncov

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_pnbd_dyncov(small_walks, maxiter=1, hessian=True)
        assert fit.hessian is not None
        assert fit.hessian.shape == (len(fit.names), len(fit.names))


class TestTheF2TermsDoNotOverflowBeforeTheyUnderflow:
    """Finding 10, and the half of it that arithmetic can fix.

    Each term was ``value / alpha ** (r + s + x)``. At ``alpha = 200`` the
    divisor passes the top of float64 by ``x = 160`` while the quotient is
    around 1e-370, so the direct form returned 0 for the wrong reason -- and a
    quotient that *is* representable, 2.4e-279 at ``x = 120``, was computed
    through an intermediate of 4.5e+278 that is one step from overflowing too.

    Forming ``exp(log value - a log alpha)`` fixes that. It does not make an
    unrepresentable number representable: at ``x = 160`` the true term is below
    float64 and the result is honestly 0. What remains -- combining the two
    terms and ``F1 * F2 + F3`` in log space, so that a genuine underflow stops
    silently selecting the alive-only branch -- is backlog item 28, because it
    changes what the oracle fixtures compare.

    The oracle cannot see any of this: CLVTools forms the same quotient the
    same way, so agreement with a fixture is agreement about the arrangement,
    not about the arithmetic.
    """

    @pytest.mark.parametrize(
        "x,expected", [(10.0, 1.234719e-26), (50.0, 2.590707e-118),
                       (120.0, 2.395442e-279)]
    )
    def test_representable_terms_are_computed(self, x, expected):
        from clvtools.pnbd.dyncov import _hyp_alpha_ge_beta

        got = float(_hyp_alpha_ge_beta(0.5, 0.6, x, 200.0, 190.0, 210.0, 195.0))
        assert got == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("x", [160.0, 200.0])
    def test_unrepresentable_terms_are_zero_and_not_nan(self, x):
        """The pre-fix code gave 0 here by dividing by ``inf``, and could give
        ``nan``; the true value is below float64 either way. The point is that
        it is now zero for the honest reason."""
        from clvtools.pnbd.dyncov import _hyp_alpha_ge_beta

        got = float(_hyp_alpha_ge_beta(0.5, 0.6, x, 200.0, 190.0, 210.0, 195.0))
        assert got == 0.0
        assert not np.isnan(got)
