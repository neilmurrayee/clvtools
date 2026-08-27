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
        assert w.elem(0) == 10.0 and w.elem(2) == 30.0

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
        scaled = TransactionWalk(base.values * 3.0, d1=0.5, tjk=2.5)
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

    @pytest.fixture(scope="class", params=CASES)
    def compared(self, request, dyncov_walks):
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
        case, got, want = compared
        akt = want["akt"].to_numpy(dtype=float)
        aT = want["aT"].to_numpy(dtype=float)
        finite = np.isfinite(akt) & np.isfinite(aT)
        np.testing.assert_array_equal(akt[finite], aT[finite])

        f2 = np.abs(want["F2"].to_numpy(dtype=float))
        f2_2 = np.abs(got["F2.2"].to_numpy(dtype=float))
        both = np.isfinite(f2) & np.isfinite(f2_2) & (f2 > 0)
        assert np.all(f2_2[both] < 1e-6 * f2[both])

    def test_the_log_likelihood_matches(self, compared):
        case, got, want = compared
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
