r"""What the time-varying likelihood does where no oracle can follow it.

Backlog item 28, and the second half of finding 10. :math:`F_2` is a sum of
per-interval terms of the form :math:`{}_2F_1(\cdot)/\alpha^{r+s+x}`, and the
divisor grows with the customer's transaction count: past :math:`x = 160` for
one of the two customers below and :math:`x = 190` for the other, *every* term
is under float64 and :math:`F_2` was therefore exactly zero.
``log_likelihood_customer`` then took its ``F2 == 0`` branch --
``log F_0 + log F_3``, the likelihood of a customer who is certainly dead --
and said nothing about it.

That is not a rounding error. :math:`F_3` is
:math:`(B_{k_T}{+}\alpha_0)^{-(x+r)}`, which underflows at the *same* rate, so
the ratio :math:`F_1F_2/F_3` that the branch throws away is O(1). At
:math:`x = 200` the answer was wrong by 79 log-units for one of them and 225
for the other.

**CLVTools cannot be asked about this.** Its C++ forms the same quotient the
same way and underflows in the same place, so the committed fixtures agree with
the broken arrangement by construction -- and the apparel cohort's largest
:math:`x` is 21, so no fixture reaches the regime at all. What replaces the
oracle is the nesting S3.3 asserts: with every covariate coefficient zero the
time-varying model *is* the standard Pareto/NBD, whose likelihood
``clvtools.pnbd`` computes in closed form and in log space throughout. That
holds at any :math:`x`, so it can be evaluated exactly where the fixtures stop.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from clvtools.pnbd import log_likelihood as plain_log_likelihood
from clvtools.pnbd import probability_alive as plain_probability_alive
from clvtools.pnbd.dyncov import (
    _log_diff_exp,
    _signed_logsumexp,
    log_likelihood_customer,
)

#: CLVTools' own fitted Pareto/NBD parameters for the apparel cohort.
PARAMS = (1.4490, 48.6361, 0.5613, 46.8844)

#: 5 to 100 are representable for both customers; 160 is past the first's
#: threshold and at the second's; 200 to 400 are past both, and are where the
#: value form silently answered a different question.
COUNTS = [5.0, 20.0, 50.0, 100.0, 160.0, 200.0, 300.0, 400.0]


@pytest.fixture(scope="module", params=["longest-walk", "mid-period"])
def heavy(request, dyncov_walks):
    """A real customer's walks, with the transaction count dialled up.

    Two of them, because the two things that drive the underflow pull in
    opposite directions. ``longest-walk`` has 105 covariate intervals to
    combine and the largest :math:`B_{k_T}`, so it underflows soonest and by
    the widest margin; ``mid-period`` last purchased halfway through the
    estimation window, which is the shape a customer with this many
    transactions would actually have, and has a real lifetime walk where the
    first has none -- both of ``d_i``'s forms.

    Only ``x`` is varied. The walks stay the customer's own, and the standard
    model is handed exactly the same :math:`(x, t_x, T_{cal})`, so the two are
    answering one question however implausible the customer.

    Zero coefficients throughout: that is the limit S3.3 names, in which every
    covariate multiplier is 1 and the standard model applies.
    """
    customers = dyncov_walks.customers(np.zeros(3), np.zeros(3))
    if request.param == "longest-walk":
        return max(customers, key=lambda c: c.aux_walk_life.n_elem)
    return min(customers, key=lambda c: abs(c.t_x - 52.0))


def _dyncov_ll(customer, x):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return log_likelihood_customer(
            *PARAMS, dataclasses.replace(customer, x=x)
        )


def _plain(fn, customer, x):
    """``fn`` from ``clvtools.pnbd`` for this customer alone.

    ``log_likelihood`` returns the sample total and ``probability_alive`` one
    value per customer; with one customer those are the same number.
    """
    return float(np.ravel(fn(
        np.array([x]), np.array([customer.t_x]), np.array([customer.T_cal]),
        *PARAMS,
    ))[0])


class TestAHeavyBuyerAgainstTheStandardModel:
    """S3.3's nesting, evaluated past the point where ``F_2`` is representable."""

    @pytest.mark.parametrize("x", COUNTS)
    def test_the_likelihood_matches(self, heavy, x):
        got = _dyncov_ll(heavy, x)["LL"]
        want = _plain(plain_log_likelihood, heavy, x)
        assert got == pytest.approx(want, rel=1e-12, abs=1e-9)

    @pytest.mark.parametrize("x", COUNTS)
    def test_and_so_does_palive(self, heavy, x):
        r""":math:`P(\text{alive})` is the ratio the old branch destroyed.

        The alive-only numerator *is* :math:`F_0F_3`, so when the denominator
        collapsed to it the ratio was exactly 1 -- certainty that a customer
        with hundreds of transactions and none for two years is still active.
        See ``test_the_value_form_would_have_been_wrong_by_225_log_units``.
        """
        row = _dyncov_ll(heavy, x)
        # ``log F_0 + log F_3`` is the alive-only numerator: expand both and
        # every term of ``dyncov.probability_alive``'s ``log F_1`` is there.
        got = float(np.exp(row["log_F0"] + row["log_F3"] - row["LL"]))
        want = _plain(plain_probability_alive, heavy, x)
        assert got == pytest.approx(want, rel=1e-9)

    def test_the_size_of_what_the_value_form_used_to_lose(self, heavy):
        """Pinned, so that a regression is unmistakable rather than plausible.

        Reconstructing what the old arrangement returned needs nothing but this
        row: ``F2`` reported as a value is still zero out here -- so is
        CLVTools' -- and the branch that selected is ``log F_0 + log F_3``,
        which is also the alive-only numerator. So the old likelihood was too
        small by tens of log-units and the old ``PAlive`` was exactly 1.
        """
        row = _dyncov_ll(heavy, 200.0)
        assert row["F2"] == 0.0, "the value form still has nothing to report"
        old = row["log_F0"] + row["log_F3"]
        assert row["LL"] - old > 70.0
        assert row["LL"] == pytest.approx(
            _plain(plain_log_likelihood, heavy, 200.0), rel=1e-12
        )

        # ``PAlive`` is ``exp(numerator - LL)``. With the old ``LL`` that is
        # exp(0); with this one it is what the standard model says, which is
        # not close to certainty by any margin the eye can measure.
        want = _plain(plain_probability_alive, heavy, 200.0)
        assert float(np.exp(old - row["LL"])) == pytest.approx(want, rel=1e-9)
        assert want < 1e-30

    def test_there_is_no_step_where_the_terms_stop_being_representable(
        self, heavy
    ):
        """Swept, rather than sampled at the eight counts above.

        The old code's error switches on at whichever :math:`x` underflows the
        last surviving term, and grows from there; the parametrised cases could
        in principle straddle it. This walks the crossing in steps of two and
        holds every one of them to the standard model.
        """
        for x in np.arange(150.0, 230.0, 2.0):
            got = _dyncov_ll(heavy, x)["LL"]
            want = _plain(plain_log_likelihood, heavy, x)
            assert got == pytest.approx(want, rel=1e-12), f"x = {x}"


class TestTheAliveOnlyBranchIsNowReachedOnlyWhenItIsRight:
    r""":math:`F_2 = 0` should mean an auxiliary walk of no length.

    Customer 262 of the apparel cohort made their last purchase exactly at the
    estimation end, so :math:`t_x = T_{cal}` and there is no interval left for
    the customer to have died in. Both hypergeometrics are then evaluated at
    identical arguments and cancel term for term -- an exact zero, and
    ``log F_0 + log F_3`` is the right answer. That is the only customer for
    whom it is, at either grid vector.
    """

    def test_the_one_customer_with_a_zero_length_auxiliary_walk(
        self, dyncov_walks
    ):
        customers = dyncov_walks.customers(np.zeros(3), np.zeros(3))
        exact = [c for c in customers if c.t_x == c.T_cal]
        assert len(exact) == 1
        row = _dyncov_ll(exact[0], exact[0].x)
        assert row["F2"] == 0.0
        assert row["LL"] == pytest.approx(row["log_F0"] + row["log_F3"], rel=1e-15)

    def test_and_a_heavy_buyer_no_longer_joins_them(self, heavy):
        """The same branch, for a customer who has a lifetime to integrate over.

        Their ``F2`` reports as zero in the value column and is not zero in the
        likelihood -- which is the whole of item 28 in one assertion.
        """
        row = _dyncov_ll(heavy, 400.0)
        assert row["F2"] == 0.0
        assert row["LL"] != pytest.approx(row["log_F0"] + row["log_F3"])


class TestTheSignedLogArithmetic:
    """The two primitives, on the cases the models do not reach."""

    def test_a_difference_of_equal_terms_is_an_exact_zero(self):
        magnitude, sign = _log_diff_exp(np.float64(3.5), np.float64(3.5))
        assert float(magnitude) == -np.inf
        assert float(sign) == 0.0

    def test_a_zero_term_leaves_the_other_untouched(self):
        magnitude, sign = _log_diff_exp(np.float64(-np.inf), np.float64(-4.0))
        assert (float(magnitude), float(sign)) == (-4.0, -1.0)

    def test_a_sum_of_zero_terms_is_zero_and_not_nan(self):
        """Every term cancelled, which is not the same as a failure."""
        magnitude, sign = _signed_logsumexp(
            np.array([-np.inf, 0.0]), np.array([0.0, 0.0])
        )
        assert (magnitude, sign) == (-np.inf, 0.0)

    def test_an_unusable_term_carries_to_the_whole_sum(self):
        """Rather than being dropped, which would give a plausible answer."""
        for bad in (np.nan, np.inf):
            magnitude, sign = _signed_logsumexp(
                np.array([-3.0, bad]), np.array([1.0, 1.0])
            )
            assert not np.isfinite(magnitude)
            assert sign == 1.0

    def test_terms_below_float64_still_cancel_correctly(self):
        r""":math:`e^{-800} - e^{-801}`, which no float64 can hold.

        Both are zero as values, so the value form gives 0; the signed logs
        give :math:`-800 + \log(1 - e^{-1})`, and the sign is positive.
        """
        magnitude, sign = _signed_logsumexp(
            np.array([-800.0, -801.0]), np.array([1.0, -1.0])
        )
        assert sign == 1.0
        assert magnitude == pytest.approx(-800.0 + np.log1p(-np.exp(-1.0)))
        assert float(np.exp(-800.0) - np.exp(-801.0)) == 0.0
