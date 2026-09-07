"""The Pareto/NBD without covariates, paired expression by expression.

The first family onto the paired oracle, and the one the rest should follow.
Five expressions -- S3.2's likelihood in both its per-customer and summed
forms, and the three quantities S4 derives from it -- at six parameter
vectors, which is 30 paired evaluations.

The six vectors are the grid ``tools/oracle/generate_fixtures.R`` already
uses, and they are chosen rather than convenient. ``mle`` is the optimum the
paper reports; the next three straddle the ``alpha >= beta`` branch that the
Kummer U arrangement in :mod:`clvtools.special` switches on, including the
equality case that sits exactly on it; the last two push the shape parameters
to either end, where the hypergeometric series is slowest to converge. A pair
that agreed only at the optimum would say very little.

``p`` is ``(r, alpha, s, beta)`` on the natural scale in both languages. The R
snippets that need the log scale write ``log(p)`` at the call, which is where
the reader can see it -- CLAUDE.md lists that convention first among the traps
that have already cost time, and pairing is what turns it from a note into
something a run can check.
"""

from __future__ import annotations

import numpy as np
from pairs import pair, prelude_inputs

from clvtools.pnbd import aggregate

#: The estimation-period sufficient statistics both sides are evaluated on,
#: read from the prelude's own recording rather than recomputed or taken from a
#: second copy. That is what makes the input check in live mode mean something:
#: the vectors compared against R are literally the ones passed in below, so a
#: difference between the two implementations can only be the *expression*.
#: ``tests/test_pairs.py`` also holds this recording to ``cbs_estimation.csv``,
#: so the paired oracle and the fixture suite describe the same 600 customers.
_INPUTS = prelude_inputs("apparel")
X = _INPUTS["x"]
TX = _INPUTS["t.x"]
TC = _INPUTS["T.cal"]

#: S6.2.1's optimum, then the five points chosen to exercise the branches.
#: Identical to the grid in ``tools/oracle/generate_fixtures.R``, so a pair and
#: the fixture it will eventually replace are evaluated at the same places.
GRID = {
    "mle": (1.449, 48.6361, 0.5613, 46.8844),
    "alpha.gt.beta": (1.0, 10.0, 1.0, 5.0),
    "alpha.lt.beta": (1.0, 5.0, 1.0, 10.0),
    "alpha.eq.beta": (0.5, 7.0, 2.0, 7.0),
    "small.shapes": (0.25, 4.0, 0.3, 15.0),
    "large.shapes": (5.0, 80.0, 4.0, 60.0),
}

#: S4.3's prediction horizon and discount rate, as the paper's case study sets
#: them. Both sides read these, so the horizon cannot drift between languages.
HORIZON_WEEKS = 52.0
CONTINUOUS_DISCOUNT_FACTOR = float(np.log(1 + 0.075) / 52)

COMMON = {"family": "pnbd_nocov", "prelude": "apparel", "inputs": GRID}


@pair(
    id="pnbd.nocov.LL_ind",
    spec="M-01",
    tol=1e-12,
    r='cpp("pnbd_nocov_LL_ind")(vLogparams = log(p), vX = x, vT_x = tx, vT_cal = Tc)',
    **COMMON,
)
def ll_ind(p):
    """S3.2 eq. 8, per customer."""
    return aggregate.log_likelihood_ind(X, TX, TC, *p)


@pair(
    id="pnbd.nocov.LL_sum",
    spec="M-01",
    tol=1e-10,
    # Negated at the call. The C++ entry point returns an objective for a
    # minimiser, not a log-likelihood, and the sign is applied on the R side so
    # that the Python function under test is the one users call.
    r='-cpp("pnbd_nocov_LL_sum")(log(p), x, tx, Tc, vN)',
    **COMMON,
)
def ll_sum(p):
    """The sample log-likelihood S3.2 maximises."""
    return aggregate.log_likelihood(X, TX, TC, *p)


@pair(
    id="pnbd.nocov.PAlive",
    spec="M-02",
    tol=1e-12,
    r='cpp("pnbd_nocov_PAlive")(r = p[1], alpha_0 = p[2], s = p[3], beta_0 = p[4],'
      " vX = x, vT_x = tx, vT_cal = Tc)",
    **COMMON,
)
def palive(p):
    """S4.1: P(alive at T | x, t_x, T)."""
    return aggregate.probability_alive(X, TX, TC, *p)


@pair(
    id="pnbd.nocov.CET",
    spec="M-02",
    tol=1e-11,
    # S4.2 divides by (s - 1). Two of the six grid points set s = 1 exactly,
    # and the two implementations part company there: CLVTools returns NaN,
    # this package raises. The pair records both, so the divergence is checked
    # rather than left out -- and so that a change to either side reports here.
    undefined_at=("alpha.gt.beta", "alpha.lt.beta"),
    r='cpp("pnbd_nocov_CET")(r = p[1], alpha_0 = p[2], s = p[3], beta_0 = p[4],'
      " dPeriods = 52, vX = x, vT_x = tx, vT_cal = Tc)",
    **COMMON,
)
def cet(p):
    """S4.2: expected transactions over the next 52 weeks."""
    return aggregate.conditional_expected_transactions(X, TX, TC, HORIZON_WEEKS, *p)


@pair(
    id="pnbd.nocov.DERT",
    spec="M-02",
    tol=1e-11,
    r='cpp("pnbd_nocov_DERT")(r = p[1], alpha_0 = p[2], s = p[3], beta_0 = p[4],'
      " continuous_discount_factor = log(1 + 0.075) / 52,"
      " vX = x, vT_x = tx, vT_cal = Tc)",
    **COMMON,
)
def dert(p):
    """S4.3: discounted expected residual transactions."""
    return aggregate.discounted_expected_residual_transactions(
        X, TX, TC, CONTINUOUS_DISCOUNT_FACTOR, *p
    )
