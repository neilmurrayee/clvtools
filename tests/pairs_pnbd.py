"""The Pareto/NBD without covariates, paired expression by expression.

The first family onto the paired oracle, and the one the rest should follow.
Seven expressions -- S3.2's likelihood in both its per-customer and summed
forms, and the five quantities S4 derives from it -- at six parameter vectors.

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
from pairs import pair

from clvtools.pnbd import aggregate


def cbs(d):
    """``(x, t_x, T)`` from the prelude's data.

    Every pair below is handed ``d`` rather than reading a fixture of its own,
    so the vectors these functions see are literally the ones live mode
    compares against R. ``tests/test_pairs.py`` also holds that recording to
    ``cbs_estimation.csv``, so the paired oracle and the 247 fixture-based
    checks describe the same 600 customers.
    """
    return d["x"], d["t.x"], d["T.cal"]

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
def ll_ind(p, d):
    """S3.2 eq. 8, per customer."""
    return aggregate.log_likelihood_ind(*cbs(d), *p)


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
def ll_sum(p, d):
    """The sample log-likelihood S3.2 maximises."""
    return aggregate.log_likelihood(*cbs(d), *p)


@pair(
    id="pnbd.nocov.PAlive",
    spec="M-02",
    tol=1e-12,
    r='cpp("pnbd_nocov_PAlive")(r = p[1], alpha_0 = p[2], s = p[3], beta_0 = p[4],'
      " vX = x, vT_x = tx, vT_cal = Tc)",
    **COMMON,
)
def palive(p, d):
    """S4.1: P(alive at T | x, t_x, T)."""
    return aggregate.probability_alive(*cbs(d), *p)


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
def cet(p, d):
    """S4.2: expected transactions over the next 52 weeks."""
    return aggregate.conditional_expected_transactions(*cbs(d), HORIZON_WEEKS, *p)


@pair(
    id="pnbd.nocov.DERT",
    spec="M-02",
    tol=1e-11,
    r='cpp("pnbd_nocov_DERT")(r = p[1], alpha_0 = p[2], s = p[3], beta_0 = p[4],'
      " continuous_discount_factor = log(1 + 0.075) / 52,"
      " vX = x, vT_x = tx, vT_cal = Tc)",
    **COMMON,
)
def dert(p, d):
    """S4.3: discounted expected residual transactions."""
    return aggregate.discounted_expected_residual_transactions(
        *cbs(d), CONTINUOUS_DISCOUNT_FACTOR, *p
    )


@pair(
    id="pnbd.nocov.expectation",
    spec="M-03",
    tol=1e-12,
    # Read the argument order twice. `pnbd_nocov_expectation` takes
    # (r, s, alpha_0, beta_0) -- s and alpha transposed relative to every one
    # of its siblings, which take (r, alpha_0, s, beta_0). CLAUDE.md lists this
    # among the traps that have already cost time, and it is the single
    # clearest argument for pairing: the R call and the Python call sit four
    # lines apart, so the transposition is visible instead of remembered.
    undefined_at=("alpha.gt.beta", "alpha.lt.beta"),
    r='cpp("pnbd_nocov_expectation")(r = p[1], s = p[3], alpha_0 = p[2],'
      " beta_0 = p[4], vT_i = rep(52, length(x)))",
    **COMMON,
)
def expectation(p, d):
    """S4.4's unconditional expectation at 52 weeks."""
    return aggregate.expectation(np.full(d["x"].shape, HORIZON_WEEKS), *p)


@pair(
    id="pnbd.nocov.PMF",
    spec="PMF-01",
    tol=1e-12,
    r='cpp("pnbd_nocov_PMF")(r = p[1], alpha_0 = p[2], s = p[3], beta_0 = p[4],'
      " x = 3, vT_i = Tc)",
    **COMMON,
)
def pmf(p, d):
    """P(X = 3) over each customer's own observation window."""
    return aggregate.pmf(3, d["T.cal"], *p)
