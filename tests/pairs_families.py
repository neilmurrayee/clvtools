"""The other three families without covariates: BG/NBD, GGompertz/NBD, Gamma-Gamma.

``tests/pairs_pnbd.py`` pairs the Pareto/NBD and ``tests/pairs_staticcov.py``
the covariate arms of all three latent-attrition families. This is the rest of
the no-covariate surface, so that every family's likelihood and every quantity
predicted from it is stated once, in both languages, at parameters chosen to
exercise it rather than to flatter it.

Each family is evaluated at its own fitted optimum -- read from the committed
fit fixtures to full precision, not to the four decimals the paper prints --
and at two points off it. The optima are worth looking at, because two of the
three are uncomfortable and that is the point of including them:

* The BG/NBD's ``a = 1.28``, ``b = 8.86`` is ordinary.
* The GGompertz/NBD's ``b = 8.1e-07`` and ``beta = 3.8e-05`` are not. This
  family's ``b`` runs to the floor of the space on apparel, and the numerical
  integration behind ``CET`` is worst-conditioned exactly there -- the same
  effect measured in ``pairs_staticcov.py`` and recorded in the README.
* The Gamma-Gamma is fitted on repeat transactions only, so its prelude has
  fewer customers than every other pair here. Pairing catches a mismatch there
  as a shape error rather than as a silently broadcast comparison.
"""

from __future__ import annotations

import numpy as np
from pairs import pair

from clvtools import bgnbd, gg, ggomnbd

HORIZON_WEEKS = 52.0
PMF_K = 3


def cbs(d):
    """``(x, t_x, T)`` from a latent-attrition prelude."""
    return d["x"], d["t.x"], d["T.cal"]


# -- BG/NBD -------------------------------------------------------------------

BGNBD_GRID = {
    #: From `bgnbd_fit.json`: the optimum `fit_bgnbd` reaches on apparel.
    "mle": (0.6073107652694043, 20.95673058721078, 1.2755434957498923,
            8.860815963388228),
    #: `a < 1`, where the beta function in the likelihood is at its most
    #: awkward and where the geometric dropout is heaviest.
    "small.a": (0.9, 12.0, 0.4, 3.0),
    #: Both shape parameters large, which is the flat, nearly-deterministic
    #: end of the mixing distribution.
    "large.shapes": (4.0, 60.0, 6.0, 25.0),
}

BGNBD = {"family": "bgnbd_nocov", "prelude": "apparel", "inputs": BGNBD_GRID}


@pair(
    id="bgnbd.nocov.LL_ind",
    spec="M-05",
    tol=1e-12,
    r='cpp("bgnbd_nocov_LL_ind")(vLogparams = log(p), vX = x, vT_x = tx, vT_cal = Tc)',
    **BGNBD,
)
def bgnbd_ll_ind(p, d):
    """S3.2's BG/NBD likelihood, per customer."""
    return bgnbd.log_likelihood_ind(*cbs(d), *p)


@pair(
    id="bgnbd.nocov.LL_sum",
    spec="M-05",
    tol=1e-10,
    r='-cpp("bgnbd_nocov_LL_sum")(log(p), x, tx, Tc, vN)',
    **BGNBD,
)
def bgnbd_ll_sum(p, d):
    """The sample log-likelihood, negated out of the minimiser's convention."""
    return bgnbd.log_likelihood(*cbs(d), *p)


@pair(
    id="bgnbd.nocov.PAlive",
    spec="M-05",
    tol=1e-12,
    r='cpp("bgnbd_nocov_PAlive")(r = p[1], alpha = p[2], a = p[3], b = p[4],'
      " vX = x, vT_x = tx, vT_cal = Tc)",
    **BGNBD,
)
def bgnbd_palive(p, d):
    """P(alive), which for the BG/NBD is exactly 1 for a customer with x = 0."""
    return bgnbd.probability_alive(*cbs(d), *p)


@pair(
    id="bgnbd.nocov.CET",
    spec="M-05",
    tol=1e-11,
    r='cpp("bgnbd_nocov_CET")(r = p[1], alpha = p[2], a = p[3], b = p[4],'
      " dPeriods = 52, vX = x, vT_x = tx, vT_cal = Tc)",
    **BGNBD,
)
def bgnbd_cet(p, d):
    """Expected transactions over the next 52 weeks."""
    return bgnbd.conditional_expected_transactions(*cbs(d), HORIZON_WEEKS, *p)


@pair(
    id="bgnbd.nocov.expectation",
    spec="M-05",
    tol=1e-11,
    r='cpp("bgnbd_nocov_expectation")(r = p[1], alpha = p[2], a = p[3], b = p[4],'
      " vT_i = rep(52, length(x)))",
    **BGNBD,
)
def bgnbd_expectation(p, d):
    """The unconditional expectation at 52 weeks."""
    return bgnbd.expectation(np.full(d["x"].shape, HORIZON_WEEKS), *p)


@pair(
    id="bgnbd.nocov.PMF",
    spec="PMF-01",
    tol=1e-11,
    r='cpp("bgnbd_nocov_PMF")(r = p[1], alpha = p[2], a = p[3], b = p[4],'
      " x = 3, vT_i = Tc)",
    **BGNBD,
)
def bgnbd_pmf(p, d):
    """P(X = 3) over each customer's own observation window."""
    return bgnbd.pmf(PMF_K, d["T.cal"], *p)


# -- GGompertz/NBD ------------------------------------------------------------

GGOMNBD_GRID = {
    #: From `ggomnbd_fit.json`. `b = 8.1e-07` is the real optimum, not a
    #: degenerate start: this family's `b` runs to the floor on apparel.
    "mle": (1.4490483109712962, 48.634490201568624, 8.127624052936936e-07,
            0.5599392548541099, 3.7970719927076935e-05),
    #: `b` at a value where the Gompertz hazard actually bends, so the
    #: quadrature behind CET is in its comfortable region.
    "b.moderate": (1.0, 30.0, 0.05, 1.2, 2.5),
    #: And `b` large, the fast-ageing end.
    "b.large": (2.0, 40.0, 1.5, 0.8, 6.0),
}

GGOMNBD = {"family": "ggomnbd_nocov", "prelude": "apparel", "inputs": GGOMNBD_GRID}


@pair(
    id="ggomnbd.nocov.LL_ind",
    spec="M-06",
    # 1e-6, and the bound is CLVTools' accuracy rather than ours. The
    # likelihood's second term integrates numerically on both sides; refining
    # the quadrature here moves nothing, and moves CLVTools a great deal.
    # Sweeping `b` with (r, alpha, s, beta) = (2, 40, 0.8, 6), each side
    # against this package's own integrand at `epsrel = 1e-14`:
    #
    #     b       ours      CLVTools      b       ours      CLVTools
    #     0.01    3.7e-16   3.7e-16       1.0     6.7e-16   3.4e-09
    #     0.1     4.3e-16   4.0e-12       1.5     4.0e-16   2.3e-07
    #     0.5     4.4e-16   2.7e-06       6.0     2.9e-16   7.6e-04
    #
    # The two agree to 4e-16 at small `b`, so the expression is the same one;
    # what parts company is the quadrature. `b.large` below stops at 1.5, and
    # `tests/test_families.py` pins the b = 6 end without needing R. See the
    # README's findings -- and note that `bT << 1` is the identified region
    # for this family, so nothing a fit reaches is affected.
    tol=1e-6,
    r='cpp("ggomnbd_nocov_LL_ind")(vLogparams = log(p), vX = x, vT_x = tx, vT_cal = Tc)',
    **GGOMNBD,
)
def ggomnbd_ll_ind(p, d):
    """Bemmaor & Glady's likelihood, per customer. Five parameters, not four."""
    return ggomnbd.log_likelihood_ind(*cbs(d), *p)


@pair(
    id="ggomnbd.nocov.LL_sum",
    spec="M-06",
    tol=1e-10,
    r='-cpp("ggomnbd_nocov_LL_sum")(log(p), x, tx, Tc, vN)',
    **GGOMNBD,
)
def ggomnbd_ll_sum(p, d):
    """The sample log-likelihood."""
    return ggomnbd.log_likelihood(*cbs(d), *p)


@pair(
    id="ggomnbd.nocov.PAlive",
    spec="M-07",
    tol=1e-11,
    r='cpp("ggomnbd_nocov_PAlive")(r = p[1], alpha_0 = p[2], b = p[3], s = p[4],'
      " beta_0 = p[5], vX = x, vT_x = tx, vT_cal = Tc)",
    **GGOMNBD,
)
def ggomnbd_palive(p, d):
    """P(alive) under the Gompertz lifetime."""
    return ggomnbd.probability_alive(*cbs(d), *p)


@pair(
    id="ggomnbd.nocov.CET",
    spec="M-09",
    # 1e-6, for the reason `pairs_staticcov.py` measures: both sides integrate
    # numerically and the integrand's conditioning degrades as `b` goes to
    # zero, which is where the `mle` point sits.
    tol=1e-6,
    r='cpp("ggomnbd_nocov_CET")(r = p[1], alpha_0 = p[2], b = p[3], s = p[4],'
      " beta_0 = p[5], dPeriods = 52, vX = x, vT_x = tx, vT_cal = Tc)",
    **GGOMNBD,
)
def ggomnbd_cet(p, d):
    """Expected transactions over the next 52 weeks."""
    return ggomnbd.conditional_expected_transactions(*cbs(d), HORIZON_WEEKS, *p)


@pair(
    id="ggomnbd.nocov.expectation",
    spec="M-06",
    tol=1e-9,
    r='cpp("ggomnbd_nocov_expectation")(r = p[1], alpha_0 = p[2], b = p[3],'
      " s = p[4], beta_0 = p[5], vT_i = rep(52, length(x)))",
    **GGOMNBD,
)
def ggomnbd_expectation(p, d):
    """The unconditional expectation at 52 weeks."""
    return ggomnbd.expectation(np.full(d["x"].shape, HORIZON_WEEKS), *p)


@pair(
    id="ggomnbd.nocov.PMF",
    spec="PMF-01",
    tol=1e-9,
    r='cpp("ggomnbd_nocov_PMF")(r = p[1], alpha_0 = p[2], b = p[3], s = p[4],'
      " beta_0 = p[5], x = 3, vT_i = Tc)",
    **GGOMNBD,
)
def ggomnbd_pmf(p, d):
    """P(X = 3) over each customer's own observation window."""
    return ggomnbd.pmf(PMF_K, d["T.cal"], *p)


# -- Gamma-Gamma --------------------------------------------------------------
#
# A different prelude, and a shorter one: the spending model is fitted on
# repeat transactions, so its CBS drops the customers who made none.

GG_GRID = {
    #: From `gg_fit.json`.
    "mle": (3.0990326501152476, 5.653672869349912, 56.504084430535976),
    #: `q` close to 1, where the mean of the mixing distribution is largest
    #: and the expected-spending expression is most sensitive.
    "small.q": (2.0, 1.2, 30.0),
    "large.shapes": (8.0, 12.0, 200.0),
}


@pair(
    id="gg.LL_sum",
    family="gg",
    prelude="apparel_spending",
    spec="M-10",
    tol=1e-10,
    inputs=GG_GRID,
    # Negated, like every other `_sum` here. `vM_x` is mean spending per
    # transaction, not total: reading it as a total is the mistake this pair
    # would catch, because the two differ by a factor of `x` per customer.
    r='-cpp("gg_LL")(log(p), x, mx, vN)',
)
def gg_ll_sum(p, d):
    """S3.5's Gamma-Gamma log-likelihood over the repeat-transaction sample."""
    return gg.log_likelihood(d["x"], d["m.x"], *p)
