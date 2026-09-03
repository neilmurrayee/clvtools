r"""S3.3 and S6.4.2 - the Pareto/NBD with time-varying covariates.

S3.3: "in many real-world applications, customer transaction and attrition
behavior may be influenced by covariates that *vary over time*. Consequently,
the timing of a purchase and the corresponding value of the covariate at that
time become relevant."

The rates become functions of time,

.. math::
    \lambda(t) = \lambda_0 \exp(\boldsymbol{\gamma}_{purch}'\mathbf{x}^P(t)),
    \qquad
    \mu(t) = \mu_0 \exp(\boldsymbol{\gamma}_{attr}'\mathbf{x}^A(t)),

with :math:`\mathbf{x}^P(t)` and :math:`\mathbf{x}^A(t)` piecewise constant on
equidistant intervals -- S3.3: "these functions are parameterized by assuming
equidistant time intervals (e.g., one week) during which the covariates are
assumed to be constant."

Two consequences run through everything here. The number of transactions in
:math:`(s_1, s_2]` depends on :math:`\Lambda(s_1,s_2) = \int \lambda(s)ds`
rather than on :math:`\lambda \cdot (s_2-s_1)`, so integrals become sums over
covariate intervals; and, as S3.3 notes, "the times between purchases are no
longer independently and identically distributed", so the sufficient statistic
is no longer :math:`(x, t_x, T)` -- "this extension considers the exact date of
all transactions".

Walks
-----
The bookkeeping that makes this tractable is the *walk*: for a span of time, the
sequence of :math:`\exp(\boldsymbol{\gamma}'\mathbf{x}_k)` values over the
covariate intervals it crosses, together with ``d1``, the distance from its
start to the end of the first interval it touches, and ``tjk``, its total
length. Each customer has four kinds of walk, built once and reused across every
evaluation of a fit. :mod:`clvtools.pnbd.dyncov_walks` defines them and
constructs them from the transaction log and the covariate table; everything
below takes them as given, which is what keeps that dependency one-way.
:class:`~clvtools.pnbd.dyncov_walks.Walk`,
:class:`~clvtools.pnbd.dyncov_walks.TransactionWalk`,
:class:`~clvtools.pnbd.dyncov_walks.DyncovWalks` and
:func:`~clvtools.pnbd.dyncov_walks.build_walks` are re-exported here, so this
module remains the one import a caller of the time-varying model needs.

The likelihood
--------------
S3.3 says only that the closed form "contains a sum of expressions similar to
the likelihood of the standard model, each being calculated for a certain time
interval", pointing to Bachmann et al. (2021) Appendix A.1 for the derivation.
This module follows CLVTools' own arrangement, which is

.. math::
    \log L = \log F_0 + \log(F_1 F_2 + F_3)

with :math:`F_2` the sum over covariate intervals of Gaussian-hypergeometric
terms, and each of :math:`F_0, F_1, F_3` a product that logs cleanly. The
building blocks -- ``A1sum``, ``Bjsum``, ``Bksum``, ``B_i``, ``D_i`` and the
rest -- are each a separate function here, and each is held to CLVTools'
per-customer output in ``tests/test_pnbd_dyncov.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import special

from clvtools._validate import finished, start_values
from clvtools.inference import Fitted, numerical_hessian

# The walk primitives moved out when this module outgrew the size gate; they
# are imported for real, and re-exported below, so that no caller of the
# time-varying model has to know that happened. `EMPTY_WALK` and `build_walks`
# are unused here and kept alive by `__all__` alone.
from clvtools.pnbd.dyncov_walks import (
    EMPTY_WALK,
    Customer,
    DyncovWalks,
    TransactionWalk,
    Walk,
    build_walks,
)

__all__ = [
    "EMPTY_WALK",
    "DyncovWalks",
    "PnbdDynCovParams",
    "TransactionWalk",
    "Walk",
    "a1sum",
    "b_i",
    "bjsum",
    "bksum",
    "build_walks",
    "d_i",
    "fit_pnbd_dyncov",
    "log_likelihood",
    "log_likelihood_ind",
    "probability_alive",
    "walk_integral",
]

#: Names of the thirty quantities CLVTools returns per customer, in its order.
INTERMEDIATE_NAMES = [
    "LL",
    "A1T", "AkT", "A1sum", "B1", "BT", "Bjsum", "Bksum",
    "C1T", "CkT", "D1", "DT", "DkT",
    "log_F0", "log_F1", "F2", "log_F3",
    "Akprod",
    "dT", "a1T", "b1T", "a1", "b1",
    "akt", "bkT", "aT", "bT",
    "F2.1", "F2.2", "F2.3",
]


# -- the transaction process --------------------------------------------------


def a1sum(real_walks: list[TransactionWalk]) -> float:
    r"""Log of the covariate active at each repeat transaction.

    Each real walk runs *to* a transaction, so its last element is the
    multiplier in force when that transaction happened. A zero-repeater has no
    real walks and contributes :math:`\log 1 = 0`.

    >>> a1sum([])
    0.0
    >>> import numpy as np
    >>> w = TransactionWalk(np.array([1.0, 2.0]), d1=0.5, tjk=1.5)
    >>> round(a1sum([w, w]), 6)
    1.386294
    """
    return float(sum(np.log(w.last) for w in real_walks))


def walk_integral(w: TransactionWalk) -> float:
    r"""The integrated rate multiplier over a walk, :math:`\int \exp(\gamma'x)`.

    The first covariate interval contributes only its tail ``d1``, the last only
    the remainder, and each whole interval between contributes 1.

    >>> import numpy as np
    >>> # A walk entirely inside one covariate interval: value times length.
    >>> round(walk_integral(TransactionWalk(np.array([2.0]), d1=0.3, tjk=0.4)), 6)
    0.8
    >>> # Two intervals: d1 of the first, the rest at the second.
    >>> round(walk_integral(TransactionWalk(np.array([2.0, 3.0]), d1=0.25, tjk=1.0)), 6)
    2.75
    """
    n = w.n_elem
    if n == 1:
        # Both ends fall in the same covariate period.
        return w.first * w.tjk
    if n == 2:
        return w.first * w.d1 + w.last * (w.tjk - w.d1)
    return (
        w.first * w.d1
        + w.sum_middle()
        + w.last * (w.tjk - w.d1 - (n - 2.0))
    )


def bjsum(real_walks: list[TransactionWalk]) -> float:
    """The integrated transaction rate over all repeat-purchase intervals."""
    return float(sum(walk_integral(w) for w in real_walks))


def bksum(bjsum_value: float, aux_walk: TransactionWalk) -> float:
    """``bjsum`` extended to the end of the estimation period."""
    return bjsum_value + walk_integral(aux_walk)


def b_i(i: int, t_x: float, aux_walk: TransactionWalk) -> float:
    r"""The transaction-rate integral over the auxiliary walk up to interval ``i``.

    ``i`` counts covariate intervals from 1, not array positions.
    """
    if i == 1:
        return aux_walk.first * (-t_x)
    if i == 2:
        return aux_walk.first * aux_walk.d1 + aux_walk.elem(1) * (-t_x - aux_walk.d1)
    return (
        aux_walk.first * aux_walk.d1
        + aux_walk.sum_from_to(1, i - 2)
        + aux_walk.elem(i - 1) * (-t_x - aux_walk.d1 - (i - 2.0))
    )


# -- the attrition process ----------------------------------------------------


def _real_life_sum(real_walk: Walk, d_omega: float) -> float:
    r"""The real lifetime walk's contribution to :math:`D_i`, the same for every ``i``.

    Split out because :func:`d_i` recomputes it per interval while
    :func:`_f2_middle` needs it once for the whole sweep; both must get the
    same number in the same order of operations.
    """
    if real_walk.n_elem == 1:
        return real_walk.first * d_omega
    if real_walk.n_elem == 2:
        return real_walk.first * d_omega + real_walk.last
    return real_walk.first * d_omega + real_walk.sum_middle() + real_walk.last


def d_i(i: int, real_walk: Walk, aux_walk: Walk, d_omega: float) -> float:
    r"""The attrition-rate integral from coming alive to interval ``i``.

    The real and auxiliary lifetime walks are treated as one continuous span:
    the covariate interval containing the last transaction belongs to the
    auxiliary walk alone, so the two never overlap. ``d_omega`` is the distance
    from the customer coming alive to the end of the interval they came alive
    in.
    """
    if real_walk.n_elem == 0:
        if i == 1:
            # First and last element coincide and cancel.
            return 0.0
        if i == 2:
            return aux_walk.first * d_omega + aux_walk.elem(1) * (-d_omega)
        last_mult = -d_omega - (1.0 + i - 3.0)
        return (
            aux_walk.first * d_omega
            + aux_walk.sum_from_to(1, i - 2)
            + aux_walk.elem(i - 1) * last_mult
        )

    sum_real = _real_life_sum(real_walk, d_omega)

    # +1 to count the interval containing the last transaction, which opens the
    # auxiliary walk.
    k0x = real_walk.n_elem + 1.0
    last_mult = -d_omega - (k0x + i - 3.0)

    if i == 1:
        sum_aux = aux_walk.first * last_mult
    elif i == 2:
        sum_aux = aux_walk.first + aux_walk.elem(1) * last_mult
    else:
        sum_aux = (
            aux_walk.first
            + aux_walk.sum_from_to(1, i - 2)
            + aux_walk.elem(i - 1) * last_mult
        )

    return sum_real + sum_aux


# -- log-space arithmetic -----------------------------------------------------


def _log_diff_exp(
    log_a: NDArray[np.float64], log_b: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r""":math:`\log|e^{a} - e^{b}|` and the sign of the difference, elementwise.

    Written with :func:`numpy.expm1` rather than as
    :math:`\log(e^{a} - e^{b})`, so that the near-cancellation :math:`F_{2.2}`
    is made of stays accurate: subtracting the smaller term inside the exponent
    leaves the larger one's magnitude intact, where subtracting the two
    exponentials loses every digit they share.

    Two terms equal to the last bit give :math:`-\infty` and a sign of zero --
    an exact zero, which is what :math:`F_{2.2}` is for 599 of the 600 apparel
    customers, and is not the same thing as a term that underflowed.

    >>> import numpy as np
    >>> magnitude, sign = _log_diff_exp(np.log(5.0), np.log(3.0))
    >>> round(float(np.exp(magnitude)), 12), float(sign)
    (2.0, 1.0)
    >>> magnitude, sign = _log_diff_exp(np.float64(-800.0), np.float64(-800.0))
    >>> float(magnitude), float(sign)
    (-inf, 0.0)
    """
    hi = np.maximum(log_a, log_b)
    lo = np.minimum(log_a, log_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = hi + np.log(-np.expm1(lo - hi))
        # Both terms zero: their difference is zero, not the ``nan`` that
        # ``-inf - (-inf)`` produces inside the ``expm1``.
        zero = hi == -np.inf
        return (
            np.where(zero, -np.inf, magnitude),
            np.where(zero, 0.0, np.sign(log_a - log_b)),
        )


def _signed_logsumexp(
    logs: NDArray[np.float64], signs: NDArray[np.float64]
) -> tuple[float, float]:
    r""":math:`\log|\sum_i s_i e^{l_i}|` and the sum's sign.

    The offset is the largest term's own log, so every term is weighed against
    the one that dominates it. Backlog item 28 records a cheaper version that
    does not work: scaling a customer's whole :math:`F_2` by the *first* term's
    :math:`(r{+}s{+}x)\log\alpha_1` keeps the sum :math:`O(1)` and looks
    equivalent, but the terms have different :math:`\alpha`, and customer 93 of
    the apparel cohort -- whose :math:`F_2` is 5.6e-165, comfortably
    representable -- moved by 2.4e-3 in log-likelihood under it.

    A term that is not usable (``nan``, or an overflowed :math:`+\infty`) is
    returned as it stands rather than mixed in, so that it reaches
    :func:`log_likelihood_customer` and becomes a ``nan`` likelihood there.

    >>> import numpy as np
    >>> magnitude, sign = _signed_logsumexp(
    ...     np.array([-800.0, -801.0]), np.array([1.0, -1.0]))
    >>> bool(magnitude > -801.0), sign
    (True, 1.0)
    """
    logs = np.where(signs == 0.0, -np.inf, logs)
    offset = float(np.max(logs))
    if not np.isfinite(offset):
        return offset, 0.0 if offset == -np.inf else 1.0
    total = float(np.sum(signs * np.exp(logs - offset)))
    with np.errstate(divide="ignore"):
        return offset + float(np.log(abs(total))), float(np.sign(total))


def _usable(log_magnitude: float) -> bool:
    r"""Whether a log magnitude can still be summed against another.

    :math:`-\infty` can: it is an exact zero. ``nan`` and :math:`+\infty`
    cannot, and stop the customer's likelihood rather than being carried into
    it.

    >>> [_usable(v) for v in (0.0, float("-inf"), float("inf"), float("nan"))]
    [True, True, False, False]
    """
    return log_magnitude < np.inf


def _value(log_magnitude: float, sign: float) -> float:
    """A signed log back in the direct domain, for the intermediates table.

    The thirty columns CLVTools reports are values, and are compared as values
    against ``tests/fixtures/``; only the likelihood itself is now formed from
    the logs. A term below float64 therefore still *reports* as zero -- so does
    CLVTools' own -- while no longer being zero where it counts.

    >>> _value(float("-inf"), 0.0), round(_value(0.0, -1.0), 12)
    (0.0, -1.0)
    """
    return float(sign * np.exp(log_magnitude))


# -- the hypergeometric pair --------------------------------------------------


def _hyp_alpha_ge_beta(
    r: float, s: float, x: float,
    alpha_1: ArrayLike, beta_1: ArrayLike, alpha_2: ArrayLike, beta_2: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The :math:`\alpha \ge \beta` arm of each :math:`F_2` term.

    .. math::
        \frac{{}_2F_1(r{+}s{+}x,\, s{+}1,\, r{+}s{+}x{+}1;\, z_1)}
             {\alpha_1^{r+s+x}}
        - \frac{{}_2F_1(\cdots;\, z_2)}{\alpha_2^{r+s+x}},
        \qquad z_j = 1 - \beta_j/\alpha_j

    Returned as ``(log magnitude, sign)`` rather than as a value. Both
    quotients are formed in log space, and so is their difference: at
    :math:`\alpha = 200` and :math:`x = 200` the term is around 1e-370 --
    below float64, but perfectly well determined, and the customer whose
    :math:`F_1F_2` it is has an :math:`F_3` of the same order. Reporting it as
    zero cost that customer 225 log-units; see
    ``tests/test_pnbd_dyncov_logspace.py``.

    Where the series will not converge, CLVTools substitutes the limiting form
    :math:`(1-z)^{r+x} C / \beta^{r+s+x}`; the same fallback is used here so the
    two agree everywhere, including where neither is accurate.

    ``alpha_1`` and its siblings may each be a float or an array of one value
    per covariate interval; :func:`_hyp_terms` passes whole batches through.
    """
    # numpy arithmetic throughout, even for a single term: `np.where` selects
    # the fallback elementwise, and the arms hand `_log_diff_exp` arrays so
    # that its own `np.where` sees one.
    alpha_1, beta_1 = np.asarray(alpha_1, float), np.asarray(beta_1, float)
    alpha_2, beta_2 = np.asarray(alpha_2, float), np.asarray(beta_2, float)
    a = r + s + x
    logs = []
    for alpha, beta in ((alpha_1, beta_1), (alpha_2, beta_2)):
        z = 1.0 - beta / alpha
        value = special.hyp2f1(a, s + 1.0, a + 1.0, z)
        with np.errstate(divide="ignore"):
            log_term = np.log(value) - a * np.log(alpha)
        failed = ~np.isfinite(value)
        if np.any(failed):
            # Computed here rather than up front: the fallback fires rarely,
            # and four gammaln calls on every term is most of this function.
            log_c = (
                special.gammaln(a + 1.0) + special.gammaln(s)
                - special.gammaln(a) - special.gammaln(s + 1.0)
            )
            with np.errstate(divide="ignore"):
                log_term = np.where(
                    failed,
                    (r + x) * np.log1p(-z) + log_c - a * np.log(beta),
                    log_term,
                )
        logs.append(log_term)
    return _log_diff_exp(logs[0], logs[1])


def _hyp_beta_gt_alpha(
    r: float, s: float, x: float,
    alpha_1: ArrayLike, beta_1: ArrayLike, alpha_2: ArrayLike, beta_2: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The :math:`\beta > \alpha` arm, with the roles exchanged.

    .. math::
        \frac{{}_2F_1(r{+}s{+}x,\, r{+}x,\, r{+}s{+}x{+}1;\, z_1)}
             {\beta_1^{r+s+x}} - \cdots,
        \qquad z_j = 1 - \alpha_j/\beta_j

    ``(log magnitude, sign)``, and vectorised over covariate intervals, in the
    same way as its sibling. The fallback's :math:`(1-z)^{s+1}C/\alpha^{a}` is
    formed in log space too, where it used to be a product of three factors of
    which the last overflows first.
    """
    alpha_1, beta_1 = np.asarray(alpha_1, float), np.asarray(beta_1, float)
    alpha_2, beta_2 = np.asarray(alpha_2, float), np.asarray(beta_2, float)
    a = r + s + x
    logs = []
    for alpha, beta in ((alpha_1, beta_1), (alpha_2, beta_2)):
        z = 1.0 - alpha / beta
        value = special.hyp2f1(a, r + x, a + 1.0, z)
        with np.errstate(divide="ignore"):
            log_term = np.log(value) - a * np.log(beta)
        failed = ~np.isfinite(value)
        if np.any(failed):
            log_c = (
                special.gammaln(a + 1.0) + special.gammaln(r + x - 1.0)
                - special.gammaln(a) - special.gammaln(r + x)
            )
            with np.errstate(divide="ignore"):
                log_term = np.where(
                    failed,
                    (s + 1.0) * np.log1p(-z) + log_c - a * np.log(alpha),
                    log_term,
                )
        logs.append(log_term)
    return _log_diff_exp(logs[0], logs[1])


def _scale_by_ratio(
    log_magnitude: NDArray[np.float64], sign: NDArray[np.float64],
    s: float, ratio: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Multiply a signed log by :math:`(A/C)^s`, which is positive.

    An exactly zero term stays exactly zero whatever the ratio is, which the
    addition alone would not give: :math:`-\infty + \infty` is ``nan``, and the
    ratio does overflow at the covariate parameters
    ``test_a_non_finite_f2_gives_a_non_finite_likelihood`` uses.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = log_magnitude + s * np.log(ratio)
    return np.where(sign == 0.0, -np.inf, scaled), sign


def _hyp_term(
    r: float, s: float, x: float,
    alpha_1: float, beta_1: float, alpha_2: float, beta_2: float,
    ratio: float,
) -> tuple[float, float]:
    """One :math:`F_2` term, as ``(log magnitude, sign)``."""
    branch = _hyp_alpha_ge_beta if alpha_1 >= beta_1 else _hyp_beta_gt_alpha
    log_magnitude, sign = _scale_by_ratio(
        *branch(r, s, x, alpha_1, beta_1, alpha_2, beta_2), s, ratio
    )
    return float(log_magnitude), float(sign)


def _hyp_terms(
    r: float, s: float, x: float,
    alpha_1: NDArray[np.float64], beta_1: NDArray[np.float64],
    alpha_2: NDArray[np.float64], beta_2: NDArray[np.float64],
    ratio: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r""":func:`_hyp_term` for a whole batch of covariate intervals at once.

    The arm is chosen per interval, not per batch: :math:`\alpha_1 \ge \beta_1`
    holds for some intervals of a customer and not others, and near the optimum
    both happen. Each arm is therefore handed the sub-batch that selected it --
    possibly an empty one, which the arms handle without a special case, so
    neither is skipped by a branch that a test could fail to reach.
    """
    logs = np.empty(np.shape(alpha_1))
    signs = np.empty(np.shape(alpha_1))
    ge = alpha_1 >= beta_1
    for chosen, branch in ((ge, _hyp_alpha_ge_beta), (~ge, _hyp_beta_gt_alpha)):
        logs[chosen], signs[chosen] = branch(
            r, s, x,
            alpha_1[chosen], beta_1[chosen], alpha_2[chosen], beta_2[chosen],
        )
    return _scale_by_ratio(logs, signs, s, ratio)


def _prefix_sums(values: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""``sum_from_to(1, i-2)`` for every :math:`i` from 2 to ``len(values) - 1``.

    :func:`b_i` and :func:`d_i` each re-sum a growing prefix of the walk for
    every covariate interval, which is quadratic in the length of the walk and
    is where a fifth of this package's calls went. These are the same prefixes,
    accumulated once.

    ``np.cumsum`` adds strictly left to right; ``ndarray.sum``, which
    :meth:`~clvtools.pnbd.dyncov_walks.Walk.sum_from_to` calls, adds pairwise.
    On a walk of more than eight intervals the two therefore agree to a
    rounding unit rather than bit for bit -- see ``docs/performance.md``. The
    scalar path is left in place for :math:`B_1, B_T, D_1, D_T`, which is what
    keeps ``F2.2``'s exact cancellation exact.

    >>> import numpy as np
    >>> _prefix_sums(np.array([1.0, 2.0, 4.0, 8.0, 16.0]))
    array([0., 2., 6.])
    """
    return np.concatenate(([0.0], np.cumsum(values[1:-2])))


def _f2_middle(
    r: float, alpha_0: float, s: float, beta_0: float,
    c: Customer, dT: float, Bjsum: float,
) -> tuple[float, float]:
    r""":math:`\sum_{i=2}^{k_T-1} Y_i` -- every covariate interval in between, at once.

    This is the same sum the scalar :func:`b_i`, :func:`d_i` and
    :func:`_hyp_term` used to build one interval at a time. There are about 66
    of those per customer, and removing the loop took an evaluation on the
    apparel cohort from 0.328 s to 0.097 s -- so roughly 70% of the cost of the
    time-varying likelihood was here. The same 39,754 hypergeometrics are still
    evaluated; they arrive in four ``scipy.special.hyp2f1`` calls per customer
    rather than two per interval.
    """
    aux_trans, aux_life = c.aux_walk_trans, c.aux_walk_life
    # `Walk.values` is the numpy array of covariate multipliers; the pandas
    # rule that fires on the name does not apply.
    A, C = aux_trans.values, aux_life.values  # noqa: PD011
    n = A.size
    if n < 3:
        # The first interval is also the last; there is nothing in between.
        return -np.inf, 0.0

    i = np.arange(2.0, n)
    Ai, Ci = A[1 : n - 1], C[1 : n - 1]
    elapsed = c.t_x + dT + (i - 2.0)

    # B_i, in the order `b_i` writes it.
    Bi = (
        A[0] * aux_trans.d1
        + _prefix_sums(A)
        + Ai * (-c.t_x - aux_trans.d1 - (i - 2.0))
    )
    ai = Bjsum + Bi + Ai * elapsed

    # D_i, likewise. Which of `d_i`'s two forms applies is a property of the
    # customer rather than of the interval, so the choice is made once.
    if c.real_walk_life.n_elem == 0:
        Di = (
            C[0] * c.d_omega
            + _prefix_sums(C)
            + Ci * (-c.d_omega - (1.0 + i - 3.0))
        )
    else:
        k0x = c.real_walk_life.n_elem + 1.0
        Di = _real_life_sum(c.real_walk_life, c.d_omega) + (
            C[0] + _prefix_sums(C) + Ci * (-c.d_omega - (k0x + i - 3.0))
        )
    bi = Di + Ci * elapsed

    logs, signs = _hyp_terms(
        r, s, c.x,
        ai + alpha_0, (bi + beta_0) * Ai / Ci,
        ai + Ai + alpha_0, (bi + Ci + beta_0) * Ai / Ci,
        Ai / Ci,
    )
    # Summed against the largest term rather than left to right. The scalar
    # loop this replaced stopped at the first running total that stopped being
    # finite; `_signed_logsumexp` propagates an unusable term to the whole sum
    # instead, which reaches `log_likelihood_customer` the same way.
    return _signed_logsumexp(logs, signs)


# The arguments are the terms of F2 itself, in the paper's notation. They are
# computed together in `log_likelihood_customer` and consumed only here; a
# wrapper object would add an allocation per customer per likelihood
# evaluation, on the path this module is already tuned around.


def _f2(  # noqa: PLR0913, PLR0917
    r: float, alpha_0: float, s: float, beta_0: float,
    c: Customer,
    B1: float, D1: float, BT: float, DT: float,
    A1T: float, C1T: float, AkT: float, CkT: float, Bjsum: float,
) -> tuple[float, float, dict[str, float]]:
    r""":math:`F_2 = Y_1 + Y_{k_T} + \sum_{i=2}^{k_T-1} Y_i`, as a signed log.

    One term per covariate interval the auxiliary walk crosses: the first, the
    last, and a sum over those between. S3.3: the likelihood "contains a sum of
    expressions similar to the likelihood of the standard model, each being
    calculated for a certain time interval".

    Returns ``(log|F_2|, sign, intermediates)``. The three terms are combined
    with a signed log-sum-exp rather than added as values, so that a customer
    whose :math:`F_2` is below float64 -- which is a matter of how many
    transactions they made, not of anything going wrong -- still contributes
    :math:`F_1F_2` to their likelihood. The ``intermediates`` keep the value
    form, because that is what the oracle fixtures hold.
    """
    dT = c.aux_walk_trans.d1

    a1T = Bjsum + B1 + c.T_cal * A1T
    b1T = D1 + c.T_cal * C1T
    a1 = Bjsum + B1 + A1T * (c.t_x + dT - 1.0)
    b1 = D1 + C1T * (c.t_x + dT - 1.0)

    parts = {"dT": dT, "a1T": a1T, "b1T": b1T, "a1": a1, "b1": b1}

    if c.aux_walk_life.n_elem == 1:
        # A single covariate interval: no first/last split and no middle sum.
        # The one apparel customer here, 262, is also the degenerate case --
        # they bought on the last day of the estimation period, so the walk has
        # no length, both hypergeometrics take identical arguments, and the
        # sign is zero. That is an exact zero F2 and the alive-only likelihood
        # for the right reason; an underflowed one is a large negative log.
        log_f2, sign = _hyp_term(
            r, s, c.x,
            a1 + (1.0 - dT) * A1T + alpha_0,
            (b1 + (1.0 - dT) * C1T + beta_0) * A1T / C1T,
            a1T + alpha_0,
            (b1T + beta_0) * A1T / C1T,
            A1T / C1T,
        )
        parts.update(dict.fromkeys(
            ("akt", "bkT", "aT", "bT", "F2.1", "F2.2", "F2.3"), float("nan")
        ))
        return log_f2, sign, parts

    n_walks = float(c.aux_walk_life.n_elem)
    akt = Bjsum + BT + AkT * (c.t_x + dT + n_walks - 2.0)
    bkT = DT + CkT * (c.t_x + dT + n_walks - 2.0)
    aT = Bjsum + BT + c.T_cal * AkT
    bT = DT + c.T_cal * CkT
    parts.update({"akt": akt, "bkT": bkT, "aT": aT, "bT": bT})

    # Y_1 -- the first covariate interval after the last transaction.
    log_1, sign_1 = _hyp_term(
        r, s, c.x,
        a1 + (1.0 - dT) * A1T + alpha_0,
        (b1 + (1.0 - dT) * C1T + beta_0) * A1T / C1T,
        a1 + A1T + alpha_0,
        (b1 + C1T + beta_0) * A1T / C1T,
        A1T / C1T,
    )
    parts["F2.1"] = _value(log_1, sign_1)
    # `_usable` and not `np.isfinite`: an exactly zero term is -inf here, and
    # is a term like any other. Only `nan` and an overflowed +inf stop the sum.
    if not _usable(log_1):
        parts.update({"F2.2": float("nan"), "F2.3": float("nan")})
        return log_1, sign_1, parts

    # Y_kT -- the last interval, running to the end of the estimation period.
    log_2, sign_2 = _hyp_term(
        r, s, c.x,
        akt + alpha_0, (bkT + beta_0) * AkT / CkT,
        aT + alpha_0, (bT + beta_0) * AkT / CkT,
        AkT / CkT,
    )
    parts["F2.2"] = _value(log_2, sign_2)
    if not _usable(log_2):
        parts["F2.3"] = float("nan")
        return log_2, sign_2, parts

    # The intervals in between.
    log_3, sign_3 = _f2_middle(r, alpha_0, s, beta_0, c, dT, Bjsum)

    parts["F2.3"] = _value(log_3, sign_3)
    log_f2, sign = _signed_logsumexp(
        np.array([log_1, log_2, log_3]), np.array([sign_1, sign_2, sign_3])
    )
    return log_f2, sign, parts


def log_likelihood_customer(
    r: float, alpha_0: float, s: float, beta_0: float, c: Customer
) -> dict[str, float]:
    r"""One customer's log-likelihood, with every intermediate quantity.

    .. math::
        \log L = \log F_0 + \log(F_1 F_2 + F_3)

    with

    .. math::
        \log F_0 &= r\log\alpha_0 + s\log\beta_0
                    + \ln\Gamma(x{+}r) - \ln\Gamma(r) + \mathrm{A1sum} \\
        \log F_1 &= \log s - \log(r{+}s{+}x) \\
        \log F_3 &= -s\log(D_{k_T}{+}\beta_0) - (x{+}r)\log(\mathrm{Bksum}{+}\alpha_0)

    :math:`F_2` may be negative -- the difference of two hypergeometrics is not
    sign-definite -- while :math:`F_1 F_2 + F_3` must stay positive for the
    likelihood to mean anything. The three sign cases are handled separately
    rather than by taking a log that may not exist.

    :math:`F_2` arrives as a log magnitude and a sign, and the sum is formed
    from logs throughout. That matters for a heavy buyer: at :math:`x = 200`
    both :math:`F_1F_2` and :math:`F_3` are around 1e-274, so both underflow
    while their *ratio* is O(1). Adding them as values gave :math:`F_2 = 0` and
    the ``sign == 0`` branch below -- the alive-only likelihood, silently, and
    wrong by 225 log-units. Finding 10 of ``docs/review-2026-09-02.md``.
    """
    A1T = c.aux_walk_trans.first
    AkT = c.aux_walk_trans.last
    A1sum_v = a1sum(c.real_walks_trans)

    B1 = b_i(1, c.t_x, c.aux_walk_trans)
    BT = b_i(c.aux_walk_trans.n_elem, c.t_x, c.aux_walk_trans)
    Bjsum_v = bjsum(c.real_walks_trans)
    Bksum_v = bksum(Bjsum_v, c.aux_walk_trans)

    C1T = c.aux_walk_life.first
    CkT = c.aux_walk_life.last
    D1 = d_i(1, c.real_walk_life, c.aux_walk_life, c.d_omega)
    DT = d_i(c.aux_walk_life.n_elem, c.real_walk_life, c.aux_walk_life, c.d_omega)
    DkT = CkT * c.T_cal + DT

    log_F0 = (
        r * np.log(alpha_0) + s * np.log(beta_0)
        + special.gammaln(c.x + r) - special.gammaln(r) + A1sum_v
    )
    log_F1 = np.log(s) - np.log(r + s + c.x)
    log_F3 = -s * np.log(DkT + beta_0) - (c.x + r) * np.log(Bksum_v + alpha_0)

    log_F2, sign_F2, parts = _f2(
        r, alpha_0, s, beta_0, c,
        B1, D1, BT, DT, A1T, C1T, AkT, CkT, Bjsum_v,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if not _usable(log_F2):
            ll = float("nan")
        elif sign_F2 < 0.0:
            # F1*F2 is negative but smaller in magnitude than F3, so the sum is
            # still positive; log1p keeps the near-cancellation accurate.
            ll = log_F0 + log_F3 + np.log1p(-np.exp(log_F1 + log_F2 - log_F3))
        elif sign_F2 > 0.0:
            ll = log_F0 + np.logaddexp(log_F1 + log_F2, log_F3)
        else:
            # F2 is exactly zero -- an auxiliary walk of no length, whose two
            # hypergeometrics cancel term for term. Not an underflow: that now
            # arrives as a large negative log and takes the branch above.
            ll = log_F0 + log_F3

    return {
        "LL": float(ll),
        "A1T": A1T, "AkT": AkT, "A1sum": A1sum_v,
        "B1": B1, "BT": BT, "Bjsum": Bjsum_v, "Bksum": Bksum_v,
        "C1T": C1T, "CkT": CkT, "D1": D1, "DT": DT, "DkT": DkT,
        "log_F0": float(log_F0), "log_F1": float(log_F1),
        "F2": _value(log_F2, sign_F2), "log_F3": float(log_F3),
        "Akprod": float(np.exp(A1sum_v)),
        **parts,
    }


@overload
def log_likelihood_ind(
    walks: DyncovWalks,
    r: float, alpha: float, s: float, beta: float,
    gamma_life: ArrayLike, gamma_trans: ArrayLike,
    intermediates: Literal[False] = False,
) -> NDArray[np.float64]: ...


@overload
def log_likelihood_ind(
    walks: DyncovWalks,
    r: float, alpha: float, s: float, beta: float,
    gamma_life: ArrayLike, gamma_trans: ArrayLike,
    intermediates: Literal[True],
) -> pd.DataFrame: ...


def log_likelihood_ind(
    walks: DyncovWalks,
    r: float, alpha: float, s: float, beta: float,
    gamma_life: ArrayLike, gamma_trans: ArrayLike,
    intermediates: bool = False,
) -> NDArray[np.float64] | pd.DataFrame:
    """Per-customer log-likelihood.

    With ``intermediates=True`` returns the whole thirty-column table CLVTools
    produces, which is what makes each block of the likelihood testable on its
    own.
    """
    customers = walks.customers(gamma_life, gamma_trans)
    # Entered once for the whole sweep. Per-term, the context manager alone
    # accounted for an eighth of the runtime.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rows = [log_likelihood_customer(r, alpha, s, beta, c) for c in customers]
    if not intermediates:
        return np.array([row["LL"] for row in rows])
    return pd.DataFrame(rows, index=walks.ids)[INTERMEDIATE_NAMES]


def probability_alive(
    walks: DyncovWalks,
    r: float, alpha: float, s: float, beta: float,
    gamma_life: ArrayLike, gamma_trans: ArrayLike,
) -> NDArray[np.float64]:
    r"""``PAlive`` with time-varying covariates. S6.4.2.

    The same ratio as the standard model's: the likelihood of the customer's
    history *and* their still being alive, over the likelihood of the history
    however it came about. Both halves are already computed for the
    log-likelihood, so this is one exponential away from it.

    .. math::
        \log F_1 &= \mathrm{A1sum} + \ln\Gamma(r{+}x) - \ln\Gamma(r)
                  + r\log\frac{\alpha_0}{\alpha_0 {+} \mathrm{Bksum}}
                  - x\log(\alpha_0 {+} \mathrm{Bksum})
                  + s\log\frac{\beta_0}{\beta_0 {+} D_{k_T}} \\
        \mathrm{PAlive} &= \exp(\log F_1 - \mathrm{LL})

    Only the *alive* branch appears in :math:`F_1`: a customer who has died
    contributes through the integral over their possible death times, which is
    the part of the likelihood this ratio divides out.

    Examples
    --------
    At CLVTools' own fitted parameters, for the first three customers:

    >>> from clvtools import ClvData, ClvDataDynCov
    >>> from clvtools import load_apparel_dyn_cov, load_apparel_trans
    >>> names = ["High.Season", "Gender", "Channel"]
    >>> data = ClvDataDynCov(
    ...     ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
    ...     load_apparel_dyn_cov(), names_cov_life=names, names_cov_trans=names)
    >>> alive = probability_alive(
    ...     data.walks(), r=1.977706, alpha=115.177940, s=2.012683,
    ...     beta=158.181797, gamma_life=[-2.482678, -0.512544, 0.505730],
    ...     gamma_trans=[0.718314, 0.264898, 0.613721])
    >>> [round(float(v), 6) for v in alive[:3]]
    [0.965822, 0.98131, 0.314156]
    """
    table = log_likelihood_ind(
        walks, r, alpha, s, beta, gamma_life, gamma_trans, intermediates=True
    )
    x = walks.x
    with np.errstate(divide="ignore", invalid="ignore"):
        log_f1 = (
            table["A1sum"].to_numpy()
            + special.gammaln(r + x) - special.gammaln(r)
            + r * (np.log(alpha) - np.log(alpha + table["Bksum"].to_numpy()))
            - x * np.log(alpha + table["Bksum"].to_numpy())
            + s * (np.log(beta) - np.log(beta + table["DkT"].to_numpy()))
        )
        return np.exp(log_f1 - table["LL"].to_numpy())


def log_likelihood(
    walks: DyncovWalks,
    r: float, alpha: float, s: float, beta: float,
    gamma_life: ArrayLike, gamma_trans: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    """The sample log-likelihood maximised in S6.4.2."""
    ll = log_likelihood_ind(walks, r, alpha, s, beta, gamma_life, gamma_trans)
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


# -- estimation ---------------------------------------------------------------


@dataclass(frozen=True)
class PnbdDynCovParams(Fitted):
    """A fitted Pareto/NBD with time-varying covariates."""

    r: float
    alpha: float
    s: float
    beta: float
    gamma_life: NDArray[np.float64]
    gamma_trans: NDArray[np.float64]
    names_cov_life: list[str]
    names_cov_trans: list[str]
    log_likelihood: float
    converged: bool
    n_customers: int
    n_evaluations: int = 0
    hessian: NDArray[np.float64] | None = field(default=None, repr=False)

    def __iter__(self):
        yield from (self.r, self.alpha, self.s, self.beta)
        yield from self.gamma_life
        yield from self.gamma_trans

    @property
    def names(self) -> list[str]:
        return (
            ["r", "alpha", "s", "beta"]
            + [f"life.{n}" for n in self.names_cov_life]
            + [f"trans.{n}" for n in self.names_cov_trans]
        )

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.names, list(self), strict=True))


def fit_pnbd_dyncov(
    walks: DyncovWalks,
    names_cov_life: list[str] | None = None,
    names_cov_trans: list[str] | None = None,
    start: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    start_cov: float = 0.1,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = False,
    options: dict | None = None,
    weights: ArrayLike | None = None,
) -> PnbdDynCovParams:
    r"""Estimate the model of S6.4.2.

    S6.4: "the model estimation with time-varying covariates is computationally
    much more demanding than the previously detailed alternatives. It is
    recommended to keep an eye on the progress of model optimization." That
    holds here too, and more so: each likelihood evaluation walks 600 customers
    and makes some 80,000 scalar hypergeometric calls, so a fit runs in minutes.

    The search vector is
    ``[log r, log alpha, log s, log beta, gamma_life..., gamma_trans...]``.
    """
    from clvtools._optimize import options_for

    n_life, n_trans = walks.n_cov_life, walks.n_cov_trans
    names_cov_life = list(
        names_cov_life if names_cov_life is not None
        else [f"life{i}" for i in range(n_life)]
    )
    names_cov_trans = list(
        names_cov_trans if names_cov_trans is not None
        else [f"trans{i}" for i in range(n_trans)]
    )

    start_arr = start_values(start, count=4, parameters="values (r, alpha, s, beta)")

    x0 = np.concatenate(
        [np.log(start_arr), np.full(n_life + n_trans, float(start_cov))]
    )
    evaluations = 0

    def negative_ll(v: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        r, alpha, s, beta = np.exp(v[:4])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(
                walks, r, alpha, s, beta,
                v[4 : 4 + n_life], v[4 + n_life :],
                weights=weights,
            )
        return np.inf if not np.isfinite(value) else -value

    from scipy import optimize

    result = optimize.minimize(
        negative_ll, x0=x0, method=method,
        options=options_for(method, maxiter, x0, options),
    )
    result = finished(result, "time-varying covariate Pareto/NBD")

    r, alpha, s, beta = (float(v) for v in np.exp(result.x[:4]))

    hess = None
    if hessian:
        # Off by default, and alone among the fits in that. A central
        # difference over this many parameters is about 350 evaluations of a
        # likelihood that costs ~0.1 s, so ``summary()`` would take a minute on
        # this model and nothing at all on every other one. The argument exists
        # so the error from ``summary()`` names something a caller can actually
        # do -- which is finding 8 of ``docs/review-2026-09-02.md`` -- and the
        # default says what it costs.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            def natural_ll(v):
                r_, alpha_, s_, beta_ = (float(p) for p in v[:4])
                return -log_likelihood(
                    walks, r_, alpha_, s_, beta_,
                    v[4 : 4 + n_life], v[4 + n_life :],
                    weights=weights,
                )

            natural = np.concatenate([np.exp(result.x[:4]), result.x[4:]])
            hess = numerical_hessian(natural_ll, natural)
    return PnbdDynCovParams(
        r=r, alpha=alpha, s=s, beta=beta,
        gamma_life=np.asarray(result.x[4 : 4 + n_life], dtype=float),
        gamma_trans=np.asarray(result.x[4 + n_life :], dtype=float),
        names_cov_life=names_cov_life,
        names_cov_trans=names_cov_trans,
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        # Weighted, matching `fit_pnbd`. `weights` multiplies the per-customer
        # log-likelihoods a few lines up, so the objective is that of a sample
        # of `sum(weights)` customers and BIC's `n` has to be the same number.
        # This reported `walks.n_customers` -- the cohort size -- so a bootstrap
        # draw's BIC was computed against a different n from its own likelihood.
        n_customers=int(
            walks.n_customers if weights is None else np.sum(
                np.asarray(weights, dtype=float)
            )
        ),
        n_evaluations=evaluations,
        hessian=hess,
    )
