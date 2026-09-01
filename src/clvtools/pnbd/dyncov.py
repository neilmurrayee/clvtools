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
covariate intervals it crosses. Each customer has

``real_walks_trans``
    one per repeat transaction, covering the interval since the previous one;
``aux_walk_trans``
    from the last transaction to the end of the estimation period;
``real_walk_life``
    from coming alive to the last transaction;
``aux_walk_life``
    from the last transaction to the end of the estimation period.

A walk carries two extra numbers: ``d1``, the distance from its start to the end
of the first covariate interval it touches, and ``tjk``, its total length.

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

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import special

# `build_walks` annotates its argument with `ClvData`, and `py.typed`
# promises that annotation resolves -- so this is a real import, not a
# `TYPE_CHECKING` one. The cycle stays broken at the other end:
# `ClvDataDynCov` imports `build_walks` inside the method that calls it,
# and must keep doing so.
from clvtools.data import ClvData
from clvtools.inference import Fitted

__all__ = [
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


@dataclass(frozen=True)
class Walk:
    r"""A span of time, as the covariate multipliers over the intervals it crosses.

    ``values`` holds :math:`\exp(\boldsymbol{\gamma}'\mathbf{x}_k)` for each
    covariate interval :math:`k` the span touches, first to last.
    """

    values: NDArray[np.float64]

    @property
    def n_elem(self) -> int:
        return int(self.values.size)

    @property
    def first(self) -> float:
        return float(self.values[0])

    @property
    def last(self) -> float:
        return float(self.values[-1])

    def elem(self, i: int) -> float:
        """Zero-based element access."""
        return float(self.values[i])

    def sum_middle(self) -> float:
        """Everything but the first and last element. Needs at least three."""
        if self.n_elem < 3:
            raise ValueError(
                f"sum_middle needs at least 3 elements, walk has {self.n_elem}"
            )
        return float(self.values[1:-1].sum())

    def sum_from_to(self, start: int, stop: int) -> float:
        """Zero-based, inclusive of both ends -- Armadillo's ``subvec``."""
        return float(self.values[start : stop + 1].sum())


@dataclass(frozen=True)
class TransactionWalk(Walk):
    r"""A walk over a transaction interval.

    ``d1`` is the distance from the walk's start to the end of the covariate
    interval it starts in; ``tjk`` is the walk's total length. Both are in the
    data's time units.
    """

    d1: float = float("nan")
    tjk: float = float("nan")


EMPTY_WALK = Walk(np.empty(0))


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

    if real_walk.n_elem == 1:
        sum_real = real_walk.first * d_omega
    elif real_walk.n_elem == 2:
        sum_real = real_walk.first * d_omega + real_walk.last
    else:
        sum_real = real_walk.first * d_omega + real_walk.sum_middle() + real_walk.last

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


# -- the hypergeometric pair --------------------------------------------------


def _hyp_alpha_ge_beta(
    r: float, s: float, x: float,
    alpha_1: float, beta_1: float, alpha_2: float, beta_2: float,
) -> float:
    r"""The :math:`\alpha \ge \beta` arm of each :math:`F_2` term.

    .. math::
        \frac{{}_2F_1(r{+}s{+}x,\, s{+}1,\, r{+}s{+}x{+}1;\, z_1)}
             {\alpha_1^{r+s+x}}
        - \frac{{}_2F_1(\cdots;\, z_2)}{\alpha_2^{r+s+x}},
        \qquad z_j = 1 - \beta_j/\alpha_j

    Where the series will not converge, CLVTools substitutes the limiting form
    :math:`(1-z)^{r+x} C / \beta^{r+s+x}`; the same fallback is used here so the
    two agree everywhere, including where neither is accurate.
    """
    a = r + s + x
    out = 0.0
    for alpha, beta, sign in ((alpha_1, beta_1, 1.0), (alpha_2, beta_2, -1.0)):
        z = 1.0 - beta / alpha
        value = special.hyp2f1(a, s + 1.0, a + 1.0, z)
        if np.isfinite(value):
            term = value / alpha**a
        else:
            # Computed here rather than up front: the fallback fires rarely,
            # and four gammaln calls on every term is most of this function.
            log_c = (
                special.gammaln(a + 1.0) + special.gammaln(s)
                - special.gammaln(a) - special.gammaln(s + 1.0)
            )
            term = (1.0 - z) ** (r + x) * np.exp(log_c) / beta**a
        out += sign * term
    return float(out)


def _hyp_beta_gt_alpha(
    r: float, s: float, x: float,
    alpha_1: float, beta_1: float, alpha_2: float, beta_2: float,
) -> float:
    r"""The :math:`\beta > \alpha` arm, with the roles exchanged.

    .. math::
        \frac{{}_2F_1(r{+}s{+}x,\, r{+}x,\, r{+}s{+}x{+}1;\, z_1)}
             {\beta_1^{r+s+x}} - \cdots,
        \qquad z_j = 1 - \alpha_j/\beta_j
    """
    a = r + s + x
    out = 0.0
    for alpha, beta, sign in ((alpha_1, beta_1, 1.0), (alpha_2, beta_2, -1.0)):
        z = 1.0 - alpha / beta
        value = special.hyp2f1(a, r + x, a + 1.0, z)
        if np.isfinite(value):
            term = value / beta**a
        else:
            log_c = (
                special.gammaln(a + 1.0) + special.gammaln(r + x - 1.0)
                - special.gammaln(a) - special.gammaln(r + x)
            )
            term = (1.0 - z) ** (s + 1.0) * np.exp(log_c) / alpha**a
        out += sign * term
    return float(out)


def _hyp_term(
    r: float, s: float, x: float,
    alpha_1: float, beta_1: float, alpha_2: float, beta_2: float,
    ratio: float,
) -> float:
    """One :math:`F_2` term: the scaled difference of two hypergeometrics."""
    branch = _hyp_alpha_ge_beta if alpha_1 >= beta_1 else _hyp_beta_gt_alpha
    return ratio**s * branch(r, s, x, alpha_1, beta_1, alpha_2, beta_2)


@dataclass(frozen=True)
class Customer:
    """One customer's sufficient statistics and walks."""

    x: float
    t_x: float
    T_cal: float
    d_omega: float
    real_walks_trans: list[TransactionWalk]
    aux_walk_trans: TransactionWalk
    real_walk_life: Walk
    aux_walk_life: Walk


# The arguments are the terms of F2 itself, in the paper's notation. They are
# computed together in `log_likelihood_customer` and consumed only here; a
# wrapper object would add an allocation per customer per likelihood
# evaluation, on the path this module is already tuned around.
def _f2(  # noqa: PLR0913, PLR0917
    r: float, alpha_0: float, s: float, beta_0: float,
    c: Customer,
    B1: float, D1: float, BT: float, DT: float,
    A1T: float, C1T: float, AkT: float, CkT: float, Bjsum: float,
) -> tuple[float, dict[str, float]]:
    r""":math:`F_2 = Y_1 + Y_{k_T} + \sum_{i=2}^{k_T-1} Y_i`.

    One term per covariate interval the auxiliary walk crosses: the first, the
    last, and a sum over those between. S3.3: the likelihood "contains a sum of
    expressions similar to the likelihood of the standard model, each being
    calculated for a certain time interval".
    """
    dT = c.aux_walk_trans.d1

    a1T = Bjsum + B1 + c.T_cal * A1T
    b1T = D1 + c.T_cal * C1T
    a1 = Bjsum + B1 + A1T * (c.t_x + dT - 1.0)
    b1 = D1 + C1T * (c.t_x + dT - 1.0)

    parts = {"dT": dT, "a1T": a1T, "b1T": b1T, "a1": a1, "b1": b1}

    if c.aux_walk_life.n_elem == 1:
        # A single covariate interval: no first/last split and no middle sum.
        f2 = _hyp_term(
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
        return f2, parts

    n_walks = float(c.aux_walk_life.n_elem)
    akt = Bjsum + BT + AkT * (c.t_x + dT + n_walks - 2.0)
    bkT = DT + CkT * (c.t_x + dT + n_walks - 2.0)
    aT = Bjsum + BT + c.T_cal * AkT
    bT = DT + c.T_cal * CkT
    parts.update({"akt": akt, "bkT": bkT, "aT": aT, "bT": bT})

    # Y_1 -- the first covariate interval after the last transaction.
    f2_1 = _hyp_term(
        r, s, c.x,
        a1 + (1.0 - dT) * A1T + alpha_0,
        (b1 + (1.0 - dT) * C1T + beta_0) * A1T / C1T,
        a1 + A1T + alpha_0,
        (b1 + C1T + beta_0) * A1T / C1T,
        A1T / C1T,
    )
    parts["F2.1"] = f2_1
    if not np.isfinite(f2_1):
        parts.update({"F2.2": float("nan"), "F2.3": float("nan")})
        return f2_1, parts

    # Y_kT -- the last interval, running to the end of the estimation period.
    f2_2 = _hyp_term(
        r, s, c.x,
        akt + alpha_0, (bkT + beta_0) * AkT / CkT,
        aT + alpha_0, (bT + beta_0) * AkT / CkT,
        AkT / CkT,
    )
    parts["F2.2"] = f2_2
    if not np.isfinite(f2_2):
        parts["F2.3"] = float("nan")
        return f2_2, parts

    # The intervals in between.
    f2_3 = 0.0
    for i in range(2, c.aux_walk_trans.n_elem):
        Ai = c.aux_walk_trans.elem(i - 1)
        Bi = b_i(i, c.t_x, c.aux_walk_trans)
        ai = Bjsum + Bi + Ai * (c.t_x + dT + (i - 2.0))

        Ci = c.aux_walk_life.elem(i - 1)
        Di = d_i(i, c.real_walk_life, c.aux_walk_life, c.d_omega)
        bi = Di + Ci * (c.t_x + dT + (i - 2.0))

        f2_3 += _hyp_term(
            r, s, c.x,
            ai + alpha_0, (bi + beta_0) * Ai / Ci,
            ai + Ai + alpha_0, (bi + Ci + beta_0) * Ai / Ci,
            Ai / Ci,
        )
        if not np.isfinite(f2_3):
            break

    parts["F2.3"] = f2_3
    return f2_1 + f2_2 + f2_3, parts


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

    F2, parts = _f2(
        r, alpha_0, s, beta_0, c,
        B1, D1, BT, DT, A1T, C1T, AkT, CkT, Bjsum_v,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if not np.isfinite(F2):
            ll = float("nan")
        elif F2 < 0.0:
            # F1*F2 is negative but smaller in magnitude than F3, so the sum is
            # still positive; log1p keeps the near-cancellation accurate.
            ll = log_F0 + log_F3 + np.log1p(np.exp(log_F1 - log_F3) * F2)
        elif F2 > 0.0:
            max_ab = max(log_F1 + np.log(F2), log_F3)
            ll = log_F0 + max_ab + np.log(
                np.exp(log_F1 + np.log(F2) - max_ab) + np.exp(log_F3 - max_ab)
            )
        else:
            ll = log_F0 + log_F3

    return {
        "LL": float(ll),
        "A1T": A1T, "AkT": AkT, "A1sum": A1sum_v,
        "B1": B1, "BT": BT, "Bjsum": Bjsum_v, "Bksum": Bksum_v,
        "C1T": C1T, "CkT": CkT, "D1": D1, "DT": DT, "DkT": DkT,
        "log_F0": float(log_F0), "log_F1": float(log_F1),
        "F2": float(F2), "log_F3": float(log_F3),
        "Akprod": float(np.exp(A1sum_v)),
        **parts,
    }


@dataclass(frozen=True)
class DyncovWalks:
    r"""The walk structures for a whole customer base.

    Rather than one array per customer, the covariate rows for every walk of a
    kind are stacked into one matrix and each customer holds a ``[from, to]``
    index into it. That is how CLVTools stores them, and it keeps
    :math:`\exp(\text{covariates} \cdot \gamma)` a single matrix product for the
    entire sample instead of one per customer.

    Indices are one-based and inclusive at both ends.
    """

    ids: pd.Index
    x: NDArray[np.float64]
    t_x: NDArray[np.float64]
    T_cal: NDArray[np.float64]
    d_omega: NDArray[np.float64]

    #: ``(n_customers, 2)``: from, to.
    walkinfo_aux_life: NDArray[np.float64]
    walkinfo_real_life: NDArray[np.float64]
    #: ``(n_customers, 4)``: from, to, d1, tjk.
    walkinfo_aux_trans: NDArray[np.float64]
    #: ``(n_walks, 4)``, with each customer owning a contiguous block.
    walkinfo_real_trans: NDArray[np.float64]
    #: ``(n_customers,)`` block bounds into ``walkinfo_real_trans``; NaN for a
    #: zero-repeater, who has no repeat-purchase intervals at all.
    real_trans_from: NDArray[np.float64]
    real_trans_to: NDArray[np.float64]

    covdata_aux_life: NDArray[np.float64]
    covdata_real_life: NDArray[np.float64]
    covdata_aux_trans: NDArray[np.float64]
    covdata_real_trans: NDArray[np.float64]

    @property
    def n_customers(self) -> int:
        return int(self.x.size)

    @property
    def n_cov_life(self) -> int:
        return int(self.covdata_aux_life.shape[1])

    @property
    def n_cov_trans(self) -> int:
        return int(self.covdata_aux_trans.shape[1])

    def customers(
        self, gamma_life: ArrayLike, gamma_trans: ArrayLike
    ) -> list[Customer]:
        r"""Build every customer's walks at the given covariate parameters."""
        gamma_life = np.asarray(gamma_life, dtype=float)
        gamma_trans = np.asarray(gamma_trans, dtype=float)
        if gamma_life.size != self.n_cov_life:
            raise ValueError(
                f"{self.n_cov_life} attrition covariates but {gamma_life.size} "
                "parameters"
            )
        if gamma_trans.size != self.n_cov_trans:
            raise ValueError(
                f"{self.n_cov_trans} transaction covariates but "
                f"{gamma_trans.size} parameters"
            )

        adj_aux_life = np.exp(self.covdata_aux_life @ gamma_life)
        adj_real_life = np.exp(self.covdata_real_life @ gamma_life)
        adj_aux_trans = np.exp(self.covdata_aux_trans @ gamma_trans)
        adj_real_trans = np.exp(self.covdata_real_trans @ gamma_trans)

        out = []
        for i in range(self.n_customers):
            al_from, al_to = self.walkinfo_aux_life[i]
            aux_life = Walk(adj_aux_life[int(al_from) - 1 : int(al_to)])

            rl_from, rl_to = self.walkinfo_real_life[i]
            if np.isfinite(rl_from) and np.isfinite(rl_to):
                real_life = Walk(adj_real_life[int(rl_from) - 1 : int(rl_to)])
            else:
                real_life = EMPTY_WALK

            at_from, at_to, at_d1, at_tjk = self.walkinfo_aux_trans[i]
            aux_trans = TransactionWalk(
                adj_aux_trans[int(at_from) - 1 : int(at_to)], d1=at_d1, tjk=at_tjk
            )

            real_trans: list[TransactionWalk] = []
            if np.isfinite(self.real_trans_from[i]):
                block = self.walkinfo_real_trans[
                    int(self.real_trans_from[i]) - 1 : int(self.real_trans_to[i])
                ]
                for w_from, w_to, d1, tjk in block:
                    real_trans.append(
                        TransactionWalk(
                            adj_real_trans[int(w_from) - 1 : int(w_to)],
                            d1=d1, tjk=tjk,
                        )
                    )

            out.append(
                Customer(
                    x=float(self.x[i]), t_x=float(self.t_x[i]),
                    T_cal=float(self.T_cal[i]), d_omega=float(self.d_omega[i]),
                    real_walks_trans=real_trans,
                    aux_walk_trans=aux_trans,
                    real_walk_life=real_life,
                    aux_walk_life=aux_life,
                )
            )
        return out


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


# -- building the walks from raw data -----------------------------------------


def _interval_index(grid: NDArray[np.int64], when: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""Which covariate interval each timepoint falls in.

    S3.3: the covariate functions "return the covariates :math:`x^P_k, x^A_k` of
    the interval :math:`k` in which :math:`s` lies". Intervals are half-open,
    :math:`[\text{grid}_k, \text{grid}_{k+1})`, because S6.1 has the covariate
    date marking the *start* of the period it describes.
    """
    return np.searchsorted(grid, when, side="right") - 1


def _distance_to_interval_end(
    grid: NDArray[np.int64], when: int, span: Callable[[int, int], float]
) -> float:
    r"""``d`` -- from a timepoint to the end of the covariate interval it is in.

    A timepoint sitting exactly on an interval boundary gets a whole period, not
    zero: CLVTools' comment is "d shall be 1 if it is exactly on the time unit
    lower boundary". Without that, a transaction on a boundary would contribute
    nothing to its own interval and the walk arithmetic would double-count.

    ``span`` measures a gap between two grid days in time units, so this works
    for a calendar unit as well as a fixed one.
    """
    k = int(np.searchsorted(grid, when, side="right") - 1)
    if grid[k] == when:
        return 1.0
    return span(when, int(grid[k + 1]))


@dataclass(frozen=True)
class _WalkSpec:
    """One walk, before the flat arrays are assembled."""

    customer: int
    lo: int
    hi: int
    d1: float = float("nan")
    tjk: float = float("nan")


def _to_days(values) -> NDArray[np.int64]:
    """Whole days since the epoch, so every span is exact.

    All of :func:`build_walks`' arithmetic is done in whole days and converted
    to time units only at the end. Working in nanoseconds instead leaves ``d1``
    and ``tjk`` off by around 4e-13, which is enough to break the exact
    cancellation that makes ``F2.2`` vanish.
    """
    return (
        pd.to_datetime(pd.Series(values))
        .to_numpy(dtype="datetime64[D]")
        .astype("int64")
    )


def _prepare_covariates(
    frame: pd.DataFrame, names: list[str] | None, name_date_cov: str
) -> tuple[dict, dict, list[str]]:
    """Split a covariate table into a per-customer date grid and value matrix."""
    out = frame.copy()
    out["Id"] = out["Id"].astype(str)
    if name_date_cov not in out.columns:
        raise ValueError(f"covariate data has no {name_date_cov!r} column")
    chosen = list(
        names
        if names is not None
        else [c for c in out.columns if c not in ("Id", name_date_cov)]
    )
    missing = [n for n in chosen if n not in out.columns]
    if missing:
        raise ValueError(f"covariate data is missing columns: {missing}")
    out = out.sort_values(["Id", name_date_cov], kind="stable")
    grids, rows = {}, {}
    for cid, group in out.groupby("Id", sort=True):
        grids[cid] = _to_days(group[name_date_cov])
        rows[cid] = group[chosen].to_numpy(dtype=float)
    return grids, rows, chosen


def _check_covariate_coverage(ids, grids_life: dict, grids_trans: dict) -> None:
    """Every customer covered, and the two grids aligned where they overlap.

    Every walk's interval indices are derived once, from the lifetime grid, and
    then used to slice *both* covariate matrices (see :func:`_stack`). That is
    sound only while the two grids agree on the intervals a walk can reach.
    The two series may legitimately run to different dates -- whether they
    reach far enough to predict over is checked at prediction time -- so only
    the overlapping prefix has to match here. A misalignment within it would
    silently shift the transactional walks and return a wrong likelihood.
    """
    missing_cov = (set(ids) - set(grids_life)) | (set(ids) - set(grids_trans))
    if missing_cov:
        raise ValueError(
            f"{len(missing_cov)} customers have no covariate data, "
            f"e.g. {sorted(missing_cov)[:3]}"
        )
    mismatched = [
        cid
        for cid in ids
        if not np.array_equal(
            grids_life[cid][: min(len(grids_life[cid]), len(grids_trans[cid]))],
            grids_trans[cid][: min(len(grids_life[cid]), len(grids_trans[cid]))],
        )
    ]
    if mismatched:
        raise ValueError(
            "lifetime and transaction covariates must share one date grid; "
            f"{len(mismatched)} customers differ, e.g. {mismatched[:3]}"
        )


@dataclass
class _CustomerSpecs:
    """The per-customer summary, and which covariate intervals each walk spans."""

    x: NDArray[np.float64]
    t_x: NDArray[np.float64]
    T_cal: NDArray[np.float64]
    d_omega: NDArray[np.float64]
    aux_life: list[_WalkSpec]
    aux_trans: list[_WalkSpec]
    real_life: dict[int, _WalkSpec]
    real_trans: list[_WalkSpec]


def _customer_specs(
    trans: pd.DataFrame,
    ids: list,
    grids_life: dict,
    end: int,
    span: Callable[[int, int], float],
) -> _CustomerSpecs:
    """Walk out each customer's transactions, recording the four kinds of walk."""
    index_of = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    out = _CustomerSpecs(
        x=np.zeros(n), t_x=np.zeros(n), T_cal=np.zeros(n), d_omega=np.zeros(n),
        aux_life=[], aux_trans=[], real_life={}, real_trans=[],
    )

    for cid, group in trans.groupby("Id", sort=True):
        i = index_of[cid]
        grid = grids_life[cid]
        dates = np.sort(_to_days(group["Date"]))
        first, last = int(dates[0]), int(dates[-1])

        out.x[i] = len(dates) - 1
        out.t_x[i] = span(first, last)
        out.T_cal[i] = span(first, end)
        out.d_omega[i] = _distance_to_interval_end(grid, first, span)

        k_first = int(_interval_index(grid, np.array([first]))[0])
        k_last = int(_interval_index(grid, np.array([last]))[0])
        k_end = int(_interval_index(grid, np.array([end]))[0])

        # The auxiliary walks run from the last transaction to the window end.
        out.aux_life.append(_WalkSpec(i, k_last, k_end))
        out.aux_trans.append(_WalkSpec(
            i, k_last, k_end,
            d1=_distance_to_interval_end(grid, last, span),
            tjk=span(last, end),
        ))

        # The real lifetime walk stops short of the auxiliary walk's first
        # interval, so the two never overlap.
        if k_first <= k_last - 1:
            out.real_life[i] = _WalkSpec(i, k_first, k_last - 1)

        # One transaction walk per repeat purchase, over the interval since the
        # previous one.
        for previous, this in pairwise(dates):
            out.real_trans.append(_WalkSpec(
                i,
                int(_interval_index(grid, np.array([previous]))[0]),
                int(_interval_index(grid, np.array([this]))[0]),
                d1=_distance_to_interval_end(grid, int(previous), span),
                tjk=span(int(previous), int(this)),
            ))
    return out


def _stack(
    specs: list[_WalkSpec], rows: dict[str, NDArray[np.float64]], ids: list
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flatten walks into one matrix plus one-based inclusive indices."""
    blocks, info, cursor = [], [], 1
    for spec in specs:
        block = rows[ids[spec.customer]][spec.lo : spec.hi + 1]
        if len(block) != spec.hi - spec.lo + 1:
            # A short covariate series would slice to fewer rows than the walk
            # spans, shifting every later walk in the stacked matrix.
            raise ValueError(
                f"customer {ids[spec.customer]!r} has covariate data for "
                f"{len(block)} of the {spec.hi - spec.lo + 1} periods its walk spans"
            )
        blocks.append(block)
        info.append((cursor, cursor + len(block) - 1, spec.d1, spec.tjk))
        cursor += len(block)
    covdata = np.vstack(blocks) if blocks else np.empty((0, 0))
    return covdata, np.array(info, dtype=float)


def _real_trans_bounds(
    specs: list[_WalkSpec], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """First and last transaction walk belonging to each customer, one-based."""
    real_from = np.full(n, np.nan)
    real_to = np.full(n, np.nan)
    for position, spec in enumerate(specs, start=1):
        if np.isnan(real_from[spec.customer]):
            real_from[spec.customer] = position
        real_to[spec.customer] = position
    return real_from, real_to


def build_walks(
    clv_data: ClvData,
    covariates_life: pd.DataFrame,
    covariates_trans: pd.DataFrame | None = None,
    names_cov_life: list[str] | None = None,
    names_cov_trans: list[str] | None = None,
    name_date_cov: str = "Cov.Date",
) -> DyncovWalks:
    r"""Assemble every customer's walks from transaction and covariate data.

    Takes a :class:`~clvtools.data.ClvData` rather than a raw log, so the
    day-level aggregation of S6.1 and the estimation split are already applied.
    Passing the raw log instead would double-count a customer who bought twice
    on one day, which changes both ``x`` and the number of transaction walks.

    The four kinds of walk, and the span each covers:

    ``real_walks_trans``
        :math:`[t_{j-1}, t_j]` for each repeat transaction :math:`j`. The first
        transaction gets none -- there is no preceding interval for it to
        depend on.
    ``aux_walk_trans``
        :math:`[t_x, T]`, from the last transaction to the end of the
        estimation period.
    ``real_walk_life``
        the covariate intervals from coming alive up to, but not including,
        the one holding the last transaction. A customer whose first and last
        transactions share an interval has none.
    ``aux_walk_life``
        the same span as ``aux_walk_trans``, over the attrition covariates.

    ``real_walk_life`` and ``aux_walk_life`` deliberately do not overlap: the
    interval containing the last transaction belongs to the auxiliary walk
    alone, which is what lets :func:`d_i` treat the pair as one continuous span.
    """
    if covariates_trans is None:
        covariates_trans = covariates_life

    trans = clv_data.transactions
    trans = trans[trans["Date"] <= clv_data.estimation_end]
    if trans.empty:
        raise ValueError("no transactions fall within the estimation period")

    unit = clv_data.time
    ids = sorted(trans["Id"].unique())

    def span(day_a: int, day_b: int) -> float:
        """The gap between two epoch-days, in time units."""
        return unit.elapsed(
            pd.Timestamp(day_a, unit="D"), pd.Timestamp(day_b, unit="D")
        )

    grids_life, rows_life, names_cov_life = _prepare_covariates(
        covariates_life, names_cov_life, name_date_cov
    )
    grids_trans, rows_trans, names_cov_trans = _prepare_covariates(
        covariates_trans, names_cov_trans, name_date_cov
    )
    _check_covariate_coverage(ids, grids_life, grids_trans)

    end = int(_to_days([clv_data.estimation_end])[0])
    specs = _customer_specs(trans, ids, grids_life, end, span)

    covdata_aux_life, wi_aux_life = _stack(specs.aux_life, rows_life, ids)
    covdata_aux_trans, wi_aux_trans = _stack(specs.aux_trans, rows_trans, ids)

    ordered_real_life = [specs.real_life[i] for i in sorted(specs.real_life)]
    covdata_real_life, wi_real_life_packed = _stack(ordered_real_life, rows_life, ids)
    # Customers without a real lifetime walk carry NaN, which the walk builder
    # reads as "empty".
    wi_real_life = np.full((len(ids), 2), np.nan)
    for spec, row in zip(ordered_real_life, wi_real_life_packed, strict=True):
        wi_real_life[spec.customer] = row[:2]

    covdata_real_trans, wi_real_trans = _stack(specs.real_trans, rows_trans, ids)
    real_from, real_to = _real_trans_bounds(specs.real_trans, len(ids))

    return DyncovWalks(
        ids=pd.Index(ids, name="Id"),
        x=specs.x, t_x=specs.t_x, T_cal=specs.T_cal, d_omega=specs.d_omega,
        walkinfo_aux_life=wi_aux_life[:, :2],
        walkinfo_real_life=wi_real_life,
        walkinfo_aux_trans=wi_aux_trans,
        walkinfo_real_trans=wi_real_trans,
        real_trans_from=real_from,
        real_trans_to=real_to,
        covdata_aux_life=covdata_aux_life,
        covdata_real_life=covdata_real_life,
        covdata_aux_trans=covdata_aux_trans,
        covdata_real_trans=covdata_real_trans,
    )


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

    @property
    def n_parameters(self) -> int:
        return len(self.names)

    @property
    def aic(self) -> float:
        return 2 * self.n_parameters - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_parameters * np.log(self.n_customers) - 2 * self.log_likelihood


def fit_pnbd_dyncov(
    walks: DyncovWalks,
    names_cov_life: list[str] | None = None,
    names_cov_trans: list[str] | None = None,
    start: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    start_cov: float = 0.1,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
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

    start_arr = np.asarray(start, dtype=float)
    if start_arr.shape != (4,):
        raise ValueError("start must give four values (r, alpha, s, beta)")
    if np.any(start_arr <= 0):
        raise ValueError("start values must be strictly positive")

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

    r, alpha, s, beta = (float(v) for v in np.exp(result.x[:4]))
    return PnbdDynCovParams(
        r=r, alpha=alpha, s=s, beta=beta,
        gamma_life=np.asarray(result.x[4 : 4 + n_life], dtype=float),
        gamma_trans=np.asarray(result.x[4 + n_life :], dtype=float),
        names_cov_life=names_cov_life,
        names_cov_trans=names_cov_trans,
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_customers=walks.n_customers,
        n_evaluations=evaluations,
    )
