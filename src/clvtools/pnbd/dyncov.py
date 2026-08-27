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

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import special

__all__ = [
    "DyncovWalks",
    "TransactionWalk",
    "Walk",
    "a1sum",
    "b_i",
    "bjsum",
    "bksum",
    "d_i",
    "log_likelihood",
    "log_likelihood_ind",
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
    log_c = (
        special.gammaln(a + 1.0) + special.gammaln(s)
        - special.gammaln(a) - special.gammaln(s + 1.0)
    )
    out = 0.0
    for alpha, beta, sign in ((alpha_1, beta_1, 1.0), (alpha_2, beta_2, -1.0)):
        z = 1.0 - beta / alpha
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = special.hyp2f1(a, s + 1.0, a + 1.0, z)
            if np.isfinite(value):
                term = value / alpha**a
            else:
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
    log_c = (
        special.gammaln(a + 1.0) + special.gammaln(r + x - 1.0)
        - special.gammaln(a) - special.gammaln(r + x)
    )
    out = 0.0
    for alpha, beta, sign in ((alpha_1, beta_1, 1.0), (alpha_2, beta_2, -1.0)):
        z = 1.0 - alpha / beta
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = special.hyp2f1(a, r + x, a + 1.0, z)
            if np.isfinite(value):
                term = value / beta**a
            else:
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


def _f2(
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
    rows = [log_likelihood_customer(r, alpha, s, beta, c) for c in customers]
    if not intermediates:
        return np.array([row["LL"] for row in rows])
    return pd.DataFrame(rows, index=walks.ids)[INTERMEDIATE_NAMES]


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
