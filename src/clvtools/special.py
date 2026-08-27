"""Special functions the Pareto/NBD expressions are built from.

Appendix A writes the marginalised likelihood in terms of the Gaussian
hypergeometric function: "The symbol :math:`_2F_1(a,b,c,z)` refers to the
integral representation of the Gaussian hypergeometric function". The
discounted-transactions expression additionally needs Tricomi's confluent
hypergeometric :math:`U`.

``scipy.special.hyp2f1`` agrees with the GSL routine CLVTools calls to about
5e-15 across the range these models visit, but returns ``nan`` in one corner --
large first parameter with ``z`` close to 1. Every Pareto/NBD call has the
special shape ``c = a + 1``, which admits a series that stays well behaved
there, so :func:`hyp2f1_ratio` falls back to it only for the entries SciPy
could not evaluate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

__all__ = ["hyp2f1_ratio", "kummer_u"]

#: Terms are added until they contribute less than this, relatively.
_SERIES_TOL = 1e-16
#: Ceiling on the number of terms summed. Successive terms decay like
#: :math:`z^n`, so reaching the tolerance needs about
#: :math:`\ln(\text{tol}) / \ln z` of them: ~3,700 at ``z = 0.99``, ~37,000 at
#: ``z = 0.999``, ~370,000 at ``z = 0.9999``. Past that the series is abandoned
#: rather than pursued -- see :func:`_hyp2f1_series`.
_SERIES_MAX_TERMS = 400_000


def _hyp2f1_series(a: float, b: float, z: float) -> float:
    r"""``2F1(a, b; a+1; z)`` by Euler's transformation, summed directly.

    Euler's transformation

    .. math::
        {}_2F_1(a, b; c; z) = (1-z)^{c-a-b} \, {}_2F_1(c-a, c-b; c; z)

    with :math:`c = a+1` gives

    .. math::
        {}_2F_1(a, b; a+1; z)
          = (1-z)^{1-b} \sum_{n \ge 0} \frac{(a+1-b)_n}{(a+1)_n} z^n .

    Successive terms are in ratio :math:`\frac{a+1-b+n}{a+1+n} z`, which is
    below :math:`z` for :math:`b > 0` and tends to it. So the sum converges
    geometrically, every term is positive, and there is no cancellation to lose
    digits to.

    The number of terms needed is decided up front from :math:`z` and summed
    with a single :func:`numpy.cumprod`. Doing it in a Python loop instead costs
    a second or more per call as :math:`z \to 1`, and an optimiser that wanders
    into that corner then appears to hang.

    Returns ``nan`` when :math:`z` is so close to 1 that the series would need
    more than :data:`_SERIES_MAX_TERMS`. Callers treat a non-finite likelihood
    as strictly worse than any real one, so the search moves away from such a
    point rather than stalling on it.
    """
    if z == 0.0:
        return 1.0
    if not 0.0 < z < 1.0:
        return float("nan")

    n_terms = int(np.ceil(np.log(_SERIES_TOL) / np.log(z))) + 100
    if n_terms > _SERIES_MAX_TERMS:
        return float("nan")

    upper, lower = a + 1.0 - b, a + 1.0
    n = np.arange(n_terms, dtype=float)
    terms = np.cumprod((upper + n) / (lower + n) * z)
    return float((1.0 - z) ** (1.0 - b) * (1.0 + terms.sum()))


def hyp2f1_ratio(a: ArrayLike, b: ArrayLike, z: ArrayLike) -> NDArray[np.float64]:
    r"""``2F1(a, b; a+1; z)`` -- the only shape the Pareto/NBD ever needs.

    Appendix A's :math:`A_1` and :math:`A_2` both call
    :math:`{}_2F_1(r+s+x,\, \cdot,\, r+s+x+1;\, \cdot)`, whose third parameter
    is always one more than its first.

    Examples
    --------
    Two values that follow from elementary identities. With ``b = 1`` the sum
    telescopes to :math:`-a z^{-a} \ln(1-z)` scaled, and at ``z = 0`` every
    hypergeometric is 1:

    >>> float(hyp2f1_ratio(2.0, 1.0, 0.0))
    1.0
    >>> import numpy as np
    >>> bool(np.isclose(hyp2f1_ratio(1.0, 1.0, 0.5), np.log(2) / 0.5))
    True

    It is finite where a direct SciPy call is not:

    >>> from scipy import special
    >>> bool(np.isnan(special.hyp2f1(200.0, 20.0, 201.0, 0.999)))
    True
    >>> float(hyp2f1_ratio(200.0, 20.0, 0.999))
    1.042158...e+58
    """
    a, b, z = (np.asarray(v, dtype=float) for v in (a, b, z))
    a, b, z = np.broadcast_arrays(a, b, z)

    out = special.hyp2f1(a, b, a + 1.0, z)

    unresolved = ~np.isfinite(out)
    if unresolved.any():
        flat_out, flat = out.ravel().copy(), unresolved.ravel()
        fa, fb, fz = a.ravel(), b.ravel(), z.ravel()
        for i in np.flatnonzero(flat):
            flat_out[i] = _hyp2f1_series(fa[i], fb[i], fz[i])
        out = flat_out.reshape(out.shape)

    return out


def kummer_u(a: float, b: float, z: ArrayLike) -> NDArray[np.float64]:
    r"""Tricomi's confluent hypergeometric :math:`U(a, b, z)`.

    Used by the discounted expected residual transactions of S6.3, where it
    appears as :math:`U(s, s, \delta(\beta + T))` with :math:`\delta` the
    continuous discount factor.

    A thin pass-through to ``scipy.special.hyperu``, matching the GSL routine
    CLVTools calls. It is named separately so the model code reads like the
    literature rather than like SciPy.

    Examples
    --------
    :math:`U(1, 1, z) = e^{z} E_1(z)`:

    >>> import numpy as np
    >>> from scipy import special
    >>> z = np.array([0.1, 1.0, 5.0])
    >>> bool(np.allclose(kummer_u(1.0, 1.0, z), np.exp(z) * special.exp1(z)))
    True
    """
    return special.hyperu(a, b, np.asarray(z, dtype=float))
