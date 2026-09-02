r"""Shared input validation and optimiser-result handling.

Every fit in this package took its own view of what a customer history may
contain and none of them said anything when the search failed. Two reviews
found the same two consequences, and both are in ``docs/review-2026-09-02.md``:

* **A recency a hair above the observation window collapses the whole fit.**
  The likelihoods need :math:`t_x \le T` exactly -- a ratio above one makes an
  intermediate negative and its logarithm ``NaN`` -- while the validators
  accepted ``t_x <= T + 1e-9``. Floating-point date arithmetic produces that
  slack routinely: a customer at ``t_x = T + 1e-10`` passes validation, every
  objective evaluation returns ``inf``, and the fit comes back at its start
  values with ``log_likelihood = -inf`` and ``converged = False``, having
  raised nothing. :func:`customer_history` clamps instead, which is the only
  reading of the input that is both safe and faithful: a purchase cannot
  happen after the window closed, so a value that far over is arithmetic
  rather than data.

* **Nothing warned.** There was no ``warnings.warn`` anywhere in ``src/``, so a
  fit that stopped at its iteration limit was distinguishable from a converged
  one only by reading ``converged``. CLVTools warns. :func:`finished` does too,
  naming the family and quoting the optimiser's own message, and it raises
  outright when the objective is not finite at the point the search returned --
  which is not a fit at all, and is the shape the clamped input above used to
  produce.

Both are deliberately loud rather than silent-and-wrong, and quiet on the happy
path: a converged fit warns about nothing.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "ConvergenceWarning",
    "customer_history",
    "finished",
    "spending_history",
    "start_values",
]


class ConvergenceWarning(UserWarning):
    """A fit did not converge, or converged somewhere it should not have.

    Its own category so that a caller can promote it to an error --
    ``warnings.simplefilter("error", ConvergenceWarning)`` -- or silence it,
    without touching every other warning NumPy and SciPy raise.
    """


def customer_history(
    x: np.ndarray, t_x: np.ndarray, T: np.ndarray, *, clamp_tolerance: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Validate a customer-by-sufficient-statistic triple, and clamp ``t_x``.

    The rules are the Pareto/NBD's, which were the strictest of the three
    families' and are now all of theirs: matching shapes, at least one
    customer, :math:`x \ge 0`, :math:`t_x \ge 0`, :math:`T > 0`,
    :math:`t_x \le T`, and :math:`t_x = 0` wherever :math:`x = 0`.

    ``t_x`` exceeding ``T`` by no more than ``clamp_tolerance`` is taken as
    arithmetic and clamped; anything further is data, and raises naming how
    many customers and by how much.

    >>> import numpy as np
    >>> x = np.array([2.0, 0.0])
    >>> t_x = np.array([50.0 + 1e-10, 0.0])
    >>> T = np.array([50.0, 50.0])
    >>> _, clamped, _ = customer_history(x, t_x, T)
    >>> bool(clamped[0] == 50.0)
    True

    A recency genuinely past the window is an error, not a rounding artefact:

    >>> customer_history(x, np.array([51.0, 0.0]), T)
    Traceback (most recent call last):
        ...
    ValueError: t_x exceeds T for 1 customer by up to 1 (tolerance 1e-06):
    a purchase cannot fall after the observation window closed
    """
    if not (x.shape == t_x.shape == T.shape):
        raise ValueError("x, t_x and T must have the same shape")
    if x.size == 0:
        raise ValueError("no customers to fit")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(t_x)) or np.any(~np.isfinite(T)):
        raise ValueError("x, t_x and T must all be finite")
    if np.any(x < 0):
        raise ValueError("frequencies x must be non-negative")
    if np.any(t_x < 0) or np.any(T <= 0):
        raise ValueError("t_x must be non-negative and T strictly positive")

    excess = t_x - T
    over = excess > clamp_tolerance
    if np.any(over):
        raise ValueError(
            f"t_x exceeds T for {int(over.sum())} customer"
            f"{'s' if over.sum() > 1 else ''} by up to {excess[over].max():g} "
            f"(tolerance {clamp_tolerance:g}):\n"
            "a purchase cannot fall after the observation window closed"
        )
    if np.any((x == 0) & (t_x != 0)):
        raise ValueError("t_x must be 0 where x == 0")

    return x, np.minimum(t_x, T), T


def spending_history(
    x: np.ndarray, z_bar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Validate a frequency and mean-spend pair for the Gamma-Gamma.

    The spending model's counterpart to :func:`customer_history`, and its rules
    are the ones eq. (17) needs: matching shapes, at least one customer,
    :math:`x \ge 0`, :math:`\bar{z} \ge 0`, and at least one customer with
    both a transaction and a spend -- without one the likelihood is constant
    and there is nothing to maximise.

    >>> import numpy as np
    >>> x, z_bar = spending_history(np.array([2.0, 0.0]), np.array([30.0, 0.0]))
    >>> x.tolist(), z_bar.tolist()
    ([2.0, 0.0], [30.0, 0.0])

    >>> spending_history(np.array([0.0]), np.array([0.0]))
    Traceback (most recent call last):
        ...
    ValueError: no customer has both a transaction and a spend: nothing to estimate
    """
    if x.shape != z_bar.shape:
        raise ValueError("x and z_bar must have the same shape")
    if x.size == 0:
        raise ValueError("no customers to fit")
    if np.any(x < 0):
        raise ValueError("frequencies x must be non-negative")
    if np.any(z_bar < 0):
        raise ValueError("mean spending must be non-negative")
    if not np.any((x > 0) & (z_bar > 0)):
        raise ValueError(
            "no customer has both a transaction and a spend: nothing to estimate"
        )
    return x, z_bar


def start_values(
    start, *, count: int, parameters: str
) -> np.ndarray:
    """The caller's start vector, checked against what the family expects.

    Every fit took the same two views of a start vector -- it must have the
    family's own length, and every value must be strictly positive, because
    the search runs over their logarithms. Five modules said so in five
    near-identical pairs of lines. ``parameters`` is the noun phrase that
    names them, so each family's message stays as specific as it was.

    >>> start_values((1.0, 2.0), count=2, parameters="values (r, alpha)")
    array([1., 2.])
    >>> start_values((1.0,), count=2, parameters="values (r, alpha)")
    Traceback (most recent call last):
        ...
    ValueError: start must give 2 values (r, alpha)
    >>> start_values((1.0, 0.0), count=2, parameters="values (r, alpha)")
    Traceback (most recent call last):
        ...
    ValueError: start values must be strictly positive
    """
    values = np.asarray(start, dtype=float)
    if values.shape != (count,):
        raise ValueError(f"start must give {count} {parameters}")
    if np.any(values <= 0):
        raise ValueError("start values must be strictly positive")
    return values


def finished(result, family: str):
    """Hand back an optimiser result, having said what happened to it.

    Raises when the objective is not finite at the returned point, because that
    is not a fit: the search never left, or left for somewhere the likelihood
    does not exist, and returning start values dressed as estimates is how that
    reaches a user as a number. Warns when the optimiser reports failure,
    quoting its own message, since a stopped-early fit is often still useful
    and the caller may want it -- but should not have to read a flag to learn
    it happened.

    >>> import warnings
    >>> from types import SimpleNamespace
    >>> ok = SimpleNamespace(success=True, fun=5.0, message="CONVERGENCE")
    >>> finished(ok, "Pareto/NBD") is ok
    True
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     stopped = SimpleNamespace(
    ...         success=False, fun=5.0, message="MAX ITERATIONS")
    ...     _ = finished(stopped, "BG/NBD")
    >>> print(caught[0].message)
    the BG/NBD fit did not converge: MAX ITERATIONS
    """
    if not np.isfinite(result.fun):
        raise ValueError(
            f"the {family} objective is not finite at the point the search "
            f"returned ({result.fun}), so there is no fit to report. The usual "
            "cause is an input the likelihood is undefined on; check for "
            "customers with t_x > T, zero or negative spending, or covariates "
            "that are not finite."
        )
    if not result.success:
        warnings.warn(
            f"the {family} fit did not converge: {result.message}",
            ConvergenceWarning,
            stacklevel=3,
        )
    return result
