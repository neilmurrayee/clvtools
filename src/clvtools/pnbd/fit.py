r"""S3.2 and S6.2.1 - estimating :math:`(r, \alpha, s, \beta)`.

S3.2: "Given transactional records for a total of :math:`N` customers, these
parameters can be estimated by maximizing the log-likelihood

.. math::
    \sum_{i=1}^N \log L(r, \alpha, s, \beta \mid x_i, t_{x_i}, T_i)."

S6.2.1 notes that CLVTools "by default uses the optimization method
``L-BFGS-B``. If the optimization is not feasible, switching to the more robust
but often slower method ``Nelder-Mead`` is recommended", which is the same
choice offered here through ``method``.

All four parameters must stay strictly positive. Rather than constraining the
search, the optimiser works on their logarithms -- the same substitution
CLVTools' own C++ entry points make, which take log parameters throughout.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from clvtools._optimize import options_for
from clvtools._validate import customer_history, finished
from clvtools.inference import Fitted, numerical_hessian
from clvtools.pnbd.aggregate import log_likelihood

__all__ = ["PnbdParams", "fit_pnbd"]

#: S6.4: "If not given, the start values are set to 0.1 for all covariates";
#: for the model parameters CLVTools starts every one at 1.
_DEFAULT_START = (1.0, 1.0, 1.0, 1.0)

_NAMES = ("r", "alpha", "s", "beta")


@dataclass(frozen=True)
class PnbdParams(Fitted):
    r"""A fitted Pareto/NBD.

    S3.2: ":math:`r` and :math:`\alpha` are the shape and scale parameters of
    the Gamma distribution of the transaction process, and :math:`s` and
    :math:`\beta` are the shape and scale parameters of the Gamma distribution
    of the attrition process. Consequently, :math:`r/\alpha` and
    :math:`s/\beta` represent the purchase rate and the dropout rate of an
    average customer, respectively."
    """

    r: float
    alpha: float
    s: float
    beta: float
    log_likelihood: float
    converged: bool
    n_customers: int
    n_evaluations: int = 0
    hessian: np.ndarray | None = field(default=None, repr=False)

    def __iter__(self) -> Iterator[float]:
        r"""Yield :math:`(r, \alpha, s, \beta)` in the paper's order."""
        yield from (self.r, self.alpha, self.s, self.beta)

    def as_dict(self) -> dict[str, float]:
        """The four parameters, as keyword arguments for the ``pnbd`` functions."""
        return {"r": self.r, "alpha": self.alpha, "s": self.s, "beta": self.beta}

    @property
    def mean_purchase_rate(self) -> float:
        r""":math:`r/\alpha` -- "the mean transaction [rate]" of S6.2.1.

        >>> from clvtools import ClvData, load_apparel_trans
        >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
        >>> fit = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"])
        >>> round(fit.mean_purchase_rate, 3)
        0.03
        """
        return self.r / self.alpha

    @property
    def mean_attrition_rate(self) -> float:
        r""":math:`s/\beta` -- "the mean attrition rate" of S6.2.1."""
        return self.s / self.beta

    @property
    def n_parameters(self) -> int:
        return 4

    @property
    def aic(self) -> float:
        """Akaike information criterion, as reported by ``summary()`` in S6.2.1."""
        return 2 * self.n_parameters - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion, as reported by ``summary()``."""
        return self.n_parameters * np.log(self.n_customers) - 2 * self.log_likelihood

    @property
    def names(self) -> list[str]:
        r""":math:`(r, \alpha, s, \beta)`, the order everything else uses."""
        return list(_NAMES)


def fit_pnbd(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    weights: ArrayLike | None = None,
    start: tuple[float, float, float, float] = _DEFAULT_START,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    options: dict | None = None,
) -> PnbdParams:
    r"""Maximise the sample log-likelihood over :math:`(r, \alpha, s, \beta)`.

    Parameters
    ----------
    x, t_x, T
        Per-customer frequency, recency and observation window, as produced by
        :meth:`ClvData.customer_summary <clvtools.data.ClvData.customer_summary>`.
    weights
        Row multiplicities, for compressed customer tables.
    start
        Starting :math:`(r, \alpha, s, \beta)`; all ones, as CLVTools does.
    method
        Any method ``scipy.optimize.minimize`` accepts. S6.2.1 recommends
        ``"Nelder-Mead"`` when ``"L-BFGS-B"`` struggles.
    options
        Extra options for the optimiser, merged over the tightened defaults in
        ``_METHOD_OPTIONS``. S6.2.1 offers the same escape hatch through
        ``optimx.args``: "If questions on optimality arise, the parameter
        ``optimx.args`` allows the optimization routine to be controlled."
    hessian
        Whether to compute the Hessian for standard errors. S6.4.1: "the
        calculation of the Hessian matrix can be skipped to decrease runtime.
        The variance-covariance matrix is then not available, including
        statistics based on it such as standard deviation and p-values."

    Returns
    -------
    PnbdParams

    Examples
    --------
    S6.2.1 fits the apparel cohort and prints
    ``r = 1.4490, alpha = 48.6361, s = 0.5613, beta = 46.8844``:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
    >>> fit = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"])
    >>> [round(v, 3) for v in fit]
    [1.449, 48.635, 0.561, 46.88...]

    The last digit of ``beta`` is elided because it is not the same on every
    platform: the ridge S6.2.1 warns about moves the estimate by about 3e-4
    between macOS/ARM and x86-64 Linux for a change in the log-likelihood
    below 1e-9. Every digit printed is one both agree on.
    >>> fit.converged
    True

    That is ``alpha = 48.635`` against the paper's ``48.6361`` -- the two
    differ by 3e-5 relative. The reason is visible in the log-likelihood, which
    agrees far past the point where the parameters do:

    >>> round(fit.log_likelihood, 4)
    -5848.0978

    The Pareto/NBD likelihood has a long flat ridge, and this fit sits at a
    point on it fractionally *better* than the published one -- so the
    difference is where each optimiser stopped, not a discrepancy in the
    likelihood being maximised:

    >>> from clvtools.pnbd import log_likelihood
    >>> published = log_likelihood(cbs["x"], cbs["t_x"], cbs["T"],
    ...                            1.4490, 48.6361, 0.5613, 46.8844)
    >>> bool(fit.log_likelihood >= published)
    True
    >>> bool(fit.log_likelihood - published < 1e-6)
    True

    S6.2.1 also reads off the two average rates, 0.030 and 0.012:

    >>> round(fit.mean_purchase_rate, 3), round(fit.mean_attrition_rate, 3)
    (0.03, 0.012)
    """
    x, t_x, T = (np.asarray(v, dtype=float).ravel() for v in (x, t_x, T))
    x, t_x, T = customer_history(x, t_x, T)

    start_arr = np.asarray(start, dtype=float)
    if start_arr.shape != (4,):
        raise ValueError("start must give four values (r, alpha, s, beta)")
    if np.any(start_arr <= 0):
        raise ValueError("start values must be strictly positive")

    w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    evaluations = 0

    def negative_ll(log_params: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        r, alpha, s, beta = np.exp(log_params)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(x, t_x, T, r, alpha, s, beta, weights=w)
        # A parameter vector that produces a non-finite value is reported as
        # +inf: strictly worse than any real point, so the search moves away
        # rather than propagating a nan into the optimiser's state.
        return np.inf if not np.isfinite(value) else -value

    result = optimize.minimize(
        negative_ll,
        x0=np.log(start_arr),
        method=method,
        options=options_for(method, maxiter, np.log(start_arr), options),
    )
    result = finished(result, "Pareto/NBD")

    hess = None
    if hessian:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # The Hessian of the log-likelihood in the *natural* parameters is
            # what standard errors refer to, so it is differenced there rather
            # than in the log coordinates the search used.
            natural = np.exp(result.x)
            hess = numerical_hessian(
                lambda p: -log_likelihood(x, t_x, T, *p, weights=w), natural
            )

    r, alpha, s, beta = (float(v) for v in np.exp(result.x))
    return PnbdParams(
        r=r,
        alpha=alpha,
        s=s,
        beta=beta,
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_customers=int(x.size if w is None else w.sum()),
        n_evaluations=evaluations,
        hessian=hess,
    )
