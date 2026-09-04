r"""The BG/NBD model of Fader, Hardie & Lee, one of the alternatives in Table 4.

S6.2.1: "As an alternative to the Pareto/NBD model, \pkg{CLVTools} features the
Beta-Geometric/NBD model and the Gamma-Gompertz/NBD model."

S3.2 gives the reference rather than the derivation: "The derivations for the
Beta-Geometric/NBD (BG/NBD) model are found in Fader et al. (2005)". Table 4
gives its shape: the transaction process is Poisson with gamma heterogeneity, as
in the Pareto/NBD, but attrition is *geometric* with beta heterogeneity rather
than exponential with gamma.

The behavioural difference is where dropout can happen. The Pareto/NBD lets a
customer die at any instant; the BG/NBD only lets them drop out immediately
*after* a transaction, with probability :math:`p`. A customer with no repeat
purchases has had no opportunity to drop out, which is why :math:`x = 0` appears
as an indicator throughout rather than as a limit.

Parameters are :math:`(r, \alpha, a, b)`: :math:`r, \alpha` the gamma shape and
rate for the purchase rate, :math:`a, b` the beta parameters for the dropout
probability.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special

# `ClvDataStaticCov` and `StaticCovResult` appear only in annotations, but
# they are imported for real rather than under `TYPE_CHECKING` so that
# `typing.get_type_hints()` resolves them -- `py.typed` promises the
# signatures are usable downstream. Neither module reaches back here, so
# this closes no cycle; the covariate fit still imports
# `fit_static_covariates` inside the function, where it is needed.
from clvtools._optimize import options_for, start_scale
from clvtools._staticcov import DelegatesToCovariates, StaticCovResult, design
from clvtools._validate import customer_history, finished, start_values
from clvtools.data import ClvDataStaticCov
from clvtools.inference import Fitted, numerical_hessian

__all__ = [
    "BgnbdParams",
    "BgnbdStaticCovParams",
    "a_i",
    "alpha_i",
    "b_i",
    "conditional_expected_transactions",
    "expectation",
    "fit_bgnbd",
    "fit_bgnbd_staticcov",
    "log_likelihood",
    "log_likelihood_ind",
    "log_likelihood_staticcov",
    "pmf",
    "probability_alive",
]


def _log_beta_ratio(a1, b1, a2, b2):
    r""":math:`\log[B(a_1,b_1) / B(a_2,b_2)]`, formed in log space throughout.

    The beta functions themselves overflow for the parameter ranges the
    likelihood visits; only their ratio is ever finite.
    """
    return (
        special.gammaln(a1) + special.gammaln(b1) - special.gammaln(a1 + b1)
        - special.gammaln(a2) - special.gammaln(b2) + special.gammaln(a2 + b2)
    )


def _broadcast(x, t_x, T, alpha, a, b):
    x, t_x, T = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, t_x, T))
    )
    return (
        x, t_x, T,
        np.broadcast_to(np.asarray(alpha, dtype=float), x.shape),
        np.broadcast_to(np.asarray(a, dtype=float), x.shape),
        np.broadcast_to(np.asarray(b, dtype=float), x.shape),
    )


def log_likelihood_ind(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: ArrayLike, a: ArrayLike, b: ArrayLike,
) -> NDArray[np.float64]:
    r"""Per-customer log-likelihood.

    .. math::
        L = \frac{\Gamma(r+x)\alpha^{r}}{\Gamma(r)(\alpha+t_x)^{r+x}}
            \left[\frac{B(a, b+x)}{B(a,b)}
                  \left(\frac{\alpha+t_x}{\alpha+T}\right)^{r+x}
                  + \delta_{x>0}\frac{B(a+1, b+x-1)}{B(a,b)}\right]

    The bracket holds the two ways the history could have arisen: the customer
    survived every one of their :math:`x` opportunities to drop out and is still
    alive at :math:`T`, or they dropped out immediately after the last purchase.
    The second term carries :math:`\delta_{x>0}` because a customer with no
    repeat purchase has never had the chance.

    Examples
    --------
    The first three apparel customers at the fitted parameters:

    >>> import numpy as np
    >>> x = np.array([6, 2, 0]); t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T = np.full(3, 104.0)
    >>> np.round(log_likelihood_ind(x, t_x, T, 0.6073, 20.9567, 1.2755, 8.8608), 4)
    array([-25.0813, -10.8877,  -1.0843])
    """
    x, t_x, T, alpha, a, b = _broadcast(x, t_x, T, alpha, a, b)

    part1 = (
        r * np.log(alpha)
        + special.gammaln(r + x)
        - special.gammaln(r)
        - (r + x) * np.log(alpha + t_x)
    )
    survived = np.exp(_log_beta_ratio(a, b + x, a, b)) * (
        (alpha + t_x) / (alpha + T)
    ) ** (r + x)
    dropped = np.where(
        x > 0, np.exp(_log_beta_ratio(a + 1.0, b + x - 1.0, a, b)), 0.0
    )
    return part1 + np.log(survived + dropped)


def log_likelihood(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: ArrayLike, a: ArrayLike, b: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    """The sample log-likelihood.

    >>> from clvtools import ClvData, load_apparel_trans
    >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
    >>> round(log_likelihood(cbs["x"], cbs["t_x"], cbs["T"],
    ...                      0.6073, 20.9567, 1.2755, 8.8608), 3)
    -5857.02
    """
    ll = log_likelihood_ind(x, t_x, T, r, alpha, a, b)
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


def probability_alive(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: ArrayLike, a: ArrayLike, b: ArrayLike,
) -> NDArray[np.float64]:
    r"""``PAlive`` at the end of the estimation period.

    .. math::
        P(\text{alive}) = \left[1 + \delta_{x>0}\frac{a}{b+x-1}
            \left(\frac{\alpha+T}{\alpha+t_x}\right)^{r+x}\right]^{-1}

    A customer with no repeat purchase is alive with certainty: under this
    model dropout only happens after a transaction, so having made none they
    cannot have dropped out.

    >>> float(probability_alive(0, 0.0, 104.0, 0.6073, 20.9567, 1.2755, 8.8608))
    1.0
    """
    x, t_x, T, alpha, a, b = _broadcast(x, t_x, T, alpha, a, b)
    odds = np.where(
        x > 0,
        (a / (b + x - 1.0)) * ((alpha + T) / (alpha + t_x)) ** (r + x),
        0.0,
    )
    return 1.0 / (1.0 + odds)


def conditional_expected_transactions(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike, t: float,
    r: float, alpha: ArrayLike, a: ArrayLike, b: ArrayLike,
) -> NDArray[np.float64]:
    r"""``CET`` -- expected transactions in the next :math:`t` periods.

    .. math::
        E[Y(t) \mid \cdot] = \frac{\frac{a+b+x-1}{a-1}
            \left[1 - \left(\frac{\alpha+T}{\alpha+T+t}\right)^{r+x}
            {}_2F_1\!\left(r{+}x, b{+}x; a{+}b{+}x{-}1;
            \frac{t}{\alpha+T+t}\right)\right]}
            {1 + \delta_{x>0}\frac{a}{b+x-1}
             \left(\frac{\alpha+T}{\alpha+t_x}\right)^{r+x}}

    The denominator is the reciprocal of :func:`probability_alive`, so this is
    the expectation conditional on being alive, weighted by that probability.

    >>> import numpy as np
    >>> x = np.array([6, 2, 0]); t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T = np.full(3, 104.0)
    >>> np.round(conditional_expected_transactions(
    ...     x, t_x, T, 52.0, 0.6073, 20.9567, 1.2755, 8.8608), 4)
    array([2.1021, 0.8824, 0.2428])
    """
    x, t_x, T, alpha, a, b = _broadcast(x, t_x, T, alpha, a, b)
    if np.any(np.isclose(a, 1.0)):
        raise ValueError("CET is undefined at a = 1: the expression divides by (a - 1)")

    term1 = (a + b + x - 1.0) / (a - 1.0)
    term2 = 1.0 - ((alpha + T) / (alpha + T + t)) ** (r + x) * special.hyp2f1(
        r + x, b + x, a + b + x - 1.0, t / (alpha + T + t)
    )
    term3 = 1.0 + np.where(
        x > 0, (a / (b + x - 1.0)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0
    )
    return term1 * term2 / term3


def expectation(
    t: ArrayLike, r: float, alpha: ArrayLike, a: ArrayLike, b: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`E[X(t)]` for a customer with no history.

    .. math::
        E[X(t)] = \frac{a+b-1}{a-1}
            \left[1 - \left(\frac{\alpha}{\alpha+t}\right)^{r}
            {}_2F_1\!\left(r, b; a{+}b{-}1; \frac{t}{\alpha+t}\right)\right]

    >>> bool(expectation(0.0, 0.6073, 20.9567, 1.2755, 8.8608) == 0.0)
    True
    """
    t = np.asarray(t, dtype=float)
    alpha, a, b = (np.asarray(v, dtype=float) for v in (alpha, a, b))
    if np.any(np.isclose(a, 1.0)):
        raise ValueError(
            "the expectation is undefined at a = 1: it divides by (a - 1)"
        )
    term1 = (a + b - 1.0) / (a - 1.0)
    term2 = (alpha / (alpha + t)) ** r
    term3 = special.hyp2f1(r, b, a + b - 1.0, t / (alpha + t))
    return term1 * (1.0 - term2 * term3)


def pmf(
    k: int, T: ArrayLike, r: float, alpha: ArrayLike, a: ArrayLike, b: ArrayLike
) -> NDArray[np.float64]:
    r"""``P(X(T) = k)`` -- exactly :math:`k` repeat purchases in the window.

    The first term covers surviving all :math:`k` dropout opportunities; the
    second covers having dropped out after the :math:`k`-th, which requires
    :math:`k > 0`.

    >>> import numpy as np
    >>> total = sum(float(pmf(k, 104.0, 0.6073, 20.9567, 1.2755, 8.8608))
    ...             for k in range(600))
    >>> bool(np.isclose(total, 1.0, atol=1e-6))
    True
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    k = int(k)
    T = np.asarray(T, dtype=float)
    alpha, a, b = (
        np.broadcast_to(np.asarray(v, dtype=float), T.shape) for v in (alpha, a, b)
    )

    log_part1 = (
        _log_beta_ratio(a, b + k, a, b)
        + special.gammaln(r + k)
        - special.gammaln(r)
        - special.gammaln(k + 1.0)
        + r * (np.log(alpha) - np.log(alpha + T))
        + k * (np.log(T) - np.log(alpha + T))
    )
    part1 = np.exp(log_part1)
    if k == 0:
        return part1

    running = np.zeros_like(part1)
    for j in range(k):
        running += np.exp(
            special.gammaln(r + j)
            - special.gammaln(r)
            - special.gammaln(j + 1.0)
            + j * (np.log(T) - np.log(alpha + T))
        )
    part2 = np.exp(_log_beta_ratio(a + 1.0, b + k - 1.0, a, b)) * (
        1.0 - np.exp(r * (np.log(alpha) - np.log(alpha + T)) + np.log(running))
    )
    return part1 + part2


# -- time-invariant covariates ------------------------------------------------


def alpha_i(
    alpha: float, gamma_trans: ArrayLike, covariates: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`\alpha_i = \alpha \exp(-\boldsymbol{\gamma}_{purch}'\mathbf{x})`.

    Same convention as the Pareto/NBD: a covariate that raises the purchase
    rate lowers the rate parameter.
    """
    covariates, gamma_trans = design(covariates, gamma_trans, "transaction")
    return alpha * np.exp(-(covariates @ gamma_trans))


def _life_scaled(value: float, gamma_life: ArrayLike, covariates: ArrayLike):
    covariates, gamma_life = design(covariates, gamma_life, "attrition")
    return value * np.exp(covariates @ gamma_life)


def a_i(
    a: float, gamma_life: ArrayLike, covariates: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`a_i = a \exp(+\boldsymbol{\gamma}_{attr}'\mathbf{x})`.

    .. note::
       The sign is **positive** here, unlike :func:`alpha_i` and unlike the
       Pareto/NBD's :math:`\beta_i`. :math:`a` and :math:`b` are beta
       parameters, not rates, so raising them raises the dropout probability
       directly rather than through a reciprocal. Both scale by the same factor,
       leaving the beta mean :math:`a/(a+b)` unchanged and its variance reduced.
    """
    return _life_scaled(a, gamma_life, covariates)


def b_i(
    b: float, gamma_life: ArrayLike, covariates: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`b_i = b \exp(+\boldsymbol{\gamma}_{attr}'\mathbf{x})`. See :func:`a_i`."""
    return _life_scaled(b, gamma_life, covariates)


# -- estimation ---------------------------------------------------------------


@dataclass(frozen=True)
class BgnbdParams(Fitted):
    r"""A fitted BG/NBD. Parameters are :math:`(r, \alpha, a, b)`, per Table 4."""

    r: float
    alpha: float
    a: float
    b: float
    log_likelihood: float
    converged: bool
    n_customers: int
    hessian: np.ndarray | None = field(default=None, repr=False)

    @property
    def names(self) -> list[str]:
        return ["r", "alpha", "a", "b"]

    def __iter__(self) -> Iterator[float]:
        yield from (self.r, self.alpha, self.a, self.b)

    def as_dict(self) -> dict[str, float]:
        return {"r": self.r, "alpha": self.alpha, "a": self.a, "b": self.b}


def fit_bgnbd(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    weights: ArrayLike | None = None,
    start: tuple[float, float, float, float] | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    options: dict | None = None,
) -> BgnbdParams:
    r"""Maximise the sample log-likelihood over :math:`(r, \alpha, a, b)`.

    >>> from clvtools import ClvData, load_apparel_trans
    >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
    >>> fit = fit_bgnbd(cbs["x"], cbs["t_x"], cbs["T"])
    >>> fit.converged
    True
    >>> round(fit.log_likelihood, 3)
    -5857.02

    On this data the BG/NBD fits slightly worse than the Pareto/NBD's
    -5848.098, with the same number of parameters:

    >>> bool(fit.log_likelihood < -5848.09)
    True
    """
    x, t_x, T = (np.asarray(v, dtype=float).ravel() for v in (x, t_x, T))
    x, t_x, T = customer_history(x, t_x, T)

    w = None if weights is None else np.asarray(weights, dtype=float).ravel()

    # See :func:`clvtools._optimize.start_scale`. ``a`` and ``b`` are the shapes
    # of a Beta distribution and carry no unit, so ``alpha`` alone is scaled.
    if start is None:
        start_arr = np.array([1.0, start_scale(T, w), 1.0, 1.0])
    else:
        start_arr = start_values(
            start, count=4, parameters="values (r, alpha, a, b)"
        )

    def negative_ll(log_params: np.ndarray) -> float:
        r_, alpha_, a_, b_ = np.exp(log_params)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(x, t_x, T, r_, alpha_, a_, b_, weights=w)
        return np.inf if not np.isfinite(value) else -value

    result = optimize.minimize(
        negative_ll, x0=np.log(start_arr), method=method,
        options=options_for(method, maxiter, np.log(start_arr), options),
    )
    result = finished(result, "BG/NBD")
    r_, alpha_, a_, b_ = (float(v) for v in np.exp(result.x))
    hess = None
    if hessian:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hess = numerical_hessian(
                lambda v: -log_likelihood(x, t_x, T, *v, weights=w),
                np.array([r_, alpha_, a_, b_]),
            )
    return BgnbdParams(
        r=r_, alpha=alpha_, a=a_, b=b_,
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_customers=int(x.size if w is None else w.sum()),
        hessian=hess,
    )


# -- estimation with time-invariant covariates --------------------------------


def log_likelihood_staticcov(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: float, a: float, b: float,
    gamma_life: ArrayLike, gamma_trans: ArrayLike,
    cov_life: ArrayLike, cov_trans: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    r"""The sample log-likelihood with time-invariant covariates.

    Table 4 marks the BG/NBD as supporting them. The rates are built by
    :func:`alpha_i`, :func:`a_i` and :func:`b_i` -- note that the two attrition
    parameters take a *positive* exponent where the transaction parameter takes
    a negative one.
    """
    ll = log_likelihood_ind(
        x, t_x, T,
        r=r,
        alpha=alpha_i(alpha, gamma_trans, cov_trans),
        a=a_i(a, gamma_life, cov_life),
        b=b_i(b, gamma_life, cov_life),
    )
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


@dataclass(frozen=True)
class BgnbdStaticCovParams(DelegatesToCovariates):
    r"""A fitted BG/NBD with time-invariant covariates."""

    r: float
    alpha: float
    a: float
    b: float
    covariates: StaticCovResult

    def __iter__(self) -> Iterator[float]:
        yield from (self.r, self.alpha, self.a, self.b)
        yield from self.covariates.covariate_values()

    @property
    def names(self) -> list[str]:
        return ["r", "alpha", "a", "b", *self.covariates.names]


def fit_bgnbd_staticcov(
    data: ClvDataStaticCov,
    names_cov_constr: list[str] | None = None,
    reg_lambdas: tuple[float, float] | None = None,
    start: tuple[float, float, float, float] | None = None,
    start_cov: float | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    polish: bool = True,
    options: dict | None = None,
) -> BgnbdStaticCovParams:
    r"""Estimate the BG/NBD with time-invariant covariates.

    Table 4 marks the BG/NBD as taking time-invariant covariates, equality
    constraints and regularization -- but not time-varying covariates or
    process correlation, both of which are Pareto/NBD only.

    Examples
    --------
    >>> from clvtools import ClvData, ClvDataStaticCov
    >>> from clvtools import load_apparel_static_cov, load_apparel_trans
    >>> data = ClvDataStaticCov(
    ...     ClvData(load_apparel_trans(), estimation_split=104),
    ...     load_apparel_static_cov(),
    ...     names_cov_life=["Gender", "Channel"],
    ...     names_cov_trans=["Gender", "Channel"])
    >>> fit = fit_bgnbd_staticcov(data, hessian=False)
    >>> fit.names
    ['r', 'alpha', 'a', 'b', 'life.Gender', 'life.Channel',
     'trans.Gender', 'trans.Channel']
    >>> round(fit.log_likelihood, 3)
    -5839.862

    Covariates improve on the BG/NBD without them, which reached -5857.020:

    >>> bool(fit.log_likelihood > -5857.02)
    True
    """
    from clvtools._staticcov import SearchSettings, fit_static_covariates

    cbs = data.customer_summary()

    def objective(model, g_life, g_trans, cov_life, cov_trans):
        return log_likelihood_staticcov(
            cbs["x"], cbs["t_x"], cbs["T"],
            model[0], model[1], model[2], model[3],
            g_life, g_trans, cov_life, cov_trans,
        )

    result = fit_static_covariates(
        x=cbs["x"], t_x=cbs["t_x"], T=cbs["T"],
        cov_life=data.design_life(), cov_trans=data.design_trans(),
        names_cov_life=data.names_cov_life,
        names_cov_trans=data.names_cov_trans,
        log_likelihood=objective,
        n_model_params=4, model_start=(1.0, 1.0, 1.0, 1.0),
        names_cov_constr=names_cov_constr,
        search=SearchSettings(
            start=start, start_cov=start_cov, reg_lambdas=reg_lambdas,
            method=method, maxiter=maxiter, hessian=hessian,
            polish=polish, options=options,
        ),
    )
    r_, alpha_, a_, b_ = (float(v) for v in result.model)
    return BgnbdStaticCovParams(
        r=r_, alpha=alpha_, a=a_, b=b_, covariates=result
    )
