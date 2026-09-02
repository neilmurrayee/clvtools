r"""The GGom/NBD model of Bemmaor & Glady, the other alternative in Table 4.

S6.2.1: "As an alternative to the Pareto/NBD model, \pkg{CLVTools} features the
Beta-Geometric/NBD model and the Gamma-Gompertz/NBD model." S3.2 points to
Bemmaor and Glady (2012) for the derivation.

Table 4 gives the shape: the transaction process is Poisson with gamma
heterogeneity, as in the other two, but the lifetime is **Gompertz** with gamma
heterogeneity rather than exponential (Pareto/NBD) or geometric (BG/NBD). A
Gompertz hazard rises exponentially with age, so a customer's chance of dropping
out grows the longer they have been a customer -- where the Pareto/NBD's
exponential lifetime is memoryless and gives a constant hazard.

That extra flexibility costs a fifth parameter and a closed form. The likelihood
holds an integral with no elementary antiderivative, so it is evaluated
numerically here, as CLVTools does with GSL. Parameters are
:math:`(r, \alpha, b, s, \beta)`.

The Pareto/NBD sits at the boundary :math:`b \to 0`, where the Gompertz hazard
stops growing and the lifetime becomes exponential. On the apparel data the
fitted :math:`b` is 8.1e-07 and the two models reach log-likelihoods of
-5848.097871 and -5848.097827 -- the same to four decimal places, on five
parameters against four. The extra flexibility buys nothing here, which
``tests/test_ggomnbd.py`` records.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special

# `ClvDataStaticCov` and `StaticCovResult` appear only in annotations, but
# they are imported for real rather than under `TYPE_CHECKING` so that
# `typing.get_type_hints()` resolves them -- `py.typed` promises the
# signatures are usable downstream. Neither module reaches back here, so
# this closes no cycle; the covariate fit still imports
# `fit_static_covariates` inside the function, where it is needed.
from clvtools._optimize import options_for
from clvtools._staticcov import DelegatesToCovariates, StaticCovResult
from clvtools._validate import customer_history, finished
from clvtools.data import ClvDataStaticCov
from clvtools.inference import Fitted, numerical_hessian

__all__ = [
    "GgomnbdParams",
    "GgomnbdStaticCovParams",
    "alpha_i",
    "beta_i",
    "conditional_expected_transactions",
    "expectation",
    "fit_ggomnbd",
    "fit_ggomnbd_staticcov",
    "log_likelihood",
    "log_likelihood_ind",
    "log_likelihood_staticcov",
    "pmf",
    "probability_alive",
]

#: Quadrature settings for the per-customer integrals. CLVTools allocates a
#: GSL workspace of 1000 intervals for the same integrands.
_QUAD = {"limit": 1000, "epsabs": 1e-12, "epsrel": 1e-10}


def _integrate(fn, lower: NDArray, upper: NDArray) -> NDArray[np.float64]:
    """Integrate ``fn(y, i)`` over ``[lower[i], upper[i]]`` for each customer."""
    out = np.empty(lower.shape, dtype=float)
    flat_lo, flat_hi = lower.ravel(), upper.ravel()
    for i in range(flat_lo.size):
        out.ravel()[i] = integrate.quad(
            fn, flat_lo[i], flat_hi[i], args=(i,), **_QUAD
        )[0]
    return out


def _as_arrays(x, t_x, T, alpha, beta):
    x, t_x, T = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, t_x, T))
    )
    return (
        x, t_x, T,
        np.broadcast_to(np.asarray(alpha, dtype=float), x.shape).ravel(),
        np.broadcast_to(np.asarray(beta, dtype=float), x.shape).ravel(),
    )


def log_likelihood_ind(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: ArrayLike, b: float, s: float, beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""Per-customer log-likelihood.

    Two terms, summed in log space:

    .. math::
        \log L_1 &= \ln\frac{\Gamma(r{+}x)}{\Gamma(r)}
            + r\log\frac{\alpha}{\alpha+T} - x\log(\alpha+T)
            + s\log\frac{\beta}{\beta - 1 + e^{bT}} \\
        \log L_2 &= \ln\frac{\Gamma(r{+}x)}{\Gamma(r)}
            + \log b + r\log\alpha + \log s + s\log\beta
            + \log \int_{t_x}^{T} \frac{e^{by}}
              {(y+\alpha)^{r+x}(\beta - 1 + e^{by})^{s+1}} \, dy

    :math:`L_1` is the customer surviving past :math:`T`; :math:`L_2` integrates
    over the instant they dropped out, which must lie between the last purchase
    and the end of the window. The integral has no elementary form -- that is
    the price of the Gompertz lifetime -- and is taken numerically.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([6, 2, 0]); t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T = np.full(3, 104.0)
    >>> np.round(log_likelihood_ind(x, t_x, T, 1.4490483109712962, 48.634490201568624,
    ...     8.127624052936936e-07, 0.5599392548541099, 3.7970719927076935e-05), 4)
    array([-24.8701, -11.0851,  -1.0347])
    """
    x, t_x, T, alpha_flat, beta_flat = _as_arrays(x, t_x, T, alpha, beta)
    shape = x.shape
    x_flat, t_x_flat, T_flat = x.ravel(), t_x.ravel(), T.ravel()

    shared = special.gammaln(r + x_flat) - special.gammaln(r)

    log_l1 = shared + (
        r * (np.log(alpha_flat) - np.log(alpha_flat + T_flat))
        - x_flat * np.log(alpha_flat + T_flat)
        + s * (np.log(beta_flat) - np.log(beta_flat - 1.0 + np.exp(b * T_flat)))
    )

    def log_integrand(y: float, i: int) -> float:
        return (
            -(r + x_flat[i]) * np.log(y + alpha_flat[i])
            - (s + 1.0) * np.log(beta_flat[i] - 1.0 + np.exp(b * y))
            + b * y
        )

    # Scaled by the integrand's value at the lower limit, where it is largest:
    # ``(alpha + y)^-(r + x)`` decreases in ``y``, and ``b`` is small enough on
    # real data that the other factor is nearly flat. Integrating
    # ``exp(log f(y) - log f(t_x))`` therefore integrates something that starts
    # at 1 and decays, and the offset goes back on in log space afterwards.
    #
    # Unscaled, this underflowed: at the apparel fit's parameters
    # ``(r + x) log(y + alpha)`` is about 808 by ``x = 160``, ``exp(-808)`` is
    # exactly 0, its log is ``-inf``, and ``logaddexp`` then returned the alive
    # branch alone -- ``PAlive`` exactly 1.0 for a heavy buyer, with no warning.
    # On daily data, where ``T`` runs to a thousand, that starts around
    # ``x = 105``. Finding 4 of ``docs/review-2026-09-02.md``.
    offset = np.array(
        [log_integrand(t_x_flat[i], i) for i in range(t_x_flat.size)]
    )

    def scaled(y: float, i: int) -> float:
        return np.exp(log_integrand(y, i) - offset[i])

    integrals = _integrate(scaled, t_x_flat, T_flat)
    with np.errstate(divide="ignore"):
        log_l2 = shared + (
            np.log(b) + r * np.log(alpha_flat) + np.log(s)
            + s * np.log(beta_flat) + offset + np.log(integrals)
        )

    return np.logaddexp(log_l1, log_l2).reshape(shape)


def log_likelihood(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: ArrayLike, b: float, s: float, beta: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    """The sample log-likelihood."""
    ll = log_likelihood_ind(x, t_x, T, r, alpha, b, s, beta)
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


def probability_alive(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: ArrayLike, b: float, s: float, beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""``PAlive`` -- the survival term of the likelihood over the whole of it.

    >>> import numpy as np
    >>> x = np.array([6, 2, 0]); t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T = np.full(3, 104.0)
    >>> np.round(probability_alive(x, t_x, T, 1.4490483109712962, 48.634490201568624,
    ...     8.127624052936936e-07, 0.5599392548541099, 3.7970719927076935e-05), 4)
    array([0.9469, 0.9826, 0.2785])
    """
    x, t_x, T, alpha_flat, beta_flat = _as_arrays(x, t_x, T, alpha, beta)
    shape = x.shape
    x_flat, T_flat = x.ravel(), T.ravel()

    p1 = special.gammaln(r + x_flat) - special.gammaln(r)
    p2 = (
        r * np.log(alpha_flat / (alpha_flat + T_flat))
        + x_flat * np.log(1.0 / (alpha_flat + T_flat))
        + s * np.log(
            beta_flat / (beta_flat - 1.0 + np.exp(b * T_flat))
        )
    )
    p3 = log_likelihood_ind(x, t_x, T, r, alpha, b, s, beta).ravel()
    return np.exp(p1 + p2 - p3).reshape(shape)


def _hyp2f1_1_s_splus1(s: float, z: NDArray[np.float64]) -> NDArray[np.float64]:
    r""":math:`{}_2F_1(1, s; s+1; z)` by its integral representation.

    .. math::
        {}_2F_1(1, s; s+1; z) = \frac{1}{B(1,s)}
            \int_0^1 \frac{t^{s-1}}{1 - zt} \, dt

    CLVTools evaluates it this way too, noting that the direct hypergeometric
    is "problematic" for the arguments ``CET`` produces and that the alternative
    Pfaff transformations would each need their own case analysis. The integral
    needs none, and ``CET`` is not performance critical.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape, dtype=float)
    for i, z_i in enumerate(z.ravel()):
        value, _ = integrate.quad(
            lambda t, z_i=z_i: t ** (s - 1.0) / (1.0 - z_i * t), 0.0, 1.0, **_QUAD
        )
        out.ravel()[i] = value / special.beta(1.0, s)
    return out


def conditional_expected_transactions(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike, t: float,
    r: float, alpha: ArrayLike, b: float, s: float, beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""``CET`` -- expected transactions in the next :math:`t` periods.

    >>> import numpy as np
    >>> x = np.array([6, 2, 0]); t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T = np.full(3, 104.0)
    >>> np.round(conditional_expected_transactions(
    ...     x, t_x, T, 52.0, 1.4490483109712962, 48.634490201568624,
    ...     8.127624052936936e-07, 0.5599392548541099, 3.7970719927076935e-05), 4)
    array([2.2051, 1.0595, 0.1261])
    """
    x, t_x, T, alpha_flat, beta_flat = _as_arrays(x, t_x, T, alpha, beta)
    shape = x.shape
    x_flat, t_x_flat, T_flat = x.ravel(), t_x.ravel(), T.ravel()

    beta_minus_1 = beta_flat - 1.0
    at_T = np.exp(b * T_flat) + beta_minus_1
    at_T_plus = np.exp(b * (T_flat + t)) + beta_minus_1

    upper = _hyp2f1_1_s_splus1(s, beta_minus_1 / at_T) - (
        at_T / at_T_plus
    ) ** s * _hyp2f1_1_s_splus1(s, beta_minus_1 / at_T_plus)

    def log_integrand(tau: float, i: int) -> float:
        return (
            b * tau
            - (s + 1.0) * np.log(np.exp(b * tau) + beta_minus_1[i])
            - (r + x_flat[i]) * np.log(alpha_flat[i] + tau)
        )

    # The same rescaling as the likelihood's, and here it fixes a worse
    # failure. ``(alpha + T)^(r + x)`` *overflows* to ``inf`` at about the same
    # frequency at which the integral *underflows* to 0, so the product was
    # ``inf * 0`` -- NaN, returned as a conditional expectation, from x = 140 on
    # the apparel fit. Finding 4 of ``docs/review-2026-09-02.md``.
    offset = np.array(
        [log_integrand(t_x_flat[i], i) for i in range(t_x_flat.size)]
    )

    def scaled(tau: float, i: int) -> float:
        return np.exp(log_integrand(tau, i) - offset[i])

    integral = _integrate(scaled, t_x_flat, T_flat)

    # ``log P`` where ``P = (alpha + T)^(r + x) * at_T^s * integral``: the
    # product that used to be formed from two doomed factors.
    with np.errstate(divide="ignore"):
        log_product = (
            (r + x_flat) * np.log(alpha_flat + T_flat)
            + s * np.log(at_T)
            + offset
            + np.log(integral)
        )

    front = (r + x_flat) / (alpha_flat + T_flat)
    log_front_upper = np.log(front) + np.log(upper)
    log_bs = np.log(b) + np.log(s)

    # ``lower = b*s * (1 + b*s*P)``. Where ``P`` is representable, form it and
    # divide; where it is not, the 1 is negligible beside it and the whole
    # expression is done in logs instead of being allowed to become NaN.
    with np.errstate(over="ignore"):
        product = np.exp(log_product)
    result = np.empty_like(product)
    finite = np.isfinite(product)
    result[finite] = np.exp(
        log_front_upper[finite] - log_bs - np.log1p(np.exp(log_bs) * product[finite])
    )
    result[~finite] = np.exp(
        log_front_upper[~finite] - 2.0 * log_bs - log_product[~finite]
    )
    return result.reshape(shape)


def expectation(
    t: ArrayLike, r: float, alpha: ArrayLike, b: float, s: float, beta: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`E[X(t)]` for a customer with no history.

    .. math::
        E[X(t)] = \frac{r}{\alpha}\left[
            \left(\frac{\beta}{\beta - 1 + e^{bt}}\right)^{s} t
            + b s \beta^{s} \int_0^{t}
              \frac{\tau e^{b\tau}}{(\beta - 1 + e^{b\tau})^{s+1}} \, d\tau
        \right]

    >>> bool(expectation(0.0, 1.4490483109712962, 48.634490201568624,
    ...     8.127624052936936e-07, 0.5599392548541099, 3.7970719927076935e-05) == 0.0)
    True
    """
    t = np.asarray(t, dtype=float)
    alpha_flat = np.broadcast_to(np.asarray(alpha, dtype=float), t.shape).ravel()
    beta_flat = np.broadcast_to(np.asarray(beta, dtype=float), t.shape).ravel()
    t_flat = t.ravel()

    f1 = r / alpha_flat
    f2 = (beta_flat / (beta_flat - 1.0 + np.exp(b * t_flat))) ** s * t_flat
    f3 = b * s * beta_flat**s

    def integrand(tau: float, i: int) -> float:
        return (
            tau * np.exp(b * tau)
            * (beta_flat[i] + np.exp(b * tau) - 1.0) ** (-(s + 1.0))
        )

    f4 = _integrate(integrand, np.zeros_like(t_flat), t_flat)
    return (f1 * (f2 + f3 * f4)).reshape(t.shape)


def pmf(
    k: int, T: ArrayLike,
    r: float, alpha: ArrayLike, b: float, s: float, beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""``P(X(T) = k)`` -- exactly :math:`k` repeat purchases in the window."""
    if k < 0:
        raise ValueError("k must be non-negative")
    k = int(k)
    T = np.asarray(T, dtype=float)
    alpha_flat = np.broadcast_to(np.asarray(alpha, dtype=float), T.shape).ravel()
    beta_flat = np.broadcast_to(np.asarray(beta, dtype=float), T.shape).ravel()
    T_flat = T.ravel()

    log_b = (
        special.gammaln(r) + special.gammaln(k + 1.0)
        - special.gammaln(r + k + 1.0)
    )
    front = np.exp(
        r * np.log(alpha_flat) + s * np.log(beta_flat)
        - np.log(r + k) - log_b
    )
    inner = np.exp(
        k * np.log(T_flat)
        - (k + r) * np.log(T_flat + alpha_flat)
        - s * np.log(np.exp(b * T_flat) + beta_flat - 1.0)
    )

    def integrand(tau: float, i: int) -> float:
        return np.exp(
            k * np.log(tau) + b * tau
            - (r + k) * np.log(tau + alpha_flat[i])
            - (s + 1.0) * np.log(np.exp(b * tau) + beta_flat[i] - 1.0)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        integral = _integrate(integrand, np.zeros_like(T_flat), T_flat)
    return (front * (inner + b * s * integral)).reshape(T.shape)


# -- time-invariant covariates ------------------------------------------------


def alpha_i(
    alpha: float, gamma_trans: ArrayLike, covariates: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`\alpha_i = \alpha \exp(-\boldsymbol{\gamma}_{purch}'\mathbf{x})`."""
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    gamma_trans = np.asarray(gamma_trans, dtype=float)
    if covariates.shape[1] != gamma_trans.size:
        raise ValueError(
            f"{covariates.shape[1]} transaction covariates but "
            f"{gamma_trans.size} parameters"
        )
    return alpha * np.exp(-(covariates @ gamma_trans))


def beta_i(
    beta: float, gamma_life: ArrayLike, covariates: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`\beta_i = \beta \exp(-\boldsymbol{\gamma}_{attr}'\mathbf{x})`.

    Negative, as in the Pareto/NBD and unlike the BG/NBD's :math:`a_i, b_i`:
    :math:`\beta` scales the Gompertz hazard, so it behaves as a rate.
    """
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    gamma_life = np.asarray(gamma_life, dtype=float)
    if covariates.shape[1] != gamma_life.size:
        raise ValueError(
            f"{covariates.shape[1]} attrition covariates but "
            f"{gamma_life.size} parameters"
        )
    return beta * np.exp(-(covariates @ gamma_life))


# -- estimation ---------------------------------------------------------------


@dataclass(frozen=True)
class GgomnbdParams(Fitted):
    r"""A fitted GGom/NBD. Table 4's five parameters :math:`(r, \alpha, \beta, b, s)`."""

    r: float
    alpha: float
    b: float
    s: float
    beta: float
    log_likelihood: float
    converged: bool
    n_customers: int
    hessian: np.ndarray | None = field(default=None, repr=False)

    @property
    def names(self) -> list[str]:
        return ["r", "alpha", "b", "s", "beta"]

    def __iter__(self) -> Iterator[float]:
        yield from (self.r, self.alpha, self.b, self.s, self.beta)

    def as_dict(self) -> dict[str, float]:
        return {
            "r": self.r, "alpha": self.alpha,
            "b": self.b, "s": self.s, "beta": self.beta,
        }


def fit_ggomnbd(
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    weights: ArrayLike | None = None,
    start: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    options: dict | None = None,
) -> GgomnbdParams:
    r"""Maximise the sample log-likelihood over :math:`(r, \alpha, b, s, \beta)`.

    Slower than the other two families by a wide margin: every evaluation takes
    one numerical integral per customer.
    """
    x, t_x, T = (np.asarray(v, dtype=float).ravel() for v in (x, t_x, T))
    x, t_x, T = customer_history(x, t_x, T)

    start_arr = np.asarray(start, dtype=float)
    if start_arr.shape != (5,):
        raise ValueError("start must give five values (r, alpha, b, s, beta)")
    if np.any(start_arr <= 0):
        raise ValueError("start values must be strictly positive")

    w = None if weights is None else np.asarray(weights, dtype=float).ravel()

    def negative_ll(log_params: np.ndarray) -> float:
        r_, alpha_, b_, s_, beta_ = np.exp(log_params)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(x, t_x, T, r_, alpha_, b_, s_, beta_, weights=w)
        return np.inf if not np.isfinite(value) else -value

    result = optimize.minimize(
        negative_ll, x0=np.log(start_arr), method=method,
        options=options_for(method, maxiter, np.log(start_arr), options),
    )
    result = finished(result, "GGom/NBD")
    r_, alpha_, b_, s_, beta_ = (float(v) for v in np.exp(result.x))
    hess = None
    if hessian:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hess = numerical_hessian(
                lambda v: -log_likelihood(x, t_x, T, *v, weights=w),
                np.array([r_, alpha_, b_, s_, beta_]),
            )
    return GgomnbdParams(
        r=r_, alpha=alpha_, b=b_, s=s_, beta=beta_,
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_customers=int(x.size if w is None else w.sum()),
        hessian=hess,
    )


# -- estimation with time-invariant covariates --------------------------------


# The signature every family's covariate likelihood shares -- data, model
# parameters, both gamma vectors, both design matrices. The GGom/NBD is the
# one family with five model parameters rather than four, which is the whole
# of why it lands one argument over the limit.


def log_likelihood_staticcov(  # noqa: PLR0913, PLR0917
    x: ArrayLike, t_x: ArrayLike, T: ArrayLike,
    r: float, alpha: float, b: float, s: float, beta: float,
    gamma_life: ArrayLike, gamma_trans: ArrayLike,
    cov_life: ArrayLike, cov_trans: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    r"""The sample log-likelihood with time-invariant covariates.

    Both rate parameters take a negative exponent, as in the Pareto/NBD -- see
    :func:`beta_i`. Only the BG/NBD differs, because its attrition parameters
    are beta shapes rather than rates.
    """
    ll = log_likelihood_ind(
        x, t_x, T,
        r=r,
        alpha=alpha_i(alpha, gamma_trans, cov_trans),
        b=b, s=s,
        beta=beta_i(beta, gamma_life, cov_life),
    )
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


@dataclass(frozen=True)
class GgomnbdStaticCovParams(DelegatesToCovariates):
    r"""A fitted GGom/NBD with time-invariant covariates."""

    r: float
    alpha: float
    b: float
    s: float
    beta: float
    covariates: StaticCovResult

    def __iter__(self) -> Iterator[float]:
        yield from (self.r, self.alpha, self.b, self.s, self.beta)
        yield from self.covariates.covariate_values()

    @property
    def names(self) -> list[str]:
        return ["r", "alpha", "b", "s", "beta", *self.covariates.names]


def fit_ggomnbd_staticcov(
    data: ClvDataStaticCov,
    names_cov_constr: list[str] | None = None,
    reg_lambdas: tuple[float, float] | None = None,
    start: tuple[float, float, float, float, float] | None = None,
    start_cov: float | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    polish: bool = True,
    options: dict | None = None,
) -> GgomnbdStaticCovParams:
    r"""Estimate the GGom/NBD with time-invariant covariates.

    Slow: every likelihood evaluation takes one numerical integral per customer,
    and a covariate fit needs several hundred of them.

    Examples
    --------
    >>> from clvtools import ClvData, ClvDataStaticCov
    >>> from clvtools import load_apparel_static_cov, load_apparel_trans
    >>> data = ClvDataStaticCov(
    ...     ClvData(load_apparel_trans(), estimation_split=104),
    ...     load_apparel_static_cov(),
    ...     names_cov_life=["Gender", "Channel"],
    ...     names_cov_trans=["Gender", "Channel"])
    >>> from clvtools.ggomnbd import log_likelihood_staticcov
    >>> cbs = data.customer_summary()
    >>> round(log_likelihood_staticcov(
    ...     cbs["x"], cbs["t_x"], cbs["T"],
    ...     r=1.8378, alpha=92.9124, b=1e-06, s=0.5920, beta=4.9e-05,
    ...     gamma_life=[-0.6430, 0.7907], gamma_trans=[0.2859, 0.6241],
    ...     cov_life=data.design_life(), cov_trans=data.design_trans()), 3)
    -5821.067
    """
    from clvtools._staticcov import SearchSettings, fit_static_covariates

    cbs = data.customer_summary()

    def objective(model, g_life, g_trans, cov_life, cov_trans):
        return log_likelihood_staticcov(
            cbs["x"], cbs["t_x"], cbs["T"],
            model[0], model[1], model[2], model[3], model[4],
            g_life, g_trans, cov_life, cov_trans,
        )

    result = fit_static_covariates(
        x=cbs["x"], t_x=cbs["t_x"], T=cbs["T"],
        cov_life=data.design_life(), cov_trans=data.design_trans(),
        names_cov_life=data.names_cov_life,
        names_cov_trans=data.names_cov_trans,
        log_likelihood=objective,
        n_model_params=5, model_start=(1.0, 1.0, 1.0, 1.0, 1.0),
        names_cov_constr=names_cov_constr,
        search=SearchSettings(
            start=start, start_cov=start_cov, reg_lambdas=reg_lambdas,
            method=method, maxiter=maxiter, hessian=hessian,
            polish=polish, options=options,
        ),
    )
    r_, alpha_, b_, s_, beta_ = (float(v) for v in result.model)
    return GgomnbdStaticCovParams(
        r=r_, alpha=alpha_, b=b_, s=s_, beta=beta_, covariates=result
    )
