r"""S3.4 - correlating the transaction and attrition processes.

S3.4: "All models discussed so far assume independence between mean transaction
and attrition rates. This may not be a realistic assumption. Thus, we use
Sarmanov distributions to correlate the mean transaction and mean attrition
rates."

The Sarmanov family writes a joint density as the independent product plus a
term that induces dependence, eq. (11):

.. math::
    g(\lambda,\mu \mid \alpha,r,\beta,s,m)
      = g(\lambda \mid r,\alpha) g(\mu \mid s,\beta)
      + m \left(\frac{\alpha}{1+\alpha}\right)^{r}
          \left(\frac{\beta}{1+\beta}\right)^{s}
        \big[(g(\lambda \mid r,\alpha{+}1) - g(\lambda \mid r,\alpha))
             (g(\mu \mid s,\beta{+}1) - g(\mu \mid s,\beta))\big]

S3.4 reads the sign off directly: "When :math:`m` is positive, the formula
increases probability density in regions where both variables are similarly
positioned (both high or both low) and decreases it elsewhere."

Because the dependence term is a product of shifted gammas, the likelihood
inherits the same shape -- eq. (12):

.. math::
    L(\alpha,r,\beta,s,m) = \tilde{L}(\alpha,\beta)
      + m \left(\frac{\alpha}{1+\alpha}\right)^{r}
          \left(\frac{\beta}{1+\beta}\right)^{s}
        \big[\tilde{L}(\alpha{+}1,\beta{+}1) - \tilde{L}(\alpha{+}1,\beta)
             - \tilde{L}(\alpha,\beta{+}1) + \tilde{L}(\alpha,\beta)\big]

so nothing new has to be derived: the correlated likelihood is four evaluations
of the uncorrelated one.

S3.4 warns that ``m`` is not itself a correlation -- "this coefficient must not
be directly interpreted as a correlation coefficient" -- and gives eq. (13) to
convert it. CLVTools reports the converted value as ``Cor(life,trans)``, and
:func:`correlation_coefficient` and :func:`m_from_correlation` move between the
two.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from clvtools._optimize import options_for
from clvtools._validate import finished, start_values
from clvtools.inference import Fitted, numerical_hessian
from clvtools.pnbd.aggregate import log_likelihood_ind

__all__ = [
    "PnbdCorrelatedParams",
    "correlated_log_likelihood",
    "correlated_log_likelihood_ind",
    "correlation_bounds",
    "correlation_coefficient",
    "fit_pnbd_correlated",
    "m_from_correlation",
]


def _laplace(r: float, alpha: float, s: float, beta: float) -> tuple[float, float]:
    r"""The two Laplace-transform factors of eqs. (11) and (12).

    :math:`L_A = (\alpha/(1+\alpha))^r` and :math:`L_B = (\beta/(1+\beta))^s`,
    both in :math:`(0, 1)`.
    """
    return (alpha / (1.0 + alpha)) ** r, (beta / (1.0 + beta)) ** s


def correlation_bounds(
    r: float, alpha: float, s: float, beta: float
) -> tuple[float, float]:
    r"""The interval of ``m`` for which eq. (11) stays a density.

    .. math::
        m \in \left[\frac{-1}{\max(L_A L_B,\, (1-L_A)(1-L_B))},\;
                    \frac{1}{\max(L_A(1-L_B),\, (1-L_A)L_B)}\right]

    Outside it the Sarmanov density goes negative somewhere. The bounds depend
    on the other four parameters, so they move during estimation.

    Examples
    --------
    >>> lo, hi = correlation_bounds(1.4490, 48.6361, 0.5613, 46.8844)
    >>> lo < 0 < hi
    True
    >>> round(lo, 3), round(hi, 3)
    (-1.042, 34.822)
    """
    la, lb = _laplace(r, alpha, s, beta)
    upper = 1.0 / max(la * (1.0 - lb), (1.0 - la) * lb)
    lower = -1.0 / max(la * lb, (1.0 - la) * (1.0 - lb))
    return float(lower), float(upper)


def correlation_coefficient(
    m: float, r: float, alpha: float, s: float, beta: float
) -> float:
    r"""Eq. (13) -- the correlation ``m`` implies.

    .. math::
        p_m = m \frac{\sqrt{r}}{1+\alpha}
              \left(\frac{\alpha}{1+\alpha}\right)^{r}
              \frac{\sqrt{s}}{1+\beta}
              \left(\frac{\beta}{1+\beta}\right)^{s}

    This is what CLVTools prints as ``Cor(life,trans)``, and what S6.5.2 tells
    the reader to interpret: "If the correlation is positive and significant,
    customers with a higher (lower) transaction rate are more (less) likely to
    churn."

    >>> round(correlation_coefficient(0.5, 1.4490, 48.6361, 0.5613, 46.8844), 6)
    0.000182
    """
    la, lb = _laplace(r, alpha, s, beta)
    return float(
        m * (np.sqrt(r) / (1.0 + alpha)) * la * (np.sqrt(s) / (1.0 + beta)) * lb
    )


def m_from_correlation(
    p_m: float, r: float, alpha: float, s: float, beta: float
) -> float:
    """Invert eq. (13), recovering ``m`` from a reported correlation.

    Needed to evaluate the likelihood at a fit reported in CLVTools' terms,
    since eq. (12) is written in ``m`` while ``Cor(life,trans)`` is ``p_m``.

    >>> import numpy as np
    >>> args = (1.4490, 48.6361, 0.5613, 46.8844)
    >>> p = correlation_coefficient(0.37, *args)
    >>> bool(np.isclose(m_from_correlation(p, *args), 0.37))
    True
    """
    scale = correlation_coefficient(1.0, r, alpha, s, beta)
    if scale == 0.0:
        raise ValueError("the correlation is identically zero at these parameters")
    return float(p_m / scale)


def correlated_log_likelihood_ind(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: float,
    s: float,
    beta: float,
    m: float,
) -> NDArray[np.float64]:
    r"""Eq. (12), per customer.

    Four evaluations of the uncorrelated likelihood, at
    :math:`(\alpha,\beta)`, :math:`(\alpha{+}1,\beta)`,
    :math:`(\alpha,\beta{+}1)` and :math:`(\alpha{+}1,\beta{+}1)`, combined as
    the Sarmanov construction prescribes.

    Examples
    --------
    ``m = 0`` removes the dependence term entirely and recovers the
    uncorrelated model, as eq. (12) requires:

    >>> import numpy as np
    >>> from clvtools.pnbd import log_likelihood_ind
    >>> args = ([6, 2, 0], [93.285714, 99.571429, 0.0], [104.0] * 3)
    >>> params = (1.4490, 48.6361, 0.5613, 46.8844)
    >>> correlated = correlated_log_likelihood_ind(*args, *params, m=0.0)
    >>> plain = log_likelihood_ind(*args, *params)
    >>> bool(np.allclose(correlated, plain))
    True
    """
    lo, hi = correlation_bounds(r, alpha, s, beta)
    if not lo <= m <= hi:
        raise ValueError(
            f"m = {m:.6g} is outside [{lo:.6g}, {hi:.6g}], where the Sarmanov "
            "density of eq. (11) is not a density"
        )

    ll_00 = log_likelihood_ind(x, t_x, T, r, alpha, s, beta)
    if m == 0.0:
        return ll_00

    ll_10 = log_likelihood_ind(x, t_x, T, r, alpha + 1.0, s, beta)
    ll_01 = log_likelihood_ind(x, t_x, T, r, alpha, s, beta + 1.0)
    ll_11 = log_likelihood_ind(x, t_x, T, r, alpha + 1.0, s, beta + 1.0)

    la, lb = _laplace(r, alpha, s, beta)
    combined = np.exp(ll_00) + m * la * lb * (
        np.exp(ll_00) + np.exp(ll_11) - np.exp(ll_10) - np.exp(ll_01)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(combined)


def correlated_log_likelihood(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: float,
    s: float,
    beta: float,
    m: float,
    weights: ArrayLike | None = None,
) -> float:
    """The sample log-likelihood of the correlated model."""
    ll = correlated_log_likelihood_ind(x, t_x, T, r, alpha, s, beta, m)
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


@dataclass(frozen=True)
class PnbdCorrelatedParams(Fitted):
    """A fitted Pareto/NBD with correlated processes."""

    r: float
    alpha: float
    s: float
    beta: float
    m: float
    log_likelihood: float
    converged: bool
    n_customers: int
    hessian: np.ndarray | None = field(default=None, repr=False)

    @property
    def names(self) -> list[str]:
        return ["r", "alpha", "s", "beta", "m"]

    def __iter__(self) -> Iterator[float]:
        yield from (self.r, self.alpha, self.s, self.beta, self.m)

    def as_dict(self) -> dict[str, float]:
        """The four uncorrelated parameters, for the plain expressions."""
        return {"r": self.r, "alpha": self.alpha, "s": self.s, "beta": self.beta}

    @property
    def correlation(self) -> float:
        """``Cor(life,trans)`` -- eq. (13) applied to ``m``."""
        return correlation_coefficient(self.m, self.r, self.alpha, self.s, self.beta)


def _validated_start_m(start_m, start: np.ndarray) -> float:
    """S6.5.2's mixing parameter's start value, checked before the search.

    Spec `X-13` asks for a "single-value/NA/[-1,1] check". The last of those
    misreads what ``m`` is: it is **not** a correlation, and
    :func:`correlation_bounds` gives its admissible interval -- ``[-1.042,
    34.822]`` at the paper's own parameters, and moving with them during the
    search. :func:`correlation_coefficient` is the thing that lives in
    ``[-1, 1]``, and CLVTools prints *that* as ``Cor(life,trans)``.

    What is real is the rest. A ``NaN`` reached the objective and came back as
    "the correlated Pareto/NBD objective is not finite at the point the search
    started" -- the model blamed for the argument, which is the fifth place this
    package did that (see `V-01`, `V-02`, `X-14` and `PR-15`). And a start well
    outside the bounds, ``-50``, earned the same message where the bounds
    themselves are computable at the start point and say so exactly.
    """
    try:
        value = float(start_m)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"start_m must be a single number, not {type(start_m).__name__}"
        ) from error
    if not np.isfinite(value):
        raise ValueError(f"start_m must be a finite number, got {start_m!r}")
    lower, upper = correlation_bounds(*start)
    if not lower <= value <= upper:
        raise ValueError(
            f"start_m={value} is outside the Sarmanov density's admissible "
            f"interval at the start parameters, [{lower:.4g}, {upper:.4g}]: "
            f"outside it eq. (11) goes negative somewhere. Note `m` is the "
            f"mixing parameter, not the correlation -- "
            f"`correlation_coefficient` is what lies in [-1, 1]"
        )
    return value


def fit_pnbd_correlated(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    weights: ArrayLike | None = None,
    start: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    start_m: float = 0.0,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    options: dict | None = None,
) -> PnbdCorrelatedParams:
    r"""Estimate :math:`(r, \alpha, s, \beta, m)` jointly.

    S6.5.2: "The argument ``start.param.cor`` allows us to optionally specify a
    starting value for the correlation parameter." The default of 0 starts from
    the uncorrelated model, which is the natural null.

    ``m`` is bounded by :func:`correlation_bounds`, and those bounds move with
    the other four parameters, so the objective returns ``+inf`` outside them
    rather than the search being constrained -- the same device CLVTools uses.

    Examples
    --------
    On the apparel cohort the fitted correlation is small, which is what S6.5.2
    reports: "adding this correlation does indeed have a limited impact on
    predictive accuracy."

    >>> from clvtools import ClvData, load_apparel_trans
    >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
    >>> fit = fit_pnbd_correlated(cbs["x"], cbs["t_x"], cbs["T"])
    >>> 0 < fit.correlation < 0.02
    True

    The value itself is not printed because it is not portable: ``m`` sits
    close to its Sarmanov bound, where the likelihood is nearly flat, so the
    fitted correlation is 0.0106 on macOS/ARM and 0.006 on x86-64 Linux. Both
    say what S6.5.2 says -- "a limited impact" -- and neither digit is a fact
    about the model.

    The correlated model nests the uncorrelated one at ``m = 0``, so its
    optimum can never be worse:

    >>> from clvtools.pnbd import log_likelihood
    >>> plain = log_likelihood(cbs["x"], cbs["t_x"], cbs["T"],
    ...                        1.4490, 48.6361, 0.5613, 46.8844)
    >>> bool(fit.log_likelihood >= plain)
    True

    .. note::
       CLVTools 0.12.1 does *not* satisfy that inequality on this data: its
       correlated fit attains -5850.82 against -5848.10 uncorrelated, because
       the search drives ``m`` onto its lower Sarmanov bound and stalls there.
       Evaluated at CLVTools' own reported parameters this implementation
       returns its log-likelihood to 4e-12, so the expressions agree and only
       the optimisation differs. ``tests/test_pnbd_correlation.py`` records
       both facts.
    """
    x, t_x, T = (np.asarray(v, dtype=float).ravel() for v in (x, t_x, T))
    w = None if weights is None else np.asarray(weights, dtype=float).ravel()

    start_arr = start_values(start, count=4, parameters="values (r, alpha, s, beta)")
    start_m = _validated_start_m(start_m, start_arr)

    x0 = np.concatenate([np.log(start_arr), [float(start_m)]])

    def negative_ll(v: np.ndarray) -> float:
        r, alpha, s, beta = np.exp(v[:4])
        m = float(v[4])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            try:
                value = correlated_log_likelihood(
                    x, t_x, T, r, alpha, s, beta, m, weights=w
                )
            except ValueError:
                # m outside the Sarmanov bounds at this (r, alpha, s, beta).
                return np.inf
        return np.inf if not np.isfinite(value) else -value

    result = optimize.minimize(
        negative_ll,
        x0=x0,
        method=method,
        options=options_for(method, maxiter, x0, options),
    )
    result = finished(result, "correlated Pareto/NBD")

    r, alpha, s, beta = (float(v) for v in np.exp(result.x[:4]))

    hess = None
    if hessian:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # As everywhere else, in the *natural* parameters, which is what a
            # standard error refers to. ``m`` is already natural -- the search
            # carries it unlogged, being signed -- so only the first four are
            # exponentiated. Until this existed the fit's Hessian was always
            # None and ``summary()`` raised advising ``hessian=True``, which
            # was not an argument this function had: finding 8 of
            # the 2026-09 review.
            natural = np.concatenate([np.exp(result.x[:4]), result.x[4:]])
            hess = numerical_hessian(
                lambda p: -correlated_log_likelihood(
                    x, t_x, T, *p[:4], m=p[4], weights=w
                ),
                natural,
            )

    return PnbdCorrelatedParams(
        r=r,
        alpha=alpha,
        s=s,
        beta=beta,
        m=float(result.x[4]),
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_customers=int(x.size if w is None else w.sum()),
        hessian=hess,
    )
