r"""Appendix A and S6.3 - the Pareto/NBD for a randomly chosen customer.

S3.2: "Customers' latent characteristics :math:`\lambda` and :math:`\mu` are
then marginalized over the two Gamma distributions. The result is a rather
intricate closed-form expression from which values for :math:`r`, :math:`\alpha`,
:math:`s`, and :math:`\beta` are estimated using a maximum likelihood approach".

Two forms of that likelihood live here:

:func:`likelihood_appendix`
    Appendix A transcribed as written, with the :math:`A_1` / :math:`A_2` branch
    on :math:`\alpha \gtrless \beta`. Readable, and it overflows for large
    :math:`x`.
:func:`log_likelihood_ind`
    The same quantity rearranged for numerical stability, following Fader &
    Hardie's "A Note on Deriving the Pareto/NBD Model and Related Expressions"
    eq. (18). This is what everything else is built on.

They are held to each other by the tests wherever the first is viable.

The managerial expressions of S6.3 -- ``PAlive``, ``CET``, ``DERT`` -- and the
diagnostics of S6.2.2 -- ``PMF``, the unconditional expectation -- follow.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from clvtools._validate import PrecisionWarning
from clvtools.special import hyp2f1_ratio, kummer_u

__all__ = [
    "conditional_expected_transactions",
    "discounted_expected_residual_transactions",
    "expectation",
    "likelihood_appendix",
    "log_likelihood",
    "log_likelihood_ind",
    "pmf",
    "probability_alive",
]


def _as_arrays(*values: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    return tuple(np.asarray(v, dtype=float) for v in values)


def _broadcast_rates(
    alpha: ArrayLike, beta: ArrayLike, like: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-customer ``alpha`` and ``beta``, broadcast to the shape of ``x``.

    Without covariates every customer shares one :math:`\\alpha` and one
    :math:`\\beta`; with the covariate extension of S3.3 each customer has their
    own. Accepting either here is what lets the static-covariate model reuse
    every expression in this module unchanged.
    """
    alpha_i = np.broadcast_to(np.asarray(alpha, dtype=float), like.shape)
    beta_i = np.broadcast_to(np.asarray(beta, dtype=float), like.shape)
    return alpha_i, beta_i


# -- the likelihood -----------------------------------------------------------


def likelihood_appendix(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: ArrayLike,
    s: float,
    beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""Appendix A, eq. (24), transcribed literally.

    .. math::
        L(r, \alpha, s, \beta \mid x, t_x, T)
          = \frac{\Gamma(r+x)\, \alpha^{r} \beta^{s}}{\Gamma(r)}
            \left(\frac{s}{r+s+x} A_1 + \frac{r+x}{r+s+x} A_2\right)

    with, for :math:`\alpha \ge \beta`,

    .. math::
        A_1 = \frac{{}_2F_1\!\left(r+s+x,\, s+1,\, r+s+x+1;\,
                    \frac{\alpha-\beta}{\alpha+t_x}\right)}{(\alpha+t_x)^{r+s+x}},
        \quad
        A_2 = \frac{{}_2F_1\!\left(r+s+x,\, s,\, r+s+x+1;\,
                    \frac{\alpha-\beta}{\alpha+T}\right)}{(\alpha+T)^{r+s+x}}

    and for :math:`\alpha < \beta` the same with :math:`\alpha` and
    :math:`\beta` exchanged and the second parameter becoming :math:`r+x` and
    :math:`r+x+1` respectively.

    This is kept because it is the form the paper states, and because agreeing
    with it is a real check on :func:`log_likelihood_ind`. It is **not** used
    for fitting: :math:`\Gamma(r+x)` and :math:`(\alpha+t_x)^{r+s+x}` both
    overflow well before ``x`` reaches the largest counts in real data.

    Examples
    --------
    >>> import numpy as np
    >>> from clvtools.pnbd.aggregate import log_likelihood_ind
    >>> args = (6, 93.285714, 104.0, 1.4490, 48.6361, 0.5613, 46.8844)
    >>> bool(np.isclose(np.log(likelihood_appendix(*args)),
    ...                 log_likelihood_ind(*args)))
    True
    """
    x, t_x, T = _as_arrays(x, t_x, T)
    alpha_i, beta_i = _broadcast_rates(alpha, beta, x)

    rsx = r + s + x
    swap = alpha_i < beta_i

    # A_1 and A_2 differ only in which of t_x / T they are evaluated at, and in
    # the 2F1's second parameter.
    b1 = np.where(swap, r + x, s + 1.0)
    b2 = np.where(swap, r + x + 1.0, s)
    hi = np.where(swap, beta_i, alpha_i)
    gap = np.abs(alpha_i - beta_i)

    a1 = hyp2f1_ratio(rsx, b1, gap / (hi + t_x)) / (hi + t_x) ** rsx
    a2 = hyp2f1_ratio(rsx, b2, gap / (hi + T)) / (hi + T) ** rsx

    prefactor = special.gamma(r + x) * alpha_i**r * beta_i**s / special.gamma(r)
    return prefactor * (s / rsx * a1 + (r + x) / rsx * a2)


def log_likelihood_ind(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: ArrayLike,
    s: float,
    beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""Per-customer log-likelihood, in the numerically stable arrangement.

    Writing Appendix A's expression as :math:`L = X \cdot (Y + Z)`,

    .. math::
        \log X &= r\log\alpha + s\log\beta - \ln\Gamma(r) + \ln\Gamma(r+x) \\
        \log Y &= -(r+x)\log(\alpha+T) - s\log(\beta+T) \\
        \log Z &= \log s - \log(r+s+x) + \log A_0

    and summing :math:`Y + Z` with the log-sum-exp trick keeps every term
    representable. :math:`A_0` is in turn split as :math:`a_1 \tilde{A}`, with

    .. math::
        \log a_1 &= -(r+s+x)\log\!\left(\max(\alpha,\beta) + t_x\right) \\
        \tilde{A} &= {}_2F_1\!\left(\cdot;\tfrac{|\alpha-\beta|}{\max+t_x}\right)
          - {}_2F_1\!\left(\cdot;\tfrac{|\alpha-\beta|}{\max+T}\right)
            \left(\frac{\max+t_x}{\max+T}\right)^{r+s+x}

    so the factor that grows with :math:`x` is logged rather than formed.

    Two cases are handled apart:

    * :math:`\alpha = \beta` sends both hypergeometrics to 1 and
      :math:`\tilde{A}` to :math:`1 - (\cdot)^{r+s+x}`;
    * :math:`t_x = T` makes :math:`A_0` vanish, so :math:`\log Z` is undefined
      and the likelihood collapses to :math:`\log X + \log Y`. This is every
      customer with no repeat purchase, so it is the common case, not a corner.

    Examples
    --------
    The first three apparel customers at the fitted parameters of S6.2.1:

    >>> import numpy as np
    >>> x   = np.array([6, 2, 0])
    >>> t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T   = np.array([104.0, 104.0, 104.0])
    >>> ll = log_likelihood_ind(x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844)
    >>> np.round(ll, 4)
    array([-24.8703, -11.0852,  -1.0347])
    """
    x, t_x, T = _as_arrays(x, t_x, T)
    x, t_x, T = np.broadcast_arrays(x, t_x, T)
    alpha_i, beta_i = _broadcast_rates(alpha, beta, x)

    rsx = r + s + x
    hi = np.maximum(alpha_i, beta_i)
    gap = np.abs(alpha_i - beta_i)

    # 2F1's second parameter: s+1 when alpha >= beta, r+x otherwise.
    b = np.where(alpha_i < beta_i, r + x, s + 1.0)

    ratio = (hi + t_x) / (hi + T)
    same_rate = alpha_i == beta_i

    # A_tilde, guarded so the alpha == beta branch never evaluates the 2F1 form
    # (which is correct there too, but wastes work and can round to 0 - 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        a_tilde = np.where(
            same_rate,
            1.0 - ratio**rsx,
            hyp2f1_ratio(rsx, b, gap / (hi + t_x))
            - hyp2f1_ratio(rsx, b, gap / (hi + T)) * ratio**rsx,
        )
        log_a0 = -rsx * np.log(hi + t_x) + np.log(a_tilde)

        log_x = (
            r * np.log(alpha_i)
            + s * np.log(beta_i)
            - special.gammaln(r)
            + special.gammaln(r + x)
        )
        log_y = -(r + x) * np.log(alpha_i + T) - s * np.log(beta_i + T)
        log_z = np.log(s) - np.log(rsx) + log_a0

        ll = log_x + np.logaddexp(log_y, log_z)

    # t_x == T: the customer's window closed on their last purchase, so there is
    # no interval in which they could have died unobserved and Z is exactly 0.
    return np.where(t_x == T, log_x + log_y, ll)


def log_likelihood(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: ArrayLike,
    s: float,
    beta: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    r"""The sample log-likelihood maximised in S3.2.

    .. math::
        \sum_{i=1}^{N} \log L(r, \alpha, s, \beta \mid x_i, t_{x_i}, T_i)

    ``weights`` repeats each row, for the compressed customer tables CLVTools
    builds when many customers share the same :math:`(x, t_x, T)`.

    Examples
    --------
    The apparel cohort at the S6.2.1 estimates reaches the value the oracle
    reports for that fit:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
    >>> round(log_likelihood(cbs["x"], cbs["t_x"], cbs["T"],
    ...                      1.4490, 48.6361, 0.5613, 46.8844), 4)
    -5848.0978
    """
    ll = log_likelihood_ind(x, t_x, T, r, alpha, s, beta)
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


# -- managerial expressions, S6.3 ---------------------------------------------


def probability_alive(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: ArrayLike,
    s: float,
    beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""``PAlive`` -- "the probability of a customer being alive" at :math:`T`.

    S6.3: metric (2) of the three a latent attrition model predicts. It is the
    survival term of the likelihood divided by the whole likelihood:

    .. math::
        P(\omega > T \mid x, t_x, T) = \frac{X \cdot Y}{L}

    in the notation of :func:`log_likelihood_ind`, where :math:`X \cdot Y` is
    exactly the branch in which the customer never died.

    S6.3.2: "``PAlive`` is unaffected by both parameters as it describes
    customers at the end of the estimation period."

    Examples
    --------
    Customer 1 of S6.3.2 has all but abandoned the firm; customer 100 has not:

    >>> import numpy as np
    >>> x   = np.array([6, 2, 0])
    >>> t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T   = np.array([104.0, 104.0, 104.0])
    >>> np.round(probability_alive(x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844), 4)
    array([0.9468, 0.9826, 0.2784])
    """
    x, t_x, T = _as_arrays(x, t_x, T)
    x, t_x, T = np.broadcast_arrays(x, t_x, T)
    alpha_i, beta_i = _broadcast_rates(alpha, beta, x)

    log_x = (
        r * np.log(alpha_i)
        + s * np.log(beta_i)
        - special.gammaln(r)
        + special.gammaln(r + x)
    )
    log_y = -(r + x) * np.log(alpha_i + T) - s * np.log(beta_i + T)
    ll = log_likelihood_ind(x, t_x, T, r, alpha, s, beta)
    return np.exp(log_x + log_y - ll)


def conditional_expected_transactions(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    t: float,
    r: float,
    alpha: ArrayLike,
    s: float,
    beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""``CET`` -- expected transactions in the next :math:`t` periods.

    S6.3: metric (1), "the number of transactions to expect from the end of the
    estimation period until the end of the prediction period".

    .. math::
        E[Y(t) \mid x, t_x, T]
        = \frac{(r+x)(\beta+T)}{(\alpha+T)(s-1)}
          \left[1 - \left(\frac{\beta+T}{\beta+T+t}\right)^{s-1}\right]
          P(\omega > T \mid x, t_x, T)

    The expression divides by :math:`s-1`, so it is undefined at :math:`s = 1`;
    the fitted :math:`s = 0.5613` of S6.2.1 is comfortably away from it, and a
    value there raises rather than returning a silent infinity.

    Examples
    --------
    Predicting 52 weeks ahead for the first three apparel customers:

    >>> import numpy as np
    >>> x   = np.array([6, 2, 0])
    >>> t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T   = np.array([104.0, 104.0, 104.0])
    >>> np.round(conditional_expected_transactions(
    ...     x, t_x, T, 52.0, 1.4490, 48.6361, 0.5613, 46.8844), 4)
    array([2.2047, 1.0593, 0.1261])
    """
    if np.isclose(s, 1.0):
        raise ValueError(
            "CET is undefined at s = 1: the expression divides by (s - 1)"
        )
    x, t_x, T = _as_arrays(x, t_x, T)
    x, t_x, T = np.broadcast_arrays(x, t_x, T)
    alpha_i, beta_i = _broadcast_rates(alpha, beta, x)

    p1 = (r + x) * (beta_i + T) / ((alpha_i + T) * (s - 1.0))
    p2 = 1.0 - ((beta_i + T) / (beta_i + T + t)) ** (s - 1.0)
    return p1 * p2 * probability_alive(x, t_x, T, r, alpha, s, beta)


def discounted_expected_residual_transactions(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    continuous_discount_factor: float,
    r: float,
    alpha: ArrayLike,
    s: float,
    beta: ArrayLike,
) -> NDArray[np.float64]:
    r"""``DERT`` -- discounted expected residual transactions.

    S6.3: metric (3), "the total number of transactions for the residual
    lifetime of a customer discounted to the end of the estimation period. Note
    that this metric has an infinite prediction horizon".

    .. math::
        DERT = \frac{\alpha^{r}\beta^{s}\, \delta^{s-1}\, \Gamma(r+x+1)\,
                     U(s, s, \delta(\beta+T))}
                    {\Gamma(r)\,(\alpha+T)^{r+x+1}\, L}

    with :math:`U` Tricomi's confluent hypergeometric function and
    :math:`\delta` the continuous discount factor. Multiplying ``DERT`` by
    predicted mean spending gives CLV, S6.3.

    S6.3.2 fixes :math:`\delta` from a discrete annual rate :math:`d` and
    :math:`k` time units per year as :math:`\delta_k = \ln(1+d)/k`: "The natural
    logarithm appears because continuous compounding models growth as
    :math:`e^{\delta}`; equating this to the discrete one-year growth factor
    :math:`1+d` and solving for :math:`\delta` gives :math:`\delta=\ln(1+d)`."

    Examples
    --------
    At the 7.5% annual rate of S6.3.2, in weekly units:

    >>> import numpy as np
    >>> delta = np.log(1.075) / 52
    >>> x   = np.array([6, 2, 0])
    >>> t_x = np.array([93.285714, 99.571429, 0.0])
    >>> T   = np.array([104.0, 104.0, 104.0])
    >>> np.round(discounted_expected_residual_transactions(
    ...     x, t_x, T, delta, 1.4490, 48.6361, 0.5613, 46.8844), 4)
    array([16.025 ,  7.6997,  0.9167])
    """
    if not 0.0 <= continuous_discount_factor < 1.0:
        # CLVTools admits [0, 1) and this admitted (0, inf): zero was refused
        # where R returns the undiscounted expectation, and 1.5 or 100 were
        # accepted silently, returning a number for a per-period discount rate
        # of 10,000%. The parameter carries CLVTools' exact semantics --
        # ``DEFAULT_DISCOUNT_FACTOR`` is ``log(1.1)`` -- so its range transfers
        # with it. Finding A3 of ``docs/spec-audit.md``, spec PR-11.
        raise ValueError(
            "continuous_discount_factor must lie in [0, 1); got "
            f"{continuous_discount_factor}. It is a *per-period* rate -- see "
            "clvtools.predict.discount_factor to convert an annual one"
        )
    x, t_x, T = _as_arrays(x, t_x, T)
    x, t_x, T = np.broadcast_arrays(x, t_x, T)
    alpha_i, beta_i = _broadcast_rates(alpha, beta, x)

    delta = float(continuous_discount_factor)
    u = kummer_u(s, s, delta * (beta_i + T))
    ll = log_likelihood_ind(x, t_x, T, r, alpha, s, beta)

    log_dert = (
        r * np.log(alpha_i)
        + s * np.log(beta_i)
        + (s - 1.0) * np.log(delta)
        + special.gammaln(r + x + 1.0)
        + np.log(u)
        - special.gammaln(r)
        - (r + x + 1.0) * np.log(alpha_i + T)
        - ll
    )
    return np.exp(log_dert)


# -- diagnostics, S6.2.2 ------------------------------------------------------


def expectation(
    t: ArrayLike, r: float, alpha: ArrayLike, s: float, beta: ArrayLike
) -> NDArray[np.float64]:
    r""":math:`E[X(t)]` -- expected repeat transactions with no customer history.

    .. math::
        E[X(t)] = \frac{r\beta}{\alpha(s-1)}
                  \left[1 - \left(\frac{\beta}{\beta+t}\right)^{s-1}\right]

    This drives two things. It is the model line of the tracking plot of
    S6.2.2 -- "the sum of all transactions expected by all customers in each
    period" -- and, per S6.3.4, the prediction for a customer who does not exist
    yet: "the unconditional expectation [...] gives the expected number of orders
    for a customer for which no information is available."

    S6.3.4 adds one to it, "to account for all transactions that a prospective
    customer will make, including the first one". At the parameters of the
    full-data fit that S6.3.4 uses:

    >>> full = (1.376786710655611, 47.293726061187556,
    ...         0.6745790427245901, 62.83250597832072)
    >>> float(round(1 + expectation(52.0, *full), 6))
    2.218635

    which is the 2.218635 the paper prints. It starts at zero, "and this fact
    gives the plot its characteristic shape":

    >>> bool(expectation(0.0, 1.4490, 48.6361, 0.5613, 46.8844) == 0.0)
    True
    """
    if np.isclose(s, 1.0):
        raise ValueError(
            "the expectation is undefined at s = 1: it divides by (s - 1)"
        )
    t = np.asarray(t, dtype=float)
    alpha_i = np.asarray(alpha, dtype=float)
    beta_i = np.asarray(beta, dtype=float)
    return (
        (r * beta_i)
        / (alpha_i * (s - 1.0))
        * (1.0 - (beta_i / (beta_i + t)) ** (s - 1.0))
    )


def pmf(
    k: int, T: ArrayLike, r: float, alpha: ArrayLike, s: float, beta: ArrayLike
) -> NDArray[np.float64]:
    r"""``P(X(T) = k)`` -- the probability of exactly :math:`k` repeat purchases.

    S6.2.2: the PMF plot "shows the actual and expected number of customers who
    make a given number of repeat transactions during the estimation period.
    [...] For each bin, the expected number of customers is the sum of all
    customers' individual PMF values for this number of purchases."

    Unlike :func:`nbd_pmf <clvtools.pnbd.individual.nbd_pmf>` this accounts for
    the customer possibly having died partway through :math:`(0, T]`: the first
    term covers surviving the whole window, the sum covers dying inside it.

    Examples
    --------
    A proper distribution -- the apparel cohort's window is 104 weeks:

    >>> import numpy as np
    >>> total = sum(float(pmf(k, 104.0, 1.4490, 48.6361, 0.5613, 46.8844))
    ...             for k in range(400))
    >>> bool(np.isclose(total, 1.0, atol=1e-6))
    True
    >>> float(np.round(pmf(0, 104.0, 1.4490, 48.6361, 0.5613, 46.8844), 4))
    0.3553
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    k = int(k)
    T = np.asarray(T, dtype=float)
    alpha_i, beta_i = _broadcast_rates(alpha, beta, T)

    # Part 1: still alive at T, so the count is negative binomial.
    log_p1 = (
        special.gammaln(r + k)
        - special.gammaln(r)
        - special.gammaln(k + 1.0)
        + r * (np.log(alpha_i) - np.log(alpha_i + T))
        + k * (np.log(T) - np.log(alpha_i + T))
        + s * (np.log(beta_i) - np.log(beta_i + T))
    )
    part1 = np.exp(log_p1)

    # Part 2: died at some point in (0, T].
    log_p2 = (
        r * np.log(alpha_i)
        + s * np.log(beta_i)
        + special.betaln(r + k, s + 1.0)
        - special.betaln(r, s)
    )

    gap = np.abs(alpha_i - beta_i)
    hi = np.maximum(alpha_i, beta_i)
    b = np.where(alpha_i >= beta_i, s + 1.0, r + k)
    rsk1 = r + s + k + 1.0

    # hyp2f1_ratio only covers c = a+1; here c = r+s+k+1 against a = r+s+i, so
    # the general SciPy call is needed.
    b1 = special.hyp2f1(r + s, b, rsk1, gap / hi) / hi ** (r + s)

    b2 = np.zeros_like(part1)
    for i in range(k + 1):
        log_part = (
            special.gammaln(r + s + i)
            + i * np.log(T)
            - special.gammaln(r + s)
            - special.gammaln(i + 1.0)
        )
        term = special.hyp2f1(r + s + i, b, rsk1, gap / (hi + T))
        b2 = b2 + np.exp(
            log_part + np.log(term) - (r + s + i) * np.log(hi + T)
        )

    difference = b1 - b2
    if np.any(difference <= 0.0):
        # `b1` and `b2` are each O(1e-7) here and their difference O(1e-22):
        # fifteen digits of cancellation, past which float64 has nothing left
        # and the difference lands on zero or goes negative. `np.log` of that
        # is a silent `NaN` -- and a `NaN` is contagious, so one negligible
        # term poisons `sum(pmf(k) for k in ...)` entirely.
        #
        # Not repairable by a fallback to `part1`: measured against a 60-digit
        # evaluation at `alpha=500, beta=1, s=1.5, T=52`, the dying-inside-the-
        # window term is **9% of the answer** at `k = 18`, not a rounding
        # correction. The fix is to form the difference without cancelling,
        # which is item 28's treatment of `F2` applied here -- backlog item 32.
        # Until then this says so rather than returning `NaN` quietly.
        warnings.warn(
            f"pmf lost the second term to cancellation at k={k}: the closed "
            f"form subtracts two values of order {float(np.max(b1)):.2e} whose "
            f"difference is below what float64 can carry, so the result is NaN "
            f"rather than a number. Large k with a large alpha/beta ratio is "
            f"where this happens; see docs/backlog.md item 32.",
            PrecisionWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return part1 + np.exp(log_p2 + np.log(difference))
