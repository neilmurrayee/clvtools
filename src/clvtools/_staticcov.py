r"""Shared estimation machinery for the time-invariant covariate models.

Table 3 marks all three latent attrition families as supporting time-invariant
covariates, equality constraints and regularization. What differs between them
is only the likelihood and how many model parameters it takes; the search
vector, the constraint bookkeeping of eq. (14), the penalty of eq. (13) and the
Hessian are the same three times over.

The search runs over

.. code-block:: text

    [log model parameters..., free life gammas..., free trans gammas...,
     shared gammas...]

with the model parameters on the log scale because they must stay positive and
the covariate parameters unconstrained because they may be either sign. A
covariate named in ``names_cov_constr`` occupies a single coordinate that is
written into *both* processes, which is eq. (14) implemented directly:
:math:`\boldsymbol{\gamma}_{purch} \equiv \boldsymbol{\gamma}_{attr}`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from clvtools._optimize import options_for

__all__ = ["StaticCovResult", "fit_static_covariates", "numerical_hessian"]

#: S6.4: "If not given, the start values are set to 0.1 for all covariates."
DEFAULT_COV_START = 0.1


def numerical_hessian(fn, at: NDArray[np.float64], step: float = 1e-5):
    """Central-difference Hessian of ``fn`` at ``at``.

    CLVTools uses ``numDeriv`` for the same purpose. The step is relative to
    each coordinate's magnitude, so parameters on very different scales are
    differenced comparably.
    """
    n = at.size
    h = step * np.maximum(np.abs(at), 1.0)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = h[i]
            ej = np.zeros(n); ej[j] = h[j]
            out[i, j] = out[j, i] = (
                fn(at + ei + ej) - fn(at + ei - ej)
                - fn(at - ei + ej) + fn(at - ei - ej)
            ) / (4 * h[i] * h[j])
    return out


@dataclass(frozen=True)
class StaticCovResult:
    """What a covariate fit produces, before a family wraps it in its own type."""

    model: NDArray[np.float64]
    gamma_life: NDArray[np.float64]
    gamma_trans: NDArray[np.float64]
    names_cov_life: list[str]
    names_cov_trans: list[str]
    names_cov_constr: list[str]
    log_likelihood: float
    unpenalised_log_likelihood: float
    converged: bool
    n_customers: int
    reg_lambdas: tuple[float, float] | None = None
    hessian: NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def names(self) -> list[str]:
        """Coefficient names for the covariate part, in CLVTools' convention.

        A constrained covariate appears once as ``constr.<name>`` rather than
        twice, which is how S6.5.3's output reads.
        """
        constrained = set(self.names_cov_constr)
        return (
            [f"life.{n}" for n in self.names_cov_life if n not in constrained]
            + [f"trans.{n}" for n in self.names_cov_trans if n not in constrained]
            + [f"constr.{n}" for n in self.names_cov_constr]
        )

    def covariate_values(self) -> list[float]:
        """The covariate estimates, in the order :attr:`names` lists them."""
        constrained = set(self.names_cov_constr)
        out = [
            float(v) for n, v in zip(self.names_cov_life, self.gamma_life)
            if n not in constrained
        ]
        out += [
            float(v) for n, v in zip(self.names_cov_trans, self.gamma_trans)
            if n not in constrained
        ]
        out += [
            float(self.gamma_life[self.names_cov_life.index(n)])
            for n in self.names_cov_constr
        ]
        return out


def fit_static_covariates(
    *,
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    cov_life: NDArray[np.float64],
    cov_trans: NDArray[np.float64],
    names_cov_life: list[str],
    names_cov_trans: list[str],
    log_likelihood: Callable[..., float],
    n_model_params: int,
    model_start: tuple[float, ...],
    names_cov_constr: list[str] | None = None,
    reg_lambdas: tuple[float, float] | None = None,
    start: tuple[float, ...] | None = None,
    start_cov: float | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = False,
    polish: bool = True,
    options: dict | None = None,
) -> StaticCovResult:
    r"""Estimate a latent attrition model with time-invariant covariates.

    Parameters
    ----------
    log_likelihood
        ``(model_params, gamma_life, gamma_trans, cov_life, cov_trans) -> float``
        with ``model_params`` on the natural scale. Each family supplies its
        own; the sign conventions of its rate builders live there, not here.
    n_model_params, model_start
        How many parameters the family has, and the default start for them.
    names_cov_constr
        Covariates forced to one coefficient across both processes, eq. (14).
    polish
        Follow the search with a derivative-free pass from wherever it stopped,
        keeping whichever point attains more. Covariate models can be very
        ill-conditioned in one direction -- on the BG/NBD the two beta
        parameters are barely identified, and L-BFGS-B reports convergence at
        ``a + b`` around 1,200 where the likelihood is still climbing past
        300,000 -- so the polish is on by default. It can only improve the
        objective, since the worse of the two points is discarded.
    reg_lambdas
        ``(life, trans)`` L2 penalties, eq. (13).

        .. warning::
           With a penalty applied, :attr:`~StaticCovResult.log_likelihood` holds
           the penalised *mean* objective that was minimised, matching what
           CLVTools' ``logLik()`` returns. Use
           :attr:`~StaticCovResult.unpenalised_log_likelihood` for anything
           comparable across models.
    """
    x, t_x, T = (np.asarray(v, dtype=float).ravel() for v in (x, t_x, T))
    if not (x.shape == t_x.shape == T.shape):
        raise ValueError("x, t_x and T must have the same shape")
    if x.size == 0:
        raise ValueError("no customers to fit")

    names_life, names_trans = list(names_cov_life), list(names_cov_trans)
    constrained = list(names_cov_constr or [])
    for name in constrained:
        if name not in names_life or name not in names_trans:
            raise ValueError(
                f"cannot constrain {name!r}: it must be a covariate of both processes"
            )

    free_life = [n for n in names_life if n not in constrained]
    free_trans = [n for n in names_trans if n not in constrained]
    idx_life = [names_life.index(n) for n in free_life]
    idx_trans = [names_trans.index(n) for n in free_trans]
    idx_constr_life = [names_life.index(n) for n in constrained]
    idx_constr_trans = [names_trans.index(n) for n in constrained]
    n_free_life, n_free_trans, n_constr = (
        len(free_life), len(free_trans), len(constrained)
    )

    if start is not None:
        start_arr = np.asarray(start, dtype=float)
        if start_arr.shape != (n_model_params,):
            raise ValueError(f"start must give {n_model_params} model parameters")
        if np.any(start_arr <= 0):
            raise ValueError("start values must be strictly positive")

    if reg_lambdas is not None:
        reg_lambdas = tuple(float(v) for v in reg_lambdas)
        if len(reg_lambdas) != 2:
            raise ValueError("reg_lambdas must give two values (life, trans)")
        if any(v < 0 for v in reg_lambdas):
            raise ValueError("regularization weights must be non-negative")

    n_cov_coords = n_free_life + n_free_trans + n_constr

    def starting_point(model_values, cov_value) -> NDArray[np.float64]:
        return np.concatenate([
            np.log(np.asarray(model_values, dtype=float)),
            np.full(n_cov_coords, float(cov_value)),
        ])

    if start is not None:
        candidates = [
            starting_point(
                start_arr,
                DEFAULT_COV_START if start_cov is None else start_cov,
            )
        ]
    elif reg_lambdas is None:
        candidates = [
            starting_point(
                model_start,
                DEFAULT_COV_START if start_cov is None else start_cov,
            )
        ]
    else:
        # Regularized fits get two starting points, and the better one wins.
        #
        # Warm-starting from the unpenalised solution is the usual way to trace
        # a regularization path, and on the Pareto/NBD it is necessary: dividing
        # the likelihood by n flattens the objective enough that a cold start
        # converges in a clearly worse basin. But it is not universally right.
        # The BG/NBD's unpenalised optimum sits far out on the ridge where its
        # two beta parameters are unidentified, at a + b in the millions, and
        # the penalised optimum is back near a + b = 10 -- warm-starting there
        # strands the search. Running both costs one extra fit and removes the
        # need to guess which model is which.
        baseline = fit_static_covariates(
            x=x, t_x=t_x, T=T, cov_life=cov_life, cov_trans=cov_trans,
            names_cov_life=names_life, names_cov_trans=names_trans,
            log_likelihood=log_likelihood,
            n_model_params=n_model_params, model_start=model_start,
            names_cov_constr=constrained,
            method=method, maxiter=maxiter, hessian=False,
            polish=polish, options=options,
        )
        cold = DEFAULT_COV_START if start_cov is None else start_cov
        candidates = [
            starting_point(model_start, cold),
            starting_point(baseline.model, 0.0 if start_cov is None else start_cov),
        ]

    def unpack(v: NDArray[np.float64]):
        """Expand the search vector into full per-process coefficient vectors."""
        model = np.exp(v[:n_model_params])
        cursor = n_model_params
        free_l = v[cursor : cursor + n_free_life]
        cursor += n_free_life
        free_t = v[cursor : cursor + n_free_trans]
        cursor += n_free_trans
        shared = v[cursor:]

        g_life = np.zeros(len(names_life))
        g_trans = np.zeros(len(names_trans))
        g_life[idx_life] = free_l
        g_trans[idx_trans] = free_t
        g_life[idx_constr_life] = shared
        g_trans[idx_constr_trans] = shared
        return model, g_life, g_trans

    def objective(v: NDArray[np.float64]) -> float:
        model, g_life, g_trans = unpack(v)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(model, g_life, g_trans, cov_life, cov_trans)
        if not np.isfinite(value):
            return np.inf
        if reg_lambdas is None:
            return -value
        # Eq. (13), on the mean so the weight means the same at any sample size.
        lam_life, lam_trans = reg_lambdas
        penalty = lam_life * np.sum(g_life**2) + lam_trans * np.sum(g_trans**2)
        return -value / x.size + penalty

    result = None
    for x0 in candidates:
        attempt = optimize.minimize(
            objective, x0=x0, method=method,
            options=options_for(method, maxiter, x0, options),
        )
        if result is None or attempt.fun < result.fun:
            result = attempt

    if polish and method != "Nelder-Mead":
        # A simplex spanning a factor of e per axis, centred on where the
        # gradient method stopped.
        simplex = np.repeat(result.x[None, :], result.x.size + 1, axis=0)
        simplex[1:] += np.eye(result.x.size)
        polished = optimize.minimize(
            objective, x0=result.x, method="Nelder-Mead",
            options={
                # Bounded: on a genuinely flat direction the simplex will
                # keep creeping for as long as it is allowed to, for a gain
                # far below anything that matters.
                "maxiter": 20_000, "maxfev": 20_000,
                "xatol": 1e-10, "fatol": 1e-10, "initial_simplex": simplex,
            },
        )
        if polished.fun < result.fun:
            result = polished

    model, g_life, g_trans = unpack(result.x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        unpenalised = float(
            log_likelihood(model, g_life, g_trans, cov_life, cov_trans)
        )

    hess = None
    if hessian:
        natural = np.concatenate([model, g_life, g_trans])
        n_life_all = len(names_life)

        def natural_objective(v: NDArray[np.float64]) -> float:
            return -log_likelihood(
                v[:n_model_params],
                v[n_model_params : n_model_params + n_life_all],
                v[n_model_params + n_life_all :],
                cov_life, cov_trans,
            )

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hess = numerical_hessian(natural_objective, natural)

    return StaticCovResult(
        model=np.asarray(model, dtype=float),
        gamma_life=np.asarray(g_life, dtype=float),
        gamma_trans=np.asarray(g_trans, dtype=float),
        names_cov_life=names_life,
        names_cov_trans=names_trans,
        names_cov_constr=constrained,
        log_likelihood=float(-result.fun),
        unpenalised_log_likelihood=unpenalised,
        converged=bool(result.success),
        n_customers=int(x.size),
        reg_lambdas=reg_lambdas,
        hessian=hess,
    )
