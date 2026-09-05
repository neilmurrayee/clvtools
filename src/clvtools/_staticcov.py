r"""Shared estimation machinery for the time-invariant covariate models.

Table 4 marks all three latent attrition families as supporting time-invariant
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

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from clvtools._optimize import _accepted_options, options_for
from clvtools._validate import ConvergenceWarning, finished, start_values
from clvtools.inference import Fitted, numerical_hessian

__all__ = ["SearchSettings", "StaticCovResult", "fit_static_covariates"]

#: S6.4: "If not given, the start values are set to 0.1 for all covariates."
DEFAULT_COV_START = 0.1


def design(
    covariates: ArrayLike, gamma: ArrayLike, process: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""A design matrix and its coefficients as arrays, checked against each other.

    The prologue every per-customer rate builder shares -- the Pareto/NBD's
    :math:`\alpha_i` and :math:`\beta_i`, the BG/NBD's :math:`\alpha_i`,
    :math:`a_i` and :math:`b_i`, the GGom/NBD's :math:`\alpha_i` and
    :math:`b_i`. Only the equation itself differs between them, and each writes
    that out where its own ``.. math::`` can be read beside it.

    ``process`` is ``"transaction"`` or ``"attrition"``, and names which set of
    covariates did not line up.

    >>> import numpy as np
    >>> matrix, coefficients = design([[0.0, 1.0]], [0.2, 0.6], "transaction")
    >>> matrix.shape, coefficients.shape
    ((1, 2), (2,))
    >>> design([[0.0, 1.0]], [0.2], "transaction")
    Traceback (most recent call last):
        ...
    ValueError: 2 transaction covariates but 1 parameters
    """
    matrix = np.atleast_2d(np.asarray(covariates, dtype=float))
    coefficients = np.asarray(gamma, dtype=float)
    if matrix.shape[1] != coefficients.size:
        raise ValueError(
            f"{matrix.shape[1]} {process} covariates but "
            f"{coefficients.size} parameters"
        )
    return matrix, coefficients


@dataclass(frozen=True)
class StaticCovResult(Fitted):
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
            float(v) for n, v in zip(self.names_cov_life, self.gamma_life, strict=True)
            if n not in constrained
        ]
        out += [
            float(v) for n, v in zip(self.names_cov_trans, self.gamma_trans, strict=True)
            if n not in constrained
        ]
        out += [
            float(self.gamma_life[self.names_cov_life.index(n)])
            for n in self.names_cov_constr
        ]
        return out


class DelegatesToCovariates(Fitted):
    """A family's covariate fit, holding a :class:`StaticCovResult` and forwarding.

    The BG/NBD and the GGom/NBD keep their covariate estimates in a
    ``StaticCovResult`` rather than spreading them across their own fields, so
    everything except the model parameters is that object's to answer. Both
    wrote the same ten forwarding properties out; they live here instead.

    A subclass supplies its model parameters as dataclass fields, a
    ``covariates`` field, and the two things that genuinely differ between
    families -- :attr:`~clvtools.inference.Fitted.names` and ``__iter__``,
    which are where the model parameters go in front of the covariate ones.

    The Pareto/NBD deliberately does not inherit this: its covariate class
    predates ``StaticCovResult`` and carries the estimates as its own fields,
    which its ``__iter__`` and the constrained-covariate handling in its
    ``names`` are written against.
    """

    #: Supplied by each subclass as a dataclass field. An annotation rather
    #: than a property, so that the field is not shadowed by a descriptor.
    covariates: StaticCovResult

    @property
    def names_cov_life(self) -> list[str]:
        return self.covariates.names_cov_life

    @property
    def names_cov_trans(self) -> list[str]:
        return self.covariates.names_cov_trans

    @property
    def names_cov_constr(self) -> list[str]:
        """S6.5.3's tied covariates, as the Pareto/NBD's result reports them.

        Carried by :class:`StaticCovResult` all along and forwarded nowhere, so
        the same question -- which covariates share one coefficient? -- could be
        asked of a Pareto/NBD fit and not of a BG/NBD or GGom/NBD one.
        """
        return self.covariates.names_cov_constr

    @property
    def reg_lambdas(self) -> tuple[float, float] | None:
        """Eq. (13)'s two penalties, or ``None`` where the fit was unregularized.

        The same gap as :attr:`names_cov_constr`: without it, nothing on a
        BG/NBD or GGom/NBD result says whether its standard errors are the
        ridge ones the README's findings warn about.
        """
        return self.covariates.reg_lambdas

    @property
    def gamma_life(self):
        return self.covariates.gamma_life

    @property
    def gamma_trans(self):
        return self.covariates.gamma_trans

    @property
    def log_likelihood(self) -> float:
        return self.covariates.log_likelihood

    @property
    def unpenalised_log_likelihood(self) -> float:
        return self.covariates.unpenalised_log_likelihood

    @property
    def _comparable_log_likelihood(self) -> float:
        """A regularized fit's :attr:`log_likelihood` is the penalised mean.

        :class:`StaticCovResult` always carries the unpenalised sum beside it,
        so AIC and BIC are built on that one and stay comparable with an
        unregularized model's.
        """
        return self.unpenalised_log_likelihood

    @property
    def converged(self) -> bool:
        return self.covariates.converged

    @property
    def n_customers(self) -> int:
        return self.covariates.n_customers

    @property
    def hessian(self):
        """Curvature over :attr:`names`, from the covariate fit."""
        return self.covariates.hessian


@dataclass(frozen=True)
class SearchSettings:
    """How the estimate is obtained, as distinct from what is estimated.

    Every family exposes this same set of knobs on its own ``fit_*_staticcov``,
    and they are only ever passed straight through to the search. Grouping them
    keeps :func:`fit_static_covariates` about the model rather than about the
    optimiser.

    ``reg_lambdas`` sits here too: eq. (13)'s penalty changes the objective the
    search minimises, not the model being fitted.
    """

    start: tuple[float, ...] | None = None
    start_cov: float | None = None
    reg_lambdas: tuple[float, float] | None = None
    method: str = "L-BFGS-B"
    maxiter: int = 10_000
    hessian: bool = False
    polish: bool = True
    options: dict | None = None


@dataclass(frozen=True)
class _Layout:
    r"""Where each estimated coefficient sits in the search vector.

    This is eq. (14) as bookkeeping: a covariate named in ``constrained``
    occupies a *single* coordinate that is written into both processes, so the
    search never sees the two coefficients it ties together.
    """

    names_life: list[str]
    names_trans: list[str]
    constrained: list[str]
    n_model_params: int
    idx_life: list[int]
    idx_trans: list[int]
    idx_constr_life: list[int]
    idx_constr_trans: list[int]

    @classmethod
    def build(
        cls,
        names_cov_life: list[str],
        names_cov_trans: list[str],
        names_cov_constr: list[str] | None,
        n_model_params: int,
    ) -> _Layout:
        """Resolve names to coordinates, rejecting a constraint that cannot hold.

        >>> layout = _Layout.build(["a", "b"], ["a", "c"], ["a"], 4)
        >>> layout.idx_life, layout.idx_constr_life, layout.n_cov_coords
        ([1], [0], 3)
        """
        names_life, names_trans = list(names_cov_life), list(names_cov_trans)
        constrained = _validated_constraints(
            names_cov_constr, names_life, names_trans
        )
        free_life = [n for n in names_life if n not in constrained]
        free_trans = [n for n in names_trans if n not in constrained]
        return cls(
            names_life=names_life,
            names_trans=names_trans,
            constrained=constrained,
            n_model_params=n_model_params,
            idx_life=[names_life.index(n) for n in free_life],
            idx_trans=[names_trans.index(n) for n in free_trans],
            idx_constr_life=[names_life.index(n) for n in constrained],
            idx_constr_trans=[names_trans.index(n) for n in constrained],
        )

    @property
    def n_cov_coords(self) -> int:
        """Free life gammas, then free trans gammas, then one per constraint."""
        return len(self.idx_life) + len(self.idx_trans) + len(self.constrained)

    def unpack(
        self, v: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Expand the search vector into full per-process coefficient vectors."""
        model = np.exp(v[: self.n_model_params])
        cursor = self.n_model_params
        free_l = v[cursor : cursor + len(self.idx_life)]
        cursor += len(self.idx_life)
        free_t = v[cursor : cursor + len(self.idx_trans)]
        cursor += len(self.idx_trans)
        shared = v[cursor:]

        g_life = np.zeros(len(self.names_life))
        g_trans = np.zeros(len(self.names_trans))
        g_life[self.idx_life] = free_l
        g_trans[self.idx_trans] = free_t
        g_life[self.idx_constr_life] = shared
        g_trans[self.idx_constr_trans] = shared
        return model, g_life, g_trans

    def starting_point(
        self, model_values: ArrayLike, cov_value: float
    ) -> NDArray[np.float64]:
        """Model parameters on the log scale, covariates flat at ``cov_value``."""
        return np.concatenate([
            np.log(np.asarray(model_values, dtype=float)),
            np.full(self.n_cov_coords, float(cov_value)),
        ])


def _polish_overrides(overrides: dict | None) -> dict:
    """The caller's optimiser overrides, narrowed to what the polish can read.

    The polish is always Nelder-Mead while the search above it is usually
    L-BFGS-B, so the two accept different spellings of the same bound --
    ``maxfun`` against ``maxfev``. Forwarding the intersection means a caller
    who caps the fit caps all of it, without this stage inheriting a key that
    :func:`~clvtools._optimize.options_for` would refuse.
    """
    if not overrides:
        return {}
    accepted = _accepted_options("Nelder-Mead")
    return {k: v for k, v in overrides.items() if k in accepted}


def _validated_cov_start(start_cov) -> float:
    """The covariate coefficients' common start value, checked before the fit.

    Spec `V-02`. Unlike the model start, this is a **single scalar** applied to
    every covariate coefficient, where R takes a named vector with one entry per
    covariate -- so five of that claim's seven failure modes (an unnamed entry,
    a duplicate, a name that is not a covariate, a covariate left out) cannot
    arise here at all. The two that can are that it be numeric and finite, and
    a `NaN` used to pass straight through to the objective, which then reported
    "the static-covariate objective is not finite at the point the search
    started" -- a statement about the model for a fault in the argument. The
    same shape as `V-01` and `X-14`.
    """
    if start_cov is None:
        return DEFAULT_COV_START
    try:
        value = float(start_cov)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"start_cov must be a single number applied to every covariate "
            f"coefficient, not {type(start_cov).__name__}"
        ) from error
    if not np.isfinite(value):
        raise ValueError(f"start_cov must be a finite number, got {start_cov!r}")
    return value


def _validated_constraints(
    names_cov_constr, names_life: list[str], names_trans: list[str]
) -> list[str]:
    """S6.5.3's tied covariates, checked before they reach the layout.

    Spec `X-15` names six failure claims. Three of them landed badly:

    * A **bare string** was iterated character by character, so
      ``names_cov_constr="Gender"`` reported ``cannot constrain 'G'``. Python's
      oldest sharp edge, and the message gave no hint of it.
    * A **duplicate** was accepted, silently tying one covariate twice and
      leaving the search a coordinate it could not move.
    * An **unknown name** was told "it must be a covariate of both processes",
      which is true and not the problem: ``Nope`` is a covariate of neither, and
      the message sent the reader looking for an asymmetry that is not there.
    """
    if names_cov_constr is None:
        return []
    if isinstance(names_cov_constr, str):
        raise TypeError(
            f"names_cov_constr must be a list of covariate names, not the "
            f"string {names_cov_constr!r}: pass [{names_cov_constr!r}]"
        )
    constrained = list(names_cov_constr)
    duplicates = [
        name for i, name in enumerate(constrained) if name in constrained[:i]
    ]
    if duplicates:
        raise ValueError(
            f"cannot constrain {duplicates[0]!r} twice: names_cov_constr has "
            f"{len(constrained)} entries and {len(set(constrained))} distinct"
        )
    known = set(names_life) | set(names_trans)
    for name in constrained:
        if name not in known:
            raise ValueError(
                f"cannot constrain {name!r}: the data carries "
                f"{', '.join(sorted(known))}"
            )
        if name not in names_life or name not in names_trans:
            side = "transaction" if name in names_life else "lifetime"
            raise ValueError(
                f"cannot constrain {name!r}: it must be a covariate of both "
                f"processes and is missing from the {side} one"
            )
    return constrained


def _validated_reg_lambdas(
    reg_lambdas: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Eq. (13)'s two weights, as floats."""
    if reg_lambdas is None:
        return None
    if np.isscalar(reg_lambdas) or isinstance(reg_lambdas, str):
        # `reg_lambdas=1.0` used to reach `tuple(float(v) for v in ...)` and
        # come back as "'float' object is not iterable", which names Python's
        # problem rather than the caller's. Eq. (13) has two weights.
        raise ValueError(
            f"reg_lambdas must give two values (life, trans), not the single "
            f"{reg_lambdas!r}; pass ({reg_lambdas!r}, {reg_lambdas!r}) to "
            f"penalise both processes equally"
        )
    try:
        reg = tuple(float(v) for v in reg_lambdas)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"regularization weights must be numbers, got {tuple(reg_lambdas)!r}"
        ) from error
    if len(reg) != 2:
        raise ValueError("reg_lambdas must give two values (life, trans)")
    if any(np.isnan(v) for v in reg):
        # Accepted until now, and it escaped into the objective: the fit then
        # failed with "objective is not finite", which points at the model
        # rather than at the argument. Spec `X-14`.
        raise ValueError(f"regularization weights must be numbers, got {reg}")
    if any(v < 0 for v in reg):
        raise ValueError("regularization weights must be non-negative")
    return reg


def _starting_points(
    layout: _Layout,
    settings: SearchSettings,
    start_arr: NDArray[np.float64] | None,
    model_start: tuple[float, ...],
    cov_start: float,
    reg_lambdas: tuple[float, float] | None,
    baseline: Callable[[], StaticCovResult],
) -> list[NDArray[np.float64]]:
    """Where the search starts: one point, or two for a regularized fit.

    A caller's own ``start`` is taken as given, and an unregularized fit runs
    from the family's default. A regularized one gets both, and the better
    result wins.

    Warm-starting from the unpenalised solution is the usual way to trace a
    regularization path, and on the Pareto/NBD it is necessary: dividing the
    likelihood by ``n`` flattens the objective enough that a cold start
    converges in a clearly worse basin. But it is not universally right. The
    BG/NBD's unpenalised optimum sits far out on the ridge where its two beta
    parameters are unidentified, at ``a + b`` in the millions, and the
    penalised optimum is back near ``a + b = 10`` -- warm-starting there
    strands the search. Running both costs one extra fit and removes the need
    to guess which model is which.

    ``baseline`` is a thunk rather than a fitted result because only the last
    branch needs it, and it is a whole extra optimisation.
    """
    if start_arr is not None:
        return [layout.starting_point(start_arr, cov_start)]
    if reg_lambdas is None:
        return [layout.starting_point(model_start, cov_start)]
    return [
        layout.starting_point(model_start, cov_start),
        layout.starting_point(
            baseline().model,
            0.0 if settings.start_cov is None else settings.start_cov,
        ),
    ]


def _search(
    objective: Callable[[NDArray[np.float64]], float],
    candidates: list[NDArray[np.float64]],
    settings: SearchSettings,
) -> optimize.OptimizeResult:
    """Minimise from each candidate start, then optionally polish the winner.

    The polish is a derivative-free pass from wherever the gradient method
    stopped. Covariate models can be very ill-conditioned in one direction --
    on the BG/NBD the two beta parameters are barely identified, and L-BFGS-B
    reports convergence at ``a + b`` around 1,200 where the likelihood is still
    climbing past 300,000 -- so it is on by default. It can only improve the
    objective, since the worse of the two points is discarded.

    Every candidate is run and the best kept, so the search is written as a
    comprehension and a ``min``: there is then no point at which the winner is
    ``None``, which is what the return type promises.
    """
    attempts = [
        optimize.minimize(
            objective,
            x0=x0,
            method=settings.method,
            options=options_for(settings.method, settings.maxiter, x0, settings.options),
        )
        for x0 in candidates
    ]
    result = min(attempts, key=lambda attempt: attempt.fun)

    if settings.polish and settings.method != "Nelder-Mead":
        # A simplex spanning a factor of e per axis, centred on where the
        # gradient method stopped.
        simplex = np.repeat(result.x[None, :], result.x.size + 1, axis=0)
        simplex[1:] += np.eye(result.x.size)
        polished = optimize.minimize(
            objective,
            x0=result.x,
            method="Nelder-Mead",
            options={
                # Bounded: on a genuinely flat direction the simplex will
                # keep creeping for as long as it is allowed to, for a gain
                # far below anything that matters.
                "maxiter": 20_000, "maxfev": 20_000,
                "xatol": 1e-10, "fatol": 1e-10, "initial_simplex": simplex,
                # A caller who bounded the search meant the whole fit, not its
                # first stage. This stage used to keep the 20,000 above
                # whatever was asked for, so `options={"maxiter": 3}` returned
                # in 10.5 s where `polish=False` returned in 0.005 s -- a
                # factor of 2,100 between what was requested and what ran.
                # Only the keys Nelder-Mead reads are carried
                # over: the search above may be L-BFGS-B, whose `maxfun` this
                # stage would reject.
                **_polish_overrides(settings.options),
            },
        )
        if polished.fun < result.fun:
            if result.success and not polished.success:
                # Otherwise this reads as a failed fit when what happened is
                # that a *better* point was found by a stage with a cap on it.
                warnings.warn(
                    "the Nelder-Mead polish improved the objective from "
                    f"{result.fun:.6f} to {polished.fun:.6f} but stopped at its "
                    f"own limit ({polished.message}); reporting the better "
                    "point, so `converged` describes the polish rather than "
                    "the gradient search that preceded it",
                    ConvergenceWarning,
                    stacklevel=3,
                )
            result = polished
    return finished(result, "static-covariate")


def _penalised(
    negative_ll: float,
    g_life: NDArray[np.float64],
    g_trans: NDArray[np.float64],
    reg_lambdas: tuple[float, float] | None,
    n_customers: int,
) -> float:
    r"""Eq. (13) applied to a negative log-likelihood, or it left alone.

    One function, called from both the search's objective and the one the
    Hessian is differenced on, so that the two cannot drift apart. They had:
    the search minimised the penalised **mean** while the reported curvature
    came from the unpenalised **sum**, which is finding 9 of the outside
    review. The estimates and their standard errors now describe the same
    function.

    The division by ``n`` is CLVTools', not the paper's -- see the README's
    findings on both the objective and what it does to a standard error.

    >>> import numpy as np
    >>> g = np.array([1.0, 2.0])
    >>> float(_penalised(-100.0, g, g, None, 10))
    -100.0
    >>> float(_penalised(-100.0, g, g, (0.5, 0.5), 10))
    -5.0
    """
    lam_life, lam_trans = reg_lambdas or (0.0, 0.0)
    if not (lam_life or lam_trans):
        # A zero penalty is no penalty. Without this, ``reg_lambdas=(0, 0)``
        # still divided the likelihood by ``n`` -- eq. (13)'s mean convention
        # -- so the *estimates* matched an unregularized fit (scaling does not
        # move an optimum) while its Hessian was 1/600 of one and every
        # standard error came out sqrt(600) = 24.5 times too large: 8.47
        # against 0.346 for `r`. The R suite asserts that lambda = 0
        # reproduces the unpenalised fit including its summary table, and it
        # is right to: which objective is minimised should not turn on a
        # weight that contributes nothing. Finding B6 of
        # the 2026-09 spec audit, spec X-06.
        return negative_ll
    penalty = lam_life * np.sum(g_life**2) + lam_trans * np.sum(g_trans**2)
    return negative_ll / n_customers + penalty


def _reported_hessian(
    layout: _Layout,
    negative_log_likelihood: Callable[..., float],
    model: NDArray[np.float64],
    g_life: NDArray[np.float64],
    g_trans: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Curvature in the parameters actually estimated.

    Under an equality constraint that is one coefficient per constrained
    covariate rather than two. Differencing the full unconstrained vector
    instead would produce one standard error too many, and they would line up
    with the wrong names. The model parameters are taken on their natural
    scale, since that is what a standard error refers to.
    """
    n = layout.n_model_params
    reported = np.concatenate([
        model,
        g_life[layout.idx_life],
        g_trans[layout.idx_trans],
        g_life[layout.idx_constr_life],
    ])

    def reported_objective(v: NDArray[np.float64]) -> float:
        m, gl, gt = layout.unpack(np.concatenate([np.log(v[:n]), v[n:]]))
        return negative_log_likelihood(m, gl, gt)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return numerical_hessian(reported_objective, reported)


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
    search: SearchSettings | None = None,
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
    search
        :class:`SearchSettings`: start values, eq. (13)'s penalty, and the
        optimiser knobs. Defaults to the settings every family defaults to.

        .. warning::
           With a penalty applied, :attr:`~StaticCovResult.log_likelihood` holds
           the penalised *mean* objective that was minimised, matching what
           CLVTools' ``logLik()`` returns. Use
           :attr:`~StaticCovResult.unpenalised_log_likelihood` for anything
           comparable across models.
    """
    settings = SearchSettings() if search is None else search
    x, t_x, T = (np.asarray(v, dtype=float).ravel() for v in (x, t_x, T))
    if not (x.shape == t_x.shape == T.shape):
        raise ValueError("x, t_x and T must have the same shape")
    if x.size == 0:
        raise ValueError("no customers to fit")

    layout = _Layout.build(
        names_cov_life, names_cov_trans, names_cov_constr, n_model_params
    )
    start_arr = (
        None
        if settings.start is None
        else start_values(
            settings.start, count=n_model_params, parameters="model parameters"
        )
    )
    reg_lambdas = _validated_reg_lambdas(settings.reg_lambdas)

    cov_start = _validated_cov_start(settings.start_cov)
    candidates = _starting_points(
        layout, settings, start_arr, model_start, cov_start, reg_lambdas,
        baseline=lambda: fit_static_covariates(
            x=x, t_x=t_x, T=T, cov_life=cov_life, cov_trans=cov_trans,
            names_cov_life=layout.names_life, names_cov_trans=layout.names_trans,
            log_likelihood=log_likelihood,
            n_model_params=n_model_params, model_start=model_start,
            names_cov_constr=layout.constrained,
            search=SearchSettings(
                method=settings.method, maxiter=settings.maxiter, hessian=False,
                polish=settings.polish, options=settings.options,
            ),
        ),
    )

    def negative_log_likelihood(model, g_life, g_trans) -> float:
        """The function a standard error should refer to.

        For an unregularized fit that is the negative log-likelihood. For a
        regularized one it is eq. (13)'s **penalised mean** -- the objective the
        search actually minimised -- rather than the unpenalised sum this used
        to difference. Reporting curvature of a function the estimates were not
        obtained from is the inconsistency finding 9 of the outside review
        names: the two described different problems.

        The consequence is worth stating plainly, and the README's findings do:
        these are ridge standard errors. The penalty contributes curvature of
        its own, so they are smaller than an unregularized fit's and are not
        comparable with them, in the same way and for the same reason that
        CLVTools' regularized AIC is not comparable with its unregularized one.
        """
        return _penalised(
            -log_likelihood(model, g_life, g_trans, cov_life, cov_trans),
            g_life, g_trans, reg_lambdas, x.size,
        )

    def objective(v: NDArray[np.float64]) -> float:
        model, g_life, g_trans = layout.unpack(v)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(model, g_life, g_trans, cov_life, cov_trans)
        if not np.isfinite(value):
            return np.inf
        return _penalised(-value, g_life, g_trans, reg_lambdas, x.size)

    result = _search(objective, candidates, settings)

    model, g_life, g_trans = layout.unpack(result.x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        unpenalised = float(
            log_likelihood(model, g_life, g_trans, cov_life, cov_trans)
        )

    hess = None
    if settings.hessian:
        hess = _reported_hessian(
            layout, negative_log_likelihood, model, g_life, g_trans
        )

    return StaticCovResult(
        model=np.asarray(model, dtype=float),
        gamma_life=np.asarray(g_life, dtype=float),
        gamma_trans=np.asarray(g_trans, dtype=float),
        names_cov_life=layout.names_life,
        names_cov_trans=layout.names_trans,
        names_cov_constr=layout.constrained,
        log_likelihood=float(-result.fun),
        unpenalised_log_likelihood=unpenalised,
        converged=bool(result.success),
        n_customers=int(x.size),
        reg_lambdas=reg_lambdas,
        hessian=hess,
    )
