r"""Shared optimiser setup for the maximum-likelihood fits.

Every model here estimates strictly positive parameters, and every one does it
by searching over their logarithms rather than constraining the search. That is
also what CLVTools' C++ entry points expect -- they take log parameters and
exponentiate on the way in.

Working in log space has one consequence worth handling centrally. The default
start is all parameters at 1, so the search starts at the origin, and SciPy's
Nelder-Mead builds its initial simplex by perturbing each coordinate by 5% --
falling back to a fixed 0.00025 when a coordinate is exactly zero. Every
coordinate is zero here, so the simplex is microscopic in a space where the
answer can be several units away. It then converges, reporting success, on
whatever local optimum happens to be nearest.

That is not hypothetical: on the Gamma-Gamma model of S6.2.3 it settles at
``p = 128610, q = 2.645, gamma = 0.001`` with a log-likelihood of -1705.06,
against -1670.66 at the published estimates. :func:`options_for` gives
Nelder-Mead a simplex spanning a factor of e in each direction instead, which
finds the published optimum.
"""

from __future__ import annotations

import functools
import inspect

import numpy as np

__all__ = ["options_for"]

#: Convergence tolerances well inside SciPy's defaults. These likelihoods are
#: flat near their optima -- the Pareto/NBD's ridge moves 3e-5 for a change of
#: 1e-10 in the objective -- so the default ``ftol`` stops early enough to shift
#: the third decimal of the estimates.
#:
#: Tight, but not tighter than double precision, which is a distinction the
#: first values here got wrong. L-BFGS-B tests ``(f_k - f_k+1) / max(|f_k|,
#: |f_k+1|, 1) <= ftol``, so ``ftol`` is a *relative* reduction and SciPy turns
#: it into ``factr = ftol / eps``. At ``ftol = 1e-16`` that is ``factr = 0.45``:
#: a demand for better than machine precision, which no line search can report
#: satisfying. ``gtol = 1e-14`` was unreachable for the same reason -- on an
#: objective of order 5e3, the gradient cannot be resolved below about 1e-9 --
#: so the only ways out were the impossible ``ftol`` test or a failed line
#: search. On macOS/ARM the reduction reached exactly zero and the test passed;
#: on x86-64 Linux the line search failed first and the *same optimum*, to
#: twelve significant figures, was returned with ``success = False``. See the
#: README's findings. ``1e-14`` is ``factr = 45``, still 200,000x tighter than
#: SciPy's default ``factr = 1e7``, and reachable on both.
_TOLERANCES = {
    "L-BFGS-B": {"ftol": 1e-14, "gtol": 1e-10, "maxfun": 100_000},
    "Nelder-Mead": {"xatol": 1e-12, "fatol": 1e-12, "maxfev": 100_000},
}

#: Half-width of the initial Nelder-Mead simplex, in log-parameter units. One
#: unit is a factor of e in the parameter itself, which is wide enough to see
#: past a nearby local optimum without starting the search somewhere absurd.
_SIMPLEX_STEP = 1.0


def _initial_simplex(x0: np.ndarray) -> np.ndarray:
    r"""An ``(n+1, n)`` simplex around ``x0``, one vertex stepped per axis."""
    n = x0.size
    simplex = np.repeat(x0[None, :], n + 1, axis=0)
    simplex[1:] += np.eye(n) * _SIMPLEX_STEP
    return simplex


#: Option keys a method actually reads, asked of SciPy rather than listed here.
#:
#: A hard-coded list is a second copy of SciPy's contract and would drift from
#: it; the keyword parameters of the ``_minimize_*`` implementations *are* the
#: contract. Looked up once per method and cached, since it is a signature
#: inspection rather than anything numerical.
_NOT_AN_OPTION = frozenset(
    {"fun", "func", "x0", "args", "jac", "hess", "hessp", "bounds", "constraints",
     "callback", "return_all", "unknown_options"}
)


@functools.cache
def _accepted_options(method: str) -> frozenset[str]:
    """The option keys ``method`` reads, from its SciPy implementation."""
    from scipy import optimize
    from scipy.optimize import _minimize

    implementations = {
        "Nelder-Mead": _minimize._minimize_neldermead,
        "L-BFGS-B": optimize._lbfgsb_py._minimize_lbfgsb,
    }
    implementation = implementations.get(method)
    if implementation is None:
        # An unrecognised method: nothing to check against, so check nothing
        # rather than refusing a method SciPy may well accept.
        return frozenset()
    return frozenset(
        name
        for name, parameter in inspect.signature(implementation).parameters.items()
        if parameter.default is not parameter.empty and name not in _NOT_AN_OPTION
    )


#: Keys that are valid for one method and mean the same thing under another
#: name for the other. `maxfun` and `maxfev` are the pair that matters: both
#: cap function evaluations, L-BFGS-B spells it the first way and Nelder-Mead
#: the second, so passing one where the other belongs looks entirely right.
_NEAR_MISSES = {"maxfun": "maxfev", "maxfev": "maxfun"}


def _reject_unknown_options(method: str, overrides: dict) -> None:
    """Refuse an override key the method will silently ignore.

    SciPy warns here -- ``UserWarning: Unknown solver options: maxfun`` -- and
    then drops the key. R errors, and so does this: an option that was asked for
    and not applied is a fit that did something other than what was requested,
    which is the one thing an escape hatch must not do quietly. See finding 20
    of ``docs/review-2026-09-02.md`` and spec ``V-03``.
    """
    accepted = _accepted_options(method)
    if not accepted:
        return
    unknown = sorted(set(overrides) - accepted)
    if not unknown:
        return
    hints = []
    for key in unknown:
        near = _NEAR_MISSES.get(key)
        if near is not None and near in accepted:
            hints.append(f"{key!r} (did you mean {near!r}? {method} spells it that way)")
        else:
            hints.append(repr(key))
    raise ValueError(
        f"{method} does not accept {', '.join(hints)}. SciPy would warn and "
        f"ignore these, leaving the fit unbounded by an option that was asked "
        f"for. It accepts: {', '.join(sorted(accepted))}."
    )


def options_for(
    method: str,
    maxiter: int,
    x0: np.ndarray,
    overrides: dict | None = None,
) -> dict:
    """Optimiser options: tightened tolerances, then any caller overrides.

    ``overrides`` is the escape hatch S6.2.1 describes for CLVTools: "If
    questions on optimality arise, the parameter ``optimx.args`` allows the
    optimization routine to be controlled."

    A key ``method`` does not read is refused rather than passed on. SciPy
    warns and drops it, which means a caller who asked for a bound gets a fit
    that ran without one -- and the warning is easy to miss.
    """
    options: dict = {"maxiter": maxiter}
    options.update(_TOLERANCES.get(method, {}))
    if method == "Nelder-Mead":
        options["initial_simplex"] = _initial_simplex(np.asarray(x0, dtype=float))
    if overrides:
        _reject_unknown_options(method, overrides)
        options.update(overrides)
    return options
