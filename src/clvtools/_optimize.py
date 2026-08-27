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

import numpy as np

__all__ = ["options_for"]

#: Convergence tolerances well inside SciPy's defaults. These likelihoods are
#: flat near their optima -- the Pareto/NBD's ridge moves 3e-5 for a change of
#: 1e-10 in the objective -- so the default ``ftol`` stops early enough to shift
#: the third decimal of the estimates.
_TOLERANCES = {
    "L-BFGS-B": {"ftol": 1e-16, "gtol": 1e-14, "maxfun": 100_000},
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
    """
    options: dict = {"maxiter": maxiter}
    options.update(_TOLERANCES.get(method, {}))
    if method == "Nelder-Mead":
        options["initial_simplex"] = _initial_simplex(np.asarray(x0, dtype=float))
    if overrides:
        options.update(overrides)
    return options
