#!/usr/bin/env python
"""Run times for the Pareto/NBD, in the shape of the paper's Appendix B.

Appendix B benchmarks CLVTools across sample sizes and estimation periods, on
data simulated from r = 1, alpha = 0.5, s = 1, beta = 0.5, with L-BFGS-B and no
Hessian, reporting the median of three runs. This does the same for this
package so the two are comparable:

    uv run python tools/benchmark.py                 # 1,000 and 10,000
    uv run python tools/benchmark.py --sizes 1000 10000 100000
    uv run python tools/benchmark.py --repeats 5

Timings are not tests. Nothing here is asserted; the numbers depend on the
machine, and the paper's own were taken on a different one.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from clvtools import ClvDataStaticCov
from clvtools._staticcov import fit_static_covariates
from clvtools.pnbd import fit_pnbd
from clvtools.pnbd.staticcov import log_likelihood as staticcov_log_likelihood

#: Appendix B: "We used the following base parameters for the Pareto/NBD model
#: for this: r = 1, alpha = 0.5, s = 1, beta = 0.5."
BASE = {"r": 1.0, "alpha": 0.5, "s": 1.0, "beta": 0.5}

#: Its scenarios, in weeks.
PERIODS = (26, 52, 104)


def simulate(n: int, T: float, seed: int, base: dict[str, float] = BASE):
    """Draw ``(x, t_x, T)`` for ``n`` customers from the Pareto/NBD.

    The generative story of S3.2, run forwards: each customer draws a purchase
    rate and an attrition rate from their gamma distributions, dies at an
    exponential time, and purchases as a Poisson process until then.
    """
    rng = np.random.default_rng(seed)
    lam = rng.gamma(base["r"], 1.0 / base["alpha"], size=n)
    mu = rng.gamma(base["s"], 1.0 / base["beta"], size=n)
    death = rng.exponential(1.0 / mu)
    observed = np.minimum(death, T)

    counts = rng.poisson(lam * observed)
    x = np.zeros(n, dtype=float)
    t_x = np.zeros(n, dtype=float)
    # Only customers with at least one transaction have a last one; their
    # purchase times are uniform on the interval they were alive for.
    has_any = counts > 0
    for i in np.flatnonzero(has_any):
        times = np.sort(rng.uniform(0.0, observed[i], size=counts[i]))
        x[i] = counts[i] - 1
        t_x[i] = times[-1] - times[0] if counts[i] > 1 else 0.0
    return x, t_x, np.full(n, float(T))


def _time(call, repeats: int) -> float:
    """Median wall-clock seconds over ``repeats`` runs, as Appendix B reports."""
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def _static_fit(x, t_x, T, covariate):
    """One dummy-coded time-invariant covariate on both processes."""
    design = covariate.reshape(-1, 1)

    def objective(model, g_life, g_trans, cov_life, cov_trans):
        return staticcov_log_likelihood(
            x, t_x, T, model[0], model[1], model[2], model[3],
            g_life, g_trans, cov_life, cov_trans,
        )

    return fit_static_covariates(
        x=x, t_x=t_x, T=T, cov_life=design, cov_trans=design,
        names_cov_life=["Dummy"], names_cov_trans=["Dummy"],
        log_likelihood=objective, n_model_params=4,
        model_start=(1.0, 1.0, 1.0, 1.0), method="L-BFGS-B",
        maxiter=10_000, hessian=False, polish=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000])
    parser.add_argument("--periods", type=int, nargs="+", default=list(PERIODS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print(f"Pareto/NBD, {BASE}, L-BFGS-B, no Hessian, "
          f"median of {args.repeats}\n")
    print(f"{'customers':>10} {'weeks':>6} {'no covariate':>14} "
          f"{'time-invariant':>16}")
    for n in args.sizes:
        for weeks in args.periods:
            x, t_x, T = simulate(n, float(weeks), args.seed)
            rng = np.random.default_rng(args.seed)
            covariate = rng.integers(0, 2, size=n).astype(float)
            plain = _time(
                lambda: fit_pnbd(x, t_x, T, hessian=False), args.repeats
            )
            static = _time(
                lambda: _static_fit(x, t_x, T, covariate), args.repeats
            )
            print(f"{n:>10,} {weeks:>6} {plain:>13.2f}s {static:>15.2f}s")


if __name__ == "__main__":
    main()
