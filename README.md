# clvtools

A from-scratch, test-driven Python implementation of the **CLVTools** R package,
following Meierer, Bachmann, Näf, Schilter & Algesheimer, *"Estimating
Individual Customer Lifetime Values with R: The CLVTools Package"* (Journal of
Statistical Software, submission 5634).

Every section of the paper maps to a module; every module's docstrings carry the
paper's own equations and printed numbers as **doctests that pytest executes**,
so the documentation cannot drift from what the code returns.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Oracle harness, datasets, fixtures | ✅ done |
| 1 | `clvdata()` — transaction log → `(x, tₓ, T)` | ✅ done |
| 2 | Pareto/NBD — likelihood, PAlive, CET, DERT, PMF, MLE | next |
| 3 | Gamma-Gamma spending model | |
| 4 | `predict()` — combined CLV | |
| 5 | Time-invariant covariates | |
| 6 | Correlation, regularization, equality constraints | |
| 7 | Time-varying covariates | |
| 8 | BG/NBD and GGom/NBD families | |

## The oracle

The R package CLVTools is the reference implementation, and it reproduces every
number printed in the paper. Rather than testing only against the ~40 values the
paper prints, `tools/oracle/generate_fixtures.R` calls CLVTools' internal
per-customer entry points to dump expectations for **every** model expression, at
several parameter vectors — including points deliberately off the optimum, and
both branches of the `α ≥ β` / `α < β` split in the appendix likelihood.

That makes each equation testable on its own, before an optimiser exists to find
its maximum.

The generated fixtures are committed under `tests/fixtures/`, so **the test suite
needs no R**. R is needed only to re-baseline them:

```bash
./tools/setup_oracle.sh                             # installs into ./.Rlib only
R_LIBS=.Rlib Rscript tools/extract_data.R           # datasets  -> data/
R_LIBS=.Rlib Rscript tools/oracle/generate_fixtures.R   # -> tests/fixtures/
```

`setup_oracle.sh` never touches your system R library. CRAN's macOS binaries lag
the newest R release, so it falls back through recent R series; the 4.5 binary
installs and runs correctly under R 4.6.

The generator asserts its own conventions before writing anything — the C++ entry
points take **log-scale** model parameters, return the **negated** sum, and order
static-covariate arguments life-then-trans. Each fixture family is checked against
a public generic (`logLik()`, `coef()`) so a sign or ordering slip cannot ship as a
plausible-looking expectation.

## Verified against the paper

The committed fixtures reproduce the paper's printed values exactly:

| Quantity | Paper | Fixture |
|---|---|---|
| Pareto/NBD `r, α, s, β` | 1.4490, 48.6361, 0.5613, 46.8844 | identical |
| Gamma-Gamma `p, q, γ` | 3.099, 5.654, 56.504 | identical |
| Static-cov log-likelihood | −5821.0627 | −5821.0627 |
| Static-cov AIC | 11658.1254 | 11658.1254 |
| `predict()` table, S6.3.2 | 18 values across 3 customers | all identical |
| New-customer transactions | 2.218635 | 2.218635 |

Two numbers differ in the 5th significant digit: `mae.cet` (2.039620 vs the
paper's 2.039532) and `rmse.cet` (3.329425 vs 3.329395). These come from
optimiser convergence tolerance in CLVTools 0.12.1 versus the version used for
the paper, not from a discrepancy in the expressions — the underlying
coefficients agree to every printed digit.

## Usage

```python
from clvtools import ClvData, load_apparel_trans

clv = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
clv.customer_summary()   # Id, x, t_x, T, date_first_transaction
clv.spending_summary()   # Id, x, Spending
```

## Testing

```bash
uv run pytest                  # everything, including doctests in src/
uv run pytest -m paper         # only the published-number checks
uv run pytest -m oracle        # only checks against R CLVTools fixtures
uv run pytest --cov=clvtools --cov-report=term-missing
```

## Dependencies

NumPy, SciPy and pandas — nothing else. The R package is a test-time oracle
invoked out-of-process; nothing in `src/` depends on it.

## Data

`data/` holds the datasets bundled with CLVTools 0.12.1, exported to CSV:
`apparelTrans` (3,187 transactions from 600 customers, a single acquisition
cohort whose first purchase was 2005-01-02), `apparelStaticCov`,
`apparelDynCov`, `apparelDynCovFuture` and `cdnow`.

The paper's LaTeX source and PDF are in `arXiv-2602.09845v1/` and
`2602.09845v1.pdf`.
