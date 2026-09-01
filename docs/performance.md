# Performance

Everything in this repo is gated on being *correct* — 901 tests, the paper's
numbers, the R package's numbers, oracle fixtures expression by expression —
and on being *tidy*: ruff, complexity, module size, 100% line coverage. Nothing
has ever asked whether it is *fast*. This document is the first pass at that
question. The invariants in "What a performance gate should look like" below
*are* now a gate — `tests/test_performance.py`, backlog item 7 — and every
profile below is **regenerable**: `tools/profile.py`, backlog item 8, emits
these tables as markdown. The first version of this document was assembled from
ad-hoc scripts that no longer existed by the time anyone could check them,
which is how one of its figures came to be wrong; see the correction under the
dyncov table.

Measured 2026-09-01 on an M-series Mac (macOS 26.6, arm64), CPython 3.12.11,
numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, against `apparelTrans` (600 customers)
and `cdnow` (2,357). Reproduce with:

    uv run python -X importtime -c "import clvtools"      # import cost
    uv run tools/benchmark.py                             # Appendix B run times
    uv run tools/profile.py                               # the profiles below

`tools/benchmark.py` already existed and reports run times in the shape of the
paper's Appendix B. That is a *report* of wall-clock, which is the right home
for numbers that move with hardware. What follows is the complementary
question: where does the time actually go, and is any of it avoidable.

`tools/profile.py` is its sibling and produced every profile table below. It
reports **call counts and shares of `tottime`** rather than seconds, so two
runs of the same code diff cleanly — a call count is a property of the code,
while seconds move with the machine. Both tools are reports: neither asserts
anything, neither is imported by a test, and neither runs in CI.

Two denominators are worth keeping straight, because the first version of this
document mixed them. cProfile charges every Python-level call, so a profiled
run is slower than an unprofiled one: `fit_pnbd` is 0.063 s on its own and
0.071 s under the profiler, and the dyncov likelihood 0.328 s against 0.486 s.
Shares below are of the **profiled** run's own total self-time, which is the
self-consistent figure; dividing a profiled `tottime` by an unprofiled
wall-clock, as the first version did, flatters every share by the difference.

---

## Summary

| Path | Cost | Verdict |
|---|---|---|
| `ClvData` + `summary()`, 2,357 customers | 0.103 s | linear, fine |
| `fit_pnbd` (600 customers, no Hessian) | 0.063 s | **at the floor** |
| `fit_pnbd_staticcov` (600, with Hessian) | 0.334 s | fine |
| `build_walks` (dyncov, 600) | 0.432 s, once per fit | fine |
| **dyncov `log_likelihood`, one evaluation** | **0.328 s** | **the one real finding** |

Unprofiled, median of five, from `tools/profile.py`. These are the numbers in
this document that move with the machine and the library versions; they are all
within ~10% of the first measurement, taken before numpy 2.5, scipy 1.18 and
pandas 3.0.

Two very different pictures, and the difference is the point of the document.

---

## The vectorised models are already at the floor

`fit_pnbd` takes 0.063 s over 290 likelihood evaluations, and **half its
self-time is inside `clvtools.special.hyp2f1_ratio`**:

| Function | Calls | Per likelihood evaluation | tottime |
|---|---|---|---|
| `clvtools/special.py:hyp2f1_ratio` | 580 | 2 | 49.4% |
| `clvtools/pnbd/aggregate.py:log_likelihood_ind` | 290 | 1 | 27.9% |
| `numpy/lib/_stride_tricks_impl.py:_broadcast_to` | 580 | 2 | 1.7% |
| `scipy/optimize/_numdiff.py:approx_derivative` | 58 | 0.2 | 1.3% |
| `clvtools/pnbd/fit.py:negative_ll` | 290 | 1 | 1.2% |
| `numpy/lib/_stride_tricks_impl.py:broadcast_arrays` | 870 | 3 | 1.2% |
| `method 'reduce' of 'numpy.ufunc' objects` | 1,051 | 3.6 | 0.8% |
| `numpy.array` | 3,831 | 13.2 | 0.8% |
| `numpy/lib/_stride_tricks_impl.py:_broadcast_shape` | 870 | 3 | 0.8% |
| `scipy/optimize/_numdiff.py:_dense_difference` | 58 | 0.2 | 0.8% |
| `clvtools/pnbd/aggregate.py:log_likelihood` | 290 | 1 | 0.7% |
| `numpy.asarray` | 3,431 | 11.8 | 0.6% |

Two calls per evaluation, exactly the `A_1` and `A_2` terms of Appendix A. That
half looks like a target until you measure what it is made of:

* the function is already vectorised: one `scipy.special.hyp2f1` call over the
  whole 600-element array, with a scalar Python series fallback only for
  entries SciPy returns non-finite;
* during a complete fit the fallback fires for **0 of 348,000 elements**;
* cProfile does not record numpy ufunc calls, so the `scipy.special.hyp2f1`
  evaluation is charged to `hyp2f1_ratio` itself — that half *is* the library
  call, not a wrapper around it. Measured separately, 580 bare
  `scipy.special.hyp2f1` calls on 600-element arrays cost 0.043 s, which is to
  say all of it.

So there is no overhead to remove. The only way to make a Pareto/NBD fit
meaningfully faster is to evaluate the likelihood *fewer* times — analytic
gradients, or better starting values — not to make an evaluation cheaper.

`fit_pnbd_staticcov` is the same picture with the same top two rows —
`hyp2f1_ratio` at 35.6% and `log_likelihood_ind` at 19.3%, both twice and once
per evaluation — over 1,271 evaluations rather than 290, because the Hessian's
differencing calls the likelihood too. Nothing below those two rows reaches
1.8%, and most of what is there is pandas column access rather than
arithmetic.

The descriptive layer is likewise sound. `summary()` is 85% of the cost of
building and describing a data set, and almost all of that is
`mean_interpurchase_times`, which loops over customers in Python. Measured
across four sizes it is flat:

| Customers | `summary()` | Per customer |
|---|---|---|
| 294 | 0.018 s | 0.061 ms |
| 589 | 0.031 s | 0.052 ms |
| 1,178 | 0.057 s | 0.048 ms |
| 2,357 | 0.103 s | 0.044 ms |

Linear, with a constant that is falling rather than rising. A Python loop is
not automatically a problem, and this one is not.

Only the last row of that table is regenerable — `tools/profile.py` profiles
the full CDNOW log, not a sweep of sizes — so it carries today's figure while
the first three are as first measured. The O(*n*) claim no longer rests on this
table anyway: `tests/test_performance.py` asserts flatness in *n* by operation
counts at two sizes.

---

## The time-varying covariate likelihood is Python-bound

This is the finding worth acting on. One evaluation of the dyncov
`log_likelihood` on 600 customers takes **0.328 s** at the parameters measured.
The fit itself is 13.5 minutes over the 1,870 evaluations the test docstring
records -- an average of **0.43 s** each, so the single-point figure below
understates the cost at the parameters the optimiser actually visits. Where
that 0.328 s goes — one evaluation, so every count below is per evaluation:

| Function | Calls | tottime |
|---|---|---|
| `clvtools/pnbd/dyncov.py:_hyp_beta_gt_alpha` | 38,542 | 35.7% |
| `clvtools/pnbd/dyncov.py:_f2` | 600 | 15.5% |
| `method 'reduce' of 'numpy.ufunc' objects` | 95,719 | 7.5% |
| `clvtools/pnbd/dyncov.py:d_i` | 39,755 | 7.0% |
| `clvtools/pnbd/dyncov.py:b_i` | 39,755 | 5.5% |
| `clvtools/pnbd/dyncov_walks.py:elem` | **155,418** | 4.5% |
| `clvtools/pnbd/dyncov_walks.py:sum_from_to` | 77,110 | 4.4% |
| `method 'sum' of 'numpy.ndarray' objects` | 95,718 | 3.4% |
| `clvtools/pnbd/dyncov_walks.py:first` | 99,647 | 2.9% |
| `clvtools/pnbd/dyncov_walks.py:n_elem` | 114,978 | 2.8% |
| `numpy/_core/_methods.py:_sum` | 95,718 | 2.3% |
| `clvtools/pnbd/dyncov.py:_hyp_term` | 39,754 | 2.1% |

**Correction, and the reason it is more interesting than a typo.** The first
version of this table gave `_hyp_beta_gt_alpha` 39,754 calls. That is
`_hyp_term`'s count — the dispatcher, which chooses an arm per covariate
interval. **Only `_hyp_term`'s 39,754 is a property of the data**; how it
splits between the arms is a property of *where in the parameter space the
profile was taken*, and both figures above are measurements of the same code:

| Profiled at | `beta > alpha` | `alpha >= beta` |
|---|---|---|
| CLVTools' fitted parameters (this table) | 38,542 | 1,212 |
| `r=0.5, alpha=10, s=0.6, beta=12, gamma=0.1` (the first version) | 39,754 | 0 |

So the original number was not wrong about its own run; it was wrong to present
a parameter-dependent split as if it were a fixed cost, and to attribute the
dispatcher's total to one arm. The `alpha >= beta` arm is reached at all only
near the optimum, which is exactly why the oracle fixtures deliberately
straddle that branch (`CLAUDE.md`) — a profile taken at a convenient starting
vector will never enter it, and would leave the arm looking like dead code.
Both this table and the tool that regenerates it now use the fitted
parameters. None of this was visible while the profile was ad-hoc; it is now.

That is over **600,000 calls into this package's own functions for one
number**, and 0.9 million counting numpy's. Unlike the plain model, almost none
of it is library work — it is interpreter overhead. Two things drive it:

1. **The hypergeometric arm is scalar.** `_hyp_beta_gt_alpha(r, s, x, alpha_1,
   beta_1, alpha_2, beta_2)` takes seven floats and is called once per covariate
   interval per customer — 39,754 intervals per evaluation across both arms,
   against *two* calls for the whole sample in the plain model.
2. **`Walk` uses numpy as a scalar container.** (`pnbd/dyncov_walks.py` since
   the split of backlog item 3; the counts did not move.) `elem(i)` is
   `float(self.values[i])`, called 155,418 times; `n_elem` is
   `int(self.values.size)`, called 114,978 times. Extracting one element from a
   numpy array and boxing it into a Python float is close to the most expensive
   way to touch an array, and this inner loop does nothing else.

The shape of the computation is a sum over covariate intervals, which is
exactly the shape that vectorises. The upside is plausibly large — but it is
**unproven**, and this document should not pretend otherwise. A spike that
vectorises `_f2` across the intervals of one customer would settle it in an
afternoon, and would tell us whether to do the harder version that vectorises
across customers too.

Two cautions for whoever picks this up. The dyncov likelihood is the most
intricate expression in the package and is checked against oracle fixtures
column by column at several parameter vectors — that suite is what makes this
safe to attempt, and it must stay green expression by expression, not merely in
total. And `CLAUDE.md` already records that whole-day arithmetic is load-bearing
here: working in nanoseconds shifts `d1` and `tjk` by ~4e-13 and breaks the
exact cancellation that makes `F2.2` vanish. Any rewrite has to preserve the
order of operations, not just the algebra.

---

## What a performance gate should look like

Not `assert elapsed < 2.0`. Every gate in this repo is deterministic — tests,
coverage, lint, size limits — and a wall-clock assertion would be the first one
that fails for reasons unrelated to the change under test. The first response
to a flaky gate is to loosen it, and `docs/backlog.md` explicitly forbids
loosening a limit to get green. A gate nobody trusts is worse than no gate.

Count operations instead. The regressions actually worth catching are
structural, and every one of them is visible without a clock:

* **`hyp2f1_ratio` stays vectorised** — called a bounded number of times per
  likelihood evaluation on arrays of length *n*, not *n* times on scalars.
  De-vectorising it is a ~100× regression that no current test would notice.
* **The scalar series fallback stays cold** — 0 of 348,000 elements today. If a
  change moves the optimiser into a region where it fires per customer, fits
  get much slower and nothing fails.
* **Likelihood evaluations per fit stay in a measured band** — 290 for
  `fit_pnbd` on the apparel data. This catches an optimiser or start-value
  regression, which is the thing that actually costs minutes.
* **Cost per customer stays flat** — asserted by comparing operation counts at
  two input sizes, so it is O(*n*) and not O(*n*²) by construction rather than
  by stopwatch.

Wall-clock still belongs in `tools/benchmark.py`, and *where* the time goes in
`tools/profile.py` — both reported, neither asserted.

## Next

- ~~`docs/backlog.md` item 7~~ — done: `tests/test_performance.py`, marker
  `performance`. The four invariants above, 1.0 s on every run, and each one
  demonstrated to fail against a deliberately broken implementation.
- ~~`docs/backlog.md` item 8~~ — done: `tools/profile.py`, 6 s, which emits
  every profile table above as markdown. Running it is what turned up the
  `_hyp_beta_gt_alpha` correction, which is the argument for it in one line.
- The dyncov vectorisation spike is deliberately *not* a backlog item yet. It
  now has item 7's counters to prove it changed nothing structural, but it
  should not be started at all unless someone wants the 13.5 minutes back.
