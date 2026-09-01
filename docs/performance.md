# Performance

Everything in this repo is gated on being *correct* — 888 tests, the paper's
numbers, the R package's numbers, oracle fixtures expression by expression —
and on being *tidy*: ruff, complexity, module size, 100% line coverage. Nothing
has ever asked whether it is *fast*. This document is the first pass at that
question. Nothing here is a gate yet; `docs/backlog.md` items 7 and 8 are what
would make it one.

Measured 2026-09-01 on an M-series Mac, Python 3.12, against `apparelTrans`
(600 customers) and `cdnow` (2,357). Reproduce with:

    uv run python -X importtime -c "import clvtools"      # import cost
    uv run tools/benchmark.py                             # Appendix B run times
    uv run python -m cProfile -s tottime <a script>       # the profiles below

`tools/benchmark.py` already existed and reports run times in the shape of the
paper's Appendix B. That is a *report* of wall-clock, which is the right home
for numbers that move with hardware. What follows is the complementary
question: where does the time actually go, and is any of it avoidable.

---

## Summary

| Path | Cost | Verdict |
|---|---|---|
| `ClvData` + `summary()`, 2,357 customers | 0.111 s | linear, fine |
| `fit_pnbd` (600 customers, no Hessian) | 0.065 s | **at the floor** |
| `fit_pnbd_staticcov` (600, with Hessian) | 0.33 s | fine |
| `build_walks` (dyncov, 600) | 0.454 s, once per fit | fine |
| **dyncov `log_likelihood`, one evaluation** | **0.290 s** | **the one real finding** |

Two very different pictures, and the difference is the point of the document.

---

## The vectorised models are already at the floor

`fit_pnbd` takes 0.065 s and spends **57% of it inside
`clvtools.special.hyp2f1_ratio`** — 0.037 s across 580 calls, two per
likelihood evaluation. That looks like a target until you measure what it is
made of:

* the function is already vectorised: one `scipy.special.hyp2f1` call over the
  whole 600-element array, with a scalar Python series fallback only for
  entries SciPy returns non-finite;
* during a complete fit the fallback fires for **0 of 348,000 elements**;
* 580 bare `scipy.special.hyp2f1` calls on 600-element arrays cost 0.043 s on
  their own — which is to say, all of it.

So there is no overhead to remove. The only way to make a Pareto/NBD fit
meaningfully faster is to evaluate the likelihood *fewer* times — analytic
gradients, or better starting values — not to make an evaluation cheaper.

The descriptive layer is likewise sound. `summary()` is 85% of the cost of
building and describing a data set, and almost all of that is
`mean_interpurchase_times`, which loops over customers in Python. Measured
across four sizes it is flat:

| Customers | `summary()` | Per customer |
|---|---|---|
| 294 | 0.018 s | 0.061 ms |
| 589 | 0.031 s | 0.052 ms |
| 1,178 | 0.057 s | 0.048 ms |
| 2,357 | 0.111 s | 0.047 ms |

Linear, with a constant that is falling rather than rising. A Python loop is
not automatically a problem, and this one is not.

---

## The time-varying covariate likelihood is Python-bound

This is the finding worth acting on. One evaluation of the dyncov
`log_likelihood` on 600 customers takes **0.290 s**, which puts the
~17-minute fit at roughly **3,500 evaluations**. Where that 0.29 s goes:

| Function | Calls **per evaluation** | `tottime` |
|---|---|---|
| `_hyp_beta_gt_alpha` | 39,754 | 0.142 s |
| `_f2` | 600 | 0.077 s |
| `d_i` | 39,755 | 0.035 s |
| `b_i` | 39,755 | 0.028 s |
| `Walk.elem` | **155,418** | 0.023 s |
| `Walk.sum_from_to` | 77,110 | 0.022 s |
| `Walk.n_elem` | 114,978 | 0.014 s |
| `Walk.first` | 99,647 | 0.014 s |

That is roughly **half a million Python-level calls for one number**. Unlike
the plain model, none of this is library cost — it is interpreter overhead. Two
things drive it:

1. **The hypergeometric arm is scalar.** `_hyp_beta_gt_alpha(r, s, x, alpha_1,
   beta_1, alpha_2, beta_2)` takes seven floats and is called once per covariate
   interval per customer — 39,754 times per evaluation, against *two* calls for
   the whole sample in the plain model.
2. **`Walk` uses numpy as a scalar container.** `elem(i)` is
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

Wall-clock still belongs in `tools/benchmark.py`, reported and not asserted.

## Next

- `docs/backlog.md` item 7 — the deterministic invariants above, as tests.
- `docs/backlog.md` item 8 — a committed profile report, so hot spots are
  reviewable rather than rediscovered.
- The dyncov vectorisation spike is deliberately *not* a backlog item yet. It
  should not be started until item 7 exists to prove it changed nothing, and
  not at all unless someone wants the 17 minutes back.
