# Performance

Everything in this repo is gated on being *correct* — 1,605 tests, the paper's
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

## What the suite itself costs

A separate question from what the *library* costs, and the one a contributor
feels: **4:50 plain and 7:40 with coverage**, on the machine above, for 1,605
tests. That is where it stands after the audit rounds, which added 453 tests.

The distribution is very uneven, and worth knowing before optimising anything:

| | share | what it is |
|---|---|---|
| `TestBemmaorGlady2012` setup | 56 s | one GGompertz/NBD fit on CDNOW's 2,357 customers, for spec `F-07` |
| `TestTheThreeViewsOfAFitAgreeOnItsNames` setup | 15 s | four static-covariate fits |
| the hourly GGom/NBD fit (`F-12`) | 10 s | 600 customers, one `quad` per customer per evaluation |
| everything else | ~3:30 | 1,600 tests |

Two of the top three are the GGompertz/NBD, and for one reason: its likelihood
runs `scipy.integrate.quad` **once per customer per evaluation**, where the
other three families are closed forms vectorised over the cohort. That is
inherent to the model as the paper states it, not a defect, and it is why
`tools/profile.py`'s tables put it in a class of its own.

**One of those was 99.4 s until it was measured.** The four covariate fits above
existed to check that `coef()`, `vcov()` and `summary()` name the same
parameters in the same order — a claim about where names go, fixed when the
params object is built, and nothing to do with where the optimiser lands. The
GGom/NBD alone was 98.2 s of it, 14% of the whole suite, spent reaching an
optimum no assertion read and whose Hessian this package already records as
untrustworthy. Bounding *that one* fit to a single iteration costs nothing the
tests check. Bounding the other three as well saved a further second and would
have cost something real: after one iteration their Hessians are not positive
definite, so the "no `NaN`" assertion beside them would have been made at a
point where the curvature is meaningless. The fixture says so at the site.

That is the same trade as backlog item 27, which found parity tests fitting
three families over 600 customers to compare *shapes*, and the same trap: a
fixture that fits something is easy to write and its cost is invisible until
someone runs `--durations`.

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
| **dyncov `log_likelihood`, one evaluation** | **0.328 s → 0.097 s** | **was the one real finding; now fixed** |

Unprofiled, median of five, from `tools/profile.py`. These are the numbers in
this document that move with the machine and the library versions; they are all
within ~10% of the first measurement, taken before numpy 2.5, scipy 1.18 and
pandas 3.0.

Two very different pictures, and the difference is the point of the document.

---

## The vectorised models are already at the floor

`fit_pnbd` takes 0.051 s over 210 likelihood evaluations, and **half its
self-time is inside `clvtools.special.hyp2f1_ratio`**:

| Function | Calls | Per likelihood evaluation | tottime |
|---|---|---|---|
| `clvtools/special.py:hyp2f1_ratio` | 420 | 2 | 53.5% |
| `clvtools/pnbd/aggregate.py:log_likelihood_ind` | 210 | 1 | 25.4% |
| `numpy/lib/_stride_tricks_impl.py:_broadcast_to` | 420 | 2 | 1.6% |
| `scipy/optimize/_numdiff.py:approx_derivative` | 42 | 0.2 | 1.3% |
| `clvtools/pnbd/fit.py:negative_ll` | 210 | 1 | 1.1% |
| `numpy/lib/_stride_tricks_impl.py:broadcast_arrays` | 630 | 3 | 1.1% |
| `numpy/lib/_stride_tricks_impl.py:_broadcast_shape` | 630 | 3 | 0.8% |
| `scipy/optimize/_numdiff.py:_dense_difference` | 42 | 0.2 | 0.8% |
| `method 'reduce' of 'numpy.ufunc' objects` | 763 | 3.6 | 0.7% |
| `numpy.array` | 2,775 | 13.2 | 0.7% |
| `clvtools/pnbd/aggregate.py:log_likelihood` | 210 | 1 | 0.6% |
| `numpy.asarray` | 2,487 | 11.8 | 0.6% |

Two calls per evaluation, exactly the `A_1` and `A_2` terms of Appendix A. That
half looks like a target until you measure what it is made of:

* the function is already vectorised: one `scipy.special.hyp2f1` call over the
  whole 600-element array, with a scalar Python series fallback only for
  entries SciPy returns non-finite;
* during a complete fit the fallback fires for **0 of 252,000 elements**;
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

## The time-varying covariate likelihood was Python-bound

This was the finding worth acting on, and backlog item 9 acted on it. What
follows is the profile as it stood, then what the rewrite did to it.

One evaluation of the dyncov `log_likelihood` on 600 customers took **0.328 s**
at the parameters measured. The fit itself was 13.5 minutes over the 1,870
evaluations the test docstring records -- an average of **0.43 s** each, so the
single-point figure below understated the cost at the parameters the optimiser
actually visits. Where that 0.328 s went — one evaluation, so every count below
is per evaluation:

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
exactly the shape that vectorises.

## What vectorising it actually bought

**One evaluation, 0.328 s to 0.097 s — and the fit only 13:27 to 10:07.** The
gap between those two numbers is the more useful half of this section.

`_f2`'s inner loop — one trip through `b_i`, `d_i` and `_hyp_term` per covariate
interval — became four array operations per customer. Profiled at CLVTools'
fitted parameters, the same point as the table above:

| Function | Calls | tottime |
|---|---|---|
| `clvtools/pnbd/dyncov.py:_hyp_beta_gt_alpha` | 1,481 | 45.9% |
| `clvtools/pnbd/dyncov.py:_f2_middle` | 599 | 7.5% |
| `clvtools/pnbd/dyncov.py:_hyp_alpha_ge_beta` | 904 | 5.7% |
| `clvtools/pnbd/dyncov_walks.py:customers` | 1 | 4.3% |
| `clvtools/pnbd/dyncov.py:log_likelihood_customer` | 600 | 4.2% |
| `method 'reduce' of 'numpy.ufunc' objects` | 9,425 | 3.4% |
| `clvtools/pnbd/dyncov.py:_hyp_terms` | 593 | 3.2% |
| `numpy/_core/fromnumeric.py:_wrapreduction_any_all` | 4,770 | 2.4% |
| `clvtools/pnbd/dyncov.py:_f2` | 600 | 2.3% |
| `clvtools/pnbd/dyncov.py:_prefix_sums` | 1,186 | 1.5% |
| `clvtools/pnbd/dyncov_walks.py:n_elem` | 12,383 | 1.3% |
| `method 'cumsum' of 'numpy.ndarray' objects` | 1,779 | 1.3% |

Read that against the table above it. The 39,754 scalar arm dispatches became
2,385 — 3.98 per customer rather than 66.3 — while still evaluating **39,754
hypergeometrics**, one per covariate interval as before. `Walk.elem` and
`Walk.sum_from_to`, 232,528 calls between them, are gone from the profile
entirely: the growing prefix each of `b_i` and `d_i` re-summed for every
interval is now one `cumsum` per walk.

### The single-point profile was measuring the wrong place

A 3.4x speedup that turns 13:27 into 10:07 is a 1.33x fit. The arithmetic only
works if the evaluations the optimiser performs are not like the one profiled,
and they are not. Timing every one of the 1,925 evaluations of a complete fit,
in order, mean seconds per decile:

| Decile of the search | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Seconds per evaluation | 0.065 | 0.066 | 0.071 | 0.222 | 0.457 | 0.454 | 0.453 | 0.453 | 0.453 | 0.452 |

The search runs cheap for a third of its length and then steps up sevenfold and
stays there. The slowest vector it visits is
`life.High.Season = -8.12` — which `test_reaches_at_least_the_oracles_optimum`
already names as the coefficient where this implementation's optimum differs
most from CLVTools' (-8.12 against -2.48). Once the attrition covariate runs
that far out, the hypergeometrics are evaluated at arguments SciPy has to work
much harder for, and profiling *there* gives a different answer entirely:

| Parameter vector | Before | After | Speedup |
|---|---|---|---|
| `fit_pnbd_dyncov`'s start, `(1, 1, 1, 1)`, γ = 0.1 | 0.277 s | 0.054 s | **5.1x** |
| `r=0.5, alpha=10, s=0.6, beta=12` (the original profile) | 0.279 s | 0.056 s | **5.0x** |
| CLVTools' fitted parameters | 0.322 s | 0.097 s | **3.3x** |
| The vector the search dwells on | 0.676 s | 0.449 s | **1.5x** |

At that last vector, **83.8% of self-time is inside `_hyp_beta_gt_alpha` over
948 calls**. That is `scipy.special.hyp2f1` itself — cProfile charges a ufunc to
its caller, the same accounting note this document makes about `hyp2f1_ratio`
above. Nothing is left to vectorise there; the interpreter overhead is already
gone and what remains is the library doing real work.

So the correction to the section above is not that it measured wrong, but that
it generalised from one point. **The dyncov likelihood is Python-bound in the
easy part of the parameter space and library-bound in the part the optimiser
spends two thirds of its time in.** Vectorising was still worth doing — it is
3.3-5.1x on every evaluation a *user* makes, which is what `predict()`,
`probability_alive()` and every fixture test pay — but the fit was never going
to fall by 3.4x, and no profile taken at a single vector could have said so.
The next lever on the fit is not this one: it is either the cost of `hyp2f1` in
that region or keeping the search out of it.

### What it cost, in the last two digits

The rewrite was held to the standard `CLAUDE.md` sets — preserve the order of
operations, not merely the algebra — and it very nearly is. Against the scalar
implementation at both oracle parameter vectors, **27 of the 30 per-customer
intermediates are bit-identical**, including `F2.1` and `F2.2`. Three move:
`F2.3` by up to 2.2e-15 relative, and `F2` and `LL` because they contain it —
the sample log-likelihood at CLVTools' fitted parameters shifts from
`-5752.936720222679` to `-5752.9367202226795`.

One thing causes all of it, and it was isolated by rerunning with the prefixes
computed the slow way: `Walk.sum_from_to` calls `ndarray.sum`, which adds
**pairwise**, while `np.cumsum` adds **left to right**. On a walk of more than
eight intervals those differ in the last bits. Substitute a pairwise
`_prefix_sums` and all thirty columns come back bit-identical at both vectors,
which is the proof that nothing else moved: the batched hypergeometric is
elementwise identical to the scalar dispatcher, including on a mixed batch, and
`cumsum` over the terms reproduces the old `+=` loop exactly.

`cumsum` is what is kept, for two measured reasons:

* The disagreement it introduces is **smaller than the disagreement that was
  already there**. Against the R fixtures, `F2.3` differs by 2.7e-15 (`mle`) and
  2.3e-13 (`offset`) — an order of magnitude more than the 2e-15 between the two
  summation orders. Neither order is a specification: pairwise is an artefact of
  `ndarray.sum`'s blocking, and CLVTools accumulates sequentially in C++, so if
  either is the reference's order it is `cumsum`'s. Measured, `cumsum` is
  marginally the closer of the two on both vectors.
* Bit-exactness costs 37% of a single evaluation — 0.132 s against 0.097 s,
  because the pairwise version keeps the quadratic re-summation.

It is a close call, and the honest version of it is that exactness is nearly
free *for the fit*: measured over a complete fit the two are 0.320 s and 0.312 s
per evaluation, because in the region the search dwells in `hyp2f1` dominates
both. The 37% is paid on single evaluations, which is where users live. The
choice is one function, `_prefix_sums`, and its docstring says as much —
swapping it back is a four-line change.

One caution on reading fit wall-clocks here at all. The two variants ran to
convergence in 1,925 and 1,562 evaluations (600.6 s and 499.7 s), against 1,870
before. That spread is the optimiser's path on a very flat likelihood, not a
property of either implementation, and it is why the per-evaluation figures
above are the ones to compare.

`F2.2` was deliberately left on the scalar path. Its two hypergeometrics are
evaluated at arguments that are equal by construction, and that exact
cancellation runs through `b_i` and `d_i` at `i = 1` and `i = k_T` — neither of
which the rewrite touches. It is bit-identical, as it must be.

### What guards it now

Nothing in the suite would have noticed this going back: the oracle fixtures
check the numbers, and a scalar loop produces the same ones.
`tests/test_performance.py::TestDyncovStaysVectorised` counts the shape of the
call instead — arm dispatches per customer (≤ 4, measured 3.98; the scalar loop
gives 66.3 and was checked to fail), one hypergeometric per covariate interval
so the batching cannot be dropping work, and **both arms non-empty at the
fitted parameters**. That last one is the reason the gate runs at
CLVTools' fitted point and not at a convenient starting vector: it is the only
place the `alpha >= beta` arm is entered at all.

### What backlog item 28 cost, and why it was still the right trade

Combining the `F_2` terms in log space rather than as values took one
evaluation from **0.104 s to 0.132 s — 26%** (same machine, same parameters,
the best of three runs of five, after a warm-up). The extra work is per term and unavoidable:
`_log_diff_exp` replaces two `exp` and a subtraction with a `maximum`, a
`minimum`, an `expm1`, a `log`, a `sign` and two `where`, and `_scale_by_ratio`
and `_signed_logsumexp` add a `log` and a reduction on top.

Hoisting the `errstate` context managers out of the per-term loops was tried
first, because CLAUDE.md records that the context manager alone was once an
eighth of the runtime here. It bought **nothing measurable** (0.132 s either
way) and was reverted: the profiler's call count made it look like the cost,
and it is not — the array work is.

The trade is not close. The 26% buys a likelihood that is right for a customer
whose `F_2` is below float64, where it was previously wrong by 225 log-units
and reported `PAlive` as exactly 1.0.

No fit time is quoted for it, for the reason this document already gives above:
a dyncov fit's wall clock is the optimiser's path on a very flat likelihood,
not a property of the implementation. Measured anyway, `-m dyncov_fit` passed
in 7:31 against the 10:07 recorded for item 9 — *faster*, while sharing the
machine with other work. The per-evaluation figure is the one that means
something.

---

### Backlog item 14: why `hyp2f1` is slow there, and what actually fixes it

Measured 2026-09-03, same machine. The claim above reproduces: at
`life.High.Season = -8.12` one evaluation is 0.480 s against 0.120 s at
CLVTools' fitted parameters, and **85.2% of it is inside `scipy.special.hyp2f1`**
(the section above said 83.8%).

**The mechanism is `z`, and nothing else.** Capturing every argument at both
vectors: the *same* 79,508 hypergeometrics in the same 4,770 array calls, and
`a` and `b` with identical ranges (`a` 3.99–24.99, `b` 1.98–22.98). Only the
argument moves:

| | max `z` | `z > 0.9` | `z > 0.99` | `z > 0.999` | `hyp2f1` alone |
|---|---|---|---|---|---|
| CLVTools' fit | 0.9844 | 27.6% | 0.0% | 0.0% | 0.045 s |
| the dwell vector | 0.9999 | 27.6% | 27.6% | 27.6% | 0.409 s |

The same 27.6% of calls — one arm's worth — is pushed from `z ≈ 0.98` to
`z ≈ 0.9999` as the attrition coefficient runs out, and SciPy's `hyp2f1` costs
9x more there. Nothing else about the problem changes.

**Both exact rewrites fail, and they fail as a pair.** `c = a + 1` in both arms,
which is a strong special case, so two classical routes are available:

* **The `1-z` connection formula** (DLMF 15.8.4) collapses neatly here, because
  `2F1(a,b;b;x) = (1-x)^-a` kills one of its two hypergeometrics outright:
  `2F1(a,b;a+1;z) = Γ(a+1)Γ(1-b)/Γ(a+1-b) · z^-a + (1-z)^(1-b) · a/(b-1) ·
  2F1(1,a+1-b;2-b;1-z)`. The residual argument is `1-z ≈ 1e-4`, so it converges
  in a few terms: **4.8x faster, 0.403 s → 0.083 s.** It is also useless — the
  maximum relative error is **5.7e35**. With `b` up to 23 and `1-z ≈ 1e-4`, the
  factor `(1-z)^(1-b)` reaches 1e88 while the answer is O(1). That is real
  cancellation between the two terms, not a representation problem, so carrying
  it in log space with signs — the fix that worked for item 28 — cannot help.
* **Euler's transformation** has one term and therefore cannot cancel, and it
  collapses just as neatly: `a+1-b = s+1` in the `β > α` arm, so
  `2F1(a,b;a+1;z) = (1-z)^(1-b) · 2F1(1, s+1; a+1; z)`. It is accurate to
  5e-15 — and **not faster**: 0.407 s against 0.403 s. SciPy is evidently
  already doing this.

Read together those two are the finding: *the fast transformation and the
accurate one are the same transformation, and it cannot be both.* SciPy is slow
at `z → 1` here for a reason, and a cheaper hypergeometric is not sitting
unclaimed in the algebra.

**The lever that does work is not analytic.** The covariates are categorical, so
`exp(γ'x)` takes very few values, and so does `z`. Of the 79,508 hypergeometrics
one evaluation asks for, there are **5,303 distinct `(a, b, z)` triples — 93.3%
are duplicates**, from only 1,570 distinct `z` and 31 distinct `(a, b)`.
Deduplicating is *bit-exact* (`np.array_equal`, not a tolerance: it is the same
function on the same arguments) and, with the cost of `np.unique` included:

| | evaluation | `hyp2f1` | share | deduplicated | projected evaluation |
|---|---|---|---|---|---|
| CLVTools' fit | 0.120 s | 0.045 s | 37.5% | 0.059 s | 0.135 s |
| the dwell vector | 0.480 s | 0.409 s | 85.2% | 0.067 s | 0.138 s |

**It is not a local change.** Within a single customer's call, *zero* percent is
removable — every `z` a customer sees is distinct. The 93.3% is entirely
*across* customers, and the median call is one element wide. So the win needs
the likelihood batched over the cohort, exactly as item 9 batched it over
covariate intervals inside a customer, one level up.

**Which overturns the prediction this document closes with.** "Vectorising
`log_likelihood_customer` across customers as well — the obvious next refactor —
would not touch it, and the numbers above are the reason not to start it." That
was reasoned from the profile: cross-customer batching removes Python dispatch,
and Python dispatch is not what is expensive there. The reasoning is sound and
the conclusion is wrong, because batching does something the profile could not
show — it puts the duplicate arguments *in the same array*, where they can be
collapsed before SciPy sees them. Weighting the two vectors 1:2 as the decile
table does, 0.360 s → 0.137 s is **2.6x on the fit**, ~10:07 → ~4 minutes.

That figure is a projection and is labelled as one: the restructure has not been
done, and it would additionally remove 4,770 NumPy dispatches per evaluation
that the projection gives it no credit for. Carried as backlog item 30.

### Backlog item 30: the duplicates, collapsed without a restructure

Item 14 projected 2.6x from deduplicating the hypergeometrics and said it needed
the likelihood batched over the cohort, because the duplication is *across*
customers. It does not. A memo shared by one evaluation reaches exactly the same
duplicates, and it is both cheaper and less invasive than the restructure.

`_hyp_alpha_ge_beta` and `_hyp_beta_gt_alpha` both want one shape,
:math:`{}_2F_1(a, b; a{+}1; z)`. Both now go through `_hyp2f1`, which consults a
`ContextVar` memo when one is open, gathers the misses, and evaluates them in a
**single** vectorised call — so the batching item 9 bought is not handed back one
element at a time. `log_likelihood_ind` opens the memo once for the whole sweep,
beside the `errstate` context manager that is already hoisted there.

| | evaluation before | after | `hyp2f1` before | after |
|---|---|---|---|---|
| CLVTools' fitted parameters | 0.120 s | **0.110 s** | 0.045 s | 0.032 s |
| the vector the search dwells on | 0.480 s | **0.119 s** | 0.409 s | 0.042 s |

The dwell vector is now as cheap as the easy one, which was the whole object of
items 14 and 30: two thirds of a fit was being spent at 4x the cost of the other
third, and that gap is gone rather than narrowed.

**Why a memo beats the `np.unique` dedup item 14 measured.** That approach sorted
79,508×3 values to find 5,303 distinct ones, and the sort cost more than the
hypergeometrics saved wherever they were cheap — 6.0x at the dwell vector but
**0.72x, a loss**, at CLVTools' fit. A dict keyed on `(a, b, z)` pays per lookup
instead of per sort, so it wins at both: 1.6x where there is little to save and
9.9x where there is much. Item 14's table has the losing version; this one
replaces it.

**Scope is the whole design, and it is load-bearing in both directions.**
Narrower than one evaluation and the memo catches nothing, since within a single
customer's call every :math:`z` differs. Wider and it is worse than useless: the
parameters move every evaluation, so a key from the last one can never hit, and
a fit's ~1,900 evaluations would grow an unbounded dictionary of pure misses.
`tests/test_performance.py::TestDyncovDeduplicatesItsHypergeometrics` gates
both — that SciPy is asked for no argument twice, and that the memo does not
outlive its evaluation.

**On the fit itself, 2:53.** `-m dyncov_fit` passed in **173.87 s** against the
**7:31** item 28 recorded and the **10:07** item 9 did — 2.6x and 3.5x. That
figure is corroboration and not the measurement, for the reason this document
gives twice already: a fit's wall clock is the optimiser's path on a very flat
likelihood, and item 28 saw 7:31 against 10:07 for a change that made every
evaluation 26% *slower*. The per-evaluation numbers in the table above are the
ones to compare. What the fit run does establish is that the optimiser still
arrives: the assertion it passed is that this implementation reaches at least
CLVTools' optimum.

**What it cost in the last digits: nothing at all.** Not "agrees to 1e-14" —
**bit-identical**. All **30 intermediate columns over all 600 customers at both
oracle grid vectors** compare equal under `np.array_equal`, because a memo
returns the same function's value for the same arguments; there is no
rearrangement to lose a digit to. That is a stronger guarantee than item 9's
rewrite could give (27 of 30) or item 28's (3e-14 relative), and it is the
reason this was worth preferring over the restructure item 30 specified: the
restructure would have re-associated the arithmetic and had to argue about the
last two bits.

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
* **The scalar series fallback stays cold** — 0 of 252,000 elements today. If a
  change moves the optimiser into a region where it fires per customer, fits
  get much slower and nothing fails.
* **Likelihood evaluations per fit stay in a coarse band** — 210 for
  `fit_pnbd` on the apparel data. This catches an optimiser or start-value
  regression, which is the thing that actually costs minutes.
* **Cost per customer stays flat** — asserted by comparing operation counts at
  two input sizes, so it is O(*n*) and not O(*n*²) by construction rather than
  by stopwatch.
* **The dyncov likelihood stays batched over covariate intervals** — four
  hypergeometric-arm dispatches per customer, not one per interval, while still
  evaluating one hypergeometric per interval. Unrolling `_f2_middle` back into a
  loop is a 3.4× regression that leaves every number right, so nothing else
  would fail. The same gate asserts that both arms are non-empty at CLVTools'
  fitted parameters, which is the only point either of them is.

Wall-clock still belongs in `tools/benchmark.py`, and *where* the time goes in
`tools/profile.py` — both reported, neither asserted.

## Next

- ~~`docs/backlog.md` item 7~~ — done: `tests/test_performance.py`, marker
  `performance`. The four invariants above, 1.0 s on every run, and each one
  demonstrated to fail against a deliberately broken implementation.
- ~~`docs/backlog.md` item 8~~ — done: `tools/profile.py`, 6 s, which emits
  every profile table above as markdown. Running it is what turned up the
  `_hyp_beta_gt_alpha` correction, which is the argument for it in one line.
- ~~`docs/backlog.md` item 9~~ — done: the dyncov vectorisation spike, which
  paid, though not where it was expected to. 3.3-5.1x per evaluation, 1.33x on
  the fit, 27 of 30 oracle intermediates bit-identical. Written up above, gated
  by `TestDyncovStaysVectorised`.
- ~~`docs/backlog.md` item 14~~ — done: the `hyp2f1` spike, written up above.
  Two thirds of a fit is spent where 85% of self-time is inside SciPy, and
  neither exact rewrite of the hypergeometric helps — the fast one cancels to
  35 digits of error, the accurate one is what SciPy already does. What does
  help is that 93.3% of the hypergeometrics are duplicate arguments.
- **This bullet used to say the opposite, and it was wrong.** It read:
  "Vectorising `log_likelihood_customer` across customers as well — the obvious
  next refactor — would not touch it, and the numbers above are the reason not
  to start it." The duplicate arguments were the thing it could not see.
- ~~`docs/backlog.md` item 30~~ — done, and *not* by the restructure that item
  specified. A per-evaluation memo reaches the same duplicates: the dwell vector
  falls 0.480 s to 0.119 s, all 30 intermediates stay **bit-identical** at both
  grid vectors, and `log_likelihood_customer` keeps its shape. Written up above.
