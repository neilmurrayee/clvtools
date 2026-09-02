# TASK — verify the spec-derived audit of this port

You are picking up finished work from another session. Nothing here needs
re-deriving; it needs checking. Read `docs/spec.md` and `docs/spec-audit.md`
first — they stand alone and contain everything you need.

## What was done, and why

`docs/audit.md` (an earlier pass) indexed the paper and the R `NAMESPACE`
*against this implementation*: for each thing the sources contain, does a
counterpart exist here. That direction finds missing features. It cannot find a
claim that is present but wrongly pinned, because the implementation supplies
the index.

This work inverts the index. `docs/spec.md` states what the sources demand —
222 items, derived from the paper, the 91 R man pages, and **CLVTools' own
testthat suite**, written without reading `src/` or `tests/`. `docs/spec-audit.md`
is the join against the suite we actually have: 64 covered, 76 weak, 75 absent,
6 out of scope.

The testthat suite is the new source and did most of the work: 81 files, 12,647
lines, 645 named claims. It is **not installed** with the package, so nothing in
this repository had ever consulted it. It is the only source that says what must
happen on bad input.

## The files

- `docs/spec.md` — 924 lines, 222 items. Each item: the claim, its source, its
  oracle, and the tolerance it deserves.
- `docs/spec-audit.md` — 608 lines. Findings in four classes (A: unrecorded
  divergences, B: tests that don't test what they claim, C: absent published
  oracles, D: no-R invariants), then section counts, a suggested work order,
  and three appendices.
  - **Appendix 1** — the two commands to refetch the R suite, plus the version
    fingerprint (`Version: 0.12.1`, `Date: 2025-11-06`) so you can confirm you
    have the same source the spec was derived from. The tarball is not
    committed: CRAN's terms cover CRAN's redistribution, not ours. This follows
    the pattern `.gitignore` already uses for the paper.
  - **Appendix 2** — provenance. Which findings were verified by running code,
    which by read-only probe, and which are reported from reading only. Weight
    them differently.
  - **Appendix 3** — all 229 per-item verdict rows with `file:line` evidence.

## Check these first — they are the load-bearing claims

1. **`bootstrap.py:213`** — `ClvDataDynCov` subclasses `ClvData`, not
   `ClvDataStaticCov`, so the resample branch never fires and a dyncov
   bootstrap silently refits without covariates. Reproduce it. This is the only
   *wrong answer* in the audit; everything else is a missing assertion.
2. **The literature tier** — fit cdnow at `estimation.split="1997-09-30"` and
   confirm the port reproduces Fader/Hardie/Lee 2005 (Pareto/NBD and BG/NBD),
   Fader/Hardie 2013 (Gamma-Gamma), and CLVTools' own standard errors. If it
   does, that is five published oracles from three papers for roughly 60 lines.
   None are pinned anywhere today; a literal grep hits only `docs/spec.md`.
3. **The four mis-named tests in section B** — each is a two-minute read.
   `test_covariates_separate_the_scenarios` never calls `predict`;
   `test_real_and_auxiliary_lifetime_walks_do_not_overlap` asserts something
   else; the `name_*` rename is only ever exercised as identity; two dyncov
   oracles are degenerate (`d_omega ≡ 1` for all 600 customers).

## Caveats to carry forward

- The section counts in the verdict table are tallied from the section audits,
  not independently recounted. If you recompute from Appendix 3 and find small
  discrepancies, **trust the appendix over the summary table.**
- `D-17` and `NC-13` were never reached. They are marked `—` rather than
  silently given a verdict.
- `weak` is the least certain verdict class. It is a judgement about whether an
  assertion pins a claim, and reasonable readers will disagree on some.
- Two spec items were corrected mid-audit and the corrections are recorded in
  place in `docs/spec.md` rather than silently patched: **DY-02** (the R test's
  own *title* contradicts its body — it says `Ai = 0`, the body asserts
  `Ai == 1`) and **DY-04** (two tables in the R file share column names and
  disagree at `i = 1`). The R suite is a strong oracle but not an infallible
  one; check bodies, not titles.

## Suggested order of work, if you are fixing rather than checking

1. **A2** — the dyncov bootstrap. Raise on `ClvDataDynCov` in `bootstrap_apply`
   until resampling exists.
2. **A4's first item** — reject `NA` in `Id`/`Date` instead of silently
   dropping the row; empty-frame and non-DataFrame input in the same change.
3. **C** — the literature tier.
4. **D1, D2, D3** — the no-R invariants: DY-07 (static-as-dynamic
   cross-check), permuted covariate rows/columns, DY-10's `α = β` arm.
5. **B1, B2, B4, B6** — fix the four tests that do not test what they claim.
6. **A1, A3, A5, A7** — decide each divergence and record it, or close it.
   Several are already pinned in the divergent direction; they need the
   Findings entry, not new code.

Items marked `out-of-scope` in `docs/spec.md` need a recorded decision rather
than a test — six of the divergences above sat in exactly that ambiguity, where
the audit could not tell a gap from a choice.

## Housekeeping

Both files were swept into commit `61cd5ba` alongside unrelated README and
`pyproject.toml` changes from a parallel session. Worth splitting them into
their own commit so the audit has a clean history to point at.


---

## Outcome, 2026-09-02

Worked in the suggested order. Every claim was reproduced before it was acted
on, which the file asks for and which changed the answer twice.

| | Claim | Verdict |
|---|---|---|
| A2 | dyncov bootstrap drops covariates | **confirmed** — `apply` received a plain `ClvData`; now raises |
| A4 | five silent acceptances | **confirmed** — NA ids and dates, empty frame, non-frame; all now named |
| C | the literature tier reproduces | **confirmed exactly** — five oracles, three papers, `tests/test_literature.py` |
| B1 | never calls `predict` | **confirmed** — it read the fixture; now predicts |
| B2 | asserts something else | **confirmed**, and the honest replacement found customer 129 |
| B4 | rename only exercised as identity | **confirmed** — now renames all three columns |
| B6 | one scalar at `abs=1e-4` | **confirmed, and it was hiding a defect**: `λ=0` gave standard errors 24.5× too large |
| D1 | DY-07 absent | **written** — and the first draft ran on an empty table |
| D2 | permuted covariates unchecked | **written** — both hold |
| D3 | `α = β` arm never taken | **written** — the arms agree to 1e-12 |
| A1 | zero-length windows refused | **confirmed** — R answers both; so do we now |
| A3 | discount factor range | **confirmed** — and its test asserted our divergence |
| A5 | time-unit spellings | **confirmed** — R's `match.arg` forms now resolve |
| A6 | timezone half-broken | **confirmed** — refused, with the route out named |
| A7 / S-13 | remaining bin emitted empty | **not a divergence** — R does the same; pinned as agreement |
| A7 / C-05 | covariate names not coerced | **divergence, ours kept** — R mangles `my var!` to `my.var.` |

Two things the audit did not mention turned up while checking it: **CLVTools has
no month unit at all** (it rejects `"month"` and `"months"`; this package
implements calendar months, which S5 describes), and the discount-factor test
was asserting this package's divergence rather than the claim, so the suite was
defending the defect.

Still open from the audit, and deliberately not started here: the `weak` verdicts
beyond those listed, B5 (two degenerate oracles), B7 (restricted samples
presented as general), and DY-22's seven weekday splits. `docs/backlog.md` items
27 and 28 are also open, and item 28's cheap route was tried and reverted — the
finding is recorded there.

---

## Outcome, round 2 — findings D5 and D6

The audit's suggested order stopped at D3. The rest of section D, and section
C's leftovers, were worked next; `tests/test_invariants.py` is what came of
D5 and D6.

| | Claim | Verdict |
|---|---|---|
| X-01 | all-zero covariate data fits the plain estimates | **holds** — to 3e-5, well inside R's 0.001; the two coefficients are then unidentified and are not compared |
| X-04 | γ = 0 predicts the plain table, three ways | **holds exactly** — `exp(0) = 1`, so `check_exact=True` rather than a tolerance |
| X-05 | γ = 0 gives the plain PMF and tracking plots | **holds** — the PMF exactly; the tracking series to 1e-13, because `600 × E[X(t)]` and a sum of 600 copies of it part company in the last two bits |
| PR-08 | `predict()`'s spending column is the Gamma-Gamma's own | **holds bit for bit** |
| FI-12 | the spending cbs `x` equals the Pareto/NBD's | **holds**, with and without a holdout — two different methods on `ClvData`, separately oracle-pinned, agreement stated nowhere until now |
| B-02 / B-11 | drawing every customer once returns the original | **holds bit for bit** — cbs, spending summary, periods, both design matrices, and the estimate |

The nesting tests discriminate: perturbing `alpha_i` by 0.01 fails five of
them, and by 1e-9 fails none, which is why the three that can be exact are.

Still open from the audit: B3 (a self-referential doctest), B5 (two degenerate
dyncov oracles), B7 (restricted samples presented as general), and D4 —
DY-22's seven weekday splits.
