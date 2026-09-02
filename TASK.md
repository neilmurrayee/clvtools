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
