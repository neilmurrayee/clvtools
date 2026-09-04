# Audit of the suite against `docs/spec.md`

Date: 2026-09-02, at commit `61cd5ba`. `docs/spec.md` states what the sources
demand, derived without reading `src/` or `tests/`. This file is the join:
for each spec item, what the suite actually pins.

**Status: every finding worked**, 2026-09-02, **and every `weak` verdict
re-checked**, 2026-09-03 — round 5 of `docs/backlog.md`, items 34 and 35. Of the
76 `weak` rows, 14 had been closed by items 21-33 without the verdict being
updated, 20 hid real defects, and the rest were either genuinely untested claims
that hold or divergences that needed recording rather than fixing. **None
remains.** A1–A7, B1–B7, C, D1–D6, the six
`out-of-scope` items that needed a recorded decision, and D-17 and NC-13, the
two the audit never reached. Appendix 4 records what each one turned out to be —
fourteen confirmed on the first pass, one overturned, two kept as deliberate
divergences, and four defects that no test here could see: standard errors 24.5×
too large at a zero penalty, a dyncov bootstrap refitting without covariates, a
`NaN` horizon becoming a `NaN` prediction, and a mistyped covariate name being
dropped. The suite went from 906 tests to 1,146, still at 100% line coverage of
`src/`. What remains of this document is the `weak` verdicts it did not
individually list, which its own caveat below calls its least certain class.

**How this differs from `docs/audit.md`.** That audit indexed the paper and the
R `NAMESPACE` *against this implementation* — for each thing the sources have,
does a counterpart exist here. It found missing features, and closed all
fourteen. This one runs the other way, so it can also find a claim that is
present but wrongly pinned. Every finding below is of that second kind or of a
kind the first direction structurally could not reach.

**New source.** CLVTools' own testthat suite, from the CRAN source tarball
(`CLVTools_0.12.1.tar.gz`, 2025-11-06) — 81 files, 12,647 lines, 645 named
claims. It is not installed with the package, so nothing in this repository had
ever consulted it. It supplies most of what follows, and it is the only source
that says what must happen on bad input.

## Verdicts

Of 222 spec items, 220 were verdicted (D-17 and NC-13 were not reached):

| | As audited | After rounds 5 and 6 |
|---|---|---|
| `covered` — the claim is genuinely pinned | 64 | **184** |
| `weak` — touched but not pinned | 76 | **0** |
| `absent` — nothing covers it | 75 | **0** |
| `out-of-scope` or a recorded divergence | 6 | **16** |

Round 5 (`docs/backlog.md`, item 34) worked the 76 `weak` rows and round 6
(item 36) the 69 `absent` ones that remained after it. Between them, **23 rows
were stale** — the behaviour was already pinned, by work done after the audit
was written — and **19 were real defects**, of which the largest is `F-12`: on
hourly data the Pareto/NBD stopped 223 log-units short of the optimum and
reported `converged = True`. The per-row notes below say which was which; the
narrative is in `docs/backlog.md`.

Read those numbers carefully. `absent` is not a defect list: much of it is
behaviour this port has and simply never asserts, and a good deal of it passes
today. The four sections below are the part that matters.

---

# A. Divergences from R that nothing records

The house rule is "deviations get a test, not a comment — add to both"
(`CLAUDE.md`), and the README's Findings section does this well for the
deviations it knows about. These are the ones it does not. Each was confirmed
against the running code. Several are *pinned in the opposite direction* — a
test asserts the divergent behaviour — which means they cannot drift, but also
that nobody chose them on the record.

**A1. Zero-length prediction windows are refused where R returns zero.**
Three places, one root: `predict(prediction_end=0)` raises rather than
returning `CET = 0` (`test_predict.py:349` pins the refusal), the dyncov path
raises `"does not reach past"` (`test_pnbd_dyncov_predict.py:385`), and
`newcustomer(0)` raises `ValueError("strictly positive")` where R returns
**1** (`predict.py:144,157,175`; refusal pinned at `test_predict.py:564` and
`test_pnbd_dyncov_predict.py:415`). Spec PR-05, DY-12, T-20, NC-02. In none of
README Findings, `docs/audit.md`, `docs/backlog.md`.

**A2. The dyncov bootstrap silently refits a model with no covariates.**
`ClvDataDynCov` subclasses `ClvData`, not `ClvDataStaticCov`, so the resample
branch at `bootstrap.py:213` never fires; the caller's `apply` receives a plain
`ClvData` with the dynamic covariates gone. A correctly written refit closure
still cannot produce a dyncov fit. The result is confidence intervals that are
plausible, wrong, and unflagged. Spec B-09. **This is the most serious item in
the audit** — everything else is a missing assertion; this is a wrong answer.

**A3. The discount factor has no upper bound and rejects zero.** R admits
`[0, 1)`. This port rejects `0` (`aggregate.py:401`) and silently returns a
number for `1.5` and `100` — I ran it. The parameter carries CLVTools' exact
semantics (`DEFAULT_DISCOUNT_FACTOR = log(1.1)`), so R's range does transfer.
`predict()` itself performs no validation at all. Spec PR-11.

**A4. Five silent acceptances where R errors.** Each probe-confirmed:
`NA` in `Id` or `Date` **silently drops the transaction** (the groupby in
`_aggregate_to_day`, `data.py:262`, drops NaN keys — a 3-row frame returns 2
rows, no warning, customer summary quietly wrong); a single-category covariate
yields a `(600, 0)` design, i.e. a covariate model with no covariates
(`data.py:746`); covariate `Id`s absent from the transaction data are dropped
by `.loc` rather than rejected (`data.py:731`); re-setting covariates on an
object that already has them overwrites without complaint (`data.py:669`); an
empty DataFrame is accepted and a non-DataFrame gives `AttributeError` rather
than a named error. Spec V-06, C-09, C-10, C-13. The first is the one to fix
first: it is exactly the silent-wrong class commit `b75c15e` was written
against, and the one input path it missed.

**A5. Time-unit spellings R accepts are rejected.** `"w"`, `"weeks"`,
`"Weeks"` all fail; only five exact lowercase singulars are admitted
(`timeunit.py:TIME_UNITS`, pinned at `test_timeunit.py:292`). R accepts cases,
plurals and full names. Spec T-07.

**A6. Timezone-aware input is half-broken.** A date or string split raises
`TypeError: Cannot compare tz-naive and tz-aware timestamps`; a *numeric* split
builds a fully usable object with a tz-aware `estimation_end`, and spans are
computed from `total_seconds()`, so a DST transition inside the window shifts
`t_x`/`T` by up to an hour against the tz-naive answer. Usable and silently
inconsistent. Spec T-04.

**A7. Smaller, each unrecorded.** `prediction_end=14.4` and `14` give different
windows where R makes them agree (T-22). Tracking-plot periods between the last
transaction and `data_end` report `0.0` where R gives `NA` (S-12) — a zero and
a missing value are not the same thing on a plot. The `"10+"` frequency row is
emitted with zero customers when the bins already cover everyone (S-13).
Covariate column names are never coerced to valid identifiers; `my var!`
survives verbatim (C-05). Correlation is refused with static covariates
(`estimate.py:193`) — README:39 says "Pareto/NBD only", which is about families,
not covariates (X-09). `fit_pnbd_dyncov` takes no constraint, regularization or
correlation argument, removing 12 of I-05's 29 Hessian configurations. The
formula parser accepts `Gender*Channel`, `.-Gender` and bare `log(x+2)`, then
fails on each with a misleading "not in the data" (FI-06, FI-07). There is no
`date.format` argument at all, and `time_unit` has a default where R has none.

---

# B. Tests that do not test what they claim

The class you cannot find by reading coverage, because every one of these lines
executes.

**B1. `test_covariates_separate_the_scenarios`** (`test_predict.py:537`).
Docstring: *"S6.3.4's 'region A versus region B' comparison."* Body: reads four
numbers out of `newcustomer_static.json` and asserts `len(set(...)) == 4`.
**It never calls `predict`.** It asserts that R's fixture file contains four
distinct numbers. The port's own ability to separate scenarios follows only
indirectly, from four parametrized oracle comparisons, and only for
static-covariate Pareto/NBD.

**B2. `test_real_and_auxiliary_lifetime_walks_do_not_overlap`**
(`test_pnbd_dyncov.py:373`). Named and documented for the walk non-overlap
invariant. Body asserts `(cbs.loc[has_real, "x"] > 0).all()` — that customers
with a real lifetime walk have repeat purchases. A different claim entirely,
and the invariant in the name is pinned nowhere.

**B3. `fitted_data`'s doctest** (`diagnostics.py:338`) is reachable from no
test file, and its printed values come from this implementation rather than
from R's `fitted()`. Self-referential: it cannot fail if the function is wrong,
only if it changes.

**B4. The `name_id` / `name_date` / `name_price` rename is only ever exercised
as identity.** The three uses in the suite are the default value passed
explicitly, a misspelling checked for raising, and `None`. The rename mapping
in `data.py:__init__` never actually renames anything, so a broken non-identity
mapping would pass all 906 tests. Spec D-14.

**B5. Two degenerate oracles.** `d_omega ≡ 1` for all 600 customers and
`d1 ≡ 1` throughout the ABCD table, so comparing them against the oracle cannot
discriminate, and the non-boundary branch of `_distance_to_interval_end`
(`dyncov_walks.py:275`) is never reached through `d_omega`. Spec DY-01, DY-15.

**B6. `test_zero_weight_reproduces_the_unpenalised_fit`**
(`test_pnbd_advanced.py:454`) is three lines and compares one scalar at
`abs=1e-4` — about `1.7e-8` relative on a log-likelihood near −5821. R's claim
is that λ=0 reproduces the coefficient vector *and* the summary coefficient
table, `expect_equal`, no loosened tolerance. Spec X-06.

**B7. Restricted samples presented as general.** `test_zero_coefficients_make_
every_multiplier_one` checks the two auxiliary walks over `customers[:20]`;
real walks are unchecked and the `Bbar_i = Dbar_i = 0` half of the claim is not
checked at all (DY-03). `i` and the window start are compared only over
`settings["sample.ids"]` (DY-06). Bootstrap resampling is verified by four
hand-picked ids and `x`/`t_x`/`T` only (B-02), and covariate resampling by a
`(600, 2)` shape assertion (B-08).

---

# C. Published oracles that are absent — and that already pass

The highest value-per-line finding in the audit. **No literature value is
pinned anywhere**; a literal grep for each hits only `docs/spec.md`. The R
suite asserts them at `estimation.split = "1997-09-30"`, a different split from
the 37-week default this port fits, which is why `paper_values.py` does not
already cover them.

`audit-models` fitted cdnow at that split and the port reproduces every one:

| Source | Published | This port |
|---|---|---|
| Fader/Hardie/Lee 2005, Pareto/NBD | `r=0.553, α=10.578, s=0.606, β=11.669`, LL `−9595.0` | `0.5533, 10.5777, 0.6062, 11.6687`, LL `−9594.9762` |
| Fader/Hardie/Lee 2005, BG/NBD | `r=0.243, α=4.414, a=0.793, b=2.426`, LL `−9582.4` | `0.2425948, 4.4136079, 0.7929248, 2.4259154`, LL `−9582.4292` |
| Fader/Hardie 2013, Gamma-Gamma | `p=6.25, q=3.74, γ=15.44`, LL `−4055.9177` | `6.2496, 3.7442, 15.4435`, LL `−4055.9177` |
| CLVTools' own Pareto/NBD SEs | `0.0476264, 0.8427222, 0.1872594, 6.2105448` | `0.0476278, 0.8427222, 0.187277, 6.2109628` — all inside R's 0.001 |

Three published papers, five oracles, roughly sixty lines of a slow module.
BG/NBD and GGompertz/NBD are **never fitted on cdnow at all** (F-04, F-06).

Two more that pass today and are pinned nowhere:

- **The Pareto/NBD `PAlive` numerical-stability case** (M-04). `x=(221,254,161,204)`,
  `T.cal=(103.57,97.29,98.00,99.43)` at `r=0.5143, α=2.8845, s=0.2856, β=14.1087`
  — inputs that produced `NaN` in an earlier implementation. Verified finite:
  `[0.99960, 0.99956, 0.74949, 0.99426]`. A named regression case, exact
  inputs, no R needed.
- **`P(X=0)` strictly decreasing in `T.cal`** (PMF-04). Verified to hold, for
  no family asserted.
- **`dyncov_palive.csv` is already committed and orphaned**, slated for
  deletion at `docs/backlog.md:1101`. Wiring it up is DY-13 for free rather
  than a fixture to throw away.

---

# D. Invariants checkable with no R at all

These need no fixture and no oracle. They are the cheapest tests in the audit
and they cover the mechanisms most likely to break silently.

1. **DY-07, the static-as-dynamic cross-check.** Feed constant covariates as a
   dynamic series; require `A_i`/`C_i` to equal the static values, `Dbar_i = 0`,
   `Bbar_i = −T·A_i` in the CET table. The only whole-machinery validation of
   the walk construction, and absent entirely.
2. **X-07 / X-08, permuted covariate data.** Nothing in the suite shuffles
   covariate rows or reverses covariate columns; every oracle frame arrives in
   the order the implementation sorts into. A design matrix mis-joined to the
   customer summary, or `names_cov_*` drifting from column position, is
   invisible — and `get_dummies` moving dummies to the end (C-03/C-04) is
   exactly that mechanism.
3. **DY-10's `α = β` arm.** `dyncov.py:355` and `:374` branch on
   `alpha_1 >= beta_1`; only one side is ever taken. R runs both, at
   `α = β = 1.234`.
4. **DY-22, seven weekday splits.** Every dyncov test in both files uses
   `estimation_split=104`, so all 600-customer oracle comparisons run at one
   alignment against the weekly covariate grid.
5. **PR-08, X-01..X-05, FI-12** — the nesting and cross-model invariants:
   γ=0 recovering the plain model in `predict`/`plot`/`pmf` (not only in the
   likelihood), and a spending model's cbs `x` equalling the Pareto/NBD's.
   Both sides of FI-12 are separately oracle-pinned; the agreement between them
   is never stated, and they come from genuinely different code paths
   (`data.py:355` vs `:395`).
6. **B-02 / B-11, the bootstrap identity.** Draw every id once; expect the
   original cbs, covariates and coefficients back. The strongest available test
   of the whole resampling path, and the one that would have caught A2.

---

# Section verdicts

These counts are tallied from the section audits, not independently recounted.
Where they disagree with Appendix 3, **trust the appendix**: it carries the
per-item evidence and these do not.

They were also a **snapshot of 2026-09-02**, and round 5
(`docs/backlog.md` item 34) has since re-verdicted every `weak` row against the
suite as it stands. Recounted from Appendix 3 on 2026-09-03, that round having
closed:

| | Count |
|---|---|
| `covered` | **119** |
| `absent` | 69 |
| `out-of-scope` | 12 |
| `weak` | **0** |

`I-05` was the last, and a capability gap rather than a testing one:
`fit_pnbd_dyncov` took no constraint, regularization or correlation argument, so
12 of the 29 configurations it names could not be constructed. Item 35 added the
first two; the third is a recorded divergence.

The section table below is the original tally and is left as the snapshot it
is; Appendix 3 is the record, as the paragraph above says.

| Spec section | covered | weak | absent | out-of-scope |
|---|---|---|---|---|
| S1 Transaction data (D) | 7 | 2 | 8 | 0 |
| S2 Time and splits (T) | 6 | 8 | 8 | 0 |
| S3 Descriptives (S) | 9 | 4 | 2 | 1 |
| S4 Covariate data (C) | 1 | 6 | 7 | 0 |
| S5 Model expressions (M) | 10 | 1 | 1 | 1 |
| S6 PMF (PMF) | 1 | 3 | 2 | 0 |
| S7 Estimation (F) | 3 | 2 | 10 | 0 |
| S8 Covariate fits (X) | 2 | 6 | 7 | 0 |
| S9 Dyncov (DY) | 4 | 8 | 13 | 0 |
| S10 Prediction (PR) | 7 | 5 | 2 | 2 |
| S11 newcustomer (NC) | 3 | 4 | 5 | 0 |
| S12 Bootstrapping (B) | 5 | 6 | 3 | 0 |
| S13 Inference (I) | 2 | 7 | 2 | 0 |
| S14 Formula interface (FI) | 4 | 8 | 3 | 0 |
| S15 Input validation (V) | 0 | 6 | 2 | 0 |

Where the suite is strongest: `DY-08` compares all 30 intermediate columns
across 2 parameter vectors and 600 customers at `rtol 1e-8`; `M-01..M-08` pin
every family's expressions against per-customer oracles; the constrained-Hessian
alignment that README Findings records is genuinely pinned. Fixture provenance
is clean — all 124 come from R generators that self-check against a public
generic before writing, so there is no self-referential oracle anywhere.

Where it is thinnest: **S15 has no `covered` item at all.** Roughly 25–30% of
the applicable validation claims are pinned, discounting ~20 that cannot cross
the language boundary. The split is sharp — the *numeric* input surface is
solid and "Make bad input loud" clearly landed there, but the *argument-shape*
surface is unguarded: optimiser overrides accept anything (`_optimize.py:82`),
no single-logical argument is validated anywhere, and a `NaN` start parameter
passes `np.any(start <= 0)` and then misreports as a data fault.

---

# Suggested order of work

1. **A2** — the dyncov bootstrap. The only wrong answer in the audit. Raise on
   `ClvDataDynCov` in `bootstrap_apply` until resampling exists.
2. **A4's first item** — reject `NA` in `Id`/`Date` instead of dropping the
   row; empty-frame and non-DataFrame input in the same change.
3. **C** — the literature tier. Five published oracles from three papers that
   the port already reproduces, for ~60 lines.
4. **D1, D2, D3** — the no-R invariants: DY-07, permuted covariates, DY-10's
   `α = β` arm.
5. **B1, B2, B4, B6** — fix the four tests that do not test what they claim.
6. **A1, A3, A5, A7** — decide each divergence and record it, or close it.
   Several are pinned in the divergent direction already; they need the
   Findings entry, not new code.

Items marked `out-of-scope` in `docs/spec.md` need a recorded decision rather
than a test — the audit could not otherwise tell a gap from a choice, and six
of the divergences above sat in that ambiguity.

---

# Appendix 1 — Reproducing the sources

The R test suite is not installed with the package and is not committed here
(CRAN's terms cover CRAN's redistribution, not ours). To re-derive or check any
`Rtest:` citation in `docs/spec.md`:

```bash
curl -sSLO https://cran.r-project.org/src/contrib/CLVTools_0.12.1.tar.gz
tar xzf CLVTools_0.12.1.tar.gz          # -> CLVTools/tests/testthat/ (81 files)
```

Verify you have the same version the spec was derived from: `CLVTools/DESCRIPTION`
must read `Version: 0.12.1`, `Date: 2025-11-06`. The man pages cited as `Rdoc:`
are in `CLVTools/man/` (91 files) — the installed `.Rlib/CLVTools/help/` holds
them only in binary `.rdb` form.

# Appendix 2 — Provenance of the findings

Not every verdict below carries the same weight, and a checker should know which
is which.

**Verified by running the code, in this session:** the dyncov bootstrap
returning a plain `ClvData` (A2); the discount factor accepting `1.5` and `100`
and rejecting `0` (A3); `test_covariates_separate_the_scenarios` never calling
`predict` (B1); the `name_*` rename never being exercised (B4);
`test_zero_weight_reproduces_the_unpenalised_fit` asserting one scalar (B6);
the absence of every literature value by literal grep (C); all 124 fixtures
coming from R generators via `paste0()` naming.

**Verified by section audits running read-only probes against the installed
package:** the port reproducing the literature values at
`estimation.split="1997-09-30"` (C); `PAlive` finite on the M-04 inputs;
`P(X=0)` monotone in `T.cal`; the five silent acceptances in A4; the timezone
behaviour in A6; the formula-parser acceptances in A7.

**Reported from reading, not executed:** the remaining per-item verdicts in
Appendix 3. Treat `weak` verdicts as the least certain class — they are a
judgement about whether an assertion pins a claim, and reasonable readers will
disagree on some.

**Two spec items were corrected during the audit** and the corrections are
recorded in place in `docs/spec.md` rather than silently patched: DY-02 (the R
test's *title* contradicts its body) and DY-04 (two tables in the R file share
column names and disagree at `i = 1`).

# Appendix 3 — Per-item verdicts

`c` covered · `w` weak · `a` absent · `o` out-of-scope · `!` divergence from R
that nothing in README Findings / `docs/audit.md` / `docs/backlog.md` records.

## S1 Transaction data

| | | evidence / note |
|---|---|---|
| D-01 | c | `test_data.py:174` |
| D-02 | c | `test_data.py:185`, exact `[10.0, 25.0]` |
| D-03 | c | re-verdicted 2026-09-04: **round 6**: offsetting the whole log by 37 minutes gives the identical customer summary, which is the claim the hour floor exists to make. No committed date is anything but midnight, so nothing had run it end to end |
| D-04 | c | re-verdicted 2026-09-04: **round 6**: input row order asserted not to matter, which the audit verified and left unasserted |
| D-05 | c | re-verdicted 2026-09-04: **round 6**: a duplicate on the *first* day, which is the one that sets `t_x`'s origin -- an aggregation slip there moves every subsequent recency rather than one row. Prices sum, as CLVTools' same-day aggregation does |
| D-06 | c | `test_data.py:152`, `test_descriptives.py:327` |
| D-07 | c | `test_data.py:124`, rtol 1e-12 |
| D-08 | c ! | re-verdicted 2026-09-04: **round 6**: a *defect*, not a non-port — a float `Id` column spelled customer 1 as `"1.0"` where R gives `"1"`, and pandas types a numeric id column as float the moment it holds one `NaN`. Fixed; a genuinely fractional id keeps its point |
| D-09 | c | re-verdicted 2026-09-04: **round 6**: an integer `Price` gives the same spending summary as a float one |
| D-10 | c | re-verdicted 2026-09-04: **round 6**: covered by the same shuffle test as `D-04` |
| D-11 | c | `test_data.py:338`, `test_descriptives.py:178` — flag *and* descriptives consequence |
| D-12 | c | `test_descriptives.py:67`, rel 1e-12 |
| D-13 | c | re-verdicted 2026-09-04: **round 6**: the caller's frame is asserted unmodified |
| D-14 | c | re-verdicted 2026-09-03: closed by **B4** — `TestTheColumnRenameActuallyRenames` renames all three columns |
| D-15 | c | re-verdicted 2026-09-04: **round 6**: `Timestamp`, `str` and `datetime.date` all accepted and agreeing |
| D-16 | c | `test_cdnow.py:73,95` — incidental but genuine |
| D-17 | — | not reached by the audit |
| D-18 | c | re-verdicted 2026-09-03: **round 5**: the sort-order half asserted — shuffling the input rows leaves the summary, the cbs and the transactions identical, which is where an assumed sort would show as a negative interpurchase gap |

## S2 Time and splits

| | | evidence / note |
|---|---|---|
| T-01 | c | re-verdicted 2026-09-03: **round 5**: the hour epsilon really is 1 hour, and it is unobservable — `_aggregate_to_day` floors every transaction to the unit first, so nothing can land in the gap. The partition it exists to guarantee is asserted for hour/day/week |
| T-02 | c | `test_data.py:107` |
| T-03 | c | `test_data.py:115`, `test_cdnow.py:206` |
| T-04 | c ! | re-verdicted 2026-09-03: closed by **A6** — `test_timezone_aware_dates_are_refused_rather_than_half_supported` |
| T-05 | c ! | re-verdicted 2026-09-04: **round 6**: a fractional split is honoured and silent where R warns — the same choice as `T-22`'s fractional `prediction_end`, now pinned and in the README |
| T-06 | c | |
| T-07 | c | re-verdicted 2026-09-04: closed by **A5** — `week`, `weeks`, `Weeks` and `w` all accepted |
| T-08 | c | |
| T-09 | c | re-verdicted 2026-09-04: **round 6**: a one-day holdout is accepted, where R imposes a minimum |
| T-10 | c ! | re-verdicted 2026-09-04: **round 6**: covered with `T-05`; the fraction reaches the timestamp rather than being rounded away |
| T-11 | c | re-verdicted 2026-09-03: **round 5**: satisfied by a stricter rule than the spec states — any `data_end` before the last purchase is refused, which subsumes one before the split. Both halves pinned |
| T-12 | c | re-verdicted 2026-09-04: **round 6**: `data_end` at the last transaction gives the same cbs and summary as omitting it |
| T-13 | c | re-verdicted 2026-09-04: **round 6**: int, str, `date` and `Timestamp` all accepted |
| T-14 | c | |
| T-15 | c | re-verdicted 2026-09-03: **round 5**: the spec's own two dates, 1998-07-16 and 1998-07-30, asserted to the day |
| T-16 | c | re-verdicted 2026-09-04: **round 6**: covered with `T-13` — only `datetime64[ns]` had ever been passed |
| T-17 | c | re-verdicted 2026-09-03: **round 5**: idempotence on a boundary for all five units, plus that `week` floors to the *day* on purpose |
| T-18 | c | re-verdicted 2026-09-03: **round 5**: all four combinations build — and a grid stopping short of the estimation end raised a bare `IndexError` from numpy; now a named `ValueError` |
| T-19 | c | re-verdicted 2026-09-03: **round 5**: 1- and 2-period horizons, numeric and as a date, and the two spellings agree |
| T-20 | c | re-verdicted 2026-09-04: **round 6**: stale -- `A1` was fixed and recorded in the README's findings; `predict(prediction_end=0)` now returns `CET = 0` as R does |
| T-21 | c | |
| T-22 | c ! | re-verdicted 2026-09-03: **round 5**: the divergence had a README finding and no test; now pinned three ways |

## S3 Descriptives

| | | evidence / note |
|---|---|---|
| S-01 | c | |
| S-02 | c | |
| S-03 | c | re-verdicted 2026-09-03: **round 5**: the whole object compared, not two cells |
| S-04 | c | |
| S-05 | c ! | re-verdicted 2026-09-03: **round 5**: the values agree — a no-holdout customer's holdout column is empty — and the spelling differs, R printing `-` where pandas prints `NaN`/`NaT`. Recorded rather than reformatted |
| S-06 | c | |
| S-07 | c | re-verdicted 2026-09-04: **round 6**: mutating the caller's frame afterwards -- `Price`, `Date` and whole rows -- leaves the data object unchanged. A defensive copy is the sort of thing an optimisation removes, and the failure is silent and remote |
| S-08 | c | |
| S-09 | o | no coercion generic; `ClvData()` is the only constructor |
| S-10 | c | |
| S-11 | c | |
| S-12 | o | re-verdicted 2026-09-04: **round 6**: a divergence, recorded rather than matched. R reports `NA` between the last transaction and `data.end`; this reports `0.0`. `data_end` is an argument the caller supplies rather than something inferred from the log, so "you told me the window runs here and nobody bought" is what was asked for, and an `NA` would discard a real observation -- and hide the weeks where the model predicted transactions and none happened, which is the part of a tracking plot a reader most wants |
| S-13 | c ! | re-verdicted 2026-09-03: closed by **A7** — `test_descriptives.py:388`, pinned as agreement with R |
| S-14 | c | |
| S-15 | c | |
| S-16 | o | re-verdicted 2026-09-03: **round 5**: `annotate.ids` has no counterpart because `diagnostics` returns frames and leaves rendering to the caller; same disposition as `V-08` |

## S4 Covariate data

| | | evidence / note |
|---|---|---|
| C-01 | c | re-verdicted 2026-09-03: **round 5**: character and pandas `category` give the same dummies, asserted with *and* without a holdout |
| C-02 | c | re-verdicted 2026-09-03: **round 5**: 2 categories -> 1 dummy, 3 -> 2 |
| C-03 | c | re-verdicted 2026-09-03: **round 5**: the mixed arm reached — `get_dummies` does reorder (numerics first), and the names are asserted to describe the matrix column for column, which is the claim the reordering threatens |
| C-04 | c | re-verdicted 2026-09-03: **round 5**: numeric covariates stay numeric with and without categoricals present |
| C-05 | o ! | re-verdicted 2026-09-04: closed as a divergence by **A7** — names are kept verbatim where R mangles them, already in the README's findings |
| C-06 | c | re-verdicted 2026-09-04: **round 6**: the covariate frame is copied; mutating the caller's afterwards leaves the data object unchanged, asserted alongside `S-07` |
| C-07 | c | re-verdicted 2026-09-04: **round 6**: 26 extra weeks of history before the estimation start leave the likelihood **bit-identical**. Asserted there rather than on the walk arrays, whose `NaN` entries make an elementwise comparison report a difference where there is none |
| C-08 | c ! | **round 5**, a recorded divergence: R requires the two series to be equally long, this package allows them to differ and asks the questions equal length would have answered at three points instead — the lifetime grid reaching the estimation end, the transaction grid covering its walks, and the prediction horizon being reachable. Pinned, and in the README's findings |
| C-09 | c ! | re-verdicted 2026-09-04: **round 6**: a *defect*, and larger than the row says — a categorical covariate could not be selected by its own name at all, because the check ran against the encoded frame. Only the apparel cohort's 0/1 numeric covariates, which keep their names, made that survivable. Fixed, with the single-level case earning its own message |
| C-10 | c ! | re-verdicted 2026-09-03: closed by **A4** — an unrecognised covariate name now raises; `test_predict.py:651` |
| C-11 | c ! | re-verdicted 2026-09-04: **round 6**: five claims, and this port had **none**. A repeated `(Id, Cov.Date)`, a missing one, and an `NA` in `Id`, the date or any covariate all built a data object in silence -- the last being the README's static-covariate finding on a path with no guard. Completeness is checked as a rectangle, which catches a *missing* pair that counting rows would not |
| C-12 | c | |
| C-13 | c | re-verdicted 2026-09-04: closed by **A4** — re-setting covariates no longer silently overwrites |
| C-14 | c | re-verdicted 2026-09-04: **round 6**: `name_id` on the covariate frame, compared against the default-named build rather than merely run |

## S5 Model expressions · S6 PMF · S7 Estimation

| | | evidence / note |
|---|---|---|
| M-01..M-03 | c | per-customer oracles |
| M-04 | c | re-verdicted 2026-09-04: **round 6**: stale -- `TestNumericalStabilityCases` has pinned the four heavy buyers since round 4 |
| M-05..M-08 | c | |
| M-09 | c ! | re-verdicted 2026-09-03: **round 5**: the post-erratum `CET` was pinned to 1e-6 with no record of *which* value or why — a deviation with a test and no findings entry. Both halves now exist |
| M-10..M-12 | c | |
| M-13 | o | BG/BB not ported |
| PMF-01 | c | re-verdicted 2026-09-03: **round 5**: partial sums over k = 0..20, strictly increasing and never exceeding 1 |
| PMF-02 | c | re-verdicted 2026-09-03: **round 5**: every value in [0, 1], no NaN, over the whole range rather than one scalar T |
| PMF-03 | c | holds structurally — `pmf(k,T,params)` cannot see `x`/`t_x` |
| PMF-04 | c | re-verdicted 2026-09-04: **round 6**: stale -- asserted for all three families in the same class |
| PMF-05 | c | re-verdicted 2026-09-04: **round 6**: added `diagnostics.pmf_table` -- one row per customer, one `pmf.x.<k>` column per count, `0:5` by default. `pmf_data` answers a different question (bins for the S6.2.2 plot) and cannot say what *this* customer's probability of buying twice is |
| PMF-06 | c | re-verdicted 2026-09-03: **round 5**: a non-integer `k` was silently truncated — `pmf(2.7, ...)` returned `pmf(2, ...)`, a different question answered confidently. Now refused |
| F-01 | c | re-verdicted 2026-09-04: **round 6**: stale -- `test_literature.py` has fitted CDNOW at the date split since round 4. Gains CLVTools' own four-decimal estimates, a tighter oracle than the paper's three |
| F-02..F-06 | a | no literature value pinned; BG/NBD and GGomNBD never fitted on cdnow at all (**C**) |
| F-07 | c ! | re-verdicted 2026-09-04: **round 6**: the comparison cannot be made coordinate by coordinate. Three sources' `(b, beta)` span four orders of magnitude at the same likelihood; the identified quantity is `beta/b`, which is the nested Pareto/NBD's `beta`, and all three agree on it within 11%. `s` tilts along the same ridge. Asserted as ratio, likelihood and spread; recorded in the README |
| F-08 | c | re-verdicted 2026-09-04: **round 6**: stale -- `TestFaderHardie2013` fits the Gamma-Gamma on CDNOW, not apparel |
| F-09 | c | re-verdicted 2026-09-03: **round 5**: the finiteness sweep — estimates, standard errors and every numeric column of `predict()` |
| F-10 | c | |
| F-11 | c | re-verdicted 2026-09-03: **round 5**: Powell run alongside Nelder-Mead and L-BFGS-B, since 'accepted' and 'works' were two claims and only the first was covered |
| F-12 | c ! | re-verdicted 2026-09-04: **round 6**: it did not work. The Pareto/NBD stopped **223 log-units short** at a degenerate `s = 0.0011` and reported `converged = True`; the GGom/NBD raised; the BG/NBD was fine. Fixed by scaling the default start (`_optimize.start_scale`), and the exact time-unit invariance that makes the true optimum knowable is now a test of its own |
| F-13 | c | re-verdicted 2026-09-04: **round 6**: fit on a log read with `name_price=None`, then predicted on the priced one; the customer summaries are compared row by row first, so the claim is that the two agree and not merely that neither raised |
| F-14, F-15 | c | |

## S8 Covariate fits

| | | evidence / note |
|---|---|---|
| X-01 | c | re-verdicted 2026-09-04: **round 6**: already covered by `TestZeroCovariatesRecoverThePlainModel` — the verdict was stale. Round 6 adds the other direction, zero *data* with non-zero coefficients |
| X-02 | c | re-verdicted 2026-09-03: **round 5**: eight seeded random gammas over [-40, 40], all three families, and bit-exact (`np.array_equal`) for the Pareto/NBD — the randomness being the claim |
| X-03 | c | |
| X-04 | c | re-verdicted 2026-09-04: **round 6**: stale, as `X-01`; `predict`'s three columns now also checked with zero data and arbitrary gamma, bit for bit |
| X-05 | c | re-verdicted 2026-09-04: **round 6**: stale — the pmf and tracking frames are covered by the same class, from item 25's D5/D6 work |
| X-06 | c | re-verdicted 2026-09-03: closed by **B6** — the coefficient vector and the summary table, not one scalar |
| X-07 | c | re-verdicted 2026-09-04: closed by **D2** — `test_pnbd_staticcov.py` permutes covariate rows |
| X-08 | c | re-verdicted 2026-09-04: closed by **D2** — and columns |
| X-09 | c ! | re-verdicted 2026-09-03: closed by **A7** — `test_estimate.py:244` pins correlation + covariates raising |
| X-10 | c | |
| X-11 | c | re-verdicted 2026-09-04: **round 6**: works, and now says so. There is no name-mangling step in this port at all, which is exactly why the claim needed an assertion -- its absence would be invisible until someone used such a column |
| X-12 | c | re-verdicted 2026-09-03: **round 5**: `start_m` given a non-default value and shown to reach the search |
| X-13 | c ! | re-verdicted 2026-09-04: **round 6**: the `[-1, 1]` half is a *misreading* — `m` is the Sarmanov mixing parameter, whose admissible interval is `correlation_bounds`, [-1.042, 34.822] at the paper's parameters and moving with them; `correlation_coefficient` is what lies in [-1, 1]. The rest was real: a `NaN` start reached the objective (the fifth such argument), and a start outside the bounds earned the same message where the bounds are computable at the start point. Both now named |
| X-14 | c | re-verdicted 2026-09-03: **round 5**: a `NaN` lambda was accepted and surfaced later as "objective is not finite"; now refused, as is a non-numeric pair |
| X-15 | c | re-verdicted 2026-09-03: **round 5**: three of six landed badly — a bare string was iterated character by character, a duplicate was accepted, and an unknown name was blamed on "both processes". All three now name the actual problem |

## S9 Dyncov

| | | evidence / note |
|---|---|---|
| DY-01 | c | re-verdicted 2026-09-04: closed by **B5** — `TestDOmegaOffTheBoundary`, four synthetic births reaching 7/7, 4/7, 2/7 and 1/7 |
| DY-02 | c | re-verdicted 2026-09-04: **round 6**: zero covariates make `Ai` and `Ci` exactly 1, bit for bit |
| DY-03 | c | re-verdicted 2026-09-03: closed by **B7** — `test_pnbd_dyncov.py:374`, all 600 customers and all four walks |
| DY-04 | o ! | re-verdicted 2026-09-04: **round 6**: does not hold as stated, and should not. `Bbar_i` is 0 at `i = 1` only under *zero* covariates; `Dbar_i` never is. `docs/spec.md` already warns that the two R tables share column names and disagree here — 'check bodies, not titles'. Both columns are pinned against `pnbd_dyncov_ABCD` at rtol 1e-10, which is the stronger check |
| DY-05 | o ! | re-verdicted 2026-09-04: **round 6**: does not hold as stated. `Bbar` integrates from the estimation end and `Dbar` from the customer's birth, so with identical data *and* identical coefficients `Ai == Ci` bit for bit while `Bbar_i` is -140 where `Dbar_i` is +17. Both are oracle-pinned at 1e-10, so the asymmetry is CLVTools' too and a test to the audit's reading would fail against the oracle it means to agree with |
| DY-06 | c | re-verdicted 2026-09-03: closed by **B7** — `test_pnbd_dyncov_predict.py:125`, over all 600 |
| DY-07 | c | re-verdicted 2026-09-04: closed by **D1** — the static-as-dynamic cross-check, `test_pnbd_dyncov_predict.py:482` |
| DY-08 | c | `test_pnbd_dyncov.py:194` — 30 columns × 2 parameter vectors × 600 customers, rtol 1e-8. The strongest item in the audit |
| DY-09 | c | re-verdicted 2026-09-04: **round 6**: trimming the covariate series to the estimation end leaves the likelihood *bit-identical*, so the walks are not reading past their own end |
| DY-10 | c | re-verdicted 2026-09-03: closed by **D3** — `test_pnbd_dyncov.py:935` takes the `alpha = beta` arm |
| DY-11 | c ! | re-verdicted 2026-09-04: **round 6**: PAlive and CET recover the plain model at gamma = 0 to 2.1e-14. `DECT` does **not** equal `DERT` and should not: `DERT` discounts an infinite horizon, `DECT` each period of a finite one, because with covariates there is no infinite horizon to discount over. The test asserts they *disagree* -- a change making `DECT` fall back to the closed form would pass everything else |
| DY-12 | c | re-verdicted 2026-09-04: **round 6**: stale with `T-20`/`PR-05` -- the zero-length window answers here too |
| DY-13 | c | re-verdicted 2026-09-04: **round 6**: its orphaned oracle was deleted by item 25 as bit-identical to the `PAlive` column already checked at rtol 1e-11; the claim is covered by that comparison |
| DY-14 | c | `test_pnbd_dyncov.py:476,495,456`, atol 1e-12 |
| DY-15 | c | re-verdicted 2026-09-03: closed by **B5** — four synthetic births reach `d_omega` of 7/7, 4/7, 2/7, 1/7 |
| DY-16 | c | incidental — exactly one of 600 customers has `t_x == T_cal` with `x > 0` |
| DY-17 | c | re-verdicted 2026-09-03: **round 5**: the 2-period auxiliary walk is constructed (birth +21d, `T` on a week start, no real life walk), with the surrounding 5/4/3/2 lengths so the 2 is not a coincidence |
| DY-18 | c | re-verdicted 2026-09-04: **round 6**: CLVTools issue #134. With the series cut at the estimation end -- the shortest `T-18` accepts -- all 600 customers still have finite aux walks on both processes |
| DY-19 | o | **decided 2026-09-03**, was `w` and undecided: claims 1-2 are pinned, and the epsilon-apart claim is unreachable by construction — S6.1's day aggregation makes two purchases an epsilon apart *one* transaction, so there is no second walk to lose. The aggregation step is asserted instead, being the thing that could regress. Same shape as `T-01` |
| DY-20 | c | re-verdicted 2026-09-03: **round 5**: the round-trip is asserted for all 600, bit for bit, at gamma != 0 |
| DY-21 | c | `test_pnbd_dyncov.py:484,495` |
| DY-22 | c | re-verdicted 2026-09-04: closed by **D4** — `TestEveryWeekdaySplit`, all seven alignments against the closed-form nesting |
| DY-23 | c | re-verdicted 2026-09-04: **round 6**: `_to_days` floors, so an epsilon and a `shift()` build the same half-open interval by construction. Asserted with a control (a step back over the boundary *does* move a day), since a `_to_days` returning a constant would pass the first half |
| DY-24 | c | re-verdicted 2026-09-03: **round 5**: the three uncovered claims pinned — 1- and 2-period horizons (issue #128), a 20-customer sample, and a horizon past the covariates refused by name |
| DY-25 | c | re-verdicted 2026-09-04: **round 6**: covered by the extreme-split runs -- a one-week estimation period leaves 590 of 600 customers at `x = 0` and the likelihood and prediction stay finite |

## S10 Prediction

| | | evidence / note |
|---|---|---|
| PR-01 | c | `test_predict.py:78-126`; CLVTools emits `actual.period.spending` — the Rdoc name in the spec is stale |
| PR-02 | c | re-verdicted 2026-09-03: **round 5**: the `level=0.9` and `num_boots=100` defaults pinned beside the `log(1.1)` one |
| PR-03 | o | no `newdata` parameter — the data object is the first positional argument |
| PR-04 | c | re-verdicted 2026-09-03: **round 5**: one row per customer in the data given, and the sampled customers' `PAlive`, `CET` and `DERT` bit-identical either way |
| PR-05 | c | re-verdicted 2026-09-04: **round 6**: stale, with `T-20` -- the same fix, the same finding |
| PR-06 | c | `test_predict.py:100` — genuinely pinned, but incidentally: apparel id 262 has a transaction on the estimation end, and `>=` would give 21 not 20 |
| PR-07 | c | `test_predict.py:267`, strict `<` for every customer |
| PR-08 | c | `test_predict.py:78`, rtol 1e-12 — though structurally unfalsifiable: one code path, not two |
| PR-09 | o | `predict.spending=TRUE` and `verbose` forwarding not ported |
| PR-10 | c | `test_predict.py:335` |
| PR-11 | c ! | re-verdicted 2026-09-03: closed by **A3** — `TestTheDiscountFactorRange` pins both ends |
| PR-12 | c | `test_predict.py:354` |
| PR-13 | c | re-verdicted 2026-09-03: **round 5**: the scenario half was closed by item 27; the data half surfaced as `KeyError: "['Gender'] not in index"` and now names the covariate and lists what the data carries |
| PR-14 | c ! | re-verdicted 2026-09-04: **round 6**: was a defect, now refused. `r = nan` produced a full table with `PAlive`, `CET` and `DERT` entirely `NaN` and nothing to say why -- the **sixth** argument here diagnosed by whatever consumed it first. The message lists every offending parameter and points at `converged` |
| PR-15 | c | re-verdicted 2026-09-03: **round 5**: a `NaN` horizon came back as "cannot convert float NaN to integer" and a list as pandas' "Cannot convert input"; both now name the argument |
| PR-16 | c | `test_predict.py:684-720` |

## S11 newcustomer

| | | evidence / note |
|---|---|---|
| NC-01 | c | all four constructors exercised |
| NC-02 | c | re-verdicted 2026-09-04: **round 6**: stale -- `predict(newcustomer(0))` returns exactly 1.0 |
| NC-03 | c | re-verdicted 2026-09-04: **round 6**: the prospective-customer path, which shares no code with `predict` over a cohort |
| NC-04 | c | re-verdicted 2026-09-04: **round 6**: covered with `NC-03`, at zero data rather than zero coefficients |
| NC-05 | c | re-verdicted 2026-09-04: **round 6**: holds by construction — `row()` looks up by name — and the arbitrary-gamma nesting above exercises it |
| NC-06 | c | re-verdicted 2026-09-03: closed by **B1** — the test predicts four scenarios and asserts the spread |
| NC-07 | c | re-verdicted 2026-09-04: **round 6**: three dates fourteen months apart give the identical prediction over a flat series, with a control over the real one. The control gives **two** distinct answers and not three: two of the dates see `High.Season` at zero throughout the 7.89-week window, which is the covariate agreeing with itself rather than the date being ignored |
| NC-08 | c | `test_pnbd_dyncov_predict.py:277,334`, rel 1e-12, both branches |
| NC-09 | c | re-verdicted 2026-09-03: **round 5**: fractional horizons (0.25, 0.5) for the plain model, not dyncov alone, and asserted monotone so a horizon dropped on the floor would show |
| NC-10 | c | re-verdicted 2026-09-03: **round 5**: the working side asserted, the refusals having been covered already |
| NC-11 | c | re-verdicted 2026-09-03: **round 5**: `str`, `date`, `datetime` and `Timestamp` all accepted and shown to normalise to the same timestamp |
| NC-12 | c | |
| NC-13 | — | not reached by the audit |

## S12 Bootstrapping

| | | evidence / note |
|---|---|---|
| B-01 | c | re-verdicted 2026-09-03: **round 5**: the `num_boots=100` default pinned |
| B-02 | c | re-verdicted 2026-09-03: closed by **D6** — the bootstrap identity over the full id list, `test_invariants.py` |
| B-03 | c | |
| B-04 | c | re-verdicted 2026-09-04: **round 6**: the split is a *date*, so it survives only if the rebuild carries it; asserted on both ends, on the holdout rows themselves, and on a draw with repeats, where a customer picked twice must contribute their holdout twice |
| B-05 | c | |
| B-06 | c | |
| B-07 | c | |
| B-08 | c | re-verdicted 2026-09-03: **round 5**: every design row asserted to carry its own customer's covariates, through `bootstrap_apply` and under duplicated ids — the `(600, 2)` shape a wrongly ordered matrix would also satisfy |
| B-09 | c | re-verdicted 2026-09-04: **round 6**: stale -- fixed and tested at `test_diagnostics.py:816`, and in the CHANGELOG |
| B-10 | o | the library never holds a specification: `apply` does its own fitting, so nothing can drop `use_cor` or lambdas. Architectural, not tested — and B-09 breaks the dyncov arm regardless |
| B-11 | c | re-verdicted 2026-09-04: **round 6**: stale -- `TestDrawingEveryCustomerOnce` refits the whole pool and compares coefficients |
| B-12 | c | re-verdicted 2026-09-03: **round 5**: every resampled draw keeps the cohort's estimation and data ends, so the interval is built from answers to one question |
| B-13 | c | |
| B-14 | o | re-verdicted 2026-09-03: **round 5**: `predict` takes no `uncertainty=` argument — intervals are composed by the caller from `bootstrap_apply` and `confidence_intervals`, the same choice the README records for `clv.bootstrapped.apply`. Pinned as a divergence |
| B-15 | c | re-verdicted 2026-09-03: **round 5**: a non-callable `apply` ran all 100 draws and reported all 100 failures; now refused before any runs. The draw-count and level bounds pinned |

## S13 Inference

| | | evidence / note |
|---|---|---|
| I-01 | c | re-verdicted 2026-09-03: **round 5**: `coef`/`vcov`/`summary` name the same parameters in the same order, across all three covariate families and a constrained fit |
| I-02 | c | re-verdicted 2026-09-03: **round 5**: the `vcov` index and columns checked for covariate and constrained fits, not the plain Pareto/NBD alone |
| I-03 | c | re-verdicted 2026-09-04: **round 6**: `parm` did not exist; added, taking names or 0-based positions and returning a `NaN` row for an unknown name, which is R's own behaviour and lets a caller assemble one table across models |
| I-04 | c | re-verdicted 2026-09-03: **round 5**: R's four columns in order, one row per estimated parameter, and that the frame prints |
| I-05 | c ! | **round 5, closed by item 35**: `fit_pnbd_dyncov` now takes `names_cov_constr` and `reg_lambdas`, reusing `_staticcov`'s `_Layout` and `_penalised` — both act on the parameter vector, so neither cares that the covariates vary over time. Three of the four dyncov configurations are constructible and pinned. `use_cor` is not: Sarmanov correlation is a different likelihood rather than a reparameterisation, and the README states the two as separate Pareto/NBD-only features. That last quarter is the `!` |
| I-06 | c | warns rather than raising — recorded in README Findings |
| I-07 | o | no named-parameter accessor exists; all likelihood functions take positional parameters |
| I-08 | c | re-verdicted 2026-09-03: **round 5**: `nobs()` added to `Fitted` — the count was reachable as `n_customers` while the data spelled it `nobs()`, so one question had two spellings. Asserted equal to the data's and to the count `bic` scores against |
| I-09 | c | re-verdicted 2026-09-03: closed by **B3** — `fitted_pnbd` read at `test_diagnostics.py:217`, rtol 1e-10 |
| I-10 | c | re-verdicted 2026-09-03: **round 5**: family-agnostic *by construction* — `likelihood_ratio_test` reads only `log_likelihood` and `n_parameters`, asserted over one constructed result per family rather than six covariate fits |
| I-11 | c | |

## S14 Formula interface

| | | evidence / note |
|---|---|---|
| FI-01..FI-03 | c | |
| FI-04 | c | re-verdicted 2026-09-03: **round 5**: `.` beside another term was read as a literal column name; it now expands against the data in `_narrowed`, all three claims pinned |
| FI-05 | c ! | re-verdicted 2026-09-04: **round 6**: was a defect, now refused. A name given twice built a `(600, 2)` design of two identical columns; the fit reported two coefficients whose *sum* is what the data identifies, each with a standard error the Hessian cannot support |
| FI-06 | c ! | re-verdicted 2026-09-04: **round 6**: the bare-call half was a real gap. In R `I()` protects *operators* from the formula grammar; a call is not formula syntax and needs no protection, so `log(Gender + 2)` should never have needed wrapping. It was refused as "covariates not in the data" -- which reads as a typo in a term that is not one. Both spellings now evaluate; the naming half was already recorded |
| FI-07 | c ! | re-verdicted 2026-09-04: **round 6**: the exclusion half already worked (a README finding); the interaction half did not exist. `*` gives main effects and their product, `:` the product alone, named `Gender.Channel` as `make.names` renders R's `Gender:Channel`. `~Gender*Channel|.-Gender` now gives exactly the life and transaction sets the spec names |
| FI-08 | c | re-verdicted 2026-09-03: **round 5**: the type claim asserted both ways — static narrows to static and is not dynamic, dynamic stays dynamic |
| FI-09 | c ! | **round 5**, a recorded divergence: narrowing shares the transaction and covariate frames where R copies. `ClvData.__init__` copies the *caller's* frame, so nothing a caller holds is reachable from a fitted object; the sharing is between two of this package's own objects. Pinned in both directions and in the README's findings |
| FI-10 | c | |
| FI-11 | c | re-verdicted 2026-09-03: **round 5**: all four claims — every coefficient, every cbs `x`, the id set unchanged and equal to the log's |
| FI-12 | c | re-verdicted 2026-09-03: closed by **D5** — the invariant is stated in `test_invariants.py` |
| FI-13 | c | re-verdicted 2026-09-03: **round 5**: a left-hand side was swallowed into the first covariate name; now refused by name. Non-`ClvData` data was a bare `AttributeError`; now a `TypeError` naming the constructor |
| FI-14 | c | re-verdicted 2026-09-03: **round 5**: non-`ClvData` data now names what the caller got wrong rather than an internal method |
| FI-15 | c | re-verdicted 2026-09-03: **round 5**: `.` with exclusions implemented (`~ . - Gender | .` mis-parsed as one literal term), and R's `constraint()` refused by name with the argument that replaces it |

## S15 Input validation

| | | evidence / note |
|---|---|---|
| V-01 | c | re-verdicted 2026-09-03: **round 5**: a `NaN` or `inf` start passed `np.any(start <= 0)` and was blamed on the objective; now refused by name, with the positivity and length checks shown not to be shadowed |
| V-02 | c | re-verdicted 2026-09-03: **round 5**: `start_cov` is a single scalar, so 5 of 7 claims cannot arise — the divergence is now in the README's findings — and the two that can are pinned; a `NaN` reached the objective and no longer does |
| V-03 | c | re-verdicted 2026-09-04: **round 6**: stale -- item 31's `_reject_unknown_options` refuses an unknown key by name and lists what the method accepts |
| V-04 | o | re-verdicted 2026-09-03: **round 5**: `**kwargs` forwarding lands an unknown argument at the inner signature. Recorded rather than reworked — the message names a private function, which is the cost of the forwarding, and item 31's `options` validation is the precedent for checking where it is given |
| V-05 | c ! | re-verdicted 2026-09-04: **round 6**: nothing existed. Python's truthiness took `None`, `'no'`, `1` and `[True, False]` as `hessian`, and the failure is that the argument *works* -- `hessian='no'` computes a Hessian and says nothing. `single_logical` now refuses all four across the four fits; `np.True_` is accepted, since that is what a comparison on an array yields |
| V-06 | c ! | re-verdicted 2026-09-03: closed by **A4** — `TestBadInputIsLoud`, `TestATransactionMustSayWhoAndWhen` |
| V-07 | o | re-verdicted 2026-09-03: **round 5**: `time_unit` defaults to `"week"` where R requires it — a divergence, now in the README's findings |
| V-08 | o | re-verdicted 2026-09-03: **round 5**: `label`, `other.models` and `annotate.ids` have no counterpart because `diagnostics` returns frames and leaves rendering to the caller — recorded in the README's findings as a smaller surface rather than a gap |

# Appendix 4 — Outcomes: what each finding turned out to be

The three rounds below were written as each was worked, and lived in a
root-level `TASK.md` until 2026-09-03, when they were folded in here: they are
the other half of this document, and a finding's verdict belongs beside the
finding. The briefing that surrounded them -- a reading order, a suggested work
order, a note about splitting a commit -- was spent, and is not carried.

### Outcome, 2026-09-02

Worked in the order suggested above. Every claim was reproduced before it was
acted on -- the standing rule for this audit -- which changed the answer twice.

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

### Outcome, round 2 — findings D5 and D6

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

#### B3, B5 and B7, same round

| | Claim | Verdict |
|---|---|---|
| B3 | `fitted_data`'s doctest is self-referential and reachable from no test file | **confirmed** — and `fitted_pnbd.csv`, R's own `fitted()` over all 313 periods, was committed and orphaned; wired in at rtol 1e-10 |
| B5 / `d_omega` | the oracle is degenerate | **confirmed, and the cause is the data**: all 600 apparel customers were born on a **Sunday**, and the covariate grid starts on one, so `d_omega ≡ 1` and the distance branch of `_distance_to_interval_end` was never reached through it. Four synthetic births fix 7/7, 4/7, 2/7 and 1/7 |
| B5 / `d1` | `d1 ≡ 1` throughout the ABCD table | **confirmed** — the apparel split lands on a covariate boundary. A Wednesday split gives `d1 = 4/7`, and the window is unmoved: the grid is the covariates', not the split's |
| B7 / DY-03 | zero coefficients checked over `customers[:20]`, auxiliary walks only | **confirmed** — now all 600, all four walks, and the 1,866 walk integrals reach all three branches of `walk_integral` |
| B7 / DY-06 | `i` and the window start compared only over the fixture's sample ids | **confirmed** — now over all 600, which needs no oracle |
| B7 / B-02, B-08 | four hand-picked ids, and a `(600, 2)` shape assertion | **superseded** by the bootstrap identity above, which compares both design matrices row for row |

#### D4, and the audit is worked through

DY-22, "all walks are basically correct for an `estimation.split` on every day
of the week". Every dyncov test in this repository ran at
`estimation_split=104`, which lands exactly on the weekly covariate grid, so
all 600-customer oracle comparisons had been made at one alignment out of
seven. No oracle is needed for the other six: S3.3's nesting holds at any
split, so the plain Pareto/NBD's closed form gives each alignment an
independent answer. All seven agree to 1e-12, the cbs the walks carry is the
split's own, one real transaction walk stands per repeat purchase, `d_omega` is
unmoved (it is fixed by the customer's birth) and `T_cal` moves by a day at a
time. The seven likelihoods fall monotonically from −5848.1 to −5879.7, which
is asserted so that a split that never reached the walks would show as one
number repeated.

Splitting the module was part of it: `test_pnbd_dyncov.py` reached 748 code
lines against the 700 limit, and CLAUDE.md's rule is to split rather than
raise. `tests/test_pnbd_dyncov_walks.py` now holds walk *construction* — the
oracle tables, the calendar, the validation — and the shared parameter grid
moved to `conftest.py`.

#### The six `out-of-scope` items, decided

The audit's closing note: "Items marked `out-of-scope` in `docs/spec.md` need a
recorded decision rather than a test — the audit could not otherwise tell a gap
from a choice." Nothing recorded them. The README's *What it implements* now
carries a **Deliberately not ported** list: `bgbb`, `as.clv.data()`, `newdata`
as a keyword, `predict.spending = TRUE`, a specification-carrying bootstrap,
and a named-parameter likelihood accessor.

One of the six turned out to be a discrepancy rather than a decision, and it
runs the other way. The paper states that BG/BB "is not currently included in
CLVTools"; `.Rlib/CLVTools/NAMESPACE` exports `bgbb` and `args(bgbb)` returns a
full fitting signature. The package has moved past the paper there, so M-13's
"scope question: is the BG/BB model in or out?" is answered by the port's own
rule — it follows the paper.

> **Corrected 2026-09-03 — this paragraph is wrong, and the answer it reaches is
> right by luck.** `args(bgbb)` returns a signature because three S4 methods are
> registered; every one of their bodies is a `stop()`. The man page is titled
> "BG/BB models - Work In Progress" and says "Not yet implemented… No value is
> returned", and `bgbb(clvdata(apparelTrans, …))` raises `This model has not yet
> been implemented!`. CLVTools has *not* moved past the paper here — the paper
> is simply accurate. M-13 is not a scope question at all; there is nothing to
> port. Left in place rather than rewritten, because reading a signature as
> behaviour is the mistake worth keeping visible. `docs/backlog.md` item 16.

#### D-17 and NC-13, the two items the audit never reached

Both were marked `—` rather than given a verdict. Both are reachable.

**D-17** holds, and the reason is worth stating: dropping first transactions
and cutting at the split commute *because* the estimation period contains every
customer's first transaction — the estimation start is the earliest of them.
The 1,266 repeat transactions are now reached three ways in
`tests/test_invariants.py`: by construction, by `customer_summary()`'s `x`, and
by the descriptive tracking series restricted to the estimation period.

**NC-13** found two silent acceptances, both settled by asking CLVTools rather
than by argument.

| Input | Was | CLVTools 0.12.1 | Now |
|---|---|---|---|
| `newcustomer("52")` | `TypeError: '<' not supported between instances of 'str' and 'int'` | `num.periods has to be numeric!` | named `TypeError` |
| `newcustomer(nan)` | **accepted** — `nan < 0` is `False` — and became a `NaN` prediction frames later | same error as a string | named `ValueError` |
| a covariate the fit does not carry | **silently dropped**, so a typo returned a plausible number from the covariates that *were* recognised | "has to contain **exactly** the following columns" | named `ValueError` |

The third is the same shape as A4's: a scenario built on `{"Gender", "Channel",
"Gendre"}` answered 2.234 with nothing said. R's word is *exactly*, so both
directions are errors there; only the missing one was an error here.

**Every finding in `docs/spec-audit.md` has now been worked**: A1–A7, B1–B7,
C, D1–D6, the six `out-of-scope` decisions, and D-17 and NC-13. What remains of that document
is the `weak` verdicts it did not individually list, which its own caveat calls
its least certain class — a judgement call rather than a task list.
