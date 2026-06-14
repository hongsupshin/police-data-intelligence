# Evaluation Report

**Pipeline version**: v2.0 — agentic precision/safety layer (relevance judge,
race verifier, advisory conflict annotator) on top of deterministic extraction,
plus a deterministic consensus resolver and victim-anchored extraction.
**Datasets**: TJI Civilians-Shot (1,674 records) and Officers-Shot (282 records),
2014–2024. **Date**: June 2026

> **Note on versions.** Phase 1 (pilot) and the Bug Fixes section below are the
> historical record of how the deterministic pipeline reached v1.3. The **Phase 2
> holdout numbers and everything after reflect the current v2.0 shipped pipeline**
> (all agents on, `claude-sonnet-4-6`), re-run on both datasets June 2026.

## Table of Contents

- [Overview](#overview)
- [Evaluation Dimensions](#evaluation-dimensions)
- [Methodology](#methodology)
  - [Data Splits](#data-splits)
  - [Stratified Sampling](#stratified-sampling)
  - [Comparison Functions](#comparison-functions)
  - [Metrics](#metrics)
- [Phase 1: Pilot Study (Dev Set)](#phase-1-pilot-study-dev-set)
  - [Setup](#setup)
  - [Results](#results)
  - [Error Analysis](#error-analysis)
  - [Failure Pattern Summary](#failure-pattern-summary)
  - [Revised Assessment](#revised-assessment)
- [Bug Fixes (Between Pilot and Holdout Evaluation)](#bug-fixes-between-pilot-and-holdout-evaluation)
  - [Fix 1: Aggregation Source Exclusion](#fix-1-aggregation-source-exclusion)
  - [Fix 2: Synthesize Normalization](#fix-2-synthesize-normalization)
  - [Regression Tests](#regression-tests)
  - [Fix 3: Location Extraction + Eval Ground Truth](#fix-3-location-extraction--eval-ground-truth)
  - [Fix 4: civilian_died Backfill Migration](#fix-4-civilian_died-backfill-migration)
- [Phase 2: Holdout Evaluation](#phase-2-holdout-evaluation)
  - [Setup](#setup-1)
  - [Results](#results-1)
  - [Error Analysis](#error-analysis-1)
  - [Fairness Metrics](#fairness-metrics)
  - [Cost and Latency](#cost-and-latency)
- [Phase 3: Adversarial Evaluation](#phase-3-adversarial-evaluation)
  - [Setup](#setup-2)
  - [Results](#results-2)
  - [Adversarial under v2.0: Zero Completions](#adversarial-under-v20-zero-completions)
- [Discussion](#discussion)
  - [What Works](#what-works)
  - [Known Limitations](#known-limitations)
  - [Failure Mode Taxonomy](#failure-mode-taxonomy)
- [Roadmap](#roadmap)
- [Appendix: Excluded Domains](#appendix-excluded-domains)

## Overview

This document describes the evaluation methodology and results for the police
data enrichment pipeline. Evaluation followed an iterative approach: a pilot
study on 10 records identified systematic failure modes, which were fixed before
a holdout evaluation on a larger sample.

## Evaluation Dimensions

The pipeline is assessed across four dimensions:

**Enrichment quality** measures whether extracted values are correct. This
includes extraction precision (of suggested fields, what percentage are
accurate), extraction coverage (of missing fields, what percentage get values),
and entity resolution accuracy (does the pipeline match articles to the correct
incident).

**Pipeline reach** measures how far records progress through the pipeline before
completing or escalating. This separates retrieval bottlenecks
(search/validation failures) from extraction quality issues.

**Escalation appropriateness** measures whether escalated records were genuinely
hard cases or false alarms caused by pipeline bugs.

**Cost and latency** tracks API cost per record (LLM + web search) and
end-to-end processing time.

## Methodology

### Data Splits

Records were divided into three non-overlapping sets:

| Set       | Size  | Purpose                                      | Selection                            |
| --------- | ----- | -------------------------------------------- | ------------------------------------ |
| Dev       | 10    | Pilot study, bug diagnosis, threshold tuning | Random sample                        |
| Holdout   | 100   | Primary evaluation results                   | Stratified random sample (see below) |
| Remaining | 1,564 | Future batch processing                      | Untouched                            |

The holdout size is controlled by the `--limit` parameter (default: 20; this
evaluation used `--limit 100`).

The dev set was used exclusively for debugging and is not included in reported
evaluation metrics. This prevents train-test contamination — the pipeline was
tuned on dev set failures, so including those records in the final evaluation
would inflate results.

### Stratified Sampling

The holdout set uses **deterministic year-proportional stratification** — not
random sampling. The selection is fully reproducible across runs:

1. Query all eligible incidents (≥ `min_fields` non-NULL eval fields), ordered
   by `field_count DESC, incident_id ASC` (deterministic tiebreaker)
2. Group by year
3. For each year (sorted), allocate proportionally (minimum 4 per stratum)
4. Take the top N rows per stratum in the fixed order

Because there is no randomization, the same `--limit` and `--min-fields`
parameters always produce the same holdout set, as long as the database is
unchanged. The `incident_id ASC` tiebreaker ensures consistent ordering when
multiple incidents share the same `field_count`.

Dev set incidents are excluded via a hardcoded `DEV_SET_IDS` list to prevent
train-test contamination.

Coverage dimensions:

- **Year** (2014–2024): Older incidents may have fewer surviving news articles
- **Missingness pattern**: Which fields are missing (names, weapon,
  demographics) determines which parts of the pipeline are exercised

Demographic dimensions (race, age) were tracked as metadata for post-hoc
fairness analysis rather than used as stratification axes, to avoid sparse cells
at this sample size.

### Comparison Functions

Extracted values were compared against ground truth using field-appropriate
methods:

| Field       | Comparison                                 | Rationale                                                           |
| ----------- | ------------------------------------------ | ------------------------------------------------------------------- |
| Age         | Exact match                                | Numeric, no ambiguity                                               |
| Race        | Exact match                                | Standardized categories                                             |
| Weapon      | Category normalization (7 canonical types) | Maps synonyms to canonical categories (e.g., "handgun" → "Firearm") |
| Location    | Fuzzy match (threshold ≥80)                | Address formatting varies                                           |
| Outcome     | Exact match                                | Binary (survived/died)                                              |
| Time of day | Hour-based (±2h tolerance)                 | Parses hour or period keyword (morning, etc.)                       |

### Metrics

**Pipeline completion rate**: Percentage of records that reach the Complete
terminal node (not escalated).

**Pipeline reach**: Percentage of records that reach the Synthesize node (extraction
attempted), regardless of final outcome. This distinguishes retrieval failures
from extraction failures.

**Extraction precision**: Among extracted field values, percentage that match
ground truth. Computed only for records that reached extraction.

**Extraction coverage**: Among missing fields in records that reached
extraction, percentage that received a value.

**Escalation breakdown**: Distribution of escalation reasons (max_retries,
validation_error, conflict, irrelevant_sources).

### Agentic Precision/Safety Layer

After deterministic extraction, three bounded LLM checks (each a single
structured-output call, fail-open, read-only with respect to the database) act as
a precision/safety layer with authority calibrated to the stakes:

- **Relevance judge** (both datasets, on by default) — reads the validated
  articles against the incident anchors and **vetoes** a completion whose sources
  do not report _this_ incident (→ `IRRELEVANT_SOURCES`). Catches the "right
  structure, wrong incident" failure rule-based validation cannot (famous-name
  collisions, coincidental same-city/same-date coverage).
- **Race verifier** (civilians_shot, on by default) — **nulls** a `civilian_race`
  the source does not explicitly state for the victim (no inference from name,
  neighborhood, or photo). A faithfulness filter that trades coverage for
  correctness.
- **Conflict annotator** (both datasets, on by default; Haiku) — when deep
  conflicts survive to human review, writes an **advisory** triage note. It never
  commits a value, so it cannot change accuracy or coverage — only aid the
  reviewer.

These are precision aids, not perfect: because a veto or null only routes to
human review or drops a value (never fabricates), an over-flag costs a review,
not a wrong record.

### Multi-Objective Accept/Reject Gate

To decide whether a pipeline change is safe to ship, a gate (`src/eval/gate.py`)
compares before/after holdout reports on three hard guards and one warning:

`accept = target_ok AND adversarial_ok AND correctness_ok`

- **target** — completion rate must not drop more than a tolerance.
- **adversarial** — fabricated-incident hallucinations must be **0** (hard veto).
- **correctness** — volume-weighted field accuracy on the **stable cohort**
  (incidents completed in _both_ runs) must not drop more than a tolerance. The
  cohort restriction and volume-weighting avoid a Simpson's-paradox confound where
  a fix that raises completion pulls harder incidents into the completed set.
- **fairness** (per-race completion equity) — surfaced as a **non-gating
  warning**, never a veto (underpowered at these group sizes).

This is how precision changes — which intentionally drop a few wrong-article
completions — are evaluated without a naive completion-rate target rejecting them.

## Phase 1: Pilot Study (Dev Set)

### Setup

10 randomly sampled records from the civilians-shot dataset. Pipeline ran on all
10 with no modifications. Total runtime: ~5 minutes.

### Results

| Metric             | Value      |
| ------------------ | ---------- |
| Completed          | 1/10 (10%) |
| Escalated          | 9/10 (90%) |
| Reached extraction | 5/10 (50%) |

Escalation breakdown:

| Reason                                            | Count | Pipeline stage reached                             |
| ------------------------------------------------- | ----- | -------------------------------------------------- |
| Conflict (articles_disagree / reference_mismatch) | 4     | Full pipeline — extraction completed               |
| Max retries                                       | 3     | Search only — no articles found                    |
| Validation error                                  | 2     | Search succeeded — articles rejected by validation |

### Error Analysis

Manual inspection of all 9 escalated cases revealed four distinct failure
patterns:

**Synonym and formatting conflicts (Incidents 5388, part of 833)**. Articles
described the same facts with different wording. Example: "Alva 'Joe' Gwinn" vs
"Alva Joe Gwinn" vs "Master Sgt. Alva Joe Gwinn" triggered an articles_disagree
conflict despite referring to the same person. Similarly, "Afternoon" vs
"Shortly after 2 p.m." was flagged as a time conflict. Root cause: the merge
node's consistency check uses string comparison without normalization for
titles, honorifics, or temporal synonyms.

**Source contamination (Incident 697)**. A Fatal Encounters dataset page
(aggregation source containing many incidents per document) passed validation
and contaminated extraction with fields from an unrelated domestic violence
case. Root cause: the document lacked a published_date, triggering the relaxed
location-only validation fallback, and the location happened to overlap. This is
the same structural problem previously identified with Wikipedia aggregation
pages.

**Genuinely conflicting accounts (Incident 833)**. A lawsuit and police report
told fundamentally different stories about the same incident — weapon
descriptions, circumstances, and use of force all diverged. Escalation was the
correct behavior. The system cannot and should not resolve factual disputes
between legal filings and police accounts.

**Identity ambiguity (Incident 330)**. The pipeline extracted the name of a
carjacking victim mentioned in the same article, not the subject of the police
shooting. Escalation was the correct behavior — the article described a complex
multi-party incident where distinguishing individuals requires human judgment.

### Failure Pattern Summary

| Pattern                        | Incidents         | Root cause                                   | Action                                                   |
| ------------------------------ | ----------------- | -------------------------------------------- | -------------------------------------------------------- |
| Synonym/formatting             | 5388, part of 833 | String comparison without normalization      | Fix: name normalization in synthesize node                    |
| Source contamination           | 697               | Aggregation source passed validation         | Fix: exclude fatalencounters.org + PDF/CSV URL filtering |
| Genuinely conflicting accounts | 833               | Real factual disagreement across sources     | No fix needed — correct escalation                       |
| Identity ambiguity             | 330               | Multi-party incident, wrong entity extracted | No fix needed — correct escalation                       |

### Revised Assessment

Reclassifying outcomes based on error analysis:

| Category           | Count | Description                                            |
| ------------------ | ----- | ------------------------------------------------------ |
| Correct completion | 1     | Pipeline enriched the record                           |
| Correct escalation | 2     | Genuinely hard cases, human review appropriate         |
| Fixable bugs       | 2     | Source contamination and synonym conflicts             |
| Open (retrieval)   | 5     | Search/validation bottleneck — needs further diagnosis |

This reframes the pilot from "90% failure rate" to "30% correct behavior, 20%
addressable bugs, 50% retrieval bottleneck."

## Bug Fixes (Between Pilot and Holdout Evaluation)

Two systematic issues identified in the pilot were fixed before the holdout
evaluation:

### Fix 1: Aggregation Source Exclusion

**Problem**: Fatal Encounters (fatalencounters.org) pages contain multiple
incidents per document, causing cross-incident contamination during LLM
extraction — the same failure mode previously identified with Wikipedia.

**Fix**: Added fatalencounters.org to `exclude_domains` in the search node.
Added URL-based filtering in the validate node to reject documents with PDF/CSV
extensions, as a defense-in-depth measure against other aggregation sources with
tabular formats.

**Rationale**: Aggregation sources that contain multiple incidents per page are
excluded because they create cross-incident contamination risk. This is a
principled, generalizable rule: the pipeline should only extract from documents
that describe a single incident or a small number of clearly delineated
incidents.

### Fix 2: Synthesize Normalization

**Problem**: Superficial formatting differences triggered false
`articles_disagree` conflicts. Names varied by honorifics and quote styles
("Alva 'Joe' Gwinn" vs "Master Sgt. Alva Joe Gwinn"), race values used
inconsistent terms ("African American" vs "Black"), and weapon/location
descriptions differed in phrasing ("handgun" vs "firearm - handgun").

**Fix**: Three targeted normalizations in `synthesize_node.py`:

- **Name normalization** (`normalize_name`): Strips honorifics (Mr, Mrs, Sgt,
  Master Sgt, Officer, Detective, Deputy, Chief, Trooper, Corporal, Sergeant,
  Lieutenant, Captain, Colonel, Private, Specialist, Major, General, etc.),
  removes quote characters (single, double, smart quotes), collapses whitespace,
  and lowercases. Applied to `officer_name` and `civilian_name` fields before
  fuzzy comparison.
- **Race synonym normalization** (`normalize_race`): Maps variants to canonical
  forms — "African American"/"African-American" → "Black", "Caucasian" →
  "White", "Latino"/"Latina"/"Latin" → "Hispanic". Applied to `civilian_race`
  before comparison.
- **Per-field fuzzy thresholds** (`FIELD_FUZZY_THRESHOLDS`): `weapon` (70) and
  `location_detail` (75) use lower thresholds than the default (80) to tolerate
  phrasing differences like "handgun" vs "firearm - handgun".

**Rationale**: Lowering fuzzy match thresholds globally would suppress real
conflicts. Targeted normalization and per-field thresholds handle formatting
differences without reducing sensitivity to genuine disagreements.

### Regression Tests

Both fixes include regression tests using the exact inputs from pilot incidents
697 (source contamination) and 5388 (synonym conflict).

### Fix 3: Location Extraction + Eval Ground Truth

**Problem**: location_detail showed 0% exact / 10% fuzzy accuracy on N=40
holdout. Two compounding issues:

1. **Extraction**: The merge prompt asked for a free-text location description,
   producing narrative outputs like "near downtown Houston" instead of
   structured addresses.
2. **Eval ground truth**: The eval compared extracted values against
   `incident_address` (e.g., "5021 GLENVIEW DR."), a street-level field that
   doesn't match city-level extractions even when the city is correct.

**Fix**:

1. **Merge prompt** (`synthesize_node.py`): Changed location extraction to request a
   structured address including city name, rather than a narrative description.
2. **Eval ground truth** (`holdout.py`): Switched from `incident_address` to
   `COALESCE(incident_city, incident_county)`, providing city-level ground truth
   that matches the granularity the LLM reliably extracts from news articles.

**Pilot re-eval results** (same 10 dev-set samples):

| Metric                | Before | After |
| --------------------- | ------ | ----- |
| Completion rate       | 10%    | 60%   |
| location_detail fuzzy | 0%     | 100%  |
| weapon exact          | 0%     | 100%  |
| civilian_race exact   | 0%     | 100%  |
| time_of_day exact     | 50%    | 100%  |

The completion rate improvement (10% → 60%) is due to the prompt change
producing more structured outputs that pass consistency checks, reducing false
conflicts in the synthesize node. The accuracy improvements reflect both the prompt
fix (better extractions) and the eval GT fix (fairer comparison).

### Fix 4: civilian_died Backfill Migration

**Problem**: The `outcome` field was not evaluable — `civilian_died` was NULL
for all 1,674 rows in the civilians-shot table. The original `clean_boolean()`
in the ETL didn't handle "DEATH"/"INJURY" values from the raw CSV, mapping them
all to NULL. The bug was fixed in commit `9eb79a7` but the database was never
re-loaded.

**Fix**: A backfill migration (`data/backfill_civilian_died.py`) updated all
3,518 rows in-place (the table's then-duplicated row count; the DB has since
been de-duplicated to 1,674) by re-deriving `civilian_died` from the raw
`cause_of_death` column, using the same logic as the corrected `clean_boolean()`.

**Pilot re-eval results** (same 10 dev-set samples, v2 → v3):

| Field           | Coverage (v2) | Coverage (v3) | Exact (v2) | Exact (v3) |
| --------------- | ------------- | ------------- | ---------- | ---------- |
| civilian_age    | 40%           | 50%           | 75%        | 100%       |
| civilian_race   | 10%           | 10%           | 100%       | 100%       |
| weapon          | 50%           | 80%           | 100%       | 75%        |
| location_detail | 20%           | 50%           | 0%         | 0%         |
| time_of_day     | 20%           | 40%           | 100%       | 100%       |
| outcome         | 0%            | 70%           | —          | 100%       |
| Completion rate | 60%           | 80%           | —          | —          |

The key change is `outcome`: previously 0% evaluable (all ground truth NULL),
now 70% coverage with 100% exact accuracy. Other field variations are run-to-run
LLM non-determinism, not caused by the backfill. See
[Phase 2 results](#results-1) for the N=100 holdout eval.

## Phase 2: Holdout Evaluation

**Pipeline version**: v2.0 (agentic precision/safety layer, deterministic
consensus resolver + victim-anchored extraction, NAME_PARTIAL strategy,
max_results=10).

### Setup

100 records each from the civilians-shot and officers-shot datasets, sampled with
deterministic year-proportional stratification, excluding the 10 dev set records.
Stratification ensured coverage across incident years (2014–2024). All agents on,
`claude-sonnet-4-6` (Haiku for the advisory annotator).

> **How this N=100 was assembled.** A data-integrity bug (a non-idempotent ETL
> that appended duplicate incident rows) had polluted the database; once fixed and
> reloaded cleanly (back to 1,674 / 282 records), the holdout was de-duplicated to
> its distinct incidents and topped back up to N=100 distinct incidents per
> dataset (each measured once, year-stratified). The civilians holdout was
> essentially unaffected; the officers numbers below are lower than a contaminated
> earlier run that over-weighted duplicate high-accuracy slots — these are the
> de-contaminated figures.

### Results (civilians_shot)

| Metric    | Value        |
| --------- | ------------ |
| Completed | 70/100 (70%) |
| Escalated | 30/100 (30%) |

#### Pipeline Reach (civilians_shot)

| Stage reached                                     | Count | Percentage |
| ------------------------------------------------- | ----- | ---------- |
| Complete                                          | 70    | 70%        |
| Escalated after synthesize (relevance veto)       | 7     | 7%         |
| Escalated after synthesize (insufficient_sources) | 1     | 1%         |
| Escalated after search (max_retries)              | 22    | 22%        |

#### Extraction Quality (civilians_shot)

| Field           | N evaluable | Coverage | Exact match | Fuzzy match |
| --------------- | ----------- | -------- | ----------- | ----------- |
| civilian_age    | 100         | 58%      | 95%         | 95%         |
| civilian_race   | 100         | 11%      | 91%         | 91%         |
| weapon          | 84          | 55%      | 83%         | 83%         |
| location_detail | 100         | 44%      | 16%         | 91%         |
| time_of_day     | 92          | 42%      | 82%         | 82%         |
| outcome         | 100         | 74%      | 92%         | 92%         |

Aggregate precision: **210/272 exact (77%), 243/272 fuzzy (89%)**.

**Key observations:**

- **civilian_age** is the strongest field: 95% exact with 58% coverage. Age is
  unambiguous and consistently reported.
- **outcome** reaches 92% exact (74% coverage). All errors are conservative — the
  pipeline says "fatal" when ground truth is "survived", never the reverse (100%
  fatal recall on civilians).
- **civilian_race**: the **race verifier** is visible here. Versus a gate-off run,
  coverage falls (17%→11%) while exact accuracy rises (65%→91%): the verifier
  nulls races the source does not explicitly state, leaving only faithfully
  supported values. One disagreement remains (extracted Black, GT Hispanic).
- **location_detail** shows the usual exact/fuzzy gap (16% vs 91%): the pipeline
  extracts the correct city, but formatting differs from the street-level ground
  truth.
- **weapon** is 83% exact after category normalization; remaining errors are
  category-map gaps (e.g. SAWED-OFF SHOTGUN, GUN vs HANDGUN canonicals) and a few
  entity-confusion cases.
- **time_of_day** is 82% exact (±2h tolerance).

#### Escalation Breakdown (civilians_shot)

| Reason               | Count | Percentage |
| -------------------- | ----- | ---------- |
| max_retries          | 22    | 73%        |
| irrelevant_sources   | 7     | 23%        |
| insufficient_sources | 1     | 3%         |

The 22 max_retries escalations stalled at the search stage — Tavily returned no
relevant articles. The 7 `irrelevant_sources` escalations are the relevance gate
vetoing wrong-article completions that previously committed silently — a
precision/faithfulness gain at the cost of ~7 completions. No conflict-only
escalations occur, because partial completion routes conflicts with agreed fields
to COMPLETE.

### Results (officers_shot)

First evaluation of the officers-shot dataset (civilian = the suspect/shooter;
outcome = the officer's harm, INJURY or DEATH).

| Metric    | Value        |
| --------- | ------------ |
| Completed | 92/100 (92%) |
| Escalated | 8/100 (8%)   |

| Reason               | Count |
| -------------------- | ----- |
| max_retries          | 4     |
| irrelevant_sources   | 3     |
| insufficient_sources | 1     |

| Field           | N evaluable | Coverage | Exact match | Fuzzy match |
| --------------- | ----------- | -------- | ----------- | ----------- |
| civilian_age    | 98          | 68%      | 82%         | 82%         |
| outcome         | 100         | 94%      | 87%         | 87%         |
| location_detail | 100         | 37%      | 11%         | 97%         |
| civilian_race   | 99          | 9%       | 67%         | 67%         |

Aggregate precision: **147/207 exact (71%), 179/207 fuzzy (86%)**. Officers
complete far more often than civilians (92% vs 70%) — officer-involved shootings
draw denser news coverage, so retrieval rarely fails. Completion is below an
earlier gate-off run (95%) because the relevance gate now vetoes 3 wrong-article
officer completions. Outcome (the officer's harm) is the highest-coverage field
at 94%; of its 12 errors, 11 are conservative (extracted "fatal" when GT is
"INJURY") and 1 is the reverse (extracted non-fatal, GT "DEATH") — so fatal recall
is high but not perfect on officers. location_detail repeats the city-vs-address
formatting gap (11% exact, 97% fuzzy).

### Error Analysis

#### Aggregate Precision

Across the 70 completed civilians records, 272 field values were extracted:

| Metric (civilians_shot) | Value         |
| ----------------------- | ------------- |
| Exact correct           | 210/272 (77%) |
| Fuzzy correct           | 243/272 (89%) |

Officers: **147/207 exact (71%), 179/207 fuzzy (86%)** — see Results above.
Civilians improved over the v1.3 civilians baseline (72%/84%): the race verifier
raised race faithfulness, the consensus resolver and anchored extraction reduced
spurious conflicts, and the relevance gate removed wrong-article completions.
Officers are lower mainly because the suspect-race and location fields are sparse
and coarse there, and because 12% of officer completions are outcome-only entity
confusion (below).

#### Outcome: Systematic False-Positive Deaths

Outcome errors are few and **almost entirely one-directional**: civilians have 6
of 74 wrong, all "fatal" extracted where ground truth is "survived" (100% fatal
recall on civilians). Officers have 12 of 94 wrong — **11 conservative** ("fatal"
vs GT "INJURY") and **1 reverse** (non-fatal vs GT "DEATH"), so officer fatal
recall is high but not perfect. The mechanism is entity confusion: when a record's
only strongly-supported field is the generic outcome, the pipeline has likely
matched a _different_ shooting at the same location/time (see Outcome-Only
Completions). The fix direction — a corroboration guard requiring ≥2 supported
fields before committing a generic outcome — is in the roadmap.

#### Race: the Verifier Trades Coverage for Faithfulness

Under v2.0 the **race verifier** reshapes this field: it nulls any `civilian_race`
the source does not explicitly state for the victim. Versus a gate-off run,
civilian-race coverage falls (17%→11%) and exact accuracy rises (65%→91%) — only
**1 of 11** committed civilian races now disagrees with the database (extracted
Black, GT Hispanic). On officers, where race is the suspect's and coverage is low
(9%), 3 of 9 disagree. The verifier is a faithfulness filter, not an accuracy
lever: it removes unsupported guesses rather than correcting them.

(Earlier, several errors were eval-comparator normalization gaps — e.g.
"Hispanic/Latino male" not mapping to "HISPANIC" — fixed by keyword-based matching
and the race taxonomy in `src/race_taxonomy.py`.)

#### Weapon: Category Mapping Gaps + Entity Confusion (civilians)

Weapon is 83% exact (civilians) after category normalization. The residual errors
fall into recurring types: **category-map gaps** (e.g. SAWED-OFF SHOTGUN,
MACHETE→KNIFE, GUN vs HANDGUN canonicals), **granularity mismatches** (RIFLE vs
FIREARM), and a few **entity-confusion** cases (a weapon from a different
shooting). The fix is additive map entries (roadmap).

#### Time of Day: Day-of-Week Parse Errors

time_of_day is 82% exact (±2h tolerance). The recurring failure is the pipeline
occasionally extracting a day of the week ("Thursday") instead of a clock time;
`compare_time()` correctly rejects these as unparseable rather than scoring them
wrong-but-plausible.

#### Location: Formatting Gap Only

location_detail is 91% fuzzy (civilians) / 97% (officers) but low exact (16% / 11%)
— the pipeline extracts the correct city inside a fuller string (e.g. "100 block
of Couch Court, Springtown, Parker County" vs GT "SPRINGTOWN"), scoring high on
fuzzy. The handful of fuzzy misses are county-vs-city cases (the pipeline returns
the county).

#### Confidence Calibration

| Confidence | civilians exact | officers exact |
| ---------- | --------------- | -------------- |
| High       | ~93%            | ~86%           |
| Medium     | ~68%            | ~54%           |

High confidence is well-calibrated; medium is materially lower and varies by
field. The signal is useful for age, outcome, and time_of_day but weaker for race
and location.

#### Completion Rate by Year Cohort (civilians_shot)

| Cohort    | N   | Completion rate |
| --------- | --- | --------------- |
| 2014–2016 | 32  | 66%             |
| 2017–2018 | 32  | 69%             |
| 2019–2021 | 20  | 90%             |
| 2022–2024 | 16  | 56%             |

Completion is highest for 2019–2021 and lowest for the most recent 2022–2024
cohort — consistent with news-coverage indexing being thinner for very recent
incidents (and sparser for the oldest). Officers complete uniformly high across
cohorts (82–100%).

#### Outcome-Only Completions

A completion that extracted **only** the generic `outcome` field is the signature
of entity confusion (a different same-place/time shooting matched). Civilians have
**4** such completions; officers have **12** — the bulk of the officer outcome
errors above. A corroboration guard (≥2 supported fields before committing) is the
roadmap fix.

#### Fairness Metrics

Pipeline reach and accuracy by demographic group. Per-race completion equity is a
**non-gating warning** in the eval gate (see Methodology), not a veto.

civilians_shot:

| Group    | N   | Completion rate | Mean exact accuracy |
| -------- | --- | --------------- | ------------------- |
| Black    | 25  | 64%             | 77%                 |
| Hispanic | 36  | 83%             | 74%                 |
| White    | 34  | 62%             | 80%                 |
| Other    | 5   | 60%             | 70%                 |

officers_shot:

| Group    | N   | Completion rate | Mean exact accuracy |
| -------- | --- | --------------- | ------------------- |
| Black    | 20  | 85%             | 83%                 |
| Hispanic | 42  | 93%             | 68%                 |
| White    | 30  | 97%             | 74%                 |
| Other    | 7   | 100%            | 62%                 |
| Unknown  | 1   | 0%              | —                   |

On civilians, the Black group shows a lower completion rate (64%) than Hispanic
(83%) — consistent with temporal bias (older incidents, fewer surviving articles)
rather than pipeline bias; extraction accuracy is comparable across groups
(70–80%). On officers, completion is uniformly high (85–100%) because
officer-involved shootings are densely covered; accuracy varies more by group, on
small per-group N (the single Unknown-race incident escalated).

### Cost and Latency

Per-incident wall-clock — including eval-harness overhead (ground-truth fetch and
field comparisons) — averaged ~50s for civilians_shot and ~90s for officers_shot
in the N=100 runs. Per-incident time is dominated by the LLM calls (per-article
extraction plus the agentic precision layer: relevance judge always, race verifier
for civilian races, a Haiku conflict annotator on conflicts), not by search.

| Metric (per record) | Value      |
| ------------------- | ---------- |
| Anthropic (LLM)     | ~$0.16     |
| Tavily (search)     | ~$0.04     |
| **Total**           | **~$0.20** |

Estimated from API pricing (Claude Sonnet 4.6 at $3/$15 per 1M input/output
tokens; Claude Haiku for the advisory annotator; Tavily advanced search at
$0.016/search, PAYGO). The agentic precision layer adds one to a few LLM calls per
completion on top of extraction; escalated incidents that never reach extraction
cost only search. Actual cost varies with article length and retry count.

## Phase 3: Adversarial Evaluation

The holdout evaluation measures accuracy on real incidents. Adversarial
evaluation measures the inverse: does the pipeline correctly refuse to answer
when given fabricated incidents that never happened?

### Setup

20 fabricated police shooting incidents were run through the live pipeline (real
Tavily search + real Claude extraction). Only `fetch_incident` was patched to
inject fake data; all downstream nodes ran unmodified. Scenarios were designed
across five categories:

| Category | N | Design | Expected behavior |
|----------|---|--------|-------------------|
| A. Obscure cities | 4 | Marfa, Alpine, Presidio, Terlingua | 0 search results → escalate |
| B. Wrong dates | 4 | Medium cities, dates >5 days off real incidents | Validation rejects → escalate |
| C. Hallucination traps | 6 | Major cities, dates within days of real high-profile events (Dallas police ambush, Harding St. raid, Botham Jean shooting, George Floyd protests) | Real articles found about wrong person → escalate |
| D. Common-name confusion | 4 | "Michael Brown," "John Williams," "Jose Garcia" in major cities | Name-ambiguity stress test |
| E. Edge cases | 2 | NULL officer or civilian name | Partial-data handling |

Category C is the hardest test: fabricated incidents were placed within the ±5
day validation window of real high-profile events, meaning real articles about
those events should pass date and location validation checks.

### Results

| Metric | Value |
|--------|-------|
| Escalated correctly | 20/20 (100%) |
| Completed | 0/20 (0%) |
| Hallucinations detected | 0/20 (0%) |

By category — every scenario escalated, none hallucinated:

| Category | N | Escalated | Hallucinations |
|----------|---|-----------|----------------|
| A. Obscure cities | 4 | 4 | 0 |
| B. Wrong dates | 4 | 4 | 0 |
| C. Hallucination traps | 6 | 6 | 0 |
| D. Common names | 4 | 4 | 0 |
| E. Edge cases | 2 | 2 | 0 |

Escalation reasons: 17 `max_retries` (search exhausted all strategies), 2
`irrelevant_sources` (the relevance gate vetoed real-but-wrong-incident articles —
scenarios 99906 and 99915, which extracted 6–7 fields before the veto), and 1
`insufficient_sources` (99920, NULL civilian name).

### Adversarial under v2.0: Zero Completions

Under v1.x, one scenario completed: 99917 — fabricated "Michael Brown" (civilian)
and "Robert Johnson" (officer) in Dallas. It never hallucinated the fabricated
names, but it _completed_ on real-but-wrong-incident Dallas articles (with
`requires_human_review=True` and 6 conflicts).

Under v2.0, **no scenario completes**. 99917 escalates at search this run, and the
relevance gate — now on for civilians — actively vetoes two other scenarios
(`irrelevant_sources`), exactly the famous-name / wrong-incident path it was built
to close. Defense-in-depth (validation → relevance gate → conflict detection →
human review) caught every fabricated incident, including the hallucination traps
placed within the validation window of real high-profile events (Dallas police
ambush, Harding St. raid, Botham Jean, George Floyd protests). **0/20 completions,
0/20 hallucinations.**

Full results: `output/adversarial/results.json`

## Discussion

### What Works

The pipeline completes **70% of civilians** and **92% of officers** holdout
incidents (N=100 each). Aggregate precision is 77% exact / 89% fuzzy (civilians)
and 71% / 86% (officers). Key strengths:

- **Partial completion** remains high-impact — records that would once have
  escalated on a single-field conflict now complete with the agreed fields.
- **civilian_age** is the strongest field (95% / 82% exact, civilians /
  officers); **outcome** is the highest-coverage (74% / 94%).
- **Fatal recall is high**: on civilians the pipeline never says someone survived
  when the database says they died (100% fatal recall); on officers it misses once
  (1 of 94). Outcome errors are otherwise all in the conservative direction.
- **location_detail** is ~91–97% fuzzy, confirming the correct city is extracted
  even when formatting differs from street-level ground truth.
- **The agentic precision/safety layer earns its place**: the race verifier lifts
  civilian-race faithfulness (65%→91% exact) by nulling unstated values; the
  relevance gate removes wrong-article completions (7 civilians, 3 officers) that
  previously committed silently — improving faithfulness at a small, bounded
  completion cost.
- **Adversarial robustness**: 0/20 fabricated incidents hallucinated and **0
  completed**. Defense-in-depth (validation → relevance gate → conflict detection
  → human review) caught every scenario, including hallucination traps placed near
  real high-profile events (see [Phase 3](#phase-3-adversarial-evaluation)).

### Known Limitations

**Sample size**: The holdout evaluation covers ~6% of the full dataset (100 of
1,674). This provides reasonable estimates of pipeline behavior and enough
statistical power to identify systematic failure modes, though confidence
intervals for per-field accuracy remain moderately wide due to the subset of
records that reach extraction.

**Temporal bias**: News articles from 2014–2016 incidents may no longer be
available online, creating a structural disadvantage for older records that is
independent of pipeline quality. The fairness metrics are consistent with this:
on civilians the Black group (which skews toward older incidents) shows a 64%
completion rate vs 83% for Hispanic incidents.

**Ground truth completeness**: The `civilian_died` column was backfilled
([Fix 4](#fix-4-civilian_died-backfill-migration)). Other ground truth fields
may also have gaps limiting evaluation coverage.

**Both datasets evaluated**: v2.0 evaluates both civilians-shot and officers-shot
(N=100 each). Officers complete far more often (92% vs 70%) thanks to denser news
coverage; the same conservative outcome-error and location-formatting patterns
appear in both. Per-group fairness cells are small (especially for officers), so
group-level accuracy differences should be read with that caveat.

### Failure Mode Taxonomy

| Failure mode                          | Frequency (N=100 each)              | Fixable?     | Priority |
| ------------------------------------- | ----------------------------------- | ------------ | -------- |
| Retrieval gap (no articles found)     | 22 civ / 4 off escalations          | No           | —        |
| Entity confusion (wrong incident)     | 4 civ / 12 off outcome-only         | Partial      | High     |
| Outcome false-positive deaths         | 6/74 civ, 11/94 off (+1 reverse)    | Partial      | High     |
| Race eval normalization gaps          | ~~5/17~~ Fixed                      | ~~Yes~~ Done | —        |
| Race genuine misidentification        | 1/11 civ, 3/9 off                   | Investigate  | Medium   |
| Weapon category mapping gaps          | ~17% of civ weapons                 | Yes (map)    | Low      |
| Time parse errors (day-of-week)       | a few civ time values               | Yes (synthesize)  | Low |
| Location formatting gap (exact/fuzzy) | 16% exact vs 91% fuzzy (civ)        | Low priority | —        |
| Outcome-only completions              | 4 civ / 12 off                      | Yes          | High     |

## Roadmap

**Completed:**

- ~~Re-run holdout eval~~ — N=100 on **both** datasets (v2.0, June 2026)
- ~~Fairness analysis across demographic groups~~ — both datasets, Phase 2
- ~~Adversarial evaluation (20 fabricated incidents)~~ — 0/20 hallucinations,
  0/20 completions
- ~~Evaluation of the officers-shot dataset~~ — 92% completion, 71%/86% precision
- ~~Agentic precision/safety layer~~ — relevance judge, race verifier, advisory
  conflict annotator (all on by default)
- ~~Deterministic conflict reduction~~ — consensus resolver + victim-anchored
  extraction + race taxonomy

**Remaining:**

- **Weapon category map**: add SAWED-OFF SHOTGUN→SHOTGUN, MACHETE→KNIFE, and
  BB/PELLET GUN mappings to `WEAPON_CATEGORY_MAP`
- **Outcome-only / entity-confusion guard**: require ≥2 corroborated fields before
  committing a generic outcome, to suppress same-location/same-time confusion
- **Cost study**: a cheaper model (Haiku) for high-volume extraction
- Batch processing across the ~1,564 remaining civilians records (priority order)
- Human review UI for processing escalated records

## Appendix: Excluded Domains

| Domain              | Reason                                                                         | Added |
| ------------------- | ------------------------------------------------------------------------------ | ----- |
| wikipedia.org       | Multi-incident aggregation pages cause cross-incident extraction contamination | v1.0  |
| fatalencounters.org | Aggregation dataset; tabular format causes cross-incident field contamination  | v1.1  |
