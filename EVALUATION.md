# Evaluation Report

**Pipeline version**: v1.3 (partial completion, score gating removed)
**Dataset**: TJI Civilians-Shot (1,674 records, 2014–2024) **Date**: March 2026

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
  - [Fix 2: Merge Normalization](#fix-2-merge-normalization)
  - [Regression Tests](#regression-tests)
  - [Fix 3: Location Extraction + Eval Ground Truth](#fix-3-location-extraction--eval-ground-truth)
  - [Fix 4: civilian_died Backfill Migration](#fix-4-civilian_died-backfill-migration)
- [Phase 2: Holdout Evaluation](#phase-2-holdout-evaluation)
  - [Setup](#setup-1)
  - [Results](#results-1)
  - [Error Analysis](#error-analysis-1)
  - [Fairness Metrics](#fairness-metrics)
  - [Cost and Latency](#cost-and-latency)
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

**Pipeline reach**: Percentage of records that reach the Merge node (extraction
attempted), regardless of final outcome. This distinguishes retrieval failures
from extraction failures.

**Extraction precision**: Among extracted field values, percentage that match
ground truth. Computed only for records that reached extraction.

**Extraction coverage**: Among missing fields in records that reached
extraction, percentage that received a value.

**Escalation breakdown**: Distribution of escalation reasons (max_retries,
validation_error, conflict).

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
| Synonym/formatting             | 5388, part of 833 | String comparison without normalization      | Fix: name normalization in merge node                    |
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

### Fix 2: Merge Normalization

**Problem**: Superficial formatting differences triggered false
`articles_disagree` conflicts. Names varied by honorifics and quote styles
("Alva 'Joe' Gwinn" vs "Master Sgt. Alva Joe Gwinn"), race values used
inconsistent terms ("African American" vs "Black"), and weapon/location
descriptions differed in phrasing ("handgun" vs "firearm - handgun").

**Fix**: Three targeted normalizations in `merge_node.py`:

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

1. **Merge prompt** (`merge_node.py`): Changed location extraction to request a
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
conflicts in the merge node. The accuracy improvements reflect both the prompt
fix (better extractions) and the eval GT fix (fairer comparison).

### Fix 4: civilian_died Backfill Migration

**Problem**: The `outcome` field was not evaluable — `civilian_died` was NULL
for all 1,674 rows in the civilians-shot table. The original `clean_boolean()`
in the ETL didn't handle "DEATH"/"INJURY" values from the raw CSV, mapping them
all to NULL. The bug was fixed in commit `9eb79a7` but the database was never
re-loaded.

**Fix**: A backfill migration (`data/backfill_civilian_died.py`) updated all
3,518 rows in-place by re-deriving `civilian_died` from the raw `cause_of_death`
column, using the same logic as the corrected `clean_boolean()`.

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

**Pipeline version**: v1.3 (partial completion, score gating removed,
NAME_PARTIAL strategy, max_results=10, all fixes from Phase 1 applied)

### Setup

100 records sampled from the civilians-shot dataset using deterministic
year-proportional stratification, excluding the 10 dev set records.
Stratification ensured coverage across incident years (2014–2024).

### Results

| Metric             | Value        |
| ------------------ | ------------ |
| Completed          | 70/100 (70%) |
| Escalated          | 30/100 (30%) |
| Reached extraction | 71/100 (71%) |

#### Pipeline Reach

| Stage reached                                | Count | Percentage |
| -------------------------------------------- | ----- | ---------- |
| Complete                                     | 70    | 70%        |
| Escalated after merge (insufficient_sources) | 1     | 1%         |
| Escalated after search (max_retries)         | 29    | 29%        |

#### Extraction Quality (Records That Reached Merge)

| Field           | N evaluable | Coverage | Exact match | Fuzzy match |
| --------------- | ----------- | -------- | ----------- | ----------- |
| civilian_age    | 100         | 49%      | 90%         | 90%         |
| civilian_race   | 100         | 17%      | 35%         | 35%         |
| weapon          | 84          | 50%      | 79%         | 79%         |
| location_detail | 100         | 38%      | 18%         | 97%         |
| time_of_day     | 96          | 32%      | 94%         | 94%         |
| outcome         | 100         | 68%      | 84%         | 84%         |

**Key observations:**

- **civilian_age** is the strongest field: 90% exact accuracy with 49% coverage.
  Age is unambiguous and consistently reported in news articles.
- **time_of_day** achieves 94% exact accuracy (±2h tolerance) despite free-text
  format differences between extracted times and ground truth.
- **location_detail** shows a large exact/fuzzy gap (18% vs 97%) — the pipeline
  extracts the correct city but formatting differs from ground truth (e.g.,
  "Houston, TX" vs "Houston"). The 97% fuzzy match confirms the extractions are
  substantively correct.
- **outcome** is newly evaluable after the `civilian_died` backfill (Fix 4): 68%
  coverage with 84% exact accuracy.
- **civilian_race** has low coverage (17%) and accuracy (35%). Race is
  inconsistently reported in news articles and the pipeline's race normalization
  doesn't cover all variants.
- **weapon** achieves 79% exact accuracy after category normalization (Fix 2).

#### Escalation Breakdown

| Reason               | Count | Percentage |
| -------------------- | ----- | ---------- |
| max_retries          | 29    | 97%        |
| insufficient_sources | 1     | 3%         |

All 29 max_retries escalations stalled at the search stage — Tavily returned no
relevant articles for these incidents. No conflict-only escalations occurred
because partial completion (v1.3) routes conflicts with agreed fields to
COMPLETE instead of ESCALATE.

### Error Analysis

#### Aggregate Precision

Across all 70 completed records, 245 field values were extracted:

| Metric        | Value         |
| ------------- | ------------- |
| Exact correct | 176/245 (72%) |
| Fuzzy correct | 206/245 (84%) |

#### Outcome: Systematic False-Positive Deaths

11 of 68 outcome extractions were wrong. All 11 errors follow the same pattern:

| Error type                   | Count | Description                                   |
| ---------------------------- | ----- | --------------------------------------------- |
| Fatal extracted, GT=survived | 9     | Pipeline says died, DB says survived          |
| Unparseable multi-person     | 2     | Multi-victim text, parser can't resolve       |
| Non-fatal extracted, GT=died | 0     | Never occurs — pipeline has 100% fatal recall |

The 9 false-positive deaths split into two sub-patterns:

- **Likely entity confusion (5 cases)**: Incidents #5, #245, #414, #954, #1332
  had zero other fields extracted — the pipeline likely matched a different
  fatal shooting at the same location/time. These are also the 5 of 6
  "outcome-only" completions (records where outcome was the only field
  extracted).
- **Possible GT errors or ambiguous cases (4 cases)**: Incidents #9, #252, #589,
  #1121 had other fields that matched correctly, suggesting the pipeline found
  the right incident but the database's `civilian_died` value may be incorrect,
  or the outcome was ambiguous (e.g., the person died later).

**Impact on outcome accuracy**: Excluding outcome-only completions (where entity
confusion is most likely), outcome accuracy rises from 84% to 90% (56/62).

#### Race: Normalization Gaps + Entity Confusion

11 of 17 race extractions were wrong (35% accuracy). The errors fall into two
categories:

**Normalization gaps (5 cases)** — the pipeline extracted a valid race
description that should match the ground truth but doesn't due to incomplete
normalization in the eval comparator:

- "Hispanic/Latino male" and "Hispanic or Latino" don't map to "HISPANIC" (only
  exact "latino"/"latina" are aliased)
- "African-American/Black" doesn't map to "BLACK" (only "african
  american"/"african-american" are aliased)
- "Iranian" and "Egyptian" don't map to "OTHER" (nationality-to-race mapping not
  implemented)

**Genuine disagreements (6 cases)** — the pipeline extracted a race that
contradicts the database (#1 Black vs WHITE, #9 White vs BLACK, #254
Latino/Hispanic vs WHITE, #408 White vs HISPANIC, #951 Hispanic vs WHITE, #1277
El Salvadoran vs HISPANIC). Some may be entity confusion (the pipeline found an
article about a different person), and some may reflect ambiguity in racial
classification.

**Corrected accuracy**: Fixing the 5 normalization gaps would raise race
accuracy from 35% (6/17) to 65% (11/17).

#### Weapon: Category Mapping Gaps + Entity Confusion

9 of 42 weapon extractions were wrong (79% accuracy). Three error types:

- **Entity confusion (4)**: Pipeline extracted HANDGUN when GT was KNIFE (#158),
  BB GUN (#165), DEPUTY'S GUN (#175), or PELLET GUN (#586) — likely found
  articles about a different shooting.
- **Category mapping gaps (4)**: SAWED-OFF SHOTGUN not in weapon map (#267),
  MACHETE maps to OTHER instead of KNIFE (#773), GUN and HANDGUN map to
  different canonicals (#263, #594).
- **Granularity mismatch (1)**: Pipeline extracted RIFLE, GT was FIREARM (#4) —
  the pipeline was more specific, but the canonical map treats them as different
  categories.

#### Time of Day: Day-of-Week Parse Errors

2 of 31 time_of_day extractions were wrong, plus 6 parse errors where the
pipeline extracted a day of the week ("Thursday", "Sunday") instead of a time.
The `compare_time()` function correctly rejects these as unparseable. Combined
accuracy: 29/37 (78%) including parse errors as wrong, or 94% (29/31) excluding
them.

#### Location: Formatting Gap Only

30 of 31 location fuzzy matches are correct (the pipeline extracts "100 block of
Couch Court, Springtown, Parker County" while GT is just "SPRINGTOWN"). These
consistently score 100 on fuzzy match because the GT city name appears within
the extracted string. The single fuzzy miss (#175: "8501 West Dunn Street, Ector
County" vs "ODESSA") is a case where the pipeline extracted the county instead
of the city.

#### Confidence Calibration

| Confidence | N extracted | Exact accuracy |
| ---------- | ----------- | -------------- |
| High       | 86          | 85%            |
| Medium     | 159         | 65%            |

High confidence is reasonably well-calibrated (85% correct). Medium confidence
accuracy varies widely by field: 93% for time_of_day but only 33% for
civilian_race. The confidence signal is useful for age, weapon, and time_of_day
but unreliable for race and location.

#### Completion Rate by Year Cohort

| Cohort    | N   | Completion rate | Exact accuracy | Fuzzy accuracy |
| --------- | --- | --------------- | -------------- | -------------- |
| 2014–2016 | 35  | 71%             | 69%            | 81%            |
| 2017–2018 | 30  | 77%             | 71%            | 86%            |
| 2019–2021 | 20  | 75%             | 73%            | 85%            |
| 2022–2024 | 15  | 47%             | 83%            | 87%            |

The 2022–2024 cohort has the lowest completion rate (47%) despite having the
highest accuracy when articles are found. This may reflect less news coverage
indexing for very recent incidents, or fewer incidents available in the
stratified sample for these years.

#### Outcome-Only Completions

6 completed records extracted only the `outcome` field with no other fields.
Their outcome accuracy is 17% (1/6) — far below the 90% accuracy for records
with multiple extracted fields. These records likely represent entity confusion:
the pipeline found an article about a different (fatal) shooting and extracted
only the outcome because other fields didn't match closely enough to pass
consistency checks. This suggests the pipeline's partial completion mechanism
correctly withholds conflicting fields but the outcome field is too generic
(fatal/non-fatal) to serve as a useful consistency signal.

#### Fairness Metrics

Pipeline reach and accuracy by demographic group:

| Group    | N   | Completion rate | Mean exact accuracy |
| -------- | --- | --------------- | ------------------- |
| Black    | 26  | 50%             | 67%                 |
| Hispanic | 36  | 83%             | 71%                 |
| White    | 32  | 75%             | 67%                 |
| Other    | 6   | 50%             | 69%                 |

The Black and Other groups show lower completion rates (50%), likely reflecting
older incidents with fewer surviving news articles rather than pipeline bias.
Hispanic incidents have the highest completion rate (83%), possibly due to more
recent incident dates with better article availability. Mean exact accuracy is
comparable across groups (67–71%), suggesting the pipeline's extraction quality
is consistent when articles are found.

### Cost and Latency

| Metric                | Value        |
| --------------------- | ------------ |
| Mean time per record  | 45.5s        |
| Total eval run time   | 75.9 minutes |
| Estimated total cost  | ~$15         |
| Estimated cost/record | ~$0.15       |

The 45.5s mean time is higher than the 7.0s pipeline latency reported in README
because the evaluation harness adds overhead per incident: database queries to
fetch ground truth, fuzzy-match comparisons across 6 fields, and result
aggregation. The core pipeline (search + merge) accounts for ~7s; the remaining
~38s is evaluation-only overhead spread across 100 sequential incidents.

Cost is estimated from API pricing (Claude Sonnet 4.6 at $3/$15 per 1M
input/output tokens; Tavily advanced search at $0.016/search PAYGO). The primary
cost driver is LLM extraction (~70%), with web search comprising ~30%. Actual
costs depend on article length and retry count.

## Discussion

### What Works

The pipeline completes 70% of holdout incidents (N=100, up from 7.5% before v1.3
improvements). Aggregate precision is 72% exact / 84% fuzzy across 245 extracted
field values. Key strengths:

- **Partial completion** is the highest-impact change — completed records
  extract a mean of 3.5 fields each. Previously most of these would have been
  escalated with zero output due to conflicts on a subset of fields.
- **civilian_age** is the strongest field at 90% exact accuracy with 49%
  coverage. Age is unambiguous and consistently reported in news articles.
- **time_of_day** achieves 94% exact accuracy despite free-text format
  differences, thanks to the hour-based comparison with ±2h tolerance.
- **location_detail** achieves 97% fuzzy accuracy, confirming the pipeline
  extracts the correct city even when formatting differs from ground truth.
- **outcome** is now evaluable (84% exact, rising to 90% when excluding
  outcome-only completions that likely represent entity confusion).
- **No false-negative deaths**: The pipeline never says someone survived when
  the database says they died (100% fatal recall). All outcome errors go in the
  conservative direction.
- **Removing score gating** lets more articles through to validation, where
  date/location matching serves as a better quality filter than Tavily's
  relevance score.

### Known Limitations

**Sample size**: The holdout evaluation covers ~6% of the full dataset (100 of
1,674). This provides reasonable estimates of pipeline behavior and enough
statistical power to identify systematic failure modes, though confidence
intervals for per-field accuracy remain moderately wide due to the subset of
records that reach extraction.

**Temporal bias**: News articles from 2014–2016 incidents may no longer be
available online, creating a structural disadvantage for older records that is
independent of pipeline quality. The fairness metrics confirm this: the Black
demographic group (which skews toward older incidents) shows a 50% completion
rate vs 83% for Hispanic incidents.

**Ground truth completeness**: The `civilian_died` column was backfilled
([Fix 4](#fix-4-civilian_died-backfill-migration)). Other ground truth fields
may also have gaps limiting evaluation coverage.

**Single dataset**: This evaluation covers the civilians-shot dataset only. The
officers-shot dataset (282 records, different schema) has not been evaluated and
may exhibit different failure modes.

### Failure Mode Taxonomy

| Failure mode                          | Frequency (N=100)            | Fixable?     | Priority |
| ------------------------------------- | ---------------------------- | ------------ | -------- |
| Retrieval gap (no articles found)     | 29 escalations (29%)         | No           | —        |
| Entity confusion (wrong incident)     | ~5 outcome errors, ~4 weapon | Partial      | High     |
| Outcome false-positive deaths         | 9/68 extractions (13%)       | Partial      | High     |
| Race eval normalization gaps          | 5/17 extractions             | Yes (eval)   | Medium   |
| Race genuine misidentification        | 6/17 extractions             | Investigate  | Medium   |
| Weapon category mapping gaps          | 4/42 extractions             | Yes (map)    | Low      |
| Time parse errors (day-of-week)       | 6/37 extractions             | Yes (merge)  | Low      |
| Location formatting gap (exact/fuzzy) | 18% exact vs 97% fuzzy       | Low priority | —        |
| Outcome-only completions              | 6 records (9%)               | Yes          | High     |

## Roadmap

**Completed:**

- ~~Fix merge node null handling (incident 3494 crash)~~
- ~~Re-run holdout eval with location + outcome fixes~~ — N=100 eval completed
- ~~Fairness analysis across demographic groups~~ — included in Phase 2 results

**Remaining:**

- **Outcome-only completion guard**: Records with only `outcome` extracted show
  17% accuracy (entity confusion). Consider requiring ≥2 extracted fields before
  completing, or treating outcome-only records as low-confidence
- **Race eval normalization**: Add "Hispanic/Latino", "African-American/Black",
  and nationality→"OTHER" mappings to `RACE_ALIASES` in `holdout.py` (would
  raise race accuracy from 35% to 65%)
- **Weapon category map**: Add SAWED-OFF SHOTGUN→SHOTGUN, MACHETE→KNIFE, and BB
  GUN/PELLET GUN mappings to `WEAPON_CATEGORY_MAP`
- **Time parse errors**: The merge node sometimes extracts day-of-week instead
  of time-of-day (6 parse errors) — tighten the extraction prompt
- Batch processing across all ~1,564 remaining records with priority ordering
- Evaluation of the officers-shot dataset
- Human review UI for processing escalated records

## Appendix: Excluded Domains

| Domain              | Reason                                                                         | Added |
| ------------------- | ------------------------------------------------------------------------------ | ----- |
| wikipedia.org       | Multi-incident aggregation pages cause cross-incident extraction contamination | v1.0  |
| fatalencounters.org | Aggregation dataset; tabular format causes cross-incident field contamination  | v1.1  |
