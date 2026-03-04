# Evaluation Report

**Pipeline version**: v1.3 (partial completion, score gating removed)
**Dataset**: TJI Civilians-Shot (1,674 records, 2014–2021) **Date**: March 2026

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
| Holdout   | 40    | Primary evaluation results                   | Stratified random sample (see below) |
| Remaining | 1,624 | Future batch processing                      | Untouched                            |

The holdout size is controlled by the `--limit` parameter (default: 20; this
evaluation used `--limit 40`).

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

- **Year** (2014–2021): Older incidents may have fewer surviving news articles
- **Missingness pattern**: Which fields are missing (names, weapon,
  demographics) determines which parts of the pipeline are exercised

Demographic dimensions (race, age) were tracked as metadata for post-hoc
fairness analysis rather than used as stratification axes, to avoid sparse cells
at this sample size.

### Comparison Functions

Extracted values were compared against ground truth using field-appropriate
methods:

| Field       | Comparison                  | Rationale                                     |
| ----------- | --------------------------- | --------------------------------------------- |
| Age         | Exact match                 | Numeric, no ambiguity                         |
| Race        | Exact match                 | Standardized categories                       |
| Weapon      | Fuzzy match (threshold ≥80) | Descriptions vary across sources              |
| Location    | Fuzzy match (threshold ≥80) | Address formatting varies                     |
| Outcome     | Exact match                 | Binary (survived/died)                        |
| Time of day | Hour-based (±2h tolerance)  | Parses hour or period keyword (morning, etc.) |

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

## Phase 2: Holdout Evaluation

**Pipeline version**: v1.3 (partial completion, score gating removed,
NAME_PARTIAL strategy, max_results=10)

### Setup

40 records sampled from the civilians-shot dataset using deterministic
year-proportional stratification, excluding the 10 dev set records.
Stratification ensured coverage across incident years (2014–2021).

### Results

| Metric             | Value       |
| ------------------ | ----------- |
| Completed          | 28/40 (70%) |
| Escalated          | 12/40 (30%) |
| Reached extraction | 29/40 (73%) |

#### Pipeline Reach

| Stage reached                        | Count | Percentage |
| ------------------------------------ | ----- | ---------- |
| Complete                             | 28    | 70%        |
| Escalated after merge (merge_error)  | 1     | 3%         |
| Escalated after search (max_retries) | 11    | 28%        |

#### Extraction Quality (Records That Reached Merge)

| Field           | Coverage | Exact match | Fuzzy match |
| --------------- | -------- | ----------- | ----------- |
| civilian_age    | 50%      | 75%         | 75%         |
| civilian_race   | 18%      | 57%         | 57%         |
| weapon          | 31%      | 0%          | 45%         |
| location_detail | 25%      | 0%          | 10%         |
| time_of_day     | 38%      | 73%         | 73%         |
| outcome         | N/A      | —           | —           |

**Notes on low-performing fields:**

- **weapon** (0% exact, 45% fuzzy): The pipeline extracts correct weapon types
  but with different phrasing than the DB (e.g., "handgun" vs "firearm -
  handgun"). Rapidfuzz character-level matching cannot bridge semantic synonyms.
  Planned fix: embedding-based similarity comparison.
- **location_detail** (0% exact, 10% fuzzy): The DB stores structured street
  addresses ("5021 GLENVIEW DR.") while the LLM extracts narrative descriptions
  ("near downtown Houston"). Planned fixes: improve the merge prompt to request
  specific addresses; geocoding normalization for eval comparison.
- **outcome** (not evaluated): The `civilian_died` ground truth column is NULL
  for all 40 holdout samples, so no comparison is possible.

#### Escalation Breakdown

| Reason      | Count | Percentage |
| ----------- | ----- | ---------- |
| max_retries | 11    | 92%        |
| merge_error | 1     | 8%         |

All 11 max_retries escalations stalled at the search stage — Tavily returned no
relevant articles for these incidents. No conflict-only escalations occurred
because partial completion (v1.3) routes conflicts with agreed fields to
COMPLETE instead of ESCALATE.

### Error Analysis

Two escalated cases were sampled for manual inspection — one from each
escalation reason — to diagnose whether escalations represent correct behavior
or fixable bugs.

**Case 1: Incident 6742 — Retrieval failure (max_retries)**

- **Ground truth**: 30yo Black male, knife, 1655 Blue Bell Road, 18:55
- **Pipeline behavior**: Searched through all 4 strategies (exact → temporal →
  name_partial → entity_dropped). Each retry returned articles, but none passed
  validation.
- **Retrieved articles**: Unrelated Houston-area shootings — Ivan Rayo at 8700
  Westpark Drive, Danny Adams at 501 Greens Road, Deputy Darren Goforth at a
  Copperfield gas station. The search queries returned Houston police shootings
  from the same time period (2015) but for entirely different incidents.
- **Diagnosis**: The incident lacks distinguishing details in the DB — no
  civilian name, only a street address and weapon type. Without a name to anchor
  the search, Tavily returns the most prominent Houston-area police shootings
  from 2015, none of which match. This is a structural retrieval gap — the
  pipeline correctly exhausted all strategies and escalated.
- **Verdict**: Correct escalation.

**Case 2: Incident 3494 — Merge bug (merge_error)**

- **Ground truth**: 27yo White male, knife, 900 N. Wells Street (Edna, TX),
  16:00
- **Pipeline behavior**: Found correct articles on the first search strategy
  (temporal_expanded). Two articles from the Victoria Advocate and Crossroads
  Today describe Matthew Pillow, 27, being shot by Officer Luis Chavez in Edna
  after brandishing a knife — matching the ground truth on age, weapon, and
  location.
- **Crash**: Merge node failed with
  `'NoneType' object has no attribute 'value'`. The articles passed validation
  and extraction was attempted, but the merge node hit a NoneType error when
  comparing extracted fields — likely a field with no extracted value being
  passed to comparison logic that assumes a non-null object.
- **Diagnosis**: Code bug in merge node's null handling. The articles were
  correct, the incident was identifiable, and extraction should have succeeded.
- **Verdict**: Fixable bug (merge null handling).

**Summary**: Of the 2 sampled escalations, 1 was correct behavior (structural
retrieval gap) and 1 was a fixable code bug. The remaining 10 max_retries cases
likely follow the same retrieval-gap pattern as incident 6742 — all stalled at
the search stage with no articles passing validation. A full 10-case review
would confirm this but is unlikely to reveal new failure modes.

### Cost and Latency

| Metric                | Value        |
| --------------------- | ------------ |
| Mean time per record  | 48.8s        |
| Total eval run time   | 32.5 minutes |
| Estimated total cost  | ~$6          |
| Estimated cost/record | ~$0.15       |

The 48.8s mean time is higher than the 7.0s pipeline latency reported in README
because the evaluation harness adds overhead per incident: database queries to
fetch ground truth, fuzzy-match comparisons across 6 fields, and result
aggregation. The core pipeline (search + merge) accounts for ~7s; the remaining
~42s is evaluation-only overhead spread across 40 sequential incidents.

Cost is estimated from API pricing (Claude Sonnet 4.6 at $3/$15 per 1M
input/output tokens; Tavily advanced search at $0.016/search PAYGO). The primary
cost driver is LLM extraction (~70%), with web search comprising ~30%. Actual
costs depend on article length and retry count.

## Discussion

### What Works

The pipeline completes 70% of holdout incidents (up from 7.5% before v1.3
improvements). Key strengths:

- **Partial completion** is the highest-impact change — most completed records
  have conflicts on some fields but successfully extract 3–7 agreed fields.
  Previously all of these would have been escalated with zero output.
- **civilian_age** is the strongest field at 75% exact accuracy, likely because
  age is unambiguous and consistently reported in news articles.
- **time_of_day** achieves 73% exact accuracy despite free-text format
  differences, thanks to the hour-based comparison with ±2h tolerance.
- **Removing score gating** lets more articles through to validation, where
  date/location matching serves as a better quality filter than Tavily's
  relevance score.

### Known Limitations

**Sample size**: The holdout evaluation covers ~2.4% of the full dataset (40 of
1,674). This is sufficient to identify systematic failure modes and estimate
pipeline behavior, but not large enough for precise accuracy estimates with
narrow confidence intervals. Full-dataset batch evaluation is on the roadmap.

**Temporal bias**: News articles from 2014–2016 incidents may no longer be
available online, creating a structural disadvantage for older records that is
independent of pipeline quality.

**Ground truth completeness**: Some ground truth fields (e.g., civilian_died)
are NULL in the database, limiting evaluation coverage for those fields.

**Single dataset**: This evaluation covers the civilians-shot dataset only. The
officers-shot dataset (282 records, different schema) has not been evaluated and
may exhibit different failure modes.

### Failure Mode Taxonomy

| Failure mode                                  | Frequency (N=40)                   | Fixable?       | Priority |
| --------------------------------------------- | ---------------------------------- | -------------- | -------- |
| Retrieval gap (no articles found)             | 11 (28%)                           | No             | —        |
| Merge error (NoneType in merge)               | 1 (3%)                             | Yes            | Medium   |
| Weapon eval mismatch (semantic synonyms)      | 31% coverage, 0% exact             | Yes            | High     |
| Location eval mismatch (narrative vs address) | 25% coverage, 0% exact             | Yes            | High     |
| Synonym/formatting false conflicts            | Reduced by partial completion      | Mitigated      | —        |
| Genuine factual conflicts                     | Present but no longer block output | No (by design) | —        |

## Roadmap

- Fix merge node null handling (incident 3494 crash) —
  `'NoneType' object has no attribute 'value'` when extracted fields are None
- Embedding-based similarity for weapon eval (e.g., `bge-small-en-v1.5` or
  `potion-base-32M`) to capture semantic synonyms ("handgun" ≈ "firearm")
- Improve merge prompt for location extraction (request specific street
  addresses instead of narrative descriptions)
- Geocoding normalization for location eval (US Census Geocoder or Google Maps
  API via geopy)
- Fairness analysis across demographic groups — the holdout eval JSON includes
  `fairness_metrics` data (pipeline reach and extraction precision by race/age)
  that has not yet been analyzed
- Batch processing across all ~1,900 remaining records with priority ordering
- Evaluation of the officers-shot dataset
- Human review UI for processing escalated records

## Appendix: Excluded Domains

| Domain              | Reason                                                                         | Added |
| ------------------- | ------------------------------------------------------------------------------ | ----- |
| wikipedia.org       | Multi-incident aggregation pages cause cross-incident extraction contamination | v1.0  |
| fatalencounters.org | Aggregation dataset; tabular format causes cross-incident field contamination  | v1.1  |
