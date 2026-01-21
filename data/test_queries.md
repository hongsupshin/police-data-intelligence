# Test Cases for Police Data Intelligence Assistant

These test cases validate the **enrichment pipeline**, the core system
capability.

---

## Media Feature Set

Fields extracted from every article (not just missing fields):

| Field             | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `officer_name`    | Name(s), rank, years of service                         |
| `civilian_name`   | Full name                                               |
| `civilian_age`    | Age at time of incident                                 |
| `civilian_race`   | Race/ethnicity                                          |
| `weapon`          | Weapon type or "unarmed"                                |
| `location_detail` | Address, neighborhood                                   |
| `time_of_day`     | When incident occurred                                  |
| `outcome`         | Fatal/non-fatal, injuries                               |
| `circumstance`    | Traffic stop, warrant service, mental health call, etc. |

Entity detection (NER) supplements this set to capture additional names,
organizations, or locations.

---

## Test 1: Single Record Enrichment

**Scenario:** Extract all media features for a specific incident.

**Input:**

```python
record = {
    "incident_id": 142,
    "date_incident": "2016-02-08",
    "incident_city": "AUSTIN",
    "civilian_name_first": "DAVID",
    "civilian_name_last": "JOSEPH"
}
```

**Expected Agent Behavior:**

1. **Planning Agent:**
   - Constructs search query:
     `"David Joseph" Austin police shooting February 2016`

2. **Retrieval Agent:**
   - Calls Tavily API with constructed query
   - Returns 3-5 news articles

3. **Validation Agent:**
   - Anchor matching: date ±3 days, location match, name match
   - Extracts all fields in media feature set
   - Runs entity detection for additional information
   - Assigns per-field confidence based on source agreement

4. **Synthesis Agent:**
   - Formats extraction results with evidence
   - Flags items for human review

**Expected Output:**

```text
ENRICHMENT RESULT
Record: David Joseph (Austin, Feb 8, 2016)

Found 4 relevant articles:
  - Austin American-Statesman (Feb 9, 2016)
  - KVUE News (Feb 8, 2016)
  - Texas Tribune (Feb 10, 2016)

Anchor Match: 95% (HIGH)
  ✓ Date: Feb 8, 2016 (exact match)
  ✓ Location: Austin, Texas (exact match)
  ✓ Name: David Joseph (exact match)

EXTRACTED FIELDS:
  weapon: "NONE (unarmed)" [HIGH - 3 sources]
  civilian_age: 17 [HIGH - 3 sources]
  officer_name: "Geoffrey Freeman" [MEDIUM - 1 source]
  circumstance: "mental health call" [HIGH - 2 sources]
  outcome: "fatal" [HIGH - 3 sources]

ENTITIES DETECTED:
  ORG: Austin Police Department
  PERSON: Art Acevedo (police chief)

⚠️ HUMAN REVIEW REQUIRED (1 field MEDIUM confidence)
```

**What This Tests:**

- Search query construction from record details
- News article retrieval via API
- Anchor matching (entity resolution)
- Full media feature set extraction
- Entity detection for supplementary information
- Per-field confidence scoring

---

## Test 2: Batch Enrichment

**Scenario:** Process all records, prioritized by location and year.

**Priority Order:**

1. Houston (largest city) → Dallas → Austin → Other cities
2. Recent years first (2024 → 2023 → 2022 → ...)

**Input:**

```json
{
  "dataset": "civilians_shot",
  "priority": "location_year",
  "limit": 100
}
```

**Expected Agent Behavior:**

- Fetch records ordered by priority (Houston 2024 first)
- Process each through the enrichment pipeline
- Extract full media feature set per record
- Track progress, costs, and generate output file

**Expected Output (Console):**

```text
BATCH ENRICHMENT PROGRESS
Processing: civilians_shot (Houston 2024 → ...)

[████████░░] 80/100 complete

Results:
  ✓ 60 enriched (HIGH confidence)
  ⚠ 15 enriched (MEDIUM confidence - needs review)
  ✗ 5 failed (no matching articles found)

Cost: $1.20 (avg $0.012/record)

Output: enrichment_results_2024-01-19.csv
```

**Expected Output File (CSV/Markdown):**

| record_id | city    | date       | field        | value      | confidence | reasoning                   | sources |
| --------- | ------- | ---------- | ------------ | ---------- | ---------- | --------------------------- | ------- |
| 142       | Austin  | 2016-02-08 | weapon       | unarmed    | HIGH       | 3 sources agree             | [urls]  |
| 142       | Austin  | 2016-02-08 | officer_name | G. Freeman | MEDIUM     | Single source, direct quote | [url]   |
| 287       | Houston | 2018-03-15 | weapon       | knife      | MEDIUM     | 2 sources, unclear context  | [urls]  |

**What This Tests:**

- Location/year priority ordering
- Full feature set extraction per record
- Progress and cost tracking
- Structured output file generation
- Reasoning summaries for human review

---

## Test 3: Conflict Resolution

**Scenario:** Sources disagree on extracted information.

**Input:** Record with conflicting officer name in news articles.

**Expected Agent Behavior:**

1. **Retrieval Agent** returns articles with conflicting info:
   - Source A (Houston Chronicle): "Officer James Rodriguez"
   - Source B (KHOU): "Sgt. J. Ramirez"

2. **Validation Agent:**
   - Detects conflict
   - Generates reasoning about which source is more authoritative
   - Flags for human review

**Expected Output:**

```text
⚠️ CONFLICT DETECTED

Field: officer_name
Record: Houston shooting, March 15, 2018

Source A (Houston Chronicle):
  Value: "Officer James Rodriguez"
  Context: "...Austin Police Officer Geoffrey Freeman, a 10-year veteran,
            said in a direct quote from police spokesperson..."

Source B (KHOU):
  Value: "Sgt. J. Ramirez"
  Context: "...Sgt. J. Ramirez was involved in the incident..."

LLM Reasoning:
  Chronicle appears more authoritative - direct quote from spokesperson
  vs KHOU's unattributed mention. However, names are significantly
  different (Rodriguez vs Ramirez) - may be two different officers.

Suggested Resolution: FLAG FOR MANUAL REVIEW
  - Possible: Multiple officers involved
  - Possible: Reporting error in one source

⚠️ REQUIRES HUMAN DECISION
```

**What This Tests:**

- Conflict detection across sources
- LLM reasoning for ambiguous cases
- Appropriate escalation to human review

---

## Test 4: No Results Found

**Scenario:** Web search returns no matching articles.

**Input:** Obscure incident with no news coverage.

**Expected Output:**

```text
ENRICHMENT RESULT
Record: [Name Redacted] (Small Town, Jan 3, 2015)

Search performed: "[Name] Small Town police shooting January 2015"

No matching articles found.

Possible reasons:
  - Incident not covered by indexed news sources
  - Name spelling variation
  - Date uncertainty

Suggested actions:
  - Try broader search terms
  - Check local news archives manually
  - Mark as "enrichment attempted, no results"

Record status: UNCHANGED
```

**What This Tests:**

- Graceful handling of no results
- Appropriate status reporting
- Suggestions for manual follow-up

---

## Success Criteria

Each test should demonstrate:

1. **Correct Pipeline Flow**: Planning → Retrieval → Validation → Synthesis
2. **Accurate Extraction**: Correct information pulled from articles
3. **Confidence Scoring**: Reasonable scores based on source agreement
4. **Human-Friendly Output**: Clear presentation with evidence
5. **Cost Tracking**: Per-record and batch cost logging
6. **Graceful Failures**: Appropriate handling of edge cases

---

## Running the Tests

```bash
# 1. Load the database
cd data
python load_data.py

# 2. Test 1: Single Record Enrichment
curl -X POST http://localhost:8000/enrich \
  -H "Content-Type: application/json" \
  -d '{"incident_id": 142}'

# 3. Test 2: Batch Enrichment
curl -X POST http://localhost:8000/enrich/batch \
  -H "Content-Type: application/json" \
  -d '{
    "field": "weapon_reported_by_media",
    "dataset": "incidents_civilians_shot",
    "limit": 10
  }'

# 4. Monitor batch progress
curl http://localhost:8000/enrich/batch/status/{batch_id}

# 5. Review pending suggestions
curl http://localhost:8000/enrich/pending
```
