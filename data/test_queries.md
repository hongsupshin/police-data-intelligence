# Test Cases for Police Data Intelligence Assistant

These test cases validate the **enrichment pipeline**, the core system capability.

---

## Test 1: Single Record Enrichment

**Scenario:** Enrich a specific incident with missing weapon information.

**Input:**
```python
record = {
    "incident_id": 142,
    "date_incident": "2016-02-08",
    "incident_city": "AUSTIN",
    "civilian_name_first": "DAVID",
    "civilian_name_last": "JOSEPH",
    "weapon_reported_by_media": None  # Missing - needs enrichment
}
```

**Expected Agent Behavior:**

1. **Planning Agent:**
   - Identifies missing field: `weapon_reported_by_media`
   - Constructs search query: `"David Joseph" Austin police shooting February 2016`

2. **Retrieval Agent:**
   - Calls Tavily API with constructed query
   - Returns 3-5 news articles

3. **Validation Agent:**
   - Anchor matching: date ±3 days, location match, name match
   - Extracts weapon information from articles
   - Assigns confidence score based on source agreement

4. **Synthesis Agent:**
   - Formats suggestion with evidence
   - Presents for human approval

**Expected Output:**
```text
ENRICHMENT RESULT
Record: David Joseph (Austin, Feb 8, 2016)

Found 4 relevant articles:
  - Austin American-Statesman (Feb 9, 2016)
  - KVUE News (Feb 8, 2016)
  - Texas Tribune (Feb 10, 2016)

Anchor Match Confidence: 95% (HIGH)
  ✓ Date: Feb 8, 2016 (exact match)
  ✓ Location: Austin, Texas (exact match)
  ✓ Name: David Joseph (exact match)

EXTRACTED FIELD:
  weapon_reported_by_media: "NONE (unarmed)"
  Confidence: 88% (HIGH - 3 sources agree)
  Evidence: "Joseph was naked and unarmed when shot..."
  Sources: Statesman, KVUE, Tribune

SUGGESTED UPDATE:
  weapon_reported_by_media: NULL → "NONE (unarmed)"

⚠️ HUMAN REVIEW REQUIRED
Approve this update? [Y/n]
```

**What This Tests:**
- Search query construction from record details
- News article retrieval via API
- Anchor matching (entity resolution)
- Information extraction from unstructured text
- Confidence scoring based on source agreement
- Human-in-the-loop output format

---

## Test 2: Batch Enrichment

**Scenario:** Process all records missing weapon information.

**Input:**
```json
{
  "field": "weapon_reported_by_media",
  "dataset": "incidents_civilians_shot",
  "limit": 10
}
```

**Expected Agent Behavior:**
- System internally fetches 10 records where weapon is NULL
- Queue records for enrichment
- Process each through the enrichment pipeline
- Track progress and success/failure counts
- Log costs per record

**Expected Output:**
```text
BATCH ENRICHMENT PROGRESS
Target: 10 records missing weapon_reported_by_media

[████████░░] 8/10 complete

Results:
  ✓ 6 enriched (HIGH confidence)
  ⚠ 2 enriched (MEDIUM confidence - needs review)
  ✗ 2 failed (no matching articles found)

Cost: $0.12 (avg $0.012/record)
Time: 45 seconds

PENDING HUMAN REVIEW:
  - Record #142: weapon = "NONE (unarmed)" [MEDIUM]
  - Record #287: weapon = "knife" [MEDIUM]
```

**What This Tests:**
- Batch processing workflow
- Progress tracking
- Cost monitoring
- Handling of failed enrichments
- Queue management

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
