# Police Data Intelligence

**An agentic AI system for enriching missing data in police shooting databases**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/hongsupshin/police-data-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/hongsupshin/police-data-intelligence/actions/workflows/ci.yml)

## Overview

This project builds an agentic pipeline that **automatically enriches missing
data** in two Texas Justice Initiative (TJI) databases through intelligent web
search and extraction. The system's core purpose is data augmentation, not
analysis.

**Datasets**:

- **Civilians-Shot** (1,674 records): Police shooting civilians — 57% missing
  weapon info, 22.5% missing names
- **Officers-Shot** (282 records): Civilians shooting police — 40% missing
  officer names
- **Total**: 1,956 records needing enrichment

**The Problem**: TJI volunteers spend 15–30 minutes per record manually
searching news articles and extracting details.

**The Solution**: An agentic AI system that automates the enrichment workflow
while keeping humans in the loop, reducing volunteer time by 75%.

## Architecture

The system uses **7 nodes** orchestrated by a Coordinator in LangGraph:

```
Extract → Search → Validate → Merge → Complete
            ↑         |         |
            └─ Coordinator (retry/escalate) ─┘
```

Each node accepts and returns `EnrichmentState` (defined in
`src/agents/state.py`). The Coordinator reads `current_stage` to decide routing;
nodes update state fields, graph edges handle transitions.

### Pipeline Nodes

| Node            | Type          | Purpose                                                                    |
| --------------- | ------------- | -------------------------------------------------------------------------- |
| **Extract**     | Deterministic | Reads incident record from PostgreSQL, populates state fields              |
| **Search**      | Deterministic | Constructs query from incident fields, calls Tavily API for news articles  |
| **Validate**    | Rule-based    | Checks date proximity (±3 days), location match, and optional name match   |
| **Merge**       | LLM-powered   | Extracts structured fields from articles, checks cross-article consistency |
| **Coordinator** | Rule-based    | Gates after each stage — decides retry, proceed, or escalate               |
| **Complete**    | Terminal      | Writes enrichment results to JSON                                          |
| **Escalate**    | Terminal      | Writes escalation report to JSON for human review                          |

### Search Strategies

The Coordinator implements an escalating retry strategy:

| Retry | Strategy            | Description                                 |
| ----- | ------------------- | ------------------------------------------- |
| 0     | `exact_match`       | All fields, exact date                      |
| 1     | `temporal_expanded` | Month + year format, keep names             |
| 2     | `entity_dropped`    | Drop both names, keep location + date range |
| 3     | Escalate            | Flag for human review                       |

### Escalation Triggers

The Coordinator routes to human review when:

- Max retries reached without sufficient validated articles
- No articles pass validation after all strategies
- Merge detects conflicting information across sources
- Merge detects conflict between articles and database reference values
- Merge encounters an error

### Validation Logic

Articles pass validation using three-tier logic:

| Condition                    | Criteria        | Rationale                    |
| ---------------------------- | --------------- | ---------------------------- |
| Has `published_date`         | date + location | Standard check               |
| No date, has `civilian_name` | location + name | Compensates for missing date |
| No date, no name             | location only   | Last resort fallback         |

This prevents false positives from articles about different incidents that
happen to match on location alone, while still handling Tavily results that lack
parsed dates.

### Merge Logic

The merge node only processes **validated articles** (those that passed
validation), filtering out unrelated articles before extraction.

For each field extracted from validated articles:

- **Articles agree** → add to `extracted_fields` with confidence level
- **Articles disagree** → add `FieldConflict` to `conflicting_fields`, escalate
- **All articles return null** → skip (no data, not a conflict)
- **Articles agree but conflict with database** → add to both lists, escalate

Each `FieldConflict` captures the field name, conflict type (`articles_disagree`
or `reference_mismatch`), the conflicting values with source URLs, and the
database reference value when applicable.

The database is treated as immutable ground truth (official government data).

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL with TJI data loaded
- Anthropic API key
- Tavily API key

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package (enables the `enrich` CLI command)
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### Run

```bash
# Enrich a single incident
enrich <incident_id> <dataset_type>

# Examples
enrich 10 civilians_shot
enrich 42 officers_shot

# Or without installing the package
python -m src.run 10 civilians_shot
```

Results are written to `output/enrichment/` as pretty-printed JSON files:

- `civilians_shot_10_complete.json` — successful enrichment
- `civilians_shot_10_escalate.json` — flagged for human review

### Example Output

<details>
<summary>Successful enrichment (civilians_shot_792_complete.json)</summary>

```json
{
  "incident_id": "792",
  "dataset_type": "civilians_shot",
  "extracted_fields": [
    {
      "field_name": "weapon",
      "value": "Knife (possessed by civilian Donnavan Hart); firearms (used by the off-duty HPD commander and the Precinct 4 deputy)",
      "confidence": "medium",
      "sources": [
        "https://www.click2houston.com/news/local/2020/02/19/deputy-off-duty-hpd-commander-shoot-suspect-near-popeyes-restaurant-in-spring-officials-say/"
      ],
      "source_quotes": [
        "An off-duty HPD commander was working out at the gym when he saw the suspect, armed with a knife, run in and 'cause a disturbance'",
        "Deputies said Hart lunged at them with the knife in his hand. The commander and the deputy both shot the suspect multiple times."
      ],
      "extraction_method": "llm",
      "llm_reasoning": "The article clearly states the civilian was armed with a knife, and the officers responded by shooting the suspect multiple times, indicating use of firearms."
    },
    {
      "field_name": "time_of_day",
      "value": "5:16 p.m.",
      "confidence": "medium",
      "sources": [
        "https://www.click2houston.com/news/local/2020/02/19/deputy-off-duty-hpd-commander-shoot-suspect-near-popeyes-restaurant-in-spring-officials-say/"
      ],
      "source_quotes": [
        "At 5:16 p.m., Hermann wrote on Facebook that deputies responded to a report of a suspicious person at the Popeye's restaurant"
      ],
      "extraction_method": "llm",
      "llm_reasoning": "The article explicitly states the time as 5:16 p.m."
    },
    {
      "field_name": "circumstance",
      "value": "Donnavan Hart was reported as a suspicious person at a Popeyes restaurant. He entered a nearby LA Fitness gym armed with a knife and caused a disturbance. An off-duty HPD commander who was at the gym joined a Precinct 4 deputy in pursuing Hart, who then lunged at the officers with the knife, prompting both to open fire.",
      "confidence": "medium",
      "sources": [
        "https://www.click2houston.com/news/local/2020/02/19/deputy-off-duty-hpd-commander-shoot-suspect-near-popeyes-restaurant-in-spring-officials-say/"
      ],
      "extraction_method": "llm"
    }
    // ... 4 more extracted fields (officer_name, civilian_name, location_detail, outcome)
  ],
  "validation_results": [
    {
      "article": {
        "url": "https://www.click2houston.com/news/local/2020/02/19/...",
        "title": "Authorities identify the man shot by deputy, off-duty HPD ..."
      },
      "date_match": false,
      "location_match": true,
      "victim_name_match": true,
      "passed": true
    }
    // ... 4 more validation results (4 failed, 1 passed)
  ],
  "search_strategy": "entity_dropped",
  "retry_count": 2,
  "outcome_summary": "Enriched 7 fields for incident 792 (civilians_shot)"
}
```

</details>

<details>
<summary>Escalated for human review (civilians_shot_10_escalate.json)</summary>

```json
{
  "incident_id": "10",
  "dataset_type": "civilians_shot",
  "escalation_reason": "conflict",
  "error_message": null,
  "current_stage": "merge",
  "search_strategy": "exact_match",
  "retry_count": 0,
  "retrieved_articles": [
    {
      "url": "https://www.nbcdfw.com/news/local/officer-shoots-armed-man-at-dallas-apartment-complex-police/143469/",
      "title": "Officers Shoot Armed Man at Dallas Apartment Complex: Police"
    },
    {
      "url": "https://www.cbsnews.com/texas/news/police-kill-suspect-at-dallas-apartment-complex/",
      "title": "Police Kill Suspect At Dallas Apartment Complex - CBS News"
    },
    {
      "url": "https://www.dallasnews.com/news/crime/2015/09/22/man-killed-by-police-told-girlfriend-he-didnt-want-to-live/",
      "title": "Man killed by police told girlfriend he didn't want to live"
    }
    // ... 2 more articles
  ],
  "extracted_fields": [
    {
      "field_name": "officer_name",
      "value": "Rob Sherwin",
      "confidence": "high",
      "sources": [
        "https://www.nbcdfw.com/news/local/officer-shoots-armed-man-at-dallas-apartment-complex-police/143469/"
      ],
      "extraction_method": "llm",
      "llm_reasoning": "Rob Sherwin is mentioned as the Deputy Chief who held the news conference, not as one of the officers who fired."
    }
  ],
  "conflicting_fields": [
    {
      "field_name": "civilian_name",
      "conflict_type": "articles_disagree",
      "values": [
        "Gerardo Ramirez",
        "Gerardo Ramirez (plus unrelated names from compilation sources)"
      ],
      "sources": [
        [
          "https://www.nbcdfw.com/news/local/...",
          "https://www.cbsnews.com/texas/..."
        ],
        ["https://www.dallasnews.com/news/crime/2015/09/22/..."]
      ],
      "reference_value": null
    }
    // ... 7 more conflicting fields (age, race, weapon, location, time, outcome, circumstance)
  ],
  "outcome_summary": "Escalated incident 10: conflict after 0 retries"
}
```

</details>

### Evaluation

The holdout evaluation measures pipeline accuracy by comparing extracted fields
against ground truth values already in the database (age, race, weapon,
location, time, outcome). These fields exist in the DB but are never seen by the
pipeline during enrichment, creating a natural holdout.

```bash
python -m src.eval.run_eval civilians_shot --limit 10 --min-fields 2
```

Reports are saved to `output/eval/` as JSON.

### Configuration

Environment variables (see `.env.example`):

| Variable                               | Default             | Description                           |
| -------------------------------------- | ------------------- | ------------------------------------- |
| `ANTHROPIC_API_KEY`                    | (required)          | Anthropic API key                     |
| `ANTHROPIC_MODEL`                      | `claude-sonnet-4-6` | Model for LLM-powered nodes           |
| `TAVILY_API_KEY`                       | (required)          | Tavily API key for news search        |
| `LOG_LEVEL`                            | `INFO`              | Logging level                         |
| `ENRICHMENT_OUTPUT_DIR`                | `output/enrichment` | Output directory for JSON results     |
| `ENRICHMENT_MAX_SEARCH_RESULTS`        | `5`                 | Max articles per search               |
| `ENRICHMENT_SEARCH_DEPTH`              | `advanced`          | Tavily search depth                   |
| `ENRICHMENT_RELEVANCE_SCORE_THRESHOLD` | `0.5`               | Min avg relevance to proceed          |
| `ENRICHMENT_FUZZY_MATCH_THRESHOLD`     | `80`                | Min rapidfuzz score for name matching |
| `ENRICHMENT_DATE_PROXIMITY_DAYS`       | `3`                 | Max days between article and incident |

PostgreSQL connection variables (`DB_HOST`, `DB_PORT`, etc.) are configured in
`.env.example` and used by the ETL pipeline (`data/`).

## Development

### Project Structure

```text
police-data-intelligence/
├── src/
│   ├── agents/
│   │   ├── state.py             # EnrichmentState, Article, FieldExtraction models
│   │   ├── graph.py             # LangGraph wiring, complete/escalate terminal nodes
│   │   ├── coordinate_node.py   # Coordinator gates (search/validate/merge checks)
│   │   └── extract_node.py      # Extract node (PostgreSQL → state)
│   ├── retrieval/
│   │   └── search_node.py       # Search node (Tavily API)
│   ├── validation/
│   │   └── validate_node.py     # Validate node (date/location/name matching)
│   ├── merge/
│   │   └── merge_node.py        # Merge node (LLM extraction + consistency)
│   ├── database/
│   │   └── connection.py        # PostgreSQL connection
│   ├── eval/
│   │   ├── holdout.py           # Holdout evaluation (compare vs DB ground truth)
│   │   └── run_eval.py          # Eval CLI entrypoint
│   ├── config.py                # Settings (pydantic-settings, from env vars)
│   └── run.py                   # CLI entrypoint
├── data/
│   └── etl/                     # ETL pipeline (CSV → PostgreSQL), separate from agents
├── tests/
│   ├── test_extract_node.py
│   ├── test_search_node.py
│   ├── test_validate_node.py
│   ├── test_merge_node.py
│   ├── test_coordinate_node.py
│   ├── test_graph.py            # Graph wiring + terminal node tests
│   ├── test_run.py
│   ├── test_holdout.py
│   └── ...                      # ETL tests (cleaners, loaders, schemas)
├── output/
│   ├── enrichment/              # Pipeline JSON output
│   └── eval/                    # Holdout evaluation reports
├── .env.example
└── requirements.txt
```

### Commands

```bash
# Lint
ruff check src/ tests/

# Test (unit only — no PostgreSQL needed)
pytest tests/ -v -m "not integration"

# Test single module
pytest tests/test_validate_node.py -v

# Integration tests (requires PostgreSQL)
pytest tests/ -v -m "integration"
```

### Testing Patterns

- Unit tests mock external dependencies (Tavily, PostgreSQL, LLM)
- Integration tests use `@pytest.mark.integration`
- Mock LLM via `MagicMock` + dependency injection, not `@patch`
- Use `model_copy()` when fixtures are mutated by functions under test

## Key State Fields

```python
class EnrichmentState:
    # Identity
    incident_id: str
    dataset_type: DatasetType          # "civilians_shot" | "officers_shot"

    # Incident data (from Extract)
    location: str | None
    incident_date: date | None
    officer_name: str | None
    civilian_name: str | None
    severity: str | None

    # Pipeline control
    current_stage: PipelineStage       # Which node just ran
    next_stage: PipelineStage | None   # Where to go next (set by Coordinator)
    next_strategy: SearchStrategyType  # Search strategy for retry
    retry_count: int
    escalation_reason: str | None
    requires_human_review: bool

    # Search results
    retrieved_articles: list[Article]
    search_attempts: list[SearchAttempt]

    # Validation
    validation_results: list[ValidationResult]

    # Merge outputs
    extracted_fields: list[FieldExtraction]
    conflicting_fields: list[FieldConflict] | None
```

## Source Filtering

### Search-level (Tavily `exclude_domains`)

| Domain                | Reason                                          |
| --------------------- | ----------------------------------------------- |
| `wikipedia.org`       | Aggregation pages confuse LLM extraction        |
| `fatalencounters.org` | Compilation dataset, not incident-specific news |

### Validation-level (URL pattern filtering in `validate_node.py`)

| Pattern | Reason                                                 |
| ------- | ------------------------------------------------------ |
| `.pdf`  | Compilation documents (e.g., statewide victim reports) |
| `.csv`  | Raw datasets with many unrelated records               |

## Performance

Measured across 23 incidents (warm connections, claude-sonnet-4-6):

| Metric                          | Range            | Mean     |
| ------------------------------- | ---------------- | -------- |
| **Total per incident**          | **2.3s – 13.5s** | **7.0s** |
| Search (Tavily API)             | 2.2s – 13.4s     | 6.5s     |
| Merge (LLM call)                | 0.13s – 0.82s    | 0.4s     |
| Extract + Validate + Coordinate | <0.1s            | ~0s      |

The primary driver of variance is retry count — each retry adds a Tavily search
call (~3–5s). Tavily search accounts for ~93% of total runtime; the LLM merge
call is ~0.4s; all deterministic nodes (extract, validate, coordinate) are
effectively instant.

| Retries | Searches | Typical Time |
| ------- | -------- | ------------ |
| 0       | 1        | 2–5s         |
| 1       | 2        | 5–9s         |
| 2       | 3        | 9–13s        |

**Projected at scale** (1,956 records, sequential): ~3.5 hours. With async
parallelism (e.g., 10 concurrent workers): ~20 minutes.

## Responsible AI

This system operates in a sensitive domain (police accountability). Key design
principles:

- **Human-in-the-loop**: System never auto-updates the database; humans approve
  all changes
- **Transparency**: Shows article excerpts, confidence scores, and conflict
  details
- **Traceability**: Links suggestions to source articles with verbatim quotes
- **Accuracy over automation**: Conservative thresholds, escalation on conflicts
- **Immutability**: Never overwrites official government data without human
  approval

## Roadmap

**Built:**

- ETL pipeline (CSV → PostgreSQL)
- 7-node LangGraph pipeline with conditional routing and retry strategies
- CLI entrypoint for single-incident enrichment
- Holdout evaluation framework (precision, coverage against DB ground truth)

**Next:**

- Batch processing across all records
- Human review UI

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgment

The author appreciates
**[Texas Justice Initiative](https://texasjusticeinitiative.org/)** (TJI) for
collecting, analyzing, and publishing criminal justice data in Texas. TJI
maintains publicly available databases on officer-involved shootings and deaths
in law enforcement custody, making this data accessible to reporters,
researchers, policymakers, and the public. The author contributed to TJI's
[Officer-Involved Shootings in Texas](https://texasjusticeinitiative.org/publications/officer-involved-shootings-in-texas)
report (covering 2016–2019). This project extends that work using TJI's updated
datasets (2014–2024, 1,956 records) to automate the labor-intensive process of
enriching incident records with information from news sources.
