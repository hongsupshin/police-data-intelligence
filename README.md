# Police Data Intelligence

**An agentic AI system for enriching missing data in police shooting databases**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/hongsupshin/police-data-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/hongsupshin/police-data-intelligence/actions/workflows/ci.yml)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Pipeline Nodes](#pipeline-nodes)
  - [Search Strategies](#search-strategies)
  - [Escalation Triggers](#escalation-triggers)
  - [Validation Logic](#validation-logic)
  - [Synthesize Logic](#synthesize-logic)
  - [Agentic Precision/Safety Layer](#agentic-precisionsafety-layer-llm)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Run](#run)
  - [Example Output](#example-output)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
- [Development](#development)
  - [Project Structure](#project-structure)
  - [Commands](#commands)
  - [Testing Patterns](#testing-patterns)
- [Performance](#performance)
  - [Cost](#cost)
- [Responsible AI](#responsible-ai)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgment](#acknowledgment)

## Overview

This project builds an agentic pipeline that **automatically enriches missing
data** in two Texas Justice Initiative (TJI) databases through intelligent web
search and extraction. The system's core purpose is data augmentation, not
analysis.

**Datasets**:

- **Civilians-Shot** (1,674 records): Police shooting civilians — 57% missing
  weapon info, 22.5% missing names
- **Officers-Shot** (282 records): Civilians shooting police — 39% missing
  officer names
- **Total**: 1,956 records needing enrichment

**The Problem**: TJI volunteers spend 15–30 minutes per record manually
searching news articles and extracting details.

**The Solution**: An agentic AI system that automates the enrichment workflow
while keeping humans in the loop, aiming to substantially reduce the volunteer
time each record takes.

## Architecture

The system uses **7 nodes** orchestrated by a Coordinator in LangGraph:

```mermaid
flowchart TD
    Start([Start]) --> Load

    Load[Load<br/><i>DB → state fields</i>]
    Search[Search<br/><i>Tavily API</i>]
    Validate[Validate<br/><i>date + location + name</i>]
    Synthesize[Synthesize<br/><i>LLM extraction</i>]
    Coord{Coordinator}
    Complete([Complete<br/><i>write JSON</i>])
    Escalate([Escalate<br/><i>human review</i>])

    Load --> Coord
    Coord -- "fields OK" --> Search
    Search --> Coord
    Coord -- "results > 0" --> Validate
    Coord -- "retry: next strategy" --> Search
    Validate --> Coord
    Coord -- "articles valid" --> Synthesize
    Synthesize --> Coord
    Coord -- "fields extracted" --> Complete
    Coord -- "error / max retries / zero extractions" --> Escalate
```

Each node accepts and returns `EnrichmentState` (defined in
`src/agents/state.py`). The Coordinator reads `current_stage` to decide routing;
nodes update state fields, graph edges handle transitions.

### Pipeline Nodes

| Node            | Type          | Purpose                                                                                                                 |
| --------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Load**        | Deterministic | Reads incident record from PostgreSQL, populates state fields                                                           |
| **Search**      | Deterministic | Constructs query from incident fields, calls Tavily API for news articles (scoped to a date window around the incident) |
| **Validate**    | Rule-based    | Checks date proximity (±5 days), location match, and optional name match                                                |
| **Synthesize**  | LLM + agentic | Extracts structured fields, checks cross-article consistency, runs the agentic precision/safety layer (relevance, race, conflict) |
| **Coordinator** | Rule-based    | Gates after each stage — decides retry, proceed, or escalate                                                            |
| **Complete**    | Terminal      | Writes enrichment results to JSON                                                                                       |
| **Escalate**    | Terminal      | Writes escalation report to JSON for human review                                                                       |

### Search Strategies

The Coordinator implements an escalating retry strategy:

| Retry | Strategy            | Description                                        |
| ----- | ------------------- | -------------------------------------------------- |
| 0     | `exact_match`       | All fields, exact date                             |
| 1     | `temporal_expanded` | Month + year format, keep both names               |
| 2     | `name_partial`      | Drop officer name, keep civilian name + month-year |
| 3     | Escalate            | Flag for human review                              |

### Escalation Triggers

The Coordinator routes to human review when:

- Max retries reached without sufficient validated articles
- No articles pass validation after all strategies
- Synthesize detects conflicts **and** zero agreed fields if some fields agree
  while others conflict, the pipeline completes with the agreed fields and flags
  `requires_human_review = True` for the conflicts
- Synthesize encounters an error
- The relevance judge vetoes the sources as not about this incident →
  `IRRELEVANT_SOURCES` (both datasets; on by default,
  `ENRICHMENT_ENABLE_RELEVANCE_GATE`)

### Validation Logic

Articles pass validation using three-tier logic:

| Condition                    | Criteria        | Rationale                    |
| ---------------------------- | --------------- | ---------------------------- |
| Has `published_date`         | date + location | Standard check               |
| No date, has `civilian_name` | location + name | Compensates for missing date |
| No date, no name             | location only   | Last resort fallback         |

This prevents false positives from articles about different incidents that
happen to match on location alone, while still handling Tavily results that lack
parsed dates. Aggregation sites (e.g., Wikipedia, fatalencounters.org) and
compilation documents (.pdf, .csv) are excluded at search or validation level —
see [EVALUATION.md — Appendix](EVALUATION.md#appendix-excluded-domains).

### Synthesize Logic

The synthesize node only processes **validated articles** (those that passed
validation), filtering out unrelated articles before extraction. For
`civilians_shot`, extraction is **anchored to the record's victim** (name, age,
gender): when a source describes multiple people (a second victim, a bystander,
the officer), the prompt extracts fields only for the target civilian rather than
blending everyone.

For each field extracted from validated articles:

- **Articles agree** → add to `extracted_fields` with confidence level
- **Articles disagree** → add `FieldConflict` to `conflicting_fields`
- **All articles return null** → skip (no data, not a conflict)
- **Articles agree but conflict with database** → add to both lists

After synthesize, the Coordinator applies partial completion logic: if any
fields were successfully extracted (`extracted_fields` non-empty), route to
COMPLETE — even if conflicts exist on other fields. Only escalate on conflict
when zero fields were extracted. Partial completions set
`requires_human_review = True` so conflicts are still surfaced for review.

Before comparing values, the synthesize node normalizes names, race terms, and
weapon categories to reduce spurious conflicts (see
[EVALUATION.md — Fix 2](EVALUATION.md#fix-2-synthesize-normalization)). Two
further deterministic steps reduce conflicts without an LLM: a **consensus
resolver** (`src/field_normalizers.py`) commits an `outcome` or `time_of_day`
value when ≥2 articles agree on the same canonical form (instead of dropping the
field), and a **race taxonomy** (`src/race_taxonomy.py`) maps race terms to the
TJI buckets for matching while preserving the raw value and flagging any
divergence for review.

Each `FieldConflict` captures the field name, conflict type (`articles_disagree`
or `reference_mismatch`), the conflicting values with source URLs, and the
database reference value when applicable.

The database is treated as immutable ground truth (official government data).

### Agentic Precision/Safety Layer (LLM)

After deterministic extraction, three **agentic** LLM checks reason about what
rule-based logic cannot. Each is granted authority **calibrated to the stakes** of
its decision — block, null, or advise — and each is deliberately bounded: a single
structured-output call (no loops, no tools), fail-open (an error logs and the
pipeline proceeds as if it had not run), and read-only with respect to the
database. The worst case is an extra human review, never a wrong record.

| Agent                | Scope           | Question                                            | Authority                                                |
| -------------------- | --------------- | --------------------------------------------------- | -------------------------------------------------------- |
| **Relevance judge**  | both datasets   | "Do the articles report _this_ incident?"           | **Block** → veto completion, escalate `IRRELEVANT_SOURCES` |
| **Race verifier**    | civilians\_shot | "Is the extracted race _explicitly stated_?"        | **Null** → drop an unsupported `civilian_race`           |
| **Conflict annotator** | both datasets | "_Why_ do the sources disagree?"                    | **Advise** → write a reviewer triage note (never commits) |

**Relevance judge** (`ENRICHMENT_ENABLE_RELEVANCE_GATE`, on by default). After
extraction, an LLM reads the validated articles with the known incident anchors
(victim name, date, city, outcome — officer-anchored for `officers_shot`,
civilian-anchored for `civilians_shot`) and decides whether at least one article
actually reports _this_ incident. If none do, it vetoes the completion
(`relevance_vetoed`) so the Coordinator escalates (`IRRELEVANT_SOURCES`) instead
of committing fields from the wrong source.

It catches the "right structure, wrong incident" failure rule-based validation
cannot. Incident 75 is illustrative: for a March 6, 2018 San Antonio officer
shooting, the pipeline used an article — _"Capital murder trial of man accused of
killing SAPD officer during 2013 chase begins"_ — that passed the rule-based
checks (San Antonio, an SAPD officer, published on the incident date) but reports
a _2013_ case: a different officer, a different outcome. The judge, reading the
text, caught the mismatch the rules could not. The same gate catches civilian
famous-name collisions (a record sharing a name with a high-profile case).

Why an LLM and not a stricter rule? The discriminator is semantic, and the ways
an article can be the _wrong_ incident are open-ended (wrong year, coincidental
publish date, same-name different-incident, an article that never names the
victim) — you can't enumerate them up front. Rather than grow a brittle rule set,
the judge answers the general "is this the same event?" question.

**Race verifier** (`ENRICHMENT_ENABLE_RACE_VERIFICATION`, on by default;
`civilians_shot`). A faithfulness filter: it re-reads the sources and nulls a
`civilian_race` the article does not _explicitly state for this victim_ — never
inferring race from a name, neighborhood, or photo. It improves correctness
(removes unfaithful values), not coverage.

**Conflict annotator** (`ENRICHMENT_ENABLE_CONFLICT_ANNOTATION`, on by default;
cheap model). When deep conflicts (name/age/weapon/location/race) survive to human
review, a Haiku LLM writes a short advisory triage note explaining _why_ the
sources differ (a different person, different granularity, a genuine
contradiction, an unsupported value). It is **advisory only** — it never commits
or overwrites a field — so it can only aid the human-review default, never corrupt
data.

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
      "value": "Knife (possessed by civilian ...)",
      "confidence": "medium",
      "sources": ["https://www.click2houston.com/news/local/2020/02/19/..."],
      "extraction_method": "llm"
    }
    // ... 6 more fields (time_of_day, circumstance, officer_name, civilian_name, location_detail, outcome)
  ],
  "validation_results": [
    {
      "article": { "url": "...", "title": "Authorities identify the man ..." },
      "date_match": false,
      "location_match": true,
      "victim_name_match": true,
      "passed": true
    }
    // ... 4 more (4 failed, 1 passed)
  ],
  "search_strategy": "name_partial",
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
  "current_stage": "synthesize",
  "search_strategy": "exact_match",
  "retry_count": 0,
  "retrieved_articles": [
    {
      "url": "https://www.nbcdfw.com/...",
      "title": "Officers Shoot Armed Man ..."
    },
    { "url": "https://www.cbsnews.com/...", "title": "Police Kill Suspect ..." }
    // ... 3 more articles
  ],
  "validation_results": [
    {
      "article": {
        "url": "https://www.nbcdfw.com/...",
        "title": "Officers Shoot Armed Man ..."
      },
      "date_match": true,
      "location_match": true,
      "victim_name_match": false,
      "passed": false
    }
    // ... 4 more
  ],
  "validation_failure_summary": {
    "total": 5,
    "passed": 0,
    "excluded": 1,
    "date_fail": 2,
    "location_fail": 3,
    "name_fail": 1
  },
  "extracted_fields": [
    {
      "field_name": "officer_name",
      "value": "Rob Sherwin",
      "confidence": "high",
      "sources": ["https://www.nbcdfw.com/..."],
      "extraction_method": "llm"
    }
  ],
  "conflicting_fields": [
    {
      "field_name": "civilian_name",
      "conflict_type": "articles_disagree",
      "values": [
        "Gerardo Ramirez",
        "Gerardo Ramirez (plus unrelated names ...)"
      ],
      "sources": [
        ["https://www.nbcdfw.com/..."],
        ["https://www.dallasnews.com/..."]
      ]
    }
    // ... 7 more conflicting fields
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
python -m src.eval.run_eval civilians_shot --limit 100 --stratified
python -m src.eval.run_eval officers_shot  --limit 100 --stratified
```

**Holdout results (N=100 each, all agents on, `claude-sonnet-4-6`):**

| Metric           | civilians_shot | officers_shot |
| ---------------- | -------------- | ------------- |
| Completion rate  | 70% (70/100)   | 92% (92/100)  |
| Escalation rate  | 30% (30/100)   | 8% (8/100)    |
| Aggregate exact  | 77% (210/272)  | 71% (147/207) |
| Aggregate fuzzy  | 89% (243/272)  | 86% (179/207) |

Per-field accuracy (civilians_shot):

| Field           | Coverage | Exact | Fuzzy |
| --------------- | -------- | ----- | ----- |
| civilian_age    | 58%      | 95%   | 95%   |
| outcome         | 74%      | 92%   | 92%   |
| weapon          | 55%      | 83%   | 83%   |
| time_of_day     | 42%      | 82%   | 82%   |
| location_detail | 44%      | 16%   | 91%   |
| civilian_race   | 11%      | 91%   | 91%   |

Age, outcome, weapon, and time-of-day are the strongest fields; location is 91%
correct by fuzzy match (the exact gap is city-vs-address formatting only). The
**race verifier** is visible here: `civilian_race` coverage falls (17%→11% vs. a
gate-off run) while exact accuracy rises (65%→91%) — it nulls races the source
doesn't explicitly state, trading coverage for faithfulness. Of the 30 civilian
escalations, 22 are retrieval gaps (no articles found), 7 are relevance-gate
vetoes (`irrelevant_sources` — wrong-article cases that previously completed
silently), and 1 `insufficient_sources`. Officers complete at 92% (vs. an earlier
gate-off 95%): the relevance gate vetoes 3 wrong-article officer completions into
review (officer per-field: age 82%, outcome 87%, location 97% fuzzy, race 67%).
Reports save to `output/eval/`.

**Adversarial evaluation (N=20, fabricated incidents):** 20 fabricated incidents
(fake names, real Texas cities/dates) were run through the live pipeline. **All
20 escalated; zero completions; zero hallucinations** — fabricated names never
appeared in extracted fields. (17 escalated at search via `max_retries`, 2 via
the relevance gate's `irrelevant_sources`, 1 `insufficient_sources`.)

See [EVALUATION.md](EVALUATION.md) for full methodology, error analysis,
fairness metrics, adversarial evaluation, and discussion.

### Configuration

Environment variables (see `.env.example`):

| Variable                                | Default             | Description                                                                 |
| --------------------------------------- | ------------------- | --------------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`                     | (required)          | Anthropic API key                                                           |
| `ANTHROPIC_MODEL`                       | `claude-sonnet-4-6` | Model for LLM-powered nodes (extraction, relevance judge, race verifier)    |
| `ANTHROPIC_CHEAP_MODEL`                 | `claude-haiku-4-5-20251001` | Cheap model for the advisory conflict annotator                     |
| `TAVILY_API_KEY`                        | (required)          | Tavily API key for news search                                              |
| `LOG_LEVEL`                             | `INFO`              | Logging level                                                               |
| `ENRICHMENT_OUTPUT_DIR`                 | `output/enrichment` | Output directory for JSON results                                           |
| `ENRICHMENT_MAX_SEARCH_RESULTS`         | `10`                | Max articles per search                                                     |
| `ENRICHMENT_SEARCH_DEPTH`               | `advanced`          | Tavily search depth                                                         |
| `ENRICHMENT_SEARCH_WINDOW_BACK_DAYS`    | `14`                | Days before incident date for the Tavily search window                      |
| `ENRICHMENT_SEARCH_WINDOW_FORWARD_DAYS` | `60`                | Days after incident date for the Tavily search window                       |
| `ENRICHMENT_FUZZY_MATCH_THRESHOLD`      | `80`                | Min rapidfuzz score for name matching                                       |
| `ENRICHMENT_DATE_PROXIMITY_DAYS`        | `5`                 | Max days between article and incident                                       |
| `ENRICHMENT_ENABLE_RELEVANCE_GATE`      | `true`              | LLM relevance judge — wrong-article veto, both datasets (`false` disables)   |
| `ENRICHMENT_ENABLE_RACE_VERIFICATION`   | `true`              | LLM race verifier — nulls an unstated `civilian_race`, civilians (`false` disables) |
| `ENRICHMENT_ENABLE_CONFLICT_ANNOTATION` | `true`              | Advisory LLM triage note on deep conflicts (`false` disables)               |

PostgreSQL connection variables (`DB_HOST`, `DB_PORT`, etc.) are configured in
`.env.example` and used by the ETL pipeline (`data/`).

## Development

### Project Structure

```text
police-data-intelligence/
├── src/
│   ├── agents/
│   │   ├── state.py             # EnrichmentState, Article, FieldConflict, models
│   │   ├── graph.py             # LangGraph wiring, complete/escalate terminal nodes
│   │   ├── coordinate_node.py   # Coordinator gates (search/validate/synthesize checks)
│   │   └── load_node.py         # Load node (PostgreSQL → state)
│   ├── retrieval/
│   │   └── search_node.py       # Search node (Tavily API)
│   ├── validation/
│   │   └── validate_node.py     # Validate node (date/location/name matching)
│   ├── synthesize/
│   │   ├── synthesize_node.py   # Synthesize node (LLM extraction + consistency)
│   │   ├── relevance_judge.py   # LLM relevance judge (wrong-article veto, both datasets)
│   │   ├── race_verifier.py     # LLM race verifier (nulls unstated civilian_race)
│   │   ├── conflict_annotator.py # Advisory LLM triage note on deep conflicts (Haiku)
│   │   └── weapon_similarity.py # Weapon category normalization helpers
│   ├── database/
│   │   └── connection.py        # PostgreSQL connection
│   ├── eval/
│   │   ├── holdout.py           # Holdout evaluation (compare vs DB ground truth)
│   │   ├── gate.py              # Multi-objective accept/reject gate (target/adversarial/correctness)
│   │   └── run_eval.py          # Eval CLI entrypoint
│   ├── field_normalizers.py     # Deterministic outcome/time consensus normalizers
│   ├── race_taxonomy.py         # Race → TJI-bucket taxonomy + divergence flag
│   ├── config.py                # Settings (pydantic-settings, from env vars)
│   └── run.py                   # CLI entrypoint
├── scripts/
│   └── run_adversarial.py       # Adversarial suite (fabricated incidents → hallucination check)
├── data/
│   └── etl/                     # ETL pipeline (CSV → PostgreSQL), separate from agents
├── tests/
│   ├── test_synthesize_node.py
│   ├── test_relevance_judge.py
│   ├── test_race_taxonomy.py
│   ├── test_field_normalizers.py
│   ├── test_gate.py
│   ├── test_coordinate_node.py
│   ├── test_graph.py            # Graph wiring + terminal node tests
│   ├── test_holdout.py
│   └── ...                      # load/search/validate/run + ETL tests (cleaners, loaders, schemas)
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

## Performance

Per-incident wall-clock is bimodal: incidents with no news coverage escalate
quickly at search (a few seconds), while completions are dominated by LLM calls —
per-article extraction plus the agentic precision layer (relevance judge always;
race verifier for civilian races; Haiku annotator on conflicts). In the N=100
holdout runs the mean end-to-end time per incident — including eval-harness
overhead (ground-truth fetch and per-field comparisons) — was ~50s for
civilians_shot and ~90s for officers_shot. See
[EVALUATION.md — Cost and Latency](EVALUATION.md#cost-and-latency).

### Cost

Estimated per-record API cost using Claude Sonnet 4.6 (extraction + relevance
judge + race verifier), Claude Haiku (advisory annotator), and Tavily advanced
search (PAYGO pricing). The agentic precision layer adds one to a few LLM calls
per completion on top of extraction, so the LLM cost is higher than an
extraction-only pipeline:

| Component       | Per Record | 1,956 Records |
| --------------- | ---------- | ------------- |
| Anthropic (LLM) | ~$0.16     | ~$310         |
| Tavily (search) | ~$0.04     | ~$78          |
| **Total**       | **~$0.20** | **~$390**     |

Estimate — cost varies with retry count, article length, and how many agents fire
(escalated incidents that never reach extraction cost only search). See
[EVALUATION.md](EVALUATION.md) for methodology.

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
- Partial completion on synthesize conflicts (accept agreed fields, flag
  conflicts)
- 3-tier search strategy (exact → temporal → name_partial)
- Deterministic conflict reduction: victim-anchored extraction, outcome/time
  consensus resolver, race taxonomy normalization
- CLI entrypoint for single-incident enrichment
- Holdout evaluation framework + multi-objective accept/reject gate; eval on
  **both** datasets (civilians_shot and officers_shot) — see
  [Evaluation](#evaluation)
- Adversarial evaluation: 20 fabricated incidents, **0 hallucinations**
- Agentic precision/safety layer — three bounded LLM checks, on by default:
  relevance judge (wrong-article veto, both datasets), race verifier (nulls an
  unstated `civilian_race`), conflict annotator (advisory triage note)

**Next:**

- Batch processing across all records
- Human review UI for processing escalated records
- Cost study: a cheaper model (Haiku) for high-volume extraction

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
