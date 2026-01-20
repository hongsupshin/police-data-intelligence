# Police Data Intelligence Assistant

**An agentic AI system for validating, enriching, and analyzing police shooting
data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project builds a multi-agent system that **automatically enriches missing
data** in two Texas Justice Initiative (TJI) databases through intelligent web
search and extraction. The system's core purpose is data augmentation, not
analysis.

**Datasets**:

- **Civilians-Shot** (1,674 records): Police shooting civilians - 57% missing
  weapon info, 22.5% missing names
- **Officers-Shot** (282 records): Civilians shooting police - 40% missing
  officer names
- **Total**: 1,956 records needing enrichment

**The Problem**: TJI volunteers spend 15-30 minutes per record manually
searching news articles and extracting details. This takes 25-50 hours for 100
records.

**The Solution**: An agentic AI system that automates the enrichment workflow
while keeping humans in the loop, reducing volunteer time by 75%.

## System Capabilities

### Primary Functions

#### 1. Data Enrichment

- Search news articles using incident details (date, location, names)
- Extract **media feature set** (officer names, weapons, ages, circumstances,
  etc.)
- Run entity detection (NER) for supplementary information
- Validate via anchor matching and assign per-field confidence scores
- Output results to CSV/spreadsheet with reasoning summaries for human review

#### 2. Batch Processing

- Process all records, prioritized by location and year (Houston → Dallas →
  Austin, recent first)
- Track progress and cost per batch
- Generate structured output file for human review
- Resume failed jobs with exponential backoff

### Future Enhancements (Post-MVP)

#### 3. Query Answering

- Natural language → SQL generation for ad-hoc queries
- "How many shootings in Austin in 2018?" → Returns structured data
- _Not needed for enrichment workflow - deferred to v2_

#### 4. Research Synthesis

- Semantic search on academic paper corpus
- "What does research say about racial disparities?" → Returns relevant excerpts

## Architecture

The system uses **4 specialized nodes** orchestrated by a Coordinator in
LangGraph. Each node is either deterministic (rule-based) or agentic
(LLM-powered).

### Node Descriptions

#### 1. Extract Node (Deterministic)

- Selects an incident record from the database
- Reads existing fields: date, location, officer/civilian names, severity,
  outcome
- Outputs structured data for downstream processing
- No LLM needed - simple SQL query and field parsing

#### 2. Search Node (Deterministic)

- Constructs search query from extracted incident fields
- Calls **Tavily API** to retrieve 5-10 news articles
- Tracks search attempts with metadata (query used, results found, strategy
  applied)
- Implements multiple search strategies:
  - `exact_match`: All fields included, exact date
  - `temporal_expanded`: Date range ±2 days
  - `entity_dropped`: Remove officer name, keep location + date
- No LLM needed - algorithmic query construction

#### 3. Validate Node (Hybrid)

- Rule-based anchor matching (date ±2 days, location verification)
- Extracts media feature set from retrieved articles
- Runs entity detection (NER) for additional information
- Assigns per-field confidence scores
- LLM used only for fuzzy matching and conflict resolution

#### 4. Merge Node (Hybrid)

- Aggregates validated data from multiple articles
- Detects conflicts (e.g., one source says "taser" another says "firearm")
- Rule-based aggregation for consistent data
- Flags conflicts for human review escalation

#### 5. Coordinator Node (Agentic)

- **Retry orchestration**: Examines Search/Validate outputs and decides retry
  strategy
- **Escalation routing**: Routes conflicts and low-confidence results to human
  review
- **Flow control**: Manages state transitions and completion criteria
- Uses LLM reasoning to make routing decisions based on intermediate results

### Agent Walkthrough

Example: Enriching Record #142 — shooting in Houston, 2018-03-15, officer name:
`NULL`

**Initial Flow (Happy Path)**:

```text
Record #142
    ↓
Extract → {officer: NULL, date: 2018-03-15, location: "Houston", severity: "fatal"}
    ↓
Search (exact_match) → 3 articles retrieved
    ↓
Validate → Officer name extracted: "James Rodriguez" [HIGH confidence]
    ↓
Merge → No conflicts detected
    ↓
Coordinator → Route to output (database staging)
    ↓
Human reviews and approves
```

**Retry Flow (Insufficient Results)**:

```text
Record #142
    ↓
Extract → {officer: NULL, date: 2018-03-15, location: "Houston"}
    ↓
Search (exact_match) → 1 article retrieved [insufficient]
    ↓
Coordinator → Decision: "Insufficient sources (< 2), retry with temporal_expanded"
    ↓
Search (temporal_expanded) → 4 articles retrieved [sufficient]
    ↓
Validate → Extract fields from 4 sources
    ↓
Merge → 2 sources agree on "Officer Rodriguez", 2 have no officer name
    ↓
Coordinator → Route to output with MEDIUM confidence flag
```

**Escalation Flow (Conflicts Detected)**:

```text
Search → 5 articles retrieved
    ↓
Validate → All 5 articles pass anchor matching
    ↓
Merge → Conflict detected: 3 sources say "taser", 2 say "firearm"
    ↓
Coordinator → Decision: "Conflicting weapon data, escalate to human review"
    ↓
Human review queue (with source evidence provided)
```

### State Management

The system tracks enrichment progress through a shared `EnrichmentState` that
flows through all nodes:

```python
class EnrichmentState:
    # Original incident data (from Extract)
    incident_id: str
    officer_name: Optional[str]
    civilian_name: Optional[str]
    incident_date: datetime
    location: str
    severity: str

    # Search tracking
    search_attempts: List[SearchAttempt]  # History of all search tries
    retrieved_articles: List[Article]     # Current article set

    # Validation results
    validated_articles: List[ValidatedArticle]

    # Coordinator control
    retry_count: int
    max_retries: int = 3
    next_strategy: str  # "exact_match", "temporal_expanded", "entity_dropped"
    current_stage: str  # "extract", "search", "validate", "merge"

    # Escalation
    escalation_reason: Optional[str]
    requires_human_review: bool
```

**SearchAttempt** tracks each search iteration:

```python
class SearchAttempt:
    query: str                    # "Houston shooting March 2018"
    strategy: str                 # "exact_match", "temporal_expanded", etc.
    num_results: int              # How many articles returned
    avg_relevance_score: float    # Search quality metric
    timestamp: datetime
```

This design makes retry logic **visible in the graph execution trace** - you can
see exactly what searches were tried and why they succeeded or failed.

### Retry Strategy Progression

The Coordinator implements a escalating retry strategy:

| Retry # | Strategy            | Description                             | Max Retries         |
| ------- | ------------------- | --------------------------------------- | ------------------- |
| 0       | `exact_match`       | All fields, exact date                  | -                   |
| 1       | `temporal_expanded` | Date range ±2 days                      | After < 2 results   |
| 2       | `entity_dropped`    | Drop officer name, keep location + date | After retry 1 fails |
| 3       | Escalate            | Flag for human review                   | Hard limit          |

**Design Rationale**: Each retry changes the search approach in a meaningful
way. Simple parameter tweaking (e.g., lowering confidence threshold) happens
within nodes; strategy changes route through the Coordinator for visibility.

### Validation Pipeline

The Validation Node implements a multi-stage pipeline:

```text
Retrieved Articles (1-5)
        ↓
   Deduplicate (content similarity > 0.9 = same source)
        ↓
   Anchor Match (date ±2 days, location match)
        ↓
   [No match] → "No information found" → Route to Coordinator
        ↓
   Extract media feature set + entity detection
        ↓
   Per-field consistency check
        ↓
   Output to Merge with confidence scores
```

**Source Expectations**: Local news incidents typically have 1-3 unique sources.
Single-source extractions are valid but flagged for review.

**Consistency Classification**:

| Scenario      | Example                                     | Action                          |
| ------------- | ------------------------------------------- | ------------------------------- |
| Agreement     | Both sources say "Officer Rodriguez"        | High confidence                 |
| Conflict      | Source A: "Rodriguez", B: "Ramirez"         | Flag for Coordinator escalation |
| Composite     | A has name + clothing, B has name + vehicle | Flag with merged view           |
| Single source | Only one article found                      | Medium confidence               |

**Per-Field Confidence**: Each field gets its own confidence level:

```text
Record #142 Enrichment:
✓ Officer Name: "James Rodriguez" [HIGH - 2 sources agree]
⚠ Weapon: "Handgun" [MEDIUM - 1 source, unclear context]
✗ Age: Not found
```

**LLM Reasoning** (used sparingly for cost): Only generated for
flagged/medium-confidence cases:

```text
Flag: Conflicting officer names
Reasoning: Source A (Houston Chronicle) names "Officer James Rodriguez"
in a direct quote from police spokesperson. Source B (KHOU) names
"Sgt. J. Ramirez" without attribution. Chronicle appears more authoritative.
Suggested resolution: Prefer "James Rodriguez"
```

**Escalation Triggers** (Coordinator routes to human review):

- Confidence below threshold
- Conflicting information across sources
- Composite information requiring merge review
- Overwriting existing non-null values
- Soft anchor match (date ±3-7 days)
- Max retries reached without sufficient data

**Provenance Tracking**: Every extracted field records its source(s), exact
quote/context, and extraction method for audit trails.

## Data Sources

1. **TJI Database** (PostgreSQL): 1,956 incident records with known quality gaps
2. **Research Corpus** (Qdrant Vector DB): 30-50 academic papers for contextual
   research
3. **News Articles** (Tavily API): On-demand retrieval for data enrichment

### Dual-Dataset Schema Complexity

The two TJI datasets use **different field names for the same concepts**,
requiring dataset-aware extraction:

| Concept        | Civilians-Shot                 | Officers-Shot                       |
| -------------- | ------------------------------ | ----------------------------------- |
| Civilian name  | `civilian_name_first`          | `civilian_name_first_1/2/3` (array) |
| Officer count  | `num_officers_recorded` (1-11) | N/A (always single)                 |
| Civilian count | N/A (always single)            | `num_civilians_recorded` (1-3)      |
| Outcome        | `civilian_died` (boolean)      | `civilian_harm` (enum)              |
| Weapon         | `weapon_reported_by_media`     | ❌ No field exists                  |

This means the Extract and Validate nodes must use **conditional field mapping**
based on which dataset a record belongs to.

### Batch Processing Priority

Process all records, prioritized by location and year:

| Priority | Location     | Rationale                    |
| -------- | ------------ | ---------------------------- |
| 1        | Houston      | Largest city, most incidents |
| 2        | Dallas       | Second largest               |
| 3        | Austin       | State capital                |
| 4        | Other cities | Remaining records            |

Within each location, process recent years first (2024 → 2023 → ...).

**Total records: 1,956** across both datasets.

## Deployment

### Trigger Model

TJI receives data from the Texas state government on an **annual release
cycle**. The system uses a **manual trigger via REST API**:

```text
[TJI receives annual state data] → [Admin calls API endpoint] → [Batch enrichment runs]
```

### Cloud Architecture

- **AWS Lambda + API Gateway**: Serverless — no always-on infrastructure costs
- **Endpoint**: `POST /enrich/batch` triggers processing of new/incomplete
  records
- **Cost**: Near-zero (Lambda free tier covers infrequent batch runs)

## Tech Stack

| Component           | Technology           | Purpose                                                          |
| ------------------- | -------------------- | ---------------------------------------------------------------- |
| Agent Orchestration | LangGraph            | State management, graph-based execution with conditional routing |
| LLM Provider        | OpenAI (GPT-4o-mini) | Coordinator reasoning and validation fuzzy matching              |
| Database            | PostgreSQL           | TJI incident records                                             |
| Vector Store        | Qdrant               | Research paper semantic search                                   |
| Web Search          | Tavily               | News retrieval (1,000 free searches/month)                       |
| API Framework       | FastAPI              | REST endpoints                                                   |
| Deployment          | Docker + Terraform   | Infrastructure as code                                           |
| Monitoring          | LangSmith            | LLM observability and execution tracing                          |

## Success Metrics

**Quantitative Targets**:

- Enrichment precision: >80% (correct suggestions)
- Enrichment coverage: >50% (missing fields filled)
- Cost per record: <$0.15
- Latency: p90 <10s

**Baseline Comparison**:

- Manual workflow: 15-30 minutes per record
- This system: 3-5 minutes with automation + human verification
- Time reduction: 75%

## Responsible AI

This system operates in a sensitive domain (police accountability). Key design
principles:

- **Human-in-the-loop**: System never auto-updates database; humans approve all
  changes
- **Transparency**: Show evidence (article excerpts) and confidence scores
- **Traceability**: Link suggestions to original sources with provenance
  tracking
- **Accuracy over automation**: Conservative confidence thresholds, escalation
  on conflicts
- **Visible retry logic**: All search attempts logged for audit and debugging

## Development

### Project Structure

```text
police-data-intelligence/
├── src/
│   ├── agents/          # Node implementations (Extract, Search, Validate, Merge, Coordinator)
│   ├── state/           # Pydantic state models (EnrichmentState, SearchAttempt, etc.)
│   ├── retrieval/       # Tavily integration and search strategies
│   ├── validation/      # Anchor matching and confidence scoring
│   ├── database/        # PostgreSQL connection and queries
│   └── api/             # FastAPI endpoints
├── tests/
│   ├── unit/            # Unit tests per node
│   └── integration/     # End-to-end graph execution tests
├── docs/                # Documentation
└── infra/               # Infrastructure as code
    ├── terraform/       # AWS infrastructure
    └── docker/          # Docker configuration
```

### Roadmap

**Timeline**: 8 weeks (January - March 2026)

- ✅ Week 1-2: Architecture design and project setup
- ⏳ Week 3-4: Core node implementation (Extract, Search, Validate, Merge)
- ⏳ Week 5-6: Coordinator logic and retry orchestration
- ⏳ Week 7-8: Cloud deployment, evaluation, and documentation

## Why No Planning Agent?

Earlier designs included a Planning Agent for upfront strategy selection. We
removed it because:

1. **Homogeneous incident types**: All records are use-of-force incidents with
   similar enrichment needs
2. **Reactive routing is sufficient**: The Coordinator handles strategy changes
   based on actual results
3. **Simpler architecture**: Fewer nodes = easier debugging and lower LLM costs

**When Planning makes sense**: If we expand to different incident categories
(complaints, misconduct investigations, traffic stops) that need fundamentally
different graph structures, we'd add Planning to route to specialized subgraphs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgments

- **Texas Justice Initiative** for providing data and real-world context
- Built to demonstrate production-grade AI engineering practices

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.
