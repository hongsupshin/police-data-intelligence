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
- Validate via incident matching and assign per-field confidence scores
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

The system uses **5 specialized nodes** orchestrated by a Coordinator in
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

#### 3. Validate Node (Rule-based)

- **Purpose**: Verify each article describes the _same incident_ as the database
  record
- **Checks performed** (binary pass/fail per article):
  - Date match: Within ±3 days of `incident_date`
  - Location match: String similarity or geocoding verification
  - Victim name match: When available (optional check, accounts for name
    disclosure issues)
- **Pass criteria**: Article passes if `date_match AND location_match` are both
  True
- **Output**: `ValidationResult` per article with binary flags
- **Design rationale**: Simple rule-based checks are sufficient because these
  rare incidents are nearly unique by time + location alone

**ValidationResult structure**:

```python
class ValidationResult:
    article: Article
    date_match: bool          # Within ±3 days
    location_match: bool      # String similarity or geocoding
    victim_name_match: bool | None  # None if victim name unavailable
    passed: bool              # True if date AND location match
```

**Note on temporal tolerance**: Articles may describe incidents at different
reporting stages (day-of shooting vs. later outcome). An article saying "suspect
in critical condition" when the database shows "fatal" is still the _same
incident_ - this is handled in Merge as temporally-sequenced information, not
validation failure.

#### 4. Merge Node (Hybrid)

- **Sub-step 1: Extract fields** from each validated article using LLM
- **Sub-step 2: Check consistency** across articles (accounting for format
  variations)
- **Sub-step 3: Merge with database** record
- **Conflict detection**: Identifies when articles disagree on substantive facts
  (not just formatting)
- **Design principle**: Only enriches fields being added/updated; does not
  create entries for unchanged fields

**Merge logic per field**:

```
For each field:
  - Database has value + articles agree → keep database (immutable)
  - Database empty + articles agree → add from articles
  - Database has value + articles conflict with it → escalate
  - Articles conflict with each other → escalate
```

**Consistency classification**:

| Scenario         | Example                                     | Action                          |
| ---------------- | ------------------------------------------- | ------------------------------- |
| Agreement        | Both sources say "Officer Rodriguez"        | High confidence                 |
| Conflict         | Source A: "Rodriguez", B: "Ramirez"         | Flag for Coordinator escalation |
| Composite        | A has name + clothing, B has name + vehicle | Flag with merged view           |
| Single source    | Only one article found                      | Medium confidence               |
| Format variation | "123 E 6th St" vs "123 East Sixth Street"   | Normalize and merge             |

**Immutability assumption**: All database values are from official state
government data and treated as ground truth. If articles conflict with existing
database values, this triggers escalation rather than overwriting.

#### 5. Coordinator Node (Agentic)

- **Retry orchestration**: Examines Search/Validate outputs and decides retry
  strategy
- **Escalation routing**: Routes conflicts and low-confidence results to human
  review
- **Flow control**: Manages state transitions and completion criteria
- **Decision gates**: Acts as supervisor at multiple pipeline points (after
  Search, after Validate, after Merge)
- Uses LLM reasoning to make routing decisions based on intermediate results

**Coordinator decision points**:

```
Search → Coordinator (quantity check) → Validate → Coordinator (quality check) → Merge → Coordinator (conflict check) → Output or Escalation
         ↑_____retry if <2 articles_____|          ↑___retry if 0 validated___|          ↑___escalate if conflicts___|
```

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
Coordinator → Quantity check: 3 articles ≥ 2 → proceed to Validate
    ↓
Validate → All 3 articles pass (date + location match)
    ↓
Coordinator → Quality check: 3 validated articles → proceed to Merge
    ↓
Merge → Extract: Officer name "James Rodriguez" from 2 sources, weapon "handgun" from 1 source
    ↓
Merge → Consistency check: No conflicts detected
    ↓
Coordinator → Conflict check: Clean merge → route to output
    ↓
Human reviews and approves
```

**Retry Flow (Insufficient Results)**:

```text
Record #142
    ↓
Extract → {officer: NULL, date: 2018-03-15, location: "Houston"}
    ↓
Search (exact_match) → 1 article retrieved
    ↓
Coordinator → Quantity check: 1 article < 2 → retry with temporal_expanded
    ↓
Search (temporal_expanded) → 4 articles retrieved
    ↓
Coordinator → Quantity check: 4 articles ≥ 2 → proceed to Validate
    ↓
Validate → 3 articles pass matching, 1 fails (wrong date)
    ↓
Coordinator → Quality check: 3 validated articles → proceed to Merge
    ↓
Merge → 2 sources agree on "Officer Rodriguez", 1 has no officer name
    ↓
Coordinator → Route to output with MEDIUM confidence flag
```

**Escalation Flow (Conflicts Detected)**:

```text
Search → 5 articles retrieved
    ↓
Coordinator → Quantity check: 5 articles ≥ 2 → proceed
    ↓
Validate → All 5 articles pass date + location matching
    ↓
Coordinator → Quality check: 5 validated articles → proceed
    ↓
Merge → Extract weapon info: 3 sources say "taser", 2 say "firearm"
    ↓
Merge → Consistency check: Conflict detected on weapon field
    ↓
Coordinator → Conflict check: Conflicting weapon data → escalate to human review
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
    next_strategy: str                    # Coordinator sets this for Search

    # Validation results
    validation_results: List[ValidationResult]

    # Merge outputs
    extracted_fields: List[FieldExtraction]  # Only enriched/updated fields
    conflicting_fields: Optional[List[str]]  # Field names with conflicts

    # Coordinator control
    retry_count: int
    max_retries: int = 3
    current_stage: str              # "extract", "search", "validate", "merge"
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

**FieldExtraction** tracks provenance for audit trails:

```python
class FieldExtraction:
    field_name: str                    # e.g., "officer_badge_number"
    value: str | None                  # Extracted value
    confidence: ConfidenceLevel        # HIGH, MEDIUM, LOW
    sources: List[str]                 # Article URLs
    source_quotes: List[str]           # Exact quotes from sources
    extraction_method: str             # "llm", "regex", "ner"
    llm_reasoning: Optional[str]       # Only for flagged cases
```

**Provenance design principle**: Only fields being **added or updated** get
`FieldExtraction` objects. This reduces noise (no need for 17 empty entries when
only 3 fields changed) while maintaining complete audit trail for enrichments. A
separate `fields_checked` list tracks what Merge attempted to extract, enabling
"we looked for X but didn't find it" debugging without bloating the core state.

This design makes retry logic **visible in the graph execution trace** - you can
see exactly what searches were tried and why they succeeded or failed.

### Retry Strategy Progression

The Coordinator implements an escalating retry strategy:

| Retry # | Strategy            | Description                             | Trigger             |
| ------- | ------------------- | --------------------------------------- | ------------------- |
| 0       | `exact_match`       | All fields, exact date                  | -                   |
| 1       | `temporal_expanded` | Date range ±2 days                      | After < 2 results   |
| 2       | `entity_dropped`    | Drop officer name, keep location + date | After retry 1 fails |
| 3       | Escalate            | Flag for human review                   | Hard limit          |

**Design Rationale**: Each retry changes the search approach in a meaningful way
(temporal expansion, spatial expansion, entity dropping). Simple parameter
tweaking (e.g., lowering confidence threshold) happens within nodes; strategy
changes route through the Coordinator for visibility.

### Escalation Triggers

The Coordinator routes to human review when:

- Confidence below threshold (per-field or overall)
- Conflicting information across sources (substantive, not formatting)
- Articles conflict with existing database values (immutability violation)
- Max retries reached without sufficient data
- Composite information requiring human judgment

**Human review format**: System provides article excerpts, extracted values,
confidence scores, and conflict descriptions to enable informed decision-making.

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

This means the Extract and Merge nodes must use **conditional field mapping**
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
| LLM Provider        | OpenAI (GPT-4o-mini) | Coordinator reasoning and Merge field extraction                 |
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
- **Immutability respect**: Never overwrite official government data without
  human approval

## Development

### Project Structure

```text
police-data-intelligence/
├── src/
│   ├── agents/          # Node implementations (Extract, Search, Validate, Merge, Coordinator)
│   ├── state/           # Pydantic state models (EnrichmentState, SearchAttempt, FieldExtraction, etc.)
│   ├── retrieval/       # Tavily integration and search strategies
│   ├── validation/      # Incident matching and binary flag checks
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
