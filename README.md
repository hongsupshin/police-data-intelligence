# Police Data Intelligence Assistant

**An agentic AI system for validating, enriching, and analyzing police shooting data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project builds a multi-agent system that **automatically enriches missing data** in two Texas Justice Initiative (TJI) databases through intelligent web search and extraction. The system's core purpose is data augmentation, not analysis.

**Datasets**:

- **Civilians-Shot** (1,674 records): Police shooting civilians - 57% missing weapon info, 22.5% missing names
- **Officers-Shot** (282 records): Civilians shooting police - 40% missing officer names
- **Total**: 1,956 records needing enrichment

**The Problem**: TJI volunteers spend 15-30 minutes per record manually searching news articles and extracting details. This takes 25-50 hours for 100 records.

**The Solution**: An agentic AI system that automates the enrichment workflow while keeping humans in the loop, reducing volunteer time by 75%.

## System Capabilities

### Primary Functions

#### 1. Data Enrichment

- Search news articles using incident details (date, location, names)
- Extract structured information (officer names, weapons, ages, circumstances)
- Validate via entity resolution and assign per-field confidence scores
- Present suggestions to humans with evidence for approval

#### 2. Batch Processing

- Enqueue incomplete records by priority (P0 → P1 → P2)
- Track progress: "Enriched 245/1068 P0 records (23%)"
- Resume failed jobs with exponential backoff
- Cost monitoring per batch

### Future Enhancements (Post-MVP)

#### 3. Query Answering

- Natural language → SQL generation for ad-hoc queries
- "How many shootings in Austin in 2018?" → Returns structured data
- *Not needed for enrichment workflow - deferred to v2*

#### 4. Research Synthesis

- Semantic search on academic paper corpus
- "What does research say about racial disparities?" → Returns relevant excerpts

## Architecture

The system uses **5 specialized agents** orchestrated by LangGraph.

### Agent Walkthrough

Example: Enriching Record #142 — shooting in Houston, 2018-03-15, officer name: `NULL`

#### 1. Coordinator Agent (Traffic Controller)

- Receives the enrichment request
- Routes to Planning Agent first
- Manages state between agents, handles retries if something fails

#### 2. Planning Agent (Strategist)

- Analyzes what's missing: officer name
- Determines data sources needed: web search (news articles)
- Constructs search strategy: `"Houston police shooting March 2018"` + date filters

#### 3. Retrieval Agent (Data Fetcher)

- Calls **Tavily API** with the constructed query
- Returns 5-10 news articles with titles, snippets, URLs
- Uses simple database queries (by ID) to fetch record details - no NL→SQL generation needed

#### 4. Validation Agent (Quality Checker)

- Extracts candidate names from article text
- Performs entity resolution and anchor matching
- Assigns per-field confidence scores
- Flags conflicts or composite information

#### 5. Synthesis Agent (Output Formatter)

- Generates human-readable output with evidence and sources
- Presents to human for approval (never auto-updates)

### End-to-End Flow

```text
Record #142 (missing name)
        ↓
   Coordinator → "This is enrichment, route to Planning"
        ↓
   Planning → "Search news for Houston shooting March 2018"
        ↓
   Retrieval → [Tavily API] → 5 articles returned
        ↓
   Validation → "Officer James Rodriguez" (HIGH confidence)
        ↓
   Synthesis → Formatted suggestion + sources
        ↓
   Human reviews → Approve/Reject
```

### Validation Pipeline

The Validation Agent implements a multi-stage pipeline:

```text
Retrieved Articles (1-5)
        ↓
   Deduplicate (content similarity > 0.9 = same source)
        ↓
   Anchor Match (date ±2 days, location match)
        ↓
   [No match] → "No information found" → Done
        ↓
   Extract per field (name, weapon, age, etc.)
        ↓
   Per-field consistency check
        ↓
   Route to human review based on confidence
```

**Source Expectations**: Local news incidents typically have 1-3 unique sources. Single-source extractions are valid but flagged for review.

**Consistency Classification**:

| Scenario | Example | Action |
|----------|---------|--------|
| Agreement | Both sources say "Officer Rodriguez" | High confidence |
| Conflict | Source A: "Rodriguez", B: "Ramirez" | Flag with LLM reasoning |
| Composite | A has name + clothing, B has name + vehicle | Flag with merged view |
| Single source | Only one article found | Medium confidence |

**Per-Field Confidence**: Each field gets its own confidence level:

```text
Record #142 Enrichment:
✓ Officer Name: "James Rodriguez" [HIGH - 2 sources agree]
⚠ Weapon: "Handgun" [MEDIUM - 1 source, unclear context]
✗ Age: Not found
```

**LLM Reasoning** (used sparingly for cost): Only generated for flagged/medium-confidence cases:

```text
Flag: Conflicting officer names
Reasoning: Source A (Houston Chronicle) names "Officer James Rodriguez"
in a direct quote from police spokesperson. Source B (KHOU) names
"Sgt. J. Ramirez" without attribution. Chronicle appears more authoritative.
Suggested resolution: Prefer "James Rodriguez"
```

**Escalation Triggers**:

- Confidence below threshold
- Conflicting information across sources
- Composite information requiring merge review
- Overwriting existing non-null values
- Soft anchor match (date ±3-7 days)

**Provenance Tracking**: Every extracted field records its source(s), exact quote/context, and extraction method for audit trails.

## Data Sources

1. **TJI Database** (PostgreSQL): 1,956 incident records with known quality gaps
2. **Research Corpus** (Qdrant Vector DB): 30-50 academic papers for contextual research
3. **News Articles** (Tavily API): On-demand retrieval for data enrichment

### Dual-Dataset Schema Complexity

The two TJI datasets use **different field names for the same concepts**, requiring dataset-aware extraction:

| Concept | Civilians-Shot | Officers-Shot |
|---------|----------------|---------------|
| Civilian name | `civilian_name_first` | `civilian_name_first_1/2/3` (array) |
| Officer count | `num_officers_recorded` (1-11) | N/A (always single) |
| Civilian count | N/A (always single) | `num_civilians_recorded` (1-3) |
| Outcome | `civilian_died` (boolean) | `civilian_harm` (enum) |
| Weapon | `weapon_reported_by_media` | ❌ No field exists |

This means the Retrieval and Validation agents must use **conditional field mapping** based on which dataset a record belongs to.

### Enrichment Priority Targets

Focus batch enrichment on high-value, high-missing fields first:

| Priority | Dataset | Field | Missing | Count | Rationale |
|----------|---------|-------|---------|-------|-----------|
| **P0** | Officers-Shot | `officer_name` | 40% | 112 | Critical for honoring fallen/injured |
| **P0** | Civilians-Shot | `weapon_reported_by_media` | 57% | 956 | Context for force justification |
| **P1** | Civilians-Shot | `civilian_name` | 22.5% | 377 | Accountability |
| **P2** | Officers-Shot | `civilian_name_1` | 16% | 45 | Suspect identification |

**P0 targets total: 1,068 records** — the primary focus of batch enrichment.

## Deployment

### Trigger Model

TJI receives data from the Texas state government on an **annual release cycle**. The system uses a **manual trigger via REST API**:

```text
[TJI receives annual state data] → [Admin calls API endpoint] → [Batch enrichment runs]
```

### Cloud Architecture

- **AWS Lambda + API Gateway**: Serverless — no always-on infrastructure costs
- **Endpoint**: `POST /enrich/batch` triggers processing of new/incomplete records
- **Cost**: Near-zero (Lambda free tier covers infrequent batch runs)

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Agent Orchestration | LangGraph | State management, graph-based execution |
| LLM Provider | OpenAI (GPT-4o-mini) | Reasoning and generation |
| Database | PostgreSQL | TJI incident records |
| Vector Store | Qdrant | Research paper semantic search |
| Web Search | Tavily | News retrieval (1,000 free searches/month) |
| API Framework | FastAPI | REST endpoints |
| Deployment | Docker + Terraform | Infrastructure as code |
| Monitoring | LangSmith | LLM observability and tracing |

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

This system operates in a sensitive domain (police accountability). Key design principles:

- **Human-in-the-loop**: Agent never auto-updates database; humans approve all changes
- **Transparency**: Show evidence (article excerpts) and confidence scores
- **Traceability**: Link suggestions to original sources
- **Accuracy over automation**: Conservative confidence thresholds

## Development

### Project Structure

```text
police-data-intelligence/
├── src/
│   ├── agents/          # Agent implementations
│   ├── retrieval/       # Data retrieval strategies
│   ├── validation/      # Quality validation logic
│   ├── database/        # Database connection and queries
│   └── api/             # FastAPI endpoints
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── docs/                # Documentation
└── infra/               # Infrastructure as code
    ├── terraform/       # AWS infrastructure
    └── docker/          # Docker configuration
```

### Roadmap

**Timeline**: 8 weeks (March - April 2026)

- ✅ Week 1-2: Architecture design and project setup
- ⏳ Week 3-4: Core agent implementation
- ⏳ Week 5-6: Cloud deployment and data quality enhancement
- ⏳ Week 7-8: Evaluation and documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Texas Justice Initiative** for providing data and real-world context
- Built to demonstrate production-grade AI engineering practices

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.
