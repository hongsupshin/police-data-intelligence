# Police Data Intelligence Assistant

**An agentic AI system for validating, enriching, and analyzing police shooting data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project builds a multi-agent RAG system that helps the Texas Justice Initiative (TJI) improve data quality in their police officer-involved shootings database. The system combines database queries, academic research, and web news search to both answer questions about police violence and actively validate and enrich incomplete records.

## The Problem

Working with TJI volunteers revealed significant data quality challenges:
- **40% of records missing officer names** - critical for accountability
- **16% missing civilian details** - incomplete victim information
- **Manual enrichment is tedious** - volunteers spend 15-30 minutes per record searching news articles and extracting details
- **News coverage often more detailed** than official reports

**Current volunteer workflow**: Identify incomplete record → Google for news articles → Read multiple sources → Extract structured information → Manually update database. This takes 25-50 hours for 100 records.

## What This System Does

An agentic AI system that **automates the enrichment workflow** while keeping humans in the loop:

1. **Query Answering**: "How many shootings in Austin in 2018?" → Analyze database
2. **Research Context**: "What does research say about racial disparities?" → Synthesize academic papers
3. **Data Enrichment**: "Validate this record" → Search news, extract details, suggest corrections with confidence scores
4. **Statistical Guidance**: "Is this sample size sufficient?" → Provide methodological advice

**Impact**: Reduces volunteer time by 75%, increases data completeness from 40% missing to 15% missing, enables sophisticated data analysis.

## Architecture

The system uses **5 specialized agents** orchestrated by LangGraph:

```
┌──────────────────────────────────────────────────────┐
│                 Coordinator Agent                     │
│  Routes queries, manages state, orchestrates workflow │
└─────────────────────┬────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬───────────────┐
        │             │             │               │
        ▼             ▼             ▼               ▼
   ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │Planning │  │Retrieval │  │Validation│  │Synthesis │
   │ Agent   │  │  Agent   │  │  Agent   │  │  Agent   │
   └─────────┘  └──────────┘  └──────────┘  └──────────┘
        │             │             │               │
     Query        SQL/Vector/     Answer &         Response
    Analysis      Web Search    Data Quality      Generation
```

### Agent Responsibilities

- **Coordinator**: Routes queries, manages execution flow, handles errors
- **Planning**: Decomposes queries, selects data sources, determines retrieval strategy
- **Retrieval**: Fetches from SQL database, vector store (research papers), and web (news)
- **Validation**: Validates answer quality, checks data consistency, performs entity resolution
- **Synthesis**: Generates responses with citations, presents data correction suggestions

## Data Architecture

The system integrates three heterogeneous data sources:

1. **TJI Database** (PostgreSQL)
   - 282 records of officer-involved shootings (2015-2019)
   - Structured data with known quality gaps

2. **Research Corpus** (Qdrant Vector DB)
   - 30-50 academic papers on police violence and criminal justice
   - Semantic search for contextual research

3. **News Articles** (Dynamic Web Search)
   - On-demand article retrieval via Tavily/Serper API
   - Structured information extraction for data enrichment

## Key Features

### Core Capabilities
- **Multi-agent orchestration** with LangGraph
- **Multi-source retrieval** (SQL + Vector + Web)
- **Data quality validation** with entity resolution and confidence scoring
- **Human-in-the-loop** approval workflow for sensitive corrections
- **Cost tracking** per agent and data source
- **REST API** for integration
- **Cloud deployment** on AWS with monitoring

### Data Enrichment Process
1. Identify incomplete database record
2. Construct search query from available fields
3. Retrieve relevant news articles
4. Extract structured information (names, ages, circumstances)
5. Match articles to incidents using date, location, and entity resolution
6. Assign confidence scores
7. Present suggestions to humans for approval

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Orchestration | LangGraph | State management, graph-based execution |
| LLM Provider | OpenAI (GPT-4o, GPT-4o-mini) | Reasoning and generation |
| Database | PostgreSQL | TJI incident records |
| Vector Store | Qdrant | Research paper semantic search |
| Web Search | Tavily/Serper | News article retrieval |
| API Framework | FastAPI | REST endpoints |
| Deployment | Docker + Terraform | Infrastructure as code |
| Monitoring | LangSmith | LLM observability and tracing |

## Project Status

**Current Phase**: Initial setup and architecture design (Week 1-2)

**Timeline**: 8 weeks (March - April 2026)

### Roadmap
- ✅ Week 1-2: Architecture design and project setup
- ⏳ Week 3-4: Core agent implementation
- ⏳ Week 5-6: Cloud deployment and data quality enhancement
- ⏳ Week 7-8: Evaluation and documentation

## Getting Started

*Coming soon - development environment setup, Docker Compose configuration, and quick start guide*

## Development

### Project Structure
```
police-data-intelligence/
├── src/
│   ├── agents/          # Agent implementations
│   ├── retrieval/       # Data retrieval strategies
│   ├── validation/      # Quality validation logic
│   ├── database/        # Database connection and queries
│   └── api/            # FastAPI endpoints
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── docs/               # Documentation
└── infra/              # Infrastructure as code
    ├── terraform/      # AWS infrastructure
    └── docker/         # Docker configuration
```

## Success Metrics

### Quantitative Targets
- Answer quality: >4.0/5.0 (LLM-as-judge)
- Enrichment precision: >80% (correct suggestions)
- Enrichment coverage: >50% (missing fields filled)
- Cost per query: <$0.15
- Latency: p90 <10s

### Baseline Comparison
- **Manual workflow**: 15-30 minutes per record
- **This system**: 3-5 minutes with automation + human verification
- **Time reduction**: 75%

## Responsible AI

This system operates in a sensitive domain (police accountability). Key design principles:
- **Human-in-the-loop**: Agent never auto-updates database; humans approve all changes
- **Transparency**: Show evidence (article excerpts) and confidence scores
- **Traceability**: Link suggestions to original sources
- **Accuracy over automation**: Conservative confidence thresholds

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Texas Justice Initiative** for providing data and real-world context
- Inspired by research on agentic AI and multi-source retrieval systems
- Built to demonstrate production-grade AI engineering practices

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Note**: This is an active development project being built as a portfolio demonstration of agentic AI, multi-source data integration, and production ML system design.
