"""State models for the enrichment pipeline.

Defines Pydantic models for tracking state across nodes in the
LangGraph-based enrichment workflow. The pipeline consists of:
Extract → Coordinator → Search → Coordinator → Validate →
Coordinator → Merge → Coordinator → Complete | Escalate
"""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DatasetType(StrEnum):
    """Dataset identifier for conditional field mapping.

    The two TJI datasets use different field names for the same concepts,
    requiring dataset-aware extraction.
    """

    CIVILIANS_SHOT = "civilians_shot"
    OFFICERS_SHOT = "officers_shot"


class SearchStrategyType(StrEnum):
    """Search strategy for progressive query refinement.

    The Coordinator implements an escalating retry strategy:
    - exact_match: All fields, exact date (initial)
    - temporal_expanded: Date range ±2 days (after < 2 results)
    - entity_dropped: Drop officer name, keep location + date (after retry 1 fails)
    """

    EXACT_MATCH = "exact_match"
    TEMPORAL_EXPANDED = "temporal_expanded"
    ENTITY_DROPPED = "entity_dropped"


class PipelineStage(StrEnum):
    """Current stage in the enrichment pipeline."""

    EXTRACT = "extract"
    SEARCH = "search"
    VALIDATE = "validate"
    MERGE = "merge"
    COMPLETE = "complete"
    ESCALATE = "escalate"


class ConfidenceLevel(StrEnum):
    """Per-field confidence classification.

    Based on source agreement and evidence quality:
    - HIGH: Multiple sources agree
    - MEDIUM: Single source or unclear context
    - LOW: Weak evidence or soft anchor match
    - NONE: No information found
    - PENDING: confidence level is not determined yet (used in extract_field)
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"
    PENDING = "pending"


class EscalationReason(StrEnum):
    """Reasons for escalating to human review.

    Escalation triggers include:
    - Extraction failed (e.g., missing information or error message)
    - Search failed (error message)
    - Confidence below threshold
    - Conflicting information across sources
    - Composite information requiring merge review
    - Overwriting existing non-null values
    - Soft anchor match (date ±3-7 days)
    - Max retries reached without sufficient data
    """

    EXTRACTION_ERROR = "extraction_error"
    VALIDATION_ERROR = "validation_error"
    MERGE_ERROR = "merge_error"
    CONFLICT = "conflict"
    COMPOSITE = "composite"
    LOW_CONFIDENCE = "low_confidence"
    OVERWRITE = "overwrite"
    SOFT_ANCHOR = "soft_anchor"
    MAX_RETRIES = "max_retries"
    INSUFFICIENT_SOURCES = "insufficient_sources"


class MediaFeatureField(StrEnum):
    """Fields to extract from media articles (media feature set)."""

    OFFICER_NAME = "officer_name"
    CIVILIAN_NAME = "civilian_name"
    CIVILIAN_AGE = "civilian_age"
    CIVILIAN_RACE = "civilian_race"
    WEAPON = "weapon"
    LOCATION_DETAIL = "location_detail"
    TIME_OF_DAY = "time_of_day"
    OUTCOME = "outcome"
    CIRCUMSTANCE = "circumstance"


class SearchAttempt(BaseModel):
    """Tracks each search iteration for visibility in graph execution trace.

    This design makes retry logic visible - you can see exactly what searches
    were tried and why they succeeded or failed.

    Attributes:
        query: The search query string used.
        strategy: The search strategy applied for this attempt.
        num_results: Number of articles returned from search.
        avg_relevance_score: Search quality metric from Tavily.
        timestamp: When this search was executed.
    """

    query: str
    strategy: SearchStrategyType
    num_results: int = 0
    avg_relevance_score: float | None
    timestamp: datetime = Field(default_factory=datetime.now)


class Article(BaseModel):
    """Article retrieved from web search (Tavily API).

    Attributes:
        url: Full URL of the article.
        title: Article headline.
        snippet: Text snippet from search results.
        published_date: Publication date if available.
        source_name: News outlet name.
        content: Full article content if fetched.
        relevance_score: Tavily relevance score.
    """

    url: str
    title: str
    snippet: str
    published_date: date | None = None
    source_name: str | None = None
    content: str | None = None
    relevance_score: float = 0.0


class DetectedEntity(BaseModel):
    """Entity detected via NER (Named Entity Recognition).

    Attributes:
        entity_type: NER type (PERSON, ORG, GPE, etc.).
        value: The extracted entity value.
        context: Surrounding text for provenance.
    """

    entity_type: str
    value: str
    context: str | None = None


class FieldExtraction(BaseModel):
    """Extracted value for a single field with provenance.

    Every extracted field records its source(s), exact quote/context,
    and extraction method for audit trails.

    Attributes:
        field_name: Name of the field being extracted.
        value: Extracted value or None if not found.
        confidence: Confidence level for this extraction.
        sources: List of source URLs.
        source_quotes: Exact quotes from sources.
        extraction_method: Method used (llm, regex, ner).
        llm_reasoning: LLM explanation for flagged cases only.
    """

    field_name: str
    value: str | None
    confidence: ConfidenceLevel
    sources: list[str] = Field(default_factory=list)
    source_quotes: list[str] = Field(default_factory=list)
    extraction_method: str = "llm"
    llm_reasoning: str | None = None


class MergeExtractionResponse(BaseModel):
    """Structured LLM response for multi-field extraction.

    Used as the schema for ChatOpenAI.with_structured_output() in the
    merge node. The LLM returns one FieldExtraction per requested field.

    Attributes:
        extractions: List of FieldExtraction objects, one per field.
    """

    extractions: list[FieldExtraction]


class ValidationResult(BaseModel):
    """Result of validating an article against incident anchors.

    Verifies each article describes the same incident as the database
    record using binary pass/fail checks. Pass criteria: article passes
    if date_match AND location_match are both True.

    Attributes:
        article: The original retrieved article.
        date_match: Whether article date is within ±3 days of incident_date.
        location_match: Whether location matches via string similarity or geocoding.
        victim_name_match: Whether victim name matches (None if unavailable).
        passed: True if date AND location match (computed from above).
    """

    article: Article
    date_match: bool = False
    location_match: bool = False
    victim_name_match: bool | None = None
    passed: bool = False


class EnrichmentState(BaseModel):
    """Complete state for enrichment pipeline.

    Tracks all data as a record moves through the pipeline:
    Extract → Coordinator → Search → Coordinator → Validate →
    Coordinator → Merge → Coordinator → Complete | Escalate

    The Coordinator orchestrates retry logic and escalation routing
    based on intermediate results from each node.

    Attributes:
        incident_id: Unique identifier for the incident record.
        dataset_type: Which TJI dataset this record belongs to.

        officer_name: Officer name from database (may be NULL).
        civilian_name: Civilian name from database (may be NULL).
        incident_date: Date of the incident.
        location: City/county where incident occurred.
        severity: Outcome severity (fatal, injured, etc.).

        search_attempts: History of all search attempts for audit.
        retrieved_articles: Current set of articles from latest search.

        validation_results: Articles validated against incident anchors.

        extracted_fields: Only enriched/updated fields with provenance.
        conflicting_fields: Field names with conflicts (for escalation).

        retry_count: Number of retry attempts made.
        max_retries: Maximum retries before escalation (default 3).
        next_strategy: Next search strategy to try on retry.
        current_stage: Which node just ran (set by processing nodes).
        next_stage: Where to route next (set by coordinator).

        escalation_reason: Why record was ESCALATE if applicable.
        requires_human_review: Whether human review is needed.

        output_file_path: Path to output CSV for human review.
        reasoning_summary: Brief summary for flagged items.
        cost_usd: Cumulative API cost for this record.
        error_message: Error details if processing failed.
    """

    # Record identification
    incident_id: str
    dataset_type: DatasetType

    # Original incident data (from Extract Node)
    officer_name: str | None = None
    civilian_name: str | None = None
    incident_date: date | None = None
    location: str | None = None
    severity: str | None = None

    # Search tracking (Search Node)
    search_attempts: list[SearchAttempt] = Field(default_factory=list)
    retrieved_articles: list[Article] = Field(default_factory=list)

    # Validation results (Validate Node)
    validation_results: list[ValidationResult] = Field(default_factory=list)

    # Merge outputs (Merge Node)
    extracted_fields: list[FieldExtraction] = Field(default_factory=list)
    conflicting_fields: list[str] | None = None

    # Coordinator control
    retry_count: int = 0
    max_retries: int = 3
    next_strategy: SearchStrategyType = SearchStrategyType.EXACT_MATCH
    current_stage: PipelineStage = PipelineStage.EXTRACT
    next_stage: PipelineStage = PipelineStage.EXTRACT

    # Escalation
    escalation_reason: EscalationReason | None = None
    requires_human_review: bool = False

    # Output (for human review)
    output_file_path: str | None = None
    reasoning_summary: str | None = None

    # Pipeline metadata
    cost_usd: float = 0.0
    error_message: str | None = None


# Mapping for MediaFeatureField and EnrichmentState
# (only fields that exist on EnrichmentState)
FIELD_TO_STATE_ATTR = {
    MediaFeatureField.OFFICER_NAME: "officer_name",
    MediaFeatureField.CIVILIAN_NAME: "civilian_name",
}
