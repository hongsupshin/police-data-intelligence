"""State models for the enrichment pipeline.

Defines Pydantic models for tracking state across nodes in the
LangGraph-based enrichment workflow. The pipeline consists of:
Extract → Search → Validate → Merge → Coordinator
"""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    """Dataset identifier for conditional field mapping.

    The two TJI datasets use different field names for the same concepts,
    requiring dataset-aware extraction.
    """

    CIVILIANS_SHOT = "civilians_shot"
    OFFICERS_SHOT = "officers_shot"


class SearchStrategyType(str, Enum):
    """Search strategy for progressive query refinement.

    The Coordinator implements an escalating retry strategy:
    - exact_match: All fields, exact date (initial)
    - temporal_expanded: Date range ±2 days (after < 2 results)
    - entity_dropped: Drop officer name, keep location + date (after retry 1 fails)
    """

    EXACT_MATCH = "exact_match"
    TEMPORAL_EXPANDED = "temporal_expanded"
    ENTITY_DROPPED = "entity_dropped"


class PipelineStage(str, Enum):
    """Current stage in the enrichment pipeline."""

    EXTRACT = "extract"
    SEARCH = "search"
    VALIDATE = "validate"
    MERGE = "merge"
    COMPLETE = "complete"
    ESCALATED = "escalated"


class ConfidenceLevel(str, Enum):
    """Per-field confidence classification.

    Based on source agreement and evidence quality:
    - HIGH: Multiple sources agree
    - MEDIUM: Single source or unclear context
    - LOW: Weak evidence or soft anchor match
    - NONE: No information found
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class EscalationReason(str, Enum):
    """Reasons for escalating to human review.

    Escalation triggers include:
    - Confidence below threshold
    - Conflicting information across sources
    - Composite information requiring merge review
    - Overwriting existing non-null values
    - Soft anchor match (date ±3-7 days)
    - Max retries reached without sufficient data
    """

    CONFLICT = "conflict"
    COMPOSITE = "composite"
    LOW_CONFIDENCE = "low_confidence"
    OVERWRITE = "overwrite"
    SOFT_ANCHOR = "soft_anchor"
    MAX_RETRIES = "max_retries"
    INSUFFICIENT_SOURCES = "insufficient_sources"


class MediaFeatureField(str, Enum):
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
    avg_relevance_score: float = 0.0
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


class ValidatedArticle(BaseModel):
    """Article that passed anchor matching validation.

    Attributes:
        article: The original retrieved article.
        anchor_match_date: Whether date matched within tolerance.
        anchor_match_location: Whether location matched.
        date_offset_days: Days difference from incident date.
        extractions: Fields extracted from this article.
        entities: NER entities detected in this article.
    """

    article: Article
    anchor_match_date: bool = False
    anchor_match_location: bool = False
    date_offset_days: int | None = None
    extractions: list[FieldExtraction] = Field(default_factory=list)
    entities: list[DetectedEntity] = Field(default_factory=list)


class MergedField(BaseModel):
    """Result of merging extractions from multiple sources.

    Attributes:
        field_name: Name of the merged field.
        suggested_value: Final suggested value after merge.
        confidence: Overall confidence after merge.
        source_count: Number of sources contributing.
        has_conflict: Whether sources disagreed.
        conflicting_values: Values that disagreed if conflict exists.
        llm_reasoning: LLM explanation for conflict resolution.
    """

    field_name: str
    suggested_value: str | None
    confidence: ConfidenceLevel
    source_count: int = 0
    has_conflict: bool = False
    conflicting_values: list[str] = Field(default_factory=list)
    llm_reasoning: str | None = None


class EnrichmentState(BaseModel):
    """Complete state for enrichment pipeline.

    Tracks all data as a record moves through the 5-node pipeline:
    Extract → Search → Validate → Merge → Coordinator

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

        validated_articles: Articles that passed anchor matching.

        merged_fields: Final merged extractions ready for output.

        retry_count: Number of retry attempts made.
        max_retries: Maximum retries before escalation (default 3).
        next_strategy: Next search strategy to try on retry.
        current_stage: Current pipeline stage.

        escalation_reason: Why record was escalated if applicable.
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
    validated_articles: list[ValidatedArticle] = Field(default_factory=list)

    # Merge results (Merge Node)
    merged_fields: list[MergedField] = Field(default_factory=list)

    # Coordinator control
    retry_count: int = 0
    max_retries: int = 3
    next_strategy: SearchStrategyType = SearchStrategyType.EXACT_MATCH
    current_stage: PipelineStage = PipelineStage.EXTRACT

    # Escalation
    escalation_reason: EscalationReason | None = None
    requires_human_review: bool = False

    # Output (for human review)
    output_file_path: str | None = None
    reasoning_summary: str | None = None

    # Pipeline metadata
    cost_usd: float = 0.0
    error_message: str | None = None
