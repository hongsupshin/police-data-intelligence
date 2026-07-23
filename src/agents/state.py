"""State models for the enrichment pipeline.

Defines Pydantic models for tracking state across nodes in the
LangGraph-based enrichment workflow. The pipeline consists of:
Load → Coordinator → Search → Coordinator → Validate →
Coordinator → Synthesize → Coordinator → Complete | Escalate
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
    - name_partial: Drop officer name, keep civilian name + month-year
      (terminal rung — the victim name is never stripped)
    """

    EXACT_MATCH = "exact_match"
    TEMPORAL_EXPANDED = "temporal_expanded"
    NAME_PARTIAL = "name_partial"


class PipelineStage(StrEnum):
    """Current stage in the enrichment pipeline."""

    LOAD = "load"
    SEARCH = "search"
    VALIDATE = "validate"
    SYNTHESIZE = "synthesize"
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
    - Soft anchor match (reserved; defined but not currently emitted)
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
    IRRELEVANT_SOURCES = "irrelevant_sources"


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


class ConflictType(StrEnum):
    """Type of field conflict detected during merge.

    Two conflict scenarios exist:
    - articles_disagree: Extracted values differ across articles beyond
      fuzzy-match tolerance.
    - reference_mismatch: Articles agree with each other but disagree
      with the existing database value.
    """

    ARTICLES_DISAGREE = "articles_disagree"
    REFERENCE_MISMATCH = "reference_mismatch"


class FieldConflict(BaseModel):
    """Details of a field conflict detected during merge.

    Attributes:
        field_name: Name of the conflicting field.
        conflict_type: Type of conflict detected.
        values: Distinct values found across sources.
        sources: Source URLs for each value (parallel to values).
        reference_value: Database value when conflict_type is
            REFERENCE_MISMATCH, None otherwise.
    """

    field_name: str
    conflict_type: ConflictType
    values: list[str]
    sources: list[list[str]]
    reference_value: str | None = None


class MergeExtractionResponse(BaseModel):
    """Structured LLM response for multi-field extraction.

    Used as the schema for ChatAnthropic.with_structured_output() in the
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
        date_match: Whether article date is within the configured date-proximity
            tolerance (``Settings.date_proximity_days``, default ±5 days).
        location_match: Whether location matches via string similarity or geocoding.
        victim_name_match: Whether victim name matches (None if unavailable).
        passed: Whether the article passed validation via the three-tier rule
            (dated: date_match AND location_match; undated with a name:
            location_match AND victim_name_match; undated without a name:
            location_match alone).
    """

    article: Article
    date_match: bool = False
    location_match: bool = False
    victim_name_match: bool | None = None
    passed: bool = False


class RaceTaxonomyFlag(BaseModel):
    """Where an extracted race sits relative to TJI's coarse race scheme.

    TJI codes race in four buckets (WHITE/BLACK/HISPANIC/OTHER) with no Asian
    category and no nationalities. This records the bucket a source-stated race
    maps to for comparison, plus whether the source was *more specific* than the
    bucket (a taxonomy divergence) — a deterministic annotation for human review
    and the aggregate "TX coarser than the source" finding. The granular value
    is preserved on the FieldExtraction; this never replaces it.

    Attributes:
        extracted: Raw extracted race string (e.g. "Asian", "Honduran").
        tji_bucket: The TJI bucket it maps to (white/black/hispanic/other).
        divergence_type: none | nationality_as_ethnicity |
            race_absent_from_scheme | uncategorized.
        diverges: True when the source is more specific than the bucket.
    """

    extracted: str
    tji_bucket: str
    divergence_type: str
    diverges: bool


class ConflictAnnotation(BaseModel):
    """Advisory LLM triage note for a record with deep flagged conflicts.

    When articles disagree on identity/structured fields (name/age/weapon/
    location/race) or a multi-subject source blends people, an LLM reads the
    disagreeing quotes and writes a short note explaining *why* they differ and
    what the reviewer should check. It is advisory only — nothing is committed
    or overwritten — so it can only aid the human-review default, never corrupt
    data. Produced at synthesize; written to the output JSON for review.

    Attributes:
        note: The reviewer-facing triage note (cites quotes, flags uncertainty,
            never asserts a person's race or any fact the articles don't state).
    """

    note: str


class EnrichmentState(BaseModel):
    """Complete state for enrichment pipeline.

    Tracks all data as a record moves through the pipeline:
    Load → Coordinator → Search → Coordinator → Validate →
    Coordinator → Synthesize → Coordinator → Complete | Escalate

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
        civilian_outcome: Suspect/civilian outcome for officers_shot
            (killed/injured/not harmed/unknown); None for civilians_shot.
        civilian_age: Victim age from database (civilians_shot only); used as an
            extraction anchor to disambiguate multi-subject sources. None otherwise.
        civilian_gender: Victim gender from database (civilians_shot only); a
            secondary extraction anchor when the victim name is absent. None otherwise.

        search_attempts: History of all search attempts for audit.
        retrieved_articles: Current set of articles from latest search.

        validation_results: Articles validated against incident anchors.
        validation_failure_summary: Per-check failure counts from the final
            validate pass (total, passed, excluded, date_fail, location_fail,
            name_fail). None until the validate node runs.

        extracted_fields: Only enriched/updated fields with provenance.
        conflicting_fields: Field names with conflicts (for escalation).
        civilian_race_taxonomy: Taxonomy annotation for a committed
            civilian_race (any dataset) — the TJI bucket it maps to and whether
            the source was more specific than that bucket. None when no race
            was committed.
        conflict_annotation: Advisory LLM triage note for a record with deep
            flagged conflicts (any dataset). None when not generated.
        judge_failures: Fail-open events from the LLM judges, one entry per
            failure ("<judge>: <error>"). Empty when every enabled judge ran.
            Persisted to the terminal JSON reports so a skipped check is
            auditable from saved artifacts, not just run-time warnings.

        retry_count: Number of retry attempts made.
        max_retries: Maximum retries before escalation (default 3).
        next_strategy: Next search strategy to try on retry.
        current_stage: Which node just ran (set by processing nodes).
        next_stage: Where to route next (set by coordinator).

        escalation_reason: Why record was ESCALATE if applicable.
        requires_human_review: Whether human review is needed.

        output_file_path: Path to output CSV for human review.
        outcome_summary: Brief summary for flagged items.
        cost_usd: Cumulative API cost for this record.
        error_message: Error details if processing failed.
    """

    # Record identification
    incident_id: str
    dataset_type: DatasetType

    # Original incident data (from Load Node)
    officer_name: str | None = None
    civilian_name: str | None = None
    incident_date: date | None = None
    location: str | None = None
    severity: str | None = None
    civilian_outcome: str | None = None
    civilian_age: int | None = None
    civilian_gender: str | None = None

    # Search tracking (Search Node)
    search_attempts: list[SearchAttempt] = Field(default_factory=list)
    retrieved_articles: list[Article] = Field(default_factory=list)

    # Validation results (Validate Node)
    validation_results: list[ValidationResult] = Field(default_factory=list)
    validation_failure_summary: dict[str, int] | None = None

    # Synthesize outputs (Synthesize Node)
    extracted_fields: list[FieldExtraction] = Field(default_factory=list)
    conflicting_fields: list[FieldConflict] | None = None
    civilian_race_taxonomy: RaceTaxonomyFlag | None = None
    conflict_annotation: ConflictAnnotation | None = None
    judge_failures: list[str] = Field(default_factory=list)

    # Coordinator control
    retry_count: int = 0
    max_retries: int = 3
    next_strategy: SearchStrategyType = SearchStrategyType.EXACT_MATCH
    current_stage: PipelineStage = PipelineStage.LOAD
    next_stage: PipelineStage = PipelineStage.LOAD

    # Escalation
    escalation_reason: EscalationReason | None = None
    requires_human_review: bool = False
    relevance_vetoed: bool = False

    # Output (for human review)
    output_file_path: str | None = None
    outcome_summary: str | None = None

    # Pipeline metadata
    cost_usd: float = 0.0
    error_message: str | None = None


# Mapping for MediaFeatureField and EnrichmentState
# (only fields that exist on EnrichmentState)
FIELD_TO_STATE_ATTR = {
    MediaFeatureField.OFFICER_NAME: "officer_name",
    MediaFeatureField.CIVILIAN_NAME: "civilian_name",
}
