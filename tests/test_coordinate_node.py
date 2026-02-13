"""Tests for the Coordinator Node.

Tests cover four helper functions (check_extract_results, retry_helper,
check_search_results, check_validate_results, check_merge_results)
that gate progression between pipeline stages.
"""

from datetime import date

import pytest

from src.agents.coordinate_node import (
    STRATEGY_ORDER,
    check_extract_results,
    check_merge_results,
    check_search_results,
    check_validate_results,
    coordinate_node,
    retry_helper,
)
from src.agents.state import (
    Article,
    ConfidenceLevel,
    DatasetType,
    EnrichmentState,
    EscalationReason,
    FieldExtraction,
    MediaFeatureField,
    PipelineStage,
    SearchAttempt,
    SearchStrategyType,
    ValidationResult,
)

# --- Fixtures ---


@pytest.fixture()
def base_state() -> EnrichmentState:
    """State after extract with all identity fields populated."""
    return EnrichmentState(
        incident_id="142",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        location="Houston",
        incident_date=date(2018, 3, 15),
        officer_name="James Rodriguez",
        civilian_name="John Doe",
        severity="fatal",
        current_stage=PipelineStage.EXTRACT,
        next_strategy=SearchStrategyType.EXACT_MATCH,
    )


@pytest.fixture()
def search_state(base_state: EnrichmentState) -> EnrichmentState:
    """State after search with retrieved articles and search attempt."""
    state = base_state.model_copy()
    state.current_stage = PipelineStage.SEARCH
    state.next_strategy = SearchStrategyType.EXACT_MATCH
    state.retrieved_articles = [
        Article(
            url="https://example.com/article1",
            title="Houston officer James Rodriguez involved in shooting of John Doe",
            snippet="A Houston police officer fatally shot John Doe during a traffic stop on March 15.",
            content="A Houston police officer identified as James Rodriguez fatally shot John Doe, 34, during a traffic stop on the city's east side on March 15, 2018. Witnesses say the encounter escalated quickly after Doe exited his vehicle.",
            source_name="CBS",
            relevance_score=0.9,
            published_date=date(2018, 3, 15),
        ),
        Article(
            url="https://example.com/article2",
            title="Houston fatal police shooting, victim is John Doe",
            snippet="Police in Houston, TX confirmed a fatal officer-involved shooting on March 14.",
            content="Police in Houston, TX confirmed a fatal officer-involved shooting near downtown on Wednesday. The victim was identified as John Doe. Officials have not yet released the name of the officer involved.",
            source_name="NBC",
            relevance_score=0.7,
            published_date=date(2018, 3, 14),
        ),
    ]
    state.search_attempts = [
        SearchAttempt(
            query="Houston police shooting James Rodriguez, John Doe, March 14 2018",
            strategy=SearchStrategyType.EXACT_MATCH,
            num_results=2,
            avg_relevance_score=0.8,
        )
    ]
    return state


@pytest.fixture()
def validate_state(search_state: EnrichmentState) -> EnrichmentState:
    """State after validation with one passed and one failed article."""
    state = search_state.model_copy(deep=True)
    state.current_stage = PipelineStage.VALIDATE
    article1 = state.retrieved_articles[0]
    article2 = state.retrieved_articles[1]
    state.validation_results = [
        ValidationResult(
            article=article1,
            date_match=True,
            location_match=True,
            victim_name_match=True,
            passed=True,
        ),
        ValidationResult(
            article=article2,
            date_match=True,
            location_match=False,
            victim_name_match=True,
            passed=False,
        ),
    ]
    return state


@pytest.fixture()
def merge_state(validate_state: EnrichmentState) -> EnrichmentState:
    """State after merge with extracted fields and no conflicts."""
    state = validate_state.model_copy()
    state.current_stage = PipelineStage.MERGE
    state.extracted_fields = [
        FieldExtraction(
            field_name="weapon",
            value="handgun",
            confidence=ConfidenceLevel.HIGH,
            sources=["https://example.com/article1"],
            source_quotes=["the victim used a handgun"],
            llm_reasoning="Weapon type mentioned in article.",
        ),
        FieldExtraction(
            field_name="civilian_age",
            value="34",
            confidence=ConfidenceLevel.HIGH,
            sources=["https://example.com/article1"],
            source_quotes=["John Doe, 34"],
            llm_reasoning="Age mentioned alongside name.",
        ),
    ]
    state.conflicting_fields = []
    return state


# --- check_extract_results tests ---


def test_check_extract_results_happy_path(base_state: EnrichmentState) -> None:
    """All identity fields present, proceed to SEARCH."""
    extract_state = base_state.model_copy()
    state = check_extract_results(extract_state)
    assert state.next_stage == PipelineStage.SEARCH


def test_check_extract_results_error(base_state: EnrichmentState) -> None:
    """Extract error message triggers ESCALATE with EXTRACTION_ERROR."""
    extract_state = base_state.model_copy()
    extract_state.error_message = "Extract failed..."
    state = check_extract_results(extract_state)
    assert state.escalation_reason == EscalationReason.EXTRACTION_ERROR
    assert state.requires_human_review
    assert state.next_stage == PipelineStage.ESCALATE


def test_check_extract_results_all_missing(base_state: EnrichmentState) -> None:
    """All identity fields missing triggers ESCALATE with INSUFFICIENT_SOURCES."""
    extract_state = base_state.model_copy()
    extract_state.civilian_name = None
    extract_state.officer_name = None
    extract_state.incident_date = None
    state = check_extract_results(extract_state)
    assert state.escalation_reason == EscalationReason.INSUFFICIENT_SOURCES
    assert state.requires_human_review
    assert state.next_stage == PipelineStage.ESCALATE


def test_check_extract_results_partial_missing(base_state: EnrichmentState) -> None:
    """At least one identity field present, proceed to SEARCH."""
    extract_state = base_state.model_copy()
    extract_state.civilian_name = None
    state = check_extract_results(extract_state)
    assert state.next_stage == PipelineStage.SEARCH


# --- retry_helper tests ---


def test_retry_helper_happy_path(search_state: EnrichmentState) -> None:
    """Strategies remaining: advance strategy, clear articles, stay in SEARCH."""
    state = retry_helper(search_state.model_copy())
    assert state.retry_count == 1
    assert state.next_strategy == STRATEGY_ORDER[1]
    assert state.next_stage == PipelineStage.SEARCH
    assert state.retrieved_articles == []


def test_retry_helper_exhausted_strategies(search_state: EnrichmentState) -> None:
    """No strategies remaining: escalate with MAX_RETRIES."""
    updated_search_state = search_state.model_copy()
    updated_search_state.next_strategy = STRATEGY_ORDER[-1]
    state = retry_helper(updated_search_state)
    assert state.next_stage == PipelineStage.ESCALATE
    assert state.escalation_reason == EscalationReason.MAX_RETRIES
    assert state.requires_human_review


# --- check_search_results tests ---


def test_check_search_results_happy_path(search_state: EnrichmentState) -> None:
    """Good relevance score, proceed to VALIDATE."""
    state = check_search_results(search_state.model_copy())
    assert state.next_stage == PipelineStage.VALIDATE


def test_check_search_results_exhausted_retries(search_state: EnrichmentState) -> None:
    """Retry count exceeds max, escalate with MAX_RETRIES."""
    updated_search_state = search_state.model_copy()
    updated_search_state.retry_count = updated_search_state.max_retries + 1
    state = check_search_results(updated_search_state)
    assert state.next_stage == PipelineStage.ESCALATE
    assert state.escalation_reason == EscalationReason.MAX_RETRIES
    assert state.requires_human_review


def test_check_search_results_low_score_retry(search_state: EnrichmentState) -> None:
    """Low relevance score triggers retry via retry_helper."""
    updated_search_state = search_state.model_copy()
    updated_search_state.search_attempts[-1].avg_relevance_score = 0.1
    state = check_search_results(updated_search_state)
    assert state.retry_count == 1
    assert state.next_stage == PipelineStage.SEARCH


def test_check_search_results_error_retry(search_state: EnrichmentState) -> None:
    """Search error message triggers retry via retry_helper."""
    updated_search_state = search_state.model_copy()
    updated_search_state.error_message = "Search failed..."
    state = check_search_results(updated_search_state)
    assert state.retry_count == 1
    assert state.next_stage == PipelineStage.SEARCH


# --- check_validate_results tests ---


def test_check_validate_results_happy_path(validate_state: EnrichmentState) -> None:
    """At least one article passed validation, proceed to MERGE."""
    state = check_validate_results(validate_state)
    assert state.next_stage == PipelineStage.MERGE


def test_check_validate_results_all_failed(validate_state: EnrichmentState) -> None:
    """All articles failed validation, escalate with VALIDATION_ERROR."""
    updated_state = validate_state.model_copy()
    for vr in updated_state.validation_results:
        vr.passed = False
    state = check_validate_results(updated_state)
    assert state.next_stage == PipelineStage.ESCALATE
    assert state.escalation_reason == EscalationReason.VALIDATION_ERROR
    assert state.requires_human_review


def test_check_validate_results_empty(validate_state: EnrichmentState) -> None:
    """No validation results at all, escalate with VALIDATION_ERROR."""
    updated_state = validate_state.model_copy()
    updated_state.validation_results = []
    state = check_validate_results(updated_state)
    assert state.next_stage == PipelineStage.ESCALATE
    assert state.escalation_reason == EscalationReason.VALIDATION_ERROR


# --- check_merge_results tests ---


def test_check_merge_results_happy_path(merge_state: EnrichmentState) -> None:
    """No errors or conflicts, proceed to COMPLETE."""
    state = check_merge_results(merge_state)
    assert state.next_stage == PipelineStage.COMPLETE


def test_check_merge_results_error(merge_state: EnrichmentState) -> None:
    """Merge error message triggers escalation with MERGE_ERROR."""
    updated_state = merge_state.model_copy()
    updated_state.error_message = "Merge failed: LLM timeout"
    state = check_merge_results(updated_state)
    assert state.next_stage == PipelineStage.ESCALATE
    assert state.escalation_reason == EscalationReason.MERGE_ERROR
    assert state.requires_human_review


def test_check_merge_results_conflict(merge_state: EnrichmentState) -> None:
    """Conflicting fields triggers escalation with CONFLICT."""
    updated_state = merge_state.model_copy()
    updated_state.conflicting_fields = [MediaFeatureField.WEAPON]
    state = check_merge_results(updated_state)
    assert state.next_stage == PipelineStage.ESCALATE
    assert state.escalation_reason == EscalationReason.CONFLICT
    assert state.requires_human_review


def test_check_merge_results_empty_extractions(merge_state: EnrichmentState) -> None:
    """No fields extracted triggers escalation with INSUFFICIENT_SOURCES."""
    updated_state = merge_state.model_copy()
    updated_state.extracted_fields = []
    state = check_merge_results(updated_state)
    assert state.next_stage == PipelineStage.ESCALATE
    assert state.escalation_reason == EscalationReason.INSUFFICIENT_SOURCES
    assert state.requires_human_review


# --- coordinate_node tests ---


def test_coordinate_node_extract_stage(base_state: EnrichmentState) -> None:
    """Dispatches to check_extract_results when stage is EXTRACT."""
    state = coordinate_node(base_state.model_copy())
    assert state.next_stage == PipelineStage.SEARCH


def test_coordinate_node_search_stage(search_state: EnrichmentState) -> None:
    """Dispatches to check_search_results when stage is SEARCH."""
    state = coordinate_node(search_state.model_copy())
    assert state.next_stage == PipelineStage.VALIDATE


def test_coordinate_node_validate_stage(validate_state: EnrichmentState) -> None:
    """Dispatches to check_validate_results when stage is VALIDATE."""
    state = coordinate_node(validate_state)
    assert state.next_stage == PipelineStage.MERGE


def test_coordinate_node_merge_stage(merge_state: EnrichmentState) -> None:
    """Dispatches to check_merge_results when stage is MERGE."""
    state = coordinate_node(merge_state)
    assert state.next_stage == PipelineStage.COMPLETE


def test_coordinate_node_unexpected_stage(base_state: EnrichmentState) -> None:
    """Unexpected stage (COMPLETE) returns state unchanged."""
    updated_state = base_state.model_copy()
    updated_state.current_stage = PipelineStage.COMPLETE
    state = coordinate_node(updated_state)
    assert state.current_stage == PipelineStage.COMPLETE


def test_coordinate_node_escalate_stage(base_state: EnrichmentState) -> None:
    """ESCALATE stage returns state unchanged."""
    updated_state = base_state.model_copy()
    updated_state.current_stage = PipelineStage.ESCALATE
    state = coordinate_node(updated_state)
    assert state.current_stage == PipelineStage.ESCALATE
