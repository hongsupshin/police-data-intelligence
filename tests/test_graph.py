"""Tests for graph wiring, routing, terminal nodes, and end-to-end paths.

Unit tests verify the router edge function, terminal nodes, and graph
compilation. Integration tests run the compiled graph with patched node
functions to verify routing through happy and escalation paths.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from src.agents.graph import (
    build_graph,
    complete_node,
    escalate_node,
    route_after_coordinator,
)
from src.agents.state import (
    Article,
    ConfidenceLevel,
    DatasetType,
    EnrichmentState,
    FieldExtraction,
    PipelineStage,
    SearchAttempt,
    SearchStrategyType,
    ValidationResult,
)
from src.config import Settings

# --- Fake node helpers for integration tests ---

_STUB_ARTICLE = Article(url="https://stub.com", title="stub", snippet="stub")


def _fake_extract(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.EXTRACT
    return state


def _fake_search(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.SEARCH
    state.search_attempts = [
        SearchAttempt(
            query="stub query",
            strategy=state.next_strategy,
            num_results=1,
            avg_relevance_score=0.9,
        )
    ]
    state.retrieved_articles = [_STUB_ARTICLE]
    return state


def _fake_validate(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.VALIDATE
    state.validation_results = [ValidationResult(article=_STUB_ARTICLE, passed=True)]
    return state


def _fake_merge(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.MERGE
    state.extracted_fields = [
        FieldExtraction(
            field_name="weapon",
            value="handgun",
            confidence=ConfidenceLevel.HIGH,
        )
    ]
    state.conflicting_fields = []
    return state


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


# --- Unit tests ---


@pytest.mark.parametrize(
    "next_stage",
    [
        PipelineStage.SEARCH,
        PipelineStage.VALIDATE,
        PipelineStage.MERGE,
        PipelineStage.COMPLETE,
        PipelineStage.ESCALATE,
    ],
)
def test_route_after_coordinator_next_stage(
    base_state: EnrichmentState, next_stage: PipelineStage
) -> None:
    """Valid next_stage values route to the matching node name."""
    state = base_state.model_copy()
    state.next_stage = next_stage
    assert route_after_coordinator(state) == next_stage.value


def test_route_after_coordinator_fallback(base_state: EnrichmentState) -> None:
    """Unexpected next_stage (EXTRACT) falls back to escalate."""
    state = base_state.model_copy()
    state.next_stage = PipelineStage.EXTRACT
    assert route_after_coordinator(state) == "escalate"


def test_complete_node(base_state: EnrichmentState, tmp_path: Path) -> None:
    """Complete node sets COMPLETE stage and no human review."""
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(tmp_path))}}
    )
    state = complete_node(base_state.model_copy(), config)
    assert state.current_stage == PipelineStage.COMPLETE
    assert not state.requires_human_review

    file_path = tmp_path / f"{state.dataset_type}_{state.incident_id}_complete.json"
    assert file_path.exists()

    data = json.loads(file_path.read_text())
    assert data["incident_id"] == state.incident_id
    assert data["dataset_type"] == state.dataset_type
    assert data["search_strategy"] == state.next_strategy
    assert (
        data["outcome_summary"]
        == f"Enriched {len(state.extracted_fields)} fields for incident {state.incident_id} ({state.dataset_type})"
    )


def test_escalate_node(base_state: EnrichmentState, tmp_path: Path) -> None:
    """Escalate node sets ESCALATE stage and requires human review."""
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(tmp_path))}}
    )
    state = escalate_node(base_state.model_copy(), config)
    assert state.current_stage == PipelineStage.ESCALATE
    assert state.requires_human_review

    file_path = tmp_path / f"{state.dataset_type}_{state.incident_id}_escalate.json"
    assert file_path.exists()

    data = json.loads(file_path.read_text())
    assert data["incident_id"] == state.incident_id
    assert data["dataset_type"] == state.dataset_type
    assert data["search_strategy"] == state.next_strategy
    assert (
        data["outcome_summary"]
        == f"Escalated incident {state.incident_id}: {state.escalation_reason} after {state.retry_count} retries"
    )


def test_build_graph_none() -> None:
    """Graph compiles without checkpointer and registers all 7 nodes."""
    compiled_graph = build_graph(None)
    assert isinstance(compiled_graph, CompiledStateGraph)
    node_names = [
        "extract",
        "search",
        "validate",
        "merge",
        "complete",
        "escalate",
        "coordinate",
    ]
    for node_name in node_names:
        assert node_name in compiled_graph.nodes


# --- Integration tests ---


@pytest.mark.integration
@patch("src.agents.graph.validate_node")
@patch("src.agents.graph.search_node")
@patch("src.agents.graph.merge_node")
@patch("src.agents.graph.extract_node")
def test_happy_path(
    mock_extract,
    mock_merge,
    mock_search,
    mock_validate,
    base_state: EnrichmentState,
) -> None:
    """Happy path: extract → search → validate → merge → complete."""
    mock_extract.side_effect = _fake_extract
    mock_search.side_effect = _fake_search
    mock_validate.side_effect = _fake_validate
    mock_merge.side_effect = _fake_merge

    graph = build_graph(None)
    config = RunnableConfig({"configurable": {"settings": Settings()}})
    result = graph.invoke(base_state, config)

    assert result["current_stage"] == PipelineStage.COMPLETE
    assert len(result["search_attempts"]) > 0
    assert len(result["validation_results"]) > 0
    assert len(result["extracted_fields"]) > 0
    assert not result["requires_human_review"]


@pytest.mark.integration
@patch("src.agents.graph.extract_node")
def test_escalate_after_extract(
    mock_extract,
    base_state: EnrichmentState,
) -> None:
    """Escalate when extract produces no identity fields."""

    def _fake_extract_empty(state: EnrichmentState) -> EnrichmentState:
        state.current_stage = PipelineStage.EXTRACT
        state.civilian_name = None
        state.officer_name = None
        state.incident_date = None
        return state

    mock_extract.side_effect = _fake_extract_empty

    graph = build_graph(None)
    config = RunnableConfig({"configurable": {"settings": Settings()}})
    result = graph.invoke(base_state, config)

    assert result["current_stage"] == PipelineStage.ESCALATE
    assert result["requires_human_review"]


@pytest.mark.integration
@patch("src.agents.graph.search_node")
@patch("src.agents.graph.extract_node")
def test_escalate_after_search(
    mock_extract,
    mock_search,
    base_state: EnrichmentState,
) -> None:
    """Escalate when search exhausts all retry strategies."""

    def _fake_search_low_score(state: EnrichmentState) -> EnrichmentState:
        state.current_stage = PipelineStage.SEARCH
        state.search_attempts = [
            SearchAttempt(
                query="stub",
                strategy=state.next_strategy,
                num_results=0,
                avg_relevance_score=0.1,
            )
        ]
        return state

    mock_extract.side_effect = _fake_extract
    mock_search.side_effect = _fake_search_low_score

    graph = build_graph(None)
    config = RunnableConfig({"configurable": {"settings": Settings()}})
    result = graph.invoke(base_state, config)

    assert result["current_stage"] == PipelineStage.ESCALATE
    assert result["requires_human_review"]


@pytest.mark.integration
@patch("src.agents.graph.validate_node")
@patch("src.agents.graph.search_node")
@patch("src.agents.graph.extract_node")
def test_escalate_after_validate(
    mock_extract,
    mock_search,
    mock_validate,
    base_state: EnrichmentState,
) -> None:
    """Escalate when all articles fail validation."""

    def _fake_validate_fail(state: EnrichmentState) -> EnrichmentState:
        state.current_stage = PipelineStage.VALIDATE
        state.validation_results = [
            ValidationResult(article=_STUB_ARTICLE, passed=False)
        ]
        return state

    mock_extract.side_effect = _fake_extract
    mock_search.side_effect = _fake_search
    mock_validate.side_effect = _fake_validate_fail

    graph = build_graph(None)
    config = RunnableConfig({"configurable": {"settings": Settings()}})
    result = graph.invoke(base_state, config)

    assert result["current_stage"] == PipelineStage.ESCALATE
    assert result["requires_human_review"]


@pytest.mark.integration
@patch("src.agents.graph.validate_node")
@patch("src.agents.graph.search_node")
@patch("src.agents.graph.merge_node")
@patch("src.agents.graph.extract_node")
def test_escalate_after_merge(
    mock_extract,
    mock_merge,
    mock_search,
    mock_validate,
    base_state: EnrichmentState,
) -> None:
    """Escalate when merge detects conflicting fields."""

    def _fake_merge_conflict(state: EnrichmentState) -> EnrichmentState:
        state.current_stage = PipelineStage.MERGE
        state.extracted_fields = []
        state.conflicting_fields = ["weapon"]
        return state

    mock_extract.side_effect = _fake_extract
    mock_search.side_effect = _fake_search
    mock_validate.side_effect = _fake_validate
    mock_merge.side_effect = _fake_merge_conflict

    graph = build_graph(None)
    config = RunnableConfig({"configurable": {"settings": Settings()}})
    result = graph.invoke(base_state, config)

    assert result["current_stage"] == PipelineStage.ESCALATE
    assert result["requires_human_review"]


def test_nested_serialization(base_state: EnrichmentState, tmp_path: Path) -> None:
    """Nested Pydantic models serialize correctly in complete node JSON output."""
    state = base_state.model_copy()
    state.extracted_fields = [
        FieldExtraction(
            field_name="weapon",
            value="handgun",
            confidence=ConfidenceLevel.HIGH,
        )
    ]
    state.validation_results = [ValidationResult(article=_STUB_ARTICLE, passed=True)]
    state.current_stage = PipelineStage.MERGE
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(tmp_path))}}
    )
    updated_stage = complete_node(state, config)

    file_path = Path(updated_stage.output_file_path)
    assert file_path.exists()

    data = json.loads(file_path.read_text())
    assert data["extracted_fields"][0]["field_name"] == "weapon"
    assert data["extracted_fields"][0]["value"] == "handgun"
    assert data["extracted_fields"][0]["confidence"] == "high"
    assert data["validation_results"][0]["article"] == _STUB_ARTICLE.model_dump()
    assert data["validation_results"][0]["passed"]


def test_complete_node_creates_nested_directory(
    base_state: EnrichmentState, tmp_path: Path
) -> None:
    """Complete node creates output directory if it doesn't exist."""
    nested_dir = tmp_path / "nested" / "subdir"
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(nested_dir))}}
    )
    state = complete_node(base_state.model_copy(), config)

    assert nested_dir.exists()
    assert Path(state.output_file_path).exists()


def test_escalate_node_creates_nested_directory(
    base_state: EnrichmentState, tmp_path: Path
) -> None:
    """Escalate node creates output directory if it doesn't exist."""
    nested_dir = tmp_path / "deep" / "path"
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(nested_dir))}}
    )
    state = escalate_node(base_state.model_copy(), config)

    assert nested_dir.exists()
    assert Path(state.output_file_path).exists()


def test_complete_node_output_file_path(
    base_state: EnrichmentState, tmp_path: Path
) -> None:
    """Complete node sets output_file_path to the correct full path."""
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(tmp_path))}}
    )
    state = complete_node(base_state.model_copy(), config)

    expected = str(
        tmp_path / f"{state.dataset_type}_{state.incident_id}_complete.json"
    )
    assert state.output_file_path == expected


def test_escalate_node_output_file_path(
    base_state: EnrichmentState, tmp_path: Path
) -> None:
    """Escalate node sets output_file_path to the correct full path."""
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(tmp_path))}}
    )
    state = escalate_node(base_state.model_copy(), config)

    expected = str(
        tmp_path / f"{state.dataset_type}_{state.incident_id}_escalate.json"
    )
    assert state.output_file_path == expected


def test_complete_node_json_keys(
    base_state: EnrichmentState, tmp_path: Path
) -> None:
    """Complete node JSON output contains all expected keys."""
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(tmp_path))}}
    )
    state = complete_node(base_state.model_copy(), config)

    data = json.loads(Path(state.output_file_path).read_text())
    expected_keys = {
        "incident_id",
        "dataset_type",
        "extracted_fields",
        "validation_results",
        "search_strategy",
        "retry_count",
        "outcome_summary",
    }
    assert set(data.keys()) == expected_keys


def test_escalate_node_json_keys(
    base_state: EnrichmentState, tmp_path: Path
) -> None:
    """Escalate node JSON output contains all expected keys."""
    config = RunnableConfig(
        {"configurable": {"settings": Settings(output_dir=str(tmp_path))}}
    )
    state = escalate_node(base_state.model_copy(), config)

    data = json.loads(Path(state.output_file_path).read_text())
    expected_keys = {
        "incident_id",
        "dataset_type",
        "escalation_reason",
        "error_message",
        "current_stage",
        "search_strategy",
        "retry_count",
        "retrieved_articles",
        "extracted_fields",
        "outcome_summary",
    }
    assert set(data.keys()) == expected_keys
