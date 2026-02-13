from datetime import date

import pytest
from langgraph.graph.state import CompiledStateGraph

from src.agents.graph import (
    build_graph,
    complete_node,
    escalate_node,
    route_after_coordinator,
)
from src.agents.state import (
    DatasetType,
    EnrichmentState,
    PipelineStage,
    SearchStrategyType,
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


def test_complete_node(base_state: EnrichmentState) -> None:
    """Complete node sets COMPLETE stage and no human review."""
    state = complete_node(base_state.model_copy())
    assert state.current_stage == PipelineStage.COMPLETE
    assert not state.requires_human_review


def test_escalate_node(base_state: EnrichmentState) -> None:
    """Escalate node sets ESCALATE stage and requires human review."""
    state = escalate_node(base_state.model_copy())
    assert state.current_stage == PipelineStage.ESCALATE
    assert state.requires_human_review


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
