"""Edge functions and terminal nodes for LangGraph wiring.

Defines the conditional routing logic after the coordinator node and
the two terminal nodes (complete, escalate) that end the pipeline.
"""

import json
import logging
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import END, START, CompiledStateGraph, StateGraph

from src.agents.coordinate_node import coordinate_node
from src.agents.load_node import load_node
from src.agents.state import EnrichmentState, PipelineStage
from src.retrieval.search_node import search_node
from src.synthesize.synthesize_node import synthesize_node
from src.validation.validate_node import validate_node

logger = logging.getLogger(__name__)


def route_after_coordinator(state: EnrichmentState) -> str:
    """Route to the next node based on coordinator decision.

    Returns the node name string that LangGraph uses for conditional
    edge routing. Falls back to "escalate" for unexpected stages.

    Args:
        state: Pipeline state after coordinator processing.

    Returns:
        Node name string matching the next pipeline stage.
    """
    if state.next_stage not in [
        PipelineStage.SEARCH,
        PipelineStage.VALIDATE,
        PipelineStage.SYNTHESIZE,
        PipelineStage.COMPLETE,
        PipelineStage.ESCALATE,
    ]:
        return "escalate"
    return state.next_stage.value


def complete_node(state: EnrichmentState, config: RunnableConfig) -> EnrichmentState:
    """Terminal node for successfully enriched records.

    Marks the pipeline as complete, generates an outcome summary,
    writes enrichment results to a JSON file, and logs success.

    Args:
        state: Pipeline state after all enrichment stages pass.
        config: RunnableConfig containing Settings under
            ``configurable.settings``.

    Returns:
        Updated state with current_stage set to COMPLETE.
    """
    settings = config["configurable"]["settings"]

    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    state.output_file_path = str(
        Path(settings.output_dir)
        / f"{state.dataset_type}_{state.incident_id}_complete.json"
    )

    state.outcome_summary = f"Enriched {len(state.extracted_fields)} fields for incident {state.incident_id} ({state.dataset_type})"

    output = {
        "incident_id": state.incident_id,
        "dataset_type": state.dataset_type,
        "extracted_fields": [f.model_dump() for f in state.extracted_fields],
        "validation_results": [r.model_dump() for r in state.validation_results],
        "search_strategy": state.next_strategy,
        "retry_count": state.retry_count,
        "conflicting_fields": [c.model_dump() for c in state.conflicting_fields]
        if state.conflicting_fields
        else [],
        "civilian_race_taxonomy": state.civilian_race_taxonomy.model_dump()
        if state.civilian_race_taxonomy
        else None,
        "conflict_annotation": state.conflict_annotation.model_dump()
        if state.conflict_annotation
        else None,
        "outcome_summary": state.outcome_summary,
    }

    state.current_stage = PipelineStage.COMPLETE
    state.requires_human_review = bool(state.conflicting_fields)

    with open(state.output_file_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(
        "Complete: incident_id=%s, dataset_type=%s, number of extracted_fields=%d",
        state.incident_id,
        state.dataset_type,
        len(state.extracted_fields),
    )

    return state


def escalate_node(state: EnrichmentState, config: RunnableConfig) -> EnrichmentState:
    """Terminal node for records requiring human review.

    Marks the pipeline as escalated, generates an outcome summary,
    writes an escalation report to a JSON file, and logs the
    escalation details.

    Args:
        state: Pipeline state after coordinator triggers escalation.
        config: RunnableConfig containing Settings under
            ``configurable.settings``.

    Returns:
        Updated state with current_stage set to ESCALATE and
        requires_human_review set to True.
    """
    settings = config["configurable"]["settings"]

    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    state.output_file_path = str(
        Path(settings.output_dir)
        / f"{state.dataset_type}_{state.incident_id}_escalate.json"
    )

    state.outcome_summary = f"Escalated incident {state.incident_id}: {state.escalation_reason} after {state.retry_count} retries"

    output = {
        "incident_id": state.incident_id,
        "dataset_type": state.dataset_type,
        "escalation_reason": state.escalation_reason,
        "error_message": state.error_message,
        "current_stage": state.current_stage,
        "search_strategy": state.next_strategy,
        "retry_count": state.retry_count,
        "retrieved_articles": [a.model_dump() for a in state.retrieved_articles],
        "validation_results": [r.model_dump() for r in state.validation_results],
        "validation_failure_summary": state.validation_failure_summary,
        "extracted_fields": [f.model_dump() for f in state.extracted_fields],
        "conflicting_fields": [c.model_dump() for c in state.conflicting_fields] if state.conflicting_fields else state.conflicting_fields,
        "civilian_race_taxonomy": state.civilian_race_taxonomy.model_dump() if state.civilian_race_taxonomy else None,
        "conflict_annotation": state.conflict_annotation.model_dump() if state.conflict_annotation else None,
        "outcome_summary": state.outcome_summary,
    }

    state.current_stage = PipelineStage.ESCALATE
    state.requires_human_review = True

    with open(state.output_file_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(
        "Escalated: incident_id=%s, escalation_reason=%s, error_message=%s",
        state.incident_id,
        state.escalation_reason,
        state.error_message,
    )

    return state


def build_graph(checkpointer: SqliteSaver | None = None) -> CompiledStateGraph:
    """Build and compile the enrichment pipeline graph.

    Assembles a hub-and-spoke StateGraph where every processing node
    (load, search, validate, synthesize) feeds into the coordinator,
    which conditionally routes to the next stage or a terminal node.

    Args:
        checkpointer: Optional SqliteSaver for persistent checkpointing.
            Pass None to compile without checkpointing.

    Returns:
        Compiled graph ready for invocation via graph.invoke(state).
    """
    workflow = StateGraph(EnrichmentState)
    workflow.add_node("load", load_node)
    workflow.add_node("search", search_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("complete", complete_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("coordinate", coordinate_node)

    # Normal nodes
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "coordinate")
    workflow.add_edge("search", "coordinate")
    workflow.add_edge("validate", "coordinate")
    workflow.add_edge("synthesize", "coordinate")
    workflow.add_edge("complete", END)
    workflow.add_edge("escalate", END)

    # Conditional nodes
    workflow.add_conditional_edges("coordinate", route_after_coordinator)

    app = workflow.compile(checkpointer=checkpointer)
    return app
