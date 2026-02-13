"""Edge functions and terminal nodes for LangGraph wiring.

Defines the conditional routing logic after the coordinator node and
the two terminal nodes (complete, escalate) that end the pipeline.
"""

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import END, START, CompiledStateGraph, StateGraph

from src.agents.coordinate_node import coordinate_node
from src.agents.extract_node import extract_node
from src.agents.state import EnrichmentState, PipelineStage
from src.merge.merge_node import merge_node
from src.retrieval.search_node import search_node
from src.validation.validate_node import validate_node


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
        PipelineStage.MERGE,
        PipelineStage.COMPLETE,
        PipelineStage.ESCALATE,
    ]:
        return "escalate"
    return state.next_stage.value


def complete_node(state: EnrichmentState) -> EnrichmentState:
    """Terminal node for successfully enriched records.

    Marks the pipeline as complete with no human review needed.
    File writing, reasoning summary generation, and logging are
    planned for future implementation.

    Args:
        state: Pipeline state after all enrichment stages pass.

    Returns:
        Updated state with current_stage set to COMPLETE.
    """
    # TODO: file writing (I/O), reasoning summary generation, logging
    state.current_stage = PipelineStage.COMPLETE
    state.requires_human_review = False
    state.output_file_path = "pending"
    state.reasoning_summary = "pending"
    return state


def escalate_node(state: EnrichmentState) -> EnrichmentState:
    """Terminal node for records requiring human review.

    Marks the pipeline as escalated so the record is routed to
    human-in-the-loop review. File writing, reasoning summary
    generation, and logging are planned for future implementation.

    Args:
        state: Pipeline state after coordinator triggers escalation.

    Returns:
        Updated state with current_stage set to ESCALATE and
        requires_human_review set to True.
    """
    # TODO: file writing (I/O), reasoning summary generation, logging
    state.current_stage = PipelineStage.ESCALATE
    state.requires_human_review = True
    state.output_file_path = "pending"
    state.reasoning_summary = "pending"
    return state


def build_graph(checkpointer: SqliteSaver | None = None) -> CompiledStateGraph:
    """Build and compile the enrichment pipeline graph.

    Assembles a hub-and-spoke StateGraph where every processing node
    (extract, search, validate, merge) feeds into the coordinator,
    which conditionally routes to the next stage or a terminal node.

    Args:
        checkpointer: Optional SqliteSaver for persistent checkpointing.
            Pass None to compile without checkpointing.

    Returns:
        Compiled graph ready for invocation via ``graph.invoke(state)``.
    """
    workflow = StateGraph(EnrichmentState)
    workflow.add_node("extract", extract_node)
    workflow.add_node("search", search_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("complete", complete_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("coordinate", coordinate_node)

    # Normal nodes
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "coordinate")
    workflow.add_edge("search", "coordinate")
    workflow.add_edge("validate", "coordinate")
    workflow.add_edge("merge", "coordinate")
    workflow.add_edge("complete", END)
    workflow.add_edge("escalate", END)

    # Conditional nodes
    workflow.add_conditional_edges("coordinate", route_after_coordinator)

    app = workflow.compile(checkpointer=checkpointer)
    return app
