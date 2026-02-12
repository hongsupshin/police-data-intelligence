"""Edge functions and terminal nodes for LangGraph wiring.

Defines the conditional routing logic after the coordinator node and
the two terminal nodes (complete, escalate) that end the pipeline.
"""

from src.agents.state import EnrichmentState, PipelineStage


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
