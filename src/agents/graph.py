from src.agents.state import EnrichmentState, PipelineStage


def route_after_coordinator(state: EnrichmentState) -> str:
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
    # TODO: file writing (I/O), reasoning summary generation, logging
    state.current_stage = PipelineStage.COMPLETE
    state.requires_human_review = False
    state.output_file_path = "pending"
    state.reasoning_summary = "pending"
    return state


def escalate_node(state: EnrichmentState) -> EnrichmentState:
    # TODO: file writing (I/O), reasoning summary generation, logging
    state.current_stage = PipelineStage.ESCALATE
    state.requires_human_review = True
    state.output_file_path = "pending"
    state.reasoning_summary = "pending"
    return state
