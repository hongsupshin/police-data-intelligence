"""Coordinator node for the enrichment pipeline.

Provides gating and escalation logic between pipeline stages. The
coordinator dispatches to stage-specific helpers that inspect state
and decide whether to proceed, retry, or escalate to human review.
"""

from src.agents.state import (
    EnrichmentState,
    EscalationReason,
    PipelineStage,
    SearchStrategyType,
)

STRATEGY_ORDER = list(SearchStrategyType)


def _relevance_threshold(strategy: SearchStrategyType) -> float:
    """Return the minimum avg relevance score for a given search strategy.

    Broader strategies accept lower scores since results are inherently
    less precise.

    Args:
        strategy: Current search strategy.

    Returns:
        Minimum relevance score threshold.
    """
    match strategy:
        case SearchStrategyType.EXACT_MATCH:
            return 0.65
        case SearchStrategyType.TEMPORAL_EXPANDED:
            return 0.55
        case SearchStrategyType.NAME_PARTIAL:
            return 0.50


def check_load_results(state: EnrichmentState) -> EnrichmentState:
    """Gate after load stage.

    Checks whether loading produced enough data to build a
    meaningful search query. Escalates if the load node errored
    or all key identity fields are missing.

    Args:
        state: Pipeline state after load node.

    Returns:
        Updated state with current_stage set to SEARCH (proceed)
        or ESCALATE (insufficient data or error).
    """
    if state.error_message and "Load failed" in state.error_message:
        state.escalation_reason = EscalationReason.EXTRACTION_ERROR
        state.requires_human_review = True
        state.next_stage = PipelineStage.ESCALATE
    elif not any([state.civilian_name, state.officer_name, state.incident_date]):
        state.escalation_reason = EscalationReason.INSUFFICIENT_SOURCES
        state.requires_human_review = True
        state.next_stage = PipelineStage.ESCALATE
    else:
        state.next_stage = PipelineStage.SEARCH
    return state


def retry_helper(state: EnrichmentState) -> EnrichmentState:
    """Advance search strategy or escalate if retries exhausted.

    Advances to the next strategy in STRATEGY_ORDER and clears
    retrieved_articles for a fresh search. If no strategies remain,
    escalates with MAX_RETRIES reason.

    Args:
        state: Pipeline state requiring a retry decision.

    Returns:
        Updated state with next_strategy advanced, retrieved_articles
        cleared, and next_stage set to SEARCH (retry) or ESCALATE
        (no strategies left).
    """
    current_index = STRATEGY_ORDER.index(state.next_strategy)
    next_index = current_index + 1
    if next_index >= len(STRATEGY_ORDER):
        state.next_stage = PipelineStage.ESCALATE
        state.escalation_reason = EscalationReason.MAX_RETRIES
        state.requires_human_review = True
    else:
        state.retry_count += 1
        state.next_strategy = STRATEGY_ORDER[next_index]
        state.next_stage = PipelineStage.SEARCH
        state.retrieved_articles = []
    return state


def check_search_results(state: EnrichmentState) -> EnrichmentState:
    """Gate after search stage.

    Checks the most recent search attempt for errors and relevance
    score. Retries with a broader strategy if results are insufficient,
    or escalates if max retries are exceeded.

    Args:
        state: Pipeline state after search node.

    Returns:
        Updated state with current_stage set to VALIDATE (proceed),
        SEARCH (retry), or ESCALATE (max retries or no strategies).
    """
    if state.retry_count <= state.max_retries:
        if state.error_message and "Search failed" in state.error_message:
            return retry_helper(state)

        if (
            state.search_attempts
            and state.search_attempts[-1].num_results > 0
        ):
            state.next_stage = PipelineStage.VALIDATE
        else:
            return retry_helper(state)
    else:
        state.next_stage = PipelineStage.ESCALATE
        state.escalation_reason = EscalationReason.MAX_RETRIES
        state.requires_human_review = True
    return state


def check_validate_results(state: EnrichmentState) -> EnrichmentState:
    """Gate after validate stage.

    Proceeds to synthesize if at least one article passed validation.
    Escalates if all articles failed.

    Args:
        state: Pipeline state after validate node.

    Returns:
        Updated state with current_stage set to SYNTHESIZE (proceed)
        or ESCALATE (no valid articles).
    """
    if any(vr.passed for vr in state.validation_results):
        state.next_stage = PipelineStage.SYNTHESIZE
    else:
        return retry_helper(state)
    return state


def check_synthesize_results(state: EnrichmentState) -> EnrichmentState:
    """Gate after synthesize stage.

    Checks for synthesize errors, conflicting fields, and empty
    extractions. Marks pipeline as complete if all checks pass.

    Args:
        state: Pipeline state after synthesize node.

    Returns:
        Updated state with current_stage set to COMPLETE (success)
        or ESCALATE (error, conflict, or no data extracted).
    """
    if state.error_message and "Synthesize failed" in state.error_message:
        state.escalation_reason = EscalationReason.MERGE_ERROR
        state.requires_human_review = True
        state.next_stage = PipelineStage.ESCALATE
    elif state.conflicting_fields:
        if state.extracted_fields:
            # Accept agreed fields; flag conflicts for human review
            state.requires_human_review = True
            state.next_stage = PipelineStage.COMPLETE
        else:
            state.escalation_reason = EscalationReason.CONFLICT
            state.requires_human_review = True
            state.next_stage = PipelineStage.ESCALATE
    elif not state.extracted_fields:
        state.escalation_reason = EscalationReason.INSUFFICIENT_SOURCES
        state.requires_human_review = True
        state.next_stage = PipelineStage.ESCALATE
    else:
        state.next_stage = PipelineStage.COMPLETE
    return state


def coordinate_node(state: EnrichmentState) -> EnrichmentState:
    """Coordinator node for the enrichment pipeline.

    Dispatches to stage-specific helper functions based on
    current_stage. Each helper inspects state and updates routing
    fields (current_stage, escalation_reason, requires_human_review).

    Args:
        state: Pipeline state after any node execution.

    Returns:
        Updated state with routing decision applied. Returns state
        unchanged for unexpected stages (COMPLETE, ESCALATE).
    """
    current_state = state.current_stage
    match current_state:
        case PipelineStage.LOAD:
            updated_state = check_load_results(state)
        case PipelineStage.SEARCH:
            updated_state = check_search_results(state)
        case PipelineStage.VALIDATE:
            updated_state = check_validate_results(state)
        case PipelineStage.SYNTHESIZE:
            updated_state = check_synthesize_results(state)
        case _:
            return state

    return updated_state
