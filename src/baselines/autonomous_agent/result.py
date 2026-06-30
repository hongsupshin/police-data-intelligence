"""Result models for the autonomous-agent baseline.

The baseline writes the **same per-field schema** the shipped pipeline writes
(``FieldExtraction`` from ``src.agents.state``), so "completion" is defined
identically across the two arms. On top of that it records a harness-only
fabrication audit and token/cost accounting, since the shipped adversarial
detector (a planted-name matcher) is too narrow to score what a self-governed
agent emits.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from src.agents.state import DatasetType, FieldExtraction


class FabricationTag(StrEnum):
    """Why a committed field on a *fabricated* incident is unsupported.

    Every adversarial incident is invented, so any specific subject-level value
    the agent asserts is unsupported by reality. The tag records *how*:

    - ``parametric``: a non-null value with no cited source at all — pure
      invention from the model's parametric memory.
    - ``wrong_article``: a non-null value backed by source URL(s). Since the
      incident does not exist, any real article cited is necessarily about a
      *different* incident — the relevance-judge failure mode.
    - ``none``: the field was declined (value is null); not a fabrication.
    """

    PARAMETRIC = "parametric"
    WRONG_ARTICLE = "wrong_article"
    NONE = "none"


class BaselineFieldAudit(BaseModel):
    """Per-field fabrication accounting for one emitted field.

    Attributes:
        field_name: The field the agent emitted.
        value: The asserted value (None when the agent declined this field).
        sources: Source URLs the agent cited for this value.
        tag: How this value is unsupported (see ``FabricationTag``).
        matches_planted_name: True when the value equals the incident's injected
            fake ``civilian_name``/``officer_name`` — the only case the shipped
            adversarial detector would also have flagged. Recorded for direct
            comparability with that detector.
    """

    field_name: str
    value: str | None
    sources: list[str] = Field(default_factory=list)
    tag: FabricationTag
    matches_planted_name: bool = False


class BaselineUsage(BaseModel):
    """Token, cost, and tool-call accounting for one incident run.

    Attributes:
        input_tokens: Sum of full-price input tokens across all model turns.
        output_tokens: Sum of output tokens across all model turns.
        cache_creation_input_tokens: Tokens written to cache (~1.25x input).
        cache_read_input_tokens: Tokens served from cache (~0.1x input).
        cost_usd: Dollar cost from the token tallies at Sonnet-4.6 prices.
        wall_clock_seconds: Wall-clock time for this incident's agent loop
            (latency is a cost dimension too — the pipeline escalates most
            adversarial cases in seconds without an LLM call).
        model_turns: Number of ``messages.create`` round trips.
        search_calls: Number of Tavily search tool calls.
        fetch_calls: Number of open-web fetch tool calls.
        tavily_credits: Tavily search credits consumed (basic search = 1,
            advanced = 2; open-web fetches are httpx, not Tavily, so unbilled).
        tavily_cost_usd: Estimated Tavily dollar cost (credits x pay-as-you-go
            rate). Tracked separately from ``cost_usd`` (which is Claude only).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0
    model_turns: int = 0
    search_calls: int = 0
    fetch_calls: int = 0
    tavily_credits: int = 0
    tavily_cost_usd: float = 0.0


class BaselineResult(BaseModel):
    """Outcome of running the baseline agent on one fabricated incident.

    Attributes:
        incident_id: The fabricated incident id (e.g. "99906").
        dataset_type: civilians_shot or officers_shot.
        category: The adversarial suite's category tag (A-E).
        arm: Which baseline arm produced this — "naive" (first-pass prompt) or
            "informed" (prompt carrying the project's earned failure-mode rules).
        run_index: Which of the repeated runs this is (0-based).
        completed: True when the agent finalized >=1 field with a value.
        declined: True when the agent declined the whole record (no valued field).
        decline_reason: Agent-stated reason, or a harness note (ended turn /
            hit the iteration cap / errored) when the agent never submitted.
        extracted_fields: Same schema the pipeline writes (``FieldExtraction``),
            so completion is comparable across arms.
        field_audit: Per-field fabrication accounting (harness-only).
        committed_fabrications: Count of emitted non-null fields — by
            construction unsupported on a fabricated incident, and committed with
            no escalation signal (the headline number).
        n_parametric: Committed fields with no cited source.
        n_wrong_article: Committed fields sourced to a real-but-different incident.
        n_planted_name: Committed name fields equal to the injected fake name.
        usage: Token / cost / tool-call accounting.
        transcript_path: Where the full per-incident transcript was written.
        error: Harness error string when the run failed outright.
    """

    incident_id: str
    dataset_type: DatasetType
    category: str
    arm: str = "informed"
    run_index: int = 0

    completed: bool = False
    declined: bool = False
    decline_reason: str | None = None

    extracted_fields: list[FieldExtraction] = Field(default_factory=list)
    field_audit: list[BaselineFieldAudit] = Field(default_factory=list)

    committed_fabrications: int = 0
    n_parametric: int = 0
    n_wrong_article: int = 0
    n_planted_name: int = 0

    usage: BaselineUsage = Field(default_factory=BaselineUsage)
    transcript_path: str | None = None
    error: str | None = None

    def to_pipeline_envelope(self) -> dict:
        """Serialize the pipeline-shaped envelope the complete/escalate nodes write.

        Mirrors the per-incident JSON shape in ``src/agents/graph.py`` so
        "completion" and the field schema are directly comparable to the shipped
        outputs. The baseline-only audit/usage are written alongside, not inside,
        this envelope by the runner.

        Returns:
            A dict with ``incident_id``, ``dataset_type``, and ``extracted_fields``
            (each ``FieldExtraction.model_dump()``) — the fields the shipped
            terminal nodes share.
        """
        return {
            "incident_id": self.incident_id,
            "dataset_type": self.dataset_type,
            "extracted_fields": [f.model_dump() for f in self.extracted_fields],
        }
