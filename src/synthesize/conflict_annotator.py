"""Advisory LLM conflict-annotator for records with deep flagged conflicts.

When articles disagree on identity/structured fields (name/age/weapon/location/
race) or a multi-subject source blends several people, the deterministic resolver
gives up and flags the field(s) for human review. This annotator runs at
synthesize, reads the disagreeing quotes, and writes a short *advisory* triage
note explaining WHY they differ and what the reviewer should check.

It is the agentic back-up catch-all that the deterministic checks can't cover —
multi-subject attribution ("which person is this record about?"), genuine
contradictions, and candidate values not actually supported by the articles. It
is **advisory only**: it never commits or overwrites a value, so it can only aid
the human-review default, never corrupt data.

Off by default; gated by ``Settings.enable_conflict_annotation`` (any dataset).
Earned offline (Haiku sample annotations, 2026-06-13): sound, quote-cited, and
faithful (no race/unsupported assertions).
"""

from langchain_anthropic import ChatAnthropic

from src.agents.state import (
    Article,
    ConflictAnnotation,
    EnrichmentState,
    FieldConflict,
)


def _build_prompt(
    state: EnrichmentState,
    conflicts: list[FieldConflict],
    articles: list[Article],
) -> str:
    """Build the conflict-annotation prompt from the anchors, conflicts, articles.

    Args:
        state: Pipeline state carrying the record's anchors.
        conflicts: The flagged conflicts to explain (each lists the competing
            values different articles produced).
        articles: The validated articles fed to extraction.

    Returns:
        The annotator prompt string.
    """
    name = state.civilian_name or "unknown / not recorded"
    conflict_lines = "\n".join(
        f"- {c.field_name}: candidate values = {c.values}" for c in conflicts
    )
    blocks = "\n\n".join(
        f"[Article {i}] {a.title}\n{str(a.content)[:700]}"
        for i, a in enumerate(articles)
    )
    return (
        "You are helping a HUMAN REVIEWER triage a police-shooting data record "
        f"({state.dataset_type}) whose automated enrichment found CONFLICTING "
        "field values across news articles. The system has NOT committed these "
        "fields; they are flagged for the reviewer.\n\n"
        "TARGET RECORD (the ONE incident/person this record is about):\n"
        f"- civilian/suspect name: {name}\n"
        f"- officer name: {state.officer_name}\n"
        f"- age: {state.civilian_age}   gender: {state.civilian_gender}\n"
        f"- location: {state.location}   date: {state.incident_date}\n\n"
        f"CONFLICTING FIELDS (each lists the competing values articles gave):\n"
        f"{conflict_lines}\n\n"
        "Write a SHORT triage note (3-6 sentences). For each conflicting field, "
        "say WHY the articles likely disagree — e.g. they describe DIFFERENT "
        "people (a bystander, a second victim, an officer vs the suspect), "
        "DIFFERENT granularity (street vs city), a GENUINE factual contradiction, "
        "or a candidate value that does NOT actually appear in the articles. "
        "Quote the relevant phrase from an article when you can, state your "
        "confidence, and flag what the reviewer should double-check.\n"
        "RULES: rely ONLY on what the articles say; NEVER assert a person's "
        "race/ethnicity or any fact the articles don't explicitly state; do not "
        "infer from a name, neighborhood, or photo; if you are unsure, say so.\n\n"
        "ARTICLES:\n" + blocks
    )


def annotate_conflicts(
    llm_client: ChatAnthropic,
    state: EnrichmentState,
    conflicts: list[FieldConflict],
    articles: list[Article],
) -> ConflictAnnotation:
    """Write an advisory reviewer note explaining the record's flagged conflicts.

    Args:
        llm_client: Injected chat model (supports ``with_structured_output``);
            typically a cheap model (Haiku) for this high-frequency advisory task.
        state: Pipeline state carrying the record's anchors.
        conflicts: The flagged conflicts to explain.
        articles: The validated articles fed to extraction.

    Returns:
        The structured advisory note (never committed; for human review only).
    """
    prompt = _build_prompt(state, conflicts, articles)
    return llm_client.with_structured_output(ConflictAnnotation).invoke(prompt)
