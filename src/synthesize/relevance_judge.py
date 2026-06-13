"""LLM relevance judge for officer-involved-shooting completions.

A semantic precision gate that runs after the deterministic validation
(date/location/name) for ``officers_shot`` incidents. It asks an LLM whether at
least one of the validated articles actually reports THIS specific incident, and
vetoes a completion matched to the wrong article — the "right structure, wrong
incident" failure the mechanical gate cannot catch (e.g. a record "completed" on
a coincidental same-city article).

Ported from the offline A/B (vetoed ~8% of officer completions, well-calibrated).
On by default; gated by ``Settings.enable_relevance_gate`` (officers_shot only).
"""

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.agents.state import Article, EnrichmentState


class RelevanceVerdict(BaseModel):
    """Structured verdict from the relevance judge.

    Attributes:
        relevant_any: True if at least one article reports THIS specific
            incident (including its direct aftermath or trial).
        relevant_indices: 0-based indices of the relevant articles.
        reasoning: One-sentence justification.
    """

    relevant_any: bool = Field(
        description=(
            "True if >=1 article reports THIS specific incident "
            "(including its direct aftermath/trial)."
        )
    )
    relevant_indices: list[int] = Field(default_factory=list)
    reasoning: str = Field(description="One sentence.")


def _build_prompt(state: EnrichmentState, articles: list[Article]) -> str:
    """Build the relevance-judge prompt from incident anchors + the article set.

    Args:
        state: Pipeline state carrying the seed incident anchors.
        articles: The validated articles fed to extraction.

    Returns:
        The judge prompt string.
    """
    anchors = (
        f"- Officer (victim) name: {state.officer_name or 'unknown'}\n"
        f"- Suspect/civilian name: {state.civilian_name or 'unknown'}\n"
        f"- Date: {state.incident_date} (+/-7 days OK)\n"
        f"- City: {state.location}\n"
        f"- Outcome: officer {state.severity}; suspect/civilian "
        f"{state.civilian_outcome or 'unknown'}"
    )
    blocks = "\n\n".join(
        f"[{i}] title: {a.title}\n date: {a.published_date}\n {str(a.content)[:600]}"
        for i, a in enumerate(articles)
    )
    return (
        "We are enriching a record from the Texas Officer-Involved Shooting "
        "dataset (officers_shot): an incident in which a peace officer was "
        "INJURED or KILLED and/or a civilian was shot by police. Known incident "
        "details:\n" + anchors + "\n\n"
        "Below are the article(s) the pipeline used. For the SET, decide if AT "
        "LEAST ONE article reports THIS SPECIFIC incident — the same shooting at "
        "the same place and time — or its OWN later aftermath or trial.\n"
        "DATE is a hard test: the shooting itself must have happened within +/-7 "
        "days of the record's date above. Use the date the shooting OCCURRED (from "
        "the article), not the article's publication or trial date. If the "
        "shooting clearly happened outside that window, the article is NOT "
        "relevant — even if the city matches and even if it is later coverage or a "
        "trial of that different-dated event.\n"
        "If the date fits: an article that NAMES the officer or the suspect/"
        "civilian above (in the right city) is the SAME incident even if a "
        "reported outcome differs from the record — the record may be stale, so a "
        "differing outcome is a data conflict, NOT a different incident. If no "
        "name matches — early coverage often withholds names — judge by city and "
        "the people's outcomes (officer and suspect/civilian, above): an unnamed "
        "report whose city, date, and circumstances fit THIS incident is the same "
        "event. Otherwise (a different city, named people who are clearly someone "
        "else, or circumstances that do not fit) it is NOT relevant. A "
        "multi-officer incident article that names other officers from the SAME "
        "event IS relevant.\n\n"
        "ARTICLES:\n" + blocks
    )


def judge_relevance(
    llm_client: ChatAnthropic,
    state: EnrichmentState,
    articles: list[Article],
) -> RelevanceVerdict:
    """Judge whether any validated article is about THIS officer incident.

    Args:
        llm_client: Injected chat model (supports ``with_structured_output``).
        state: Pipeline state carrying the incident anchors.
        articles: The validated articles fed to extraction.

    Returns:
        The structured relevance verdict; ``relevant_any`` False means veto.
    """
    prompt = _build_prompt(state, articles)
    return llm_client.with_structured_output(RelevanceVerdict).invoke(prompt)
