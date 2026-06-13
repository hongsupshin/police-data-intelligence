"""LLM race verifier for civilian-shot completions.

A faithfulness/safety gate that runs after extraction for ``civilians_shot``
incidents. It re-reads the source article(s) and decides whether the extracted
``civilian_race`` is *explicitly stated for THIS victim* — vetoing (nulling) a
race the text never states, which the extractor often over-commits by inferring
it from a name, neighborhood, or photo. On a sensitive field, an honest null
beats an unsourced or proxy-inferred race.

Crucially this is a *faithfulness* check (verifiable against the article), not a
truth-claim against the coarse agency ``civilians.race`` label: a value the news
states but that diverges from the label (e.g. "Asian" vs the agency's "OTHER")
is faithful and kept — that taxonomy-collapse is the race-disagreement typer's
concern, not this gate's.

Off by default; gated by ``Settings.enable_race_verification`` (civilians_shot
only). Earned offline (faithfulness A/B, 2026-06-13): zero genuine false-nulls.
"""

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.agents.state import Article, EnrichmentState


class RaceVerificationVerdict(BaseModel):
    """Structured verdict from the race verifier.

    Attributes:
        supported: True only if the article(s) explicitly state THIS victim's
            race/ethnicity and it matches the extracted value.
        quote: The exact supporting quote from the article, or None.
        reasoning: One-sentence justification.
    """

    supported: bool = Field(
        description=(
            "True ONLY if the article explicitly states THIS victim's "
            "race/ethnicity AND it matches the extracted value."
        )
    )
    quote: str | None = Field(
        default=None, description="Exact supporting quote from the article, or null."
    )
    reasoning: str = Field(description="One sentence.")


def _build_prompt(
    state: EnrichmentState, race_value: str, articles: list[Article]
) -> str:
    """Build the race-verification prompt from the victim anchors + the value.

    Args:
        state: Pipeline state carrying the record's victim anchors.
        race_value: The extracted ``civilian_race`` value to verify.
        articles: The validated articles fed to extraction.

    Returns:
        The verifier prompt string.
    """
    name = state.civilian_name or "unknown / not recorded"
    blocks = "\n\n".join(
        f"[{i}] title: {a.title}\n date: {a.published_date}\n {str(a.content)[:600]}"
        for i, a in enumerate(articles)
    )
    return (
        "We are enriching a record from the Texas civilian-shot-by-police "
        "dataset (civilians_shot). A prior step extracted the race/ethnicity of "
        f'THIS specific victim from the article(s) below as: "{race_value}".\n\n'
        "TARGET VICTIM (the ONE person this record is about):\n"
        f"- name: {name}\n"
        f"- age: {state.civilian_age}\n"
        f"- gender: {state.civilian_gender}\n\n"
        "Verify whether the article(s) EXPLICITLY state the race/ethnicity of "
        f'THIS victim and whether it matches "{race_value}". Treat '
        "Latino/Latina/Hispanic as equivalent, African-American/Black as "
        "equivalent, and Caucasian/White as equivalent.\n"
        "Count ONLY an explicit statement of THIS person's race/ethnicity in the "
        "text (match the person by name; if no name is given, by age + gender). "
        "Do NOT infer race from a name, surname origin, neighborhood, language, "
        "photograph, or immigration status. If the article does not explicitly "
        "state this victim's race/ethnicity, the extraction is NOT supported.\n"
        "Return supported=true ONLY if you can give the exact quote from the "
        f'article that states this victim\'s race/ethnicity and matches "{race_value}". '
        "Otherwise supported=false.\n\n"
        "ARTICLES:\n" + blocks
    )


def verify_race(
    llm_client: ChatAnthropic,
    state: EnrichmentState,
    race_value: str,
    articles: list[Article],
) -> RaceVerificationVerdict:
    """Verify the extracted civilian race is explicitly stated for THIS victim.

    Args:
        llm_client: Injected chat model (supports ``with_structured_output``).
        state: Pipeline state carrying the record's victim anchors.
        race_value: The extracted ``civilian_race`` value to verify.
        articles: The validated articles fed to extraction.

    Returns:
        The structured verdict; ``supported`` False means null the race.
    """
    prompt = _build_prompt(state, race_value, articles)
    return llm_client.with_structured_output(RaceVerificationVerdict).invoke(prompt)
