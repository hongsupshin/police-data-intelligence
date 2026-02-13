"""Merge node for enrichment pipeline."""

from collections import Counter, defaultdict
from datetime import date

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from rapidfuzz import fuzz

from src.agents.state import (
    FIELD_TO_STATE_ATTR,
    Article,
    ConfidenceLevel,
    EnrichmentState,
    FieldExtraction,
    MediaFeatureField,
    MergeExtractionResponse,
    PipelineStage,
)

FIELD_DEFINITIONS = {
    MediaFeatureField.OFFICER_NAME: "Name of the police officer involved in the shooting. This person can be the shooter or the victim.",
    MediaFeatureField.CIVILIAN_NAME: "Name of the civilian (non-officer) involved in the shooting. This person can be the shooter or the victim.",
    MediaFeatureField.CIVILIAN_AGE: "Age of the civilian in integers",
    MediaFeatureField.CIVILIAN_RACE: "Race/ethnicity of the civilian",
    MediaFeatureField.WEAPON: "Weapon involved in the incident, including type (e.g., handgun, rifle, knife, vehicle). Note which party possessed or used it if mentioned.",
    MediaFeatureField.LOCATION_DETAIL: "Detailed location information such as street/business/landmark names",
    MediaFeatureField.TIME_OF_DAY: "Time of day when the incident occurred, as described in the article",
    MediaFeatureField.OUTCOME: "Fatal or non-fatal outcome of the victim (police officer or the civilian)",
    MediaFeatureField.CIRCUMSTANCE: "Any context or background regarding the incident such as the cause, complications",
}

assert set(FIELD_DEFINITIONS.keys()) == set(
    MediaFeatureField
), "Field definitions and MediaFeatureField do not match."

RAPIDFUZZ_THRESHOLD = 80


# helper functions
def extract_fields(
    article: Article, llm_client: ChatOpenAI, fields: list[MediaFeatureField]
) -> dict[str, FieldExtraction]:
    """Extract structured fields from a single article using an LLM.

    Builds a prompt with field definitions and article content, then
    calls the LLM with structured output to extract all fields at once.
    Returns an empty dict if article content is missing or the LLM call fails.

    Args:
        article: Article object containing content to extract from.
        llm_client: LangChain ChatOpenAI client for structured extraction.
        fields: List of MediaFeatureField enums to extract.

    Returns:
        Dictionary mapping field names to FieldExtraction objects.
        Empty dict if extraction fails or article content is None.
    """
    if article.content is None:
        # TODO: warning message
        return {}

    prompt = """
    You are extracting structured information from a police shooting incident article.
    For each of the following fields, extract the value from the article:
    """
    for field_name in fields:
        prompt += f"""
        - "{field_name}": {FIELD_DEFINITIONS[field_name]}
        """
    prompt += f"""
    Instructions:
    - Use the exact field names shown above. (example: use "weapon" not "Weapon used")
    - Quote the relevant sentence verbatim as "source_quotes".
    - Explain your rationale as "llm_reasoning".
    - If a field is not mentioned in the article, set value to null.

    Article title: {article.title}
    Published: {article.published_date}
    Content:
    ---
    {article.content}
    ---
    """

    structured_llm = llm_client.with_structured_output(MergeExtractionResponse)
    try:
        results = structured_llm.invoke(prompt)
    except Exception:
        # TODO: warning message
        return {}
    extractions = {}
    for extraction in results.extractions:
        extraction.sources = [article.url]
        extraction.confidence = ConfidenceLevel.PENDING
        extractions[extraction.field_name] = extraction
    return extractions


def check_articles_match(
    field: MediaFeatureField, extracted_results: list[FieldExtraction]
) -> tuple[bool, FieldExtraction | None]:
    """Check consistency of extracted values across multiple articles.

    Filters out null values, then checks if remaining extractions agree.
    Uses fuzzy matching (rapidfuzz) to resolve minor differences. Sets
    confidence level based on agreement: HIGH if all agree exactly,
    MEDIUM if single source or fuzzy-resolved.

    Args:
        field: The MediaFeatureField being checked (for logging).
        extracted_results: List of FieldExtraction objects for this field,
            one per article.

    Returns:
        Tuple of (matched, converged_extraction). If matched is True,
        converged_extraction contains the winning FieldExtraction with
        updated confidence. If False, returns None.
    """
    non_null_results = [r for r in extracted_results if r.value is not None]
    non_null_values = [r.value for r in non_null_results]
    if len(non_null_results) == 0:
        # TODO: warning
        return (False, None)

    # Single extraction
    if len(non_null_results) == 1:
        result = non_null_results[0]
        result.confidence = ConfidenceLevel.MEDIUM
        return (True, result)

    # All agree
    if len(set(non_null_values)) == 1:
        result = non_null_results[0]
        result.confidence = ConfidenceLevel.HIGH
        return (True, result)

    counts = Counter(non_null_values)
    most_common = counts.most_common(1)[0][0]
    others = [v for v in set(non_null_values) if v != most_common]

    if all(fuzz.ratio(most_common, other) >= RAPIDFUZZ_THRESHOLD for other in others):
        # Minor difference: return the most common
        winner = next(r for r in non_null_results if r.value == most_common)
        winner.confidence = ConfidenceLevel.MEDIUM
        return (True, winner)
    else:
        # TODO: warning message
        return (False, None)


def check_reference_match(
    field: MediaFeatureField,
    extracted_field: FieldExtraction,
    reference: str | date | None,
) -> tuple[bool, FieldExtraction | None]:
    """Check if extracted value matches the database reference value.

    Compares the converged extraction against the existing database
    value using fuzzy matching. If matched, overwrites the extracted
    value with the reference (immutability assumption). If reference
    is None, accepts the extraction as-is.

    Args:
        field: The MediaFeatureField being checked (for logging).
        extracted_field: Converged FieldExtraction from check_articles_match.
        reference: Database value to compare against, or None if missing.

    Returns:
        Tuple of (matched, extraction). If matched is True, extraction
        has its value set to the reference. If False, returns None.
    """
    if reference is None:
        # TODO: warning
        return (True, extracted_field)

    if fuzz.ratio(str(reference), extracted_field.value) < RAPIDFUZZ_THRESHOLD:
        # TODO: logging
        return (False, None)
    else:
        extracted_field.value = str(reference)
        return (True, extracted_field)


def merge_node(state: EnrichmentState, config: RunnableConfig) -> EnrichmentState:
    """Orchestrate field extraction, cross-article consistency, and reference matching.

    Extracts fields from all retrieved articles using an LLM, groups
    results by field, checks consistency across articles, and validates
    against database reference values. Populates extracted_fields and
    conflicting_fields on the state.

    Args:
        state: Current enrichment pipeline state with retrieved articles.
        config: LangGraph RunnableConfig containing the LLM client at
            ``config["configurable"]["llm_client"]``.

    Returns:
        Updated EnrichmentState with extracted_fields, conflicting_fields,
        and current_stage set to MERGE.
    """
    llm_client = config["configurable"]["llm_client"]

    # Extract from all articles
    try:
        all_extractions = []
        for article in state.retrieved_articles:
            result = extract_fields(article, llm_client, list(MediaFeatureField))
            all_extractions.append(result)

        # Group by field
        extractions_by_field = defaultdict(list)
        for extraction_dict in all_extractions:
            for field_name, field_extraction in extraction_dict.items():
                extractions_by_field[field_name].append(field_extraction)

        # Consistency check
        state.conflicting_fields = []
        for field_name in list(MediaFeatureField):
            extraction = extractions_by_field[field_name]
            if not extraction:
                # Skip empty list
                continue
            articles_match = check_articles_match(field_name, extraction)
            if articles_match[0]:  # Merge success
                # Only check the fields in FIELD_TO_STATE_ATTR, the rest goes to extracted_fields
                if field_name in FIELD_TO_STATE_ATTR:
                    reference = getattr(state, FIELD_TO_STATE_ATTR[field_name])
                    reference_match = check_reference_match(
                        field_name, articles_match[1], reference
                    )
                    if not reference_match[0]:
                        # Log conflict btw reference and extraction
                        state.conflicting_fields.append(field_name)
                # Regardless of the merge success, log extracted fields
                state.extracted_fields.append(articles_match[1])
            else:
                # Log conflicting fields
                state.conflicting_fields.append(field_name)

        state.current_stage = PipelineStage.MERGE
    except Exception as e:
        state.extracted_fields = []
        state.conflicting_fields = None
        state.error_message = f"Merge failed: {str(e)}"
        state.current_stage = PipelineStage.MERGE

    return state
