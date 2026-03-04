"""Merge node for enrichment pipeline."""

import logging
import re
from collections import Counter, defaultdict
from datetime import date

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from rapidfuzz import fuzz

from src.agents.state import (
    FIELD_TO_STATE_ATTR,
    Article,
    ConfidenceLevel,
    ConflictType,
    EnrichmentState,
    FieldConflict,
    FieldExtraction,
    MediaFeatureField,
    MergeExtractionResponse,
    PipelineStage,
)
from src.merge.weapon_similarity import weapons_match

logger = logging.getLogger(__name__)

FIELD_DEFINITIONS = {
    MediaFeatureField.OFFICER_NAME: "Name of the police officer involved in the shooting. This person can be the shooter or the victim.",
    MediaFeatureField.CIVILIAN_NAME: "Name of the civilian (non-officer) involved in the shooting. This person can be the shooter or the victim.",
    MediaFeatureField.CIVILIAN_AGE: "Age of the civilian in integers",
    MediaFeatureField.CIVILIAN_RACE: "Race/ethnicity of the civilian",
    MediaFeatureField.WEAPON: "Weapon possessed by the civilian. Choose exactly one from: HANDGUN, RIFLE, SHOTGUN, KNIFE, VEHICLE, OTHER. Use OTHER if the weapon doesn't fit these categories or is unclear. Return only the category name, nothing else.",
    MediaFeatureField.LOCATION_DETAIL: "Detailed location information such as street/business/landmark names",
    MediaFeatureField.TIME_OF_DAY: "Time of day when the incident occurred, as described in the article",
    MediaFeatureField.OUTCOME: "Fatal or non-fatal outcome of the victim (police officer or the civilian)",
    MediaFeatureField.CIRCUMSTANCE: "Any context or background regarding the incident such as the cause, complications",
}

assert set(FIELD_DEFINITIONS.keys()) == set(
    MediaFeatureField
), "Field definitions and MediaFeatureField do not match."

RAPIDFUZZ_THRESHOLD = 80

FIELD_FUZZY_THRESHOLDS: dict[MediaFeatureField, int] = {
    MediaFeatureField.LOCATION_DETAIL: 75,
}

RACE_SYNONYMS: dict[str, str] = {
    "african american": "black",
    "african-american": "black",
    "caucasian": "white",
    "latino": "hispanic",
    "latina": "hispanic",
    "latin": "hispanic",
}


def normalize_race(value: str) -> str:
    """Normalize a race/ethnicity string via synonym mapping.

    Args:
        value: Raw race string (e.g., "African American", "Caucasian").

    Returns:
        Canonical lowercase form (e.g., "black", "white").

    Examples:
        >>> normalize_race("African American")
        'black'
        >>> normalize_race("White")
        'white'
    """
    lowered = value.strip().lower()
    return RACE_SYNONYMS.get(lowered, lowered)


_HONORIFICS = re.compile(
    r"\b(mr|mrs|ms|dr|sgt|master sgt|cpl|lt|capt|"
    r"officer|detective|deputy|chief|trooper|corporal|sergeant|"
    r"lieutenant|captain|colonel|private|specialist|major|general)\b\.?",
    re.IGNORECASE,
)


def normalize_name(value: str) -> str:
    """Strip honorifics, quotes, and extra whitespace from a name.

    Used before fuzzy matching to prevent rank prefixes and nickname
    quotes from diluting similarity scores.

    Args:
        value: Raw name string (e.g., ``"Master Sgt. Alva Joe Gwinn"``).

    Returns:
        Lowercased name with honorifics, quotes, and extra whitespace
        removed.

    Examples:
        >>> normalize_name("Master Sgt. Alva Joe Gwinn")
        'alva joe gwinn'
        >>> normalize_name("Alva 'Joe' Gwinn")
        'alva joe gwinn'
        >>> normalize_name("John Doe")
        'john doe'
    """
    # Strip quotes (single, double, smart)
    result = re.sub(r"['\u2018\u2019\u201c\u201d\"']", "", value)
    # Strip honorifics/titles
    result = _HONORIFICS.sub("", result)
    # Collapse whitespace
    result = " ".join(result.split()).strip()
    return result.lower()


# helper functions
def extract_fields(
    article: Article, llm_client: ChatAnthropic, fields: list[MediaFeatureField]
) -> dict[str, FieldExtraction]:
    """Extract structured fields from a single article using an LLM.

    Builds a prompt with field definitions and article content, then
    calls the LLM with structured output to extract all fields at once.
    Returns an empty dict if article content is missing or the LLM call fails.

    Args:
        article: Article object containing content to extract from.
        llm_client: LangChain ChatAnthropic client for structured extraction.
        fields: List of MediaFeatureField enums to extract.

    Returns:
        Dictionary mapping field names to FieldExtraction objects.
        Empty dict if extraction fails or article content is None.
    """
    if article.content is None:
        logger.warning("Article content is None for %s", article.url)
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
        logger.warning("LLM extraction failed for %s", article.url)
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
        updated confidence. Returns (True, None) when all extractions
        are null (no data, not a conflict). If False, returns None.
    """
    non_null_results = [r for r in extracted_results if r.value is not None]
    non_null_values = [r.value for r in non_null_results]
    if len(non_null_results) == 0:
        logger.debug("All extractions null for field %s", field)
        return (True, None)

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

    # Name fields: normalize before fuzzy comparison to handle
    # honorifics ("Master Sgt.") and nickname quotes ("'Joe'")
    if field in (MediaFeatureField.OFFICER_NAME, MediaFeatureField.CIVILIAN_NAME):
        normalized_common = normalize_name(most_common)
        if all(
            fuzz.ratio(normalized_common, normalize_name(other))
            >= RAPIDFUZZ_THRESHOLD
            for other in others
        ):
            winner = next(r for r in non_null_results if r.value == most_common)
            winner.confidence = ConfidenceLevel.MEDIUM
            return (True, winner)

    # Race field: normalize synonyms before comparison
    if field == MediaFeatureField.CIVILIAN_RACE:
        normalized_common = normalize_race(most_common)
        if all(normalize_race(other) == normalized_common for other in others):
            winner = next(r for r in non_null_results if r.value == most_common)
            winner.confidence = ConfidenceLevel.MEDIUM
            return (True, winner)

    # Weapon field: use embedding-based similarity
    if field == MediaFeatureField.WEAPON:
        if all(weapons_match(most_common, other) for other in others):
            winner = next(r for r in non_null_results if r.value == most_common)
            winner.confidence = ConfidenceLevel.MEDIUM
            return (True, winner)

    threshold = FIELD_FUZZY_THRESHOLDS.get(field, RAPIDFUZZ_THRESHOLD)
    if all(
        max(fuzz.ratio(most_common, other), fuzz.partial_ratio(most_common, other))
        >= threshold
        for other in others
    ):
        # Minor difference: return the most common
        winner = next(r for r in non_null_results if r.value == most_common)
        winner.confidence = ConfidenceLevel.MEDIUM
        return (True, winner)
    else:
        logger.warning("Articles disagree on field %s", field)
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
        logger.debug("No reference value for field %s", field)
        return (True, extracted_field)

    if field == MediaFeatureField.WEAPON:
        if weapons_match(extracted_field.value, str(reference)):
            extracted_field.value = str(reference)
            return (True, extracted_field)
        else:
            logger.warning("Reference mismatch for field %s", field)
            return (False, None)

    if fuzz.ratio(str(reference).lower(), extracted_field.value.lower()) < RAPIDFUZZ_THRESHOLD:
        logger.warning("Reference mismatch for field %s", field)
        return (False, None)
    else:
        extracted_field.value = str(reference)
        return (True, extracted_field)


def merge_node(state: EnrichmentState, config: RunnableConfig) -> EnrichmentState:
    """Orchestrate field extraction, cross-article consistency, and reference matching.

    Filters to only validated articles (those that passed validation),
    extracts fields using an LLM, groups results by field, checks
    consistency across articles, and validates against database reference
    values. Populates extracted_fields and conflicting_fields on the state.

    Args:
        state: Current enrichment pipeline state with retrieved articles.
        config: LangGraph RunnableConfig containing the LLM client at
            ``config["configurable"]["llm_client"]``.

    Returns:
        Updated EnrichmentState with extracted_fields, conflicting_fields,
        and current_stage set to MERGE.
    """
    llm_client = config["configurable"]["llm_client"]

    # Filter to only validated articles
    validated_urls = {
        vr.article.url for vr in state.validation_results if vr.passed
    }
    validated_articles = [
        a for a in state.retrieved_articles if a.url in validated_urls
    ]

    # Extract from validated articles
    try:
        all_extractions = []
        for article in validated_articles:
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
                    if articles_match[1] is not None:
                        reference_match = check_reference_match(
                            field_name, articles_match[1], reference
                        )
                        if not reference_match[0]:
                            converged = articles_match[1]
                            state.conflicting_fields.append(
                                FieldConflict(
                                    field_name=field_name,
                                    conflict_type=ConflictType.REFERENCE_MISMATCH,
                                    values=[converged.value],
                                    sources=[converged.sources],
                                    reference_value=str(reference),
                                )
                            )
                # Regardless of the merge success, log extracted fields
                if articles_match[1]:
                    state.extracted_fields.append(articles_match[1])
            else:
                # Log conflicting fields
                non_null = [e for e in extraction if e.value is not None]
                seen: dict[str, list[str]] = {}
                for e in non_null:
                    if e.value not in seen:
                        seen[e.value] = []
                    seen[e.value].extend(e.sources)
                state.conflicting_fields.append(
                    FieldConflict(
                        field_name=field_name,
                        conflict_type=ConflictType.ARTICLES_DISAGREE,
                        values=list(seen.keys()),
                        sources=[srcs for srcs in seen.values()],
                    )
                )

        state.current_stage = PipelineStage.MERGE
    except Exception as e:
        state.extracted_fields = []
        state.conflicting_fields = None
        state.error_message = f"Merge failed: {str(e)}"
        state.current_stage = PipelineStage.MERGE

    return state
