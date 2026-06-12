"""Synthesize node for enrichment pipeline."""

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
    DatasetType,
    EnrichmentState,
    FieldConflict,
    FieldExtraction,
    MediaFeatureField,
    MergeExtractionResponse,
    PipelineStage,
)
from src.synthesize.relevance_judge import judge_relevance
from src.synthesize.weapon_similarity import weapons_match

logger = logging.getLogger(__name__)

FIELD_DEFINITIONS = {
    MediaFeatureField.OFFICER_NAME: "Name of the police officer involved in the shooting. This person can be the shooter or the victim.",
    MediaFeatureField.CIVILIAN_NAME: "Name of the civilian (non-officer) involved in the shooting. This person can be the shooter or the victim.",
    MediaFeatureField.CIVILIAN_AGE: "Age of the civilian in integers",
    MediaFeatureField.CIVILIAN_RACE: "Race/ethnicity of the civilian",
    MediaFeatureField.WEAPON: "Weapon possessed by the civilian. Choose exactly one from: HANDGUN, RIFLE, SHOTGUN, KNIFE, VEHICLE, OTHER. Use OTHER if the weapon doesn't fit these categories or is unclear. Return only the category name, nothing else.",
    MediaFeatureField.LOCATION_DETAIL: "Street address and city/county where the incident occurred. Include the street number and name if mentioned (e.g., '5021 Glenview Dr, Houston' or '900 block of Orange St, Beaumont'). Always include the city or county name. If only a landmark, intersection, or neighborhood is given, include that with the city (e.g., 'downtown Dallas' or 'I-35 near Waco').",
    MediaFeatureField.TIME_OF_DAY: "Time of day when the incident occurred, as described in the article",
    MediaFeatureField.OUTCOME: "Fatal or non-fatal outcome of the victim (police officer or the civilian)",
    MediaFeatureField.CIRCUMSTANCE: "Any context or background regarding the incident such as the cause, complications",
}

assert set(FIELD_DEFINITIONS.keys()) == set(
    MediaFeatureField
), "Field definitions and MediaFeatureField do not match."

_DEFAULT_PREAMBLE = """
    You are extracting structured information from a police shooting incident article.
    For each of the following fields, extract the value from the article:
    """

_OFFICERS_PREAMBLE = """
    You are extracting structured information from a news article about an
    officer-involved shooting in which a police OFFICER was shot. The officer is
    the VICTIM. Here, "civilian" refers to the SUSPECT/shooter - the non-officer
    person involved - NOT the officer.
    For each of the following fields, extract the value from the article:
    """

# officers_shot reframes the suspect-vs-officer roles; only these field
# definitions differ from the civilian-centric defaults.
_OFFICER_FIELD_OVERRIDES: dict[MediaFeatureField, str] = {
    MediaFeatureField.CIVILIAN_AGE: (
        "Age (integer) of the SUSPECT/civilian - the non-officer person "
        "involved. Null if not stated."
    ),
    MediaFeatureField.CIVILIAN_RACE: (
        "Race/ethnicity of the SUSPECT/civilian (the non-officer). Only if "
        "explicitly stated; otherwise null. Do not guess."
    ),
    MediaFeatureField.OUTCOME: (
        "Whether the police OFFICER (the victim who was shot) was killed "
        "(=fatal) or injured/survived (=non-fatal). Map killed/died/slain to "
        "fatal; injured/wounded/survived/recovering/hospitalized to non-fatal. "
        "Null if unstated."
    ),
}


def _prompt_parts(
    dataset_type: DatasetType,
) -> tuple[str, dict[MediaFeatureField, str]]:
    """Return the (preamble, field-definitions) for the dataset.

    officers_shot reframes the civilian fields as the suspect/shooter and
    outcome as the officer-victim's fate; civilians_shot keeps the defaults
    (its civilian-centric prompt is correct and stays byte-identical).

    Args:
        dataset_type: Which TJI dataset the incident belongs to.

    Returns:
        Tuple of (prompt preamble, field-name -> definition mapping).
    """
    if dataset_type == DatasetType.OFFICERS_SHOT:
        return _OFFICERS_PREAMBLE, {**FIELD_DEFINITIONS, **_OFFICER_FIELD_OVERRIDES}
    return _DEFAULT_PREAMBLE, FIELD_DEFINITIONS


RAPIDFUZZ_THRESHOLD = 80

FIELD_FUZZY_THRESHOLDS: dict[MediaFeatureField, int] = {
    MediaFeatureField.LOCATION_DETAIL: 75,
}

_GENDER_WORDS = re.compile(r"\b(male|female|man|woman)\b")

_RACE_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(black|african)\b"), "black"),
    (re.compile(r"\b(hispanic|latino|latina)\b"), "hispanic"),
    (re.compile(r"\b(white|caucasian)\b"), "white"),
    (re.compile(r"\basian\b"), "asian"),
]


def normalize_race(value: str) -> str:
    """Normalize a race/ethnicity string via keyword matching.

    Strips gender words, then checks for race keywords in priority
    order. Unmatched values default to "other".

    Args:
        value: Raw race string (e.g., "African American", "Hispanic/Latino male").

    Returns:
        Canonical lowercase form (e.g., "black", "hispanic", "other").

    Examples:
        >>> normalize_race("African-American/Black")
        'black'
        >>> normalize_race("Hispanic/Latino male")
        'hispanic'
        >>> normalize_race("Iranian")
        'other'
    """
    lowered = value.strip().lower()
    lowered = _GENDER_WORDS.sub("", lowered).strip()
    lowered = re.sub(r"\s+", " ", lowered)
    for pattern, canonical in _RACE_KEYWORDS:
        if pattern.search(lowered):
            return canonical
    return "other"


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
    article: Article,
    llm_client: ChatAnthropic,
    fields: list[MediaFeatureField],
    dataset_type: DatasetType,
) -> dict[str, FieldExtraction]:
    """Extract structured fields from a single article using an LLM.

    Builds a prompt with field definitions and article content, then
    calls the LLM with structured output to extract all fields at once.
    Returns an empty dict if article content is missing or the LLM call fails.

    Args:
        article: Article object containing content to extract from.
        llm_client: LangChain ChatAnthropic client for structured extraction.
        fields: List of MediaFeatureField enums to extract.
        dataset_type: Which TJI dataset the incident belongs to; selects the
            civilian- vs officer-framed extraction prompt.

    Returns:
        Dictionary mapping field names to FieldExtraction objects.
        Empty dict if extraction fails or article content is None.
    """
    if article.content is None:
        logger.warning("Article content is None for %s", article.url)
        return {}

    preamble, field_definitions = _prompt_parts(dataset_type)
    prompt = preamble
    for field_name in fields:
        prompt += f"""
        - "{field_name}": {field_definitions[field_name]}
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


def synthesize_node(state: EnrichmentState, config: RunnableConfig) -> EnrichmentState:
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
        and current_stage set to SYNTHESIZE.
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
            result = extract_fields(
                article, llm_client, list(MediaFeatureField), state.dataset_type
            )
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

        state.current_stage = PipelineStage.SYNTHESIZE
    except Exception as e:
        state.extracted_fields = []
        state.conflicting_fields = None
        state.error_message = f"Synthesize failed: {str(e)}"
        state.current_stage = PipelineStage.SYNTHESIZE

    # Relevance gate (officers_shot only, flag-gated): veto a would-be completion
    # whose validated articles aren't about THIS incident. Runs only on the
    # success path; fail-open so a judge outage never blocks a completion.
    settings = config["configurable"].get("settings")
    if (
        settings is not None
        and getattr(settings, "enable_relevance_gate", False)
        and state.dataset_type == DatasetType.OFFICERS_SHOT
        and state.extracted_fields
        and not state.error_message
        and validated_articles
    ):
        try:
            verdict = judge_relevance(llm_client, state, validated_articles)
            if not verdict.relevant_any:
                state.relevance_vetoed = True
        except Exception as e:
            logger.warning("Relevance judge failed (fail-open, no veto): %s", e)

    return state
