"""Validation Node for the enrichment pipeline.

Validates retrieved articles against incident data from the database
using rule-based checks: date proximity, location matching, and
optional name matching. No LLM calls — pure deterministic logic.

The node reads state.retrieved_articles (from Search Node) and compares
each article against state fields (from Extract Node) to determine if
the article describes the same incident.
"""

from datetime import date

from rapidfuzz import fuzz

from src.agents.state import (
    EnrichmentState,
    PipelineStage,
    ValidationResult,
)


def check_location_match(article_location: str | None, location: str | None) -> bool:
    """Check if incident location appears in article text.

    Uses fuzzy partial matching to handle variations like
    "Dallas" vs "Dallas, TX" or "Dallas, Texas".

    Args:
        article_location: Article content or title text.
        location: Incident location from database (e.g., "Dallas").

    Returns:
        True if partial match score is >= 80, False otherwise.
        Returns False if either input is None.

    Examples:
        >>> check_location_match("A shooting in Dallas, TX", "Dallas")
        True
        >>> check_location_match("A shooting in Houston", "Dallas")
        False
        >>> check_location_match(None, "Dallas")
        False
    """
    if article_location is None or location is None:
        return False
    return fuzz.partial_ratio(article_location.lower(), location.lower()) >= 80


def check_name_match(article_name: str | None, name: str | None) -> bool:
    """Check if victim name appears in article text.

    Uses fuzzy partial matching to handle variations like
    "Armando Juarez" vs "Armando L. Juarez".

    Args:
        article_name: Article content or title text.
        name: Civilian or officer name from database.

    Returns:
        True if partial match score is >= 80, False otherwise.
        Returns False if either input is None.

    Examples:
        >>> check_name_match("Officer shot Armando L. Juarez", "Armando Juarez")
        True
        >>> check_name_match("Officer shot John Doe", "Armando Juarez")
        False
        >>> check_name_match(None, "Armando Juarez")
        False
    """
    if article_name is None or name is None:
        return False
    return fuzz.partial_ratio(article_name.lower(), name.lower()) >= 80


def check_date_match(article_date: date | None, incident_date: date | None) -> bool:
    """Check if article date is within ±3 days of incident date.

    Args:
        article_date: Published date parsed from article metadata.
        incident_date: Incident date from database.

    Returns:
        True if dates are within 3 days of each other, False otherwise.
        Returns False if either date is None.

    Examples:
        >>> from datetime import date
        >>> check_date_match(date(2018, 3, 17), date(2018, 3, 15))
        True
        >>> check_date_match(date(2018, 3, 25), date(2018, 3, 15))
        False
        >>> check_date_match(None, date(2018, 3, 15))
        False
    """
    if article_date is None or incident_date is None:
        return False
    diff = abs((article_date - incident_date).days)
    return diff <= 3


def validate_node(state: EnrichmentState) -> EnrichmentState:
    """Validate retrieved articles against incident data.

    Loops through each article in state.retrieved_articles and checks
    date, location, and (optionally) name against the incident record.
    An article passes validation if both date_match and location_match
    are True.

    Args:
        state: Current enrichment state with incident fields and
            retrieved_articles populated by prior nodes.

    Returns:
        Updated EnrichmentState with:
        - validation_results: List of ValidationResult objects.
        - current_stage: Set to PipelineStage.VALIDATE.
        - error_message: Set if validation fails unexpectedly.

    Examples:
        >>> state = EnrichmentState(
        ...     incident_id="142",
        ...     dataset_type=DatasetType.CIVILIANS_SHOT,
        ...     location="Dallas",
        ...     incident_date=date(2018, 3, 15),
        ...     retrieved_articles=[...],
        ... )
        >>> updated = validate_node(state)
        >>> updated.current_stage
        <PipelineStage.VALIDATE: 'validate'>
    """
    try:
        validation_results = []
        for article in state.retrieved_articles:
            result = ValidationResult(article=article)

            result.date_match = check_date_match(
                article.published_date, state.incident_date
            )

            article_text = article.content or article.title
            result.location_match = check_location_match(article_text, state.location)
            if state.civilian_name is None:
                result.victim_name_match = None
            else:
                result.victim_name_match = check_name_match(
                    article_text, state.civilian_name
                )

            if result.date_match and result.location_match:
                result.passed = True
            else:
                result.passed = False
            validation_results.append(result)

        state.validation_results = validation_results
        state.current_stage = PipelineStage.VALIDATE
    except Exception as e:
        state.validation_results = []
        state.error_message = f"Validation failed: {str(e)}"
        state.current_stage = PipelineStage.VALIDATE
    return state
