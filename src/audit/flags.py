"""Discrepancy flag model and severity assignment.

A DiscrepancyFlag records one field where news coverage contradicts the
official record. Severity encodes accountability relevance (who was
involved and whether they survived matter more than location granularity);
extraction confidence gates whether a flag is reportable, it does not rank
severity.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from src.agents.state import (
    ConfidenceLevel,
    DatasetType,
    FieldExtraction,
    MediaFeatureField,
)
from src.eval.comparators import MatchResult

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FlagSeverity(StrEnum):
    """Accountability relevance of a flagged mismatch."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VerificationStatus(StrEnum):
    """Human verification outcome for a flag (worksheet round-trip).

    PENDING until a human verifies. The three resolved causes mirror the
    audit's error model: the official record is wrong (DB_ERROR), the news
    coverage is wrong (NEWS_ERROR), or the pipeline misread the news
    (EXTRACTION_ERROR). UNRESOLVED means a human looked and could not
    determine the cause.
    """

    PENDING = "pending"
    DB_ERROR = "db_error"
    NEWS_ERROR = "news_error"
    EXTRACTION_ERROR = "extraction_error"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Severity scheme
# ---------------------------------------------------------------------------

# Name/outcome/race mismatches are the accountability-relevant errors (wrong
# person, wrong survival status, misrecorded race). Weapon/age/time are
# consequential but noisier categories. Location is granularity-prone (city
# vs street address), so contradictions there are usually benign.
SEVERITY_BY_FIELD: dict[MediaFeatureField, FlagSeverity] = {
    MediaFeatureField.OFFICER_NAME: FlagSeverity.HIGH,
    MediaFeatureField.CIVILIAN_NAME: FlagSeverity.HIGH,
    MediaFeatureField.OUTCOME: FlagSeverity.HIGH,
    MediaFeatureField.CIVILIAN_RACE: FlagSeverity.HIGH,
    MediaFeatureField.WEAPON: FlagSeverity.MEDIUM,
    MediaFeatureField.CIVILIAN_AGE: FlagSeverity.MEDIUM,
    MediaFeatureField.TIME_OF_DAY: FlagSeverity.MEDIUM,
    MediaFeatureField.LOCATION_DETAIL: FlagSeverity.LOW,
}

# An age difference this small is rounding/birthday noise, not a discrepancy
# worth HIGH/MEDIUM attention.
AGE_LOW_SEVERITY_MAX_DIFF = 1

# Flags below this confidence are emitted with suppressed=True: kept in the
# JSON report for taxonomy analysis, excluded from the verification
# worksheet and headline counts.
REPORTABLE_CONFIDENCE: frozenset[ConfidenceLevel] = frozenset(
    {ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM}
)


# ---------------------------------------------------------------------------
# Flag model
# ---------------------------------------------------------------------------


class DiscrepancyFlag(BaseModel):
    """One field where news coverage contradicts the official record.

    Attributes:
        flag_id: Stable identifier ``{dataset}_{incident_id}_{field}`` used
            as the join key for the verification worksheet round-trip.
        incident_id: TJI incident identifier.
        dataset_type: Which dataset the incident belongs to.
        field: MediaFeatureField value that mismatched.
        db_value: Official record value (the audit object).
        news_value: Value extracted from news coverage.
        sources: Source URLs backing the news value.
        source_quotes: Exact quotes backing the news value.
        extraction_confidence: Pipeline confidence in the news value.
        fuzzy_score: Comparator fuzzy score (None for exact-only fields).
        severity: Accountability relevance of the mismatch.
        reference_source: Which official record was compared against
            (default TJI DB; a raw-OAG provider can plug in later).
        suppressed: True when extraction confidence is below the
            reportable bar; excluded from worksheet and headline counts.
        verification_status: Human verification outcome (pending until
            verified).
    """

    flag_id: str
    incident_id: int
    dataset_type: DatasetType
    field: str
    db_value: str | None
    news_value: str | None
    sources: list[str] = Field(default_factory=list)
    source_quotes: list[str] = Field(default_factory=list)
    extraction_confidence: ConfidenceLevel
    fuzzy_score: float | None = None
    severity: FlagSeverity
    reference_source: str = "tji_db"
    suppressed: bool = False
    verification_status: VerificationStatus = VerificationStatus.PENDING


# ---------------------------------------------------------------------------
# Severity assignment and flag construction
# ---------------------------------------------------------------------------


def _age_difference(match: MatchResult) -> int | None:
    """Compute the absolute age difference from a MatchResult, if parseable.

    Args:
        match: Comparison result whose values may be age strings.

    Returns:
        Absolute difference of the two ages, or None if either side does
        not parse as an integer.
    """
    try:
        extracted = int(match.extracted_value)  # type: ignore[arg-type]
        ground_truth = int(match.ground_truth_value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    return abs(extracted - ground_truth)


def assign_severity(
    field: MediaFeatureField,
    match: MatchResult,
    dataset_type: DatasetType,
) -> FlagSeverity:
    """Assign a severity to a flagged mismatch.

    Base severity comes from SEVERITY_BY_FIELD, with two demotions:
    an age difference of at most AGE_LOW_SEVERITY_MAX_DIFF is LOW
    (rounding/birthday noise), and officers_shot race mismatches are
    MEDIUM because the race verifier only runs on civilians_shot, so
    that extraction is unverified.

    Args:
        field: The mismatched field.
        match: The comparison result for the field.
        dataset_type: Which dataset the incident belongs to.

    Returns:
        Severity for the flag.
    """
    severity = SEVERITY_BY_FIELD[field]
    if field == MediaFeatureField.CIVILIAN_AGE:
        diff = _age_difference(match)
        if diff is not None and diff <= AGE_LOW_SEVERITY_MAX_DIFF:
            return FlagSeverity.LOW
    if (
        field == MediaFeatureField.CIVILIAN_RACE
        and dataset_type == DatasetType.OFFICERS_SHOT
    ):
        return FlagSeverity.MEDIUM
    return severity


def make_flag_id(
    dataset_type: DatasetType, incident_id: int, field: MediaFeatureField
) -> str:
    """Build the stable worksheet join key for a flag.

    Args:
        dataset_type: Which dataset the incident belongs to.
        incident_id: TJI incident identifier.
        field: The mismatched field.

    Returns:
        Identifier of the form ``{dataset}_{incident_id}_{field}``.
    """
    return f"{dataset_type.value}_{incident_id}_{field.value}"


def build_flag(
    incident_id: int,
    dataset_type: DatasetType,
    field: MediaFeatureField,
    match: MatchResult,
    extraction: FieldExtraction,
    reference_source: str,
) -> DiscrepancyFlag:
    """Assemble a DiscrepancyFlag from a contradicting comparison result.

    Args:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset the incident belongs to.
        field: The mismatched field.
        match: Comparison result (must be a genuine contradiction: both
            values present, no exact or fuzzy match, no error).
        extraction: The pipeline extraction backing the news value.
        reference_source: Which official record was compared against.

    Returns:
        The flag, with ``suppressed=True`` when extraction confidence is
        below the reportable bar.
    """
    return DiscrepancyFlag(
        flag_id=make_flag_id(dataset_type, incident_id, field),
        incident_id=incident_id,
        dataset_type=dataset_type,
        field=field.value,
        db_value=match.ground_truth_value,
        news_value=match.extracted_value,
        sources=list(extraction.sources),
        source_quotes=list(extraction.source_quotes),
        extraction_confidence=extraction.confidence,
        fuzzy_score=match.fuzzy_score,
        severity=assign_severity(field, match, dataset_type),
        reference_source=reference_source,
        suppressed=extraction.confidence not in REPORTABLE_CONFIDENCE,
    )
