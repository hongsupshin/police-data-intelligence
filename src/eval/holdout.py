"""Holdout evaluation for the enrichment pipeline.

Runs the pipeline on incidents with known ground truth values and compares
extracted results against database values. Measures extraction accuracy
across 6 evaluable fields using field-specific comparison strategies.

Modules:
    Data models (EvalError, PipelineOutcome, MatchResult, EvalResult,
    FieldMetrics, HoldoutReport), comparison functions, DB queries,
    orchestration, and report formatting.
"""

import json
import logging
import re
import time as time_mod
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, time
from enum import StrEnum
from pathlib import Path

from psycopg2.extensions import connection
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src.agents.state import (
    ConfidenceLevel,
    DatasetType,
    EscalationReason,
    FieldExtraction,
    MediaFeatureField,
    PipelineStage,
)
from src.synthesize.synthesize_node import RAPIDFUZZ_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvalError(StrEnum):
    """Error categories for field evaluation."""

    NO_EXTRACTION = "no_extraction"
    PARSE_ERROR = "parse_error"
    NO_GROUND_TRUTH = "no_ground_truth"


class PipelineOutcome(StrEnum):
    """Pipeline terminal outcome."""

    COMPLETE = "complete"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MatchResult(BaseModel):
    """Single field comparison result.

    Attributes:
        field_name: MediaFeatureField name being compared.
        extracted_value: Value extracted by the pipeline (None if missing).
        ground_truth_value: Database ground truth value (None if missing).
        exact_match: Whether values match exactly after normalization.
        fuzzy_match: Whether values match within fuzzy threshold.
        fuzzy_score: RapidFuzz score (None for exact-match-only fields).
        confidence: Extraction confidence level from the pipeline.
        error: Error category if comparison could not be performed.
    """

    field_name: str
    extracted_value: str | None = None
    ground_truth_value: str | None = None
    exact_match: bool = False
    fuzzy_match: bool = False
    fuzzy_score: float | None = None
    confidence: ConfidenceLevel | None = None
    error: EvalError | None = None


class EvalResult(BaseModel):
    """Evaluation result for a single incident.

    Attributes:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset this incident belongs to.
        pipeline_outcome: Whether pipeline completed or escalated.
        field_results: Comparison results for each evaluable field.
        escalation_reason: Reason for escalation (None if completed).
        elapsed_seconds: Wall-clock seconds for pipeline execution.
        stage_reached: Last pipeline stage before termination.
        validation_failure_summary: Per-check validation failure counts
            from the pipeline's final validate pass (None if unavailable).
    """

    incident_id: int
    dataset_type: DatasetType
    pipeline_outcome: PipelineOutcome
    field_results: list[MatchResult] = Field(default_factory=list)
    escalation_reason: str | None = None
    elapsed_seconds: float = 0.0
    stage_reached: str | None = None
    validation_failure_summary: dict[str, int] | None = None


class HoldoutSample(BaseModel):
    """Metadata for a single holdout evaluation sample.

    Attributes:
        incident_id: TJI incident identifier.
        year: Year of the incident.
        race: Race/ethnicity of the civilian (None if unknown).
        n_eval_fields: Number of non-NULL evaluable ground truth fields.
    """

    incident_id: int
    year: int
    race: str | None = None
    n_eval_fields: int


class FieldMetrics(BaseModel):
    """Aggregated metrics for a single field across all incidents.

    Attributes:
        field_name: MediaFeatureField name.
        n_evaluable: Incidents with ground truth for this field.
        n_extracted: Incidents where pipeline extracted a value.
        n_exact_match: Exact matches among extracted values.
        n_fuzzy_match: Fuzzy matches among extracted values.
        coverage: Fraction of evaluable incidents with extractions.
        exact_accuracy: Exact match rate among extracted values.
        fuzzy_accuracy: Fuzzy match rate among extracted values.
        confidence_breakdown: Accuracy by confidence level.
    """

    field_name: str
    n_evaluable: int
    n_extracted: int
    n_exact_match: int
    n_fuzzy_match: int
    coverage: float
    exact_accuracy: float
    fuzzy_accuracy: float
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)


class HoldoutReport(BaseModel):
    """Complete holdout evaluation report.

    Attributes:
        dataset_type: Which dataset was evaluated.
        n_incidents: Total incidents evaluated.
        n_completed: Incidents where pipeline completed.
        n_escalated: Incidents where pipeline escalated.
        completion_rate: Fraction of incidents that completed.
        field_metrics: Per-field aggregated metrics.
        per_incident: Detailed results for each incident.
        samples: Holdout sample metadata (populated by stratified eval).
        mean_elapsed_seconds: Mean wall-clock seconds per incident.
        total_elapsed_seconds: Total wall-clock seconds for all incidents.
        fairness_metrics: Per-race group metrics (populated by stratified eval).
        validation_failure_totals: Per-check validation failure counts summed
            over escalated incidents (the escalation residual).
    """

    dataset_type: DatasetType
    n_incidents: int
    n_completed: int
    n_escalated: int
    completion_rate: float
    field_metrics: list[FieldMetrics] = Field(default_factory=list)
    per_incident: list[EvalResult] = Field(default_factory=list)
    samples: list[HoldoutSample] = Field(default_factory=list)
    mean_elapsed_seconds: float = 0.0
    total_elapsed_seconds: float = 0.0
    fairness_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    validation_failure_totals: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Comparison Helpers
# ---------------------------------------------------------------------------

_GENDER_WORDS = re.compile(r"\b(male|female|man|woman)\b")

_RACE_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(black|african)\b"), "black"),
    (re.compile(r"\b(hispanic|latino|latina)\b"), "hispanic"),
    (re.compile(r"\b(white|caucasian)\b"), "white"),
    (re.compile(r"\basian\b"), "asian"),
]

TIME_PERIOD_BUCKETS: dict[str, tuple[int, int]] = {
    "morning": (6, 11),
    "afternoon": (12, 17),
    "evening": (18, 21),
    "night": (22, 5),
}

FATAL_KEYWORDS = {"killed", "died", "death", "fatal", "fatally"}
NON_FATAL_KEYWORDS = {"injured", "survived", "wounded", "non-fatal", "nonfatal"}


def _normalize_race(value: str) -> str:
    """Normalize a race string via keyword matching.

    Strips gender words, then checks for race keywords in priority
    order. Unmatched values default to "other".
    """
    lowered = value.strip().lower()
    lowered = _GENDER_WORDS.sub("", lowered).strip()
    lowered = re.sub(r"\s+", " ", lowered)
    for pattern, canonical in _RACE_KEYWORDS:
        if pattern.search(lowered):
            return canonical
    return "other"


def _normalize_outcome_str(value: str) -> str | None:
    """Normalize an outcome string to 'fatal' or 'non-fatal'.

    Returns:
        'fatal', 'non-fatal', or None if unrecognized.
    """
    lowered = value.strip().lower()
    # Check non-fatal first since "non-fatal" contains "fatal"
    if any(kw in lowered for kw in NON_FATAL_KEYWORDS):
        return "non-fatal"
    if any(kw in lowered for kw in FATAL_KEYWORDS):
        return "fatal"
    return None


def _parse_hour(time_str: str) -> int | None:
    """Extract hour from a time string (e.g., '2:30 PM', '14:30', '2 a.m.').

    Returns:
        Hour as 0-23 integer, or None if unparseable.
    """
    # Try HH:MM patterns (24h or 12h)
    match = re.search(
        r"(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?", time_str, re.IGNORECASE
    )
    if match:
        hour = int(match.group(1))
        period = match.group(3)
        if period:
            period = period.lower().replace(".", "")
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
        return hour

    # Try bare hour with AM/PM (e.g., "2 PM", "11 a.m.")
    match = re.search(
        r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)", time_str, re.IGNORECASE
    )
    if match:
        hour = int(match.group(1))
        period = match.group(2).lower().replace(".", "")
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        return hour

    return None


def _hour_in_bucket(hour: int, bucket_name: str) -> bool:
    """Check if an hour falls within a named time period bucket."""
    start, end = TIME_PERIOD_BUCKETS[bucket_name]
    if start <= end:
        return start <= hour <= end
    # Wraps around midnight (night: 22-5)
    return hour >= start or hour <= end


def _detect_period(text: str) -> str | None:
    """Detect a time period keyword in text."""
    lowered = text.strip().lower()
    for period in TIME_PERIOD_BUCKETS:
        if period in lowered:
            return period
    return None


# ---------------------------------------------------------------------------
# Comparison Functions
# ---------------------------------------------------------------------------


def compare_age(
    extracted: str | None, ground_truth: int | None, field_name: str
) -> MatchResult:
    """Compare extracted age string against integer ground truth.

    Args:
        extracted: Age string from pipeline (e.g., "25").
        ground_truth: Age integer from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match only (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=str(ground_truth) if ground_truth is not None else None,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    try:
        extracted_int = int(extracted)
    except (ValueError, TypeError):
        result.error = EvalError.PARSE_ERROR
        return result
    result.exact_match = extracted_int == ground_truth
    result.fuzzy_match = result.exact_match
    return result


def compare_race(
    extracted: str | None, ground_truth: str | None, field_name: str
) -> MatchResult:
    """Compare extracted race against ground truth with alias normalization.

    Args:
        extracted: Race string from pipeline.
        ground_truth: Race string from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match after normalization (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    norm_extracted = _normalize_race(extracted)
    norm_truth = _normalize_race(ground_truth)
    result.exact_match = norm_extracted == norm_truth
    result.fuzzy_match = result.exact_match
    return result


def compare_weapon(
    extracted: str | None, ground_truth: str | None, field_name: str
) -> MatchResult:
    """Compare extracted weapon against ground truth using category normalization.

    Both values are normalized to canonical categories (HANDGUN, RIFLE,
    SHOTGUN, KNIFE, VEHICLE, OTHER, UNKNOWN) before comparison. Ground
    truth that normalizes to ``None`` (missing/empty) is treated as
    ``NO_GROUND_TRUTH``.

    Args:
        extracted: Weapon category from pipeline.
        ground_truth: Weapon description from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match and fuzzy_match set identically.
    """
    from src.synthesize.weapon_similarity import normalize_weapon

    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )
    gt_normalized = normalize_weapon(ground_truth)
    if gt_normalized is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    ext_normalized = normalize_weapon(extracted)
    if ext_normalized is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    matched = ext_normalized == gt_normalized
    result.exact_match = matched
    result.fuzzy_match = matched
    return result


def compare_location(
    extracted: str | None, ground_truth: str | None, field_name: str
) -> MatchResult:
    """Compare extracted location against ground truth using partial fuzzy matching.

    Uses fuzz.partial_ratio with RAPIDFUZZ_THRESHOLD, since extracted
    locations may have different granularity than database addresses.

    Args:
        extracted: Location string from pipeline.
        ground_truth: Address from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with fuzzy_score set.
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    score = fuzz.partial_ratio(extracted.lower(), ground_truth.lower())
    result.fuzzy_score = score
    result.exact_match = extracted.lower().strip() == ground_truth.lower().strip()
    result.fuzzy_match = score >= RAPIDFUZZ_THRESHOLD
    return result


def compare_time(
    extracted: str | None, ground_truth: time | None, field_name: str
) -> MatchResult:
    """Compare extracted time string against database TIME value.

    Primary: parse hour from extracted string, compare +/-2h to ground truth.
    Fallback: detect period keyword and check if ground truth hour falls
    within that period bucket.

    Args:
        extracted: Time description from pipeline (e.g., "2:30 PM", "evening").
        ground_truth: Time object from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=str(ground_truth) if ground_truth is not None else None,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result

    gt_hour = ground_truth.hour

    # Primary: parse hour from extracted string
    extracted_hour = _parse_hour(extracted)
    if extracted_hour is not None:
        diff = abs(extracted_hour - gt_hour)
        # Handle wrap-around midnight
        diff = min(diff, 24 - diff)
        result.exact_match = diff <= 2
        result.fuzzy_match = result.exact_match
        return result

    # Fallback: period bucket matching
    period = _detect_period(extracted)
    if period is not None:
        result.exact_match = _hour_in_bucket(gt_hour, period)
        result.fuzzy_match = result.exact_match
        return result

    result.error = EvalError.PARSE_ERROR
    return result


def compare_outcome(
    extracted: str | None, ground_truth: bool | str | None, field_name: str
) -> MatchResult:
    """Compare extracted outcome against ground truth.

    Normalizes both sides to 'fatal' or 'non-fatal'.
    DB values: civilian_died=True -> fatal, officer_harm='DEATH' -> fatal.

    Args:
        extracted: Outcome description from pipeline.
        ground_truth: Boolean (civilian_died) or string (officer_harm).
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=str(ground_truth) if ground_truth is not None else None,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result

    # Normalize ground truth
    if isinstance(ground_truth, bool):
        gt_normalized = "fatal" if ground_truth else "non-fatal"
    elif isinstance(ground_truth, str):
        gt_normalized = "fatal" if ground_truth.upper() == "DEATH" else "non-fatal"
    else:
        result.error = EvalError.PARSE_ERROR
        return result

    # Normalize extracted
    ext_normalized = _normalize_outcome_str(extracted)
    if ext_normalized is None:
        result.error = EvalError.PARSE_ERROR
        return result

    result.exact_match = ext_normalized == gt_normalized
    result.fuzzy_match = result.exact_match
    return result


# ---------------------------------------------------------------------------
# Field-to-Comparator Mapping
# ---------------------------------------------------------------------------

FIELD_COMPARATORS: dict[MediaFeatureField, Callable] = {
    MediaFeatureField.CIVILIAN_AGE: compare_age,
    MediaFeatureField.CIVILIAN_RACE: compare_race,
    MediaFeatureField.WEAPON: compare_weapon,
    MediaFeatureField.LOCATION_DETAIL: compare_location,
    MediaFeatureField.TIME_OF_DAY: compare_time,
    MediaFeatureField.OUTCOME: compare_outcome,
}

EVAL_FIELDS: set[MediaFeatureField] = set(FIELD_COMPARATORS.keys())

# Ground truth DB column -> MediaFeatureField mapping per dataset
_CIVILIANS_GT_MAPPING: dict[str, MediaFeatureField] = {
    "age": MediaFeatureField.CIVILIAN_AGE,
    "race": MediaFeatureField.CIVILIAN_RACE,
    "weapon_reported_by_media": MediaFeatureField.WEAPON,
    "location_city": MediaFeatureField.LOCATION_DETAIL,
    "time_incident": MediaFeatureField.TIME_OF_DAY,
    "civilian_died": MediaFeatureField.OUTCOME,
}

_OFFICERS_GT_MAPPING: dict[str, MediaFeatureField] = {
    "age": MediaFeatureField.CIVILIAN_AGE,
    "race": MediaFeatureField.CIVILIAN_RACE,
    "location_city": MediaFeatureField.LOCATION_DETAIL,
    "officer_harm": MediaFeatureField.OUTCOME,
}

_DATASET_GT_MAPPINGS: dict[DatasetType, dict[str, MediaFeatureField]] = {
    DatasetType.CIVILIANS_SHOT: _CIVILIANS_GT_MAPPING,
    DatasetType.OFFICERS_SHOT: _OFFICERS_GT_MAPPING,
}


def comparators_for_dataset(
    dataset_type: DatasetType,
) -> dict[MediaFeatureField, Callable]:
    """Return the comparators evaluable for a dataset.

    Restricts FIELD_COMPARATORS to fields that have ground truth for the
    given dataset (e.g., officers_shot has no weapon/time_of_day columns),
    preserving FIELD_COMPARATORS ordering. This keeps officers reports from
    listing phantom civilian-only fields with zero evaluable incidents.

    Args:
        dataset_type: Which dataset is being evaluated.

    Returns:
        Subset of FIELD_COMPARATORS whose fields are mapped for the dataset.
    """
    mapped_fields = set(_DATASET_GT_MAPPINGS[dataset_type].values())
    return {
        field: comparator
        for field, comparator in FIELD_COMPARATORS.items()
        if field in mapped_fields
    }


# ---------------------------------------------------------------------------
# DB Queries
# ---------------------------------------------------------------------------


def fetch_ground_truth(
    conn: connection, incident_id: int, dataset_type: DatasetType
) -> dict[str, object]:
    """Fetch ground truth values for evaluable fields from the database.

    Uses dataset-specific queries to retrieve the 6 (civilians) or 4
    (officers) holdout fields that the pipeline never sees during
    extraction.

    Args:
        conn: Active PostgreSQL connection.
        incident_id: TJI incident identifier.
        dataset_type: Which dataset to query.

    Returns:
        Dict mapping MediaFeatureField values to ground truth values.
        Values may be None if the database column is NULL.

    Raises:
        KeyError: If incident_id not found in database.
    """
    cursor = conn.cursor()

    if dataset_type == DatasetType.CIVILIANS_SHOT:
        query = """
            SELECT c.age, c.race,
                   i.weapon_reported_by_media,
                   COALESCE(i.incident_city, i.incident_county) AS location_city,
                   i.time_incident, v.civilian_died
            FROM incidents_civilians_shot i
            LEFT JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN civilians c ON v.civilian_id = c.civilian_id
            WHERE i.incident_id = %s LIMIT 1;
        """
        cursor.execute(query, (incident_id,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Incident {incident_id} not found in civilians_shot")
        columns = [
            "age", "race", "weapon_reported_by_media",
            "location_city", "time_incident", "civilian_died",
        ]
        raw = dict(zip(columns, row))
        return {
            _CIVILIANS_GT_MAPPING[col].value: val
            for col, val in raw.items()
        }
    else:
        query = """
            SELECT c.age, c.race,
                   COALESCE(i.incident_city, i.incident_county) AS location_city,
                   v.officer_harm
            FROM incidents_officers_shot i
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id
            LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
            LEFT JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            WHERE i.incident_id = %s LIMIT 1;
        """
        cursor.execute(query, (incident_id,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Incident {incident_id} not found in officers_shot")
        columns = ["age", "race", "location_city", "officer_harm"]
        raw = dict(zip(columns, row))
        return {
            _OFFICERS_GT_MAPPING[col].value: val
            for col, val in raw.items()
        }


def select_holdout_incidents(
    conn: connection,
    dataset_type: DatasetType,
    min_fields: int = 2,
    limit: int = 20,
) -> list[int]:
    """Select incidents with the most populated eval fields.

    Ranks incidents by number of non-NULL evaluable columns and requires
    that date_incident and incident_city are populated (pipeline minimum).

    Args:
        conn: Active PostgreSQL connection.
        dataset_type: Which dataset to query.
        min_fields: Minimum number of non-NULL eval fields required.
        limit: Maximum number of incidents to return.

    Returns:
        List of incident IDs ordered by field count descending.
    """
    cursor = conn.cursor()

    if dataset_type == DatasetType.CIVILIANS_SHOT:
        query = """
            SELECT i.incident_id,
                   (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.weapon_reported_by_media IS NOT NULL
                         THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.time_incident IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.civilian_died IS NOT NULL THEN 1 ELSE 0 END
                   ) AS field_count
            FROM incidents_civilians_shot i
            LEFT JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN civilians c ON v.civilian_id = c.civilian_id
            WHERE i.date_incident IS NOT NULL
              AND i.incident_city IS NOT NULL
            GROUP BY i.incident_id, c.age, c.race,
                     i.weapon_reported_by_media, i.incident_address,
                     i.time_incident, v.civilian_died
            HAVING (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.weapon_reported_by_media IS NOT NULL
                         THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.time_incident IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.civilian_died IS NOT NULL THEN 1 ELSE 0 END
                   ) >= %s
            ORDER BY field_count DESC
            LIMIT %s;
        """
    else:
        query = """
            SELECT i.incident_id,
                   (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.officer_harm IS NOT NULL THEN 1 ELSE 0 END
                   ) AS field_count
            FROM incidents_officers_shot i
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id
            LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
            LEFT JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            WHERE i.date_incident IS NOT NULL
              AND i.incident_city IS NOT NULL
            GROUP BY i.incident_id, c.age, c.race,
                     i.incident_address, v.officer_harm
            HAVING (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.officer_harm IS NOT NULL THEN 1 ELSE 0 END
                   ) >= %s
            ORDER BY field_count DESC
            LIMIT %s;
        """

    cursor.execute(query, (min_fields, limit))
    return [row[0] for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


DEV_SET_IDS: set[int] = {3710, 5388, 3630, 3669, 833, 792, 5168, 697, 3744, 330}


def _infer_stage_reached(
    pipeline_outcome: PipelineOutcome,
    escalation_reason: str | None,
) -> str:
    """Infer which pipeline stage was reached before termination.

    Args:
        pipeline_outcome: Whether pipeline completed or escalated.
        escalation_reason: Escalation reason string (None if completed).

    Returns:
        Pipeline stage name (e.g., "search", "validate", "synthesize", "complete").
    """
    if pipeline_outcome == PipelineOutcome.COMPLETE:
        return "complete"
    if escalation_reason is None:
        return "unknown"
    reason = escalation_reason.lower()
    if reason == EscalationReason.MAX_RETRIES.value:
        return "search"
    if reason == EscalationReason.VALIDATION_ERROR.value:
        return "validate"
    if reason in (
        EscalationReason.CONFLICT.value,
        EscalationReason.MERGE_ERROR.value,
        EscalationReason.INSUFFICIENT_SOURCES.value,
    ):
        return "synthesize"
    if reason == EscalationReason.EXTRACTION_ERROR.value:
        return "load"
    return "unknown"


def select_holdout_stratified(
    conn: connection,
    dataset_type: DatasetType,
    min_fields: int = 2,
    limit: int = 40,
    exclude_dev_set: bool = True,
) -> list[HoldoutSample]:
    """Select holdout incidents with proportional year stratification.

    Selects incidents proportionally across years (2015-2019), ensuring
    a minimum of 4 per stratum. Excludes dev-set incidents by default.

    Args:
        conn: Active PostgreSQL connection.
        dataset_type: Which dataset to query.
        min_fields: Minimum number of non-NULL eval fields required.
        limit: Total number of incidents to return.
        exclude_dev_set: Whether to exclude dev-set IDs.

    Returns:
        List of HoldoutSample with year/race metadata.
    """
    cursor = conn.cursor()

    exclusion_clause = ""
    if exclude_dev_set and DEV_SET_IDS:
        ids_str = ", ".join(str(i) for i in DEV_SET_IDS)
        exclusion_clause = f"AND i.incident_id NOT IN ({ids_str})"

    if dataset_type == DatasetType.CIVILIANS_SHOT:
        query = f"""
            SELECT i.incident_id,
                   EXTRACT(YEAR FROM i.date_incident)::int AS year,
                   c.race,
                   (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.weapon_reported_by_media IS NOT NULL
                         THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.time_incident IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.civilian_died IS NOT NULL THEN 1 ELSE 0 END
                   ) AS field_count
            FROM incidents_civilians_shot i
            LEFT JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN civilians c ON v.civilian_id = c.civilian_id
            WHERE i.date_incident IS NOT NULL
              AND i.incident_city IS NOT NULL
              {exclusion_clause}
            GROUP BY i.incident_id, i.date_incident, c.race, c.age,
                     i.weapon_reported_by_media, i.incident_address,
                     i.time_incident, v.civilian_died
            HAVING (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.weapon_reported_by_media IS NOT NULL
                         THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.time_incident IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.civilian_died IS NOT NULL THEN 1 ELSE 0 END
                   ) >= %s
            ORDER BY field_count DESC, i.incident_id ASC;
        """
    else:
        query = f"""
            SELECT i.incident_id,
                   EXTRACT(YEAR FROM i.date_incident)::int AS year,
                   c.race,
                   (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.officer_harm IS NOT NULL THEN 1 ELSE 0 END
                   ) AS field_count
            FROM incidents_officers_shot i
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id
            LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
            LEFT JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            WHERE i.date_incident IS NOT NULL
              AND i.incident_city IS NOT NULL
              {exclusion_clause}
            GROUP BY i.incident_id, i.date_incident, c.race, c.age,
                     i.incident_address, v.officer_harm
            HAVING (CASE WHEN c.age IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN c.race IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN i.incident_address IS NOT NULL THEN 1 ELSE 0 END
                  + CASE WHEN v.officer_harm IS NOT NULL THEN 1 ELSE 0 END
                   ) >= %s
            ORDER BY field_count DESC, i.incident_id ASC;
        """

    cursor.execute(query, (min_fields,))
    rows = cursor.fetchall()

    # Group by year for proportional allocation
    by_year: dict[int, list[tuple]] = defaultdict(list)
    for row in rows:
        by_year[row[1]].append(row)

    # Proportional allocation with minimum 4 per stratum
    total_available = sum(len(v) for v in by_year.values())
    selected: list[HoldoutSample] = []
    min_per_stratum = 4

    for year in sorted(by_year.keys()):
        year_rows = by_year[year]
        # Proportional count, at least min_per_stratum
        proportion = len(year_rows) / total_available if total_available > 0 else 0
        n_for_year = max(min_per_stratum, round(proportion * limit))
        n_for_year = min(n_for_year, len(year_rows))
        for row in year_rows[:n_for_year]:
            selected.append(
                HoldoutSample(
                    incident_id=row[0],
                    year=row[1],
                    race=row[2],
                    n_eval_fields=row[3],
                )
            )
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    return selected[:limit]


def evaluate_single(
    incident_id: int,
    dataset_type: DatasetType,
    ground_truth: dict[str, object],
) -> EvalResult:
    """Evaluate a single incident by running the pipeline and comparing results.

    Calls the real pipeline via src.run.run(), then compares each
    evaluable field against ground truth using field-specific comparators.

    Args:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset this incident belongs to.
        ground_truth: Dict mapping field names to ground truth values.

    Returns:
        EvalResult with comparison results for all evaluable fields.
    """
    from src.run import run

    start = time_mod.monotonic()
    result = run(str(incident_id), dataset_type.value)
    elapsed = time_mod.monotonic() - start

    pipeline_outcome = (
        PipelineOutcome.COMPLETE
        if result.get("current_stage") == PipelineStage.COMPLETE.value
        else PipelineOutcome.ESCALATE
    )

    escalation_reason = (
        str(result.get("escalation_reason"))
        if result.get("escalation_reason")
        else None
    )

    # Build lookup from extracted fields
    extracted_lookup: dict[str, FieldExtraction] = {}
    for field in result.get("extracted_fields", []):
        if isinstance(field, FieldExtraction):
            extracted_lookup[field.field_name] = field
        elif isinstance(field, dict):
            extracted_lookup[field["field_name"]] = FieldExtraction(**field)

    field_results: list[MatchResult] = []
    for field_enum, comparator in comparators_for_dataset(dataset_type).items():
        gt_value = ground_truth.get(field_enum.value)

        extraction = extracted_lookup.get(field_enum.value)
        extracted_value = extraction.value if extraction else None

        match_result = comparator(extracted_value, gt_value, field_enum.value)

        if extraction and match_result.error is None:
            match_result.confidence = extraction.confidence

        field_results.append(match_result)

    vfs = result.get("validation_failure_summary")

    return EvalResult(
        incident_id=incident_id,
        dataset_type=dataset_type,
        pipeline_outcome=pipeline_outcome,
        field_results=field_results,
        escalation_reason=escalation_reason,
        elapsed_seconds=round(elapsed, 2),
        stage_reached=_infer_stage_reached(pipeline_outcome, escalation_reason),
        validation_failure_summary=vfs if isinstance(vfs, dict) else None,
    )


def aggregate_metrics(
    eval_results: list[EvalResult],
    dataset_type: DatasetType | None = None,
) -> list[FieldMetrics]:
    """Aggregate per-field metrics across all evaluated incidents.

    Groups MatchResults by field, excludes NO_GROUND_TRUTH from accuracy
    denominators, and computes coverage, accuracy, and confidence breakdown.

    Args:
        eval_results: List of EvalResult objects from evaluate_single.
        dataset_type: When given, report only the fields evaluable for that
            dataset. When None, report all FIELD_COMPARATORS fields.

    Returns:
        List of FieldMetrics, one per evaluable field.
    """
    comparators = (
        comparators_for_dataset(dataset_type)
        if dataset_type is not None
        else FIELD_COMPARATORS
    )

    results_by_field: dict[str, list[MatchResult]] = defaultdict(list)
    for er in eval_results:
        for mr in er.field_results:
            results_by_field[mr.field_name].append(mr)

    metrics: list[FieldMetrics] = []
    for field_enum in comparators:
        field_name = field_enum.value
        all_results = results_by_field.get(field_name, [])

        # Evaluable = has ground truth (not NO_GROUND_TRUTH)
        evaluable = [
            r for r in all_results if r.error != EvalError.NO_GROUND_TRUTH
        ]
        n_evaluable = len(evaluable)

        # Extracted = evaluable AND no error
        extracted = [r for r in evaluable if r.error is None]
        n_extracted = len(extracted)

        n_exact = sum(1 for r in extracted if r.exact_match)
        n_fuzzy = sum(1 for r in extracted if r.fuzzy_match)

        coverage = n_extracted / n_evaluable if n_evaluable > 0 else 0.0
        exact_acc = n_exact / n_extracted if n_extracted > 0 else 0.0
        fuzzy_acc = n_fuzzy / n_extracted if n_extracted > 0 else 0.0

        # Confidence breakdown: accuracy per confidence level
        conf_breakdown: dict[str, float] = {}
        for level in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
        ]:
            by_conf = [r for r in extracted if r.confidence == level]
            if by_conf:
                conf_breakdown[level.value] = (
                    sum(1 for r in by_conf if r.exact_match) / len(by_conf)
                )

        metrics.append(
            FieldMetrics(
                field_name=field_name,
                n_evaluable=n_evaluable,
                n_extracted=n_extracted,
                n_exact_match=n_exact,
                n_fuzzy_match=n_fuzzy,
                coverage=coverage,
                exact_accuracy=exact_acc,
                fuzzy_accuracy=fuzzy_acc,
                confidence_breakdown=conf_breakdown,
            )
        )

    return metrics


def sum_validation_failures(eval_results: list[EvalResult]) -> dict[str, int]:
    """Sum per-check validation-failure counts over escalated incidents.

    Aggregates the per-incident ``validation_failure_summary`` for escalated
    incidents only, producing the escalation residual that the deterministic
    fixes (and the later enable-gate) are read against.

    Args:
        eval_results: List of EvalResult objects from evaluate_single.

    Returns:
        Dict of summed counts keyed by the summary keys (total, passed,
        excluded, date_fail, location_fail, name_fail). Empty if no
        escalated incident carried a summary.
    """
    totals: dict[str, int] = {}
    for er in eval_results:
        if er.pipeline_outcome != PipelineOutcome.ESCALATE:
            continue
        if not er.validation_failure_summary:
            continue
        for key, value in er.validation_failure_summary.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def evaluate_holdout(
    dataset_type: DatasetType,
    limit: int = 20,
    min_fields: int = 2,
    incident_ids: list[int] | None = None,
) -> HoldoutReport:
    """Run holdout evaluation on a dataset.

    Connects to the database, selects incidents with the most populated
    eval fields, runs the pipeline on each, and aggregates results.

    Args:
        dataset_type: Which dataset to evaluate.
        limit: Maximum number of incidents to evaluate.
        min_fields: Minimum non-NULL eval fields per incident.
        incident_ids: Specific incident IDs to evaluate. When provided,
            skips DB-based incident selection.

    Returns:
        Complete HoldoutReport with per-field metrics and per-incident
        results.
    """
    from src.database.connection import get_connection

    conn = get_connection()
    try:
        if incident_ids is None:
            incident_ids = select_holdout_incidents(
                conn, dataset_type, min_fields, limit
            )

        eval_results: list[EvalResult] = []
        for incident_id in incident_ids:
            gt = fetch_ground_truth(conn, incident_id, dataset_type)
            result = evaluate_single(incident_id, dataset_type, gt)
            eval_results.append(result)
    finally:
        conn.close()

    n_completed = sum(
        1 for r in eval_results
        if r.pipeline_outcome == PipelineOutcome.COMPLETE
    )
    n_escalated = sum(
        1 for r in eval_results
        if r.pipeline_outcome == PipelineOutcome.ESCALATE
    )
    n_incidents = len(eval_results)

    return HoldoutReport(
        dataset_type=dataset_type,
        n_incidents=n_incidents,
        n_completed=n_completed,
        n_escalated=n_escalated,
        completion_rate=n_completed / n_incidents if n_incidents > 0 else 0.0,
        field_metrics=aggregate_metrics(eval_results, dataset_type),
        per_incident=eval_results,
        validation_failure_totals=sum_validation_failures(eval_results),
    )


def compute_fairness_metrics(
    eval_results: list[EvalResult],
    samples: list[HoldoutSample],
) -> dict[str, dict[str, float]]:
    """Compute per-race-group pipeline metrics for fairness analysis.

    Groups results by race (from HoldoutSample metadata) and computes
    completion rate, pipeline reach rate, and mean exact accuracy per group.

    Args:
        eval_results: List of EvalResult objects.
        samples: List of HoldoutSample with race metadata.

    Returns:
        Dict mapping race group to metrics dict with keys:
        n, completion_rate, mean_exact_accuracy.
    """
    # Build incident_id -> race lookup
    race_lookup: dict[int, str] = {}
    for sample in samples:
        race_lookup[sample.incident_id] = (sample.race or "unknown").lower()

    # Group results by race
    by_race: dict[str, list[EvalResult]] = defaultdict(list)
    for er in eval_results:
        race = race_lookup.get(er.incident_id, "unknown")
        by_race[race].append(er)

    metrics: dict[str, dict[str, float]] = {}
    for race, results in sorted(by_race.items()):
        n = len(results)
        n_completed = sum(
            1 for r in results if r.pipeline_outcome == PipelineOutcome.COMPLETE
        )
        # Mean exact accuracy across completed incidents
        exact_accs: list[float] = []
        for r in results:
            evaluated = [
                fr for fr in r.field_results
                if fr.error is None
            ]
            if evaluated:
                acc = sum(1 for fr in evaluated if fr.exact_match) / len(evaluated)
                exact_accs.append(acc)

        metrics[race] = {
            "n": float(n),
            "completion_rate": n_completed / n if n > 0 else 0.0,
            "mean_exact_accuracy": (
                sum(exact_accs) / len(exact_accs) if exact_accs else 0.0
            ),
        }

    return metrics


def evaluate_holdout_stratified(
    dataset_type: DatasetType,
    limit: int = 40,
    min_fields: int = 2,
    exclude_dev_set: bool = True,
) -> HoldoutReport:
    """Run stratified holdout evaluation on a dataset.

    Uses year-stratified sampling, tracks timing per incident,
    and computes fairness metrics by race group.

    Args:
        dataset_type: Which dataset to evaluate.
        limit: Total number of incidents to evaluate.
        min_fields: Minimum non-NULL eval fields per incident.
        exclude_dev_set: Whether to exclude dev-set IDs.

    Returns:
        Complete HoldoutReport with stratified samples, timing,
        and fairness metrics.
    """
    from src.database.connection import get_connection

    conn = get_connection()
    try:
        samples = select_holdout_stratified(
            conn, dataset_type, min_fields, limit, exclude_dev_set
        )

        eval_results: list[EvalResult] = []
        for i, sample in enumerate(samples):
            logger.info(
                "Evaluating %d/%d: incident_id=%d (year=%d)",
                i + 1,
                len(samples),
                sample.incident_id,
                sample.year,
            )
            gt = fetch_ground_truth(conn, sample.incident_id, dataset_type)
            result = evaluate_single(sample.incident_id, dataset_type, gt)
            eval_results.append(result)
    finally:
        conn.close()

    n_completed = sum(
        1 for r in eval_results
        if r.pipeline_outcome == PipelineOutcome.COMPLETE
    )
    n_escalated = sum(
        1 for r in eval_results
        if r.pipeline_outcome == PipelineOutcome.ESCALATE
    )
    n_incidents = len(eval_results)
    total_elapsed = sum(r.elapsed_seconds for r in eval_results)
    mean_elapsed = total_elapsed / n_incidents if n_incidents > 0 else 0.0

    return HoldoutReport(
        dataset_type=dataset_type,
        n_incidents=n_incidents,
        n_completed=n_completed,
        n_escalated=n_escalated,
        completion_rate=n_completed / n_incidents if n_incidents > 0 else 0.0,
        field_metrics=aggregate_metrics(eval_results, dataset_type),
        per_incident=eval_results,
        samples=samples,
        mean_elapsed_seconds=round(mean_elapsed, 2),
        total_elapsed_seconds=round(total_elapsed, 2),
        fairness_metrics=compute_fairness_metrics(eval_results, samples),
        validation_failure_totals=sum_validation_failures(eval_results),
    )


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------


def _format_pct(value: float | None) -> str:
    """Format a float as a percentage string, or '-' if None."""
    if value is None:
        return "-"
    return f"{value:.0%}"


def print_report(report: HoldoutReport) -> None:
    """Print a formatted evaluation report table to stdout.

    Args:
        report: Complete holdout evaluation report.
    """
    header = (
        f"{'Field':<20} | {'N':>3} | {'Coverage':>8} | {'Exact':>5} "
        f"| {'Fuzzy':>5} | {'HIGH acc':>8} | {'MED acc':>7}"
    )
    separator = "-" * len(header)

    print(f"\nHoldout Evaluation: {report.dataset_type.value}")
    print(separator)
    print(header)
    print(separator)

    for fm in report.field_metrics:
        high_acc = _format_pct(fm.confidence_breakdown.get("high"))
        med_acc = _format_pct(fm.confidence_breakdown.get("medium"))
        print(
            f"{fm.field_name:<20} | {fm.n_evaluable:>3} | "
            f"{fm.coverage:>7.0%} | {fm.exact_accuracy:>4.0%} | "
            f"{fm.fuzzy_accuracy:>4.0%} | {high_acc:>8} | {med_acc:>7}"
        )

    print(separator)
    print(
        f"Pipeline completion rate: {report.completion_rate:.0%} "
        f"({report.n_completed}/{report.n_incidents} completed, "
        f"{report.n_escalated} escalated)"
    )
    if report.validation_failure_totals:
        residual = ", ".join(
            f"{key}={value}"
            for key, value in report.validation_failure_totals.items()
        )
        print(f"Escalation validation residual: {residual}")
    print()


def save_report(report: HoldoutReport) -> str:
    """Save the evaluation report as JSON.

    Args:
        report: Complete holdout evaluation report.

    Returns:
        Path to the saved JSON file.
    """
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"holdout_{report.dataset_type.value}_{timestamp}.json"
    filepath = output_dir / filename

    data = json.loads(report.model_dump_json())
    filepath.write_text(json.dumps(data, indent=2))

    return str(filepath)
