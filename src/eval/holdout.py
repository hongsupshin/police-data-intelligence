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
import re
from collections import defaultdict
from datetime import datetime, time
from enum import StrEnum
from pathlib import Path

from psycopg2.extensions import connection
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src.agents.state import (
    ConfidenceLevel,
    DatasetType,
    FieldExtraction,
    MediaFeatureField,
)
from src.merge.merge_node import RAPIDFUZZ_THRESHOLD

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
    """

    incident_id: int
    dataset_type: DatasetType
    pipeline_outcome: PipelineOutcome
    field_results: list[MatchResult] = Field(default_factory=list)
    escalation_reason: str | None = None


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
    """

    dataset_type: DatasetType
    n_incidents: int
    n_completed: int
    n_escalated: int
    completion_rate: float
    field_metrics: list[FieldMetrics] = Field(default_factory=list)
    per_incident: list[EvalResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparison Helpers
# ---------------------------------------------------------------------------

RACE_ALIASES: dict[str, str] = {
    "african american": "black",
    "african-american": "black",
    "caucasian": "white",
    "latino": "hispanic",
    "latina": "hispanic",
    "latin": "hispanic",
}

TIME_PERIOD_BUCKETS: dict[str, tuple[int, int]] = {
    "morning": (6, 11),
    "afternoon": (12, 17),
    "evening": (18, 21),
    "night": (22, 5),
}

FATAL_KEYWORDS = {"killed", "died", "death", "fatal", "fatally"}
NON_FATAL_KEYWORDS = {"injured", "survived", "wounded", "non-fatal", "nonfatal"}


def _normalize_race(value: str) -> str:
    """Normalize a race string via alias mapping and lowercasing."""
    lowered = value.strip().lower()
    return RACE_ALIASES.get(lowered, lowered)


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
    """Compare extracted weapon against ground truth using fuzzy matching.

    Uses fuzz.ratio and fuzz.partial_ratio with RAPIDFUZZ_THRESHOLD.

    Args:
        extracted: Weapon description from pipeline.
        ground_truth: Weapon description from database.
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
    ratio = fuzz.ratio(extracted.lower(), ground_truth.lower())
    partial = fuzz.partial_ratio(extracted.lower(), ground_truth.lower())
    best = max(ratio, partial)
    result.fuzzy_score = best
    result.exact_match = extracted.lower().strip() == ground_truth.lower().strip()
    result.fuzzy_match = best >= RAPIDFUZZ_THRESHOLD
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

FIELD_COMPARATORS: dict[MediaFeatureField, callable] = {
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
    "incident_address": MediaFeatureField.LOCATION_DETAIL,
    "time_incident": MediaFeatureField.TIME_OF_DAY,
    "civilian_died": MediaFeatureField.OUTCOME,
}

_OFFICERS_GT_MAPPING: dict[str, MediaFeatureField] = {
    "age": MediaFeatureField.CIVILIAN_AGE,
    "race": MediaFeatureField.CIVILIAN_RACE,
    "incident_address": MediaFeatureField.LOCATION_DETAIL,
    "officer_harm": MediaFeatureField.OUTCOME,
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
                   i.weapon_reported_by_media, i.incident_address,
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
            "incident_address", "time_incident", "civilian_died",
        ]
        raw = dict(zip(columns, row))
        return {
            _CIVILIANS_GT_MAPPING[col].value: val
            for col, val in raw.items()
        }
    else:
        query = """
            SELECT c.age, c.race,
                   i.incident_address,
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
        columns = ["age", "race", "incident_address", "officer_harm"]
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

    result = run(str(incident_id), dataset_type.value)

    pipeline_outcome = (
        PipelineOutcome.ESCALATE
        if result.get("requires_human_review", False)
        else PipelineOutcome.COMPLETE
    )

    # Build lookup from extracted fields
    extracted_lookup: dict[str, FieldExtraction] = {}
    for field in result.get("extracted_fields", []):
        if isinstance(field, FieldExtraction):
            extracted_lookup[field.field_name] = field
        elif isinstance(field, dict):
            extracted_lookup[field["field_name"]] = FieldExtraction(**field)

    field_results: list[MatchResult] = []
    for field_enum, comparator in FIELD_COMPARATORS.items():
        gt_value = ground_truth.get(field_enum.value)

        extraction = extracted_lookup.get(field_enum.value)
        extracted_value = extraction.value if extraction else None

        match_result = comparator(extracted_value, gt_value, field_enum.value)

        if extraction and match_result.error is None:
            match_result.confidence = extraction.confidence

        field_results.append(match_result)

    return EvalResult(
        incident_id=incident_id,
        dataset_type=dataset_type,
        pipeline_outcome=pipeline_outcome,
        field_results=field_results,
        escalation_reason=str(result.get("escalation_reason"))
        if result.get("escalation_reason")
        else None,
    )


def aggregate_metrics(eval_results: list[EvalResult]) -> list[FieldMetrics]:
    """Aggregate per-field metrics across all evaluated incidents.

    Groups MatchResults by field, excludes NO_GROUND_TRUTH from accuracy
    denominators, and computes coverage, accuracy, and confidence breakdown.

    Args:
        eval_results: List of EvalResult objects from evaluate_single.

    Returns:
        List of FieldMetrics, one per evaluable field.
    """
    results_by_field: dict[str, list[MatchResult]] = defaultdict(list)
    for er in eval_results:
        for mr in er.field_results:
            results_by_field[mr.field_name].append(mr)

    metrics: list[FieldMetrics] = []
    for field_enum in FIELD_COMPARATORS:
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


def evaluate_holdout(
    dataset_type: DatasetType,
    limit: int = 20,
    min_fields: int = 2,
) -> HoldoutReport:
    """Run holdout evaluation on a dataset.

    Connects to the database, selects incidents with the most populated
    eval fields, runs the pipeline on each, and aggregates results.

    Args:
        dataset_type: Which dataset to evaluate.
        limit: Maximum number of incidents to evaluate.
        min_fields: Minimum non-NULL eval fields per incident.

    Returns:
        Complete HoldoutReport with per-field metrics and per-incident
        results.
    """
    from src.database.connection import get_connection

    conn = get_connection()
    try:
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
        field_metrics=aggregate_metrics(eval_results),
        per_incident=eval_results,
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
