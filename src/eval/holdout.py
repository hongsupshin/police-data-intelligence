"""Holdout evaluation for the enrichment pipeline.

Runs the pipeline on incidents with known ground truth values and compares
extracted results against database values. Measures extraction accuracy
across 6 evaluable fields using field-specific comparison strategies.

Modules:
    Data models (EvalResult, FieldMetrics, HoldoutReport), DB queries,
    orchestration, and report formatting. The comparison layer (EvalError,
    PipelineOutcome, MatchResult, compare_* functions, comparator mappings,
    fetch_ground_truth) lives in src.eval.comparators and is re-exported
    here for backward compatibility.
"""

import json
import logging
import time as time_mod
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from psycopg2.extensions import connection
from pydantic import BaseModel, Field

from src.agents.state import (
    ConfidenceLevel,
    DatasetType,
    EscalationReason,
    FieldExtraction,
    PipelineStage,
)
from src.config import Settings

# The comparison layer (enums, MatchResult, compare_* functions, comparator
# mappings, fetch_ground_truth) was extracted to src.eval.comparators so the
# discrepancy audit can reuse it. Names are re-imported here so all historical
# import sites (gate.py, run_eval.py, ci.py, tests) keep resolving.
from src.eval.comparators import (
    EVAL_FIELDS,  # noqa: F401
    FIELD_COMPARATORS,
    EvalError,
    MatchResult,
    PipelineOutcome,
    _normalize_race,  # noqa: F401
    comparators_for_dataset,
    compare_age,  # noqa: F401
    compare_location,  # noqa: F401
    compare_outcome,  # noqa: F401
    compare_race,  # noqa: F401
    compare_time,  # noqa: F401
    compare_weapon,  # noqa: F401
    fetch_ground_truth,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


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
# DB Queries
# ---------------------------------------------------------------------------


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

# Frozen, never-tuned-against TEST split (per dataset — the two datasets share an
# id namespace, so ids must be keyed by DatasetType). Derived offline from the
# saved PR #52 baseline reports' stratified `samples` via a year-stratified pick
# (limit=40, min 4/stratum, ordered by field_count desc then id asc), and verified
# disjoint from DEV_SET_IDS. A fix is promoted only if it passes the gate on TEST,
# not just DEV; exploratory sampling excludes DEV ∪ TEST so tuning can never touch it.
TEST_SET_IDS: dict[DatasetType, set[int]] = {
    DatasetType.OFFICERS_SHOT: {
        1, 2, 4, 5, 7, 8, 10, 12, 43, 44, 45, 46, 47, 49, 51, 52, 53, 54, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 95, 96, 97, 98, 138, 139, 140, 141, 175,
        176, 177, 178,
    },
    DatasetType.CIVILIANS_SHOT: {
        1, 46, 139, 144, 146, 147, 150, 151, 152, 154, 156, 157, 244, 245, 247,
        248, 251, 252, 253, 254, 255, 407, 408, 409, 412, 583, 584, 585, 586, 770,
        771, 772, 773, 951, 952, 953, 954, 1720, 3394, 5068,
    },
}


def _excluded_ids(
    dataset_type: DatasetType,
    exclude_dev_set: bool = True,
    exclude_test_set: bool = True,
) -> set[int]:
    """Incident ids to hold out of exploratory holdout sampling.

    Args:
        dataset_type: Which dataset's TEST set applies.
        exclude_dev_set: Exclude the 10-id DEV smoke set.
        exclude_test_set: Exclude the frozen per-dataset TEST split.

    Returns:
        The union of the requested hold-out sets (DEV is dataset-agnostic;
        TEST is per-dataset).
    """
    excluded: set[int] = set()
    if exclude_dev_set:
        excluded |= DEV_SET_IDS
    if exclude_test_set:
        excluded |= TEST_SET_IDS.get(dataset_type, set())
    return excluded


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
        EscalationReason.IRRELEVANT_SOURCES.value,
    ):
        return "synthesize"
    if reason == EscalationReason.EXTRACTION_ERROR.value:
        return "load"
    return "unknown"


# Every year stratum gets at least this many samples so sparse years are
# not drowned out by proportional allocation.
MIN_PER_STRATUM = 4


def allocate_by_stratum(
    rows_by_stratum: dict[int, list[tuple]],
    limit: int,
    min_per_stratum: int = MIN_PER_STRATUM,
) -> list[tuple]:
    """Proportionally allocate rows across strata with a per-stratum minimum.

    Shared by holdout and audit sampling. Strata are visited in sorted key
    order; each gets max(min_per_stratum, its proportional share of limit),
    capped at what it has, until limit is reached.

    Args:
        rows_by_stratum: Rows grouped by stratum key (e.g. year).
        limit: Total number of rows to select.
        min_per_stratum: Minimum rows per stratum when available.

    Returns:
        Selected rows, at most ``limit``.
    """
    total_available = sum(len(v) for v in rows_by_stratum.values())
    selected: list[tuple] = []
    for stratum in sorted(rows_by_stratum.keys()):
        stratum_rows = rows_by_stratum[stratum]
        proportion = len(stratum_rows) / total_available if total_available > 0 else 0
        n_for_stratum = max(min_per_stratum, round(proportion * limit))
        n_for_stratum = min(n_for_stratum, len(stratum_rows))
        for row in stratum_rows[:n_for_stratum]:
            selected.append(row)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return selected[:limit]


def select_holdout_stratified(
    conn: connection,
    dataset_type: DatasetType,
    min_fields: int = 2,
    limit: int = 40,
    exclude_dev_set: bool = True,
    exclude_test_set: bool = True,
) -> list[HoldoutSample]:
    """Select holdout incidents with proportional year stratification.

    Selects incidents proportionally across years (2015-2019), ensuring
    a minimum of 4 per stratum. Excludes the DEV and frozen TEST sets by
    default so exploratory sampling never touches held-out data.

    Args:
        conn: Active PostgreSQL connection.
        dataset_type: Which dataset to query.
        min_fields: Minimum number of non-NULL eval fields required.
        limit: Total number of incidents to return.
        exclude_dev_set: Whether to exclude dev-set IDs.
        exclude_test_set: Whether to exclude the frozen TEST-split IDs.

    Returns:
        List of HoldoutSample with year/race metadata.
    """
    cursor = conn.cursor()

    exclusion_clause = ""
    excluded = _excluded_ids(dataset_type, exclude_dev_set, exclude_test_set)
    if excluded:
        ids_str = ", ".join(str(i) for i in sorted(excluded))
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

    return [
        HoldoutSample(
            incident_id=row[0],
            year=row[1],
            race=row[2],
            n_eval_fields=row[3],
        )
        for row in allocate_by_stratum(by_year, limit)
    ]


def evaluate_single(
    incident_id: int,
    dataset_type: DatasetType,
    ground_truth: dict[str, object],
    settings: Settings | None = None,
) -> EvalResult:
    """Evaluate a single incident by running the pipeline and comparing results.

    Calls the real pipeline via src.run.run(), then compares each
    evaluable field against ground truth using field-specific comparators.

    Args:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset this incident belongs to.
        ground_truth: Dict mapping field names to ground truth values.
        settings: Optional pipeline settings override forwarded to run()
            (lets the eval gate run a config variant).

    Returns:
        EvalResult with comparison results for all evaluable fields.
    """
    from src.run import run

    start = time_mod.monotonic()
    result = run(str(incident_id), dataset_type.value, settings=settings)
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
    settings: Settings | None = None,
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
        settings: Optional pipeline settings override forwarded to each
            run (lets the eval gate evaluate a config variant in-process).

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
            result = evaluate_single(incident_id, dataset_type, gt, settings=settings)
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
    exclude_test_set: bool = True,
    settings: Settings | None = None,
) -> HoldoutReport:
    """Run stratified holdout evaluation on a dataset.

    Uses year-stratified sampling, tracks timing per incident,
    and computes fairness metrics by race group.

    Args:
        dataset_type: Which dataset to evaluate.
        limit: Total number of incidents to evaluate.
        min_fields: Minimum non-NULL eval fields per incident.
        exclude_dev_set: Whether to exclude dev-set IDs.
        exclude_test_set: Whether to exclude the frozen TEST-split IDs.
        settings: Optional pipeline settings override forwarded to each
            run (lets the eval gate evaluate a config variant in-process).

    Returns:
        Complete HoldoutReport with stratified samples, timing,
        and fairness metrics.
    """
    from src.database.connection import get_connection

    conn = get_connection()
    try:
        samples = select_holdout_stratified(
            conn, dataset_type, min_fields, limit, exclude_dev_set, exclude_test_set
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
            result = evaluate_single(
                sample.incident_id, dataset_type, gt, settings=settings
            )
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


def evaluate_test_split(
    dataset_type: DatasetType,
    settings: Settings | None = None,
) -> HoldoutReport:
    """Evaluate the frozen TEST split for a dataset.

    Runs the pipeline on exactly ``TEST_SET_IDS[dataset_type]`` — the
    never-tuned-against benchmark a fix must pass before promotion. Pairs with
    a baseline-vs-variant call through the gate.

    Note:
        Because this uses the ``incident_ids`` path, the returned report's
        ``fairness_metrics`` is empty (no stratified ``samples`` are built);
        the gate recomputes fairness from ``per_incident``.

    Args:
        dataset_type: Which dataset's TEST split to evaluate.
        settings: Optional pipeline settings override forwarded to each run.

    Returns:
        HoldoutReport over the TEST incidents.
    """
    return evaluate_holdout(
        dataset_type,
        incident_ids=sorted(TEST_SET_IDS[dataset_type]),
        settings=settings,
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
