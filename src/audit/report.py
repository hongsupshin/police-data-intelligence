"""Audit report models, persistence, and the human-verification worksheet.

Follows the holdout report conventions: Pydantic report saved as indented
JSON under ``output/audit/``, plus a console table. The worksheet is a CSV
(one row per reportable flag) that a human verifier fills in; verified
flag precision — the headline metric — is recomputed from the filled-in
worksheet with a Wilson interval.
"""

import csv
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.agents.state import DatasetType
from src.audit.flags import DiscrepancyFlag, FlagSeverity, VerificationStatus
from src.audit.reference import TJI_DB_SOURCE
from src.audit.sampling import AuditSample
from src.eval.ci import wilson_ci
from src.eval.comparators import MatchResult, PipelineOutcome

logger = logging.getLogger(__name__)

AUDIT_OUTPUT_DIR = Path("output/audit")

WORKSHEET_COLUMNS = [
    "flag_id",
    "incident_id",
    "field",
    "severity",
    "db_value",
    "news_value",
    "extraction_confidence",
    "source_urls",
    "source_quotes",
    "verification_status",
    "verifier_notes",
]

_SEVERITY_ORDER = {
    FlagSeverity.HIGH: 0,
    FlagSeverity.MEDIUM: 1,
    FlagSeverity.LOW: 2,
}


class IncidentAuditResult(BaseModel):
    """Audit outcome for one incident.

    Attributes:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset the incident belongs to.
        auditable: True when the pipeline completed and comparisons ran.
        pipeline_outcome: Pipeline terminal outcome (None if the run
            errored before terminating).
        escalation_reason: Why the pipeline escalated (None if completed).
        flags: Reportable contradiction flags.
        suppressed_flags: Contradictions below the confidence bar (kept
            for taxonomy analysis, excluded from headline counts).
        match_results: All field comparisons, agreements included.
        elapsed_seconds: Wall-clock seconds for the pipeline run (0 for
            offline replays).
        error: Exception text if the incident's run failed.
    """

    incident_id: int
    dataset_type: DatasetType
    auditable: bool
    pipeline_outcome: PipelineOutcome | None = None
    escalation_reason: str | None = None
    flags: list[DiscrepancyFlag] = Field(default_factory=list)
    suppressed_flags: list[DiscrepancyFlag] = Field(default_factory=list)
    match_results: list[MatchResult] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None


class FieldFlagSummary(BaseModel):
    """Flag statistics for one field across the run.

    Attributes:
        field: MediaFeatureField value.
        n_audited: Comparisons where both values were present.
        n_flags: Reportable flags emitted.
        flag_rate: n_flags / n_audited (0 when nothing audited).
        flag_rate_ci: 95% Wilson interval for the flag rate.
        by_severity: Reportable flag counts per severity.
    """

    field: str
    n_audited: int
    n_flags: int
    flag_rate: float
    flag_rate_ci: tuple[float, float]
    by_severity: dict[str, int] = Field(default_factory=dict)


class AuditReport(BaseModel):
    """Complete discrepancy-audit report for one run.

    Attributes:
        run_id: Run identifier ``{dataset}_{timestamp}`` (checkpoint dir
            name for live runs).
        dataset_type: Which dataset was audited.
        reference_source: Which official record was compared against.
        n_incidents: Incidents attempted.
        n_auditable: Incidents where the pipeline completed.
        n_escalated: Incidents the pipeline escalated (never flagged).
        n_errors: Incidents whose runs failed.
        n_flags: Reportable flags across the run.
        n_suppressed: Suppressed (low-confidence) contradictions.
        field_summaries: Per-field flag statistics.
        per_incident: Detailed per-incident results.
        samples: Sample metadata for the audited incidents.
        mean_elapsed_seconds: Mean pipeline seconds per incident.
        total_elapsed_seconds: Total pipeline seconds.
    """

    run_id: str
    dataset_type: DatasetType
    reference_source: str = TJI_DB_SOURCE
    n_incidents: int
    n_auditable: int
    n_escalated: int
    n_errors: int
    n_flags: int
    n_suppressed: int
    field_summaries: list[FieldFlagSummary] = Field(default_factory=list)
    per_incident: list[IncidentAuditResult] = Field(default_factory=list)
    samples: list[AuditSample] = Field(default_factory=list)
    mean_elapsed_seconds: float = 0.0
    total_elapsed_seconds: float = 0.0


def summarize_fields(results: list[IncidentAuditResult]) -> list[FieldFlagSummary]:
    """Aggregate per-field flag statistics across incident results.

    A field counts as audited when its comparison ran with both values
    present (no error). Only reportable flags enter the counts.

    Args:
        results: Per-incident audit results.

    Returns:
        One FieldFlagSummary per field seen, ordered by field name.
    """
    audited: dict[str, int] = {}
    flags_by_field: dict[str, list[DiscrepancyFlag]] = {}
    for result in results:
        for match in result.match_results:
            if match.error is None:
                audited[match.field_name] = audited.get(match.field_name, 0) + 1
        for flag in result.flags:
            flags_by_field.setdefault(flag.field, []).append(flag)

    summaries = []
    for field in sorted(audited):
        field_flags = flags_by_field.get(field, [])
        n_audited = audited[field]
        n_flags = len(field_flags)
        by_severity: dict[str, int] = {}
        for flag in field_flags:
            by_severity[flag.severity.value] = (
                by_severity.get(flag.severity.value, 0) + 1
            )
        summaries.append(
            FieldFlagSummary(
                field=field,
                n_audited=n_audited,
                n_flags=n_flags,
                flag_rate=n_flags / n_audited if n_audited else 0.0,
                flag_rate_ci=wilson_ci(n_flags, n_audited),
                by_severity=by_severity,
            )
        )
    return summaries


def build_report(
    run_id: str,
    dataset_type: DatasetType,
    results: list[IncidentAuditResult],
    samples: list[AuditSample] | None = None,
    reference_source: str = TJI_DB_SOURCE,
) -> AuditReport:
    """Assemble the audit report from per-incident results.

    Args:
        run_id: Run identifier.
        dataset_type: Which dataset was audited.
        results: Per-incident audit results.
        samples: Sample metadata (None for ad-hoc/offline runs).
        reference_source: Which official record was compared against.

    Returns:
        Complete AuditReport.
    """
    n_incidents = len(results)
    total_elapsed = sum(r.elapsed_seconds for r in results)
    return AuditReport(
        run_id=run_id,
        dataset_type=dataset_type,
        reference_source=reference_source,
        n_incidents=n_incidents,
        n_auditable=sum(1 for r in results if r.auditable),
        n_escalated=sum(
            1 for r in results if r.pipeline_outcome == PipelineOutcome.ESCALATE
        ),
        n_errors=sum(1 for r in results if r.error is not None),
        n_flags=sum(len(r.flags) for r in results),
        n_suppressed=sum(len(r.suppressed_flags) for r in results),
        field_summaries=summarize_fields(results),
        per_incident=results,
        samples=samples or [],
        mean_elapsed_seconds=(
            round(total_elapsed / n_incidents, 2) if n_incidents else 0.0
        ),
        total_elapsed_seconds=round(total_elapsed, 2),
    )


def save_report(report: AuditReport) -> str:
    """Save the audit report as JSON under output/audit/.

    Args:
        report: Complete audit report.

    Returns:
        Path to the saved JSON file.
    """
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = AUDIT_OUTPUT_DIR / f"audit_{report.run_id}.json"
    data = json.loads(report.model_dump_json())
    filepath.write_text(json.dumps(data, indent=2))
    return str(filepath)


def print_report(report: AuditReport) -> None:
    """Print a human-readable summary table of the audit report.

    Args:
        report: Complete audit report.
    """
    print(f"\nDiscrepancy audit: {report.run_id} (vs {report.reference_source})")
    print(
        f"Incidents: {report.n_incidents} attempted, "
        f"{report.n_auditable} auditable, {report.n_escalated} escalated, "
        f"{report.n_errors} errors"
    )
    print(
        f"Flags: {report.n_flags} reportable, {report.n_suppressed} suppressed"
    )
    if report.field_summaries:
        print(f"\n{'Field':<18} {'Audited':>8} {'Flags':>6} {'Rate':>6}  95% CI")
        for fs in report.field_summaries:
            low, high = fs.flag_rate_ci
            print(
                f"{fs.field:<18} {fs.n_audited:>8} {fs.n_flags:>6} "
                f"{fs.flag_rate:>6.1%}  [{low:.1%}, {high:.1%}]"
            )
    print()


def write_verification_worksheet(report: AuditReport) -> str:
    """Write the human-verification worksheet CSV.

    One row per reportable flag, ordered by severity (high first), then
    incident, then field. The verifier fills in ``verification_status``
    (db_error / news_error / extraction_error / unresolved) and notes.

    Args:
        report: Complete audit report.

    Returns:
        Path to the worksheet CSV.
    """
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = AUDIT_OUTPUT_DIR / f"audit_{report.run_id}_worksheet.csv"

    flags = [flag for result in report.per_incident for flag in result.flags]
    flags.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.incident_id, f.field))

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(WORKSHEET_COLUMNS)
        for flag in flags:
            writer.writerow(
                [
                    flag.flag_id,
                    flag.incident_id,
                    flag.field,
                    flag.severity.value,
                    flag.db_value,
                    flag.news_value,
                    flag.extraction_confidence.value,
                    "; ".join(flag.sources),
                    " | ".join(flag.source_quotes),
                    flag.verification_status.value,
                    "",
                ]
            )
    return str(filepath)


def summarize_verified(worksheet_path: str | Path) -> dict[str, object]:
    """Compute verified flag precision from a filled-in worksheet.

    Precision is the fraction of verified flags confirmed as errors in
    the official record (``db_error``) among all flags a human resolved.
    Pending rows are excluded.

    Args:
        worksheet_path: Path to the (partially) filled-in worksheet CSV.

    Returns:
        Dict with counts per verification status, ``n_verified``,
        ``precision``, and ``precision_ci`` (95% Wilson). Precision is
        None when nothing has been verified yet.
    """
    counts: dict[str, int] = {status.value: 0 for status in VerificationStatus}
    with open(worksheet_path, newline="") as f:
        for row in csv.DictReader(f):
            status = row["verification_status"].strip().lower()
            if status in counts:
                counts[status] += 1
            else:
                logger.warning("Unknown verification_status %r ignored", status)

    n_verified = sum(
        n
        for status, n in counts.items()
        if status != VerificationStatus.PENDING.value
    )
    n_db_errors = counts[VerificationStatus.DB_ERROR.value]
    return {
        "counts": counts,
        "n_verified": n_verified,
        "precision": n_db_errors / n_verified if n_verified else None,
        "precision_ci": wilson_ci(n_db_errors, n_verified) if n_verified else None,
    }
