"""Checkpointed batch runner for live discrepancy audits.

Runs the full enrichment pipeline per incident (the only LLM spend in the
audit) and compares COMPLETE outcomes against the official record.
Escalated incidents — including relevance-judge vetoes — are recorded as
not auditable and never produce flags.

Every incident result is written to the run directory immediately, so a
killed or failed run is always resumable: ``run_audit(..., run_id=...)``
reloads the manifest and skips incidents whose result file exists. One
Tavily/LLM failure must not lose a paid partial run.
"""

import json
import logging
import time as time_mod
from datetime import datetime
from pathlib import Path

from src.agents.state import DatasetType, PipelineStage
from src.audit.compare import compare_record
from src.audit.reference import ReferenceProvider, TjiDbReferenceProvider
from src.audit.report import (
    AUDIT_OUTPUT_DIR,
    AuditReport,
    IncidentAuditResult,
    build_report,
)
from src.audit.sampling import AuditSample, select_audit_incidents
from src.config import Settings
from src.eval.comparators import PipelineOutcome

logger = logging.getLogger(__name__)

RUNS_DIR = AUDIT_OUTPUT_DIR / "runs"
MANIFEST_FILENAME = "manifest.json"


def audit_single(
    incident_id: int,
    dataset_type: DatasetType,
    reference: dict[str, object],
    settings: Settings | None = None,
    reference_source: str = "tji_db",
) -> IncidentAuditResult:
    """Run the pipeline on one incident and compare against the record.

    Args:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset the incident belongs to.
        reference: Official record values keyed by MediaFeatureField value.
        settings: Optional pipeline settings override forwarded to run().
        reference_source: Which official record is being compared against.

    Returns:
        IncidentAuditResult; auditable (with flags) only when the
        pipeline completed.
    """
    from src.run import run

    start = time_mod.monotonic()
    result = run(str(incident_id), dataset_type.value, settings=settings)
    elapsed = time_mod.monotonic() - start

    completed = result.get("current_stage") == PipelineStage.COMPLETE.value
    escalation_reason = (
        str(result.get("escalation_reason"))
        if result.get("escalation_reason")
        else None
    )

    if not completed:
        return IncidentAuditResult(
            incident_id=incident_id,
            dataset_type=dataset_type,
            auditable=False,
            pipeline_outcome=PipelineOutcome.ESCALATE,
            escalation_reason=escalation_reason,
            elapsed_seconds=round(elapsed, 2),
        )

    audit = compare_record(
        incident_id=incident_id,
        dataset_type=dataset_type,
        extracted_fields=result.get("extracted_fields", []),
        reference=reference,
        reference_source=reference_source,
    )
    return IncidentAuditResult(
        incident_id=incident_id,
        dataset_type=dataset_type,
        auditable=True,
        pipeline_outcome=PipelineOutcome.COMPLETE,
        flags=[f for f in audit.flags if not f.suppressed],
        suppressed_flags=[f for f in audit.flags if f.suppressed],
        match_results=audit.match_results,
        elapsed_seconds=round(elapsed, 2),
    )


def _load_manifest(run_dir: Path) -> dict:
    """Load a run manifest.

    Args:
        run_dir: The run's checkpoint directory.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: If the run directory has no manifest.
    """
    return json.loads((run_dir / MANIFEST_FILENAME).read_text())


def _write_manifest(
    run_dir: Path,
    run_id: str,
    dataset_type: DatasetType,
    samples: list[AuditSample],
    reference_source: str,
) -> None:
    """Write the run manifest before the batch loop starts.

    Args:
        run_dir: The run's checkpoint directory.
        run_id: Run identifier.
        dataset_type: Which dataset is audited.
        samples: The sampled incidents (fixed for the run's lifetime).
        reference_source: Which official record is compared against.
    """
    manifest = {
        "run_id": run_id,
        "dataset_type": dataset_type.value,
        "reference_source": reference_source,
        "created_at": datetime.now().isoformat(),
        "samples": [json.loads(s.model_dump_json()) for s in samples],
    }
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


def run_audit(
    dataset_type: DatasetType,
    limit: int = 100,
    run_id: str | None = None,
    incident_ids: list[int] | None = None,
    settings: Settings | None = None,
    provider: ReferenceProvider | None = None,
    runs_dir: Path = RUNS_DIR,
) -> AuditReport:
    """Run a checkpointed batch audit.

    A new run samples complete single-victim incidents (or uses the
    explicit ``incident_ids``), writes a manifest, then processes each
    incident with per-incident error isolation, checkpointing every
    result immediately. Passing an existing ``run_id`` resumes: sampled
    ids come from the manifest and finished incidents are skipped.

    Args:
        dataset_type: Which dataset to audit.
        limit: Number of incidents to sample (ignored on resume or when
            incident_ids is given).
        run_id: Existing run to resume (None starts a new run).
        incident_ids: Explicit incident ids (skips sampling; for smokes).
        settings: Optional pipeline settings override.
        provider: Reference provider (defaults to the TJI DB provider).
        runs_dir: Root directory for run checkpoints.

    Returns:
        Complete AuditReport aggregated from the run directory.
    """
    from src.database.connection import get_connection

    provider = provider or TjiDbReferenceProvider()

    if run_id is not None:
        run_dir = runs_dir / run_id
        manifest = _load_manifest(run_dir)
        dataset_type = DatasetType(manifest["dataset_type"])
        samples = [AuditSample(**s) for s in manifest["samples"]]
        logger.info("Resuming run %s (%d incidents)", run_id, len(samples))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{dataset_type.value}_{timestamp}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        conn = get_connection()
        try:
            if incident_ids is not None:
                samples = [
                    AuditSample(incident_id=i, year=0) for i in incident_ids
                ]
            else:
                samples = select_audit_incidents(conn, dataset_type, limit)
        finally:
            conn.close()
        _write_manifest(
            run_dir, run_id, dataset_type, samples, provider.source_name
        )
        logger.info("Starting run %s (%d incidents)", run_id, len(samples))

    conn = get_connection()
    try:
        for i, sample in enumerate(samples):
            checkpoint = run_dir / f"incident_{sample.incident_id}.json"
            if checkpoint.exists():
                logger.info(
                    "Skipping %d/%d: incident_id=%d (checkpointed)",
                    i + 1,
                    len(samples),
                    sample.incident_id,
                )
                continue
            logger.info(
                "Auditing %d/%d: incident_id=%d",
                i + 1,
                len(samples),
                sample.incident_id,
            )
            try:
                reference = provider.fetch(conn, sample.incident_id, dataset_type)
                result = audit_single(
                    sample.incident_id,
                    dataset_type,
                    reference,
                    settings=settings,
                    reference_source=provider.source_name,
                )
            except Exception as exc:  # noqa: BLE001 - isolate paid partial runs
                logger.error(
                    "Incident %d failed: %s", sample.incident_id, exc
                )
                result = IncidentAuditResult(
                    incident_id=sample.incident_id,
                    dataset_type=dataset_type,
                    auditable=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            checkpoint.write_text(result.model_dump_json(indent=2))
    finally:
        conn.close()

    results = _collect_results(run_dir)
    return build_report(
        run_id=run_id,
        dataset_type=dataset_type,
        results=results,
        samples=samples,
        reference_source=provider.source_name,
    )


def _collect_results(run_dir: Path) -> list[IncidentAuditResult]:
    """Load all checkpointed incident results for a run.

    Args:
        run_dir: The run's checkpoint directory.

    Returns:
        Results sorted by incident id.
    """
    results = [
        IncidentAuditResult.model_validate_json(path.read_text())
        for path in sorted(run_dir.glob("incident_*.json"))
    ]
    return sorted(results, key=lambda r: r.incident_id)
