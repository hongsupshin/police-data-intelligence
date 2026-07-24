#!/usr/bin/env python3
"""CLI for the discrepancy audit.

Examples:
    # Stratified pilot (live pipeline runs — costs money)
    python scripts/run_audit.py civilians_shot --limit 100

    # Resume a killed/failed run
    python scripts/run_audit.py civilians_shot --resume civilians_shot_20260724_120000

    # Smoke test on explicit incidents
    python scripts/run_audit.py civilians_shot --incident-ids 1031 1040

    # Offline: replay saved enrichment outputs (zero LLM cost)
    python scripts/run_audit.py civilians_shot --from-saved output/enrichment

    # Inspect the sample + cost estimate without running anything
    python scripts/run_audit.py civilians_shot --dry-run --limit 100

    # False-flag suite: audit presumed-correct incidents, report specificity
    # (add --from-saved output/enrichment for the offline tier, zero LLM cost)
    python scripts/run_audit.py civilians_shot --false-flag-suite --limit 30
"""

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.state import DatasetType  # noqa: E402
from src.audit.offline import audit_saved_outputs  # noqa: E402
from src.audit.preflight import verify_db_clean  # noqa: E402
from src.audit.report import (  # noqa: E402
    IncidentAuditResult,
    build_report,
    print_report,
    save_report,
    write_verification_worksheet,
)
from src.audit.runner import run_audit  # noqa: E402
from src.audit.sampling import select_audit_incidents  # noqa: E402
from src.audit.specificity import (  # noqa: E402
    HOLDOUT_REPORTS_DIR,
    build_presumed_correct_set,
    print_specificity,
    specificity_from_results,
)
from src.eval.comparators import PipelineOutcome  # noqa: E402

# Approximate per-incident pipeline cost, from holdout runs (~$0.20/record
# on Sonnet; see EVALUATION.md). Used only for the --dry-run estimate.
EST_COST_PER_INCIDENT_USD = 0.20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the discrepancy audit")
    parser.add_argument("dataset_type", choices=[d.value for d in DatasetType])
    parser.add_argument(
        "--limit", type=int, default=100, help="Incidents to sample (default 100)"
    )
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume an existing run")
    parser.add_argument(
        "--incident-ids",
        type=int,
        nargs="+",
        help="Audit these incidents instead of sampling",
    )
    parser.add_argument(
        "--from-saved",
        metavar="DIR",
        type=Path,
        help="Offline mode: replay saved enrichment outputs (zero LLM cost)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sample and cost estimate; run nothing",
    )
    parser.add_argument(
        "--false-flag-suite",
        action="store_true",
        help="Audit presumed-correct incidents and report specificity",
    )
    return parser.parse_args()


def _presumed_correct(dataset_type: DatasetType) -> list[int]:
    """Build the presumed-correct id set from saved holdout reports."""
    report_paths = sorted(
        HOLDOUT_REPORTS_DIR.glob(f"holdout_{dataset_type.value}_*.json")
    )
    ids = build_presumed_correct_set(report_paths, dataset_type)
    print(
        f"Presumed-correct set: {len(ids)} incidents "
        f"from {len(report_paths)} holdout reports"
    )
    return ids


def _dry_run(dataset_type: DatasetType, limit: int) -> None:
    """Print the stratified selection and cost estimate without running."""
    from src.database.connection import get_connection

    conn = get_connection()
    try:
        verify_db_clean(conn)
        samples = select_audit_incidents(conn, dataset_type, limit)
    finally:
        conn.close()

    by_year = Counter(s.year for s in samples)
    by_race = Counter(s.race or "UNKNOWN" for s in samples)
    print(f"Would audit {len(samples)} {dataset_type.value} incidents")
    print("By year: " + ", ".join(f"{y}={n}" for y, n in sorted(by_year.items())))
    print("By race: " + ", ".join(f"{r}={n}" for r, n in sorted(by_race.items())))
    print(
        f"Estimated cost: ~${len(samples) * EST_COST_PER_INCIDENT_USD:.2f} "
        f"(~${EST_COST_PER_INCIDENT_USD:.2f}/incident)"
    )
    print("Incident ids: " + ", ".join(str(s.incident_id) for s in samples))


def _offline(
    dataset_type: DatasetType,
    directory: Path,
    false_flag_suite: bool = False,
) -> None:
    """Replay saved enrichment outputs through the flag layer (no LLM)."""
    from src.database.connection import get_connection

    presumed = set(_presumed_correct(dataset_type)) if false_flag_suite else None

    conn = get_connection()
    try:
        verify_db_clean(conn)
        audits = audit_saved_outputs(directory, dataset_type, conn)
    finally:
        conn.close()

    results = [
        IncidentAuditResult(
            incident_id=a.incident_id,
            dataset_type=a.dataset_type,
            auditable=True,
            pipeline_outcome=PipelineOutcome.COMPLETE,
            flags=[f for f in a.flags if not f.suppressed],
            suppressed_flags=[f for f in a.flags if f.suppressed],
            match_results=a.match_results,
        )
        for a in audits
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = build_report(
        run_id=f"{dataset_type.value}_offline_{timestamp}",
        dataset_type=dataset_type,
        results=results,
    )
    print_report(report)
    if presumed is not None:
        print_specificity(specificity_from_results(results, presumed))
    print(f"Report: {save_report(report)}")
    print(f"Worksheet: {write_verification_worksheet(report)}")


def _live(args: argparse.Namespace, dataset_type: DatasetType) -> None:
    """Run (or resume) a live checkpointed audit."""
    from src.database.connection import get_connection

    conn = get_connection()
    try:
        verify_db_clean(conn)
    finally:
        conn.close()

    incident_ids = args.incident_ids
    presumed: set[int] | None = None
    if args.false_flag_suite:
        presumed_ids = _presumed_correct(dataset_type)
        incident_ids = presumed_ids[: args.limit]
        presumed = set(presumed_ids)

    report = run_audit(
        dataset_type=dataset_type,
        limit=args.limit,
        run_id=args.resume,
        incident_ids=incident_ids,
    )
    print_report(report)
    if presumed is not None:
        print_specificity(
            specificity_from_results(report.per_incident, presumed)
        )
    print(f"Report: {save_report(report)}")
    print(f"Worksheet: {write_verification_worksheet(report)}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    dataset_type = DatasetType(args.dataset_type)

    if args.dry_run:
        _dry_run(dataset_type, args.limit)
    elif args.from_saved is not None:
        _offline(dataset_type, args.from_saved, args.false_flag_suite)
    else:
        _live(args, dataset_type)


if __name__ == "__main__":
    main()
