"""Comparison-only audit over saved enrichment outputs. Zero LLM cost.

Replays the flag layer against artifacts already written by past pipeline
runs (``output/enrichment/{dataset}_{id}_complete.json``). Only COMPLETE
artifacts are used — escalated runs (including relevance-judge vetoes)
never produce flags. This is the offline gate: the entire comparison layer
is exercised end-to-end against real extractions before any new LLM spend.

Saved artifacts span pipeline versions; older files lack newer keys
(``civilian_race_taxonomy``, ``conflict_annotation``, ``judge_failures``),
so parsing relies only on ``incident_id``, ``dataset_type``, and
``extracted_fields``.

Usage (needs PostgreSQL, no API keys):
    python -m src.audit.offline civilians_shot
    python -m src.audit.offline officers_shot --directory output/enrichment
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from psycopg2.extensions import connection

from src.agents.state import DatasetType
from src.audit.compare import RecordAudit, compare_record
from src.audit.reference import ReferenceProvider, TjiDbReferenceProvider

logger = logging.getLogger(__name__)

DEFAULT_ENRICHMENT_DIR = Path("output/enrichment")


def load_saved_enrichment(path: Path) -> tuple[int, DatasetType, list[dict]]:
    """Parse a saved complete-enrichment artifact.

    Args:
        path: Path to a ``{dataset}_{id}_complete.json`` artifact.

    Returns:
        Tuple of (incident_id, dataset_type, extracted field dicts).

    Raises:
        KeyError: If the artifact lacks a required key.
        ValueError: If dataset_type or incident_id is malformed.
    """
    with open(path) as f:
        data = json.load(f)
    incident_id = int(data["incident_id"])
    dataset_type = DatasetType(data["dataset_type"])
    extracted_fields = data["extracted_fields"]
    return incident_id, dataset_type, extracted_fields


def iter_saved_complete(directory: Path, dataset_type: DatasetType) -> list[Path]:
    """List saved COMPLETE artifacts for a dataset, sorted by path.

    Escalated artifacts (``*_escalate.json``) are excluded by pattern:
    escalated runs must never produce flags.

    Args:
        directory: Directory holding enrichment artifacts.
        dataset_type: Which dataset's artifacts to list.

    Returns:
        Sorted list of matching artifact paths.
    """
    return sorted(directory.glob(f"{dataset_type.value}_*_complete.json"))


def audit_saved_outputs(
    directory: Path,
    dataset_type: DatasetType,
    conn: connection,
    provider: ReferenceProvider | None = None,
) -> list[RecordAudit]:
    """Run the comparison-only audit over saved COMPLETE artifacts.

    Args:
        directory: Directory holding enrichment artifacts.
        dataset_type: Which dataset to audit.
        conn: Active PostgreSQL connection (for the reference fetch).
        provider: Reference provider (defaults to the TJI DB provider).

    Returns:
        One RecordAudit per parseable artifact whose incident exists in
        the database; unparseable or unknown-incident artifacts are
        logged and skipped.
    """
    provider = provider or TjiDbReferenceProvider()
    audits: list[RecordAudit] = []
    for path in iter_saved_complete(directory, dataset_type):
        try:
            incident_id, artifact_dataset, extracted_fields = load_saved_enrichment(
                path
            )
            reference = provider.fetch(conn, incident_id, artifact_dataset)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        audits.append(
            compare_record(
                incident_id=incident_id,
                dataset_type=artifact_dataset,
                extracted_fields=extracted_fields,
                reference=reference,
                reference_source=provider.source_name,
            )
        )
    return audits


def summarize_audits(audits: list[RecordAudit]) -> str:
    """Render a flag-count summary for a list of record audits.

    Args:
        audits: Record audits to summarize.

    Returns:
        Multi-line human-readable summary (counts by field and severity,
        reportable vs suppressed).
    """
    all_flags = [flag for audit in audits for flag in audit.flags]
    reportable = [f for f in all_flags if not f.suppressed]
    by_field = Counter(f.field for f in reportable)
    by_severity = Counter(f.severity.value for f in reportable)

    lines = [
        f"Incidents audited: {len(audits)}",
        f"Flags: {len(reportable)} reportable, "
        f"{len(all_flags) - len(reportable)} suppressed (low confidence)",
        "By severity: "
        + (
            ", ".join(f"{sev}={n}" for sev, n in sorted(by_severity.items()))
            or "none"
        ),
        "By field: "
        + (
            ", ".join(f"{field}={n}" for field, n in sorted(by_field.items()))
            or "none"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for the offline audit."""
    parser = argparse.ArgumentParser(
        description="Comparison-only discrepancy audit over saved outputs"
    )
    parser.add_argument(
        "dataset_type", choices=[d.value for d in DatasetType]
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_ENRICHMENT_DIR,
        help="Directory of saved enrichment artifacts",
    )
    args = parser.parse_args()

    from src.database.connection import get_connection

    logging.basicConfig(level=logging.INFO)
    conn = get_connection()
    try:
        audits = audit_saved_outputs(
            args.directory, DatasetType(args.dataset_type), conn
        )
    finally:
        conn.close()
    print(summarize_audits(audits))
    for audit in audits:
        for flag in audit.flags:
            if not flag.suppressed:
                print(
                    f"  [{flag.severity.value:6s}] {flag.flag_id}: "
                    f"db={flag.db_value!r} news={flag.news_value!r} "
                    f"({flag.extraction_confidence.value})"
                )


if __name__ == "__main__":
    main()
