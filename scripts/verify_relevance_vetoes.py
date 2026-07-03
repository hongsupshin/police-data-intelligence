"""Offline re-verification of the relevance judge's live holdout vetoes.

The canonical June-2026 holdout (report pair ``20260613_231717``) records ten
``irrelevant_sources`` escalations: seven civilians_shot and three
officers_shot incidents the relevance judge vetoed. This script reconstructs
each veto for human review by printing, side by side:

- the database anchor (date, city, names) via ``fetch_incident``,
- every retrieved article the judge saw (title, source, date, URL, text), and
- the fields the pipeline had extracted before the veto.

A reviewer reads each dossier and records a verdict (genuine wrong-article,
false veto, or ambiguous) in ``docs/relevance-veto-audit.md``. The script
itself is deterministic and read-only: it fetches nothing from the network,
writes nothing, and asserts the database matches the clean row counts the
holdout was scored against before printing anything.

Two provenance safeguards, both needed because the canonical holdout predates
no incident-ID renumbering guarantee: the idempotent-ETL fix (PR #67) rebuilt
the tables with ``TRUNCATE ... RESTART IDENTITY``, so report IDs are only
trustworthy if re-verified.

- Every dossier cross-checks the clean-database anchor against the canonical
  report's per-field ground truth, proving the fetched row is the same
  incident the holdout scored rather than assuming ID stability.
- A report ID beyond the clean table's range (a pre-fix, duplicate-era ID;
  officer incident 357 is the one vetoed case) cannot be fetched by ID at
  all. For those the anchor is recovered by matching the report's ground
  truth (city + suspect age) against the clean table, and every candidate
  row is printed for the reviewer.

Usage:
    python scripts/verify_relevance_vetoes.py [--max-article-chars N]

Examples:
    $ python scripts/verify_relevance_vetoes.py > /tmp/veto_dossiers.txt
    $ python scripts/verify_relevance_vetoes.py --max-article-chars 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psycopg2.extensions import connection  # noqa: E402

from src.agents.load_node import fetch_incident  # noqa: E402
from src.agents.state import DatasetType  # noqa: E402
from src.database.connection import get_connection  # noqa: E402

CANONICAL_REPORTS: dict[DatasetType, Path] = {
    DatasetType.CIVILIANS_SHOT: REPO_ROOT
    / "output/eval/holdout_civilians_shot_20260613_231717.json",
    DatasetType.OFFICERS_SHOT: REPO_ROOT
    / "output/eval/holdout_officers_shot_20260613_231717.json",
}
ENRICHMENT_DIR = REPO_ROOT / "output" / "enrichment"

# The clean-database row counts the canonical holdout was scored against
# (post PR #67 idempotent ETL). Anchors fetched from a database in any other
# state would not be the anchors the judge vetoed against.
EXPECTED_ROW_COUNTS: dict[str, int] = {
    "incidents_civilians_shot": 1674,
    "incidents_officers_shot": 282,
}

VETO_REASON = "irrelevant_sources"


def vetoed_incident_ids(report_path: Path) -> list[int]:
    """Extract the relevance-vetoed incident IDs from a saved eval report.

    Args:
        report_path: Path to a holdout eval report JSON with a
            ``per_incident`` list.

    Returns:
        Incident IDs whose ``escalation_reason`` is ``irrelevant_sources``,
        in report order.
    """
    report = json.loads(report_path.read_text())
    return [
        record["incident_id"]
        for record in report["per_incident"]
        if record.get("escalation_reason") == VETO_REASON
    ]


def verify_db_is_clean(conn: connection) -> None:
    """Assert the database matches the clean holdout row counts.

    Args:
        conn: Active PostgreSQL connection.

    Raises:
        RuntimeError: If any incident table's row count differs from the
            clean counts the canonical holdout was scored against.
    """
    cursor = conn.cursor()
    for table, expected in EXPECTED_ROW_COUNTS.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table};")  # noqa: S608
        actual = cursor.fetchone()[0]
        if actual != expected:
            raise RuntimeError(
                f"{table} has {actual} rows, expected {expected}. "
                "The database is not in the clean state the canonical "
                "holdout was scored against; refusing to print anchors."
            )


def load_escalation(dataset_type: DatasetType, incident_id: int) -> dict[str, Any]:
    """Load the saved escalation output for a vetoed incident.

    Args:
        dataset_type: Which dataset the incident belongs to.
        incident_id: Numeric incident ID from the database.

    Returns:
        The parsed ``{dataset}_{id}_escalate.json`` contents.

    Raises:
        FileNotFoundError: If the escalation file does not exist.
        ValueError: If the file's escalation reason is not a relevance veto.
    """
    path = ENRICHMENT_DIR / f"{dataset_type.value}_{incident_id}_escalate.json"
    record = json.loads(path.read_text())
    reason = record.get("escalation_reason")
    if reason != VETO_REASON:
        raise ValueError(
            f"{path.name} has escalation_reason={reason!r}, not {VETO_REASON!r}"
        )
    return record


def ground_truth_fields(report_path: Path, incident_id: int) -> dict[str, Any]:
    """Extract one incident's per-field ground truth from a saved eval report.

    Args:
        report_path: Path to a holdout eval report JSON.
        incident_id: Incident to look up in ``per_incident``.

    Returns:
        Mapping of field name to ground-truth value for the incident.

    Raises:
        KeyError: If the incident is not in the report.
    """
    report = json.loads(report_path.read_text())
    for record in report["per_incident"]:
        if record["incident_id"] == incident_id:
            return {
                field["field_name"]: field["ground_truth_value"]
                for field in record["field_results"]
            }
    raise KeyError(f"Incident {incident_id} not in {report_path.name}")


def recover_officer_anchor_by_gt(
    conn: connection, ground_truth: dict[str, Any]
) -> list[dict[str, Any]]:
    """Find clean-database officer incidents matching a report's ground truth.

    Used when a canonical-report incident ID predates the idempotent-ETL
    rebuild and no longer exists under that ID. Matches on incident city and
    suspect age, the most identifying ground-truth fields available.

    Args:
        conn: Active PostgreSQL connection.
        ground_truth: Field mapping from ``ground_truth_fields`` (needs
            ``location_detail`` and ``civilian_age``).

    Returns:
        One dict per candidate row: clean incident_id, date, city, county,
        officer/civilian names, officer harm, and suspect age/race.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            i.incident_id,
            i.date_incident::date,
            i.incident_city,
            i.incident_county,
            o.name_first, o.name_last,
            c.name_first, c.name_last,
            v.officer_harm,
            c.age, c.race
        FROM incidents_officers_shot i
        LEFT JOIN incident_officers_shot_victims v
            ON i.incident_id = v.incident_id
        LEFT JOIN officers o ON v.officer_id = o.officer_id
        LEFT JOIN incident_officers_shot_shooters s
            ON i.incident_id = s.incident_id AND s.civilian_sequence = 1
        LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
        WHERE UPPER(i.incident_city) = UPPER(%s) AND c.age = %s;
        """,
        (ground_truth.get("location_detail"), ground_truth.get("civilian_age")),
    )
    return [
        {
            "clean_incident_id": row[0],
            "incident_date": row[1],
            "location": row[2],
            "county": row[3],
            "officer_name": " ".join(p for p in row[4:6] if p) or None,
            "civilian_name": " ".join(p for p in row[6:8] if p) or None,
            "officer_harm": row[8],
            "suspect_age": row[9],
            "suspect_race": row[10],
        }
        for row in cursor.fetchall()
    ]


def format_gt_crosscheck(
    anchor: dict[str, Any], ground_truth: dict[str, Any]
) -> str:
    """Render the anchor-vs-report ground-truth consistency check.

    Proves the clean-database row fetched by ID is the same incident the
    canonical holdout scored (IDs were renumbered by the idempotent-ETL
    rebuild, so identity must be shown, not assumed).

    Args:
        anchor: Field mapping returned by ``fetch_incident``.
        ground_truth: Field mapping from ``ground_truth_fields``.

    Returns:
        One comparison line per overlapping field, flagged MATCH or REVIEW.
    """
    comparisons = [
        ("age", anchor.get("civilian_age"), ground_truth.get("civilian_age")),
        ("city", anchor.get("location"), ground_truth.get("location_detail")),
        ("outcome", anchor.get("severity"), ground_truth.get("outcome")),
    ]
    lines = []
    for label, anchor_value, gt_value in comparisons:
        if anchor_value is None and gt_value is None:
            continue
        same = str(anchor_value).strip().upper() == str(gt_value).strip().upper()
        flag = "MATCH " if same else "REVIEW"
        lines.append(
            f"    [{flag}] {label}: anchor={anchor_value!r} vs report GT={gt_value!r}"
        )
    return "\n".join(lines) if lines else "    (no overlapping fields)"


def format_anchor(anchor: dict[str, Any]) -> str:
    """Render the database anchor as aligned key/value lines.

    Args:
        anchor: Field mapping returned by ``fetch_incident``.

    Returns:
        One line per field, sorted by key, ``None`` printed as ``(none)``.
    """
    width = max(len(key) for key in anchor)
    return "\n".join(
        f"    {key.ljust(width)} : {value if value is not None else '(none)'}"
        for key, value in sorted(anchor.items())
    )


def format_article(index: int, article: dict[str, Any], max_chars: int) -> str:
    """Render one retrieved article for review.

    Args:
        index: 1-based position of the article in the retrieved list.
        article: Article dict with url/title/published_date/source_name/content.
        max_chars: Maximum characters of article text to include.

    Returns:
        A multi-line block with headline metadata and (truncated) text.
    """
    text = " ".join((article.get("content") or "").split())
    truncated = " [...truncated]" if len(text) > max_chars else ""
    return (
        f"  [{index}] {article.get('title')}\n"
        f"      source: {article.get('source_name')}"
        f" | published: {article.get('published_date')}\n"
        f"      url: {article.get('url')}\n"
        f"      text: {text[:max_chars]}{truncated}"
    )


def format_extracted_fields(fields: list[dict[str, Any]]) -> str:
    """Render the fields the pipeline extracted before the veto.

    Args:
        fields: ``extracted_fields`` entries from the escalation output.

    Returns:
        One line per field with value and confidence, or a placeholder when
        nothing was extracted.
    """
    if not fields:
        return "    (no fields extracted)"
    return "\n".join(
        f"    {field.get('field_name')} = {field.get('value')!r}"
        f" (confidence: {field.get('confidence')})"
        for field in fields
    )


def print_dossier(
    conn: connection,
    dataset_type: DatasetType,
    incident_id: int,
    max_chars: int,
    report_path: Path,
) -> None:
    """Print the full review dossier for one vetoed incident.

    Args:
        conn: Active PostgreSQL connection for anchor lookup.
        dataset_type: Which dataset the incident belongs to.
        incident_id: Numeric incident ID from the database.
        max_chars: Maximum characters of article text to include per article.
        report_path: Canonical eval report, for the ground-truth cross-check.
    """
    record = load_escalation(dataset_type, incident_id)
    ground_truth = ground_truth_fields(report_path, incident_id)
    articles = record.get("retrieved_articles") or []

    print("=" * 78)
    print(f"INCIDENT {incident_id} ({dataset_type.value}) — vetoed: {VETO_REASON}")
    print("=" * 78)
    try:
        anchor = fetch_incident(conn, incident_id, dataset_type)
    except KeyError:
        if dataset_type is not DatasetType.OFFICERS_SHOT:
            raise
        print("  DATABASE ANCHOR: id predates the idempotent-ETL rebuild;")
        print("  recovered from the clean table by ground-truth match")
        print("  (city + suspect age):")
        candidates = recover_officer_anchor_by_gt(conn, ground_truth)
        if not candidates:
            print("    (NO MATCHING ROW — resolve manually)")
        for candidate in candidates:
            print(format_anchor(candidate))
            print("    ---")
        print("\n  REPORT GROUND TRUTH:")
        for field_name, value in sorted(ground_truth.items()):
            print(f"    {field_name} : {value!r}")
    else:
        print("  DATABASE ANCHOR:")
        print(format_anchor(anchor))
        print("\n  GT CROSS-CHECK (clean-DB row vs canonical report):")
        print(format_gt_crosscheck(anchor, ground_truth))
    print(f"\n  EXTRACTED BEFORE VETO ({len(record.get('extracted_fields') or [])}):")
    print(format_extracted_fields(record.get("extracted_fields") or []))
    print(f"\n  RETRIEVED ARTICLES ({len(articles)}):")
    for index, article in enumerate(articles, start=1):
        print(format_article(index, article, max_chars))
    print()


def main() -> int:
    """Print review dossiers for every canonical-holdout relevance veto.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--max-article-chars",
        type=int,
        default=2500,
        help="Max characters of article text to print per article.",
    )
    args = parser.parse_args()

    conn = get_connection()
    try:
        verify_db_is_clean(conn)
        for dataset_type, report_path in CANONICAL_REPORTS.items():
            incident_ids = vetoed_incident_ids(report_path)
            print(
                f"# {dataset_type.value}: {len(incident_ids)} relevance vetoes "
                f"in {report_path.name}: {incident_ids}\n"
            )
            for incident_id in incident_ids:
                print_dossier(
                    conn,
                    dataset_type,
                    incident_id,
                    args.max_article_chars,
                    report_path,
                )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
