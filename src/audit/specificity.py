"""False-flag (specificity) suite for the discrepancy audit.

The audit has no free ground truth, so precision alone can mislead: a
flag-happy comparator would still surface *some* real errors. Specificity
is measured on incidents where the official record and news coverage
demonstrably agreed — saved holdout reports where every evaluated field
matched exactly. Any audit flag on those incidents is presumed false.

This is the audit's analog of the adversarial suite: instead of fabricated
incidents probing hallucination, presumed-correct incidents probe
over-flagging.
"""

import json
import logging
from pathlib import Path

from src.agents.state import DatasetType
from src.audit.report import IncidentAuditResult
from src.eval.ci import wilson_ci
from src.eval.holdout import _excluded_ids

logger = logging.getLogger(__name__)

HOLDOUT_REPORTS_DIR = Path("output/eval")

# An incident must have at least this many exact-match fields in a holdout
# report to be presumed correct — one or two agreements is weak evidence
# that the whole record matches coverage.
DEFAULT_MIN_EXACT_FIELDS = 3


def build_presumed_correct_set(
    report_paths: list[Path],
    dataset_type: DatasetType,
    min_exact_fields: int = DEFAULT_MIN_EXACT_FIELDS,
    exclude_ids: set[int] | None = None,
) -> list[int]:
    """Collect incidents whose holdout evaluations were all-exact.

    An incident qualifies from a report when the pipeline completed and
    every field result that was actually evaluated (no error) matched
    exactly, with at least ``min_exact_fields`` such fields. An incident
    that shows any evaluated non-exact field in *any* report is
    disqualified everywhere — presumed correctness must be unanimous.

    Args:
        report_paths: Saved holdout report JSON paths.
        dataset_type: Which dataset's incidents to collect.
        min_exact_fields: Minimum exact-match fields required to qualify.
        exclude_ids: Incident ids to exclude. Defaults to DEV ∪ TEST —
            iterating thresholds against the suite is tuning, and tuning
            must never touch the held-out eval splits.

    Returns:
        Sorted incident ids presumed correct.
    """
    excluded = _excluded_ids(dataset_type) if exclude_ids is None else exclude_ids
    qualified: set[int] = set()
    disqualified: set[int] = set()

    for path in report_paths:
        try:
            report = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable report %s: %s", path, exc)
            continue
        if report.get("dataset_type") != dataset_type.value:
            continue
        for incident in report.get("per_incident", []):
            incident_id = incident["incident_id"]
            if incident.get("pipeline_outcome") != "complete":
                continue
            evaluated = [
                r
                for r in incident.get("field_results", [])
                if r.get("error") is None
            ]
            if any(not r.get("exact_match") for r in evaluated):
                disqualified.add(incident_id)
            elif len(evaluated) >= min_exact_fields:
                qualified.add(incident_id)

    return sorted(qualified - disqualified - excluded)


def specificity_from_results(
    results: list[IncidentAuditResult],
    presumed_correct: set[int],
) -> dict[str, object]:
    """Compute the false-flag rate on the presumed-correct subset.

    Args:
        results: Per-incident audit results (any run).
        presumed_correct: Incident ids presumed correct.

    Returns:
        Dict with ``n_presumed_correct`` (auditable incidents in the
        subset), ``n_false_flagged`` (those with at least one reportable
        flag), ``n_false_flags`` (total reportable flags there),
        ``false_flag_rate`` with 95% Wilson ``false_flag_rate_ci``, and
        ``specificity`` (1 - rate). Rates are None when the subset has no
        auditable incidents.
    """
    subset = [
        r for r in results if r.incident_id in presumed_correct and r.auditable
    ]
    n_subset = len(subset)
    false_flagged = [r for r in subset if r.flags]
    n_false_flagged = len(false_flagged)
    n_false_flags = sum(len(r.flags) for r in subset)

    if n_subset == 0:
        return {
            "n_presumed_correct": 0,
            "n_false_flagged": 0,
            "n_false_flags": 0,
            "false_flag_rate": None,
            "false_flag_rate_ci": None,
            "specificity": None,
        }
    rate = n_false_flagged / n_subset
    return {
        "n_presumed_correct": n_subset,
        "n_false_flagged": n_false_flagged,
        "n_false_flags": n_false_flags,
        "false_flag_rate": rate,
        "false_flag_rate_ci": wilson_ci(n_false_flagged, n_subset),
        "specificity": 1 - rate,
    }


def print_specificity(summary: dict[str, object]) -> None:
    """Print the specificity summary block.

    Args:
        summary: Output of specificity_from_results().
    """
    print("\nFalse-flag suite (presumed-correct incidents):")
    if summary["false_flag_rate"] is None:
        print("  No auditable presumed-correct incidents in this run.")
        return
    low, high = summary["false_flag_rate_ci"]
    print(
        f"  {summary['n_false_flagged']}/{summary['n_presumed_correct']} "
        f"incidents falsely flagged ({summary['false_flag_rate']:.1%} "
        f"[{low:.1%}, {high:.1%}]), {summary['n_false_flags']} flags total"
    )
    print(f"  Specificity: {summary['specificity']:.1%}")
