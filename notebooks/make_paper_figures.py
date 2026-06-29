"""Generate the SciPy-paper result figure from the canonical holdout reports.

One two-panel overview figure, backed only by counts already saved in the canonical
holdout report pair (no new inference):

* ``fig_overview.png`` -- panel (a) headline completion / exact / fuzzy, civilians vs
  officers, with 95% Wilson confidence intervals; panel (b) completion rate by year
  cohort, which decomposes completion over time to show that retrieval recall (news
  coverage availability), not reasoning, sets the ceiling.

Per-field accuracy stays as the manuscript's tables (it documents the non-agentic
extraction layer and carries coverage compactly), and the outcome-composition numbers
(relevance-judge activity) are printed for the manuscript table rather than plotted.

The Wilson intervals reuse :func:`src.eval.ci.wilson_ci`. Run from the repo root with
the project interpreter (the ``police-data-intel`` conda env, Python 3.11):

    python notebooks/make_paper_figures.py

By default the PNG is written into the SciPy paper's ``figures/`` directory; pass
``--out-dir`` to override. The plotted values are printed to stdout so they can be
cross-checked against the manuscript tables without opening the image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

# Make ``src`` importable regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.ci import wilson_ci  # noqa: E402  (import after sys.path setup)

# --- Paths -----------------------------------------------------------------

REPORTS = {
    "civilians_shot": REPO_ROOT
    / "output/eval/holdout_civilians_shot_20260613_231717.json",
    "officers_shot": REPO_ROOT
    / "output/eval/holdout_officers_shot_20260613_231717.json",
}
DEFAULT_OUT_DIR = (
    REPO_ROOT.parents[1]
    / "Projects/SciPy/scipy_proceedings/papers/hongsup_shin/figures"
)

# --- Presentation ----------------------------------------------------------

# Colourblind-safe pair from seaborn's "colorblind" palette (blue / orange).
COLOR = {"civilians_shot": "#0173b2", "officers_shot": "#de8f05"}
LABEL = {"civilians_shot": "Civilians", "officers_shot": "Officers"}
DPI = 200
WIDTH = 0.38

# Year cohorts for the temporal panel.
COHORTS = [
    ("2014–16", range(2014, 2017)),
    ("2017–18", range(2017, 2019)),
    ("2019–21", range(2019, 2022)),
    ("2022–24", range(2022, 2025)),
]


# --- Data helpers ----------------------------------------------------------


def load_reports() -> dict[str, dict]:
    """Load the canonical holdout report pair as plain dicts."""
    return {ds: json.loads(path.read_text()) for ds, path in REPORTS.items()}


def aggregate_from_report(report: dict) -> dict[str, tuple[int, int]]:
    """Sum extracted / exact / fuzzy counts across all fields.

    Returns counts (numerator, denominator) for the three headline proportions,
    with the extraction count as the denominator for exact and fuzzy.
    """
    n_extracted = sum(fm["n_extracted"] for fm in report["field_metrics"])
    n_exact = sum(fm["n_exact_match"] for fm in report["field_metrics"])
    n_fuzzy = sum(fm["n_fuzzy_match"] for fm in report["field_metrics"])
    return {
        "completion": (report["n_completed"], report["n_incidents"]),
        "exact": (n_exact, n_extracted),
        "fuzzy": (n_fuzzy, n_extracted),
    }


def cohort_completion(report: dict) -> list[tuple[str, int, int]]:
    """Completion (completed, total) per year cohort.

    Joins ``samples[].year`` to ``per_incident[].pipeline_outcome`` on
    ``incident_id``.
    """
    year_by_id = {s["incident_id"]: s["year"] for s in report["samples"]}
    completed_by_id = {
        pi["incident_id"]: pi["pipeline_outcome"] == "complete"
        for pi in report["per_incident"]
    }
    rows = []
    for name, years in COHORTS:
        yrs = set(years)
        ids = [i for i, y in year_by_id.items() if y in yrs]
        total = len(ids)
        completed = sum(1 for i in ids if completed_by_id.get(i, False))
        rows.append((name, completed, total))
    binned = sum(t for _, _, t in rows)
    if binned != report["n_incidents"]:
        raise ValueError(
            f"cohort bins cover {binned} of {report['n_incidents']} incidents; "
            "some years fall outside COHORTS and would be dropped silently"
        )
    return rows


def escalation_breakdown(report: dict) -> dict[str, int]:
    """Count completions and each escalation reason in a holdout report."""
    counts = {"complete": 0, "max_retries": 0, "irrelevant_sources": 0,
              "insufficient_sources": 0}
    for pi in report["per_incident"]:
        if pi["pipeline_outcome"] == "complete":
            counts["complete"] += 1
        else:
            counts[pi["escalation_reason"]] += 1
    return counts


def _err(successes: int, total: int) -> tuple[float, float, float]:
    """Point estimate and (lower, upper) error-bar magnitudes in percent."""
    p = 100.0 * successes / total if total else 0.0
    lo, hi = wilson_ci(successes, total)
    return p, p - lo * 100.0, hi * 100.0 - p


# --- Figure ----------------------------------------------------------------


def make_overview(reports: dict[str, dict], out_dir: Path) -> Path:
    """Two-panel overview: headline metrics (a) and completion by cohort (b)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True,
                                   layout="constrained")
    datasets = ("civilians_shot", "officers_shot")

    # Panel (a): headline completion / exact / fuzzy with Wilson CIs.
    metrics = [("completion", "Completion"), ("exact", "Exact match"),
               ("fuzzy", "Fuzzy match")]
    aggs = {ds: aggregate_from_report(reports[ds]) for ds in datasets}
    x = np.arange(len(metrics))
    print("\n[Panel a] headline metrics (point [Wilson 95% CI], n/N)")
    for i, ds in enumerate(datasets):
        offset = (i - 0.5) * WIDTH
        pts, lo_err, hi_err = [], [], []
        for key, _ in metrics:
            s, t = aggs[ds][key]
            p, lo, hi = _err(s, t)
            pts.append(p)
            lo_err.append(lo)
            hi_err.append(hi)
            print(f"  {LABEL[ds]:9s} {key:11s} {p:5.1f}% "
                  f"[{p - lo:4.1f}, {p + hi:4.1f}]  ({s}/{t})")
        ax1.bar(x + offset, pts, WIDTH, yerr=[lo_err, hi_err], capsize=4,
                color=COLOR[ds], label=LABEL[ds], error_kw={"elinewidth": 1.2})
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[1] for m in metrics])
    ax1.set_ylabel("Percent")
    ax1.set_ylim(0, 112)
    ax1.set_title("(a) Overall performance")
    ax1.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    sns.despine(ax=ax1)

    # Panel (b): completion rate by year cohort with Wilson CIs.
    cohorts = {ds: cohort_completion(reports[ds]) for ds in datasets}
    names = [c[0] for c in cohorts["civilians_shot"]]
    xb = np.arange(len(names))
    print("\n[Panel b] completion by year cohort (completed/total)")
    for i, ds in enumerate(datasets):
        offset = (i - 0.5) * WIDTH
        rows = cohorts[ds]
        pts, lo_err, hi_err = [], [], []
        for _, c, t in rows:
            p, lo, hi = _err(c, t)
            pts.append(p)
            lo_err.append(lo)
            hi_err.append(hi)
        bars = ax2.bar(xb + offset, pts, WIDTH, yerr=[lo_err, hi_err], capsize=3,
                       color=COLOR[ds], error_kw={"elinewidth": 1.0})
        for b, (name, c, t) in zip(bars, rows):
            ax2.annotate(f"$N$={t}", (b.get_x() + b.get_width() / 2, 3),
                         ha="center", va="bottom", fontsize=7, color="white")
            print(f"  {LABEL[ds]:9s} {name}: {100.0 * c / t:5.1f}%  ({c}/{t})")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(names)
    ax2.set_xlabel("Incident year cohort")
    ax2.set_title("(b) Completion by year cohort")
    ax2.tick_params(labelleft=True)
    sns.despine(ax=ax2)

    fig.suptitle("Held-out evaluation ($N$=100)", fontsize=14)
    return _save(fig, out_dir / "fig_overview.png")


def print_outcome_composition(reports: dict[str, dict]) -> None:
    """Print the outcome-composition numbers backing the manuscript table.

    Not plotted: a stacked bar reads as a deterministic-vs-agentic *comparison* the
    data does not support (the methods are sequential, deterministic then judges),
    so this composition lives as a table in the paper. This printout keeps the
    table reproducible from the saved reports.
    """
    civ = escalation_breakdown(reports["civilians_shot"])
    off = escalation_breakdown(reports["officers_shot"])
    # Adversarial counts come straight from output/adversarial/summary.md:
    # 0 completed, 17 escalate@search (max_retries), 2 relevance vetoes, 1 insufficient.
    adv = {"complete": 0, "max_retries": 17, "irrelevant_sources": 2,
           "insufficient_sources": 1}
    rows = [("complete", "Completed"),
            ("max_retries", "Retrieval gap (search)"),
            ("irrelevant_sources", "Relevance-judge veto"),
            ("insufficient_sources", "Insufficient sources")]
    print("\n[Outcome-composition table] counts per evaluation set")
    print(f"  {'Outcome':24s} {'Civ(100)':>9s} {'Off(100)':>9s} {'Adv(20)':>9s}")
    for key, label in rows:
        print(f"  {label:24s} {civ[key]:9d} {off[key]:9d} {adv[key]:9d}")


def _save(fig: Figure, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Generate the overview figure and print the table numbers it backs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory to write the PNG into.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150

    reports = load_reports()
    path = make_overview(reports, args.out_dir)
    print_outcome_composition(reports)
    print("\nWrote:\n ", path)


if __name__ == "__main__":
    main()
