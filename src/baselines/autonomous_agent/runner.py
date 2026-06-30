"""Run the autonomous-agent baseline over the 20 fabricated adversarial incidents.

Reuses the shipped adversarial suite's fixtures (``FABRICATED_INCIDENTS``,
``fake_fetch``) and patches the **same** DB fetch the suite patches
(``src.agents.load_node.fetch_incident``) — every other step runs live (real
Tavily, real web, real model). Output lands in a **separate** directory
(``output/adversarial_baseline/``) so the shipped artifacts in
``output/adversarial/`` are never touched.

Usage:
    python -m src.baselines.autonomous_agent.runner            # full 20 x 3
    python -m src.baselines.autonomous_agent.runner --runs 1 --limit 2   # pilot
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import scripts.run_adversarial as adv
from src.agents.state import DatasetType
from src.baselines.autonomous_agent.agent import DEFAULT_MODEL, run_baseline_agent
from src.baselines.autonomous_agent.result import BaselineResult
from src.baselines.autonomous_agent.tools import BaselineTools
from src.config import Settings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/adversarial_baseline")
SHIPPED_RESULTS = Path("output/adversarial/results.json")
ARMS = ("naive", "informed")


def arm_dir(arm: str) -> Path:
    """Per-arm output directory (results.json + summary.md + transcripts/)."""
    return OUTPUT_DIR / arm


def build_client() -> Any:
    """Construct the real Anthropic client (mirrors run.py timeout/retry env)."""
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(override=True)
    return anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=float(os.getenv("ANTHROPIC_TIMEOUT", "90")),
        max_retries=int(os.getenv("ANTHROPIC_MAX_RETRIES", "2")),
    )


def run_one(
    scenario: dict,
    run_index: int,
    *,
    client: Any,
    settings: Settings,
    arm: str = "informed",
    model: str = DEFAULT_MODEL,
    transcript_dir: str | Path | None = None,
) -> BaselineResult:
    """Run the baseline agent on one fabricated scenario, patching only the DB fetch.

    Args:
        scenario: One entry from ``FABRICATED_INCIDENTS``.
        run_index: Which repeated run this is (0-based).
        client: Anthropic client (or a mock).
        settings: Pipeline settings.
        arm: "naive" or "informed" baseline arm.
        model: Model id.
        transcript_dir: Where to write the transcript (None to skip).

    Returns:
        The ``BaselineResult`` for this incident/run.
    """
    iid = str(scenario["id"])
    dataset = DatasetType(scenario["dataset"])
    category = scenario["category"]
    with patch("src.agents.load_node.fetch_incident", side_effect=adv.fake_fetch):
        tools = BaselineTools(iid, dataset, settings=settings)
        return run_baseline_agent(
            iid,
            dataset,
            category,
            client=client,
            tools=tools,
            arm=arm,
            run_index=run_index,
            model=model,
            transcript_dir=transcript_dir,
        )


def run_suite(
    *,
    client: Any,
    settings: Settings | None = None,
    arm: str = "informed",
    n_runs: int = 3,
    start_run: int = 0,
    prior_results: list[BaselineResult] | None = None,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    transcript_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    on_result=None,
) -> dict:
    """Run every fabricated incident through the baseline agent ``n_runs`` times.

    Args:
        client: Anthropic client (or a mock).
        settings: Pipeline settings (defaults to a fresh ``Settings()``).
        arm: "naive" or "informed" baseline arm.
        n_runs: How many fresh passes to run (for run-to-run variance).
        start_run: First run index to run (for resume — runs start_run..start_run+n_runs-1).
        prior_results: Already-completed results to fold into the aggregate (resume).
        limit: Run only the first ``limit`` incidents (for a plumbing pilot).
        model: Model id.
        transcript_dir: Where to write transcripts (None to skip).
        checkpoint_dir: If set, write results.json + summary.md after every
            incident so a partial run leaves valid output (crash safety).
        on_result: Optional callback ``(run_index, scenario, result)`` per incident.

    Returns:
        Aggregate dict with per-run summaries, all per-incident results, and the
        shipped-pipeline comparison.
    """
    settings = settings or Settings()
    scenarios = adv.FABRICATED_INCIDENTS[:limit] if limit else adv.FABRICATED_INCIDENTS

    results: list[BaselineResult] = list(prior_results or [])
    for run_index in range(start_run, start_run + n_runs):
        for scenario in scenarios:
            result = run_one(
                scenario,
                run_index,
                client=client,
                settings=settings,
                arm=arm,
                model=model,
                transcript_dir=transcript_dir,
            )
            results.append(result)
            if on_result is not None:
                on_result(run_index, scenario, result)
            if checkpoint_dir is not None:
                write_outputs(checkpoint_dir, _aggregate(results, arm=arm, model=model))

    return _aggregate(results, arm=arm, model=model)


def write_outputs(out_dir: str | Path, aggregate: dict) -> None:
    """Write results.json + summary.md for an aggregate to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(aggregate, indent=2, default=str))
    (out_dir / "summary.md").write_text(write_summary_md(aggregate))


def _aggregate(results: list[BaselineResult], *, arm: str, model: str) -> dict:
    """Summarize per-run completion / fabrication / cost across all results.

    Iterates the run indices actually present (not a fixed range), so a resumed
    run that adds run 1-2 onto an existing run 0 aggregates cleanly.
    """
    per_run = []
    for run_index in sorted({r.run_index for r in results}):
        run = [r for r in results if r.run_index == run_index]
        per_run.append(
            {
                "run_index": run_index,
                "n_incidents": len(run),
                "completed": sum(r.completed for r in run),
                "declined": sum(r.declined for r in run),
                "committed_fabrications": sum(r.committed_fabrications for r in run),
                "n_parametric": sum(r.n_parametric for r in run),
                "n_wrong_article": sum(r.n_wrong_article for r in run),
                "n_planted_name": sum(r.n_planted_name for r in run),
                "errors": sum(bool(r.error) for r in run),
                "cost_usd": round(sum(r.usage.cost_usd for r in run), 4),
                "tavily_credits": sum(r.usage.tavily_credits for r in run),
                "tavily_cost_usd": round(sum(r.usage.tavily_cost_usd for r in run), 4),
                "wall_clock_seconds": round(sum(r.usage.wall_clock_seconds for r in run), 1),
            }
        )
    claude = round(sum(r.usage.cost_usd for r in results), 4)
    tavily = round(sum(r.usage.tavily_cost_usd for r in results), 4)
    return {
        "arm": arm,
        "model": model,
        "n_runs": len(per_run),
        "n_incidents": per_run[0]["n_incidents"] if per_run else 0,
        "per_run": per_run,
        "total_claude_cost_usd": claude,
        "total_tavily_cost_usd": tavily,
        "total_cost_usd": round(claude + tavily, 4),
        "results": [r.model_dump() for r in results],
        "shipped_comparison": _shipped_summary(),
    }


def load_prior_results(out_dir: str | Path, before_run: int) -> list[BaselineResult]:
    """Load valid prior results (run_index < before_run, no error) for resume.

    Lets a resumed sweep keep an already-completed run 0 instead of re-spending on
    it. Errored prior results are dropped so they get re-run.
    """
    path = Path(out_dir) / "results.json"
    if before_run <= 0 or not path.exists():
        return []
    data = json.loads(path.read_text())
    prior = []
    for d in data.get("results", []):
        if d.get("run_index", 0) < before_run and not d.get("error"):
            prior.append(BaselineResult(**d))
    return prior


def _shipped_summary() -> dict:
    """Read the shipped adversarial results for side-by-side context (if present)."""
    if not SHIPPED_RESULTS.exists():
        return {"available": False}
    data = json.loads(SHIPPED_RESULTS.read_text())
    reasons: dict[str, int] = {}
    for r in data:
        reasons[r.get("escalation_reason")] = reasons.get(r.get("escalation_reason"), 0) + 1
    return {
        "available": True,
        "n_incidents": len(data),
        "completed": sum(1 for r in data if "complete" in str(r.get("current_stage"))),
        "hallucinations_detected": sum(1 for r in data if r.get("hallucination_detected")),
        "escalation_reasons": reasons,
    }


def write_summary_md(aggregate: dict) -> str:
    """Render a markdown summary: shipped vs baseline, per run, with cost."""
    lines = [
        "# Autonomous-agent baseline — adversarial results",
        "",
        f"**Arm**: {aggregate.get('arm', 'informed')}  ",
        f"**Model**: {aggregate['model']}  ",
        f"**Runs**: {aggregate['n_runs']} x {aggregate['n_incidents']} incidents  ",
        f"**Total cost**: ${aggregate['total_cost_usd']} "
        f"(Claude ${aggregate.get('total_claude_cost_usd', aggregate['total_cost_usd'])} "
        f"+ Tavily ${aggregate.get('total_tavily_cost_usd', 0)})",
        "",
        "## Shipped pipeline vs baseline",
        "",
        "| Arm | Completed | Committed fabrications |",
        "|-----|-----------|------------------------|",
    ]
    shipped = aggregate.get("shipped_comparison", {})
    if shipped.get("available"):
        lines.append(
            f"| Shipped pipeline | {shipped['completed']}/{shipped['n_incidents']} "
            f"| {shipped['hallucinations_detected']} (planted-name detector) |"
        )
    else:
        lines.append("| Shipped pipeline | 0/20 (see output/adversarial) | 0 |")
    for run in aggregate["per_run"]:
        lines.append(
            f"| Baseline run {run['run_index']} | {run['completed']}/{run['n_incidents']} "
            f"| {run['committed_fabrications']} "
            f"(parametric {run['n_parametric']}, wrong-article {run['n_wrong_article']}, "
            f"planted-name {run['n_planted_name']}) |"
        )
    lines.extend(
        [
            "",
            "## Per run",
            "",
            "| Run | Completed | Declined | Errors | Committed fab. | Claude $ | Tavily credits | Tavily $ | Wall-clock (min) |",
            "|-----|-----------|----------|--------|----------------|----------|----------------|----------|------------------|",
        ]
    )
    for run in aggregate["per_run"]:
        mins = round(run.get("wall_clock_seconds", 0) / 60, 1)
        lines.append(
            f"| {run['run_index']} | {run['completed']} | {run['declined']} "
            f"| {run['errors']} | {run['committed_fabrications']} "
            f"| {run['cost_usd']} | {run['tavily_credits']} "
            f"| {run.get('tavily_cost_usd', 0)} | {mins} |"
        )
    lines.append("")
    return "\n".join(lines)


def _make_progress(arm: str):
    def _print_progress(run_index: int, scenario: dict, result: BaselineResult) -> None:
        outcome = "COMPLETED" if result.completed else "declined"
        print(
            f"[{arm} run {run_index}] {scenario['id']} cat {scenario['category']} "
            f"{scenario['location']}: {outcome}, "
            f"committed_fab={result.committed_fabrications} "
            f"(parametric={result.n_parametric}, wrong_article={result.n_wrong_article}) "
            f"cost=${result.usage.cost_usd:.4f}"
            + (f" ERROR={result.error}" if result.error else "")
        )

    return _print_progress


def main() -> None:
    """Parse CLI args, run the chosen arm(s) live, and write per-arm results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="How many fresh passes to run.")
    parser.add_argument(
        "--start-run",
        type=int,
        default=0,
        help="First run index (resume: keeps valid runs < this from results.json).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N incidents (pilot)."
    )
    parser.add_argument(
        "--arm",
        choices=["naive", "informed", "both"],
        default="both",
        help="Which baseline arm(s) to run.",
    )
    parser.add_argument(
        "--model", type=str, default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    )
    args = parser.parse_args()

    arms = list(ARMS) if args.arm == "both" else [args.arm]
    client = build_client()
    for arm in arms:
        out = arm_dir(arm)
        out.mkdir(parents=True, exist_ok=True)
        prior = load_prior_results(out, args.start_run)
        print(f"\n===== ARM: {arm} =====")
        if prior:
            print(f"[{arm}] resuming: kept {len(prior)} valid prior results (runs < {args.start_run})")
        aggregate = run_suite(
            client=client,
            arm=arm,
            n_runs=args.runs,
            start_run=args.start_run,
            prior_results=prior,
            limit=args.limit,
            model=args.model,
            transcript_dir=out / "transcripts",
            checkpoint_dir=out,
            on_result=_make_progress(arm),
        )
        write_outputs(out, aggregate)
        print(f"[{arm}] Results: {out / 'results.json'}  Summary: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
