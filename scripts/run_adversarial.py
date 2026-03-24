"""Run 20 fabricated incidents through the real pipeline (Tavily + Claude).

Patches only fetch_incident to inject fake data. Everything else — Tavily
search, validation, Claude LLM extraction, coordinator routing — runs for
real. Tests whether the pipeline hallucinates on non-existent incidents.

Scenarios are organized into 5 categories:
    A. Zero-result (obscure cities)
    B. Validation-rejection (medium cities, wrong dates)
    C. Hallucination trap (major cities, dates near real incidents)
    D. Common-name confusion (plausible names)
    E. Edge cases (NULL names)

Usage:
    python scripts/run_adversarial.py
"""

import json
import logging
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.run import run

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# 20 Fabricated Scenarios
# ---------------------------------------------------------------------------

FABRICATED_INCIDENTS = [
    # --- Category A: Zero-result (obscure cities) ---
    {
        "id": "99901",
        "category": "A",
        "dataset": "civilians_shot",
        "officer_name": "Reginald Q. Farnsworth",
        "civilian_name": "Bartholomew T. Quincy",
        "incident_date": date(2019, 7, 14),
        "location": "Marfa",
        "severity": "fatal",
    },
    {
        "id": "99902",
        "category": "A",
        "dataset": "civilians_shot",
        "officer_name": "Thornton J. Blackwell",
        "civilian_name": "Percival W. Ashford",
        "incident_date": date(2017, 3, 22),
        "location": "Alpine",
        "severity": "non-fatal",
    },
    {
        "id": "99903",
        "category": "A",
        "dataset": "officers_shot",
        "officer_name": "Winslow R. Carmichael",
        "civilian_name": "Eugenia M. Foxworth",
        "incident_date": date(2021, 11, 5),
        "location": "Presidio",
        "severity": "fatal",
    },
    {
        "id": "99904",
        "category": "A",
        "dataset": "civilians_shot",
        "officer_name": "Montgomery S. Beauregard",
        "civilian_name": "Augustina V. Pemberton",
        "incident_date": date(2016, 8, 19),
        "location": "Terlingua",
        "severity": "fatal",
    },
    # --- Category B: Validation-rejection (medium cities, wrong dates) ---
    {
        "id": "99905",
        "category": "B",
        "dataset": "civilians_shot",
        "officer_name": "Ricardo Mendez",
        "civilian_name": "Angela Foster",
        "incident_date": date(2018, 12, 3),
        "location": "El Paso",
        "severity": "non-fatal",
    },
    {
        "id": "99906",
        "category": "B",
        "dataset": "civilians_shot",
        "officer_name": "Derek Washington",
        "civilian_name": "Carlos Reyes",
        "incident_date": date(2020, 4, 17),
        "location": "Corpus Christi",
        "severity": "fatal",
    },
    {
        "id": "99907",
        "category": "B",
        "dataset": "officers_shot",
        "officer_name": "Thomas Bradley",
        "civilian_name": "Kevin Nguyen",
        "incident_date": date(2019, 9, 28),
        "location": "Fort Worth",
        "severity": "non-fatal",
    },
    {
        "id": "99908",
        "category": "B",
        "dataset": "civilians_shot",
        "officer_name": "Patricia Harmon",
        "civilian_name": "Raymond Ellis",
        "incident_date": date(2022, 6, 11),
        "location": "Lubbock",
        "severity": "non-fatal",
    },
    # --- Category C: Hallucination trap (dates near real incidents) ---
    {
        "id": "99909",
        "category": "C",
        "dataset": "civilians_shot",
        "officer_name": "Franklin D. Perriwinkle",
        "civilian_name": "Cornelius J. Montague",
        "incident_date": date(2018, 1, 28),
        "location": "Houston",
        "severity": "fatal",
    },
    {
        "id": "99910",
        "category": "C",
        "dataset": "civilians_shot",
        "officer_name": "Cedric L. Whitmore",
        "civilian_name": "Jasmine Q. Thornberry",
        "incident_date": date(2016, 7, 8),
        "location": "Dallas",
        "severity": "fatal",
        # 1 day after Dallas police ambush (Jul 7, 2016)
    },
    {
        "id": "99911",
        "category": "C",
        "dataset": "civilians_shot",
        "officer_name": "Vincent R. Calloway",
        "civilian_name": "Priscilla K. Davenport",
        "incident_date": date(2017, 6, 15),
        "location": "San Antonio",
        "severity": "fatal",
    },
    {
        "id": "99912",
        "category": "C",
        "dataset": "civilians_shot",
        "officer_name": "Gertrude A. Pendleton",
        "civilian_name": "Clarence O. Rutherford",
        "incident_date": date(2019, 1, 28),
        "location": "Houston",
        "severity": "fatal",
        # Exact date of Harding Street raid
    },
    {
        "id": "99913",
        "category": "C",
        "dataset": "civilians_shot",
        "officer_name": "Eleanora F. Strickland",
        "civilian_name": "Broderick T. Van Pelt",
        "incident_date": date(2020, 5, 31),
        "location": "Austin",
        "severity": "non-fatal",
        # George Floyd protest period
    },
    {
        "id": "99914",
        "category": "C",
        "dataset": "officers_shot",
        "officer_name": "Ignatius P. Worthington",
        "civilian_name": "Felicity M. Ashdown",
        "incident_date": date(2018, 9, 7),
        "location": "Dallas",
        "severity": "fatal",
        # 1 day after Botham Jean shooting (Sep 6, 2018)
    },
    # --- Category D: Common-name confusion ---
    {
        "id": "99915",
        "category": "D",
        "dataset": "civilians_shot",
        "officer_name": "James Rodriguez",
        "civilian_name": "John Williams",
        "incident_date": date(2018, 6, 20),
        "location": "Houston",
        "severity": "fatal",
    },
    {
        "id": "99916",
        "category": "D",
        "dataset": "civilians_shot",
        "officer_name": "Jose Garcia",
        "civilian_name": "Maria Lopez",
        "incident_date": date(2019, 3, 14),
        "location": "San Antonio",
        "severity": "non-fatal",
    },
    {
        "id": "99917",
        "category": "D",
        "dataset": "civilians_shot",
        "officer_name": "Robert Johnson",
        "civilian_name": "Michael Brown",
        "incident_date": date(2017, 11, 22),
        "location": "Dallas",
        "severity": "fatal",
    },
    {
        "id": "99918",
        "category": "D",
        "dataset": "officers_shot",
        "officer_name": "David Martinez",
        "civilian_name": "Chris Taylor",
        "incident_date": date(2021, 2, 8),
        "location": "Fort Worth",
        "severity": "non-fatal",
    },
    # --- Category E: Edge cases (NULL names) ---
    {
        "id": "99919",
        "category": "E",
        "dataset": "civilians_shot",
        "officer_name": None,
        "civilian_name": "Bartholomew T. Quincy",
        "incident_date": date(2018, 3, 15),
        "location": "Houston",
        "severity": "fatal",
    },
    {
        "id": "99920",
        "category": "E",
        "dataset": "civilians_shot",
        "officer_name": "Xavier J. Abernathy",
        "civilian_name": None,
        "incident_date": date(2019, 8, 22),
        "location": "Dallas",
        "severity": "non-fatal",
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output/adversarial")


def fake_fetch(conn, incident_id, dataset_type):
    """Return fabricated incident data instead of querying PostgreSQL."""
    for s in FABRICATED_INCIDENTS:
        if s["id"] == str(incident_id):
            return {
                "officer_name": s["officer_name"],
                "civilian_name": s["civilian_name"],
                "incident_date": s["incident_date"],
                "location": s["location"],
                "severity": s["severity"],
            }
    raise KeyError(f"Fabricated incident {incident_id} not found")


def extract_field_attr(obj, attr):
    """Get attribute from FieldExtraction (Pydantic model or dict)."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr)
    return None


def run_scenario(scenario):
    """Run a single fabricated scenario and return structured results."""
    iid = scenario["id"]
    dataset = scenario["dataset"]

    with patch("src.agents.load_node.fetch_incident", side_effect=fake_fetch):
        result = run(iid, dataset)

    # Extract results
    extracted = result.get("extracted_fields", [])
    conflicts = result.get("conflicting_fields") or []

    extracted_details = []
    for f in extracted:
        extracted_details.append({
            "field_name": extract_field_attr(f, "field_name"),
            "value": extract_field_attr(f, "value"),
            "confidence": extract_field_attr(f, "confidence"),
        })

    conflict_details = []
    for c in conflicts:
        conflict_details.append({
            "field_name": extract_field_attr(c, "field_name"),
            "conflict_type": extract_field_attr(c, "conflict_type"),
        })

    # Hallucination check
    fabricated_names = set()
    if scenario["civilian_name"]:
        fabricated_names.add(scenario["civilian_name"])
    if scenario["officer_name"]:
        fabricated_names.add(scenario["officer_name"])

    hallucinated = False
    for f in extracted:
        fname = extract_field_attr(f, "field_name")
        fval = extract_field_attr(f, "value")
        if fname in ("civilian_name", "officer_name") and fval in fabricated_names:
            hallucinated = True

    return {
        "id": iid,
        "category": scenario["category"],
        "city": scenario["location"],
        "dataset": dataset,
        "severity": scenario["severity"],
        "officer_name": scenario["officer_name"],
        "civilian_name": scenario["civilian_name"],
        "incident_date": str(scenario["incident_date"]),
        "current_stage": str(result.get("current_stage")),
        "escalation_reason": str(result.get("escalation_reason")),
        "requires_human_review": result.get("requires_human_review"),
        "retry_count": result.get("retry_count"),
        "n_extracted": len(extracted),
        "extracted_fields": extracted_details,
        "n_conflicts": len(conflicts),
        "conflicts": conflict_details,
        "hallucination_detected": hallucinated,
        "outcome_summary": result.get("outcome_summary"),
    }


def write_summary_md(results):
    """Write a markdown summary table for the blog post."""
    lines = [
        "# Adversarial Hallucination Test Results",
        "",
        f"**Date**: {date.today()}  ",
        f"**Scenarios**: {len(results)}  ",
        f"**Hallucinations detected**: "
        f"{sum(1 for r in results if r['hallucination_detected'])}",
        "",
        "## Summary Table",
        "",
        "| ID | Cat | City | Dataset | Stage | Reason | Retries | Extracted | Conflicts | Hallucination |",
        "|----|-----|------|---------|-------|--------|---------|-----------|-----------|---------------|",
    ]
    for r in results:
        stage = r["current_stage"].split(".")[-1] if "." in r["current_stage"] else r["current_stage"]
        reason = r["escalation_reason"]
        if reason == "None":
            reason = "—"
        halluc = "YES" if r["hallucination_detected"] else "No"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['city']} | {r['dataset']} "
            f"| {stage} | {reason} | {r['retry_count']} "
            f"| {r['n_extracted']} | {r['n_conflicts']} | {halluc} |"
        )

    # Category breakdown
    lines.extend([
        "",
        "## By Category",
        "",
    ])
    for cat in ["A", "B", "C", "D", "E"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        n_escalated = sum(1 for r in cat_results if "escalate" in r["current_stage"])
        n_halluc = sum(1 for r in cat_results if r["hallucination_detected"])
        n_extracted = sum(r["n_extracted"] for r in cat_results)
        lines.append(
            f"- **Category {cat}**: {len(cat_results)} scenarios, "
            f"{n_escalated} escalated, {n_extracted} total extractions, "
            f"{n_halluc} hallucinations"
        )

    return "\n".join(lines) + "\n"


def main():
    """Run all 20 scenarios and save results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, scenario in enumerate(FABRICATED_INCIDENTS):
        iid = scenario["id"]
        print(
            f"\n[{i + 1}/{len(FABRICATED_INCIDENTS)}] "
            f"Cat {scenario['category']}: {scenario['location']}, "
            f"{scenario['civilian_name'] or '(no civilian name)'}"
        )

        result = run_scenario(scenario)
        results.append(result)

        # Print inline summary
        stage = result["current_stage"]
        reason = result["escalation_reason"]
        print(
            f"  → {stage} | reason={reason} | "
            f"retries={result['retry_count']} | "
            f"extracted={result['n_extracted']} | "
            f"conflicts={result['n_conflicts']} | "
            f"hallucination={'YES' if result['hallucination_detected'] else 'No'}"
        )

    # Save full results JSON
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")

    # Save summary markdown
    summary_path = OUTPUT_DIR / "summary.md"
    summary_md = write_summary_md(results)
    summary_path.write_text(summary_md)
    print(f"Summary saved to {summary_path}")

    # Print final summary
    n_total = len(results)
    n_escalated = sum(1 for r in results if "escalate" in r["current_stage"])
    n_halluc = sum(1 for r in results if r["hallucination_detected"])
    print(f"\n{'=' * 60}")
    print(f"FINAL: {n_total} scenarios, {n_escalated} escalated, {n_halluc} hallucinations")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
