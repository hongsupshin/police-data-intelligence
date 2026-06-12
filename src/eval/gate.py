"""Multi-objective accept/reject gate for pipeline changes.

A pure, offline, deterministic comparator over two :class:`HoldoutReport`
objects (a *before* and an *after*). It decides whether a pipeline change is
safe to ship by checking that a target metric improved while no *hard guard*
regressed. The three hard guards are:

* **target** (default completion rate) must not go backwards,
* **adversarial** hallucinations must stay at zero, and
* **correctness** (volume-weighted field accuracy on the *stable cohort* of
  incidents completed in both runs) must not drop.

**Fairness** (per-race completion equity) is computed and reported as a
*non-gating signal* — surfaced as warnings, never a veto. The per-race
completion-rate metric is statistically underpowered at our coarse 4-bucket
race categories and typical group sizes (n~18-39, where a 0.05 threshold is
well under one standard error), so it must not block a change on a 1-2 incident
swing; the deltas are reported for human judgment instead.

The cohort restriction and volume-weighting exist to avoid a real confound: a
fix that *raises* completion pulls harder incidents into the completed set,
which can lower naive full-set accuracy with no actual regression. Measuring
correctness on the intersection cohort keeps the comparison apples-to-apples.
Per-race ``mean_exact_accuracy`` drops are likewise surfaced as warnings.

See ``Notes/spec-p0-eval-moat.md`` for the full design.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from src.eval.holdout import (
    EvalResult,
    HoldoutReport,
    HoldoutSample,
    PipelineOutcome,
    _normalize_race,
    compute_fairness_metrics,
)


class GateTolerances(BaseModel):
    """Tolerances controlling how much regression each guard allows.

    Attributes:
        target: Maximum allowed regression in the target metric. Default 0.0
            means the target must not go backwards at all.
        correctness: Maximum allowed drop in cohort weighted field accuracy.
        fairness: Per-race completion-rate drop beyond which a (non-gating)
            fairness warning is surfaced. Fairness does not veto.
        min_group_n: Minimum (after-run) group size for a fairness completion
            drop to surface as a full warning; smaller groups produce a softer
            warning instead, since their rates are statistically noisy.
        min_cohort_n: Minimum stable-cohort size for the correctness comparison
            to be considered powered; below this a warning is emitted (an empty
            cohort means correctness is not evaluated at all).
    """

    target: float = 0.0
    correctness: float = 0.02
    fairness: float = 0.05
    min_group_n: int = 5
    min_cohort_n: int = 5


class GuardResult(BaseModel):
    """Outcome of a single guard check.

    Attributes:
        name: Guard identifier ("adversarial", "fairness", "correctness").
        passed: Whether the guard passed (no disqualifying regression).
        delta: The signed change measured by the guard (None if not numeric).
        detail: Human-readable summary of what was measured.
    """

    name: str
    passed: bool
    delta: float | None = None
    detail: str = ""


class GateDecision(BaseModel):
    """Result of a gate evaluation.

    Attributes:
        accept: True only if the target held and every hard guard passed
            (target, adversarial, correctness). Fairness is non-gating.
        target: Name of the target metric compared.
        target_before: Target value in the before report.
        target_after: Target value in the after report.
        target_delta: ``target_after - target_before``.
        guards: Per-guard results (adversarial, fairness, correctness).
        warnings: Non-blocking observations (e.g. low-coverage accuracy drops).
        reasons: Human-readable reasons the decision was a reject (empty if
            accepted).
    """

    accept: bool
    target: str
    target_before: float
    target_after: float
    target_delta: float
    guards: list[GuardResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def load_report(path: str | Path) -> HoldoutReport:
    """Load a saved holdout report JSON into a HoldoutReport.

    Args:
        path: Path to a report written by the eval harness
            (``report.model_dump_json()``).

    Returns:
        The parsed HoldoutReport.
    """
    return HoldoutReport.model_validate_json(Path(path).read_text())


def _by_id(report: HoldoutReport) -> dict[int, EvalResult]:
    """Map incident_id to its EvalResult, deduping (last entry wins).

    Saved reports can contain duplicate incident ids; downstream cohort and
    metric logic must treat each incident once.
    """
    return {er.incident_id: er for er in report.per_incident}


def _completed_ids(report: HoldoutReport) -> set[int]:
    """Incident ids whose (deduped) result completed."""
    return {
        iid
        for iid, er in _by_id(report).items()
        if er.pipeline_outcome == PipelineOutcome.COMPLETE
    }


def _result_signature(result: EvalResult) -> tuple[object, frozenset[object]]:
    """Signature of the fields the gate consumes (ignores timing/diagnostics).

    Used to compare duplicate-id entries by what actually affects the verdict
    (outcome + per-field comparison results), not by wall-clock fields like
    ``elapsed_seconds`` that legitimately differ between two evals.
    """
    return (
        result.pipeline_outcome,
        frozenset(
            (
                fr.field_name,
                fr.extracted_value,
                fr.ground_truth_value,
                fr.exact_match,
                fr.error,
            )
            for fr in result.field_results
        ),
    )


def _duplicate_conflicts(report: HoldoutReport) -> list[int]:
    """Incident ids appearing more than once with materially different results.

    Saved reports occasionally evaluate an incident twice (e.g. LLM variance),
    and :func:`_by_id` keeps only the last entry — so disagreeing duplicates
    silently drop data. This surfaces ids whose *result-relevant* fields differ
    (outcome or any field comparison), so the dedup is never invisible while
    ignoring duplicates that differ only in timing.
    """
    groups: dict[int, list[EvalResult]] = defaultdict(list)
    for er in report.per_incident:
        groups[er.incident_id].append(er)
    return sorted(
        iid
        for iid, results in groups.items()
        if len({_result_signature(er) for er in results}) > 1
    )


def _weighted_field_accuracy(
    results: list[EvalResult],
) -> tuple[float, dict[str, tuple[int, float]]]:
    """Volume-weighted exact accuracy over successfully extracted fields.

    A field counts as "extracted" when it has a non-null value and no
    comparison error. The aggregate pools every extracted field across the
    given incidents, so high-volume fields dominate and a low-coverage field's
    misses move the number only in proportion to how often it appears.

    Args:
        results: Incidents to aggregate over (typically the stable cohort).

    Returns:
        A tuple of (aggregate exact accuracy, per-field breakdown), where the
        breakdown maps field_name to (n_extracted, exact_accuracy).
    """
    per_field: dict[str, list[int]] = {}  # field -> [n_extracted, n_exact]
    for er in results:
        for fr in er.field_results:
            if fr.extracted_value is None or fr.error is not None:
                continue
            slot = per_field.setdefault(fr.field_name, [0, 0])
            slot[0] += 1
            if fr.exact_match:
                slot[1] += 1
    total_extracted = sum(slot[0] for slot in per_field.values())
    total_exact = sum(slot[1] for slot in per_field.values())
    aggregate = total_exact / total_extracted if total_extracted else 0.0
    breakdown = {
        field: (slot[0], slot[1] / slot[0] if slot[0] else 0.0)
        for field, slot in per_field.items()
    }
    return aggregate, breakdown


def _race_of(result: EvalResult) -> str:
    """Ground-truth race group for an incident, from its field_results.

    Reads the civilian_race comparison's ground-truth value (present even when
    the field was not extracted) and normalizes it. Returns "unknown" when no
    race ground truth is available.
    """
    for fr in result.field_results:
        if fr.field_name == "civilian_race" and fr.ground_truth_value:
            return _normalize_race(fr.ground_truth_value)
    return "unknown"


def recompute_fairness(report: HoldoutReport) -> dict[str, dict[str, float]]:
    """Recompute per-race fairness metrics from per-incident ground truth.

    The ``incident_ids`` eval path leaves ``report.fairness_metrics`` empty
    because it builds no samples. This reconstructs the race lookup from each
    incident's civilian_race ground truth and delegates to the existing
    :func:`compute_fairness_metrics` for the actual math, so both eval paths
    produce identical fairness numbers and before/after stay comparable.

    Args:
        report: The report to compute fairness over.

    Returns:
        Mapping of race group to ``{n, completion_rate, mean_exact_accuracy}``.
    """
    results = list(_by_id(report).values())
    samples = [
        HoldoutSample(
            incident_id=er.incident_id,
            year=0,
            race=_race_of(er),
            n_eval_fields=0,
        )
        for er in results
    ]
    return compute_fairness_metrics(results, samples)


def gate(
    before: HoldoutReport,
    after: HoldoutReport,
    adversarial_after: int,
    target: str = "completion_rate",
    tol: GateTolerances | None = None,
) -> GateDecision:
    """Decide whether the change from ``before`` to ``after`` is acceptable.

    Args:
        before: Holdout report for the baseline pipeline.
        after: Holdout report for the changed pipeline.
        adversarial_after: Hallucination count from the adversarial suite on
            the changed pipeline (must be 0 to pass the adversarial guard).
        target: Name of the (numeric) HoldoutReport attribute to treat as the
            metric the change is meant to improve.
        tol: Regression tolerances; defaults to :class:`GateTolerances`.

    Returns:
        A GateDecision recording the verdict, per-guard results, warnings, and
        reject reasons.
    """
    tol = tol or GateTolerances()
    guards: list[GuardResult] = []
    warnings: list[str] = []
    reasons: list[str] = []

    # Data-quality: surface disagreeing duplicate ids that dedup would hide.
    for label, report in (("before", before), ("after", after)):
        conflicts = _duplicate_conflicts(report)
        if conflicts:
            warnings.append(
                f"{label} report has duplicate incident ids with differing "
                f"results {conflicts} (deduped last-wins)"
            )

    # 1. Target: must not regress beyond tolerance.
    target_before = float(getattr(before, target))
    target_after = float(getattr(after, target))
    target_delta = target_after - target_before
    target_ok = target_delta >= -tol.target
    if not target_ok:
        reasons.append(f"target {target} regressed {target_delta:+.3f}")

    # 2. Adversarial: hard zero-hallucination veto.
    adversarial_ok = adversarial_after == 0
    guards.append(
        GuardResult(
            name="adversarial",
            passed=adversarial_ok,
            delta=float(adversarial_after),
            detail=f"{adversarial_after} hallucination(s) on the changed pipeline",
        )
    )
    if not adversarial_ok:
        reasons.append(f"adversarial hallucinations: {adversarial_after}")

    # 3. Fairness: non-gating signal. Per-race completion/accuracy drops surface
    #    as warnings only — never a veto. The per-race completion-rate metric is
    #    statistically underpowered at our coarse 4-bucket race categories and
    #    typical group sizes (n~18-39: a 0.05 threshold is well under one standard
    #    error), so a 1-2 incident swing must not block a change. The deltas are
    #    reported for human judgment, which is where significance is weighed.
    fair_before = recompute_fairness(before)
    fair_after = recompute_fairness(after)
    worst_completion_delta: float | None = None
    for race in sorted(set(fair_before) & set(fair_after)):
        b = fair_before[race]
        a = fair_after[race]
        n_after = a.get("n", 0.0)
        completion_delta = a["completion_rate"] - b["completion_rate"]
        if worst_completion_delta is None or completion_delta < worst_completion_delta:
            worst_completion_delta = completion_delta

        accuracy_delta = a["mean_exact_accuracy"] - b["mean_exact_accuracy"]
        if accuracy_delta < -tol.correctness:
            warnings.append(
                f"fairness: {race} mean_exact_accuracy {accuracy_delta:+.3f} "
                "(warning, non-gating)"
            )

        # tol.fairness / min_group_n now only decide which completion drops are
        # worth surfacing as a warning — not whether to veto.
        if completion_delta < -tol.fairness:
            if n_after >= tol.min_group_n:
                warnings.append(
                    f"fairness: {race} completion_rate {completion_delta:+.3f} "
                    f"(n={n_after:.0f}) (warning, non-gating)"
                )
            else:
                warnings.append(
                    f"fairness: {race} completion_rate {completion_delta:+.3f} "
                    f"but n={n_after:.0f} < {tol.min_group_n} (warning)"
                )
    guards.append(
        GuardResult(
            name="fairness",
            passed=True,
            delta=worst_completion_delta,
            detail="per-race completion equity (non-gating; reported only)",
        )
    )

    # 4. Correctness: volume-weighted accuracy on the stable cohort.
    cohort = _completed_ids(before) & _completed_ids(after)
    if len(cohort) < tol.min_cohort_n:
        detail = "not evaluated" if not cohort else "under-powered"
        warnings.append(
            f"correctness: cohort n={len(cohort)} < {tol.min_cohort_n} "
            f"({detail}) — accuracy comparison is unreliable"
        )
    before_by_id = _by_id(before)
    after_by_id = _by_id(after)
    before_acc, _ = _weighted_field_accuracy([before_by_id[i] for i in cohort])
    after_acc, _ = _weighted_field_accuracy([after_by_id[i] for i in cohort])
    correctness_delta = after_acc - before_acc
    correctness_ok = correctness_delta >= -tol.correctness
    guards.append(
        GuardResult(
            name="correctness",
            passed=correctness_ok,
            delta=correctness_delta,
            detail=f"cohort n={len(cohort)} volume-weighted accuracy",
        )
    )
    if not correctness_ok:
        reasons.append(
            f"correctness: cohort accuracy {correctness_delta:+.3f} "
            f"(n={len(cohort)})"
        )

    # Full-set per-field accuracy drops surface as warnings (not vetoes).
    before_field_metrics = {fm.field_name: fm for fm in before.field_metrics}
    for fm in after.field_metrics:
        prior = before_field_metrics.get(fm.field_name)
        if prior is None:
            continue
        field_delta = fm.exact_accuracy - prior.exact_accuracy
        if field_delta < -tol.correctness:
            warnings.append(
                f"field {fm.field_name} exact_accuracy {field_delta:+.3f} "
                f"(full-set warning, coverage {fm.coverage:.2f})"
            )

    accept = target_ok and adversarial_ok and correctness_ok
    return GateDecision(
        accept=accept,
        target=target,
        target_before=target_before,
        target_after=target_after,
        target_delta=target_delta,
        guards=guards,
        warnings=warnings,
        reasons=reasons,
    )
