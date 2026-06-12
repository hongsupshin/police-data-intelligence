"""Unit tests for src/eval/gate.py — the multi-objective accept/reject gate.

Covers the four decision axes (target, adversarial, fairness, correctness) on
synthetic reports, the intersection-cohort and volume-weighting behaviour that
isolates real regression from completion-mix noise, and a replay of the real
PR #52 before/after holdout reports (skipped when those gitignored fixtures are
absent).
"""

from pathlib import Path

import pytest

from src.agents.state import DatasetType
from src.eval.gate import (
    GateTolerances,
    _race_of,
    _weighted_field_accuracy,
    gate,
    recompute_fairness,
)
from src.eval.holdout import (
    EvalError,
    EvalResult,
    FieldMetrics,
    HoldoutReport,
    HoldoutSample,
    MatchResult,
    PipelineOutcome,
    compute_fairness_metrics,
)

DT = DatasetType.OFFICERS_SHOT
COMPLETE = PipelineOutcome.COMPLETE
ESCALATE = PipelineOutcome.ESCALATE


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _race_mr(race: str) -> MatchResult:
    """Grouping-only race comparison: ground truth present, never extracted."""
    return MatchResult(
        field_name="civilian_race",
        extracted_value=None,
        ground_truth_value=race,
        error=EvalError.NO_EXTRACTION,
    )


def _outcome_mr(state: str) -> MatchResult:
    """Outcome comparison in one of three states: exact, wrong, missing."""
    if state == "exact":
        return MatchResult(
            field_name="outcome",
            extracted_value="fatal",
            ground_truth_value="fatal",
            exact_match=True,
        )
    if state == "wrong":
        return MatchResult(
            field_name="outcome",
            extracted_value="injury",
            ground_truth_value="fatal",
            exact_match=False,
        )
    return MatchResult(
        field_name="outcome",
        extracted_value=None,
        ground_truth_value="fatal",
        error=EvalError.NO_EXTRACTION,
    )


def _inc(
    iid: int,
    pipeline: PipelineOutcome,
    outcome_state: str,
    race: str | None = None,
) -> EvalResult:
    """Build an EvalResult with an outcome comparison and optional race group."""
    fields = [_outcome_mr(outcome_state)]
    if race is not None:
        fields.append(_race_mr(race))
    return EvalResult(
        incident_id=iid,
        dataset_type=DT,
        pipeline_outcome=pipeline,
        field_results=fields,
    )


def _report(
    results: list[EvalResult],
    field_metrics: list[FieldMetrics] | None = None,
) -> HoldoutReport:
    """Build a HoldoutReport with completion_rate derived from results."""
    n = len(results)
    n_completed = sum(1 for r in results if r.pipeline_outcome == COMPLETE)
    return HoldoutReport(
        dataset_type=DT,
        n_incidents=n,
        n_completed=n_completed,
        n_escalated=n - n_completed,
        completion_rate=n_completed / n if n else 0.0,
        field_metrics=field_metrics or [],
        per_incident=results,
    )


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for the gate's pure helpers."""

    def test_weighted_field_accuracy_pools_by_extraction(self) -> None:
        """Aggregate is total exact over total extracted; misses lower it."""
        results = [
            _inc(1, COMPLETE, "exact"),
            _inc(2, COMPLETE, "exact"),
            _inc(3, COMPLETE, "wrong"),
        ]
        agg, breakdown = _weighted_field_accuracy(results)
        assert agg == pytest.approx(2 / 3)
        assert breakdown["outcome"] == (3, pytest.approx(2 / 3))

    def test_weighted_field_accuracy_skips_unextracted(self) -> None:
        """Fields with no value or a comparison error are not counted."""
        results = [_inc(1, ESCALATE, "missing", race="BLACK")]
        agg, breakdown = _weighted_field_accuracy(results)
        assert agg == 0.0
        assert breakdown == {}

    def test_race_of_normalizes_ground_truth(self) -> None:
        """Race group comes from the civilian_race GT, normalized."""
        assert _race_of(_inc(1, COMPLETE, "exact", race="BLACK")) == "black"
        assert _race_of(_inc(2, COMPLETE, "exact")) == "unknown"

    def test_recompute_fairness_matches_compute_fairness_metrics(self) -> None:
        """recompute_fairness delegates to the canonical fairness function."""
        results = [
            _inc(1, COMPLETE, "exact", race="BLACK"),
            _inc(2, ESCALATE, "missing", race="BLACK"),
            _inc(3, COMPLETE, "wrong", race="WHITE"),
        ]
        report = _report(results)
        samples = [
            HoldoutSample(
                incident_id=er.incident_id,
                year=0,
                race=_race_of(er),
                n_eval_fields=0,
            )
            for er in results
        ]
        assert recompute_fairness(report) == compute_fairness_metrics(results, samples)


# ---------------------------------------------------------------------------
# Correctness guard
# ---------------------------------------------------------------------------


class TestCorrectnessGuard:
    """The correctness guard reads the stable cohort, volume-weighted."""

    def test_cohort_isolates_new_incident_accuracy(self) -> None:
        """Low accuracy on newly completed incidents does not veto."""
        before = _report(
            [
                _inc(1, COMPLETE, "exact"),
                _inc(2, COMPLETE, "exact"),
                _inc(3, COMPLETE, "exact"),
                _inc(4, ESCALATE, "missing"),
                _inc(5, ESCALATE, "missing"),
            ]
        )
        after = _report(
            [
                _inc(1, COMPLETE, "exact"),
                _inc(2, COMPLETE, "exact"),
                _inc(3, COMPLETE, "exact"),
                _inc(4, COMPLETE, "wrong"),
                _inc(5, COMPLETE, "wrong"),
            ]
        )
        decision = gate(before, after, adversarial_after=0)
        correctness = next(g for g in decision.guards if g.name == "correctness")
        assert correctness.passed
        assert correctness.delta == pytest.approx(0.0)
        assert decision.accept

    def test_cohort_regression_vetoes(self) -> None:
        """A real accuracy drop on the cohort rejects the change."""
        before = _report([_inc(i, COMPLETE, "exact") for i in (1, 2, 3)])
        after = _report([_inc(i, COMPLETE, "wrong") for i in (1, 2, 3)])
        decision = gate(before, after, adversarial_after=0)
        correctness = next(g for g in decision.guards if g.name == "correctness")
        assert not correctness.passed
        assert not decision.accept

    def test_field_metrics_drop_is_warning_not_veto(self) -> None:
        """A full-set per-field accuracy drop warns while the cohort passes."""
        before = _report(
            [_inc(i, COMPLETE, "exact") for i in (1, 2, 3)],
            field_metrics=[
                FieldMetrics(
                    field_name="civilian_race",
                    n_evaluable=80,
                    n_extracted=3,
                    n_exact_match=2,
                    n_fuzzy_match=2,
                    coverage=0.04,
                    exact_accuracy=0.75,
                    fuzzy_accuracy=0.75,
                )
            ],
        )
        after = _report(
            [_inc(i, COMPLETE, "exact") for i in (1, 2, 3)],
            field_metrics=[
                FieldMetrics(
                    field_name="civilian_race",
                    n_evaluable=80,
                    n_extracted=9,
                    n_exact_match=4,
                    n_fuzzy_match=4,
                    coverage=0.09,
                    exact_accuracy=0.44,
                    fuzzy_accuracy=0.44,
                )
            ],
        )
        decision = gate(before, after, adversarial_after=0)
        assert decision.accept
        assert any("civilian_race" in w for w in decision.warnings)


# ---------------------------------------------------------------------------
# Fairness guard
# ---------------------------------------------------------------------------


class TestFairnessGuard:
    """Fairness is a non-gating signal: per-race drops warn, never veto."""

    def test_completion_regression_warns_not_vetoes(self) -> None:
        """A sizeable group losing completion warns but does not reject."""
        before = _report(
            [_inc(i, COMPLETE, "exact", race="BLACK") for i in range(1, 7)]
            + [_inc(i, ESCALATE, "missing", race="WHITE") for i in range(7, 13)]
        )
        after = _report(
            [_inc(1, COMPLETE, "exact", race="BLACK"), _inc(2, COMPLETE, "exact", race="BLACK")]
            + [_inc(i, ESCALATE, "missing", race="BLACK") for i in range(3, 7)]
            + [_inc(i, COMPLETE, "exact", race="WHITE") for i in range(7, 13)]
        )
        # BLACK completion crashes (6/6 -> 2/6) while overall completion rises
        # (6/12 -> 8/12) — only fairness moves, and it must not veto.
        decision = gate(before, after, adversarial_after=0)
        fairness = next(g for g in decision.guards if g.name == "fairness")
        assert fairness.passed
        assert decision.accept
        assert not any("fairness" in r for r in decision.reasons)
        assert any("fairness" in w and "black" in w for w in decision.warnings)

    def test_accuracy_regression_is_warning_not_veto(self) -> None:
        """A per-race mean_exact_accuracy drop warns but does not veto."""
        before = _report(
            [_inc(i, COMPLETE, "exact", race="BLACK") for i in (1, 2, 3)]
            + [_inc(i, ESCALATE, "missing", race="BLACK") for i in (4, 5, 6)]
        )
        after = _report(
            [_inc(i, COMPLETE, "exact", race="BLACK") for i in (1, 2, 3)]
            + [_inc(i, COMPLETE, "wrong", race="BLACK") for i in (4, 5, 6)]
        )
        decision = gate(before, after, adversarial_after=0)
        fairness = next(g for g in decision.guards if g.name == "fairness")
        assert fairness.passed
        assert decision.accept
        assert any("mean_exact_accuracy" in w for w in decision.warnings)

    def test_small_group_completion_drop_is_warning(self) -> None:
        """A completion drop in a sub-min_group_n group only warns."""
        before = _report(
            [_inc(i, COMPLETE, "exact", race="BLACK") for i in (1, 2, 3)]
            + [_inc(i, COMPLETE, "exact", race="WHITE") for i in range(4, 10)]
        )
        after = _report(
            [_inc(i, ESCALATE, "missing", race="BLACK") for i in (1, 2, 3)]
            + [_inc(i, COMPLETE, "exact", race="WHITE") for i in range(4, 10)]
        )
        decision = gate(before, after, adversarial_after=0, target="completion_rate")
        fairness = next(g for g in decision.guards if g.name == "fairness")
        # black (n=3 < 5) lost all completion -> warning, not a hard veto.
        assert fairness.passed
        assert any("black" in w for w in decision.warnings)


# ---------------------------------------------------------------------------
# Target & adversarial guards
# ---------------------------------------------------------------------------


class TestTargetAndAdversarial:
    """Target regression and any hallucination both block acceptance."""

    def test_adversarial_hallucination_vetoes(self) -> None:
        """A positive target cannot override a hallucination."""
        before = _report([_inc(i, ESCALATE, "missing") for i in (1, 2)])
        after = _report([_inc(i, COMPLETE, "exact") for i in (1, 2)])
        decision = gate(before, after, adversarial_after=1)
        adversarial = next(g for g in decision.guards if g.name == "adversarial")
        assert not adversarial.passed
        assert not decision.accept

    def test_target_regression_vetoes(self) -> None:
        """A drop in the target metric rejects the change."""
        before = _report([_inc(i, COMPLETE, "exact") for i in (1, 2, 3)])
        after = _report(
            [_inc(1, COMPLETE, "exact"), _inc(2, ESCALATE, "missing"), _inc(3, ESCALATE, "missing")]
        )
        decision = gate(before, after, adversarial_after=0)
        assert decision.target_delta < 0
        assert not decision.accept


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


class TestTolerances:
    """Tolerance knobs change the verdict on a borderline change."""

    def test_correctness_tolerance_relaxes_veto(self) -> None:
        """A cohort accuracy drop that the default rejects, a loose tol accepts."""
        before = _report([_inc(i, COMPLETE, "exact") for i in range(1, 11)])
        after = _report(
            [_inc(i, COMPLETE, "exact") for i in range(1, 10)] + [_inc(10, COMPLETE, "wrong")]
        )  # cohort 10, accuracy 1.0 -> 0.9 (delta -0.1)
        assert not gate(before, after, adversarial_after=0).accept
        loose = gate(before, after, adversarial_after=0, tol=GateTolerances(correctness=0.2))
        correctness = next(g for g in loose.guards if g.name == "correctness")
        assert correctness.passed
        assert loose.accept


# ---------------------------------------------------------------------------
# Data quality — the gate must not silently hide input problems
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Empty cohorts and disagreeing duplicate ids must surface as warnings."""

    def test_empty_cohort_warns(self) -> None:
        """Zero overlap in completed incidents leaves correctness unevaluated."""
        before = _report(
            [_inc(1, COMPLETE, "exact"), _inc(2, COMPLETE, "exact")]
            + [_inc(3, ESCALATE, "missing"), _inc(4, ESCALATE, "missing")]
        )
        after = _report(
            [_inc(1, ESCALATE, "missing"), _inc(2, ESCALATE, "missing")]
            + [_inc(3, COMPLETE, "exact"), _inc(4, COMPLETE, "exact")]
        )
        decision = gate(before, after, adversarial_after=0)
        assert any("cohort" in w and "not evaluated" in w for w in decision.warnings)

    def test_disagreeing_duplicate_ids_warn(self) -> None:
        """Duplicate ids with differing results are surfaced, not hidden."""
        before = _report(
            [
                _inc(1, COMPLETE, "exact"),
                _inc(1, COMPLETE, "wrong"),  # same id, conflicting result
                _inc(2, COMPLETE, "exact"),
            ]
        )
        after = _report([_inc(1, COMPLETE, "exact"), _inc(2, COMPLETE, "exact")])
        decision = gate(before, after, adversarial_after=0)
        assert any("duplicate incident ids" in w for w in decision.warnings)


# ---------------------------------------------------------------------------
# Replay — the real PR #52 fixtures (the moat's self-validation)
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
_BEFORE = _REPO / "output/eval/holdout_officers_shot_20260611_160440.json"
_AFTER = _REPO / "output/eval/holdout_officers_shot_20260611_233624.json"
_HAVE_FIXTURES = _BEFORE.exists() and _AFTER.exists()
_SKIP = pytest.mark.skipif(
    not _HAVE_FIXTURES, reason="PR #52 holdout fixtures not present (gitignored)"
)


@_SKIP
class TestReplayPR52:
    """The gate must reproduce our hand-verdicts on the real PR #52 reports."""

    @staticmethod
    def _load() -> tuple[HoldoutReport, HoldoutReport]:
        return (
            HoldoutReport.model_validate_json(_BEFORE.read_text()),
            HoldoutReport.model_validate_json(_AFTER.read_text()),
        )

    def test_accepts_the_real_fix(self) -> None:
        """PR #52: completion up, cohort correctness up, fairness up, adv 0."""
        before, after = self._load()
        decision = gate(before, after, adversarial_after=0)
        assert decision.accept
        assert decision.target_delta == pytest.approx(0.57, abs=0.01)
        correctness = next(g for g in decision.guards if g.name == "correctness")
        assert correctness.delta == pytest.approx(0.239, abs=0.02)
        fairness = next(g for g in decision.guards if g.name == "fairness")
        assert fairness.passed
        # The known civilian_race watch-item surfaces as a full-set warning.
        assert any("civilian_race" in w for w in decision.warnings)

    def test_rejects_outcome_degrade_variant(self) -> None:
        """Tanking the high-coverage outcome field on the cohort -> reject."""
        before, after = self._load()
        degraded = after.model_copy(deep=True)
        for er in degraded.per_incident:
            for fr in er.field_results:
                if fr.field_name == "outcome" and fr.extracted_value is not None:
                    fr.exact_match = False
        decision = gate(before, degraded, adversarial_after=0)
        correctness = next(g for g in decision.guards if g.name == "correctness")
        assert not correctness.passed
        assert not decision.accept

    def test_fairness_degrade_does_not_veto(self) -> None:
        """Escalating an entire race group warns but does not veto (non-gating).

        Overall completion-collapse protection lives in the target guard
        (test_target_regression_vetoes); fairness alone never rejects.
        """
        before, after = self._load()
        degraded = after.model_copy(deep=True)
        for er in degraded.per_incident:
            if _race_of(er) == "black":
                er.pipeline_outcome = ESCALATE
        decision = gate(before, degraded, adversarial_after=0)
        fairness = next(g for g in decision.guards if g.name == "fairness")
        assert fairness.passed
        assert decision.accept
        assert not any("fairness" in r for r in decision.reasons)
        assert any("fairness" in w and "black" in w for w in decision.warnings)
