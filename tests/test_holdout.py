"""Unit tests for src/eval/holdout.py comparison functions and models.

Tests each of the 6 comparison functions with exact match, non-match,
None/missing, parse error, fuzzy match, alias normalization, and
period bucket fallback cases. Also tests aggregate_metrics.
"""

from datetime import time

import pytest

from src.agents.state import ConfidenceLevel, DatasetType, MediaFeatureField
from src.eval.holdout import (
    EVAL_FIELDS,
    FIELD_COMPARATORS,
    EvalError,
    EvalResult,
    FieldMetrics,
    HoldoutReport,
    HoldoutSample,
    MatchResult,
    PipelineOutcome,
    _infer_stage_reached,
    aggregate_metrics,
    compare_age,
    compare_location,
    compare_outcome,
    compare_race,
    compare_time,
    compare_weapon,
    compute_fairness_metrics,
)

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    """Test data model creation and defaults."""

    def test_match_result_defaults(self) -> None:
        """Test MatchResult has sensible defaults."""
        mr = MatchResult(field_name="civilian_age")
        assert mr.exact_match is False
        assert mr.fuzzy_match is False
        assert mr.fuzzy_score is None
        assert mr.confidence is None
        assert mr.error is None

    def test_eval_result_creation(self) -> None:
        """Test EvalResult creation with required fields."""
        er = EvalResult(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            pipeline_outcome=PipelineOutcome.COMPLETE,
        )
        assert er.field_results == []
        assert er.escalation_reason is None

    def test_field_metrics_creation(self) -> None:
        """Test FieldMetrics creation."""
        fm = FieldMetrics(
            field_name="civilian_age",
            n_evaluable=10,
            n_extracted=8,
            n_exact_match=6,
            n_fuzzy_match=6,
            coverage=0.8,
            exact_accuracy=0.75,
            fuzzy_accuracy=0.75,
        )
        assert fm.confidence_breakdown == {}

    def test_holdout_report_creation(self) -> None:
        """Test HoldoutReport creation."""
        report = HoldoutReport(
            dataset_type=DatasetType.CIVILIANS_SHOT,
            n_incidents=10,
            n_completed=8,
            n_escalated=2,
            completion_rate=0.8,
        )
        assert report.field_metrics == []
        assert report.per_incident == []

    def test_eval_fields_has_six_fields(self) -> None:
        """Test EVAL_FIELDS contains exactly 6 evaluable fields."""
        assert len(EVAL_FIELDS) == 6
        assert MediaFeatureField.CIVILIAN_AGE in EVAL_FIELDS
        assert MediaFeatureField.OFFICER_NAME not in EVAL_FIELDS

    def test_field_comparators_keys_match_eval_fields(self) -> None:
        """Test FIELD_COMPARATORS keys match EVAL_FIELDS."""
        assert set(FIELD_COMPARATORS.keys()) == EVAL_FIELDS


# ---------------------------------------------------------------------------
# compare_age
# ---------------------------------------------------------------------------


class TestCompareAge:
    """Test cases for compare_age."""

    def test_exact_match(self) -> None:
        """Test matching integer values."""
        r = compare_age("25", 25, "civilian_age")
        assert r.exact_match is True
        assert r.fuzzy_match is True
        assert r.fuzzy_score is None
        assert r.error is None

    def test_non_match(self) -> None:
        """Test non-matching ages."""
        r = compare_age("30", 25, "civilian_age")
        assert r.exact_match is False
        assert r.fuzzy_match is False
        assert r.error is None

    def test_none_extracted(self) -> None:
        """Test None extracted value returns NO_EXTRACTION."""
        r = compare_age(None, 25, "civilian_age")
        assert r.error == EvalError.NO_EXTRACTION
        assert r.exact_match is False

    def test_none_ground_truth(self) -> None:
        """Test None ground truth returns NO_GROUND_TRUTH."""
        r = compare_age("25", None, "civilian_age")
        assert r.error == EvalError.NO_GROUND_TRUTH

    def test_parse_error(self) -> None:
        """Test non-numeric string returns PARSE_ERROR."""
        r = compare_age("twenty-five", 25, "civilian_age")
        assert r.error == EvalError.PARSE_ERROR

    def test_both_none(self) -> None:
        """Test both None returns NO_GROUND_TRUTH (checked first)."""
        r = compare_age(None, None, "civilian_age")
        assert r.error == EvalError.NO_GROUND_TRUTH

    def test_float_string(self) -> None:
        """Test float string returns PARSE_ERROR (int() rejects '25.0')."""
        r = compare_age("25.0", 25, "civilian_age")
        assert r.error == EvalError.PARSE_ERROR

    def test_ground_truth_value_stored(self) -> None:
        """Test ground truth value is stored as string."""
        r = compare_age("25", 25, "civilian_age")
        assert r.ground_truth_value == "25"


# ---------------------------------------------------------------------------
# compare_race
# ---------------------------------------------------------------------------


class TestCompareRace:
    """Test cases for compare_race."""

    def test_exact_match(self) -> None:
        """Test identical race strings."""
        r = compare_race("Black", "Black", "civilian_race")
        assert r.exact_match is True
        assert r.fuzzy_score is None

    def test_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        r = compare_race("black", "Black", "civilian_race")
        assert r.exact_match is True

    def test_alias_african_american(self) -> None:
        """Test 'African American' normalizes to 'black'."""
        r = compare_race("African American", "Black", "civilian_race")
        assert r.exact_match is True

    def test_alias_african_american_hyphenated(self) -> None:
        """Test 'African-American' normalizes to 'black'."""
        r = compare_race("African-American", "Black", "civilian_race")
        assert r.exact_match is True

    def test_alias_caucasian(self) -> None:
        """Test 'Caucasian' normalizes to 'white'."""
        r = compare_race("Caucasian", "White", "civilian_race")
        assert r.exact_match is True

    def test_alias_latino(self) -> None:
        """Test 'Latino' normalizes to 'hispanic'."""
        r = compare_race("Latino", "Hispanic", "civilian_race")
        assert r.exact_match is True

    def test_alias_latina(self) -> None:
        """Test 'Latina' normalizes to 'hispanic'."""
        r = compare_race("Latina", "Hispanic", "civilian_race")
        assert r.exact_match is True

    def test_non_match(self) -> None:
        """Test non-matching races."""
        r = compare_race("White", "Black", "civilian_race")
        assert r.exact_match is False

    def test_none_extracted(self) -> None:
        """Test None extracted returns NO_EXTRACTION."""
        r = compare_race(None, "Black", "civilian_race")
        assert r.error == EvalError.NO_EXTRACTION

    def test_none_ground_truth(self) -> None:
        """Test None ground truth returns NO_GROUND_TRUTH."""
        r = compare_race("Black", None, "civilian_race")
        assert r.error == EvalError.NO_GROUND_TRUTH


# ---------------------------------------------------------------------------
# compare_weapon
# ---------------------------------------------------------------------------


class TestCompareWeapon:
    """Test cases for compare_weapon using category normalization."""

    def test_exact_match(self) -> None:
        """Test identical canonical category."""
        r = compare_weapon("HANDGUN", "HANDGUN", "weapon")
        assert r.exact_match is True
        assert r.fuzzy_match is True
        assert r.fuzzy_score is None

    def test_synonym_match(self) -> None:
        """Test DB synonym maps to same category as extracted."""
        r = compare_weapon("HANDGUN", "GUN", "weapon")
        assert r.exact_match is True
        assert r.fuzzy_match is True

    def test_cross_category_no_match(self) -> None:
        """Test different categories don't match."""
        r = compare_weapon("RIFLE", "KNIFE", "weapon")
        assert r.exact_match is False
        assert r.fuzzy_match is False

    def test_none_extracted(self) -> None:
        """Test None extracted returns NO_EXTRACTION."""
        r = compare_weapon(None, "HANDGUN", "weapon")
        assert r.error == EvalError.NO_EXTRACTION
        assert r.fuzzy_score is None

    def test_none_ground_truth(self) -> None:
        """Test None ground truth returns NO_GROUND_TRUTH."""
        r = compare_weapon("HANDGUN", None, "weapon")
        assert r.error == EvalError.NO_GROUND_TRUTH

    def test_details_missing_ground_truth(self) -> None:
        """Test '(DETAILS MISSING)' ground truth returns NO_GROUND_TRUTH."""
        r = compare_weapon("HANDGUN", "(DETAILS MISSING)", "weapon")
        assert r.error == EvalError.NO_GROUND_TRUTH

    def test_unmapped_values_match_as_other(self) -> None:
        """Test unmapped values both normalize to OTHER and match."""
        r = compare_weapon("taser", "pepper spray", "weapon")
        assert r.exact_match is True


# ---------------------------------------------------------------------------
# compare_location
# ---------------------------------------------------------------------------


class TestCompareLocation:
    """Test cases for compare_location."""

    def test_exact_match(self) -> None:
        """Test identical location strings."""
        r = compare_location("123 Main St", "123 Main St", "location_detail")
        assert r.exact_match is True
        assert r.fuzzy_match is True

    def test_fuzzy_match_partial(self) -> None:
        """Test partial match on address substrings."""
        r = compare_location("Main St", "123 Main St, Austin TX", "location_detail")
        assert r.fuzzy_match is True

    def test_non_match(self) -> None:
        """Test non-matching locations."""
        r = compare_location("Elm Street", "Oak Avenue", "location_detail")
        assert r.fuzzy_match is False

    def test_none_extracted(self) -> None:
        """Test None extracted returns NO_EXTRACTION."""
        r = compare_location(None, "123 Main St", "location_detail")
        assert r.error == EvalError.NO_EXTRACTION

    def test_none_ground_truth(self) -> None:
        """Test None ground truth returns NO_GROUND_TRUTH."""
        r = compare_location("123 Main St", None, "location_detail")
        assert r.error == EvalError.NO_GROUND_TRUTH

    def test_case_insensitive(self) -> None:
        """Test case-insensitive comparison."""
        r = compare_location("main street", "Main Street", "location_detail")
        assert r.fuzzy_match is True


# ---------------------------------------------------------------------------
# compare_time
# ---------------------------------------------------------------------------


class TestCompareTime:
    """Test cases for compare_time."""

    def test_exact_hour_match(self) -> None:
        """Test exact hour match from '2:30 PM'."""
        r = compare_time("2:30 PM", time(14, 30), "time_of_day")
        assert r.exact_match is True
        assert r.fuzzy_score is None

    def test_within_tolerance(self) -> None:
        """Test match within +/-2h tolerance."""
        r = compare_time("3:00 PM", time(14, 0), "time_of_day")
        assert r.exact_match is True  # 15 vs 14, diff=1

    def test_outside_tolerance(self) -> None:
        """Test no match outside +/-2h tolerance."""
        r = compare_time("10:00 AM", time(14, 0), "time_of_day")
        assert r.exact_match is False  # 10 vs 14, diff=4

    def test_24h_format(self) -> None:
        """Test 24h time format '14:30'."""
        r = compare_time("14:30", time(14, 30), "time_of_day")
        assert r.exact_match is True

    def test_am_period(self) -> None:
        """Test AM time parsing."""
        r = compare_time("8:00 AM", time(8, 0), "time_of_day")
        assert r.exact_match is True

    def test_period_with_dots(self) -> None:
        """Test 'a.m.' and 'p.m.' format."""
        r = compare_time("2:30 p.m.", time(14, 30), "time_of_day")
        assert r.exact_match is True

    def test_bare_hour_pm(self) -> None:
        """Test bare hour with PM (e.g., '2 PM')."""
        r = compare_time("2 PM", time(14, 0), "time_of_day")
        assert r.exact_match is True

    def test_midnight_wrap(self) -> None:
        """Test midnight wrap-around tolerance."""
        r = compare_time("11:00 PM", time(1, 0), "time_of_day")
        assert r.exact_match is True  # 23 vs 1, diff=min(22,2)=2

    def test_period_bucket_morning(self) -> None:
        """Test 'morning' period bucket fallback."""
        r = compare_time("morning", time(8, 0), "time_of_day")
        assert r.exact_match is True

    def test_period_bucket_afternoon(self) -> None:
        """Test 'afternoon' period bucket."""
        r = compare_time("afternoon", time(14, 0), "time_of_day")
        assert r.exact_match is True

    def test_period_bucket_evening(self) -> None:
        """Test 'evening' period bucket."""
        r = compare_time("evening", time(20, 0), "time_of_day")
        assert r.exact_match is True

    def test_period_bucket_night(self) -> None:
        """Test 'night' period bucket (wraps midnight)."""
        r = compare_time("night", time(23, 0), "time_of_day")
        assert r.exact_match is True

    def test_period_bucket_night_early(self) -> None:
        """Test 'night' bucket covers early morning hours."""
        r = compare_time("night", time(3, 0), "time_of_day")
        assert r.exact_match is True

    def test_period_bucket_mismatch(self) -> None:
        """Test period bucket mismatch."""
        r = compare_time("morning", time(20, 0), "time_of_day")
        assert r.exact_match is False

    def test_unparseable(self) -> None:
        """Test unparseable time string returns PARSE_ERROR."""
        r = compare_time("some random text", time(14, 0), "time_of_day")
        assert r.error == EvalError.PARSE_ERROR

    def test_none_extracted(self) -> None:
        """Test None extracted returns NO_EXTRACTION."""
        r = compare_time(None, time(14, 0), "time_of_day")
        assert r.error == EvalError.NO_EXTRACTION

    def test_none_ground_truth(self) -> None:
        """Test None ground truth returns NO_GROUND_TRUTH."""
        r = compare_time("2:30 PM", None, "time_of_day")
        assert r.error == EvalError.NO_GROUND_TRUTH

    def test_12_pm_is_noon(self) -> None:
        """Test '12:00 PM' is parsed as hour 12 (noon)."""
        r = compare_time("12:00 PM", time(12, 0), "time_of_day")
        assert r.exact_match is True

    def test_12_am_is_midnight(self) -> None:
        """Test '12:00 AM' is parsed as hour 0 (midnight)."""
        r = compare_time("12:00 AM", time(0, 0), "time_of_day")
        assert r.exact_match is True


# ---------------------------------------------------------------------------
# compare_outcome
# ---------------------------------------------------------------------------


class TestCompareOutcome:
    """Test cases for compare_outcome."""

    def test_fatal_bool_true(self) -> None:
        """Test civilian_died=True matches 'killed'."""
        r = compare_outcome("killed", True, "outcome")
        assert r.exact_match is True

    def test_nonfatal_bool_false(self) -> None:
        """Test civilian_died=False matches 'wounded'."""
        r = compare_outcome("wounded", False, "outcome")
        assert r.exact_match is True

    def test_fatal_string_death(self) -> None:
        """Test officer_harm='DEATH' matches 'fatal'."""
        r = compare_outcome("fatal", "DEATH", "outcome")
        assert r.exact_match is True

    def test_nonfatal_string_injury(self) -> None:
        """Test officer_harm='INJURY' matches 'injured'."""
        r = compare_outcome("injured", "INJURY", "outcome")
        assert r.exact_match is True

    def test_died_keyword(self) -> None:
        """Test 'died' maps to fatal."""
        r = compare_outcome("died", True, "outcome")
        assert r.exact_match is True

    def test_death_keyword(self) -> None:
        """Test 'death' maps to fatal."""
        r = compare_outcome("death", True, "outcome")
        assert r.exact_match is True

    def test_fatally_keyword(self) -> None:
        """Test 'fatally shot' maps to fatal."""
        r = compare_outcome("fatally shot", True, "outcome")
        assert r.exact_match is True

    def test_survived_keyword(self) -> None:
        """Test 'survived' maps to non-fatal."""
        r = compare_outcome("survived", False, "outcome")
        assert r.exact_match is True

    def test_non_fatal_keyword(self) -> None:
        """Test 'non-fatal' maps to non-fatal."""
        r = compare_outcome("non-fatal", False, "outcome")
        assert r.exact_match is True

    def test_mismatch_fatal_vs_nonfatal(self) -> None:
        """Test mismatch: killed vs non-fatal ground truth."""
        r = compare_outcome("killed", False, "outcome")
        assert r.exact_match is False

    def test_unrecognized_extracted(self) -> None:
        """Test unrecognized outcome string returns PARSE_ERROR."""
        r = compare_outcome("unknown status", True, "outcome")
        assert r.error == EvalError.PARSE_ERROR

    def test_none_extracted(self) -> None:
        """Test None extracted returns NO_EXTRACTION."""
        r = compare_outcome(None, True, "outcome")
        assert r.error == EvalError.NO_EXTRACTION

    def test_none_ground_truth(self) -> None:
        """Test None ground truth returns NO_GROUND_TRUTH."""
        r = compare_outcome("killed", None, "outcome")
        assert r.error == EvalError.NO_GROUND_TRUTH

    def test_fuzzy_score_none(self) -> None:
        """Test outcome comparison has no fuzzy_score."""
        r = compare_outcome("killed", True, "outcome")
        assert r.fuzzy_score is None


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    """Test cases for aggregate_metrics."""

    def _make_eval_result(
        self, field_results: list[MatchResult]
    ) -> EvalResult:
        """Helper to create an EvalResult with given field results."""
        return EvalResult(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            pipeline_outcome=PipelineOutcome.COMPLETE,
            field_results=field_results,
        )

    def test_empty_results(self) -> None:
        """Test aggregate with no results."""
        metrics = aggregate_metrics([])
        assert len(metrics) == 6
        for m in metrics:
            assert m.n_evaluable == 0
            assert m.coverage == 0.0

    def test_single_exact_match(self) -> None:
        """Test aggregate with one exact match for age."""
        mr = MatchResult(
            field_name="civilian_age",
            extracted_value="25",
            ground_truth_value="25",
            exact_match=True,
            fuzzy_match=True,
            confidence=ConfidenceLevel.HIGH,
        )
        er = self._make_eval_result([mr])
        metrics = aggregate_metrics([er])

        age_metric = next(m for m in metrics if m.field_name == "civilian_age")
        assert age_metric.n_evaluable == 1
        assert age_metric.n_extracted == 1
        assert age_metric.n_exact_match == 1
        assert age_metric.coverage == 1.0
        assert age_metric.exact_accuracy == 1.0
        assert age_metric.confidence_breakdown.get("high") == 1.0

    def test_no_ground_truth_excluded(self) -> None:
        """Test NO_GROUND_TRUTH results are excluded from evaluable count."""
        mr = MatchResult(
            field_name="civilian_age",
            error=EvalError.NO_GROUND_TRUTH,
        )
        er = self._make_eval_result([mr])
        metrics = aggregate_metrics([er])

        age_metric = next(m for m in metrics if m.field_name == "civilian_age")
        assert age_metric.n_evaluable == 0

    def test_no_extraction_counted_as_evaluable(self) -> None:
        """Test NO_EXTRACTION is counted as evaluable but not extracted."""
        mr = MatchResult(
            field_name="civilian_age",
            ground_truth_value="25",
            error=EvalError.NO_EXTRACTION,
        )
        er = self._make_eval_result([mr])
        metrics = aggregate_metrics([er])

        age_metric = next(m for m in metrics if m.field_name == "civilian_age")
        assert age_metric.n_evaluable == 1
        assert age_metric.n_extracted == 0
        assert age_metric.coverage == 0.0

    def test_mixed_results(self) -> None:
        """Test aggregate with mix of match and non-match."""
        results = [
            MatchResult(
                field_name="civilian_age",
                exact_match=True,
                fuzzy_match=True,
                confidence=ConfidenceLevel.HIGH,
            ),
            MatchResult(
                field_name="civilian_age",
                exact_match=False,
                fuzzy_match=False,
                confidence=ConfidenceLevel.MEDIUM,
            ),
        ]
        evals = [self._make_eval_result([r]) for r in results]
        metrics = aggregate_metrics(evals)

        age_metric = next(m for m in metrics if m.field_name == "civilian_age")
        assert age_metric.n_extracted == 2
        assert age_metric.n_exact_match == 1
        assert age_metric.exact_accuracy == pytest.approx(0.5)
        assert age_metric.confidence_breakdown.get("high") == 1.0
        assert age_metric.confidence_breakdown.get("medium") == 0.0


# ---------------------------------------------------------------------------
# HoldoutSample
# ---------------------------------------------------------------------------


class TestHoldoutSample:
    """Test HoldoutSample model."""

    def test_creation(self) -> None:
        """Test HoldoutSample creation with required fields."""
        sample = HoldoutSample(
            incident_id=100, year=2017, race="BLACK", n_eval_fields=5
        )
        assert sample.incident_id == 100
        assert sample.year == 2017
        assert sample.race == "BLACK"
        assert sample.n_eval_fields == 5

    def test_race_optional(self) -> None:
        """Test HoldoutSample with None race."""
        sample = HoldoutSample(
            incident_id=100, year=2017, n_eval_fields=3
        )
        assert sample.race is None


# ---------------------------------------------------------------------------
# EvalResult new fields
# ---------------------------------------------------------------------------


class TestEvalResultNewFields:
    """Test new fields on EvalResult."""

    def test_defaults(self) -> None:
        """Test elapsed_seconds and stage_reached default to 0/None."""
        er = EvalResult(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            pipeline_outcome=PipelineOutcome.COMPLETE,
        )
        assert er.elapsed_seconds == 0.0
        assert er.stage_reached is None

    def test_explicit_values(self) -> None:
        """Test explicit elapsed_seconds and stage_reached."""
        er = EvalResult(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            pipeline_outcome=PipelineOutcome.ESCALATE,
            elapsed_seconds=45.3,
            stage_reached="search",
        )
        assert er.elapsed_seconds == 45.3
        assert er.stage_reached == "search"


# ---------------------------------------------------------------------------
# _infer_stage_reached
# ---------------------------------------------------------------------------


class TestInferStageReached:
    """Test _infer_stage_reached helper."""

    def test_complete(self) -> None:
        """Completed pipeline returns 'complete'."""
        assert _infer_stage_reached(PipelineOutcome.COMPLETE, None) == "complete"

    def test_max_retries(self) -> None:
        """MAX_RETRIES escalation reached 'search'."""
        assert _infer_stage_reached(PipelineOutcome.ESCALATE, "max_retries") == "search"

    def test_validation_error(self) -> None:
        """VALIDATION_ERROR escalation reached 'validate'."""
        assert (
            _infer_stage_reached(PipelineOutcome.ESCALATE, "validation_error")
            == "validate"
        )

    def test_conflict(self) -> None:
        """CONFLICT escalation reached 'merge'."""
        assert _infer_stage_reached(PipelineOutcome.ESCALATE, "conflict") == "merge"

    def test_merge_error(self) -> None:
        """MERGE_ERROR escalation reached 'merge'."""
        assert _infer_stage_reached(PipelineOutcome.ESCALATE, "merge_error") == "merge"

    def test_insufficient_sources(self) -> None:
        """INSUFFICIENT_SOURCES escalation reached 'merge'."""
        assert (
            _infer_stage_reached(PipelineOutcome.ESCALATE, "insufficient_sources")
            == "merge"
        )

    def test_extraction_error(self) -> None:
        """EXTRACTION_ERROR escalation reached 'extract'."""
        assert (
            _infer_stage_reached(PipelineOutcome.ESCALATE, "extraction_error")
            == "extract"
        )

    def test_unknown_reason(self) -> None:
        """Unknown escalation reason returns 'unknown'."""
        assert (
            _infer_stage_reached(PipelineOutcome.ESCALATE, "something_else")
            == "unknown"
        )

    def test_none_reason_escalate(self) -> None:
        """Escalation with None reason returns 'unknown'."""
        assert _infer_stage_reached(PipelineOutcome.ESCALATE, None) == "unknown"


# ---------------------------------------------------------------------------
# compute_fairness_metrics
# ---------------------------------------------------------------------------


class TestComputeFairnessMetrics:
    """Test compute_fairness_metrics."""

    def _make_eval_result(
        self,
        incident_id: int,
        outcome: PipelineOutcome,
        field_results: list[MatchResult] | None = None,
    ) -> EvalResult:
        """Helper to create EvalResult."""
        return EvalResult(
            incident_id=incident_id,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            pipeline_outcome=outcome,
            field_results=field_results or [],
        )

    def test_single_group(self) -> None:
        """Single race group computes correct metrics."""
        samples = [
            HoldoutSample(incident_id=1, year=2017, race="BLACK", n_eval_fields=5),
            HoldoutSample(incident_id=2, year=2017, race="BLACK", n_eval_fields=5),
        ]
        results = [
            self._make_eval_result(
                1,
                PipelineOutcome.COMPLETE,
                [MatchResult(field_name="civilian_age", exact_match=True)],
            ),
            self._make_eval_result(
                2,
                PipelineOutcome.ESCALATE,
                [MatchResult(field_name="civilian_age", exact_match=False)],
            ),
        ]
        metrics = compute_fairness_metrics(results, samples)
        assert "black" in metrics
        assert metrics["black"]["n"] == 2.0
        assert metrics["black"]["completion_rate"] == pytest.approx(0.5)

    def test_multiple_groups(self) -> None:
        """Multiple race groups are tracked separately."""
        samples = [
            HoldoutSample(incident_id=1, year=2017, race="BLACK", n_eval_fields=5),
            HoldoutSample(incident_id=2, year=2017, race="WHITE", n_eval_fields=5),
        ]
        results = [
            self._make_eval_result(1, PipelineOutcome.COMPLETE),
            self._make_eval_result(2, PipelineOutcome.ESCALATE),
        ]
        metrics = compute_fairness_metrics(results, samples)
        assert len(metrics) == 2
        assert metrics["black"]["completion_rate"] == 1.0
        assert metrics["white"]["completion_rate"] == 0.0

    def test_none_race_becomes_unknown(self) -> None:
        """None race is grouped as 'unknown'."""
        samples = [
            HoldoutSample(incident_id=1, year=2017, race=None, n_eval_fields=3),
        ]
        results = [self._make_eval_result(1, PipelineOutcome.COMPLETE)]
        metrics = compute_fairness_metrics(results, samples)
        assert "unknown" in metrics


# ---------------------------------------------------------------------------
# HoldoutReport new fields
# ---------------------------------------------------------------------------


class TestHoldoutReportNewFields:
    """Test new fields on HoldoutReport."""

    def test_defaults(self) -> None:
        """Test new fields default to empty/zero."""
        report = HoldoutReport(
            dataset_type=DatasetType.CIVILIANS_SHOT,
            n_incidents=10,
            n_completed=8,
            n_escalated=2,
            completion_rate=0.8,
        )
        assert report.samples == []
        assert report.mean_elapsed_seconds == 0.0
        assert report.total_elapsed_seconds == 0.0
        assert report.fairness_metrics == {}
