"""Tests for src/audit/specificity.py."""

import json

from src.agents.state import DatasetType
from src.audit.flags import DiscrepancyFlag, FlagSeverity
from src.audit.report import IncidentAuditResult
from src.audit.specificity import (
    build_presumed_correct_set,
    print_specificity,
    specificity_from_results,
)
from src.eval.comparators import PipelineOutcome


def _field_result(exact: bool = True, error: str | None = None) -> dict:
    return {
        "field_name": "civilian_age",
        "extracted_value": "30",
        "ground_truth_value": "30" if exact else "31",
        "exact_match": exact and error is None,
        "fuzzy_match": exact and error is None,
        "error": error,
    }


def _report(
    per_incident: list[dict], dataset: str = "civilians_shot"
) -> dict:
    return {"dataset_type": dataset, "per_incident": per_incident}


def _incident(
    incident_id: int,
    field_results: list[dict],
    outcome: str = "complete",
) -> dict:
    return {
        "incident_id": incident_id,
        "pipeline_outcome": outcome,
        "field_results": field_results,
    }


def _write(tmp_path, name: str, report: dict):
    path = tmp_path / name
    path.write_text(json.dumps(report))
    return path


class TestBuildPresumedCorrectSet:
    def test_all_exact_qualifies(self, tmp_path):
        report = _report([_incident(500, [_field_result()] * 3)])
        path = _write(tmp_path, "holdout_civilians_shot_1.json", report)
        ids = build_presumed_correct_set(
            [path], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
        )
        assert ids == [500]

    def test_any_inexact_field_disqualifies(self, tmp_path):
        report = _report(
            [_incident(500, [_field_result(), _field_result(exact=False)] * 2)]
        )
        path = _write(tmp_path, "holdout_civilians_shot_1.json", report)
        assert (
            build_presumed_correct_set(
                [path], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
            )
            == []
        )

    def test_min_exact_fields_required(self, tmp_path):
        report = _report([_incident(500, [_field_result()] * 2)])
        path = _write(tmp_path, "holdout_civilians_shot_1.json", report)
        assert (
            build_presumed_correct_set(
                [path], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
            )
            == []
        )
        assert build_presumed_correct_set(
            [path],
            DatasetType.CIVILIANS_SHOT,
            min_exact_fields=2,
            exclude_ids=set(),
        ) == [500]

    def test_errored_fields_do_not_count(self, tmp_path):
        # NO_EXTRACTION / NO_GROUND_TRUTH fields are not evidence either way.
        results = [_field_result()] * 3 + [
            _field_result(error="no_extraction"),
            _field_result(error="no_ground_truth"),
        ]
        report = _report([_incident(500, results)])
        path = _write(tmp_path, "holdout_civilians_shot_1.json", report)
        assert build_presumed_correct_set(
            [path], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
        ) == [500]

    def test_escalated_incidents_ignored(self, tmp_path):
        report = _report(
            [_incident(500, [_field_result()] * 3, outcome="escalate")]
        )
        path = _write(tmp_path, "holdout_civilians_shot_1.json", report)
        assert (
            build_presumed_correct_set(
                [path], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
            )
            == []
        )

    def test_disqualification_is_unanimous_across_reports(self, tmp_path):
        good = _write(
            tmp_path,
            "holdout_civilians_shot_1.json",
            _report([_incident(500, [_field_result()] * 3)]),
        )
        bad = _write(
            tmp_path,
            "holdout_civilians_shot_2.json",
            _report([_incident(500, [_field_result(exact=False)] * 3)]),
        )
        assert (
            build_presumed_correct_set(
                [good, bad], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
            )
            == []
        )

    def test_other_dataset_reports_skipped(self, tmp_path):
        report = _report(
            [_incident(500, [_field_result()] * 3)], dataset="officers_shot"
        )
        path = _write(tmp_path, "holdout_officers_shot_1.json", report)
        assert (
            build_presumed_correct_set(
                [path], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
            )
            == []
        )

    def test_dev_and_test_ids_excluded_by_default(self, tmp_path):
        from src.eval.holdout import TEST_SET_IDS

        test_id = sorted(TEST_SET_IDS[DatasetType.CIVILIANS_SHOT])[0]
        report = _report(
            [
                _incident(test_id, [_field_result()] * 3),
                _incident(500, [_field_result()] * 3),
            ]
        )
        path = _write(tmp_path, "holdout_civilians_shot_1.json", report)
        assert build_presumed_correct_set(
            [path], DatasetType.CIVILIANS_SHOT
        ) == [500]

    def test_unreadable_report_skipped(self, tmp_path):
        bad = tmp_path / "holdout_civilians_shot_bad.json"
        bad.write_text("{not json")
        good = _write(
            tmp_path,
            "holdout_civilians_shot_1.json",
            _report([_incident(500, [_field_result()] * 3)]),
        )
        assert build_presumed_correct_set(
            [bad, good], DatasetType.CIVILIANS_SHOT, exclude_ids=set()
        ) == [500]


def _audit_result(
    incident_id: int, n_flags: int = 0, auditable: bool = True
) -> IncidentAuditResult:
    flags = [
        DiscrepancyFlag(
            flag_id=f"civilians_shot_{incident_id}_weapon_{i}",
            incident_id=incident_id,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            field="weapon",
            db_value="a",
            news_value="b",
            extraction_confidence="high",
            severity=FlagSeverity.MEDIUM,
        )
        for i in range(n_flags)
    ]
    return IncidentAuditResult(
        incident_id=incident_id,
        dataset_type=DatasetType.CIVILIANS_SHOT,
        auditable=auditable,
        pipeline_outcome=PipelineOutcome.COMPLETE if auditable else None,
        flags=flags,
    )


class TestSpecificityFromResults:
    def test_rate_and_ci(self):
        results = [
            _audit_result(1, n_flags=0),
            _audit_result(2, n_flags=2),
            _audit_result(3, n_flags=0),
            _audit_result(4, n_flags=1),
            _audit_result(99, n_flags=5),  # not in presumed set — ignored
        ]
        summary = specificity_from_results(results, {1, 2, 3, 4})
        assert summary["n_presumed_correct"] == 4
        assert summary["n_false_flagged"] == 2
        assert summary["n_false_flags"] == 3
        assert summary["false_flag_rate"] == 0.5
        assert summary["specificity"] == 0.5
        low, high = summary["false_flag_rate_ci"]
        assert 0.0 <= low < 0.5 < high <= 1.0

    def test_non_auditable_excluded(self):
        results = [_audit_result(1, auditable=False)]
        summary = specificity_from_results(results, {1})
        assert summary["n_presumed_correct"] == 0
        assert summary["false_flag_rate"] is None
        assert summary["specificity"] is None

    def test_print_specificity_renders(self, capsys):
        summary = specificity_from_results([_audit_result(1)], {1})
        print_specificity(summary)
        out = capsys.readouterr().out
        assert "Specificity: 100.0%" in out

    def test_print_specificity_empty(self, capsys):
        print_specificity(specificity_from_results([], set()))
        out = capsys.readouterr().out
        assert "No auditable presumed-correct incidents" in out
