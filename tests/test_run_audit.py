"""Tests for src/audit/runner.py and src/audit/report.py."""

import csv
import json
from unittest.mock import patch

import pytest

from src.agents.state import ConfidenceLevel, DatasetType
from src.audit.flags import DiscrepancyFlag, FlagSeverity
from src.audit.report import (
    WORKSHEET_COLUMNS,
    IncidentAuditResult,
    build_report,
    print_report,
    save_report,
    summarize_fields,
    summarize_verified,
    write_verification_worksheet,
)
from src.audit.runner import audit_single, run_audit
from src.eval.comparators import MatchResult, PipelineOutcome


class _StubProvider:
    source_name = "stub"

    def __init__(self, record=None):
        self._record = record or {}

    def fetch(self, conn, incident_id, dataset_type):
        return dict(self._record)


def _flag(
    incident_id: int = 1,
    field: str = "civilian_name",
    severity: FlagSeverity = FlagSeverity.HIGH,
    suppressed: bool = False,
) -> DiscrepancyFlag:
    return DiscrepancyFlag(
        flag_id=f"civilians_shot_{incident_id}_{field}",
        incident_id=incident_id,
        dataset_type=DatasetType.CIVILIANS_SHOT,
        field=field,
        db_value="db",
        news_value="news",
        sources=["https://example.com"],
        source_quotes=["quote, with comma"],
        extraction_confidence=ConfidenceLevel.HIGH,
        severity=severity,
        suppressed=suppressed,
    )


def _match(field: str, error=None, exact: bool = True) -> MatchResult:
    return MatchResult(
        field_name=field,
        extracted_value="x",
        ground_truth_value="x" if error is None else None,
        exact_match=exact and error is None,
        fuzzy_match=exact and error is None,
        error=error,
    )


def _complete_run(extracted_fields=None):
    return {
        "current_stage": "complete",
        "extracted_fields": extracted_fields
        or [
            {
                "field_name": "civilian_name",
                "value": "John Smith",
                "confidence": "high",
                "sources": ["https://example.com"],
                "source_quotes": ["q"],
                "extraction_method": "llm",
            }
        ],
    }


class TestAuditSingle:
    @patch("src.run.run")
    def test_complete_run_produces_flags(self, mock_run):
        mock_run.return_value = _complete_run()
        result = audit_single(
            1031,
            DatasetType.CIVILIANS_SHOT,
            reference={"civilian_name": "Michael Brown"},
        )
        assert result.auditable is True
        assert result.pipeline_outcome == PipelineOutcome.COMPLETE
        assert len(result.flags) == 1
        mock_run.assert_called_once_with(
            "1031", "civilians_shot", settings=None
        )

    @patch("src.run.run")
    def test_escalated_run_never_flags(self, mock_run):
        mock_run.return_value = {
            "current_stage": "escalate",
            "escalation_reason": "irrelevant_sources",
            "extracted_fields": [
                {
                    "field_name": "civilian_name",
                    "value": "Wrong Person",
                    "confidence": "high",
                }
            ],
        }
        result = audit_single(
            1031,
            DatasetType.CIVILIANS_SHOT,
            reference={"civilian_name": "Michael Brown"},
        )
        assert result.auditable is False
        assert result.pipeline_outcome == PipelineOutcome.ESCALATE
        assert result.escalation_reason == "irrelevant_sources"
        assert result.flags == []
        assert result.match_results == []

    @patch("src.run.run")
    def test_suppressed_flags_split_out(self, mock_run):
        mock_run.return_value = _complete_run(
            [
                {
                    "field_name": "civilian_name",
                    "value": "John Smith",
                    "confidence": "low",
                    "sources": [],
                    "source_quotes": [],
                    "extraction_method": "llm",
                }
            ]
        )
        result = audit_single(
            1031,
            DatasetType.CIVILIANS_SHOT,
            reference={"civilian_name": "Michael Brown"},
        )
        assert result.flags == []
        assert len(result.suppressed_flags) == 1


class TestRunAudit:
    @patch("src.database.connection.get_connection")
    @patch("src.run.run")
    def test_new_run_checkpoints_and_reports(
        self, mock_run, mock_get_conn, tmp_path, mock_connection
    ):
        mock_get_conn.return_value = mock_connection
        mock_run.return_value = _complete_run()
        provider = _StubProvider({"civilian_name": "Michael Brown"})

        report = run_audit(
            DatasetType.CIVILIANS_SHOT,
            incident_ids=[7, 8],
            provider=provider,
            runs_dir=tmp_path,
        )
        assert report.n_incidents == 2
        assert report.n_auditable == 2
        assert report.n_flags == 2
        run_dir = tmp_path / report.run_id
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "incident_7.json").exists()
        assert (run_dir / "incident_8.json").exists()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["reference_source"] == "stub"

    @patch("src.database.connection.get_connection")
    @patch("src.run.run")
    def test_resume_skips_checkpointed_incidents(
        self, mock_run, mock_get_conn, tmp_path, mock_connection
    ):
        mock_get_conn.return_value = mock_connection
        mock_run.return_value = _complete_run()
        provider = _StubProvider({"civilian_name": "Michael Brown"})

        first = run_audit(
            DatasetType.CIVILIANS_SHOT,
            incident_ids=[7, 8],
            provider=provider,
            runs_dir=tmp_path,
        )
        assert mock_run.call_count == 2

        resumed = run_audit(
            DatasetType.CIVILIANS_SHOT,
            run_id=first.run_id,
            provider=provider,
            runs_dir=tmp_path,
        )
        assert mock_run.call_count == 2  # nothing re-run
        assert resumed.n_incidents == 2
        assert resumed.run_id == first.run_id

    @patch("src.database.connection.get_connection")
    @patch("src.run.run")
    def test_per_incident_error_isolation(
        self, mock_run, mock_get_conn, tmp_path, mock_connection
    ):
        mock_get_conn.return_value = mock_connection
        mock_run.side_effect = [RuntimeError("Tavily down"), _complete_run()]
        provider = _StubProvider({"civilian_name": "Michael Brown"})

        report = run_audit(
            DatasetType.CIVILIANS_SHOT,
            incident_ids=[7, 8],
            provider=provider,
            runs_dir=tmp_path,
        )
        assert report.n_incidents == 2
        assert report.n_errors == 1
        assert report.n_auditable == 1
        errored = next(r for r in report.per_incident if r.error)
        assert "Tavily down" in errored.error

    @patch("src.database.connection.get_connection")
    @patch("src.run.run")
    def test_resume_retries_errored_incident(
        self, mock_run, mock_get_conn, tmp_path, mock_connection
    ):
        # An errored incident still writes a checkpoint; resume must NOT
        # silently skip it — current design keeps the error result.
        mock_get_conn.return_value = mock_connection
        mock_run.side_effect = [RuntimeError("boom"), _complete_run()]
        provider = _StubProvider({"civilian_name": "Michael Brown"})
        first = run_audit(
            DatasetType.CIVILIANS_SHOT,
            incident_ids=[7, 8],
            provider=provider,
            runs_dir=tmp_path,
        )
        assert first.n_errors == 1
        # Deleting the errored checkpoint retries just that incident.
        (tmp_path / first.run_id / "incident_7.json").unlink()
        mock_run.side_effect = [_complete_run()]
        resumed = run_audit(
            DatasetType.CIVILIANS_SHOT,
            run_id=first.run_id,
            provider=provider,
            runs_dir=tmp_path,
        )
        assert resumed.n_errors == 0
        assert resumed.n_auditable == 2

    def test_resume_unknown_run_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_audit(
                DatasetType.CIVILIANS_SHOT,
                run_id="civilians_shot_nope",
                runs_dir=tmp_path,
            )


class TestReportAggregation:
    def _results(self):
        return [
            IncidentAuditResult(
                incident_id=1,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=True,
                pipeline_outcome=PipelineOutcome.COMPLETE,
                flags=[_flag(1, "civilian_name")],
                match_results=[
                    _match("civilian_name", exact=False),
                    _match("civilian_age"),
                ],
                elapsed_seconds=10.0,
            ),
            IncidentAuditResult(
                incident_id=2,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=True,
                pipeline_outcome=PipelineOutcome.COMPLETE,
                suppressed_flags=[_flag(2, "weapon", suppressed=True)],
                match_results=[_match("civilian_age")],
                elapsed_seconds=20.0,
            ),
            IncidentAuditResult(
                incident_id=3,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=False,
                pipeline_outcome=PipelineOutcome.ESCALATE,
                escalation_reason="irrelevant_sources",
            ),
            IncidentAuditResult(
                incident_id=4,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=False,
                error="RuntimeError: boom",
            ),
        ]

    def test_build_report_counts(self):
        report = build_report(
            "test_run", DatasetType.CIVILIANS_SHOT, self._results()
        )
        assert report.n_incidents == 4
        assert report.n_auditable == 2
        assert report.n_escalated == 1
        assert report.n_errors == 1
        assert report.n_flags == 1
        assert report.n_suppressed == 1
        assert report.mean_elapsed_seconds == 7.5
        assert report.total_elapsed_seconds == 30.0

    def test_summarize_fields_rates_and_ci(self):
        summaries = summarize_fields(self._results())
        by_field = {s.field: s for s in summaries}
        name = by_field["civilian_name"]
        assert name.n_audited == 1
        assert name.n_flags == 1
        assert name.flag_rate == 1.0
        assert name.by_severity == {"high": 1}
        age = by_field["civilian_age"]
        assert age.n_audited == 2
        assert age.n_flags == 0
        low, high = age.flag_rate_ci
        assert 0.0 <= low <= high <= 1.0

    def test_save_report_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.audit.report.AUDIT_OUTPUT_DIR", tmp_path)
        report = build_report(
            "test_run", DatasetType.CIVILIANS_SHOT, self._results()
        )
        path = save_report(report)
        data = json.loads(open(path).read())
        assert data["run_id"] == "test_run"
        assert data["n_flags"] == 1

    def test_print_report_renders(self, capsys):
        report = build_report(
            "test_run", DatasetType.CIVILIANS_SHOT, self._results()
        )
        print_report(report)
        out = capsys.readouterr().out
        assert "test_run" in out
        assert "civilian_name" in out


class TestWorksheet:
    def test_columns_order_and_suppressed_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.audit.report.AUDIT_OUTPUT_DIR", tmp_path)
        results = [
            IncidentAuditResult(
                incident_id=2,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=True,
                pipeline_outcome=PipelineOutcome.COMPLETE,
                flags=[_flag(2, "location_detail", FlagSeverity.LOW)],
                suppressed_flags=[_flag(2, "weapon", suppressed=True)],
            ),
            IncidentAuditResult(
                incident_id=1,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=True,
                pipeline_outcome=PipelineOutcome.COMPLETE,
                flags=[_flag(1, "civilian_name", FlagSeverity.HIGH)],
            ),
        ]
        report = build_report("test_run", DatasetType.CIVILIANS_SHOT, results)
        path = write_verification_worksheet(report)

        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == WORKSHEET_COLUMNS
        # High severity first, suppressed weapon flag absent
        assert rows[1][0] == "civilians_shot_1_civilian_name"
        assert rows[2][0] == "civilians_shot_2_location_detail"
        assert len(rows) == 3
        assert rows[1][-2] == "pending"

    def test_summarize_verified_precision(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.audit.report.AUDIT_OUTPUT_DIR", tmp_path)
        results = [
            IncidentAuditResult(
                incident_id=i,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=True,
                pipeline_outcome=PipelineOutcome.COMPLETE,
                flags=[_flag(i, "civilian_name")],
            )
            for i in range(1, 5)
        ]
        report = build_report("test_run", DatasetType.CIVILIANS_SHOT, results)
        path = write_verification_worksheet(report)

        # Human fills in 3 of 4 rows: 2 db errors, 1 extraction error
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        rows[1][-2] = "db_error"
        rows[2][-2] = "db_error"
        rows[3][-2] = "extraction_error"
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

        summary = summarize_verified(path)
        assert summary["n_verified"] == 3
        assert summary["precision"] == pytest.approx(2 / 3)
        low, high = summary["precision_ci"]
        assert 0.0 <= low < 2 / 3 < high <= 1.0
        assert summary["counts"]["pending"] == 1

    def test_summarize_verified_nothing_verified(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.audit.report.AUDIT_OUTPUT_DIR", tmp_path)
        results = [
            IncidentAuditResult(
                incident_id=1,
                dataset_type=DatasetType.CIVILIANS_SHOT,
                auditable=True,
                pipeline_outcome=PipelineOutcome.COMPLETE,
                flags=[_flag(1)],
            )
        ]
        report = build_report("test_run", DatasetType.CIVILIANS_SHOT, results)
        path = write_verification_worksheet(report)
        summary = summarize_verified(path)
        assert summary["precision"] is None
        assert summary["precision_ci"] is None
