"""Tests for src/audit/offline.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.state import DatasetType
from src.audit.offline import (
    audit_saved_outputs,
    iter_saved_complete,
    load_saved_enrichment,
    main,
    summarize_audits,
)


class _StubProvider:
    """Reference provider returning a canned record (no DB)."""

    source_name = "stub"

    def __init__(self, record: dict[str, object]):
        self._record = record
        self.fetched: list[int] = []

    def fetch(self, conn, incident_id, dataset_type):
        self.fetched.append(incident_id)
        if isinstance(self._record, Exception):
            raise self._record
        return dict(self._record)


def _write_artifact(
    directory: Path,
    incident_id: int,
    dataset: str = "civilians_shot",
    suffix: str = "complete",
    extracted_fields: list[dict] | None = None,
) -> Path:
    """Write a minimal saved-enrichment artifact (older schema, no newer keys)."""
    if extracted_fields is None:
        extracted_fields = [
            {
                "field_name": "civilian_name",
                "value": "John Smith",
                "confidence": "high",
                "sources": ["https://example.com"],
                "source_quotes": ["q"],
                "extraction_method": "llm",
            }
        ]
    path = directory / f"{dataset}_{incident_id}_{suffix}.json"
    path.write_text(
        json.dumps(
            {
                "incident_id": str(incident_id),
                "dataset_type": dataset,
                "extracted_fields": extracted_fields,
                "outcome_summary": "ok",
            }
        )
    )
    return path


class TestLoadSavedEnrichment:
    def test_parses_minimal_artifact(self, tmp_path):
        path = _write_artifact(tmp_path, 1031)
        incident_id, dataset_type, fields = load_saved_enrichment(path)
        assert incident_id == 1031
        assert dataset_type == DatasetType.CIVILIANS_SHOT
        assert fields[0]["field_name"] == "civilian_name"

    def test_missing_key_raises(self, tmp_path):
        path = tmp_path / "civilians_shot_1_complete.json"
        path.write_text(json.dumps({"incident_id": "1"}))
        with pytest.raises(KeyError):
            load_saved_enrichment(path)


class TestIterSavedComplete:
    def test_selects_only_complete_for_dataset(self, tmp_path):
        _write_artifact(tmp_path, 1, "civilians_shot", "complete")
        _write_artifact(tmp_path, 1, "civilians_shot", "escalate")
        _write_artifact(tmp_path, 2, "officers_shot", "complete")
        paths = iter_saved_complete(tmp_path, DatasetType.CIVILIANS_SHOT)
        assert [p.name for p in paths] == ["civilians_shot_1_complete.json"]


class TestAuditSavedOutputs:
    def test_flags_from_saved_artifacts(self, tmp_path, mock_connection):
        _write_artifact(tmp_path, 1031)
        provider = _StubProvider({"civilian_name": "Michael Brown"})
        audits = audit_saved_outputs(
            tmp_path, DatasetType.CIVILIANS_SHOT, mock_connection, provider
        )
        assert len(audits) == 1
        assert audits[0].reference_source == "stub"
        assert len(audits[0].flags) == 1
        assert provider.fetched == [1031]

    def test_agreeing_artifact_yields_no_flags(self, tmp_path, mock_connection):
        _write_artifact(tmp_path, 1031)
        provider = _StubProvider({"civilian_name": "John Smith"})
        audits = audit_saved_outputs(
            tmp_path, DatasetType.CIVILIANS_SHOT, mock_connection, provider
        )
        assert len(audits) == 1
        assert audits[0].flags == []

    def test_unknown_incident_skipped(self, tmp_path, mock_connection, caplog):
        _write_artifact(tmp_path, 99999)
        provider = _StubProvider(KeyError("Incident 99999 not found"))
        audits = audit_saved_outputs(
            tmp_path, DatasetType.CIVILIANS_SHOT, mock_connection, provider
        )
        assert audits == []

    def test_malformed_artifact_skipped(self, tmp_path, mock_connection):
        bad = tmp_path / "civilians_shot_7_complete.json"
        bad.write_text("{not json")
        _write_artifact(tmp_path, 1031)
        provider = _StubProvider({"civilian_name": "John Smith"})
        audits = audit_saved_outputs(
            tmp_path, DatasetType.CIVILIANS_SHOT, mock_connection, provider
        )
        assert [a.incident_id for a in audits] == [1031]

    def test_escalated_artifacts_never_audited(self, tmp_path, mock_connection):
        _write_artifact(tmp_path, 1, suffix="escalate")
        provider = _StubProvider({"civilian_name": "Michael Brown"})
        audits = audit_saved_outputs(
            tmp_path, DatasetType.CIVILIANS_SHOT, mock_connection, provider
        )
        assert audits == []


class TestSummarizeAudits:
    def test_counts_reportable_and_suppressed(self, tmp_path, mock_connection):
        _write_artifact(tmp_path, 1)
        _write_artifact(
            tmp_path,
            2,
            extracted_fields=[
                {
                    "field_name": "civilian_name",
                    "value": "John Smith",
                    "confidence": "low",
                    "sources": [],
                    "source_quotes": [],
                    "extraction_method": "llm",
                }
            ],
        )
        provider = _StubProvider({"civilian_name": "Michael Brown"})
        audits = audit_saved_outputs(
            tmp_path, DatasetType.CIVILIANS_SHOT, mock_connection, provider
        )
        summary = summarize_audits(audits)
        assert "Incidents audited: 2" in summary
        assert "1 reportable" in summary
        assert "1 suppressed" in summary
        assert "high=1" in summary
        assert "civilian_name=1" in summary

    def test_empty_audits(self):
        summary = summarize_audits([])
        assert "Incidents audited: 0" in summary
        assert "none" in summary


class TestMain:
    def test_prints_summary_and_reportable_flags(
        self, tmp_path, mock_connection, capsys
    ):
        _write_artifact(tmp_path, 1031)
        provider = _StubProvider({"civilian_name": "Michael Brown"})
        argv = ["offline.py", "civilians_shot", "--directory", str(tmp_path)]
        with (
            patch("sys.argv", argv),
            patch(
                "src.database.connection.get_connection",
                return_value=mock_connection,
            ),
            patch(
                "src.audit.offline.TjiDbReferenceProvider",
                return_value=provider,
            ),
        ):
            main()
        out = capsys.readouterr().out
        assert "Incidents audited: 1" in out
        assert "civilians_shot_1031_civilian_name" in out
        mock_connection.close.assert_called_once()
