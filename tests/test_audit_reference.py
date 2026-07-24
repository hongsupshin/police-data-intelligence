"""Tests for src/audit/reference.py."""

from unittest.mock import patch

import pytest

from src.agents.state import DatasetType
from src.audit.reference import TJI_DB_SOURCE, TjiDbReferenceProvider


class TestTjiDbReferenceProvider:
    def test_source_name(self):
        assert TjiDbReferenceProvider().source_name == TJI_DB_SOURCE

    @patch("src.audit.reference.fetch_incident")
    @patch("src.audit.reference.fetch_ground_truth")
    def test_composes_ground_truth_and_names(
        self, mock_gt, mock_incident, mock_connection
    ):
        mock_gt.return_value = {
            "civilian_age": 34,
            "civilian_race": "HISPANIC",
            "weapon": "handgun",
            "location_detail": "Houston",
            "time_of_day": None,
            "outcome": True,
        }
        mock_incident.return_value = {
            "officer_name": "John Duran",
            "civilian_name": "Jose Ibarra-Ruiz",
            "incident_date": None,
            "location": "Houston",
        }
        record = TjiDbReferenceProvider().fetch(
            mock_connection, 1031, DatasetType.CIVILIANS_SHOT
        )
        assert record["officer_name"] == "John Duran"
        assert record["civilian_name"] == "Jose Ibarra-Ruiz"
        assert record["civilian_age"] == 34
        assert record["outcome"] is True
        mock_gt.assert_called_once_with(
            mock_connection, 1031, DatasetType.CIVILIANS_SHOT
        )
        mock_incident.assert_called_once_with(
            mock_connection, 1031, DatasetType.CIVILIANS_SHOT
        )

    @patch("src.audit.reference.fetch_incident")
    @patch("src.audit.reference.fetch_ground_truth")
    def test_missing_names_map_to_none(self, mock_gt, mock_incident, mock_connection):
        mock_gt.return_value = {"civilian_age": None}
        mock_incident.return_value = {}
        record = TjiDbReferenceProvider().fetch(
            mock_connection, 5, DatasetType.OFFICERS_SHOT
        )
        assert record["officer_name"] is None
        assert record["civilian_name"] is None

    @patch(
        "src.audit.reference.fetch_ground_truth",
        side_effect=KeyError("Incident 99999 not found in civilians_shot"),
    )
    def test_unknown_incident_raises(self, mock_gt, mock_connection):
        with pytest.raises(KeyError, match="not found"):
            TjiDbReferenceProvider().fetch(
                mock_connection, 99999, DatasetType.CIVILIANS_SHOT
            )
