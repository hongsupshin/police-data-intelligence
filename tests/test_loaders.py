#!/usr/bin/env python3
"""Unit tests for data/etl/loaders.py dataset loading functions.

This module tests the ETL workflow functions that transform CSV data into
the normalized database schema. These tests verify proper data transformation,
error handling, transaction management, and schema-driven data processing.

Test coverage includes:
- load_civilians_shot: Loading civilian shooting data
- load_officers_shot: Loading officer shooting data
- Schema integration: Verifying schema-driven approach
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data.etl.loaders import load_civilians_shot, load_officers_shot


class TestLoadCiviliansShot:
    """Test cases for the load_civilians_shot function."""

    @patch("data.etl.loaders.pd.read_csv")
    def test_loads_valid_csv(self, mock_read_csv):
        """Test successful loading of valid civilian shooting data."""
        # Mock DataFrame
        mock_df = pd.DataFrame(
            [
                {
                    "ois_report_no": "OIS-2020-001",
                    "date_incident": "2020-01-15",
                    "incident_city": "AUSTIN",
                    "civilian_name_first": "John",
                    "civilian_name_last": "Doe",
                    "civilian_age": "30",
                    "civilian_race": "White",
                    "civilian_gender": "M",
                    "civilian_died": "true",
                    "weapon_reported_by_media": "Handgun",
                    "officer_age_1": "35",
                    "officer_race_1": "Hispanic",
                    "officer_gender_1": "M",
                    "agency_name_1": "Austin PD",
                    "agency_city_1": "Austin",
                    "agency_county_1": "Travis",
                    "news_coverage_1": "http://example.com/article",
                }
            ]
        )
        mock_read_csv.return_value = mock_df

        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # Mock cursor.fetchone() to return IDs
        mock_cursor.fetchone.side_effect = [
            (1,),  # incident_id
            (10,),  # civilian_id
            (20,),  # officer_id
            (30,),  # agency_id
        ]

        # Run function
        csv_path = "/fake/path.csv"
        incidents, errors = load_civilians_shot(mock_conn, csv_path)

        # Assertions
        assert incidents == 1
        assert errors == 0
        mock_read_csv.assert_called_once_with(csv_path, low_memory=False)

        # Verify incident was created
        assert mock_cursor.execute.call_count > 0

        # Verify commit was called
        mock_conn.commit.assert_called()

    @patch("data.etl.loaders.pd.read_csv")
    def test_handles_errors_gracefully(self, mock_read_csv):
        """Test that database errors are handled gracefully with rollback."""
        # Mock DataFrame with invalid data that will cause errors
        mock_df = pd.DataFrame(
            [{"ois_report_no": "OIS-2020-001", "date_incident": "2020-01-15"}]
        )
        mock_read_csv.return_value = mock_df

        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # Make execute raise an exception
        mock_cursor.execute.side_effect = Exception("Database error")

        # Run function
        incidents, errors = load_civilians_shot(mock_conn, "/fake/path.csv")

        # Should handle error gracefully
        assert incidents == 0
        assert errors == 1
        mock_conn.rollback.assert_called()

    @patch("data.etl.loaders.apply_schema")
    @patch("data.etl.loaders.clean_entity_fields")
    @patch("data.etl.loaders.pd.read_csv")
    def test_uses_schema_driven_approach(
        self, mock_read_csv, mock_clean_entity, mock_apply_schema
    ):
        """Test that loader uses schema-driven approach for data cleaning."""
        # Mock DataFrame
        mock_df = pd.DataFrame(
            [
                {
                    "ois_report_no": "OIS-2020-001",
                    "date_incident": "2020-01-15",
                    "civilian_age": "30",
                    "civilian_race": "White",
                    "officer_age": "35",
                }
            ]
        )
        mock_read_csv.return_value = mock_df

        # Mock schema functions to return cleaned data
        mock_apply_schema.return_value = [
            "OIS-2020-001",
            "2020-01-10",
            "2020-01-15",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]  # 22 values
        # Mock returns different values for civilian vs officer
        mock_clean_entity.side_effect = [
            {  # First call: civilian entity (has name_full)
                "age": 30,
                "race": "White",
                "gender": None,
                "name_first": None,
                "name_last": None,
                "name_full": None,
            },
            {  # Second call: officer entity (no name_full)
                "age": 35,
                "race": "White",
                "gender": None,
            },
        ]

        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (1,),  # incident_id
            (10,),  # civilian_id
            (20,),  # officer_id
        ]

        # Run function
        incidents, errors = load_civilians_shot(mock_conn, "/fake/path.csv")

        # Verify schema functions were called
        assert mock_apply_schema.called
        assert mock_clean_entity.called

        # Should successfully process the row
        assert incidents == 1
        assert errors == 0

    @patch("data.etl.loaders.clean_entity_fields_with_suffix")
    @patch("data.etl.loaders.pd.read_csv")
    def test_handles_multiple_officers_with_suffix(
        self, mock_read_csv, mock_clean_with_suffix
    ):
        """Test that multiple officers are processed using suffix pattern."""
        # Mock DataFrame with multiple officers
        mock_df = pd.DataFrame(
            [
                {
                    "ois_report_no": "OIS-2020-001",
                    "officer_age_1": "35",
                    "officer_race_1": "White",
                    "officer_gender_1": "M",
                    "officer_age_2": "40",
                    "officer_race_2": "Black",
                    "officer_gender_2": "F",
                }
            ]
        )
        mock_read_csv.return_value = mock_df

        # Mock suffix cleaner
        mock_clean_with_suffix.return_value = {
            "age": 35,
            "race": "White",
            "gender": "M",
        }

        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        # Return IDs for incident, civilian, officer 1, officer 2
        mock_cursor.fetchone.side_effect = [(1,), (10,), (20,), (21,)]

        # Run function
        incidents, errors = load_civilians_shot(mock_conn, "/fake/path.csv")

        # Verify clean_entity_fields_with_suffix was called for officers
        # Should be called for each officer iteration
        assert mock_clean_with_suffix.call_count > 0


class TestLoadOfficersShot:
    """Test cases for the load_officers_shot function."""

    @patch("data.etl.loaders.pd.read_csv")
    def test_loads_valid_csv(self, mock_read_csv):
        """Test successful loading of valid officer shooting data."""
        # Mock DataFrame
        mock_df = pd.DataFrame(
            [
                {
                    "ois_report_no": "OIS-2020-002",
                    "date_incident": "2020-02-20 15:30:00",
                    "incident_city": "HOUSTON",
                    "officer_name_first": "Jane",
                    "officer_name_last": "Smith",
                    "officer_age": "40",
                    "officer_race": "White",
                    "officer_gender": "F",
                    "officer_harm": "INJURY",
                    "civilian_name_first_1": "Bob",
                    "civilian_name_last_1": "Jones",
                    "civilian_age_1": "25",
                    "civilian_race_1": "Black",
                    "civilian_gender_1": "M",
                    "agency_name_1": "Houston PD",
                    "agency_city_1": "Houston",
                    "agency_county_1": "Harris",
                    "media_link_1": "http://example.com/news",
                }
            ]
        )
        mock_read_csv.return_value = mock_df

        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.side_effect = [
            (2,),  # incident_id
            (40,),  # officer_id
            (50,),  # civilian_id
            (60,),  # agency_id
        ]

        # Run function
        incidents, errors = load_officers_shot(mock_conn, "/fake/path.csv")

        # Assertions
        assert incidents == 1
        assert errors == 0
        mock_conn.commit.assert_called()

    @patch("data.etl.loaders.apply_schema")
    @patch("data.etl.loaders.clean_entity_fields")
    @patch("data.etl.loaders.pd.read_csv")
    def test_uses_schema_for_officers_shot(
        self, mock_read_csv, mock_clean_entity, mock_apply_schema
    ):
        """Test that officers_shot loader uses schema-driven approach."""
        # Mock DataFrame
        mock_df = pd.DataFrame(
            [
                {
                    "ois_report_no": "OIS-2020-002",
                    "date_incident": "2020-02-20",
                    "officer_age": "40",
                    "officer_race": "White",
                    "officer_harm": "INJURY",
                }
            ]
        )
        mock_read_csv.return_value = mock_df

        # Mock schema functions
        mock_apply_schema.return_value = [
            "OIS-2020-002",
            None,
            "2020-02-20",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]  # 10 values for officers_shot
        mock_clean_entity.return_value = {
            "age": 40,
            "race": "White",
            "gender": None,
            "name_first": None,
            "name_last": None,
        }

        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(2,), (40,)]  # incident_id, officer_id

        # Run function
        incidents, errors = load_officers_shot(mock_conn, "/fake/path.csv")

        # Verify schema-driven approach was used
        assert mock_apply_schema.called
        assert mock_clean_entity.called
        assert incidents == 1
        assert errors == 0

    @patch("data.etl.loaders.clean_entity_fields_with_suffix")
    @patch("data.etl.loaders.pd.read_csv")
    def test_handles_multiple_civilians_in_officers_shot(
        self, mock_read_csv, mock_clean_with_suffix
    ):
        """Test that multiple civilian shooters are processed correctly."""
        # Mock DataFrame with multiple civilians
        mock_df = pd.DataFrame(
            [
                {
                    "ois_report_no": "OIS-2020-002",
                    "civilian_age_1": "25",
                    "civilian_race_1": "Black",
                    "civilian_gender_1": "M",
                    "civilian_age_2": "30",
                    "civilian_race_2": "Hispanic",
                    "civilian_gender_2": "M",
                }
            ]
        )
        mock_read_csv.return_value = mock_df

        # Mock suffix cleaner
        mock_clean_with_suffix.return_value = {
            "age": 25,
            "race": "Black",
            "gender": "M",
        }

        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        # Return IDs for incident, officer, civilian 1, civilian 2
        mock_cursor.fetchone.side_effect = [(2,), (40,), (50,), (51,)]

        # Run function
        incidents, errors = load_officers_shot(mock_conn, "/fake/path.csv")

        # Verify multiple civilians were processed
        assert mock_clean_with_suffix.call_count > 0


class TestSchemaIntegrationInLoaders:
    """Integration tests for schema usage in loader functions."""

    @patch("data.etl.loaders.pd.read_csv")
    def test_civilians_shot_processes_full_row(self, mock_read_csv):
        """Test that all fields in a realistic row are processed correctly."""
        # Create a comprehensive test row
        mock_df = pd.DataFrame(
            [
                {
                    # Incident fields
                    "ois_report_no": "OIS-2020-001",
                    "date_ag_received": "2020-01-10",
                    "date_incident": "2020-01-15",
                    "time_incident": "14:30",
                    "incident_address": "123 Main St",
                    "incident_city": "Austin",
                    "incident_county": "Travis",
                    "incident_zip": "78701",
                    "incident_result_of": "Call for service",
                    "incident_call_other": "",
                    "weapon_reported_by_media": "Handgun",
                    "weapon_reported_by_media_category": "Firearm",
                    "deadly_weapon": "true",
                    "num_officers_recorded": "2",
                    "multiple_officers_involved": "true",
                    "officer_on_duty": "true",
                    "num_reports_filed": "1",
                    "num_rows_about_this_incident": "1",
                    "cdr_narrative": "Narrative",
                    "custodial_death_report": "false",
                    "lea_narrative_published": "Published",
                    "lea_narrative_shorter": "Short",
                    # Civilian fields
                    "civilian_age": "30",
                    "civilian_race": "Hispanic",
                    "civilian_gender": "M",
                    "civilian_name_first": "John",
                    "civilian_name_last": "Doe",
                    "civilian_name_full": "John Doe",
                    "civilian_died": "DEATH",
                    # Officer fields
                    "officer_age": "35",
                    "officer_race": "White",
                    "officer_gender": "M",
                    # Agency fields
                    "agency_name_1": "Austin PD",
                    "agency_city_1": "Austin",
                    "agency_county_1": "Travis",
                    "agency_zip_1": "78701",
                    # Media
                    "news_coverage_1": "http://example.com/news",
                }
            ]
        )
        mock_read_csv.return_value = mock_df

        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        # Return IDs for incident, civilian, officer, agency
        mock_cursor.fetchone.side_effect = [(1,), (10,), (20,), (30,)]

        # Run function
        incidents, errors = load_civilians_shot(mock_conn, "/fake/path.csv")

        # Should successfully process complete row
        assert incidents == 1
        assert errors == 0

        # Verify incident INSERT was called
        calls = mock_cursor.execute.call_args_list
        assert len(calls) > 0

        # First call should be incident INSERT with 22 values
        first_call = calls[0]
        assert "INSERT INTO incidents_civilians_shot" in first_call[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
