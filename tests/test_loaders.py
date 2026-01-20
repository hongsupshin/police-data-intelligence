#!/usr/bin/env python3
"""Unit tests for data/etl/loaders.py dataset loading functions.

This module tests the ETL workflow functions that transform CSV data into
the normalized database schema. These tests verify proper data transformation,
error handling, and transaction management.

Test coverage includes:
- load_civilians_shot: Loading civilian shooting data
- load_officers_shot: Loading officer shooting data
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
