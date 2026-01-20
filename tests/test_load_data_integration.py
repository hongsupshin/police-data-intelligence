#!/usr/bin/env python3
"""Integration tests for data/load_data.py main orchestration function.

This module tests the end-to-end ETL workflow including database connection,
schema loading, data loading, and summary statistics. These tests verify that
all components work together correctly.

Test coverage includes:
- main() orchestration function
- End-to-end workflow with mock database
"""

from unittest.mock import Mock, patch

import pytest

from data.load_data import main


class TestMain:
    """Test cases for the main orchestration function."""

    @patch("data.load_data.psycopg2.connect")
    @patch("data.load_data.load_civilians_shot")
    @patch("data.load_data.load_officers_shot")
    @patch("data.load_data.Path.exists")
    @patch("builtins.open", create=True)
    def test_main_happy_path(
        self,
        mock_open,
        mock_exists,
        mock_load_officers,
        mock_load_civilians,
        mock_connect,
    ):
        """Test the happy path for the main orchestration function."""
        # Mock file existence checks
        mock_exists.return_value = True

        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock row counts
        mock_cursor.fetchone.side_effect = [
            (1674,),  # incidents_civilians_shot
            (282,),  # incidents_officers_shot
            (500,),  # officers
            (600,),  # civilians
            (100,),  # agencies
            (2000,),  # officer links
            (300,),  # civilian links
        ]

        # Mock loading functions
        mock_load_civilians.return_value = (1674, 0)
        mock_load_officers.return_value = (282, 0)

        # Run main function
        main()

        # Verify connections were made
        mock_connect.assert_called_once()

        # Verify loaders were called
        mock_load_civilians.assert_called_once()
        mock_load_officers.assert_called_once()

        # Verify summary statistics were queried
        assert mock_cursor.execute.call_count >= 7  # Schema + 7 count queries


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests that require a real database connection.

    These tests verify end-to-end functionality with an actual PostgreSQL
    database. They are marked with the 'integration' pytest marker and can
    be skipped in CI/CD environments.

    Run with: pytest -m integration
    """

    def test_placeholder(self):
        """Placeholder for future integration tests.

        Integration tests require a test database setup. This test is skipped
        by default to avoid database dependencies in CI/CD environments.
        """
        pytest.skip("Integration tests require test database setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
