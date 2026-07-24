"""Tests for src/audit/preflight.py."""

import pytest

from src.audit.preflight import EXPECTED_ROW_COUNTS, verify_db_clean


class TestVerifyDbClean:
    def test_passes_on_expected_counts(self, mock_connection, mock_cursor):
        mock_cursor.fetchone.side_effect = [(1674,), (282,)]
        verify_db_clean(mock_connection)

    def test_raises_on_doubled_civilians(self, mock_connection, mock_cursor):
        mock_cursor.fetchone.side_effect = [(3348,), (282,)]
        with pytest.raises(RuntimeError, match="3348 rows, expected 1674"):
            verify_db_clean(mock_connection)

    def test_raises_on_wrong_officers(self, mock_connection, mock_cursor):
        mock_cursor.fetchone.side_effect = [(1674,), (564,)]
        with pytest.raises(RuntimeError, match="564 rows, expected 282"):
            verify_db_clean(mock_connection)

    def test_expected_counts_match_readme(self):
        assert sum(EXPECTED_ROW_COUNTS.values()) == 1956
