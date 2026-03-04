"""Unit tests for data/backfill_civilian_died.py migration script."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.backfill_civilian_died import (
    backfill_by_composite_key,
    backfill_by_report_no,
    diagnose,
)


@pytest.fixture()
def mock_conn() -> MagicMock:
    """Create a mock database connection with cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn


class TestDiagnose:
    """Test cases for the diagnose function."""

    def test_prints_stats(self, mock_conn: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify diagnose prints correct stats."""
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = (1674, 1674, 963, 711, 0)

        result = diagnose(mock_conn)

        assert result == {
            "total": 1674,
            "non_null": 1674,
            "true_count": 963,
            "false_count": 711,
            "null_count": 0,
        }
        output = capsys.readouterr().out
        assert "1674" in output
        assert "963" in output
        assert "711" in output

    def test_all_null(self, mock_conn: MagicMock) -> None:
        """Verify diagnose handles all-NULL case."""
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = (1674, 0, 0, 0, 1674)

        result = diagnose(mock_conn)

        assert result["null_count"] == 1674
        assert result["non_null"] == 0


class TestBackfillByReportNo:
    """Test cases for backfill_by_report_no."""

    def test_updates_death_rows(self, mock_conn: MagicMock) -> None:
        """Verify DEATH maps to TRUE via ois_report_no match."""
        cursor = mock_conn.cursor.return_value
        cursor.rowcount = 1

        df = pd.DataFrame({
            "ois_report_no": ["OIS-001"],
            "civilian_died": ["DEATH"],
        })

        updated = backfill_by_report_no(mock_conn, df)

        assert updated == 1
        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args[0]
        assert "ois_report_no = %s" in sql
        assert "civilian_died IS NULL" in sql
        assert params == (True, "OIS-001")

    def test_updates_injury_rows(self, mock_conn: MagicMock) -> None:
        """Verify INJURY maps to FALSE via ois_report_no match."""
        cursor = mock_conn.cursor.return_value
        cursor.rowcount = 1

        df = pd.DataFrame({
            "ois_report_no": ["OIS-002"],
            "civilian_died": ["INJURY"],
        })

        updated = backfill_by_report_no(mock_conn, df)

        assert updated == 1
        _, params = cursor.execute.call_args[0]
        assert params == (False, "OIS-002")

    def test_skips_null_report_no(self, mock_conn: MagicMock) -> None:
        """Verify rows without ois_report_no are skipped."""
        cursor = mock_conn.cursor.return_value

        df = pd.DataFrame({
            "ois_report_no": [None, pd.NA],
            "civilian_died": ["DEATH", "INJURY"],
        })

        updated = backfill_by_report_no(mock_conn, df)

        assert updated == 0
        cursor.execute.assert_not_called()

    def test_multiple_rows(self, mock_conn: MagicMock) -> None:
        """Verify multiple rows are processed correctly."""
        cursor = mock_conn.cursor.return_value
        cursor.rowcount = 1

        df = pd.DataFrame({
            "ois_report_no": ["OIS-001", "OIS-002", "OIS-003"],
            "civilian_died": ["DEATH", "INJURY", "DEATH"],
        })

        updated = backfill_by_report_no(mock_conn, df)

        assert updated == 3
        assert cursor.execute.call_count == 3


class TestBackfillByCompositeKey:
    """Test cases for backfill_by_composite_key."""

    def test_uses_composite_key(self, mock_conn: MagicMock) -> None:
        """Verify composite key match uses IS NOT DISTINCT FROM."""
        cursor = mock_conn.cursor.return_value
        cursor.rowcount = 1

        df = pd.DataFrame({
            "ois_report_no": [None],
            "civilian_died": ["DEATH"],
            "date_incident": ["2015-09-02"],
            "incident_city": ["HOUSTON"],
            "civilian_name_full": ["JOHN DOE"],
            "civilian_age": [30.0],
            "civilian_race": ["BLACK"],
        })

        updated = backfill_by_composite_key(mock_conn, df)

        assert updated == 1
        sql, params = cursor.execute.call_args[0]
        assert "IS NOT DISTINCT FROM" in sql
        assert "civilian_died IS NULL" in sql
        assert params[0] is True  # DEATH -> True
        assert str(params[1]) == "2015-09-02"
        assert params[2] == "HOUSTON"
        assert params[3] == "JOHN DOE"
        assert params[4] == 30
        assert params[5] == "BLACK"

    def test_handles_null_fields(self, mock_conn: MagicMock) -> None:
        """Verify NULL fields are passed through for IS NOT DISTINCT FROM."""
        cursor = mock_conn.cursor.return_value
        cursor.rowcount = 1

        df = pd.DataFrame({
            "ois_report_no": [None],
            "civilian_died": ["INJURY"],
            "date_incident": ["2015-09-02"],
            "incident_city": [None],
            "civilian_name_full": [None],
            "civilian_age": [None],
            "civilian_race": [None],
        })

        updated = backfill_by_composite_key(mock_conn, df)

        assert updated == 1
        _, params = cursor.execute.call_args[0]
        assert params[0] is False  # INJURY -> False
        assert params[2] is None  # incident_city
        assert params[3] is None  # name_full
        assert params[4] is None  # age
        assert params[5] is None  # race

    def test_skips_rows_with_report_no(self, mock_conn: MagicMock) -> None:
        """Verify rows with ois_report_no are skipped (handled by pass 1)."""
        cursor = mock_conn.cursor.return_value

        df = pd.DataFrame({
            "ois_report_no": ["OIS-001"],
            "civilian_died": ["DEATH"],
            "date_incident": ["2015-09-02"],
            "incident_city": ["HOUSTON"],
            "civilian_name_full": ["JOHN DOE"],
            "civilian_age": [30.0],
            "civilian_race": ["BLACK"],
        })

        updated = backfill_by_composite_key(mock_conn, df)

        assert updated == 0
        cursor.execute.assert_not_called()


class TestIdempotency:
    """Test that re-running the backfill updates 0 rows."""

    def test_rerun_updates_zero(self, mock_conn: MagicMock) -> None:
        """Verify idempotency via civilian_died IS NULL guard."""
        cursor = mock_conn.cursor.return_value
        # Simulate already-updated rows (WHERE civilian_died IS NULL matches nothing)
        cursor.rowcount = 0

        df = pd.DataFrame({
            "ois_report_no": ["OIS-001"],
            "civilian_died": ["DEATH"],
        })

        updated = backfill_by_report_no(mock_conn, df)

        assert updated == 0


class TestDryRun:
    """Test --dry-run flag behavior."""

    @patch("data.backfill_civilian_died.psycopg2")
    @patch("data.backfill_civilian_died.pd")
    def test_dry_run_rolls_back(
        self, mock_pd: MagicMock, mock_psycopg2: MagicMock
    ) -> None:
        """Verify --dry-run calls rollback, not commit."""
        from data.backfill_civilian_died import main

        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        cursor = MagicMock()
        mock_conn.cursor.return_value = cursor
        cursor.fetchone.return_value = (0, 0, 0, 0, 0)
        cursor.rowcount = 0

        mock_df = MagicMock()
        mock_df.__len__ = lambda self: 0
        mock_df.__getitem__ = lambda self, key: MagicMock(
            value_counts=lambda: MagicMock(to_dict=lambda: {})
        )
        mock_pd.read_csv.return_value = mock_df

        with patch("sys.argv", ["backfill", "--dry-run"]):
            main()

        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()

    @patch("data.backfill_civilian_died.psycopg2")
    @patch("data.backfill_civilian_died.pd")
    def test_no_flag_commits(
        self, mock_pd: MagicMock, mock_psycopg2: MagicMock
    ) -> None:
        """Verify no --dry-run calls commit, not rollback."""
        from data.backfill_civilian_died import main

        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        cursor = MagicMock()
        mock_conn.cursor.return_value = cursor
        cursor.fetchone.return_value = (0, 0, 0, 0, 0)
        cursor.rowcount = 0

        mock_df = MagicMock()
        mock_df.__len__ = lambda self: 0
        mock_df.__getitem__ = lambda self, key: MagicMock(
            value_counts=lambda: MagicMock(to_dict=lambda: {})
        )
        mock_pd.read_csv.return_value = mock_df

        with patch("sys.argv", ["backfill"]):
            main()

        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
