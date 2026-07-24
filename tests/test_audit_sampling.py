"""Tests for src/audit/sampling.py and the shared stratum allocator."""

from src.agents.state import DatasetType
from src.audit.sampling import AuditSample, select_audit_incidents
from src.eval.holdout import DEV_SET_IDS, TEST_SET_IDS, allocate_by_stratum


class TestAllocateByStratum:
    def test_proportional_allocation(self):
        rows = {
            2016: [("a", 2016)] * 60,
            2017: [("b", 2017)] * 40,
        }
        selected = allocate_by_stratum(rows, 10)
        assert len(selected) == 10

    def test_min_per_stratum_protects_sparse_years(self):
        # Sparse 2015 would get round(4/104 * 20) = 1 proportionally; the
        # minimum lifts it to 4. (Strata fill in sorted order, so the
        # protection applies before the limit is exhausted.)
        rows = {
            2015: [("a", 2015)] * 4,
            2016: [("b", 2016)] * 100,
        }
        selected = allocate_by_stratum(rows, 20)
        n_2015 = sum(1 for r in selected if r[1] == 2015)
        assert n_2015 == 4

    def test_respects_limit(self):
        rows = {y: [(f"r{y}", y)] * 50 for y in (2016, 2017, 2018)}
        assert len(allocate_by_stratum(rows, 12)) == 12

    def test_stratum_smaller_than_minimum(self):
        rows = {2016: [("a", 2016)] * 2}
        assert len(allocate_by_stratum(rows, 10)) == 2

    def test_empty_input(self):
        assert allocate_by_stratum({}, 10) == []


class TestSelectAuditIncidents:
    def _rows(self):
        # (incident_id, year, race) rows as the query would return them
        return [
            (10, 2016, "WHITE"),
            (11, 2016, "HISPANIC"),
            (12, 2017, "BLACK"),
            (13, 2017, None),
        ]

    def test_returns_samples(self, mock_connection, mock_cursor):
        mock_cursor.fetchall.return_value = self._rows()
        samples = select_audit_incidents(
            mock_connection, DatasetType.CIVILIANS_SHOT, limit=4
        )
        assert all(isinstance(s, AuditSample) for s in samples)
        assert {s.incident_id for s in samples} == {10, 11, 12, 13}
        assert samples[0].year == 2016

    def test_civilians_query_requires_completeness(
        self, mock_connection, mock_cursor
    ):
        mock_cursor.fetchall.return_value = []
        select_audit_incidents(mock_connection, DatasetType.CIVILIANS_SHOT, limit=5)
        query = mock_cursor.execute.call_args[0][0]
        for predicate in (
            "c.age IS NOT NULL",
            "c.race IS NOT NULL",
            "c.name_first IS NOT NULL",
            "c.name_last IS NOT NULL",
            "i.weapon_reported_by_media IS NOT NULL",
            "i.time_incident IS NOT NULL",
            "v.civilian_died IS NOT NULL",
        ):
            assert predicate in query
        # Officer names are structurally absent from civilians_shot
        # (0/1,674) — requiring them would empty the pool.
        assert "o.name_first" not in query

    def test_civilians_query_requires_single_victim(
        self, mock_connection, mock_cursor
    ):
        mock_cursor.fetchall.return_value = []
        select_audit_incidents(mock_connection, DatasetType.CIVILIANS_SHOT, limit=5)
        query = mock_cursor.execute.call_args[0][0]
        assert "SELECT COUNT(*) FROM incident_civilians_shot_victims" in query
        assert "= 1" in query

    def test_officers_query_requires_single_victim_and_names(
        self, mock_connection, mock_cursor
    ):
        mock_cursor.fetchall.return_value = []
        select_audit_incidents(mock_connection, DatasetType.OFFICERS_SHOT, limit=5)
        query = mock_cursor.execute.call_args[0][0]
        assert "SELECT COUNT(*) FROM incident_officers_shot_victims" in query
        assert "o.name_first IS NOT NULL" in query
        assert "c.name_first IS NOT NULL" in query
        # Structural: officers_shot has no weapon/time columns
        assert "weapon_reported_by_media" not in query
        assert "time_incident" not in query

    def test_dev_and_test_ids_excluded_by_default(
        self, mock_connection, mock_cursor
    ):
        mock_cursor.fetchall.return_value = []
        select_audit_incidents(mock_connection, DatasetType.CIVILIANS_SHOT, limit=5)
        query = mock_cursor.execute.call_args[0][0]
        assert "NOT IN" in query
        excluded = DEV_SET_IDS | TEST_SET_IDS[DatasetType.CIVILIANS_SHOT]
        for incident_id in sorted(excluded)[:3]:
            assert str(incident_id) in query

    def test_explicit_exclude_ids_override(self, mock_connection, mock_cursor):
        mock_cursor.fetchall.return_value = []
        select_audit_incidents(
            mock_connection,
            DatasetType.CIVILIANS_SHOT,
            limit=5,
            exclude_ids={424242},
        )
        query = mock_cursor.execute.call_args[0][0]
        assert "424242" in query

    def test_empty_exclude_ids_omits_clause(self, mock_connection, mock_cursor):
        mock_cursor.fetchall.return_value = []
        select_audit_incidents(
            mock_connection, DatasetType.CIVILIANS_SHOT, limit=5, exclude_ids=set()
        )
        query = mock_cursor.execute.call_args[0][0]
        assert "NOT IN" not in query
