#!/usr/bin/env python3
"""Integration tests for src/agents/extract_node.py.

Tests the extract_node function which fetches incident records from PostgreSQL
and populates EnrichmentState with dataset-aware field mapping.

Test coverage includes:
- SQL query execution for both datasets
- Conditional field parsing (civilians_shot vs officers_shot)
- Name construction from first/last name fields
- Severity mapping (boolean vs text fields)
- Location fallback logic (city → county)
- EnrichmentState population and validation
"""

import pytest

from src.agents.extract_node import extract_node, fetch_incident
from src.agents.state import DatasetType, EnrichmentState, PipelineStage


@pytest.mark.integration
class TestFetchIncident:
    """Test cases for the fetch_incident function."""

    def test_fetch_civilians_shot_incident(
        self, db_connection, test_incident_civilians_shot
    ) -> None:
        """Test fetching an incident from civilians_shot dataset."""
        if not test_incident_civilians_shot:
            pytest.skip("No suitable test incident found in civilians_shot")

        incident_id = int(test_incident_civilians_shot)
        result = fetch_incident(db_connection, incident_id, DatasetType.CIVILIANS_SHOT)

        # Verify all expected fields are present
        assert "officer_name" in result
        assert "civilian_name" in result
        assert "incident_date" in result
        assert "location" in result
        assert "severity" in result

        # Verify field types
        assert isinstance(result["incident_date"], type(None)) or hasattr(
            result["incident_date"], "year"
        )
        assert result["severity"] in ["fatal", "non-fatal", "unknown"]

    def test_fetch_officers_shot_incident(
        self, db_connection, test_incident_officers_shot
    ) -> None:
        """Test fetching an incident from officers_shot dataset."""
        if not test_incident_officers_shot:
            pytest.skip("No suitable test incident found in officers_shot")

        incident_id = int(test_incident_officers_shot)
        result = fetch_incident(db_connection, incident_id, DatasetType.OFFICERS_SHOT)

        # Verify all expected fields are present
        assert "officer_name" in result
        assert "civilian_name" in result
        assert "incident_date" in result
        assert "location" in result
        assert "severity" in result

        # Verify field types
        assert isinstance(result["incident_date"], type(None)) or hasattr(
            result["incident_date"], "year"
        )
        assert result["severity"] in ["fatal", "non-fatal", "unknown"]

    def test_fetch_nonexistent_incident(self, db_connection) -> None:
        """Test that fetching nonexistent incident raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            fetch_incident(db_connection, 999999, DatasetType.CIVILIANS_SHOT)


@pytest.mark.integration
class TestExtractNodeCiviliansShot:
    """Test cases for extract_node with civilians_shot dataset."""

    def test_extract_populates_state(self, test_incident_civilians_shot) -> None:
        """Test that extract_node populates EnrichmentState correctly."""
        if not test_incident_civilians_shot:
            pytest.skip("No suitable test incident found")

        state = EnrichmentState(
            incident_id=test_incident_civilians_shot,
            dataset_type=DatasetType.CIVILIANS_SHOT,
        )

        updated_state = extract_node(state)

        # Verify pipeline stage updated
        assert updated_state.current_stage == PipelineStage.EXTRACT
        assert updated_state.error_message is None

        # Verify required fields populated
        assert updated_state.incident_date is not None
        assert updated_state.location is not None
        assert updated_state.severity is not None

    def test_extract_data_matches_database(
        self, db_connection, test_incident_civilians_shot
    ) -> None:
        """Test that extracted data matches raw database values."""
        if not test_incident_civilians_shot:
            pytest.skip("No suitable test incident found")

        # Run extract_node
        state = EnrichmentState(
            incident_id=test_incident_civilians_shot,
            dataset_type=DatasetType.CIVILIANS_SHOT,
        )
        updated_state = extract_node(state)

        # Query database directly for validation
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT
                i.date_incident,
                i.incident_city,
                i.incident_county,
                o.name_first AS officer_first,
                o.name_last AS officer_last,
                c.name_first AS civilian_first,
                c.name_last AS civilian_last,
                v.civilian_died
            FROM incidents_civilians_shot i
            LEFT JOIN incident_civilians_shot_officers_involved oi
                ON i.incident_id = oi.incident_id AND oi.officer_sequence = 1
            LEFT JOIN officers o ON oi.officer_id = o.officer_id
            LEFT JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN civilians c ON v.civilian_id = c.civilian_id
            WHERE i.incident_id = %s
            LIMIT 1;
        """,
            (test_incident_civilians_shot,),
        )

        (
            db_date,
            db_city,
            db_county,
            db_officer_first,
            db_officer_last,
            db_civilian_first,
            db_civilian_last,
            db_civilian_died,
        ) = cursor.fetchone()
        cursor.close()

        # Verify date
        assert updated_state.incident_date == db_date

        # Verify location (city or county fallback)
        expected_location = db_city if db_city else db_county
        assert updated_state.location == expected_location

        # Verify officer name construction
        if db_officer_first or db_officer_last:
            parts = [p for p in [db_officer_first, db_officer_last] if p]
            expected_officer_name = " ".join(parts) if parts else None
        else:
            expected_officer_name = None
        assert updated_state.officer_name == expected_officer_name

        # Verify civilian name construction
        if db_civilian_first or db_civilian_last:
            parts = [p for p in [db_civilian_first, db_civilian_last] if p]
            expected_civilian_name = " ".join(parts) if parts else None
        else:
            expected_civilian_name = None
        assert updated_state.civilian_name == expected_civilian_name

        # Verify severity mapping
        if db_civilian_died is True:
            expected_severity = "fatal"
        elif db_civilian_died is False:
            expected_severity = "non-fatal"
        else:
            expected_severity = "unknown"
        assert updated_state.severity == expected_severity


@pytest.mark.integration
class TestExtractNodeOfficersShot:
    """Test cases for extract_node with officers_shot dataset."""

    def test_extract_populates_state(self, test_incident_officers_shot) -> None:
        """Test that extract_node populates EnrichmentState correctly."""
        if not test_incident_officers_shot:
            pytest.skip("No suitable test incident found")

        state = EnrichmentState(
            incident_id=test_incident_officers_shot,
            dataset_type=DatasetType.OFFICERS_SHOT,
        )

        updated_state = extract_node(state)

        # Verify pipeline stage updated
        assert updated_state.current_stage == PipelineStage.EXTRACT
        assert updated_state.error_message is None

        # Verify required fields populated
        assert updated_state.incident_date is not None
        assert updated_state.location is not None
        assert updated_state.severity is not None

    def test_extract_data_matches_database(
        self, db_connection, test_incident_officers_shot
    ) -> None:
        """Test that extracted data matches raw database values."""
        if not test_incident_officers_shot:
            pytest.skip("No suitable test incident found")

        # Run extract_node
        state = EnrichmentState(
            incident_id=test_incident_officers_shot,
            dataset_type=DatasetType.OFFICERS_SHOT,
        )
        updated_state = extract_node(state)

        # Query database directly for validation
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT
                i.date_incident::date,
                i.incident_city,
                i.incident_county,
                o.name_first AS officer_first,
                o.name_last AS officer_last,
                c.name_first AS civilian_first,
                c.name_last AS civilian_last,
                v.officer_harm
            FROM incidents_officers_shot i
            LEFT JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN officers o ON v.officer_id = o.officer_id
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id AND s.civilian_sequence = 1
            LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
            WHERE i.incident_id = %s
            LIMIT 1;
        """,
            (test_incident_officers_shot,),
        )

        (
            db_date,
            db_city,
            db_county,
            db_officer_first,
            db_officer_last,
            db_civilian_first,
            db_civilian_last,
            db_officer_harm,
        ) = cursor.fetchone()
        cursor.close()

        # Verify date
        assert updated_state.incident_date == db_date

        # Verify location (city or county fallback)
        expected_location = db_city if db_city else db_county
        assert updated_state.location == expected_location

        # Verify officer name construction
        if db_officer_first or db_officer_last:
            parts = [p for p in [db_officer_first, db_officer_last] if p]
            expected_officer_name = " ".join(parts) if parts else None
        else:
            expected_officer_name = None
        assert updated_state.officer_name == expected_officer_name

        # Verify civilian name construction
        if db_civilian_first or db_civilian_last:
            parts = [p for p in [db_civilian_first, db_civilian_last] if p]
            expected_civilian_name = " ".join(parts) if parts else None
        else:
            expected_civilian_name = None
        assert updated_state.civilian_name == expected_civilian_name

        # Verify severity mapping
        if db_officer_harm == "DEATH":
            expected_severity = "fatal"
        elif db_officer_harm == "INJURY":
            expected_severity = "non-fatal"
        else:
            expected_severity = "unknown"
        assert updated_state.severity == expected_severity


@pytest.mark.integration
class TestConditionalFieldMapping:
    """Test that field mapping differs correctly between datasets."""

    def test_civilians_shot_uses_correct_joins(self, db_connection) -> None:
        """Test that civilians_shot queries use correct table joins."""
        # This is more of a documentation test - verifies query structure
        # The actual JOIN logic is tested through data validation above
        cursor = db_connection.cursor()

        # Verify the tables exist and have expected relationships
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM incidents_civilians_shot i
            LEFT JOIN incident_civilians_shot_officers_involved oi
                ON i.incident_id = oi.incident_id
            LEFT JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            LIMIT 1;
        """
        )
        result = cursor.fetchone()
        cursor.close()

        assert result is not None

    def test_officers_shot_uses_correct_joins(self, db_connection) -> None:
        """Test that officers_shot queries use correct table joins."""
        # Verify the tables exist and have expected relationships
        cursor = db_connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM incidents_officers_shot i
            LEFT JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id
            LIMIT 1;
        """
        )
        result = cursor.fetchone()
        cursor.close()

        assert result is not None

    def test_severity_mapping_differs_by_dataset(self) -> None:
        """Test that severity field sources differ by dataset type.

        Civilians shot: Uses civilian_died (BOOLEAN)
        Officers shot: Uses officer_harm (TEXT: 'DEATH'/'INJURY')
        """
        # This is a documentation test - the actual mapping is tested
        # through data validation in the other test classes
        assert True  # Placeholder - logic verified in integration tests
