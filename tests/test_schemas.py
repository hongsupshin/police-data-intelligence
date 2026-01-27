#!/usr/bin/env python3
"""Unit tests for schema definitions and schema processing utilities.

This module tests the schema definitions (config.py) and helper functions
(schema_utils.py) that enable schema-driven data transformation in the ETL
pipeline.

Test coverage includes:
- apply_schema: Applying cleaning functions based on table schema
- clean_entity_fields: Extracting entity fields with prefix pattern
- clean_entity_fields_with_suffix: Handling numbered entity fields
- Schema definitions: Validating schema structure and completeness
"""


import pandas as pd
import pytest

from data.etl.cleaners import clean_boolean, clean_date, clean_integer, clean_text
from data.etl.config import (
    AGENCY_ENTITY_SCHEMA,
    CIVILIAN_ENTITY_SCHEMA,
    CIVILIANS_SHOT_INCIDENT_SCHEMA,
    OFFICER_ENTITY_SCHEMA,
    OFFICERS_SHOT_INCIDENT_SCHEMA,
)
from data.etl.schema_utils import (
    apply_schema,
    clean_entity_fields,
    clean_entity_fields_with_suffix,
)


class TestApplySchema:
    """Test cases for the apply_schema function."""

    def test_applies_cleaners_in_order(self):
        """Test that cleaners are applied in schema order."""
        schema = [
            ("name", clean_text),
            ("age", clean_integer),
            ("active", clean_boolean),
        ]

        row = pd.Series({"name": "  John Doe  ", "age": "25", "active": "true"})

        result = apply_schema(row, schema)

        assert result == ["John Doe", 25, True]

    def test_handles_missing_values(self):
        """Test that missing values are handled correctly."""
        schema = [
            ("name", clean_text),
            ("age", clean_integer),
            ("date", clean_date),
        ]

        row = pd.Series({"name": None, "age": "", "date": pd.NA})

        result = apply_schema(row, schema)

        assert result == [None, None, None]

    def test_handles_mixed_valid_and_invalid(self):
        """Test schema with mix of valid and invalid values."""
        schema = [
            ("city", clean_text),
            ("count", clean_integer),
            ("flag", clean_boolean),
        ]

        row = pd.Series({"city": "Austin", "count": "invalid", "flag": "maybe"})

        result = apply_schema(row, schema)

        assert result[0] == "Austin"
        assert result[1] is None  # Invalid integer
        assert result[2] is None  # Invalid boolean

    def test_empty_schema(self):
        """Test that empty schema returns empty list."""
        schema = []
        row = pd.Series({"name": "Test", "age": "25"})

        result = apply_schema(row, schema)

        assert result == []

    def test_preserves_order_with_many_fields(self):
        """Test that field order is preserved with many columns."""
        schema = [
            ("field1", clean_text),
            ("field2", clean_integer),
            ("field3", clean_text),
            ("field4", clean_boolean),
            ("field5", clean_date),
        ]

        row = pd.Series({
            "field1": "A",
            "field2": "1",
            "field3": "B",
            "field4": "true",
            "field5": "2020-01-15",
        })

        result = apply_schema(row, schema)

        assert len(result) == 5
        assert result[0] == "A"
        assert result[1] == 1
        assert result[2] == "B"
        assert result[3] is True
        assert str(result[4]) == "2020-01-15"


class TestCleanEntityFields:
    """Test cases for the clean_entity_fields function."""

    def test_extracts_fields_with_prefix(self):
        """Test basic field extraction with prefix."""
        schema = [("age", clean_integer), ("race", clean_text), ("gender", clean_text)]

        row = pd.Series({
            "civilian_age": "30",
            "civilian_race": "Hispanic",
            "civilian_gender": "M",
        })

        result = clean_entity_fields(row, "civilian_", schema)

        assert result == {"age": 30, "race": "Hispanic", "gender": "M"}

    def test_handles_empty_prefix(self):
        """Test that empty prefix works correctly."""
        schema = [("name", clean_text), ("count", clean_integer)]

        row = pd.Series({"name": "Test", "count": "42"})

        result = clean_entity_fields(row, "", schema)

        assert result == {"name": "Test", "count": 42}

    def test_handles_missing_fields(self):
        """Test handling of missing fields in row."""
        schema = [
            ("age", clean_integer),
            ("race", clean_text),
            ("missing_field", clean_text),
        ]

        row = pd.Series({"officer_age": "35", "officer_race": "Asian"})

        result = clean_entity_fields(row, "officer_", schema)

        assert result["age"] == 35
        assert result["race"] == "Asian"
        assert result["missing_field"] is None

    def test_unpacks_correctly_to_function(self):
        """Test that result can be unpacked with ** operator."""
        schema = [("x", clean_integer), ("y", clean_integer)]

        row = pd.Series({"point_x": "10", "point_y": "20"})

        result = clean_entity_fields(row, "point_", schema)

        # Simulate function call with **kwargs
        def mock_function(x=None, y=None):
            return x + y

        value = mock_function(**result)
        assert value == 30


class TestCleanEntityFieldsWithSuffix:
    """Test cases for the clean_entity_fields_with_suffix function."""

    def test_extracts_fields_with_suffix(self):
        """Test field extraction with prefix and suffix pattern."""
        schema = [("age", clean_integer), ("race", clean_text), ("gender", clean_text)]

        row = pd.Series({
            "civilian_age_1": "25",
            "civilian_race_1": "White",
            "civilian_gender_1": "F",
        })

        result = clean_entity_fields_with_suffix(row, "civilian_", "_1", schema)

        assert result == {"age": 25, "race": "White", "gender": "F"}

    def test_handles_different_suffix_numbers(self):
        """Test handling of different numbered suffixes."""
        schema = [("name", clean_text), ("age", clean_integer)]

        # Test with suffix "_2"
        row = pd.Series({"officer_name_2": "Smith", "officer_age_2": "40"})

        result = clean_entity_fields_with_suffix(row, "officer_", "_2", schema)

        assert result == {"name": "Smith", "age": 40}

        # Test with suffix "_10"
        row = pd.Series({"officer_name_10": "Jones", "officer_age_10": "35"})

        result = clean_entity_fields_with_suffix(row, "officer_", "_10", schema)

        assert result == {"name": "Jones", "age": 35}

    def test_empty_suffix(self):
        """Test that empty suffix works like clean_entity_fields."""
        schema = [("city", clean_text), ("zip", clean_text)]

        row = pd.Series({"agency_city": "Austin", "agency_zip": "78701"})

        result = clean_entity_fields_with_suffix(row, "agency_", "", schema)

        assert result == {"city": "Austin", "zip": "78701"}

    def test_handles_missing_numbered_fields(self):
        """Test handling when numbered fields don't exist."""
        schema = [("age", clean_integer), ("race", clean_text)]

        row = pd.Series({"civilian_age_1": "30"})  # Missing race

        result = clean_entity_fields_with_suffix(row, "civilian_", "_1", schema)

        assert result["age"] == 30
        assert result["race"] is None


class TestSchemaDefinitions:
    """Test cases validating the schema definitions themselves."""

    def test_civilians_shot_incident_schema_structure(self):
        """Test that civilians_shot incident schema is well-formed."""
        schema = CIVILIANS_SHOT_INCIDENT_SCHEMA

        # Should have 22 fields
        assert len(schema) == 22

        # Each entry should be (column_name, cleaner_function)
        for column_name, cleaner in schema:
            assert isinstance(column_name, str)
            assert callable(cleaner)

        # Verify some key fields exist
        column_names = [name for name, _ in schema]
        assert "ois_report_no" in column_names
        assert "date_incident" in column_names
        assert "incident_city" in column_names
        assert "civilian_died" not in column_names  # Should be in victim junction table

    def test_officers_shot_incident_schema_structure(self):
        """Test that officers_shot incident schema is well-formed."""
        schema = OFFICERS_SHOT_INCIDENT_SCHEMA

        # Should have 10 fields
        assert len(schema) == 10

        # Each entry should be (column_name, cleaner_function)
        for column_name, cleaner in schema:
            assert isinstance(column_name, str)
            assert callable(cleaner)

        # Verify some key fields
        column_names = [name for name, _ in schema]
        assert "ois_report_no" in column_names
        assert "date_incident" in column_names
        assert "civilian_harm" in column_names

    def test_civilian_entity_schema_structure(self):
        """Test that civilian entity schema is well-formed."""
        schema = CIVILIAN_ENTITY_SCHEMA

        # Should have 6 fields (age, race, gender, name_first, name_last, name_full)
        assert len(schema) == 6

        # Verify field names
        field_names = [name for name, _ in schema]
        assert "age" in field_names
        assert "race" in field_names
        assert "gender" in field_names
        assert "name_first" in field_names
        assert "name_last" in field_names
        assert "name_full" in field_names

    def test_officer_entity_schema_structure(self):
        """Test that officer entity schema is well-formed."""
        schema = OFFICER_ENTITY_SCHEMA

        # Should have 5 fields (age, race, gender, name_first, name_last)
        assert len(schema) == 5

        # Verify field names
        field_names = [name for name, _ in schema]
        assert "age" in field_names
        assert "race" in field_names
        assert "gender" in field_names
        assert "name_first" in field_names
        assert "name_last" in field_names

    def test_agency_entity_schema_structure(self):
        """Test that agency entity schema is well-formed."""
        schema = AGENCY_ENTITY_SCHEMA

        # Should have 4 fields
        assert len(schema) == 4

        # Verify field names
        field_names = [name for name, _ in schema]
        assert "name" in field_names
        assert "city" in field_names
        assert "county" in field_names
        assert "zip_code" in field_names

    def test_schema_cleaners_are_correct_type(self):
        """Test that schemas use appropriate cleaner functions."""
        # Check civilians_shot incident schema uses correct cleaners
        schema_dict = dict(CIVILIANS_SHOT_INCIDENT_SCHEMA)

        # Text fields should use clean_text
        assert schema_dict["ois_report_no"] == clean_text
        assert schema_dict["incident_city"] == clean_text

        # Date fields should use clean_date
        assert schema_dict["date_ag_received"] == clean_date
        assert schema_dict["date_incident"] == clean_date

        # Integer fields should use clean_integer
        assert schema_dict["num_officers_recorded"] == clean_integer
        assert schema_dict["num_reports_filed"] == clean_integer

        # Boolean fields should use clean_boolean
        assert schema_dict["deadly_weapon"] == clean_boolean
        assert schema_dict["officer_on_duty"] == clean_boolean


class TestSchemaIntegration:
    """Integration tests combining schemas with helper functions."""

    def test_realistic_civilians_shot_row(self):
        """Test applying civilians_shot schema to realistic row data."""
        # Simulate a real CSV row
        row = pd.Series({
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
            "cdr_narrative": "Incident description...",
            "custodial_death_report": "false",
            "lea_narrative_published": "Published narrative...",
            "lea_narrative_shorter": "Short narrative...",
        })

        result = apply_schema(row, CIVILIANS_SHOT_INCIDENT_SCHEMA)

        # Should return 22 values
        assert len(result) == 22

        # Verify some key values
        assert result[0] == "OIS-2020-001"  # ois_report_no
        assert result[5] == "Austin"  # incident_city
        assert result[12] is True  # deadly_weapon
        assert result[13] == 2  # num_officers_recorded

    def test_realistic_entity_extraction(self):
        """Test extracting entity fields from realistic row."""
        row = pd.Series({
            "civilian_age": "30",
            "civilian_race": "Hispanic",
            "civilian_gender": "M",
            "civilian_name_first": "John",
            "civilian_name_last": "Doe",
            "civilian_name_full": "John Doe",
            "other_field": "ignored",
        })

        result = clean_entity_fields(row, "civilian_", CIVILIAN_ENTITY_SCHEMA)

        assert result["age"] == 30
        assert result["race"] == "Hispanic"
        assert result["gender"] == "M"
        assert result["name_first"] == "John"
        assert result["name_last"] == "Doe"
        assert result["name_full"] == "John Doe"
        assert "other_field" not in result

    def test_multiple_numbered_entities(self):
        """Test extracting multiple numbered entities (e.g., officer_1, officer_2)."""
        row = pd.Series({
            "officer_age_1": "35",
            "officer_race_1": "White",
            "officer_gender_1": "M",
            "officer_age_2": "42",
            "officer_race_2": "Black",
            "officer_gender_2": "F",
            "officer_age_3": "28",
            "officer_race_3": "Asian",
            "officer_gender_3": "M",
        })

        # Schema for officers (no names in civilians_shot dataset)
        schema = [("age", clean_integer), ("race", clean_text), ("gender", clean_text)]

        # Extract officer 1
        officer1 = clean_entity_fields_with_suffix(row, "officer_", "_1", schema)
        assert officer1 == {"age": 35, "race": "White", "gender": "M"}

        # Extract officer 2
        officer2 = clean_entity_fields_with_suffix(row, "officer_", "_2", schema)
        assert officer2 == {"age": 42, "race": "Black", "gender": "F"}

        # Extract officer 3
        officer3 = clean_entity_fields_with_suffix(row, "officer_", "_3", schema)
        assert officer3 == {"age": 28, "race": "Asian", "gender": "M"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
