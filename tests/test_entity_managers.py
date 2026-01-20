#!/usr/bin/env python3
"""Unit tests for data/etl/entity_managers.py entity deduplication functions.

This module tests the database entity creation functions that handle
deduplication using PostgreSQL's INSERT...ON CONFLICT pattern. These
functions ensure that duplicate entities are not created and existing
entity IDs are reused.

Test coverage includes:
- get_or_create_officer: Officer record deduplication
- get_or_create_civilian: Civilian record deduplication
- get_or_create_agency: Agency record deduplication
"""

from unittest.mock import Mock

import pytest

from data.etl.entity_managers import (
    get_or_create_agency,
    get_or_create_civilian,
    get_or_create_officer,
)


class TestGetOrCreateOfficer:
    """Test cases for the get_or_create_officer function."""

    def test_all_none_returns_none(self):
        """Test that function returns None when all parameters are None."""
        cursor = Mock()
        result = get_or_create_officer(cursor, None, None, None, None, None)
        assert result is None
        cursor.execute.assert_not_called()

    def test_creates_new_officer(self):
        """Test creation of a new officer record with full data."""
        cursor = Mock()
        cursor.fetchone.return_value = (123,)

        result = get_or_create_officer(
            cursor, age=35, race="White", gender="M", name_first="John", name_last="Doe"
        )

        assert result == 123
        cursor.execute.assert_called_once()

        # Verify SQL contains INSERT...ON CONFLICT
        sql_call = cursor.execute.call_args[0][0]
        assert "INSERT INTO officers" in sql_call
        assert "ON CONFLICT" in sql_call
        assert "RETURNING officer_id" in sql_call

        # Verify parameters
        params = cursor.execute.call_args[0][1]
        assert params == (35, "White", "M", "John", "Doe")

    def test_partial_data(self):
        """Test creation with partial officer data."""
        cursor = Mock()
        cursor.fetchone.return_value = (456,)

        result = get_or_create_officer(
            cursor, age=None, race="Hispanic", gender="F"
        )

        assert result == 456
        params = cursor.execute.call_args[0][1]
        assert params == (None, "Hispanic", "F", None, None)


class TestGetOrCreateCivilian:
    """Test cases for the get_or_create_civilian function."""

    def test_all_none_returns_none(self):
        """Test that function returns None when all parameters are None."""
        cursor = Mock()
        result = get_or_create_civilian(
            cursor, None, None, None, None, None, None
        )
        assert result is None
        cursor.execute.assert_not_called()

    def test_creates_new_civilian(self):
        """Test creation of a new civilian record with full data."""
        cursor = Mock()
        cursor.fetchone.return_value = (789,)

        result = get_or_create_civilian(
            cursor,
            age=28,
            race="Black",
            gender="M",
            name_first="David",
            name_last="Joseph",
            name_full="David Joseph",
        )

        assert result == 789
        cursor.execute.assert_called_once()

        sql_call = cursor.execute.call_args[0][0]
        assert "INSERT INTO civilians" in sql_call
        assert "ON CONFLICT" in sql_call

        params = cursor.execute.call_args[0][1]
        assert params == (28, "Black", "M", "David", "Joseph", "David Joseph")


class TestGetOrCreateAgency:
    """Test cases for the get_or_create_agency function."""

    def test_all_none_returns_none(self):
        """Test that function returns None when required parameters are None."""
        cursor = Mock()
        result = get_or_create_agency(cursor, None, None, None, None)
        assert result is None
        cursor.execute.assert_not_called()

    def test_creates_new_agency(self):
        """Test creation of a new agency record with full data."""
        cursor = Mock()
        cursor.fetchone.return_value = (101,)

        result = get_or_create_agency(
            cursor,
            name="Austin Police Department",
            city="Austin",
            county="Travis",
            zip_code="78701",
        )

        assert result == 101
        cursor.execute.assert_called_once()

        sql_call = cursor.execute.call_args[0][0]
        assert "INSERT INTO agencies" in sql_call
        assert "ON CONFLICT" in sql_call

        params = cursor.execute.call_args[0][1]
        assert params == ("Austin Police Department", "Austin", "Travis", "78701")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
