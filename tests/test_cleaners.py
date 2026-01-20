#!/usr/bin/env python3
"""Unit tests for data/etl/cleaners.py data cleaning functions.

This module tests the pure data transformation functions that convert CSV values
into appropriate Python types for database insertion. These functions handle
missing values (NA, empty strings) and various input formats gracefully.

Test coverage includes:
- Boolean conversion (various true/false representations)
- Integer conversion (with float truncation)
- Date conversion (multiple date formats)
- Timestamp conversion
- Text conversion (whitespace stripping)
"""

from datetime import date

import pandas as pd
import pytest

from data.etl.cleaners import (
    clean_boolean,
    clean_date,
    clean_integer,
    clean_text,
    clean_timestamp,
)


class TestCleanBoolean:
    """Test cases for the clean_boolean function."""

    def test_none_values(self) -> None:
        """Test that None and empty values return None."""
        assert clean_boolean(None) is None
        assert clean_boolean("") is None
        assert clean_boolean(pd.NA) is None

    def test_true_values(self) -> None:
        """Test that various true representations are converted correctly."""
        assert clean_boolean(True) is True
        assert clean_boolean("true") is True
        assert clean_boolean("True") is True
        assert clean_boolean("TRUE") is True
        assert clean_boolean("t") is True
        assert clean_boolean("yes") is True
        assert clean_boolean("1") is True

    def test_false_values(self) -> None:
        """Test that various false representations are converted correctly."""
        assert clean_boolean(False) is False
        assert clean_boolean("false") is False
        assert clean_boolean("False") is False
        assert clean_boolean("FALSE") is False
        assert clean_boolean("f") is False
        assert clean_boolean("no") is False
        assert clean_boolean("0") is False

    def test_invalid_values(self):
        """Test that invalid boolean values return None."""
        assert clean_boolean("invalid") is None
        assert clean_boolean("maybe") is None


class TestCleanInteger:
    """Test cases for the clean_integer function."""

    def test_none_values(self):
        """Test that None and empty values return None."""
        assert clean_integer(None) is None
        assert clean_integer("") is None
        assert clean_integer(pd.NA) is None

    def test_valid_integers(self):
        """Test conversion of valid integer values."""
        assert clean_integer(42) == 42
        assert clean_integer("42") == 42
        assert clean_integer(42.0) == 42
        assert clean_integer("42.7") == 42  # Truncates floats

    def test_invalid_values(self):
        """Test that invalid values return None."""
        assert clean_integer("not a number") is None
        assert clean_integer("12abc") is None


class TestCleanDate:
    """Test cases for the clean_date function."""

    def test_none_values(self):
        """Test that None and empty values return None."""
        assert clean_date(None) is None
        assert clean_date("") is None
        assert clean_date(pd.NA) is None

    def test_valid_dates(self):
        """Test conversion of valid date strings."""
        result = clean_date("2020-01-15")
        assert result == date(2020, 1, 15)

        result = clean_date("01/15/2020")
        assert result == date(2020, 1, 15)

    def test_invalid_dates(self):
        """Test that invalid dates return None."""
        assert clean_date("not a date") is None
        assert clean_date("2020-13-45") is None


class TestCleanTimestamp:
    """Test cases for the clean_timestamp function."""

    def test_none_values(self):
        """Test that None and empty values return None."""
        assert clean_timestamp(None) is None
        assert clean_timestamp("") is None
        assert clean_timestamp(pd.NA) is None

    def test_valid_timestamps(self):
        """Test conversion of valid timestamp strings."""
        result = clean_timestamp("2020-01-15 14:30:00")
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2020
        assert result.month == 1
        assert result.day == 15

    def test_invalid_timestamps(self):
        """Test that invalid timestamps return None."""
        assert clean_timestamp("not a timestamp") is None


class TestCleanText:
    """Test cases for the clean_text function."""

    def test_none_values(self):
        """Test that None and empty values return None."""
        assert clean_text(None) is None
        assert clean_text("") is None
        assert clean_text(pd.NA) is None

    def test_valid_text(self):
        """Test text conversion and whitespace stripping."""
        assert clean_text("hello") == "hello"
        assert clean_text("  hello  ") == "hello"  # Strips whitespace
        assert clean_text(42) == "42"  # Converts to string


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
