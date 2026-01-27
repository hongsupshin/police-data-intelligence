"""Data cleaning and type conversion functions.

This module provides pure functions for cleaning and converting data values
from CSV files into appropriate Python types for database insertion. These
functions are designed to handle missing values (NA, empty strings) and various
input formats gracefully.
"""

from datetime import date
from typing import Any

import pandas as pd


def clean_boolean(value: Any) -> bool | None:
    """Convert various boolean representations to Python bool or None.

    Handles standard boolean values plus TJI-specific outcome values:
    - "DEATH" maps to True (fatal outcome)
    - "INJURY" maps to False (non-fatal outcome)

    Args:
        value: A value that may represent a boolean (bool, str, pd.NA, etc.).

    Returns:
        True, False, or None. Returns None for missing values or invalid inputs.

    Examples:
        >>> clean_boolean("DEATH")
        True
        >>> clean_boolean("INJURY")
        False
        >>> clean_boolean("true")
        True
        >>> clean_boolean(None)
        None
    """
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_upper = value.upper().strip()
        # Handle TJI-specific outcome values
        if value_upper == "DEATH":
            return True
        elif value_upper == "INJURY":
            return False
        # Handle standard boolean representations
        value_lower = value.lower().strip()
        if value_lower in ("true", "t", "yes", "1"):
            return True
        elif value_lower in ("false", "f", "no", "0"):
            return False
    return None


def clean_integer(value: Any) -> int | None:
    """Convert value to integer or None.

    Args:
        value: A value that may represent an integer (int, str, float, etc.).

    Returns:
        An integer, or None if the value is missing or cannot be converted.
        Floats are truncated (not rounded).
    """
    if pd.isna(value) or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def clean_date(value: Any) -> date | None:
    """Convert value to date or None.

    Args:
        value: A value that may represent a date (str, datetime, etc.).

    Returns:
        A date object, or None if the value is missing or cannot be parsed.
    """
    if pd.isna(value) or value == "":
        return None
    try:
        return pd.to_datetime(value).date()  # type: ignore[no-any-return]
    except Exception:
        return None


def clean_timestamp(value: Any) -> pd.Timestamp | None:
    """Convert value to timestamp or None.

    Args:
        value: A value that may represent a timestamp (str, datetime, etc.).

    Returns:
        A pandas Timestamp object, or None if the value is missing or
        cannot be parsed.
    """
    if pd.isna(value) or value == "":
        return None
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def clean_text(value: Any) -> str | None:
    """Convert value to text or None.

    Args:
        value: A value that may represent text (str, int, etc.).

    Returns:
        A stripped string, or None if the value is missing.
    """
    if pd.isna(value) or value == "":
        return None
    return str(value).strip()
