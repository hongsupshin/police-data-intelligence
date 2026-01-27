"""Database configuration and table schemas for ETL processes.

This module contains:
1. Database connection parameters for PostgreSQL
2. Table schemas that map CSV columns to cleaning functions
3. Helper function to apply schema-based cleaning

The schema-driven approach eliminates repetitive clean_* calls and
makes the ETL pipeline more maintainable and less error-prone.
"""

import os
from collections.abc import Callable
from typing import Any

from data.etl.cleaners import (
    clean_boolean,
    clean_date,
    clean_integer,
    clean_text,
    clean_timestamp,
)

# Database configuration
# Uses environment variables if set, otherwise falls back to defaults
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "tji_police_data"),
    "user": os.getenv("POSTGRES_USER", os.getenv("USER", "postgres")),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}


# ============================================================================
# Table Schemas: Map CSV columns to cleaning functions
# ============================================================================
# Each schema is a list of (column_name, cleaner_function) tuples.
# Order matters - must match the INSERT statement column order.

CIVILIANS_SHOT_INCIDENT_SCHEMA: list[tuple[str, Callable[[Any], Any]]] = [
    ("ois_report_no", clean_text),
    ("date_ag_received", clean_date),
    ("date_incident", clean_date),
    ("time_incident", clean_text),
    ("incident_address", clean_text),
    ("incident_city", clean_text),
    ("incident_county", clean_text),
    ("incident_zip", clean_text),
    ("incident_result_of", clean_text),
    ("incident_call_other", clean_text),
    ("weapon_reported_by_media", clean_text),
    ("weapon_reported_by_media_category", clean_text),
    ("deadly_weapon", clean_boolean),
    ("num_officers_recorded", clean_integer),
    ("multiple_officers_involved", clean_boolean),
    ("officer_on_duty", clean_boolean),
    ("num_reports_filed", clean_integer),
    ("num_rows_about_this_incident", clean_integer),
    ("cdr_narrative", clean_text),
    ("custodial_death_report", clean_boolean),
    ("lea_narrative_published", clean_text),
    ("lea_narrative_shorter", clean_text),
]

OFFICERS_SHOT_INCIDENT_SCHEMA: list[tuple[str, Callable[[Any], Any]]] = [
    ("ois_report_no", clean_text),
    ("date_ag_received", clean_date),
    ("date_incident", clean_timestamp),  # Officers shot uses timestamp
    ("incident_address", clean_text),
    ("incident_city", clean_text),
    ("incident_county", clean_text),
    ("incident_zip", clean_text),
    ("num_civilians_recorded", clean_integer),
    ("civilian_harm", clean_text),
    ("civilian_suicide", clean_boolean),
]

CIVILIAN_ENTITY_SCHEMA: list[tuple[str, Callable[[Any], Any]]] = [
    ("age", clean_integer),
    ("race", clean_text),
    ("gender", clean_text),
    ("name_first", clean_text),
    ("name_last", clean_text),
    ("name_full", clean_text),
]

OFFICER_ENTITY_SCHEMA: list[tuple[str, Callable[[Any], Any]]] = [
    ("age", clean_integer),
    ("race", clean_text),
    ("gender", clean_text),
    ("name_first", clean_text),
    ("name_last", clean_text),
]

AGENCY_ENTITY_SCHEMA: list[tuple[str, Callable[[Any], Any]]] = [
    ("name", clean_text),
    ("city", clean_text),
    ("county", clean_text),
    ("zip_code", clean_text),
]


def apply_schema(row: Any, schema: list[tuple[str, Callable[[Any], Any]]]) -> list[Any]:
    """Apply cleaning functions to row data based on schema.

    Takes a pandas Series (CSV row) and a schema definition, applies
    the appropriate cleaning function to each column value, and returns
    a list of cleaned values in schema order.

    This function enables schema-driven ETL processing, eliminating
    repetitive clean_* function calls and reducing errors.

    Args:
        row: A pandas Series representing a CSV row with named columns.
        schema: List of (column_name, cleaner_function) tuples defining
            the cleaning pipeline. Order determines output order.

    Returns:
        List of cleaned values in the same order as the schema.
        Values are cleaned according to their corresponding functions.

    Examples:
        >>> schema = [("age", clean_integer), ("name", clean_text)]
        >>> row = pd.Series({"age": "25", "name": "  John  "})
        >>> apply_schema(row, schema)
        [25, 'John']

        >>> row = pd.Series({"age": "", "name": None})
        >>> apply_schema(row, schema)
        [None, None]
    """
    return [cleaner(row.get(col_name)) for col_name, cleaner in schema]


def clean_entity_fields(
    row: Any, prefix: str, schema: list[tuple[str, Callable[[Any], Any]]]
) -> dict[str, Any]:
    """Apply cleaning to entity fields with optional column prefix.

    Extracts entity fields from a row, applies appropriate cleaning
    functions, and returns a dictionary suitable for passing as
    keyword arguments to entity creation functions.

    Args:
        row: A pandas Series representing a CSV row.
        prefix: Column name prefix (e.g., "civilian_", "officer_1_").
            Use empty string "" if columns don't have a prefix.
        schema: List of (field_name, cleaner_function) tuples.
            Field names should NOT include the prefix.

    Returns:
        Dictionary mapping field names to cleaned values.
        Ready to be unpacked with ** into function calls.

    Examples:
        >>> schema = [("age", clean_integer), ("race", clean_text)]
        >>> row = pd.Series({"civilian_age": "30", "civilian_race": "Asian"})
        >>> clean_entity_fields(row, "civilian_", schema)
        {'age': 30, 'race': 'Asian'}

        >>> row = pd.Series({"officer_age_2": "45", "officer_race_2": "White"})
        >>> clean_entity_fields(row, "officer_", schema)
        {'age': 45, 'race': 'White'}
    """
    result = {}
    for field_name, cleaner in schema:
        col_name = f"{prefix}{field_name}"
        result[field_name] = cleaner(row.get(col_name))
    return result


def clean_entity_fields_with_suffix(
    row: Any, prefix: str, suffix: str, schema: list[tuple[str, Callable[[Any], Any]]]
) -> dict[str, Any]:
    """Apply cleaning to entity fields with prefix and suffix pattern.

    Handles CSV columns following the pattern: prefix + field_name + suffix.
    This is common for numbered entities like "civilian_age_1", "officer_race_2", etc.

    Args:
        row: A pandas Series representing a CSV row.
        prefix: Column name prefix (e.g., "civilian_", "officer_").
        suffix: Column name suffix (e.g., "_1", "_2").
        schema: List of (field_name, cleaner_function) tuples.
            Field names should NOT include prefix or suffix.

    Returns:
        Dictionary mapping field names to cleaned values.
        Ready to be unpacked with ** into function calls.

    Examples:
        >>> schema = [("age", clean_integer), ("race", clean_text)]
        >>> row = pd.Series({"civilian_age_1": "25", "civilian_race_1": "Hispanic"})
        >>> clean_entity_fields_with_suffix(row, "civilian_", "_1", schema)
        {'age': 25, 'race': 'Hispanic'}

        >>> row = pd.Series({"officer_age_3": "40", "officer_race_3": "Black"})
        >>> clean_entity_fields_with_suffix(row, "officer_", "_3", schema)
        {'age': 40, 'race': 'Black'}
    """
    result = {}
    for field_name, cleaner in schema:
        col_name = f"{prefix}{field_name}{suffix}"
        result[field_name] = cleaner(row.get(col_name))
    return result
