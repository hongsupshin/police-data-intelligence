"""Schema-driven data transformation utilities.

This module provides helper functions for applying cleaning transformations
based on schema definitions. These utilities enable declarative ETL pipelines
where column-to-cleaner mappings drive the transformation process.

Functions in this module work with schema definitions from config.py to
automatically apply appropriate cleaning functions to row data.
"""

from collections.abc import Callable
from typing import Any


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
