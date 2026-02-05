"""Validation module for the enrichment pipeline.

Parse the retrieved articles, and validate the data against extracted metadata.
"""

from src.validation.validate_node import (
    check_date_match,
    check_location_match,
    check_name_match,
    validate_node,
)

__all__ = [
    "check_date_match",
    "check_location_match",
    "check_name_match",
    "validate_node",
]
