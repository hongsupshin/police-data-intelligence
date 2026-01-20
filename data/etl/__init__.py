"""ETL subpackage for data processing modules.

This subpackage contains:
- cleaners: Data cleaning and type conversion functions
- entity_managers: Database entity creation with deduplication
- loaders: Dataset-specific ETL workflows
- config: Database configuration
"""

from data.etl.cleaners import (
    clean_boolean,
    clean_date,
    clean_integer,
    clean_text,
    clean_timestamp,
)
from data.etl.entity_managers import (
    get_or_create_agency,
    get_or_create_civilian,
    get_or_create_officer,
)
from data.etl.loaders import load_civilians_shot, load_officers_shot

__all__ = [
    "clean_boolean",
    "clean_date",
    "clean_integer",
    "clean_text",
    "clean_timestamp",
    "get_or_create_agency",
    "get_or_create_civilian",
    "get_or_create_officer",
    "load_civilians_shot",
    "load_officers_shot",
]
