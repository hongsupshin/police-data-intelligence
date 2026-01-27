"""Database configuration and table schemas for ETL processes.

This module contains:
1. Database connection parameters for PostgreSQL
2. Table schemas that map CSV columns to cleaning functions

The schema-driven approach eliminates repetitive clean_* calls and
makes the ETL pipeline more maintainable and less error-prone.

For schema processing utilities, see schema_utils.py.
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

# Type alias for schema definitions
SchemaType = list[tuple[str, Callable[[Any], Any]]]

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
# Use with schema_utils functions for automatic data transformation.

CIVILIANS_SHOT_INCIDENT_SCHEMA: SchemaType = [
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

OFFICERS_SHOT_INCIDENT_SCHEMA: SchemaType = [
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

CIVILIAN_ENTITY_SCHEMA: SchemaType = [
    ("age", clean_integer),
    ("race", clean_text),
    ("gender", clean_text),
    ("name_first", clean_text),
    ("name_last", clean_text),
    ("name_full", clean_text),
]

OFFICER_ENTITY_SCHEMA: SchemaType = [
    ("age", clean_integer),
    ("race", clean_text),
    ("gender", clean_text),
    ("name_first", clean_text),
    ("name_last", clean_text),
]

AGENCY_ENTITY_SCHEMA: SchemaType = [
    ("name", clean_text),
    ("city", clean_text),
    ("county", clean_text),
    ("zip_code", clean_text),
]
