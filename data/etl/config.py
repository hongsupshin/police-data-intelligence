"""Database configuration for ETL processes.

This module contains database connection parameters for PostgreSQL.
"""

import os

# Database configuration
# Uses environment variables if set, otherwise falls back to defaults
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "tji_police_data"),
    "user": os.getenv("POSTGRES_USER", os.getenv("USER", "postgres")),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}
