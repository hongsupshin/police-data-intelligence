"""Database utilities for the Police Data Intelligence Assistant.

This package provides database connection management and query helpers
for accessing the TJI PostgreSQL database.
"""

from src.database.connection import get_connection

__all__ = ["get_connection"]
