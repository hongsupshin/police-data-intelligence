#!/usr/bin/env python3
"""Shared test fixtures for the test suite.

This module provides reusable pytest fixtures for database connections,
cursors, and mock objects used across multiple test files.
"""

from typing import Any
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_cursor() -> Mock:
    """Provide a mock database cursor for testing.

    Returns:
        A Mock object that simulates a psycopg2 cursor.
    """
    cursor = Mock()
    cursor.fetchone.return_value = (1,)  # Default return value
    return cursor


@pytest.fixture
def mock_connection(mock_cursor: Mock) -> Mock:
    """Provide a mock database connection for testing.

    Args:
        mock_cursor: The mock cursor fixture.

    Returns:
        A Mock object that simulates a psycopg2 connection.
    """
    conn = Mock()
    conn.cursor.return_value = mock_cursor
    return conn


@pytest.fixture
def db_config() -> dict[str, Any]:
    """Provide test database configuration.

    Returns:
        A dict with PostgreSQL connection parameters for the test database.
    """
    return {
        "dbname": "tji_police_data_test",
        "user": "postgres",
        "password": "",
        "host": "localhost",
        "port": 5432,
    }


@pytest.fixture
def db_connection():
    """Provide a real database connection for integration tests.

    Yields:
        Active PostgreSQL connection to the actual database.
        Connection is automatically closed after test completes.
    """
    from src.database.connection import get_connection

    conn = get_connection()
    yield conn
    conn.close()


@pytest.fixture
def test_incident_civilians_shot(db_connection):
    """Find a test incident from civilians_shot dataset.

    Args:
        db_connection: Database connection fixture.

    Returns:
        Incident ID as string, or None if no suitable incident found.
    """
    cursor = db_connection.cursor()
    cursor.execute(
        """
        SELECT i.incident_id
        FROM incidents_civilians_shot i
        WHERE i.date_incident IS NOT NULL
            AND i.incident_city IS NOT NULL
        LIMIT 1;
    """
    )
    result = cursor.fetchone()
    cursor.close()
    return str(result[0]) if result else None


@pytest.fixture
def test_incident_officers_shot(db_connection):
    """Find a test incident from officers_shot dataset.

    Args:
        db_connection: Database connection fixture.

    Returns:
        Incident ID as string, or None if no suitable incident found.
    """
    cursor = db_connection.cursor()
    cursor.execute(
        """
        SELECT i.incident_id
        FROM incidents_officers_shot i
        WHERE i.date_incident IS NOT NULL
            AND i.incident_city IS NOT NULL
        LIMIT 1;
    """
    )
    result = cursor.fetchone()
    cursor.close()
    return str(result[0]) if result else None
