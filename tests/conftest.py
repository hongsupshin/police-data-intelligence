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
