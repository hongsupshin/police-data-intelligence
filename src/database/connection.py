"""PostgreSQL database connection helper.

Provides connection management for the TJI database using credentials
from environment variables. Uses psycopg2 for PostgreSQL connectivity.
"""

import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection

# Load environment variables from .env file
load_dotenv()


def get_connection() -> connection:
    """Create and return a PostgreSQL database connection.

    Reads database credentials from environment variables and establishes
    a connection to the TJI PostgreSQL database. The connection should be
    closed by the caller when no longer needed.

    Environment variables required:
        DB_HOST: Database host address (e.g., 'localhost')
        DB_PORT: Database port (e.g., '5432')
        DB_NAME: Database name (e.g., 'tji_data')
        DB_USER: Database username
        DB_PASSWORD: Database password

    Returns:
        Active PostgreSQL connection object.

    Raises:
        psycopg2.OperationalError: If connection fails due to invalid
            credentials or unreachable database.
        KeyError: If required environment variables are missing.

    Examples:
        >>> conn = get_connection()
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT COUNT(*) FROM incidents")
        >>> print(cursor.fetchone()[0])
        1956
        >>> conn.close()
    """
    # Validate required environment variables (DB_PASSWORD can be empty for local auth)
    required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER"]
    missing_vars = [var for var in required_vars if os.getenv(var) is None]

    if missing_vars:
        raise KeyError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Create connection
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

    return conn
