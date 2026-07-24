"""Preflight checks before spending on audit runs.

The ETL was once non-idempotent and silently doubled the database; every
paid run since is gated on verifying the expected clean row counts (the
"second load must be a no-op" precondition).
"""

from psycopg2.extensions import connection

from src.agents.state import DatasetType

# Clean-load row counts for the incident tables (data/README.md).
EXPECTED_ROW_COUNTS: dict[DatasetType, int] = {
    DatasetType.CIVILIANS_SHOT: 1674,
    DatasetType.OFFICERS_SHOT: 282,
}

_INCIDENT_TABLES: dict[DatasetType, str] = {
    DatasetType.CIVILIANS_SHOT: "incidents_civilians_shot",
    DatasetType.OFFICERS_SHOT: "incidents_officers_shot",
}


def verify_db_clean(conn: connection) -> None:
    """Verify both incident tables hold their expected clean row counts.

    Args:
        conn: Active PostgreSQL connection.

    Raises:
        RuntimeError: If any incident table's row count deviates from the
            expected clean-load count (e.g. a doubled ETL load).
    """
    cursor = conn.cursor()
    for dataset_type, expected in EXPECTED_ROW_COUNTS.items():
        table = _INCIDENT_TABLES[dataset_type]
        cursor.execute(f"SELECT COUNT(*) FROM {table};")  # noqa: S608
        actual = cursor.fetchone()[0]
        if actual != expected:
            raise RuntimeError(
                f"DB not clean: {table} has {actual} rows, expected {expected}. "
                "Re-verify the ETL load before spending on audit runs."
            )
