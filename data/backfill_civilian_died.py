"""Backfill civilian_died in the incident_civilians_shot_victims table.

The original ETL loaded with a clean_boolean() that didn't handle
"DEATH"/"INJURY" values, mapping them all to NULL. The function was fixed
(commit 9eb79a7) but the database was never re-loaded. This script updates
existing rows in-place, preserving all IDs.

Usage:
    python -m data.backfill_civilian_died --dry-run   # preview changes
    python -m data.backfill_civilian_died              # apply changes
"""

import argparse
from pathlib import Path

import pandas as pd
import psycopg2

from data.etl.cleaners import clean_boolean
from data.etl.config import DB_CONFIG

CSV_PATH = Path("data/tji_civilians-shot.csv")


def diagnose(conn: psycopg2.extensions.connection) -> dict[str, int]:
    """Print and return civilian_died stats from the database.

    Args:
        conn: A psycopg2 database connection.

    Returns:
        Dict with keys: total, non_null, true_count, false_count, null_count.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(civilian_died) AS non_null,
            COUNT(*) FILTER (WHERE civilian_died = TRUE) AS true_count,
            COUNT(*) FILTER (WHERE civilian_died = FALSE) AS false_count,
            COUNT(*) FILTER (WHERE civilian_died IS NULL) AS null_count
        FROM incident_civilians_shot_victims
    """)
    row = cursor.fetchone()
    cursor.close()

    stats = {
        "total": row[0],
        "non_null": row[1],
        "true_count": row[2],
        "false_count": row[3],
        "null_count": row[4],
    }
    print(f"  total rows:    {stats['total']}")
    print(f"  non-null:      {stats['non_null']}")
    print(f"  TRUE (died):   {stats['true_count']}")
    print(f"  FALSE (injury):{stats['false_count']}")
    print(f"  NULL:          {stats['null_count']}")
    return stats


def backfill_by_report_no(
    conn: psycopg2.extensions.connection, df: pd.DataFrame
) -> int:
    """Update civilian_died by matching on ois_report_no.

    Only updates rows where civilian_died IS NULL to ensure idempotency.

    Args:
        conn: A psycopg2 database connection.
        df: DataFrame with ois_report_no and civilian_died columns.

    Returns:
        Number of rows updated.
    """
    cursor = conn.cursor()
    updated = 0

    rows_with_report_no = df[df["ois_report_no"].notna()]
    for _, row in rows_with_report_no.iterrows():
        value = clean_boolean(row["civilian_died"])
        if value is None:
            continue

        cursor.execute(
            """
            UPDATE incident_civilians_shot_victims v
            SET civilian_died = %s
            FROM incidents_civilians_shot i
            WHERE v.incident_id = i.incident_id
              AND i.ois_report_no = %s
              AND v.civilian_died IS NULL
            """,
            (value, str(row["ois_report_no"])),
        )
        updated += cursor.rowcount

    cursor.close()
    print(f"  Updated {updated} rows via ois_report_no match")
    return updated


def backfill_by_composite_key(
    conn: psycopg2.extensions.connection, df: pd.DataFrame
) -> int:
    """Update civilian_died by matching on composite key for remaining rows.

    Uses (date_incident, incident_city, name_full, age, race) with
    IS NOT DISTINCT FROM for nullable fields. Only updates rows where
    civilian_died IS NULL.

    Args:
        conn: A psycopg2 database connection.
        df: DataFrame with date_incident, incident_city, civilian_name_full,
            civilian_age, civilian_race, and civilian_died columns.

    Returns:
        Number of rows updated.
    """
    cursor = conn.cursor()
    updated = 0

    rows_without_report_no = df[df["ois_report_no"].isna()]
    for _, row in rows_without_report_no.iterrows():
        value = clean_boolean(row["civilian_died"])
        if value is None:
            continue

        # Clean fields to match what the ETL loader stored
        date_incident = pd.to_datetime(row.get("date_incident")).date() if pd.notna(row.get("date_incident")) else None
        incident_city = str(row["incident_city"]).strip() if pd.notna(row.get("incident_city")) else None
        name_full = str(row["civilian_name_full"]).strip() if pd.notna(row.get("civilian_name_full")) else None
        age = int(float(row["civilian_age"])) if pd.notna(row.get("civilian_age")) else None
        race = str(row["civilian_race"]).strip() if pd.notna(row.get("civilian_race")) else None

        cursor.execute(
            """
            UPDATE incident_civilians_shot_victims v
            SET civilian_died = %s
            FROM incidents_civilians_shot i, civilians c
            WHERE v.incident_id = i.incident_id
              AND v.civilian_id = c.civilian_id
              AND i.date_incident IS NOT DISTINCT FROM %s
              AND i.incident_city IS NOT DISTINCT FROM %s
              AND c.name_full IS NOT DISTINCT FROM %s
              AND c.age IS NOT DISTINCT FROM %s
              AND c.race IS NOT DISTINCT FROM %s
              AND v.civilian_died IS NULL
            """,
            (value, date_incident, incident_city, name_full, age, race),
        )
        updated += cursor.rowcount

    cursor.close()
    print(f"  Updated {updated} rows via composite key match")
    return updated


def main() -> None:
    """Run the backfill migration with optional --dry-run."""
    parser = argparse.ArgumentParser(
        description="Backfill civilian_died from CSV into existing DB rows."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without committing.",
    )
    args = parser.parse_args()

    conn = psycopg2.connect(**DB_CONFIG)

    print("=== Before backfill ===")
    diagnose(conn)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"\nCSV: {len(df)} rows, "
          f"{df['civilian_died'].value_counts().to_dict()}")

    print("\n--- Pass 1: ois_report_no match ---")
    backfill_by_report_no(conn, df)

    print("\n--- Pass 2: composite key match ---")
    backfill_by_composite_key(conn, df)

    print("\n=== After backfill ===")
    diagnose(conn)

    if args.dry_run:
        conn.rollback()
        print("\n[DRY RUN] Changes rolled back.")
    else:
        conn.commit()
        print("\nChanges committed.")

    conn.close()


if __name__ == "__main__":
    main()
