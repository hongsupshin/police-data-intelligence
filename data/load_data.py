#!/usr/bin/env python3
"""ETL script to load Texas Justice Initiative (TJI) police data into PostgreSQL.

This module transforms wide-format CSV data (with repeating columns) into a
normalized relational database schema. It handles loading two datasets:
- Civilians shot by police (incidents_civilians_shot)
- Officers shot by civilians (incidents_officers_shot)

The script deduplicates records for officers, civilians, and agencies using
PostgreSQL's INSERT...ON CONFLICT pattern.

Usage:
    python load_data.py
"""

import sys
from pathlib import Path

import psycopg2

# Handle both direct execution and module execution
try:
    from data.etl.config import DB_CONFIG
    from data.etl.loaders import load_civilians_shot, load_officers_shot
except ModuleNotFoundError:
    # When running as script (python data/load_data.py), add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.etl.config import DB_CONFIG
    from data.etl.loaders import load_civilians_shot, load_officers_shot


# ============================================================================
# Main Script
# ============================================================================


def main() -> None:
    """Main entry point for the ETL script.

    Orchestrates the entire data loading process:
    1. Connects to PostgreSQL database
    2. Loads and executes schema.sql to create tables
    3. Loads civilian shooting data from CSV
    4. Loads officer shooting data from CSV
    5. Prints summary statistics of loaded data

    Exits with code 1 if CSV files or database are not found.
    """
    # Paths
    data_dir = Path(__file__).parent
    schema_file = data_dir / "schema.sql"
    civilians_csv = data_dir / "tji_civilians-shot.csv"
    officers_csv = data_dir / "tji_officers-shot.csv"

    # Check files exist
    if not civilians_csv.exists():
        print(f"Error: {civilians_csv} not found")
        sys.exit(1)
    if not officers_csv.exists():
        print(f"Error: {officers_csv} not found")
        sys.exit(1)

    # Connect to database
    print("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("\nMake sure PostgreSQL is running and database exists:")
        print("  createdb tji_police_data")
        sys.exit(1)

    # Load schema
    if schema_file.exists():
        print(f"\nLoading schema from {schema_file}...")
        with open(schema_file) as f:
            schema_sql = f.read()
        cursor = conn.cursor()
        try:
            cursor.execute(schema_sql)
            conn.commit()
            print("✓ Schema loaded")
        except Exception as e:
            print(f"Error loading schema: {e}")
            conn.rollback()
            sys.exit(1)
        finally:
            cursor.close()

    # Load data
    print("\n" + "=" * 60)
    civilians_incidents, civilians_errors = load_civilians_shot(conn, civilians_csv)

    print("\n" + "=" * 60)
    officers_incidents, officers_errors = load_officers_shot(conn, officers_csv)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    cursor = conn.cursor()

    # Count records in each table
    cursor.execute("SELECT COUNT(*) FROM incidents_civilians_shot")
    civ_incidents = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents_officers_shot")
    off_incidents = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM officers")
    officers_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM civilians")
    civilians_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM agencies")
    agencies_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incident_civilians_shot_officers_involved")
    civ_shot_off_links = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incident_officers_shot_shooters")
    off_shot_civ_links = cursor.fetchone()[0]

    cursor.close()

    print("\nIncidents:")
    print(f"  Civilians shot by police:  {civ_incidents} ({civilians_errors} errors)")
    print(f"  Officers shot by civilians: {off_incidents} ({officers_errors} errors)")
    print(f"  Total incidents:            {civ_incidents + off_incidents}")

    print("\nMaster Tables (deduplicated):")
    print(f"  Unique officers:   {officers_count}")
    print(f"  Unique civilians:  {civilians_count}")
    print(f"  Unique agencies:   {agencies_count}")

    print("\nRelationships:")
    print(f"  Officer involvements in civilian shootings: {civ_shot_off_links}")
    print(f"  Civilian involvements in officer shootings:  {off_shot_civ_links}")

    conn.close()
    print("\n✓ Done! Data loaded into normalized schema.")


if __name__ == "__main__":
    main()
