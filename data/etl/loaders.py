"""Dataset-specific ETL workflows.

This module contains the main loading functions that transform CSV data into
the normalized database schema. Each loader function handles a complete
transformation pipeline for a specific dataset.

The loaders:
1. Read CSV files using pandas
2. Apply schema-driven cleaning to each field
3. Create master table records (officers, civilians, agencies) with deduplication
4. Create incident records and relationship tables
5. Handle errors gracefully with transaction rollback

Schema-driven approach:
- Column-to-cleaner mappings defined in config.py
- apply_schema() automatically applies cleaning based on table schema
- Eliminates repetitive clean_* calls and reduces errors
"""

from pathlib import Path

import pandas as pd
from psycopg2.extensions import connection

from data.etl.cleaners import clean_boolean, clean_date, clean_integer, clean_text
from data.etl.config import (
    CIVILIAN_ENTITY_SCHEMA,
    CIVILIANS_SHOT_INCIDENT_SCHEMA,
    OFFICER_ENTITY_SCHEMA,
    OFFICERS_SHOT_INCIDENT_SCHEMA,
    apply_schema,
    clean_entity_fields,
    clean_entity_fields_with_suffix,
)
from data.etl.entity_managers import (
    get_or_create_agency,
    get_or_create_civilian,
    get_or_create_officer,
)


def load_civilians_shot(conn: connection, csv_path: Path) -> tuple[int, int]:
    """Load civilian shooting incident data from CSV into normalized tables.

    Reads a CSV file containing civilian shooting incidents and populates:
    - incidents_civilians_shot: Incident records
    - officers: Officer shooter records (deduplicated)
    - civilians: Civilian victim records (deduplicated)
    - agencies: Law enforcement agencies involved (deduplicated)
    - Junction tables: Linking incidents to officers, civilians, and agencies
    - media_coverage_civilians_shot: Media coverage links

    Args:
        conn: A psycopg2 database connection object.
        csv_path: Path to the CSV file containing civilian shooting data.

    Returns:
        A tuple of (incidents_created, errors) where:
        - incidents_created: Number of incidents successfully inserted.
        - errors: Number of rows that failed to process.
    """
    print(f"Loading civilians_shot data from {csv_path}...")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Read {len(df)} rows from CSV")

    cursor = conn.cursor()
    incidents_created = 0
    errors = 0

    for idx, row in df.iterrows():
        try:
            # ----------------------------------------------------------------
            # 1. Create incident record
            # ----------------------------------------------------------------
            incident_values = apply_schema(row, CIVILIANS_SHOT_INCIDENT_SCHEMA)
            cursor.execute(
                """
                INSERT INTO incidents_civilians_shot (
                    ois_report_no, date_ag_received, date_incident, time_incident,
                    incident_address, incident_city, incident_county, incident_zip,
                    incident_result_of, incident_call_other,
                    weapon_reported_by_media, weapon_reported_by_media_category,
                    deadly_weapon, num_officers_recorded, multiple_officers_involved,
                    officer_on_duty, num_reports_filed, num_rows_about_this_incident,
                    cdr_narrative, custodial_death_report, lea_narrative_published,
                    lea_narrative_shorter
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING incident_id
            """,
                incident_values,
            )

            incident_id = cursor.fetchone()[0]
            incidents_created += 1

            # ----------------------------------------------------------------
            # 2. Create civilian victim record and link to incident
            # ----------------------------------------------------------------
            civilian_fields = clean_entity_fields(
                row, "civilian_", CIVILIAN_ENTITY_SCHEMA
            )
            civilian_id = get_or_create_civilian(cursor, **civilian_fields)

            if civilian_id:
                cursor.execute(
                    """
                    INSERT INTO incident_civilians_shot_victims (
                        incident_id, civilian_id, civilian_died
                    )
                    VALUES (%s, %s, %s)
                """,
                    (incident_id, civilian_id, clean_boolean(row.get("civilian_died"))),
                )

            # ----------------------------------------------------------------
            # 3. Create officer records and link to incident (up to 11 officers)
            # ----------------------------------------------------------------
            # Officer schema for civilians_shot (no names in this dataset)
            officer_basic_schema = [
                ("age", clean_integer),
                ("race", clean_text),
                ("gender", clean_text),
            ]

            for i in range(1, 12):  # Officers 1-11
                # Handle officer_1 fields that might not have _1 suffix
                if i == 1 and "officer_age" in row:
                    officer_fields = clean_entity_fields(
                        row, "officer_", officer_basic_schema
                    )
                else:
                    officer_fields = clean_entity_fields_with_suffix(
                        row, "officer_", f"_{i}", officer_basic_schema
                    )

                officer_id = get_or_create_officer(cursor, **officer_fields)

                if officer_id:
                    caused_injury = (
                        clean_boolean(row.get(f"officer_caused_injury_{i}"))
                        if i > 1
                        else None
                    )
                    cursor.execute(
                        """
                        INSERT INTO incident_civilians_shot_officers_involved
                        (incident_id, officer_id, officer_sequence, caused_injury)
                        VALUES (%s, %s, %s, %s)
                    """,
                        (incident_id, officer_id, i, caused_injury),
                    )

            # ----------------------------------------------------------------
            # 4. Create agency records and link to incident (up to 11 agencies)
            # ----------------------------------------------------------------
            # Agency schema with suffix pattern (CSV: agency_name_{i})
            agency_suffix_schema = [
                ("name", clean_text),
                ("city", clean_text),
                ("county", clean_text),
                ("zip", clean_text),  # CSV uses "zip", will rename to "zip_code"
            ]

            for i in range(1, 12):  # Agencies 1-11
                agency_fields = clean_entity_fields_with_suffix(
                    row, "agency_", f"_{i}", agency_suffix_schema
                )
                # Rename 'zip' to 'zip_code' to match function signature
                agency_fields["zip_code"] = agency_fields.pop("zip")
                agency_id = get_or_create_agency(cursor, **agency_fields)

                if agency_id:
                    cursor.execute(
                        """
                        INSERT INTO incident_civilians_shot_agencies
                        (incident_id, agency_id, agency_sequence, report_date,
                         person_filling_out_name, person_filling_out_email)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                        (
                            incident_id,
                            agency_id,
                            i,
                            clean_date(row.get(f"agency_report_date_{i}")),
                            clean_text(row.get(f"agency_name_person_filling_out_{i}")),
                            clean_text(row.get(f"agency_email_person_filling_out_{i}")),
                        ),
                    )

            # ----------------------------------------------------------------
            # 5. Create media coverage records (up to 4 links)
            # ----------------------------------------------------------------
            for i in range(1, 5):  # Media coverage 1-4
                media_url = clean_text(row.get(f"news_coverage_{i}"))
                if media_url:
                    cursor.execute(
                        """
                        INSERT INTO media_coverage_civilians_shot
                        (incident_id, media_url, coverage_sequence)
                        VALUES (%s, %s, %s)
                    """,
                        (incident_id, media_url, i),
                    )

            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} rows...")
                conn.commit()  # Commit in batches

        except Exception as e:
            errors += 1
            if errors < 10:  # Print first 10 errors
                print(f"  Error on row {idx}: {e}")
            conn.rollback()  # Rollback failed transaction

    conn.commit()
    cursor.close()
    print(f"  ✓ Created {incidents_created} incidents, {errors} errors")
    return incidents_created, errors


def load_officers_shot(conn: connection, csv_path: Path) -> tuple[int, int]:
    """Load officer shooting incident data from CSV into normalized tables.

    Reads a CSV file containing officer shooting incidents and populates:
    - incidents_officers_shot: Incident records
    - officers: Officer victim records (deduplicated)
    - civilians: Civilian shooter records (deduplicated)
    - agencies: Law enforcement agencies involved (deduplicated)
    - Junction tables: Linking incidents to officers, civilians, and agencies
    - media_coverage_officers_shot: Media coverage links

    Args:
        conn: A psycopg2 database connection object.
        csv_path: Path to the CSV file containing officer shooting data.

    Returns:
        A tuple of (incidents_created, errors) where:
        - incidents_created: Number of incidents successfully inserted.
        - errors: Number of rows that failed to process.
    """
    print(f"Loading officers_shot data from {csv_path}...")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Read {len(df)} rows from CSV")

    cursor = conn.cursor()
    incidents_created = 0
    errors = 0

    for idx, row in df.iterrows():
        try:
            # ----------------------------------------------------------------
            # 1. Create incident record
            # ----------------------------------------------------------------
            incident_values = apply_schema(row, OFFICERS_SHOT_INCIDENT_SCHEMA)
            cursor.execute(
                """
                INSERT INTO incidents_officers_shot (
                    ois_report_no, date_ag_received, date_incident,
                    incident_address, incident_city, incident_county, incident_zip,
                    num_civilians_recorded, civilian_harm, civilian_suicide
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING incident_id
            """,
                incident_values,
            )

            incident_id = cursor.fetchone()[0]
            incidents_created += 1

            # ----------------------------------------------------------------
            # 2. Create officer victim record and link to incident
            # ----------------------------------------------------------------
            officer_fields = clean_entity_fields(row, "officer_", OFFICER_ENTITY_SCHEMA)
            officer_id = get_or_create_officer(cursor, **officer_fields)

            if officer_id:
                cursor.execute(
                    """
                    INSERT INTO incident_officers_shot_victims (
                        incident_id, officer_id, officer_harm
                    )
                    VALUES (%s, %s, %s)
                """,
                    (incident_id, officer_id, clean_text(row.get("officer_harm"))),
                )

            # ----------------------------------------------------------------
            # 3. Create civilian shooter records and link to incident (up to 3)
            # ----------------------------------------------------------------
            # Civilian schema for officers_shot (includes names)
            civilian_suffix_schema = [
                ("age", clean_integer),
                ("race", clean_text),
                ("gender", clean_text),
                ("name_first", clean_text),
                ("name_last", clean_text),
            ]

            for i in range(1, 4):  # Civilians 1-3
                civilian_fields = clean_entity_fields_with_suffix(
                    row, "civilian_", f"_{i}", civilian_suffix_schema
                )
                civilian_id = get_or_create_civilian(cursor, **civilian_fields)

                if civilian_id:
                    cursor.execute(
                        """
                        INSERT INTO incident_officers_shot_shooters
                        (incident_id, civilian_id, civilian_sequence)
                        VALUES (%s, %s, %s)
                    """,
                        (incident_id, civilian_id, i),
                    )

            # ----------------------------------------------------------------
            # 4. Create agency records and link to incident (up to 2 agencies)
            # ----------------------------------------------------------------
            # Agency schema with suffix pattern (CSV: agency_name_{i})
            agency_suffix_schema = [
                ("name", clean_text),
                ("city", clean_text),
                ("county", clean_text),
                ("zip", clean_text),  # CSV uses "zip", will rename to "zip_code"
            ]

            for i in range(1, 3):  # Agencies 1-2
                agency_fields = clean_entity_fields_with_suffix(
                    row, "agency_", f"_{i}", agency_suffix_schema
                )
                # Rename 'zip' to 'zip_code' to match function signature
                agency_fields["zip_code"] = agency_fields.pop("zip")
                agency_id = get_or_create_agency(cursor, **agency_fields)

                if agency_id:
                    cursor.execute(
                        """
                        INSERT INTO incident_officers_shot_agencies
                        (incident_id, agency_id, agency_sequence, report_date,
                         person_filling_out_name, person_filling_out_email)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                        (
                            incident_id,
                            agency_id,
                            i,
                            clean_date(row.get(f"agency_report_date_{i}")),
                            clean_text(row.get(f"agency_name_person_filling_out_{i}")),
                            clean_text(row.get(f"agency_email_person_filling_out_{i}")),
                        ),
                    )

            # ----------------------------------------------------------------
            # 5. Create media coverage records (up to 3 links)
            # ----------------------------------------------------------------
            for i in range(1, 4):  # Media coverage 1-3
                media_url = clean_text(row.get(f"media_link_{i}"))
                if media_url:
                    cursor.execute(
                        """
                        INSERT INTO media_coverage_officers_shot
                        (incident_id, media_url, coverage_sequence)
                        VALUES (%s, %s, %s)
                    """,
                        (incident_id, media_url, i),
                    )

            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1} rows...")
                conn.commit()  # Commit in batches

        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"  Error on row {idx}: {e}")
            conn.rollback()

    conn.commit()
    cursor.close()
    print(f"  ✓ Created {incidents_created} incidents, {errors} errors")
    return incidents_created, errors
