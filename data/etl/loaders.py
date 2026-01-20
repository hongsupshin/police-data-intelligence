"""Dataset-specific ETL workflows.

This module contains the main loading functions that transform CSV data into
the normalized database schema. Each loader function handles a complete
transformation pipeline for a specific dataset.

The loaders:
1. Read CSV files using pandas
2. Apply cleaning functions to each field
3. Create master table records (officers, civilians, agencies) with deduplication
4. Create incident records and relationship tables
5. Handle errors gracefully with transaction rollback
"""

from pathlib import Path

import pandas as pd
from psycopg2.extensions import connection

from data.etl.cleaners import (
    clean_boolean,
    clean_date,
    clean_integer,
    clean_text,
    clean_timestamp,
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
            cursor.execute(
                """
                INSERT INTO incidents_civilians_shot (
                    ois_report_no, date_ag_received, date_incident, time_incident,
                    incident_address, incident_city, incident_county, incident_zip,
                    incident_result_of, incident_call_other,
                    weapon_reported_by_media, weapon_reported_by_media_category, deadly_weapon,
                    num_officers_recorded, multiple_officers_involved, officer_on_duty,
                    num_reports_filed, num_rows_about_this_incident,
                    cdr_narrative, custodial_death_report, lea_narrative_published, lea_narrative_shorter
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING incident_id
            """,
                (
                    clean_text(row.get("ois_report_no")),
                    clean_date(row.get("date_ag_received")),
                    clean_date(row.get("date_incident")),
                    clean_text(row.get("time_incident")),
                    clean_text(row.get("incident_address")),
                    clean_text(row.get("incident_city")),
                    clean_text(row.get("incident_county")),
                    clean_text(row.get("incident_zip")),
                    clean_text(row.get("incident_result_of")),
                    clean_text(row.get("incident_call_other")),
                    clean_text(row.get("weapon_reported_by_media")),
                    clean_text(row.get("weapon_reported_by_media_category")),
                    clean_boolean(row.get("deadly_weapon")),
                    clean_integer(row.get("num_officers_recorded")),
                    clean_boolean(row.get("multiple_officers_involved")),
                    clean_boolean(row.get("officer_on_duty")),
                    clean_integer(row.get("num_reports_filed")),
                    clean_integer(row.get("num_rows_about_this_incident")),
                    clean_text(row.get("cdr_narrative")),
                    clean_boolean(row.get("custodial_death_report")),
                    clean_text(row.get("lea_narrative_published")),
                    clean_text(row.get("lea_narrative_shorter")),
                ),
            )

            incident_id = cursor.fetchone()[0]
            incidents_created += 1

            # ----------------------------------------------------------------
            # 2. Create civilian victim record and link to incident
            # ----------------------------------------------------------------
            civilian_id = get_or_create_civilian(
                cursor,
                age=clean_integer(row.get("civilian_age")),
                race=clean_text(row.get("civilian_race")),
                gender=clean_text(row.get("civilian_gender")),
                name_first=clean_text(row.get("civilian_name_first")),
                name_last=clean_text(row.get("civilian_name_last")),
                name_full=clean_text(row.get("civilian_name_full")),
            )

            if civilian_id:
                cursor.execute(
                    """
                    INSERT INTO incident_civilians_shot_victims (incident_id, civilian_id, civilian_died)
                    VALUES (%s, %s, %s)
                """,
                    (incident_id, civilian_id, clean_boolean(row.get("civilian_died"))),
                )

            # ----------------------------------------------------------------
            # 3. Create officer records and link to incident (up to 11 officers)
            # ----------------------------------------------------------------
            for i in range(1, 12):  # Officers 1-11
                # Handle the fact that officer_1 fields might not have the _1 suffix
                if i == 1:
                    age_col = (
                        "officer_age_1" if "officer_age_1" in row else "officer_age"
                    )
                    race_col = (
                        "officer_race_1" if "officer_race_1" in row else "officer_race"
                    )
                    gender_col = (
                        "officer_gender_1"
                        if "officer_gender_1" in row
                        else "officer_gender"
                    )
                else:
                    age_col = f"officer_age_{i}"
                    race_col = f"officer_race_{i}"
                    gender_col = f"officer_gender_{i}"

                officer_id = get_or_create_officer(
                    cursor,
                    age=clean_integer(row.get(age_col)),
                    race=clean_text(row.get(race_col)),
                    gender=clean_text(row.get(gender_col)),
                )

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
            for i in range(1, 12):  # Agencies 1-11
                agency_id = get_or_create_agency(
                    cursor,
                    name=clean_text(row.get(f"agency_name_{i}")),
                    city=clean_text(row.get(f"agency_city_{i}")),
                    county=clean_text(row.get(f"agency_county_{i}")),
                    zip_code=clean_text(row.get(f"agency_zip_{i}")),
                )

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
            cursor.execute(
                """
                INSERT INTO incidents_officers_shot (
                    ois_report_no, date_ag_received, date_incident,
                    incident_address, incident_city, incident_county, incident_zip,
                    num_civilians_recorded, civilian_harm, civilian_suicide
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING incident_id
            """,
                (
                    clean_text(row.get("ois_report_no")),
                    clean_date(row.get("date_ag_received")),
                    clean_timestamp(row.get("date_incident")),
                    clean_text(row.get("incident_address")),
                    clean_text(row.get("incident_city")),
                    clean_text(row.get("incident_county")),
                    clean_text(row.get("incident_zip")),
                    clean_integer(row.get("num_civilians_recorded")),
                    clean_text(row.get("civilian_harm")),
                    clean_boolean(row.get("civilian_suicide")),
                ),
            )

            incident_id = cursor.fetchone()[0]
            incidents_created += 1

            # ----------------------------------------------------------------
            # 2. Create officer victim record and link to incident
            # ----------------------------------------------------------------
            officer_id = get_or_create_officer(
                cursor,
                age=clean_integer(row.get("officer_age")),
                race=clean_text(row.get("officer_race")),
                gender=clean_text(row.get("officer_gender")),
                name_first=clean_text(row.get("officer_name_first")),
                name_last=clean_text(row.get("officer_name_last")),
            )

            if officer_id:
                cursor.execute(
                    """
                    INSERT INTO incident_officers_shot_victims (incident_id, officer_id, officer_harm)
                    VALUES (%s, %s, %s)
                """,
                    (incident_id, officer_id, clean_text(row.get("officer_harm"))),
                )

            # ----------------------------------------------------------------
            # 3. Create civilian shooter records and link to incident (up to 3)
            # ----------------------------------------------------------------
            for i in range(1, 4):  # Civilians 1-3
                civilian_id = get_or_create_civilian(
                    cursor,
                    age=clean_integer(row.get(f"civilian_age_{i}")),
                    race=clean_text(row.get(f"civilian_race_{i}")),
                    gender=clean_text(row.get(f"civilian_gender_{i}")),
                    name_first=clean_text(row.get(f"civilian_name_first_{i}")),
                    name_last=clean_text(row.get(f"civilian_name_last_{i}")),
                )

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
            for i in range(1, 3):  # Agencies 1-2
                agency_id = get_or_create_agency(
                    cursor,
                    name=clean_text(row.get(f"agency_name_{i}")),
                    city=clean_text(row.get(f"agency_city_{i}")),
                    county=clean_text(row.get(f"agency_county_{i}")),
                    zip_code=clean_text(row.get(f"agency_zip_{i}")),
                )

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
