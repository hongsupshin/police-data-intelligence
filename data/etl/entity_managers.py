"""Database entity creation with deduplication logic.

This module provides functions for creating and retrieving master table entities
(officers, civilians, agencies) with automatic deduplication using PostgreSQL's
INSERT...ON CONFLICT pattern.

These functions ensure that:
- Duplicate entities are not created in the database
- Existing entity IDs are reused when the same entity appears multiple times
- All entity relationships can be properly linked via returned IDs
"""

from psycopg2.extensions import cursor


def get_or_create_officer(
    cursor: cursor,
    age: int | None,
    race: str | None,
    gender: str | None,
    name_first: str | None = None,
    name_last: str | None = None,
) -> int | None:
    """Get or create an officer record in the database.

    Uses PostgreSQL's INSERT...ON CONFLICT pattern to handle deduplication.
    If an officer with the same name, age, race, and gender already exists,
    the existing officer_id is returned. Otherwise, a new officer is created.

    Args:
        cursor: A psycopg2 database cursor.
        age: Officer's age (int or None).
        race: Officer's race (str or None).
        gender: Officer's gender (str or None).
        name_first: Officer's first name (str or None). Defaults to None.
        name_last: Officer's last name (str or None). Defaults to None.

    Returns:
        The officer_id (int), or None if all parameters are None.
    """
    # All None means no officer data - skip
    if all(v is None for v in [age, race, gender, name_first, name_last]):
        return None

    cursor.execute(
        """
        INSERT INTO officers (age, race, gender, name_first, name_last)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (name_first, name_last, age, race, gender)
        DO UPDATE SET officer_id = officers.officer_id
        RETURNING officer_id
    """,
        (age, race, gender, name_first, name_last),
    )

    return cursor.fetchone()[0]  # type: ignore[no-any-return]


def get_or_create_civilian(
    cursor: cursor,
    age: int | None,
    race: str | None,
    gender: str | None,
    name_first: str | None = None,
    name_last: str | None = None,
    name_full: str | None = None,
) -> int | None:
    """Get or create a civilian record in the database.

    Uses PostgreSQL's INSERT...ON CONFLICT pattern to handle deduplication.
    If a civilian with the same name (first and last), age, race, and gender
    already exists, the existing civilian_id is returned. Otherwise, a new
    civilian is created.

    Args:
        cursor: A psycopg2 database cursor.
        age: Civilian's age (int or None).
        race: Civilian's race (str or None).
        gender: Civilian's gender (str or None).
        name_first: Civilian's first name (str or None). Defaults to None.
        name_last: Civilian's last name (str or None). Defaults to None.
        name_full: Civilian's full name (str or None). Defaults to None.

    Returns:
        The civilian_id (int), or None if all parameters are None.
    """
    # All None means no civilian data - skip
    if all(v is None for v in [age, race, gender, name_first, name_last, name_full]):
        return None

    cursor.execute(
        """
        INSERT INTO civilians (age, race, gender, name_first, name_last, name_full)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (name_first, name_last, age, race, gender)
        DO UPDATE SET civilian_id = civilians.civilian_id
        RETURNING civilian_id
    """,
        (age, race, gender, name_first, name_last, name_full),
    )

    return cursor.fetchone()[0]  # type: ignore[no-any-return]


def get_or_create_agency(
    cursor: cursor,
    name: str | None,
    city: str | None,
    county: str | None,
    zip_code: str | None = None,
) -> int | None:
    """Get or create an agency record in the database.

    Uses PostgreSQL's INSERT...ON CONFLICT pattern to handle deduplication.
    If an agency with the same name, city, and county already exists, the
    existing agency_id is returned. Otherwise, a new agency is created.

    Args:
        cursor: A psycopg2 database cursor.
        name: Agency name (str or None).
        city: City where agency is located (str or None).
        county: County where agency is located (str or None).
        zip_code: Zip code (str or None). Defaults to None.

    Returns:
        The agency_id (int), or None if name, city, and county are all None.
    """
    # All None means no agency data - skip
    if all(v is None for v in [name, city, county]):
        return None

    cursor.execute(
        """
        INSERT INTO agencies (name, city, county, zip)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name, city, county)
        DO UPDATE SET agency_id = agencies.agency_id
        RETURNING agency_id
    """,
        (name, city, county, zip_code),
    )

    return cursor.fetchone()[0]  # type: ignore[no-any-return]
