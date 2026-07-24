"""Stratified sampling of audit-ready incidents.

Inverts the holdout sampler's purpose: the audit needs incidents whose
official records are COMPLETE (every auditable column populated, including
both person names), because a missing DB value cannot be contradicted.

Two further restrictions:
    - Single-victim incidents only. The reference fetch uses ``LIMIT 1``
      joins, so multi-victim incidents would compare news coverage against
      an arbitrary victim and produce spurious name/age/race flags (the
      offline gate confirmed this failure mode on saved artifacts).
    - DEV and frozen TEST ids are excluded. Pilot threshold iteration is
      tuning, and tuning must never touch the held-out eval splits.

Officer names are structurally absent from civilians_shot (the OAG
civilian-shot report form records officer age/race/gender but no names,
so the officers table holds none for that dataset — verified 0/1,674).
officer_name is therefore unauditable there (its comparison yields
NO_GROUND_TRUTH, never a flag) and is not a completeness requirement.
Clean-DB pool sizes with these predicates: 506 civilians / 151 officers.
"""

from collections import defaultdict

from psycopg2.extensions import connection
from pydantic import BaseModel

from src.agents.state import DatasetType
from src.eval.holdout import _excluded_ids, allocate_by_stratum


class AuditSample(BaseModel):
    """Metadata for one sampled audit incident.

    Attributes:
        incident_id: TJI incident identifier.
        year: Year of the incident (stratification key).
        race: Race of the civilian on the record (fairness metadata).
    """

    incident_id: int
    year: int
    race: str | None = None


def select_audit_incidents(
    conn: connection,
    dataset_type: DatasetType,
    limit: int,
    exclude_ids: set[int] | None = None,
) -> list[AuditSample]:
    """Select complete, single-victim incidents with year stratification.

    Requires every auditable-and-populatable column non-NULL (the holdout
    fields for the dataset plus the person names its records can carry —
    civilian name for civilians_shot; officer and shooter names for
    officers_shot) and exactly one victim row, then allocates
    proportionally across years via the shared stratum allocator.

    Args:
        conn: Active PostgreSQL connection.
        dataset_type: Which dataset to sample.
        limit: Total number of incidents to return.
        exclude_ids: Incident ids to exclude. Defaults to DEV ∪ TEST via
            the holdout exclusion helper.

    Returns:
        List of AuditSample with year/race metadata.
    """
    cursor = conn.cursor()

    excluded = _excluded_ids(dataset_type) if exclude_ids is None else exclude_ids
    exclusion_clause = ""
    if excluded:
        ids_str = ", ".join(str(i) for i in sorted(excluded))
        exclusion_clause = f"AND i.incident_id NOT IN ({ids_str})"

    if dataset_type == DatasetType.CIVILIANS_SHOT:
        query = f"""
            SELECT i.incident_id,
                   EXTRACT(YEAR FROM i.date_incident)::int AS year,
                   c.race
            FROM incidents_civilians_shot i
            JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            JOIN civilians c ON v.civilian_id = c.civilian_id
            WHERE i.date_incident IS NOT NULL
              AND i.incident_city IS NOT NULL
              AND i.weapon_reported_by_media IS NOT NULL
              AND i.time_incident IS NOT NULL
              AND c.age IS NOT NULL
              AND c.race IS NOT NULL
              AND c.name_first IS NOT NULL
              AND c.name_last IS NOT NULL
              AND v.civilian_died IS NOT NULL
              {exclusion_clause}
              AND (SELECT COUNT(*) FROM incident_civilians_shot_victims v2
                   WHERE v2.incident_id = i.incident_id) = 1
            ORDER BY i.incident_id ASC;
        """
    else:
        query = f"""
            SELECT i.incident_id,
                   EXTRACT(YEAR FROM i.date_incident)::int AS year,
                   c.race
            FROM incidents_officers_shot i
            JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            JOIN officers o ON v.officer_id = o.officer_id
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id AND s.civilian_sequence = 1
            LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
            WHERE i.date_incident IS NOT NULL
              AND i.incident_city IS NOT NULL
              AND v.officer_harm IS NOT NULL
              AND c.age IS NOT NULL
              AND c.race IS NOT NULL
              AND c.name_first IS NOT NULL
              AND c.name_last IS NOT NULL
              AND o.name_first IS NOT NULL
              AND o.name_last IS NOT NULL
              {exclusion_clause}
              AND (SELECT COUNT(*) FROM incident_officers_shot_victims v2
                   WHERE v2.incident_id = i.incident_id) = 1
            ORDER BY i.incident_id ASC;
        """

    cursor.execute(query)
    rows = cursor.fetchall()

    by_year: dict[int, list[tuple]] = defaultdict(list)
    for row in rows:
        by_year[row[1]].append(row)

    return [
        AuditSample(incident_id=row[0], year=row[1], race=row[2])
        for row in allocate_by_stratum(by_year, limit)
    ]
