"""Extract Node for the enrichment pipeline.

Fetches incident records from the PostgreSQL database and populates
the EnrichmentState with baseline data for downstream processing.

This is a deterministic node - no LLM calls, just database queries
and field mapping.
"""

from datetime import date

from psycopg2.extensions import connection

from src.agents.state import DatasetType, EnrichmentState, PipelineStage
from src.database.connection import get_connection


def fetch_incident(
    conn: connection, incident_id: int, dataset_type: DatasetType
) -> dict[str, str | date | None]:
    """Fetch incident record from database with dataset-aware queries.

    Queries the appropriate incident table based on dataset_type and
    JOINs with related entity tables to get officer and civilian names.

    Args:
        conn: Active PostgreSQL database connection.
        incident_id: Numeric incident ID from the database.
        dataset_type: Which dataset to query (civilians_shot or officers_shot).

    Returns:
        Dictionary with extracted fields:
        - officer_name: First officer name found (may be None)
        - civilian_name: First civilian name found (may be None)
        - incident_date: Date of incident
        - location: City where incident occurred
        - severity: Outcome description (e.g., "fatal", "injured")

    Raises:
        psycopg2.Error: If database query fails.
        KeyError: If incident_id not found in database.
    """
    cursor = conn.cursor()

    if dataset_type == DatasetType.CIVILIANS_SHOT:
        # Query for civilians_shot incidents
        # Gets first officer (shooter) and civilian (victim) names via JOINs
        query = """
            SELECT
                i.date_incident,
                i.incident_city,
                i.incident_county,
                o.name_first AS officer_first,
                o.name_last AS officer_last,
                c.name_first AS civilian_first,
                c.name_last AS civilian_last,
                v.civilian_died
            FROM incidents_civilians_shot i
            LEFT JOIN incident_civilians_shot_officers_involved oi
                ON i.incident_id = oi.incident_id AND oi.officer_sequence = 1
            LEFT JOIN officers o ON oi.officer_id = o.officer_id
            LEFT JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN civilians c ON v.civilian_id = c.civilian_id
            WHERE i.incident_id = %s
            LIMIT 1;
        """
        cursor.execute(query, (incident_id,))
        row = cursor.fetchone()

        if not row:
            raise KeyError(f"Incident {incident_id} not found in civilians_shot")

        (
            incident_date,
            city,
            county,
            officer_first,
            officer_last,
            civilian_first,
            civilian_last,
            civilian_died,
        ) = row

        # Build full names
        officer_name = None
        if officer_first or officer_last:
            parts = [p for p in [officer_first, officer_last] if p]
            officer_name = " ".join(parts) if parts else None

        civilian_name = None
        if civilian_first or civilian_last:
            parts = [p for p in [civilian_first, civilian_last] if p]
            civilian_name = " ".join(parts) if parts else None

        # Map boolean outcome to severity
        # civilian_died: True=DEATH, False=INJURY, None=unknown
        if civilian_died is True:
            severity = "fatal"
        elif civilian_died is False:
            severity = "non-fatal"
        else:
            severity = "unknown"

    else:  # DatasetType.OFFICERS_SHOT
        # Query for officers_shot incidents
        # Gets first officer (victim) and civilian (shooter) names via JOINs
        query = """
            SELECT
                i.date_incident::date,
                i.incident_city,
                i.incident_county,
                o.name_first AS officer_first,
                o.name_last AS officer_last,
                c.name_first AS civilian_first,
                c.name_last AS civilian_last,
                v.officer_harm
            FROM incidents_officers_shot i
            LEFT JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN officers o ON v.officer_id = o.officer_id
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id AND s.civilian_sequence = 1
            LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
            WHERE i.incident_id = %s
            LIMIT 1;
        """
        cursor.execute(query, (incident_id,))
        row = cursor.fetchone()

        if not row:
            raise KeyError(f"Incident {incident_id} not found in officers_shot")

        (
            incident_date,
            city,
            county,
            officer_first,
            officer_last,
            civilian_first,
            civilian_last,
            officer_harm,
        ) = row

        # Build full names
        officer_name = None
        if officer_first or officer_last:
            parts = [p for p in [officer_first, officer_last] if p]
            officer_name = " ".join(parts) if parts else None

        civilian_name = None
        if civilian_first or civilian_last:
            parts = [p for p in [civilian_first, civilian_last] if p]
            civilian_name = " ".join(parts) if parts else None

        # Map outcome text to severity
        if officer_harm == "DEATH":
            severity = "fatal"
        elif officer_harm == "INJURY":
            severity = "non-fatal"
        else:
            severity = "unknown"

    cursor.close()

    # Determine location (prefer city, fallback to county)
    location = city if city else county

    return {
        "officer_name": officer_name,
        "civilian_name": civilian_name,
        "incident_date": incident_date,
        "location": location,
        "severity": severity,
    }


def extract_node(state: EnrichmentState) -> EnrichmentState:
    """Extract incident data from database and populate state.

    Fetches a single incident record from the TJI database using the
    incident_id and dataset_type (civilian vs. officer) set in the state.
    Performs dataset-aware field mapping since the two TJI datasets use
    different column names for the same concepts.

    The function expects state.incident_id and state.dataset_type to be
    pre-populated by the batch orchestrator.

    Args:
        state: Current enrichment state with incident_id and dataset_type set.

    Returns:
        Updated EnrichmentState with incident fields populated:
        - officer_name: Officer name from database (may be None)
        - civilian_name: Civilian name from database (may be None)
        - incident_date: Date of the incident
        - location: City/county where incident occurred
        - severity: Outcome severity (fatal, injured, etc.)
        - current_stage: Set to PipelineStage.EXTRACT
        - error_message: Set if extraction fails

    Examples:
        >>> state = EnrichmentState(
        ...     incident_id="123",
        ...     dataset_type=DatasetType.CIVILIANS_SHOT
        ... )
        >>> updated_state = extract_node(state)
        >>> print(updated_state.location)
        'Houston'
        >>> print(updated_state.current_stage)
        <PipelineStage.EXTRACT: 'extract'>
    """
    try:
        # Get database connection
        conn = get_connection()

        # Convert incident_id from string to int
        incident_id_int = int(state.incident_id)

        # Fetch incident data from database
        incident_data = fetch_incident(conn, incident_id_int, state.dataset_type)

        # Update state with extracted fields
        state.officer_name = incident_data["officer_name"]
        state.civilian_name = incident_data["civilian_name"]
        state.incident_date = incident_data["incident_date"]
        state.location = incident_data["location"]
        state.severity = incident_data["severity"]

        # Update pipeline stage
        state.current_stage = PipelineStage.EXTRACT

        # Close connection
        conn.close()

    except (ValueError, KeyError, Exception) as e:
        # Handle errors and populate error_message
        state.error_message = f"Extract failed: {str(e)}"
        state.current_stage = PipelineStage.EXTRACT

    return state
