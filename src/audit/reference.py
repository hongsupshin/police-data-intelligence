"""Reference-record providers for the discrepancy audit.

A reference provider fetches the official record the audit compares news
extractions against. In the audit's semantics the record is the *audit
object*, not ground truth — the pipeline's holdout machinery calls the same
values "ground truth" because there they grade extraction accuracy.

The provider protocol keeps the comparison layer pluggable: a future
raw-OAG provider (data.world ``original/OIS.xlsx``, joined on the
``ois_report_no`` column both incident tables already carry) only needs to
return the same field dict and nothing downstream changes.
"""

from typing import Protocol

from psycopg2.extensions import connection

from src.agents.load_node import fetch_incident
from src.agents.state import DatasetType, MediaFeatureField
from src.eval.comparators import fetch_ground_truth

TJI_DB_SOURCE = "tji_db"


class ReferenceProvider(Protocol):
    """Fetches the official record for one incident.

    Attributes:
        source_name: Identifier stored on flags as ``reference_source``.
    """

    source_name: str

    def fetch(
        self, conn: connection, incident_id: int, dataset_type: DatasetType
    ) -> dict[str, object]:
        """Fetch the official record's auditable field values.

        Args:
            conn: Active PostgreSQL connection.
            incident_id: TJI incident identifier.
            dataset_type: Which dataset to query.

        Returns:
            Dict mapping MediaFeatureField values to official record
            values (None where the record is empty).
        """
        ...


class TjiDbReferenceProvider:
    """Official record fetch from the TJI PostgreSQL database.

    Composes the holdout eval's field fetch (age/race/weapon/location/
    time/outcome) with the name fields the pipeline's load node already
    joins (officer_name/civilian_name as "First Last" strings). No new SQL.

    Note: for officers_shot, ``civilian_name``/``civilian_age``/
    ``civilian_race`` describe the civilian *shooter* and ``officer_name``
    is the shot officer — flag readers must interpret accordingly.
    """

    source_name = TJI_DB_SOURCE

    def fetch(
        self, conn: connection, incident_id: int, dataset_type: DatasetType
    ) -> dict[str, object]:
        """Fetch the official record's auditable field values.

        Args:
            conn: Active PostgreSQL connection.
            incident_id: TJI incident identifier.
            dataset_type: Which dataset to query.

        Returns:
            Dict mapping MediaFeatureField values to DB values, covering
            the holdout fields plus officer_name/civilian_name.

        Raises:
            KeyError: If incident_id is not found in the database.
        """
        record = fetch_ground_truth(conn, incident_id, dataset_type)
        incident = fetch_incident(conn, incident_id, dataset_type)
        record[MediaFeatureField.OFFICER_NAME.value] = incident.get("officer_name")
        record[MediaFeatureField.CIVILIAN_NAME.value] = incident.get("civilian_name")
        return record
