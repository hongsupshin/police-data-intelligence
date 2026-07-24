"""Record-level comparison and flag emission for the discrepancy audit.

Deterministic layer — no LLM calls. Runs the audit comparators over one
incident's extractions against the official record and emits a
DiscrepancyFlag for every genuine contradiction (both values present,
neither exact nor fuzzy match, no comparison error).

Circumstance is deliberately not compared: it is free text on the news
side and only semi-structured in the DB (``incident_result_of``,
narratives — civilians only), so a deterministic comparator would be
noise and an LLM entailment judge would be a new unvalidated judge. Fields
absent from AUDIT_FIELD_COMPARATORS are simply skipped, which is the
extension point if a validated comparator is ever earned.
"""

from collections.abc import Callable

from pydantic import BaseModel, Field

from src.agents.state import DatasetType, FieldExtraction, MediaFeatureField
from src.audit.flags import DiscrepancyFlag, build_flag
from src.audit.reference import TJI_DB_SOURCE
from src.eval.comparators import (
    MatchResult,
    comparators_for_dataset,
    compare_name,
)


class RecordAudit(BaseModel):
    """Audit outcome for one incident's field comparisons.

    Attributes:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset the incident belongs to.
        reference_source: Which official record was compared against.
        flags: Contradiction flags (including suppressed ones).
        match_results: All field comparisons, agreements included —
            needed for specificity measurement.
    """

    incident_id: int
    dataset_type: DatasetType
    reference_source: str = TJI_DB_SOURCE
    flags: list[DiscrepancyFlag] = Field(default_factory=list)
    match_results: list[MatchResult] = Field(default_factory=list)


def audit_fields_for_dataset(
    dataset_type: DatasetType,
) -> dict[MediaFeatureField, Callable]:
    """Return the audit comparators applicable to a dataset.

    The holdout comparators for the dataset (officers_shot has no
    weapon/time columns — structural, per schema.sql) extended with the
    person-name comparators, which apply to both datasets.

    Args:
        dataset_type: Which dataset is being audited.

    Returns:
        Mapping of auditable fields to comparator functions.
    """
    comparators = dict(comparators_for_dataset(dataset_type))
    comparators[MediaFeatureField.OFFICER_NAME] = compare_name
    comparators[MediaFeatureField.CIVILIAN_NAME] = compare_name
    return comparators


def _extraction_lookup(
    extracted_fields: list[FieldExtraction | dict],
) -> dict[str, FieldExtraction]:
    """Index extractions by field name, accepting models or raw dicts.

    Args:
        extracted_fields: Pipeline extractions (FieldExtraction instances
            from a live run, or dicts parsed from a saved JSON artifact).

    Returns:
        Mapping of field name to FieldExtraction.
    """
    lookup: dict[str, FieldExtraction] = {}
    for field in extracted_fields:
        if isinstance(field, FieldExtraction):
            lookup[field.field_name] = field
        elif isinstance(field, dict):
            lookup[field["field_name"]] = FieldExtraction(**field)
    return lookup


def compare_record(
    incident_id: int,
    dataset_type: DatasetType,
    extracted_fields: list[FieldExtraction | dict],
    reference: dict[str, object],
    reference_source: str = TJI_DB_SOURCE,
) -> RecordAudit:
    """Compare one incident's extractions against the official record.

    A flag is emitted iff the comparator finds a genuine contradiction:
    both values present (no error), and neither an exact nor a fuzzy
    match. Missing DB values and missing extractions produce comparison
    errors, not flags.

    Args:
        incident_id: TJI incident identifier.
        dataset_type: Which dataset the incident belongs to.
        extracted_fields: Pipeline extractions for the incident.
        reference: Official record values keyed by MediaFeatureField value.
        reference_source: Which official record is being compared against.

    Returns:
        RecordAudit with flags and all match results (agreements included).
    """
    lookup = _extraction_lookup(extracted_fields)

    flags: list[DiscrepancyFlag] = []
    match_results: list[MatchResult] = []
    for field_enum, comparator in audit_fields_for_dataset(dataset_type).items():
        reference_value = reference.get(field_enum.value)
        extraction = lookup.get(field_enum.value)
        extracted_value = extraction.value if extraction else None

        match = comparator(extracted_value, reference_value, field_enum.value)
        if extraction and match.error is None:
            match.confidence = extraction.confidence
        match_results.append(match)

        is_contradiction = (
            match.error is None and not match.exact_match and not match.fuzzy_match
        )
        if is_contradiction and extraction is not None:
            flags.append(
                build_flag(
                    incident_id=incident_id,
                    dataset_type=dataset_type,
                    field=field_enum,
                    match=match,
                    extraction=extraction,
                    reference_source=reference_source,
                )
            )

    return RecordAudit(
        incident_id=incident_id,
        dataset_type=dataset_type,
        reference_source=reference_source,
        flags=flags,
        match_results=match_results,
    )
