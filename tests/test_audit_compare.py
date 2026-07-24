"""Tests for src/audit/compare.py."""

from src.agents.state import (
    ConfidenceLevel,
    DatasetType,
    FieldExtraction,
    MediaFeatureField,
)
from src.audit.compare import (
    audit_fields_for_dataset,
    compare_record,
)
from src.eval.comparators import EvalError, compare_name


def _extraction(
    field: MediaFeatureField,
    value: str | None,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> FieldExtraction:
    return FieldExtraction(
        field_name=field.value,
        value=value,
        confidence=confidence,
        sources=["https://example.com/article"],
        source_quotes=["quote"],
    )


class TestAuditFieldsForDataset:
    def test_civilians_fields(self):
        fields = audit_fields_for_dataset(DatasetType.CIVILIANS_SHOT)
        assert set(fields) == {
            MediaFeatureField.CIVILIAN_AGE,
            MediaFeatureField.CIVILIAN_RACE,
            MediaFeatureField.WEAPON,
            MediaFeatureField.LOCATION_DETAIL,
            MediaFeatureField.TIME_OF_DAY,
            MediaFeatureField.OUTCOME,
            MediaFeatureField.OFFICER_NAME,
            MediaFeatureField.CIVILIAN_NAME,
        }

    def test_officers_fields_exclude_weapon_and_time(self):
        fields = audit_fields_for_dataset(DatasetType.OFFICERS_SHOT)
        assert MediaFeatureField.WEAPON not in fields
        assert MediaFeatureField.TIME_OF_DAY not in fields
        assert fields[MediaFeatureField.OFFICER_NAME] is compare_name
        assert fields[MediaFeatureField.CIVILIAN_NAME] is compare_name

    def test_circumstance_never_audited(self):
        for dataset in DatasetType:
            assert MediaFeatureField.CIRCUMSTANCE not in audit_fields_for_dataset(
                dataset
            )


class TestCompareRecord:
    def test_contradiction_emits_flag(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(MediaFeatureField.CIVILIAN_NAME, "John Smith")
            ],
            reference={"civilian_name": "Michael Brown"},
        )
        assert len(audit.flags) == 1
        assert audit.flags[0].field == "civilian_name"
        assert audit.flags[0].db_value == "Michael Brown"
        assert audit.flags[0].news_value == "John Smith"

    def test_agreement_emits_no_flag(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(MediaFeatureField.CIVILIAN_NAME, "John Smith")
            ],
            reference={"civilian_name": "John Smith"},
        )
        assert audit.flags == []

    def test_fuzzy_agreement_emits_no_flag(self):
        # Surname-only news name is consistent, not contradictory.
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(MediaFeatureField.OFFICER_NAME, "Sgt. Duran")
            ],
            reference={"officer_name": "John Duran"},
        )
        assert audit.flags == []

    def test_missing_db_value_emits_no_flag(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(MediaFeatureField.CIVILIAN_NAME, "John Smith")
            ],
            reference={"civilian_name": None},
        )
        assert audit.flags == []
        name_result = next(
            r for r in audit.match_results if r.field_name == "civilian_name"
        )
        assert name_result.error == EvalError.NO_GROUND_TRUTH

    def test_missing_extraction_emits_no_flag(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[],
            reference={"civilian_name": "Michael Brown"},
        )
        assert audit.flags == []

    def test_circumstance_extraction_is_ignored(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(MediaFeatureField.CIRCUMSTANCE, "traffic stop")
            ],
            reference={"circumstance": "welfare check"},
        )
        assert audit.flags == []
        assert all(r.field_name != "circumstance" for r in audit.match_results)

    def test_low_confidence_flag_is_suppressed_not_dropped(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(
                    MediaFeatureField.CIVILIAN_NAME,
                    "John Smith",
                    ConfidenceLevel.LOW,
                )
            ],
            reference={"civilian_name": "Michael Brown"},
        )
        assert len(audit.flags) == 1
        assert audit.flags[0].suppressed is True

    def test_dict_extractions_from_saved_artifacts(self):
        audit = compare_record(
            incident_id=1031,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                {
                    "field_name": "civilian_name",
                    "value": "John Smith",
                    "confidence": "high",
                    "sources": ["https://example.com"],
                    "source_quotes": ["q"],
                    "extraction_method": "llm",
                }
            ],
            reference={"civilian_name": "Michael Brown"},
        )
        assert len(audit.flags) == 1

    def test_match_results_include_agreements(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(MediaFeatureField.CIVILIAN_NAME, "John Smith"),
                _extraction(MediaFeatureField.CIVILIAN_AGE, "34"),
            ],
            reference={"civilian_name": "John Smith", "civilian_age": 34},
        )
        assert audit.flags == []
        matched = [r for r in audit.match_results if r.exact_match]
        assert len(matched) == 2

    def test_confidence_attached_to_match_results(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(
                    MediaFeatureField.CIVILIAN_AGE, "34", ConfidenceLevel.MEDIUM
                )
            ],
            reference={"civilian_age": 34},
        )
        age_result = next(
            r for r in audit.match_results if r.field_name == "civilian_age"
        )
        assert age_result.confidence == ConfidenceLevel.MEDIUM

    def test_reference_source_propagates_to_flags(self):
        audit = compare_record(
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            extracted_fields=[
                _extraction(MediaFeatureField.CIVILIAN_NAME, "John Smith")
            ],
            reference={"civilian_name": "Michael Brown"},
            reference_source="oag_xlsx",
        )
        assert audit.reference_source == "oag_xlsx"
        assert audit.flags[0].reference_source == "oag_xlsx"
