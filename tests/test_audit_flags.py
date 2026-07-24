"""Tests for src/audit/flags.py."""

import pytest

from src.agents.state import (
    ConfidenceLevel,
    DatasetType,
    FieldExtraction,
    MediaFeatureField,
)
from src.audit.flags import (
    SEVERITY_BY_FIELD,
    DiscrepancyFlag,
    FlagSeverity,
    VerificationStatus,
    assign_severity,
    build_flag,
    make_flag_id,
)
from src.eval.comparators import MatchResult


def _match(
    field: MediaFeatureField,
    extracted: str | None = "news value",
    ground_truth: str | None = "db value",
) -> MatchResult:
    return MatchResult(
        field_name=field.value,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )


def _extraction(
    field: MediaFeatureField,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> FieldExtraction:
    return FieldExtraction(
        field_name=field.value,
        value="news value",
        confidence=confidence,
        sources=["https://example.com/article"],
        source_quotes=["quoted evidence"],
    )


class TestSeverityScheme:
    def test_all_audit_fields_have_severity(self):
        # Every field with a comparator must map to a severity.
        from src.eval.comparators import AUDIT_FIELD_COMPARATORS

        assert set(SEVERITY_BY_FIELD) == set(AUDIT_FIELD_COMPARATORS)

    @pytest.mark.parametrize(
        "field, expected",
        [
            (MediaFeatureField.OFFICER_NAME, FlagSeverity.HIGH),
            (MediaFeatureField.CIVILIAN_NAME, FlagSeverity.HIGH),
            (MediaFeatureField.OUTCOME, FlagSeverity.HIGH),
            (MediaFeatureField.CIVILIAN_RACE, FlagSeverity.HIGH),
            (MediaFeatureField.WEAPON, FlagSeverity.MEDIUM),
            (MediaFeatureField.TIME_OF_DAY, FlagSeverity.MEDIUM),
            (MediaFeatureField.LOCATION_DETAIL, FlagSeverity.LOW),
        ],
    )
    def test_base_severity(self, field, expected):
        match = _match(field)
        assert assign_severity(field, match, DatasetType.CIVILIANS_SHOT) == expected

    def test_age_off_by_one_is_low(self):
        match = _match(MediaFeatureField.CIVILIAN_AGE, "34", "35")
        severity = assign_severity(
            MediaFeatureField.CIVILIAN_AGE, match, DatasetType.CIVILIANS_SHOT
        )
        assert severity == FlagSeverity.LOW

    def test_age_off_by_five_is_medium(self):
        match = _match(MediaFeatureField.CIVILIAN_AGE, "30", "35")
        severity = assign_severity(
            MediaFeatureField.CIVILIAN_AGE, match, DatasetType.CIVILIANS_SHOT
        )
        assert severity == FlagSeverity.MEDIUM

    def test_age_unparseable_keeps_base_severity(self):
        match = _match(MediaFeatureField.CIVILIAN_AGE, "thirties", "35")
        severity = assign_severity(
            MediaFeatureField.CIVILIAN_AGE, match, DatasetType.CIVILIANS_SHOT
        )
        assert severity == FlagSeverity.MEDIUM

    def test_officers_race_demoted_to_medium(self):
        # Race verifier only runs on civilians_shot; officers race is unverified.
        match = _match(MediaFeatureField.CIVILIAN_RACE, "WHITE", "BLACK")
        severity = assign_severity(
            MediaFeatureField.CIVILIAN_RACE, match, DatasetType.OFFICERS_SHOT
        )
        assert severity == FlagSeverity.MEDIUM

    def test_civilians_race_stays_high(self):
        match = _match(MediaFeatureField.CIVILIAN_RACE, "WHITE", "BLACK")
        severity = assign_severity(
            MediaFeatureField.CIVILIAN_RACE, match, DatasetType.CIVILIANS_SHOT
        )
        assert severity == FlagSeverity.HIGH


class TestMakeFlagId:
    def test_deterministic_format(self):
        flag_id = make_flag_id(
            DatasetType.CIVILIANS_SHOT, 1031, MediaFeatureField.OFFICER_NAME
        )
        assert flag_id == "civilians_shot_1031_officer_name"

    def test_distinct_across_datasets(self):
        civ = make_flag_id(
            DatasetType.CIVILIANS_SHOT, 1, MediaFeatureField.OUTCOME
        )
        off = make_flag_id(
            DatasetType.OFFICERS_SHOT, 1, MediaFeatureField.OUTCOME
        )
        assert civ != off


class TestBuildFlag:
    def test_populates_values_and_provenance(self):
        field = MediaFeatureField.OFFICER_NAME
        match = _match(field, "John Smith", "Michael Brown")
        match.fuzzy_score = 41.0
        flag = build_flag(
            incident_id=1031,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            field=field,
            match=match,
            extraction=_extraction(field),
            reference_source="tji_db",
        )
        assert flag.flag_id == "civilians_shot_1031_officer_name"
        assert flag.db_value == "Michael Brown"
        assert flag.news_value == "John Smith"
        assert flag.sources == ["https://example.com/article"]
        assert flag.source_quotes == ["quoted evidence"]
        assert flag.fuzzy_score == 41.0
        assert flag.severity == FlagSeverity.HIGH
        assert flag.reference_source == "tji_db"
        assert flag.verification_status == VerificationStatus.PENDING

    @pytest.mark.parametrize(
        "confidence, suppressed",
        [
            (ConfidenceLevel.HIGH, False),
            (ConfidenceLevel.MEDIUM, False),
            (ConfidenceLevel.LOW, True),
            (ConfidenceLevel.NONE, True),
        ],
    )
    def test_suppression_gated_by_confidence(self, confidence, suppressed):
        field = MediaFeatureField.WEAPON
        flag = build_flag(
            incident_id=5,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            field=field,
            match=_match(field),
            extraction=_extraction(field, confidence),
            reference_source="tji_db",
        )
        assert flag.suppressed is suppressed

    def test_reference_source_propagates(self):
        field = MediaFeatureField.WEAPON
        flag = build_flag(
            incident_id=5,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            field=field,
            match=_match(field),
            extraction=_extraction(field),
            reference_source="oag_xlsx",
        )
        assert flag.reference_source == "oag_xlsx"


class TestDiscrepancyFlagModel:
    def test_round_trips_through_json(self):
        flag = DiscrepancyFlag(
            flag_id="civilians_shot_1_weapon",
            incident_id=1,
            dataset_type=DatasetType.CIVILIANS_SHOT,
            field="weapon",
            db_value="HANDGUN",
            news_value="rifle",
            extraction_confidence=ConfidenceLevel.HIGH,
            severity=FlagSeverity.MEDIUM,
        )
        restored = DiscrepancyFlag.model_validate_json(flag.model_dump_json())
        assert restored == flag
