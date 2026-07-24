"""Tests for src/eval/comparators.py.

Covers the audit-only person-name comparator (normalize_person_name,
compare_name, AUDIT_FIELD_COMPARATORS) and regression-guards the extraction
of the comparison layer out of src/eval/holdout.py (frozen FIELD_COMPARATORS
membership, backward-compatible re-exports).
"""

from datetime import time

import pytest

from src.agents.state import DatasetType, MediaFeatureField
from src.eval.comparators import (
    AUDIT_FIELD_COMPARATORS,
    FIELD_COMPARATORS,
    EvalError,
    compare_age,
    compare_location,
    compare_name,
    compare_outcome,
    compare_race,
    compare_time,
    compare_weapon,
    fetch_ground_truth,
    normalize_person_name,
)

# ---------------------------------------------------------------------------
# Refactor regression: frozen holdout mapping + backward-compatible re-exports
# ---------------------------------------------------------------------------


class TestRefactorRegression:
    """The extraction must not change the holdout comparison layer."""

    def test_field_comparators_membership_frozen(self):
        assert FIELD_COMPARATORS == {
            MediaFeatureField.CIVILIAN_AGE: compare_age,
            MediaFeatureField.CIVILIAN_RACE: compare_race,
            MediaFeatureField.WEAPON: compare_weapon,
            MediaFeatureField.LOCATION_DETAIL: compare_location,
            MediaFeatureField.TIME_OF_DAY: compare_time,
            MediaFeatureField.OUTCOME: compare_outcome,
        }

    def test_name_fields_not_in_holdout_comparators(self):
        assert MediaFeatureField.OFFICER_NAME not in FIELD_COMPARATORS
        assert MediaFeatureField.CIVILIAN_NAME not in FIELD_COMPARATORS

    def test_audit_comparators_extend_holdout_comparators(self):
        for field, comparator in FIELD_COMPARATORS.items():
            assert AUDIT_FIELD_COMPARATORS[field] is comparator
        assert AUDIT_FIELD_COMPARATORS[MediaFeatureField.OFFICER_NAME] is compare_name
        assert AUDIT_FIELD_COMPARATORS[MediaFeatureField.CIVILIAN_NAME] is compare_name

    def test_circumstance_has_no_comparator(self):
        assert MediaFeatureField.CIRCUMSTANCE not in AUDIT_FIELD_COMPARATORS

    def test_holdout_reexports_resolve_to_same_objects(self):
        from src.eval import comparators, holdout

        for name in (
            "EVAL_FIELDS",
            "FIELD_COMPARATORS",
            "EvalError",
            "MatchResult",
            "PipelineOutcome",
            "_normalize_race",
            "compare_age",
            "compare_location",
            "compare_outcome",
            "compare_race",
            "compare_time",
            "compare_weapon",
            "comparators_for_dataset",
            "fetch_ground_truth",
        ):
            assert getattr(holdout, name) is getattr(comparators, name)


# ---------------------------------------------------------------------------
# normalize_person_name
# ---------------------------------------------------------------------------


class TestNormalizePersonName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Armando Juarez", ["armando", "juarez"]),
            ("ARMANDO JUAREZ", ["armando", "juarez"]),
            ("Sgt. Duran", ["duran"]),
            ("Officer John Duran", ["john", "duran"]),
            ("Deputy Sheriff Jane Doe", ["jane", "doe"]),
            ("Armando L. Juarez Jr.", ["armando", "l", "juarez"]),
            ("Robert Smith III", ["robert", "smith"]),
            ("IBARRA-RUIZ", ["ibarra", "ruiz"]),
            ("Jose Angel Ibarra-Ruiz", ["jose", "angel", "ibarra", "ruiz"]),
            ("José García", ["jose", "garcia"]),
            ("Juarez, Armando", ["juarez", "armando"]),
            ("O'Brien", ["o", "brien"]),
        ],
    )
    def test_tokenization(self, raw, expected):
        assert normalize_person_name(raw) == expected

    def test_title_only_yields_empty(self):
        assert normalize_person_name("Officer") == []
        assert normalize_person_name("Sgt.") == []

    def test_empty_string_yields_empty(self):
        assert normalize_person_name("") == []


# ---------------------------------------------------------------------------
# compare_name
# ---------------------------------------------------------------------------


class TestCompareNameMatches:
    """Consistent names must not become flag candidates."""

    def test_exact_match(self):
        result = compare_name("Armando Juarez", "Armando Juarez", "civilian_name")
        assert result.exact_match is True
        assert result.fuzzy_match is True
        assert result.error is None

    def test_case_and_diacritics_are_exact(self):
        result = compare_name("josé garcía", "Jose Garcia", "civilian_name")
        assert result.exact_match is True

    def test_hyphenation_is_exact(self):
        result = compare_name(
            "Jose Angel Ibarra-Ruiz", "JOSE ANGEL IBARRA RUIZ", "civilian_name"
        )
        assert result.exact_match is True

    def test_rank_prefix_stripped(self):
        result = compare_name("Officer John Duran", "John Duran", "officer_name")
        assert result.exact_match is True

    def test_generational_suffix_stripped(self):
        result = compare_name("Robert Smith Jr.", "Robert Smith", "civilian_name")
        assert result.exact_match is True

    def test_surname_only_is_compatible_not_exact(self):
        result = compare_name("Sgt. Duran", "John Duran", "officer_name")
        assert result.exact_match is False
        assert result.fuzzy_match is True

    def test_reversed_order_is_fuzzy(self):
        result = compare_name("Juarez, Armando", "Armando Juarez", "civilian_name")
        assert result.exact_match is False
        assert result.fuzzy_match is True

    def test_middle_name_extra_token_is_fuzzy(self):
        result = compare_name(
            "Jose Angel Ibarra-Ruiz", "Jose Ibarra-Ruiz", "civilian_name"
        )
        assert result.fuzzy_match is True

    def test_initial_matches_full_name(self):
        result = compare_name("Armando L. Juarez", "Armando Juarez", "civilian_name")
        assert result.fuzzy_match is True

    def test_initial_only_surname_full(self):
        result = compare_name("A. Juarez", "Armando Juarez", "civilian_name")
        assert result.exact_match is False
        assert result.fuzzy_match is True

    def test_first_name_only_is_compatible(self):
        result = compare_name("Armando", "Armando Juarez", "civilian_name")
        assert result.fuzzy_match is True

    def test_fuzzy_score_populated(self):
        result = compare_name("Armando Juarez", "Armando Juarez", "civilian_name")
        assert result.fuzzy_score is not None


class TestCompareNameContradictions:
    """Genuinely different names must be flag candidates (no match)."""

    def test_different_full_names(self):
        result = compare_name("John Smith", "Michael Brown", "officer_name")
        assert result.exact_match is False
        assert result.fuzzy_match is False
        assert result.error is None

    def test_same_first_different_surname(self):
        result = compare_name("Jose Garcia", "Jose Martinez", "civilian_name")
        assert result.fuzzy_match is False

    def test_surname_contradiction(self):
        result = compare_name("Sgt. Duran", "John Miller", "officer_name")
        assert result.fuzzy_match is False

    def test_initial_contradiction(self):
        result = compare_name("B. Juarez", "Armando Juarez", "civilian_name")
        assert result.fuzzy_match is False


class TestCompareNameErrors:
    def test_none_ground_truth(self):
        result = compare_name("John Smith", None, "officer_name")
        assert result.error == EvalError.NO_GROUND_TRUTH

    def test_none_extracted(self):
        result = compare_name(None, "John Smith", "officer_name")
        assert result.error == EvalError.NO_EXTRACTION

    def test_title_only_extraction_is_parse_error(self):
        result = compare_name("Officer", "John Smith", "officer_name")
        assert result.error == EvalError.PARSE_ERROR
        assert result.fuzzy_match is False

    def test_title_only_ground_truth_is_no_ground_truth(self):
        result = compare_name("John Smith", "Officer", "officer_name")
        assert result.error == EvalError.NO_GROUND_TRUTH

    def test_field_name_propagated(self):
        result = compare_name("John Smith", "John Smith", "officer_name")
        assert result.field_name == "officer_name"


# ---------------------------------------------------------------------------
# fetch_ground_truth
# ---------------------------------------------------------------------------


class TestFetchGroundTruth:
    def test_civilians_maps_columns_to_fields(self, mock_connection, mock_cursor):
        mock_cursor.fetchone.return_value = (
            34, "HISPANIC", "handgun", "Houston", time(22, 15), True,
        )
        result = fetch_ground_truth(mock_connection, 1031, DatasetType.CIVILIANS_SHOT)
        assert result == {
            "civilian_age": 34,
            "civilian_race": "HISPANIC",
            "weapon": "handgun",
            "location_detail": "Houston",
            "time_of_day": time(22, 15),
            "outcome": True,
        }

    def test_officers_maps_columns_to_fields(self, mock_connection, mock_cursor):
        mock_cursor.fetchone.return_value = (27, "WHITE", "Dallas", "INJURY")
        result = fetch_ground_truth(mock_connection, 42, DatasetType.OFFICERS_SHOT)
        assert result == {
            "civilian_age": 27,
            "civilian_race": "WHITE",
            "location_detail": "Dallas",
            "outcome": "INJURY",
        }

    def test_civilians_missing_incident_raises(self, mock_connection, mock_cursor):
        mock_cursor.fetchone.return_value = None
        with pytest.raises(KeyError, match="not found in civilians_shot"):
            fetch_ground_truth(mock_connection, 99999, DatasetType.CIVILIANS_SHOT)

    def test_officers_missing_incident_raises(self, mock_connection, mock_cursor):
        mock_cursor.fetchone.return_value = None
        with pytest.raises(KeyError, match="not found in officers_shot"):
            fetch_ground_truth(mock_connection, 99999, DatasetType.OFFICERS_SHOT)
