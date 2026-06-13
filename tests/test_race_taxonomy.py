"""Tests for src.race_taxonomy: normalize_race + classify_race."""

import pytest

from src.race_taxonomy import classify_race, normalize_race


class TestNormalizeRace:
    """normalize_race maps to TJI's coarse 4-bucket scheme (matching key)."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Black", "black"),
            ("African American", "black"),
            ("African-American/Black", "black"),
            ("White", "white"),
            ("Caucasian", "white"),
            ("European-American", "white"),
            ("Hispanic", "hispanic"),
            ("Latino", "hispanic"),
            ("Hispanic/Latino male", "hispanic"),
            # nationality -> hispanic (the fix)
            ("Honduran", "hispanic"),
            ("El Salvadoran", "hispanic"),
            ("Mexican", "hispanic"),
            # races absent from TJI's scheme -> other (the fix)
            ("Asian", "other"),
            ("Korean", "other"),
            ("Iranian", "other"),
            ("Egyptian", "other"),
            ("Native American", "other"),
            # unknown -> other
            ("Martian", "other"),
        ],
    )
    def test_buckets(self, value: str, expected: str) -> None:
        assert normalize_race(value) == expected

    def test_strips_gender_words(self) -> None:
        assert normalize_race("Black male") == "black"

    def test_consistent_with_classify(self) -> None:
        for v in ["Asian", "Honduran", "Black", "Iranian", "Martian"]:
            assert normalize_race(v) == classify_race(v).tji_bucket


class TestClassifyRace:
    """classify_race adds the divergence annotation."""

    def test_synonym_no_divergence(self) -> None:
        f = classify_race("Latino")
        assert (f.tji_bucket, f.divergence_type, f.diverges) == (
            "hispanic",
            "none",
            False,
        )

    def test_nationality_diverges(self) -> None:
        f = classify_race("Honduran")
        assert f.tji_bucket == "hispanic"
        assert f.divergence_type == "nationality_as_ethnicity"
        assert f.diverges is True

    def test_race_absent_diverges(self) -> None:
        for v in ("Asian", "Iranian", "Native American"):
            f = classify_race(v)
            assert f.tji_bucket == "other"
            assert f.divergence_type == "race_absent_from_scheme"
            assert f.diverges is True

    def test_uncategorized_diverges(self) -> None:
        f = classify_race("Martian")
        assert f.tji_bucket == "other"
        assert f.divergence_type == "uncategorized"
        assert f.diverges is True

    def test_preserves_raw_extracted_value(self) -> None:
        assert classify_race("Asian").extracted == "Asian"
        assert classify_race("Honduran").extracted == "Honduran"
