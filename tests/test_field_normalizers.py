"""Tests for the shared outcome/time-of-day canonical normalizers."""

import pytest

from src.field_normalizers import (
    detect_period,
    hour_in_bucket,
    normalize_outcome,
    parse_hour,
    time_period_bucket,
)


class TestNormalizeOutcome:
    """normalize_outcome maps free text to fatal / non-fatal / None."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("The suspect was shot and killed", "fatal"),
            ("He died at the scene", "fatal"),
            ("a fatally wounded man", "non-fatal"),  # non-fatal checked first
            ("the officer was injured", "non-fatal"),
            ("the victim survived", "non-fatal"),
            ("taken into custody unharmed", None),
            ("", None),
        ],
    )
    def test_normalize_outcome(self, value: str, expected: str | None) -> None:
        assert normalize_outcome(value) == expected


class TestParseHour:
    """parse_hour extracts a 0-23 hour from varied time strings."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2:30 PM", 14),
            ("14:30", 14),
            ("2 a.m.", 2),
            ("12 a.m.", 0),
            ("12 p.m.", 12),
            ("11 p.m.", 23),
            ("noon", None),
            ("Tuesday", None),
        ],
    )
    def test_parse_hour(self, value: str, expected: int | None) -> None:
        assert parse_hour(value) == expected


class TestDetectPeriod:
    """detect_period finds a period keyword in text."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("early this morning", "morning"),
            ("in the afternoon", "afternoon"),
            ("late evening", "evening"),
            ("overnight / at night", "night"),
            ("on a Tuesday", None),
        ],
    )
    def test_detect_period(self, value: str, expected: str | None) -> None:
        assert detect_period(value) == expected


class TestHourInBucket:
    """hour_in_bucket handles ranges including the midnight-wrapping night."""

    @pytest.mark.parametrize(
        "hour,bucket,expected",
        [
            (8, "morning", True),
            (14, "afternoon", True),
            (20, "evening", True),
            (23, "night", True),
            (3, "night", True),  # wraps past midnight
            (8, "night", False),
        ],
    )
    def test_hour_in_bucket(self, hour: int, bucket: str, expected: bool) -> None:
        assert hour_in_bucket(hour, bucket) is expected


class TestTimePeriodBucket:
    """time_period_bucket canonicalizes a time string to a coarse bucket."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("around 1 p.m.", "afternoon"),
            ("at 3am", "night"),
            ("this morning", "morning"),
            ("11 p.m.", "night"),
            ("Tuesday", None),
        ],
    )
    def test_time_period_bucket(self, value: str, expected: str | None) -> None:
        assert time_period_bucket(value) == expected
