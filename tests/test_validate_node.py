"""Tests for the validation node.

Tests cover the three helper functions (check_location_match,
check_name_match, check_date_match) and the main validate_node
function that orchestrates article validation against incident data.
"""

from datetime import date

import pytest

from src.agents.state import (
    Article,
    DatasetType,
    EnrichmentState,
    PipelineStage,
    SearchStrategyType,
)
from src.validation.validate_node import (
    check_date_match,
    check_location_match,
    check_name_match,
    validate_node,
)


@pytest.fixture
def base_state() -> EnrichmentState:
    """State with all incident fields with search results (after Search)."""
    return EnrichmentState(
        incident_id="142",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        location="Houston",
        incident_date=date(2018, 3, 15),
        officer_name="James Rodriguez",
        civilian_name="John Doe",
        severity="fatal",
        current_stage=PipelineStage.SEARCH,
        next_strategy=SearchStrategyType.EXACT_MATCH,
        retrieved_articles=[
            Article(
                url="https://example.com/article1",
                title="Houston officer James Rodriguez involved in shooting of John Doe",
                snippet="A Houston police officer fatally shot John Doe during a traffic stop on March 15.",
                content="A Houston police officer identified as James Rodriguez fatally shot John Doe, 34, during a traffic stop on the city's east side on March 15, 2018. Witnesses say the encounter escalated quickly after Doe exited his vehicle.",
                source_name="CBS",
                relevance_score=0.9,
                published_date=date(2018, 3, 15),
            ),
            Article(
                url="https://example.com/article2",
                title="Houston fatal police shooting, victim is John Doe",
                snippet="Police in Houston, TX confirmed a fatal officer-involved shooting on March 14.",
                content="Police in Houston, TX confirmed a fatal officer-involved shooting near downtown on Wednesday. The victim was identified as John Doe. Officials have not yet released the name of the officer involved.",
                source_name="NBC",
                relevance_score=0.7,
                published_date=date(2018, 3, 14),
            ),
        ],
    )


def test_check_location_match() -> None:
    """Fuzzy partial match of location within article text."""
    assert check_location_match("A shooting in Houston", "Houston")
    assert not check_location_match("Shooting in Dallas", "Houston")
    assert not check_location_match(None, "Houston")
    assert not check_location_match("Austin", None)


def test_check_name_match() -> None:
    """Fuzzy partial match of victim name within article text."""
    assert check_name_match("The victim was identified as John Doe", "John Doe")
    assert not check_name_match("The victim was identified as Jane Smith", "John Doe")
    assert not check_name_match(None, "John Doe")
    assert not check_name_match("The victim was John Doe", None)


def test_check_date_match() -> None:
    """Date match within +/-3 day tolerance, including boundary."""
    assert check_date_match(date(2018, 3, 10), date(2018, 3, 10))
    assert check_date_match(date(2018, 3, 10), date(2018, 3, 12))
    assert check_date_match(date(2018, 3, 9), date(2018, 3, 12))
    assert not check_date_match(date(2018, 3, 8), date(2018, 3, 12))
    assert not check_date_match(None, date(2018, 3, 12))
    assert not check_date_match(date(2018, 3, 12), None)


class TestValidateNode:
    """Tests for the validate_node orchestrator function."""

    def test_happy_path(self, base_state: EnrichmentState) -> None:
        """All articles match on date, location, and name."""
        result = validate_node(base_state)
        assert result.current_stage == PipelineStage.VALIDATE
        assert len(result.validation_results) == 2
        for vr in result.validation_results:
            assert vr.location_match
            assert vr.date_match
            assert vr.victim_name_match
            assert vr.passed

    def test_victim_name_match_none(self, base_state: EnrichmentState) -> None:
        """victim_name_match is None when civilian_name unavailable."""
        base_state.civilian_name = None
        result = validate_node(base_state)
        assert result.current_stage == PipelineStage.VALIDATE
        assert len(result.validation_results) == 2
        for vr in result.validation_results:
            assert vr.location_match
            assert vr.date_match
            assert vr.victim_name_match is None
            assert vr.passed  # only checks location and date

    def test_content_fallback_on_title(self, base_state: EnrichmentState) -> None:
        """Empty content falls back to title for location and name matching."""
        base_state.retrieved_articles = [
            Article(
                url="https://example.com/fallback",
                title="Houston police shooting John Doe",
                snippet="",
                content="",
                published_date=date(2018, 3, 14),
            )
        ]
        result = validate_node(base_state)
        for vr in result.validation_results:
            assert vr.location_match
            assert vr.date_match
            assert vr.victim_name_match
            assert vr.passed

    def test_exception_handling(self, base_state: EnrichmentState) -> None:
        """Invalid article triggers exception and sets error_message."""
        base_state.retrieved_articles = ["not an article"]
        result = validate_node(base_state)
        assert result.validation_results == []
        assert "Validation failed" in result.error_message

    def test_article_with_missing_date(self, base_state: EnrichmentState) -> None:
        """Missing published_date causes date_match=False and passed=False."""
        base_state.retrieved_articles = [
            Article(
                url="https://example.com/date_missing",
                title="Houston police shooting John Doe",
                snippet="Police in Houston, TX confirmed a fatal officer-involved shooting near downtown on Wednesday.",
                content="Police in Houston, TX confirmed a fatal officer-involved shooting near downtown on Wednesday. The victim was identified as John Doe. Officials have not yet released the name of the officer involved.",
                published_date=None,
            )
        ]
        result = validate_node(base_state)
        for vr in result.validation_results:
            assert not vr.date_match
            assert not vr.passed

    def test_article_with_missing_location(self, base_state: EnrichmentState) -> None:
        """Missing state location causes location_match=False and passed=False."""
        base_state.location = None
        result = validate_node(base_state)
        for vr in result.validation_results:
            assert not vr.location_match
            assert not vr.passed

    def test_date_match_location_mismatch(self, base_state: EnrichmentState) -> None:
        """Date matches but wrong location still fails validation."""
        base_state.location = "Dallas"
        result = validate_node(base_state)
        for vr in result.validation_results:
            assert not vr.location_match
            assert vr.date_match
            assert not vr.passed
