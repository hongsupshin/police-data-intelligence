"""Tests for the Search Node.

All tests are unit tests - Tavily API calls are mocked.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.agents.state import (
    Article,
    DatasetType,
    EnrichmentState,
    PipelineStage,
    SearchAttempt,
    SearchStrategyType,
)
from src.retrieval.search_node import (
    build_search_query,
    search_node,
)

# --- Fixtures ---


@pytest.fixture
def base_state() -> EnrichmentState:
    """State with all incident fields populated (after Extract)."""
    return EnrichmentState(
        incident_id="142",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        location="Houston",
        incident_date=date(2018, 3, 15),
        officer_name="James Rodriguez",
        civilian_name="John Doe",
        severity="fatal",
        current_stage=PipelineStage.EXTRACT,
        next_strategy=SearchStrategyType.EXACT_MATCH,
    )


@pytest.fixture
def state_missing_names() -> EnrichmentState:
    """State where both officer and civilian names are None."""
    return EnrichmentState(
        incident_id="200",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        location="Dallas",
        incident_date=date(2020, 7, 4),
        officer_name=None,
        civilian_name=None,
        severity="non-fatal",
        current_stage=PipelineStage.EXTRACT,
        next_strategy=SearchStrategyType.EXACT_MATCH,
    )


@pytest.fixture
def tavily_response() -> dict:
    """Canned Tavily API response matching the documented schema."""
    return {
        "query": "Houston Texas police shooting 2018-03-15",
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": [
            {
                "url": "https://example.com/article1",
                "title": "Houston officer involved in shooting",
                "content": "A police officer shot a suspect in Houston on March 15.",
                "score": 0.92,
            },
            {
                "url": "https://example.com/article2",
                "title": "Fatal shooting in Houston",
                "content": "Police shooting reported in Houston, TX.",
                "score": 0.85,
            },
        ],
        "response_time": 1.23,
        "request_id": "abc-123",
    }


# --- build_search_query tests ---


class TestBuildSearchQueryExactMatch:
    """Tests for EXACT_MATCH strategy."""

    def test_includes_all_fields(self, base_state: EnrichmentState) -> None:
        """Query should contain location, exact date, and names."""
        search_query = build_search_query(base_state, SearchStrategyType.EXACT_MATCH)
        assert base_state.location in search_query, "Location missing."
        assert (
            base_state.incident_date.strftime("%Y-%m-%d") in search_query
        ), "Date missing."
        assert base_state.officer_name in search_query, "Officer name missing."
        assert base_state.civilian_name in search_query, "Civilian name missing."

    def test_includes_severity_for_fatal(self, base_state: EnrichmentState) -> None:
        """Fatal incidents should include a fatality keyword."""
        search_query = build_search_query(base_state, SearchStrategyType.EXACT_MATCH)
        assert base_state.severity in search_query, "Severity ('fatal') missing."

    def test_includes_texas(self, base_state: EnrichmentState) -> None:
        """All queries should include 'Texas'."""
        search_query = build_search_query(base_state, SearchStrategyType.EXACT_MATCH)
        assert "Texas" in search_query, "'Texas' missing."

    def test_missing_names_skipped(self, state_missing_names: EnrichmentState) -> None:
        """None names should not appear in the query string."""
        search_query = build_search_query(
            state_missing_names, SearchStrategyType.EXACT_MATCH
        )
        assert (
            str(state_missing_names.civilian_name) not in search_query
        ), "'None' present in query"
        assert (
            str(state_missing_names.officer_name) not in search_query
        ), "'None' present in query"

    def test_non_fatal_excludes_fatal_keyword(
        self, state_missing_names: EnrichmentState
    ) -> None:
        """Non-fatal incidents should not include fatality keywords."""
        search_query = build_search_query(
            state_missing_names, SearchStrategyType.EXACT_MATCH
        )
        assert (
            state_missing_names.severity not in search_query
        ), "'non-fatal' severity present in query"


class TestBuildSearchQueryTemporalExpanded:
    """Tests for TEMPORAL_EXPANDED strategy."""

    def test_date_becomes_month_year(self, base_state: EnrichmentState) -> None:
        """Date should be formatted as 'Month YYYY' instead of exact date."""
        search_query = build_search_query(
            base_state, SearchStrategyType.TEMPORAL_EXPANDED
        )
        assert (
            base_state.incident_date.strftime("%B %Y") in search_query
        ), "Date format is incorrect."
        assert (
            base_state.incident_date.strftime("%Y-%m-%d") not in search_query
        ), "Date format is incorrect."

    def test_names_still_included(self, base_state: EnrichmentState) -> None:
        """Names should still be present in temporal expanded queries."""
        search_query = build_search_query(
            base_state, SearchStrategyType.TEMPORAL_EXPANDED
        )
        assert base_state.officer_name in search_query, "Officer name missing."
        assert base_state.civilian_name in search_query, "Civilian name missing."


class TestBuildSearchQueryEntityDropped:
    """Tests for ENTITY_DROPPED strategy."""

    def test_names_excluded(self, base_state: EnrichmentState) -> None:
        """Officer and civilian names should not appear even when available."""
        search_query = build_search_query(base_state, SearchStrategyType.ENTITY_DROPPED)
        assert (
            base_state.officer_name not in search_query
        ), "Officer name still present in query."
        assert (
            base_state.civilian_name not in search_query
        ), "Civilian name still present in query."

    def test_location_and_date_kept(self, base_state: EnrichmentState) -> None:
        """Location and date range should still be present."""
        search_query = build_search_query(base_state, SearchStrategyType.ENTITY_DROPPED)
        assert base_state.location in search_query, "Location missing."
        assert (
            base_state.incident_date.strftime("%B %Y") in search_query
        ), "Expanded date missing."
        assert (
            base_state.incident_date.strftime("%Y-%m-%d") not in search_query
        ), "Incorrect date format."


# --- search_node tests ---


class TestSearchNode:
    """Tests for the search_node function."""

    @patch("src.retrieval.search_node.TavilyClient")
    def test_returns_articles(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        tavily_response: dict,
    ) -> None:
        """Tavily results should be converted to Article objects."""
        mock_instance = mock_client_cls.return_value
        mock_instance.search.return_value = tavily_response

        result = search_node(base_state)
        assert isinstance(
            result.retrieved_articles, list
        ), "Articles are not in a list."
        for article in result.retrieved_articles:
            assert isinstance(article, Article), "Wrong article format."
        assert (
            len(result.retrieved_articles) == 2
        ), "Incorrect number of retrieved articles."

    @patch("src.retrieval.search_node.TavilyClient")
    def test_records_search_attempt(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        tavily_response: dict,
    ) -> None:
        """A SearchAttempt should be appended to state.search_attempts."""
        mock_client_cls.return_value.search.return_value = tavily_response
        result = search_node(base_state)
        current_search_attempt = result.search_attempts[0]
        assert (
            len(result.search_attempts) == 1
        ), "Wrong number of search attempts."  # Called once
        assert isinstance(
            current_search_attempt, SearchAttempt
        ), "Incorrect search attempt type."

        # Using the exact values from the fixture, tavily_response
        assert (
            current_search_attempt.query
            == "Houston Texas police shooting 2018-03-15 James Rodriguez John Doe fatal"
        )
        assert current_search_attempt.strategy == SearchStrategyType.EXACT_MATCH
        assert current_search_attempt.num_results == 2
        assert current_search_attempt.avg_relevance_score == (0.92 + 0.85) / 2

    @patch("src.retrieval.search_node.TavilyClient")
    def test_updates_stage_to_search(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        tavily_response: dict,
    ) -> None:
        """current_stage should be set to PipelineStage.SEARCH."""
        mock_client_cls.return_value.search.return_value = tavily_response
        result = search_node(base_state)
        assert result.current_stage == PipelineStage.SEARCH, "Incorrect PipelineStage."

    @patch("src.retrieval.search_node.TavilyClient")
    def test_handles_empty_results(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
    ) -> None:
        """Empty results should produce empty articles list and num_results=0."""
        mock_client_cls.return_value.search.return_value = {"results": []}
        result = search_node(base_state)
        assert result.retrieved_articles == [], "retrieved_articles is not empty."
        assert result.search_attempts[0].num_results == 0, "Incorrect num_results."

    @patch("src.retrieval.search_node.TavilyClient")
    def test_api_error_sets_error_message(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
    ) -> None:
        """Tavily API errors should be caught and stored in error_message."""
        mock_client_cls.return_value.search.side_effect = ValueError("API key invalid")
        result = search_node(base_state)
        assert result.error_message == "Search failed: API key invalid"

    @patch("src.retrieval.search_node.TavilyClient")
    def test_calculates_avg_relevance_score(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        tavily_response: dict,
    ) -> None:
        """SearchAttempt.avg_relevance_score should be mean of result scores."""
        mock_client_cls.return_value.search.return_value = tavily_response
        result = search_node(base_state)
        assert result.search_attempts[0].avg_relevance_score == (0.92 + 0.85) / 2
