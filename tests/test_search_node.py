"""Tests for the Search Node.

All tests are unit tests - Tavily API calls are mocked.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from src.agents.state import (
    Article,
    DatasetType,
    EnrichmentState,
    PipelineStage,
    SearchAttempt,
    SearchStrategyType,
)
from src.config import Settings
from src.retrieval.search_node import (
    _date_window,
    build_search_query,
    execute_search,
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
        current_stage=PipelineStage.LOAD,
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
        current_stage=PipelineStage.LOAD,
        next_strategy=SearchStrategyType.EXACT_MATCH,
    )


@pytest.fixture
def config() -> RunnableConfig:
    """RunnableConfig injecting default Settings, mirroring run.py."""
    return RunnableConfig({"configurable": {"settings": Settings()}})


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
        assert base_state.incident_date.strftime("%Y-%m-%d") in search_query, (
            "Date missing."
        )
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
        assert str(state_missing_names.civilian_name) not in search_query, (
            "'None' present in query"
        )
        assert str(state_missing_names.officer_name) not in search_query, (
            "'None' present in query"
        )

    def test_non_fatal_excludes_fatal_keyword(
        self, state_missing_names: EnrichmentState
    ) -> None:
        """Non-fatal incidents should not include fatality keywords."""
        search_query = build_search_query(
            state_missing_names, SearchStrategyType.EXACT_MATCH
        )
        assert state_missing_names.severity not in search_query, (
            "'non-fatal' severity present in query"
        )


class TestBuildSearchQueryTemporalExpanded:
    """Tests for TEMPORAL_EXPANDED strategy."""

    def test_date_becomes_month_year(self, base_state: EnrichmentState) -> None:
        """Date should be formatted as 'Month YYYY' instead of exact date."""
        search_query = build_search_query(
            base_state, SearchStrategyType.TEMPORAL_EXPANDED
        )
        assert base_state.incident_date.strftime("%B %Y") in search_query, (
            "Date format is incorrect."
        )
        assert base_state.incident_date.strftime("%Y-%m-%d") not in search_query, (
            "Date format is incorrect."
        )

    def test_names_still_included(self, base_state: EnrichmentState) -> None:
        """Names should still be present in temporal expanded queries."""
        search_query = build_search_query(
            base_state, SearchStrategyType.TEMPORAL_EXPANDED
        )
        assert base_state.officer_name in search_query, "Officer name missing."
        assert base_state.civilian_name in search_query, "Civilian name missing."


# --- _date_window tests ---


class TestDateWindow:
    """Tests for the _date_window helper."""

    def test_default_window(self, base_state: EnrichmentState) -> None:
        """Default settings produce a ±(14, 60)-day window as YYYY-MM-DD."""
        start, end = _date_window(base_state, Settings())
        assert start == "2018-03-01", "Incorrect start_date."
        assert end == "2018-05-14", "Incorrect end_date."

    def test_none_incident_date_returns_none(
        self, base_state: EnrichmentState
    ) -> None:
        """A None incident_date yields (None, None) so callers omit kwargs."""
        state = base_state.model_copy(update={"incident_date": None})
        assert _date_window(state, Settings()) == (None, None)

    def test_respects_overridden_window(self, base_state: EnrichmentState) -> None:
        """Window widths come from settings, not hardcoded values."""
        settings = Settings(search_window_back_days=7, search_window_forward_days=7)
        start, end = _date_window(base_state, settings)
        assert start == "2018-03-08", "Back window not honored."
        assert end == "2018-03-22", "Forward window not honored."


# --- execute_search tests ---


class TestExecuteSearch:
    """Tests for the execute_search helper."""

    @patch("src.retrieval.search_node.TavilyClient")
    def test_returns_article_list(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        tavily_response: dict,
    ) -> None:
        """Tavily results are converted to a list of Article objects."""
        mock_client_cls.return_value.search.return_value = tavily_response
        articles = execute_search(
            base_state, SearchStrategyType.EXACT_MATCH, Settings()
        )
        assert isinstance(articles, list), "Articles are not in a list."
        assert len(articles) == 2, "Incorrect number of articles."
        assert all(isinstance(a, Article) for a in articles), "Wrong article format."

    @patch("src.retrieval.search_node.TavilyClient")
    def test_wires_max_results_from_settings(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        tavily_response: dict,
    ) -> None:
        """max_results comes from settings, not a hardcoded value."""
        mock_client_cls.return_value.search.return_value = tavily_response
        settings = Settings(max_search_results=3)
        execute_search(base_state, SearchStrategyType.EXACT_MATCH, settings)
        _, kwargs = mock_client_cls.return_value.search.call_args
        assert kwargs["max_results"] == 3, "max_results not wired from settings."

    @patch("src.retrieval.search_node.TavilyClient")
    def test_passes_date_window(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        tavily_response: dict,
    ) -> None:
        """A dated state passes start_date/end_date to Tavily."""
        mock_client_cls.return_value.search.return_value = tavily_response
        execute_search(base_state, SearchStrategyType.EXACT_MATCH, Settings())
        _, kwargs = mock_client_cls.return_value.search.call_args
        assert kwargs["start_date"] == "2018-03-01", "Missing/incorrect start_date."
        assert kwargs["end_date"] == "2018-05-14", "Missing/incorrect end_date."


# --- search_node tests ---


class TestSearchNode:
    """Tests for the search_node function."""

    @patch("src.retrieval.search_node.TavilyClient")
    def test_returns_articles(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        config: RunnableConfig,
        tavily_response: dict,
    ) -> None:
        """Tavily results should be converted to Article objects."""
        mock_instance = mock_client_cls.return_value
        mock_instance.search.return_value = tavily_response

        result = search_node(base_state, config)
        assert isinstance(result.retrieved_articles, list), (
            "Articles are not in a list."
        )
        for article in result.retrieved_articles:
            assert isinstance(article, Article), "Wrong article format."
        assert len(result.retrieved_articles) == 2, (
            "Incorrect number of retrieved articles."
        )

    @patch("src.retrieval.search_node.TavilyClient")
    def test_records_search_attempt(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        config: RunnableConfig,
        tavily_response: dict,
    ) -> None:
        """A SearchAttempt should be appended to state.search_attempts."""
        mock_client_cls.return_value.search.return_value = tavily_response
        result = search_node(base_state, config)
        current_search_attempt = result.search_attempts[0]
        assert len(result.search_attempts) == 1, (
            "Wrong number of search attempts."
        )  # Called once
        assert isinstance(current_search_attempt, SearchAttempt), (
            "Incorrect search attempt type."
        )

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
        config: RunnableConfig,
        tavily_response: dict,
    ) -> None:
        """current_stage should be set to PipelineStage.SEARCH."""
        mock_client_cls.return_value.search.return_value = tavily_response
        result = search_node(base_state, config)
        assert result.current_stage == PipelineStage.SEARCH, "Incorrect PipelineStage."

    @patch("src.retrieval.search_node.TavilyClient")
    def test_handles_empty_results(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        config: RunnableConfig,
    ) -> None:
        """Empty results should produce empty articles list and num_results=0."""
        mock_client_cls.return_value.search.return_value = {"results": []}
        result = search_node(base_state, config)
        assert result.retrieved_articles == [], "retrieved_articles is not empty."
        assert result.search_attempts[0].num_results == 0, "Incorrect num_results."

    @patch("src.retrieval.search_node.TavilyClient")
    def test_api_error_sets_error_message(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        config: RunnableConfig,
    ) -> None:
        """Tavily API errors should be caught and stored in error_message."""
        mock_client_cls.return_value.search.side_effect = ValueError("API key invalid")
        result = search_node(base_state, config)
        assert result.error_message == "Search failed: API key invalid"

    @patch("src.retrieval.search_node.TavilyClient")
    def test_calculates_avg_relevance_score(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        config: RunnableConfig,
        tavily_response: dict,
    ) -> None:
        """SearchAttempt.avg_relevance_score should be mean of result scores."""
        mock_client_cls.return_value.search.return_value = tavily_response
        result = search_node(base_state, config)
        assert result.search_attempts[0].avg_relevance_score == (0.92 + 0.85) / 2

    @patch("src.retrieval.search_node.TavilyClient")
    def test_excludes_wikipedia_and_fatalencounters(
        self,
        mock_client_cls: MagicMock,
        base_state: EnrichmentState,
        config: RunnableConfig,
        tavily_response: dict,
    ) -> None:
        """Tavily search excludes aggregators and applies window + max_results."""
        mock_client_cls.return_value.search.return_value = tavily_response
        search_node(base_state, config)
        mock_client_cls.return_value.search.assert_called_once_with(
            build_search_query(base_state, SearchStrategyType.EXACT_MATCH),
            max_results=10,
            search_depth="advanced",
            exclude_domains=["wikipedia.org", "fatalencounters.org"],
            start_date="2018-03-01",
            end_date="2018-05-14",
        )
