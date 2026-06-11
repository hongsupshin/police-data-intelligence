"""Search Node for the enrichment pipeline.

Constructs search queries from incident data and calls the Tavily API
to retrieve news articles. This is a deterministic node - no LLM calls,
just algorithmic query construction and API interaction.

The node reads state.next_strategy (set by the Coordinator) and executes
a single search, bounded to a temporal window around the incident date.
Retry decisions are made by the Coordinator, not here.
"""

import os
from datetime import timedelta

from dateutil import parser
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient

from src.agents.state import (
    Article,
    EnrichmentState,
    PipelineStage,
    SearchAttempt,
    SearchStrategyType,
)
from src.config import Settings


def build_search_query(state: EnrichmentState, strategy: SearchStrategyType) -> str:
    """Construct a Tavily search query from incident fields and strategy.

    Builds a query string by combining available incident fields
    (location, date, names, severity) according to the selected
    search strategy. Fields that are None are skipped.

    Strategy behavior:
        - EXACT_MATCH: All available fields, exact date (YYYY-MM-DD).
        - TEMPORAL_EXPANDED: Replace exact date with "Month YYYY" format.
        - NAME_PARTIAL: Drop officer name, keep civilian name + month-year
          (terminal rung — the victim name is never stripped).

    Always includes "Texas" and "police shooting" as base terms.
    Appends "fatal" or "killed" for fatal-severity incidents.

    Args:
        state: Current enrichment state with incident fields populated
            by the Extract Node.
        strategy: Which search strategy to apply for query construction.

    Returns:
        Search query string ready for the Tavily API.

    Examples:
        >>> state = EnrichmentState(
        ...     incident_id="142",
        ...     dataset_type=DatasetType.CIVILIANS_SHOT,
        ...     location="Houston",
        ...     incident_date=date(2018, 3, 15),
        ...     officer_name="James Rodriguez",
        ...     severity="fatal",
        ... )
        >>> build_search_query(state, SearchStrategyType.EXACT_MATCH)
        'Houston Texas police shooting 2018-03-15 James Rodriguez fatal'
        >>> build_search_query(state, SearchStrategyType.TEMPORAL_EXPANDED)
        'Houston Texas police shooting March 2018 James Rodriguez fatal'
    """
    if strategy == SearchStrategyType.EXACT_MATCH:
        date = state.incident_date.strftime("%Y-%m-%d")
        officer = state.officer_name
        civilian = state.civilian_name
    elif strategy == SearchStrategyType.TEMPORAL_EXPANDED:
        date = state.incident_date.strftime("%B %Y")
        officer = state.officer_name
        civilian = state.civilian_name
    elif strategy == SearchStrategyType.NAME_PARTIAL:
        date = state.incident_date.strftime("%B %Y")
        officer = ""
        civilian = state.civilian_name
    search_query = []
    if state.location:
        search_query.append(state.location)
    search_query.append("Texas police shooting")
    search_query.append(date)
    if officer:
        search_query.append(officer)
    if civilian:
        search_query.append(civilian)
    if state.severity == "fatal":
        search_query.append(state.severity)
    return " ".join(search_query)


def _date_window(
    state: EnrichmentState, settings: Settings
) -> tuple[str | None, str | None]:
    """Compute absolute Tavily start/end date strings around the incident.

    Bounds the search to a window centered on the incident date so Tavily
    does not surface temporally-irrelevant articles. The window spans
    ``search_window_back_days`` before to ``search_window_forward_days``
    after ``incident_date``.

    Args:
        state: Current enrichment state; reads ``incident_date``.
        settings: Pipeline settings providing the window widths.

    Returns:
        A ``(start_date, end_date)`` tuple of ``YYYY-MM-DD`` strings, or
        ``(None, None)`` when ``incident_date`` is None so the caller can
        omit the kwargs rather than send a malformed window.

    Examples:
        >>> state = EnrichmentState(
        ...     incident_id="142",
        ...     dataset_type=DatasetType.CIVILIANS_SHOT,
        ...     incident_date=date(2018, 3, 15),
        ... )
        >>> _date_window(state, Settings())
        ('2018-03-01', '2018-05-14')
    """
    if state.incident_date is None:
        return None, None
    start_date = (
        state.incident_date - timedelta(days=settings.search_window_back_days)
    ).strftime("%Y-%m-%d")
    end_date = (
        state.incident_date + timedelta(days=settings.search_window_forward_days)
    ).strftime("%Y-%m-%d")
    return start_date, end_date


def _convert_tavily_result(result: dict) -> Article:
    """Convert a single Tavily API result dict to an Article model.

    Args:
        result: A dictionary from the Tavily response "results" array
            with keys: url, title, content, score, published_date

    Returns:
        Article instance populated from the Tavily result.
    """
    published_date = result.get("published_date")
    try:
        parsed_date = parser.parse(published_date).date()
    except (ValueError, TypeError):
        parsed_date = None

    tavily_article = Article(
        url=result["url"],
        title=result["title"],
        snippet=result["content"][:500],
        content=result["content"],
        relevance_score=result["score"],
        published_date=parsed_date,
    )

    return tavily_article


def execute_search(
    state: EnrichmentState,
    strategy: SearchStrategyType,
    settings: Settings,
) -> list[Article]:
    """Build a query, apply the temporal window, call Tavily, return Articles.

    Shared retrieval mechanism: constructs the query via
    ``build_search_query``, bounds it to the temporal window from
    ``_date_window``, calls the Tavily API, and converts the raw results to
    ``Article`` objects. It performs no state bookkeeping (SearchAttempt,
    aggregate metrics) - that stays with ``search_node``.

    The query is built from ``state`` and ``strategy``; a future free-text
    ReAct tool is expected to share the window/fetch/convert tail rather
    than calling this function directly.

    Args:
        state: Current enrichment state with incident fields populated.
        strategy: Search strategy for query construction.
        settings: Pipeline settings (max results, search depth, window).

    Returns:
        List of Article objects from the Tavily response (possibly empty).
    """
    search_query = build_search_query(state, strategy)
    start_date, end_date = _date_window(state, settings)

    search_kwargs = {
        "max_results": settings.max_search_results,
        "search_depth": settings.search_depth,  # "advanced" = 2 API credits
        "exclude_domains": ["wikipedia.org", "fatalencounters.org"],
    }
    if start_date is not None:
        search_kwargs["start_date"] = start_date
        search_kwargs["end_date"] = end_date

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(search_query, **search_kwargs)["results"]
    return [_convert_tavily_result(result) for result in results]


def search_node(state: EnrichmentState, config: RunnableConfig) -> EnrichmentState:
    """Execute a web search for news articles about the incident.

    Reads state.next_strategy to determine query construction approach,
    calls the Tavily API (bounded to a temporal window around the incident
    date), and updates state with retrieved articles and search attempt
    metadata.

    This node does NOT decide whether to retry - it executes one search
    per invocation. The Coordinator inspects the results and sets
    next_strategy for subsequent calls if needed.

    Steps:
        1. Build query string via build_search_query().
        2. Retrieve articles via execute_search(), which applies
           max_results and a date window from Settings.
        3. Convert results to Article objects.
        4. Record a SearchAttempt with query, strategy, num_results,
           and avg_relevance_score.
        5. Update state: append to search_attempts, set retrieved_articles,
           set current_stage to SEARCH.

    Args:
        state: Current enrichment state with incident fields and
            next_strategy populated.
        config: RunnableConfig containing Settings under
            ``configurable.settings``.

    Returns:
        Updated EnrichmentState with:
        - retrieved_articles: List of Article objects from this search.
        - search_attempts: Appended with this attempt's metadata.
        - current_stage: Set to PipelineStage.SEARCH.
        - error_message: Set if the search fails.

    Examples:
        >>> updated = search_node(state, config)
        >>> updated.current_stage
        <PipelineStage.SEARCH: 'search'>
    """
    settings = config["configurable"]["settings"]

    # Build the search query (recorded in the attempt even on failure)
    strategy = state.next_strategy
    search_query = build_search_query(state, strategy)

    try:
        # Retrieve articles via Tavily (temporally scoped)
        tavily_articles = execute_search(state, strategy, settings)
        num_results = len(tavily_articles)
        if num_results != 0:
            avg_relevance_score = (
                sum([article.relevance_score for article in tavily_articles])
                / num_results
            )
        else:
            avg_relevance_score = None
        state.retrieved_articles = tavily_articles

    # Error handling
    except Exception as e:
        # Handle errors and populate error_message
        state.error_message = f"Search failed: {str(e)}"
        state.current_stage = PipelineStage.SEARCH
        state.retrieved_articles = []
        num_results = 0
        avg_relevance_score = None

    # Set SearchAttempts
    current_search_attempt = SearchAttempt(
        query=search_query,
        strategy=strategy,
        num_results=num_results,
        avg_relevance_score=avg_relevance_score,
    )
    state.search_attempts.append(current_search_attempt)
    state.current_stage = PipelineStage.SEARCH

    return state
