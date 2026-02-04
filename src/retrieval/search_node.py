"""Search Node for the enrichment pipeline.

Constructs search queries from incident data and calls the Tavily API
to retrieve news articles. This is a deterministic node - no LLM calls,
just algorithmic query construction and API interaction.

The node reads state.next_strategy (set by the Coordinator) and executes
a single search. Retry decisions are made by the Coordinator, not here.
"""

import os

from tavily import TavilyClient

from src.agents.state import (
    Article,
    EnrichmentState,
    PipelineStage,
    SearchAttempt,
    SearchStrategyType,
)


def build_search_query(state: EnrichmentState, strategy: SearchStrategyType) -> str:
    """Construct a Tavily search query from incident fields and strategy.

    Builds a query string by combining available incident fields
    (location, date, names, severity) according to the selected
    search strategy. Fields that are None are skipped.

    Strategy behavior:
        - EXACT_MATCH: All available fields, exact date (YYYY-MM-DD).
        - TEMPORAL_EXPANDED: Replace exact date with "Month YYYY" format.
        - ENTITY_DROPPED: Drop officer and civilian names, keep
          location + date range.

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
        >>> build_search_query(state, SearchStrategyType.ENTITY_DROPPED)
        'Houston Texas police shooting March 2018 fatal'
    """
    if strategy == SearchStrategyType.EXACT_MATCH:
        date = state.incident_date.strftime("%Y-%m-%d")
        officer = state.officer_name
        civilian = state.civilian_name
    elif strategy == SearchStrategyType.TEMPORAL_EXPANDED:
        date = state.incident_date.strftime("%B %Y")
        officer = state.officer_name
        civilian = state.civilian_name
    elif strategy == SearchStrategyType.ENTITY_DROPPED:
        date = state.incident_date.strftime("%B %Y")  # Expand the date window
        officer = ""
        civilian = ""
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


def _convert_tavily_result(result: dict) -> Article:
    """Convert a single Tavily API result dict to an Article model.

    Args:
        result: A dictionary from the Tavily response "results" array
            with keys: url, title, content, score.

    Returns:
        Article instance populated from the Tavily result.
    """
    tavily_article = Article(
        url=result["url"],
        title=result["title"],
        snippet=result["content"][:500],
        content=result["content"],
        relevance_score=result["score"],
    )

    return tavily_article


def search_node(state: EnrichmentState) -> EnrichmentState:
    """Execute a web search for news articles about the incident.

    Reads state.next_strategy to determine query construction approach,
    calls the Tavily API, and updates state with retrieved articles and
    search attempt metadata.

    This node does NOT decide whether to retry - it executes one search
    per invocation. The Coordinator inspects the results and sets
    next_strategy for subsequent calls if needed.

    Steps:
        1. Build query string via build_search_query().
        2. Call Tavily API with max_results=5, search_depth="advanced".
        3. Convert results to Article objects.
        4. Record a SearchAttempt with query, strategy, num_results,
           and avg_relevance_score.
        5. Update state: append to search_attempts, set retrieved_articles,
           set current_stage to SEARCH.

    Args:
        state: Current enrichment state with incident fields and
            next_strategy populated.

    Returns:
        Updated EnrichmentState with:
        - retrieved_articles: List of Article objects from this search.
        - search_attempts: Appended with this attempt's metadata.
        - current_stage: Set to PipelineStage.SEARCH.
        - error_message: Set if the search fails.

    Examples:
        >>> state = EnrichmentState(
        ...     incident_id="142",
        ...     dataset_type=DatasetType.CIVILIANS_SHOT,
        ...     location="Houston",
        ...     incident_date=date(2018, 3, 15),
        ...     next_strategy=SearchStrategyType.EXACT_MATCH,
        ... )
        >>> updated = search_node(state)
        >>> len(updated.retrieved_articles)
        5
        >>> updated.current_stage
        <PipelineStage.SEARCH: 'search'>
    """
    # Build the search query
    strategy = state.next_strategy
    search_query = build_search_query(state, strategy)

    try:
        # Retrieve articles via Tavily
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(
            search_query,
            max_results=5,
            search_depth="advanced",  # 2 API credits per request
        )["results"]
        tavily_articles = [_convert_tavily_result(result) for result in results]
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
    except (ValueError, KeyError, Exception) as e:
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
