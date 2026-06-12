"""Enrichment pipeline configuration.

Centralizes settings for the enrichment pipeline. Values are loaded
from environment variables with the ``ENRICHMENT_`` prefix via
pydantic-settings. For example, the ``output_dir`` field reads from
``ENRICHMENT_OUTPUT_DIR``.

Example:
    Override defaults with environment variables::

        export ENRICHMENT_OUTPUT_DIR="/data/results"

    Or instantiate directly for testing::

        settings = Settings(output_dir="/tmp/test")
"""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Enrichment pipeline settings loaded from environment variables.

    All fields map to ``ENRICHMENT_<FIELD_NAME>`` environment variables.

    Attributes:
        output_dir: Directory for pipeline output JSON files.
        max_search_results: Maximum articles per Tavily search request.
        search_depth: Tavily search depth (``basic`` or ``advanced``).
        search_window_back_days: Days before the incident date to bound
            the Tavily search window (``start_date``). Must be at least
            ``date_proximity_days`` so search never excludes an article
            that validation would accept.
        search_window_forward_days: Days after the incident date to bound
            the Tavily search window (``end_date``). Wider than the back
            window to capture follow-up coverage (news lag).
        relevance_score_threshold: Minimum average relevance score to
            proceed from search to validation.
        fuzzy_match_threshold: Minimum rapidfuzz score for location,
            name, and cross-article consistency matching.
        date_proximity_days: Maximum days between article and incident
            dates for a date match.
        enable_relevance_gate: Feature flag for the (Tier 1) LLM relevance
            judge that vetoes a completion matched to the wrong article. Off
            by default; no node reads it yet. Establishes the ``enable_*``
            flag pattern so the eval gate can A/B a behavior in-process.
    """

    model_config = SettingsConfigDict(env_prefix="ENRICHMENT_")

    output_dir: str = "output/enrichment"
    max_search_results: int = 10
    search_depth: str = "advanced"
    search_window_back_days: int = 14
    search_window_forward_days: int = 60
    relevance_score_threshold: float = 0.5
    fuzzy_match_threshold: int = 80
    date_proximity_days: int = 5
    enable_relevance_gate: bool = False

    @model_validator(mode="after")
    def _check_search_window_covers_date_proximity(self) -> "Settings":
        """Ensure the search window never excludes an article validation accepts.

        ``search_window_back_days`` must be at least ``date_proximity_days`` so
        the Tavily search window can never drop an article that the validate
        node would otherwise accept on date proximity.
        """
        if self.search_window_back_days < self.date_proximity_days:
            raise ValueError(
                f"search_window_back_days ({self.search_window_back_days}) must be "
                f">= date_proximity_days ({self.date_proximity_days})"
            )
        return self
