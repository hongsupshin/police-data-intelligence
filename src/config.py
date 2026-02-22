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

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Enrichment pipeline settings loaded from environment variables.

    All fields map to ``ENRICHMENT_<FIELD_NAME>`` environment variables.

    Attributes:
        output_dir: Directory for pipeline output JSON files.
        max_search_results: Maximum articles per Tavily search request.
        search_depth: Tavily search depth (``basic`` or ``advanced``).
        relevance_score_threshold: Minimum average relevance score to
            proceed from search to validation.
        fuzzy_match_threshold: Minimum rapidfuzz score for location,
            name, and cross-article consistency matching.
        date_proximity_days: Maximum days between article and incident
            dates for a date match.
    """

    model_config = SettingsConfigDict(env_prefix="ENRICHMENT_")

    output_dir: str = "output/enrichment"
    max_search_results: int = 5
    search_depth: str = "advanced"
    relevance_score_threshold: float = 0.5
    fuzzy_match_threshold: int = 80
    date_proximity_days: int = 3
