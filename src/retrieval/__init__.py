"""Retrieval module for the enrichment pipeline.

Handles web search via Tavily API to find news articles
related to police shooting incidents.
"""

from src.retrieval.search_node import build_search_query, search_node

__all__ = ["build_search_query", "search_node"]
