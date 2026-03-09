"""Synthesize module for the enrichment pipeline.

Extract fields from articles using LLM, check consistency across sources,
and validate against database reference values.
"""

from src.synthesize.synthesize_node import (
    check_articles_match,
    check_reference_match,
    extract_fields,
    synthesize_node,
)

__all__ = [
    "check_articles_match",
    "check_reference_match",
    "extract_fields",
    "synthesize_node",
]
