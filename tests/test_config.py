"""Tests for enrichment pipeline settings."""

from src.config import Settings


def test_default_values():
    """All fields have expected defaults when no env vars are set."""
    settings = Settings()
    assert settings.output_dir == "output/enrichment"
    assert settings.max_search_results == 5
    assert settings.search_depth == "advanced"
    assert settings.relevance_score_threshold == 0.5
    assert settings.fuzzy_match_threshold == 80
    assert settings.date_proximity_days == 5


def test_env_var_override(monkeypatch):
    """ENRICHMENT_ prefixed env vars override defaults."""
    monkeypatch.setenv("ENRICHMENT_OUTPUT_DIR", "/custom/path")
    monkeypatch.setenv("ENRICHMENT_MAX_SEARCH_RESULTS", "10")
    monkeypatch.setenv("ENRICHMENT_RELEVANCE_SCORE_THRESHOLD", "0.7")
    settings = Settings()
    assert settings.output_dir == "/custom/path"
    assert settings.max_search_results == 10
    assert settings.relevance_score_threshold == 0.7


def test_prefix_required(monkeypatch):
    """Env vars without ENRICHMENT_ prefix do not override defaults."""
    monkeypatch.setenv("OUTPUT_DIR", "/should/be/ignored")
    monkeypatch.setenv("MAX_SEARCH_RESULTS", "99")
    settings = Settings()
    assert settings.output_dir == "output/enrichment"
    assert settings.max_search_results == 5
