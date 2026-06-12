"""Tests for enrichment pipeline settings."""

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_default_values():
    """All fields have expected defaults when no env vars are set."""
    settings = Settings()
    assert settings.output_dir == "output/enrichment"
    assert settings.max_search_results == 10
    assert settings.search_depth == "advanced"
    assert settings.search_window_back_days == 14
    assert settings.search_window_forward_days == 60
    assert settings.relevance_score_threshold == 0.5
    assert settings.fuzzy_match_threshold == 80
    assert settings.date_proximity_days == 5
    assert settings.enable_relevance_gate is True


def test_env_var_override(monkeypatch):
    """ENRICHMENT_ prefixed env vars override defaults."""
    monkeypatch.setenv("ENRICHMENT_OUTPUT_DIR", "/custom/path")
    monkeypatch.setenv("ENRICHMENT_MAX_SEARCH_RESULTS", "15")
    monkeypatch.setenv("ENRICHMENT_SEARCH_WINDOW_BACK_DAYS", "21")
    monkeypatch.setenv("ENRICHMENT_SEARCH_WINDOW_FORWARD_DAYS", "90")
    monkeypatch.setenv("ENRICHMENT_RELEVANCE_SCORE_THRESHOLD", "0.7")
    settings = Settings()
    assert settings.output_dir == "/custom/path"
    assert settings.max_search_results == 15
    assert settings.search_window_back_days == 21
    assert settings.search_window_forward_days == 90
    assert settings.relevance_score_threshold == 0.7


def test_prefix_required(monkeypatch):
    """Env vars without ENRICHMENT_ prefix do not override defaults."""
    monkeypatch.setenv("OUTPUT_DIR", "/should/be/ignored")
    monkeypatch.setenv("MAX_SEARCH_RESULTS", "99")
    settings = Settings()
    assert settings.output_dir == "output/enrichment"
    assert settings.max_search_results == 10


def test_search_window_covers_date_proximity():
    """Search window-back must be at least the validation date tolerance.

    Guarantees search never excludes an article that the validate node
    would accept on date proximity.
    """
    settings = Settings()
    assert settings.search_window_back_days >= settings.date_proximity_days


def test_search_window_validator_rejects_too_narrow():
    """Settings raises if search_window_back_days < date_proximity_days."""
    with pytest.raises(ValidationError):
        Settings(search_window_back_days=3, date_proximity_days=5)


def test_enable_relevance_gate_override(monkeypatch):
    """enable_relevance_gate (on by default) flips off via constructor and env."""
    assert Settings(enable_relevance_gate=False).enable_relevance_gate is False
    assert Settings(enable_relevance_gate=True).enable_relevance_gate is True
    monkeypatch.setenv("ENRICHMENT_ENABLE_RELEVANCE_GATE", "false")
    assert Settings().enable_relevance_gate is False
