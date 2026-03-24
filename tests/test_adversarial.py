"""Adversarial hallucination detection tests for the enrichment pipeline.

Tests fabricated (non-existent) police shooting incidents to verify the
pipeline correctly escalates instead of hallucinating structured fields.

Three scenarios:
    A. Obscure location — search returns 0 results, retries exhaust → escalate
    B. Major city — articles about wrong person pass validation, synthesize
       catches reference mismatch → escalate or human review
    C. Plausible but fake — articles with wrong dates, validation rejects → escalate
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from src.agents.graph import build_graph
from src.agents.state import (
    Article,
    ConfidenceLevel,
    DatasetType,
    EnrichmentState,
    EscalationReason,
    FieldConflict,
    FieldExtraction,
    MergeExtractionResponse,
    PipelineStage,
    SearchAttempt,
    SearchStrategyType,
)
from src.config import Settings

# --- Shared helpers ---


def _fake_load(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.LOAD
    return state


def _fake_search_zero(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.SEARCH
    state.search_attempts.append(
        SearchAttempt(
            query="stub",
            strategy=state.next_strategy,
            num_results=0,
            avg_relevance_score=None,
        )
    )
    state.retrieved_articles = []
    return state


def _fake_search_wrong_person(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.SEARCH
    state.retrieved_articles = [
        Article(
            url="https://example.com/real-incident",
            title="Houston officer shoots suspect in east side confrontation",
            content=(
                "Houston police officer James Rodriguez fatally shot "
                "Marcus Johnson, 28, during a traffic stop on January 29, "
                "2018. Witnesses say the encounter escalated quickly after "
                "Johnson exited his vehicle with a handgun."
            ),
            snippet="Houston officer shoots suspect...",
            published_date=date(2018, 1, 29),
            relevance_score=0.8,
        )
    ]
    state.search_attempts.append(
        SearchAttempt(
            query="stub",
            strategy=state.next_strategy,
            num_results=1,
            avg_relevance_score=0.8,
        )
    )
    return state


def _fake_search_wrong_date(state: EnrichmentState) -> EnrichmentState:
    state.current_stage = PipelineStage.SEARCH
    state.retrieved_articles = [
        Article(
            url="https://example.com/unrelated",
            title="San Antonio police respond to shooting",
            content=(
                "San Antonio police officers responded to a shooting at "
                "a gas station on August 5, 2020. The suspect, identified "
                "as Marcus Allen, was taken into custody."
            ),
            snippet="San Antonio police respond to shooting...",
            published_date=date(2020, 8, 5),
            relevance_score=0.6,
        )
    ]
    state.search_attempts.append(
        SearchAttempt(
            query="stub",
            strategy=state.next_strategy,
            num_results=1,
            avg_relevance_score=0.6,
        )
    )
    return state


def assert_no_hallucination(
    result: dict,
    fabricated_civilian: str,
    fabricated_officer: str,
) -> None:
    """Verify fabricated names never appear as confirmed extractions."""
    for field in result.get("extracted_fields", []):
        if isinstance(field, FieldExtraction):
            fe = field
        elif isinstance(field, dict):
            fe = FieldExtraction(**field)
        else:
            continue
        if fe.field_name in ("civilian_name", "officer_name"):
            assert fe.value != fabricated_civilian, (
                f"Hallucination: pipeline confirmed fabricated civilian name '{fe.value}'"
            )
            assert fe.value != fabricated_officer, (
                f"Hallucination: pipeline confirmed fabricated officer name '{fe.value}'"
            )


# --- Fixtures ---


@pytest.fixture()
def fabricated_obscure() -> EnrichmentState:
    """Scenario A: fake names, obscure Texas city."""
    return EnrichmentState(
        incident_id="99901",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        officer_name="Reginald Q. Farnsworth",
        civilian_name="Bartholomew T. Quincy",
        incident_date=date(2019, 7, 14),
        location="Marfa",
        severity="fatal",
        current_stage=PipelineStage.LOAD,
        next_strategy=SearchStrategyType.EXACT_MATCH,
    )


@pytest.fixture()
def fabricated_major_city() -> EnrichmentState:
    """Scenario B: fake names, Houston, near real shooting dates."""
    return EnrichmentState(
        incident_id="99902",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        officer_name="Franklin D. Perriwinkle",
        civilian_name="Cornelius J. Montague",
        incident_date=date(2018, 1, 28),
        location="Houston",
        severity="fatal",
        current_stage=PipelineStage.LOAD,
        next_strategy=SearchStrategyType.EXACT_MATCH,
    )


@pytest.fixture()
def fabricated_plausible() -> EnrichmentState:
    """Scenario C: plausible names, San Antonio, no matching incident."""
    return EnrichmentState(
        incident_id="99903",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        officer_name="Michael Torres",
        civilian_name="David Williams",
        incident_date=date(2020, 6, 10),
        location="San Antonio",
        severity="non-fatal",
        current_stage=PipelineStage.LOAD,
        next_strategy=SearchStrategyType.EXACT_MATCH,
    )


# --- Unit tests ---


class TestAdversarialUnit:
    """Adversarial tests with all external dependencies mocked."""

    @pytest.mark.integration
    @patch("src.agents.graph.search_node")
    @patch("src.agents.graph.load_node")
    def test_zero_results_escalates_max_retries(
        self,
        mock_load,
        mock_search,
        fabricated_obscure: EnrichmentState,
    ) -> None:
        """Scenario A: search returns 0 results for all strategies → escalate."""
        mock_load.side_effect = _fake_load
        mock_search.side_effect = _fake_search_zero

        graph = build_graph(None)
        config = RunnableConfig({"configurable": {"settings": Settings()}})
        result = graph.invoke(fabricated_obscure, config)

        assert result["current_stage"] == PipelineStage.ESCALATE
        assert result["escalation_reason"] == EscalationReason.MAX_RETRIES
        assert result["extracted_fields"] == []
        assert result["requires_human_review"] is True
        # Search called 4 times (one per strategy)
        assert mock_search.call_count == 4
        assert_no_hallucination(
            result, "Bartholomew T. Quincy", "Reginald Q. Farnsworth"
        )

    @pytest.mark.integration
    @patch("src.agents.graph.search_node")
    @patch("src.agents.graph.load_node")
    def test_wrong_person_reference_mismatch(
        self,
        mock_load,
        mock_search,
        fabricated_major_city: EnrichmentState,
        tmp_path: Path,
    ) -> None:
        """Scenario B: articles about wrong person → reference mismatch detected."""
        mock_load.side_effect = _fake_load
        mock_search.side_effect = _fake_search_wrong_person

        # Mock LLM extracts "Marcus Johnson" (wrong person) from the article
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            MergeExtractionResponse(
                extractions=[
                    FieldExtraction(
                        field_name="civilian_name",
                        value="Marcus Johnson",
                        confidence=ConfidenceLevel.PENDING,
                        sources=["https://example.com/real-incident"],
                        source_quotes=["fatally shot Marcus Johnson, 28"],
                        llm_reasoning="Name from article",
                    ),
                    FieldExtraction(
                        field_name="weapon",
                        value="handgun",
                        confidence=ConfidenceLevel.PENDING,
                        sources=["https://example.com/real-incident"],
                        source_quotes=["exited his vehicle with a handgun"],
                        llm_reasoning="Weapon from article",
                    ),
                ]
            )
        )

        graph = build_graph(None)
        config = RunnableConfig(
            {
                "configurable": {
                    "settings": Settings(output_dir=str(tmp_path)),
                    "llm_client": mock_llm,
                }
            }
        )
        result = graph.invoke(fabricated_major_city, config)

        # Pipeline should flag the mismatch
        assert result["requires_human_review"] is True

        # civilian_name should be in conflicting_fields as REFERENCE_MISMATCH
        conflicts = result.get("conflicting_fields", [])
        conflict_names = []
        for c in conflicts:
            if isinstance(c, FieldConflict):
                conflict_names.append(c.field_name)
            elif isinstance(c, dict):
                conflict_names.append(c["field_name"])
        assert "civilian_name" in conflict_names

        # Fabricated names must not appear as confirmed extractions
        assert_no_hallucination(
            result, "Cornelius J. Montague", "Franklin D. Perriwinkle"
        )

    @pytest.mark.integration
    @patch("src.agents.graph.search_node")
    @patch("src.agents.graph.load_node")
    def test_wrong_person_partial_conflict_flags_review(
        self,
        mock_load,
        mock_search,
        fabricated_major_city: EnrichmentState,
        tmp_path: Path,
    ) -> None:
        """Scenario B variant: some fields agree, name conflicts → complete with review."""
        mock_load.side_effect = _fake_load
        mock_search.side_effect = _fake_search_wrong_person

        # LLM extracts wrong civilian name but weapon field is valid
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            MergeExtractionResponse(
                extractions=[
                    FieldExtraction(
                        field_name="civilian_name",
                        value="Marcus Johnson",
                        confidence=ConfidenceLevel.PENDING,
                        sources=["https://example.com/real-incident"],
                        source_quotes=["fatally shot Marcus Johnson"],
                        llm_reasoning="Name from article",
                    ),
                    FieldExtraction(
                        field_name="weapon",
                        value="handgun",
                        confidence=ConfidenceLevel.PENDING,
                        sources=["https://example.com/real-incident"],
                        source_quotes=["exited his vehicle with a handgun"],
                        llm_reasoning="Weapon from article",
                    ),
                    FieldExtraction(
                        field_name="civilian_age",
                        value="28",
                        confidence=ConfidenceLevel.PENDING,
                        sources=["https://example.com/real-incident"],
                        source_quotes=["Marcus Johnson, 28"],
                        llm_reasoning="Age from article",
                    ),
                ]
            )
        )

        graph = build_graph(None)
        config = RunnableConfig(
            {
                "configurable": {
                    "settings": Settings(output_dir=str(tmp_path)),
                    "llm_client": mock_llm,
                }
            }
        )
        result = graph.invoke(fabricated_major_city, config)

        # Has extracted fields (weapon, age) AND conflicts (civilian_name)
        # → coordinator routes to COMPLETE with requires_human_review=True
        assert result["requires_human_review"] is True
        assert result["current_stage"] in (
            PipelineStage.COMPLETE,
            PipelineStage.ESCALATE,
        )

        # Fabricated names must not appear as confirmed extractions
        assert_no_hallucination(
            result, "Cornelius J. Montague", "Franklin D. Perriwinkle"
        )

    @pytest.mark.integration
    @patch("src.agents.graph.search_node")
    @patch("src.agents.graph.load_node")
    def test_wrong_date_validation_rejects(
        self,
        mock_load,
        mock_search,
        fabricated_plausible: EnrichmentState,
    ) -> None:
        """Scenario C: articles with wrong dates fail validation → escalate."""
        mock_load.side_effect = _fake_load
        mock_search.side_effect = _fake_search_wrong_date

        graph = build_graph(None)
        config = RunnableConfig({"configurable": {"settings": Settings()}})
        result = graph.invoke(fabricated_plausible, config)

        assert result["current_stage"] == PipelineStage.ESCALATE
        assert result["escalation_reason"] == EscalationReason.MAX_RETRIES
        assert result["extracted_fields"] == []
        assert result["requires_human_review"] is True
        assert_no_hallucination(result, "David Williams", "Michael Torres")
