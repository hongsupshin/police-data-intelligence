"""Unit tests for the officer relevance judge (src/synthesize/relevance_judge.py)."""

from datetime import date
from unittest.mock import MagicMock

from src.agents.state import Article, DatasetType, EnrichmentState
from src.synthesize.relevance_judge import (
    RelevanceVerdict,
    _build_prompt,
    judge_relevance,
)


def _state() -> EnrichmentState:
    return EnrichmentState(
        incident_id="51",
        dataset_type=DatasetType.OFFICERS_SHOT,
        officer_name="Officer Jane Doe",
        civilian_name="John Suspect",
        incident_date=date(2019, 5, 1),
        location="Killeen",
        severity="injury",
    )


def _article(
    url: str = "https://x.com/a",
    title: str = "Killeen officer shot during standoff",
    content: str = "An officer was shot in Killeen on May 1, 2019 during a standoff.",
) -> Article:
    return Article(
        url=url,
        title=title,
        snippet=content[:40],
        content=content,
        source_name="CBS",
        relevance_score=0.8,
        published_date=date(2019, 5, 2),
    )


class TestBuildPrompt:
    """The judge prompt carries the incident anchors and article content."""

    def test_includes_anchors_and_article_content(self) -> None:
        prompt = _build_prompt(_state(), [_article()])
        assert "Officer Jane Doe" in prompt
        assert "John Suspect" in prompt
        assert "Killeen" in prompt
        assert "An officer was shot in Killeen" in prompt
        assert "[0] title:" in prompt

    def test_truncates_long_content(self) -> None:
        long = "x" * 5000
        prompt = _build_prompt(_state(), [_article(content=long)])
        # content is truncated to 600 chars in the block
        assert "x" * 600 in prompt
        assert "x" * 601 not in prompt


class TestJudgeRelevance:
    """judge_relevance returns the structured verdict from the LLM."""

    def test_returns_structured_verdict(self) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            RelevanceVerdict(relevant_any=False, relevant_indices=[], reasoning="off-topic")
        )
        verdict = judge_relevance(mock_llm, _state(), [_article()])
        assert verdict.relevant_any is False
        mock_llm.with_structured_output.assert_called_once_with(RelevanceVerdict)
        prompt_arg = (
            mock_llm.with_structured_output.return_value.invoke.call_args.args[0]
        )
        assert "Officer Jane Doe" in prompt_arg
