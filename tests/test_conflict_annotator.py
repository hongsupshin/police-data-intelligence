"""Unit tests for the conflict annotator (src/synthesize/conflict_annotator.py)."""

from datetime import date
from unittest.mock import MagicMock

from src.agents.state import (
    Article,
    ConflictAnnotation,
    ConflictType,
    DatasetType,
    EnrichmentState,
    FieldConflict,
)
from src.synthesize.conflict_annotator import _build_prompt, annotate_conflicts


def _state() -> EnrichmentState:
    return EnrichmentState(
        incident_id="51",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        civilian_name="John Suspect",
        incident_date=date(2019, 5, 1),
        location="Killeen",
        civilian_age=41,
        civilian_gender="male",
    )


def _conflicts() -> list[FieldConflict]:
    return [
        FieldConflict(
            field_name="civilian_age",
            conflict_type=ConflictType.ARTICLES_DISAGREE,
            values=["41", "18"],
            sources=[["https://x.com/a"], ["https://x.com/b"]],
        ),
        FieldConflict(
            field_name="location_detail",
            conflict_type=ConflictType.ARTICLES_DISAGREE,
            values=["100 block of Elm St, Killeen", "Killeen County Jail"],
            sources=[["https://x.com/a"], ["https://x.com/b"]],
        ),
    ]


def _article(
    content: str = "A 41-year-old man was shot on Elm St in Killeen on May 1.",
) -> Article:
    return Article(
        url="https://x.com/a",
        title="Killeen shooting",
        snippet=content[:40],
        content=content,
        source_name="CBS",
        relevance_score=0.8,
        published_date=date(2019, 5, 2),
    )


class TestBuildPrompt:
    """The annotator prompt carries the conflicting values, anchors, articles, rules."""

    def test_includes_conflicts_anchors_articles(self) -> None:
        prompt = _build_prompt(_state(), _conflicts(), [_article()])
        assert "John Suspect" in prompt
        assert "Killeen" in prompt
        assert "civilian_age" in prompt
        assert "['41', '18']" in prompt  # candidate values surfaced
        assert "location_detail" in prompt
        assert "A 41-year-old man was shot" in prompt
        assert "[Article 0]" in prompt

    def test_includes_never_assert_rule(self) -> None:
        prompt = _build_prompt(_state(), _conflicts(), [_article()])
        assert "NEVER assert a person's race" in prompt

    def test_truncates_long_content(self) -> None:
        prompt = _build_prompt(_state(), _conflicts(), [_article(content="x" * 5000)])
        assert "x" * 700 in prompt
        assert "x" * 701 not in prompt


class TestAnnotateConflicts:
    """annotate_conflicts returns the structured advisory note from the LLM."""

    def test_returns_structured_annotation(self) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            ConflictAnnotation(note="Ages differ (41 vs 18); 18 likely a bystander.")
        )
        result = annotate_conflicts(mock_llm, _state(), _conflicts(), [_article()])
        assert isinstance(result, ConflictAnnotation)
        assert "41 vs 18" in result.note
        mock_llm.with_structured_output.assert_called_once_with(ConflictAnnotation)
        prompt_arg = (
            mock_llm.with_structured_output.return_value.invoke.call_args.args[0]
        )
        assert "civilian_age" in prompt_arg
