"""Tests for the Merge Node.

Tests cover three helper functions (check_reference_match,
check_articles_match, extract_fields) and the merge_node orchestrator.
LLM calls are mocked via MagicMock.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableConfig

from src.agents.state import (
    Article,
    ConfidenceLevel,
    DatasetType,
    EnrichmentState,
    FieldExtraction,
    MediaFeatureField,
    MergeExtractionResponse,
    PipelineStage,
    SearchStrategyType,
    ValidationResult,
)
from src.merge.merge_node import (
    check_articles_match,
    check_reference_match,
    extract_fields,
    merge_node,
)

# --- Fixtures ---


@pytest.fixture
def base_field_extraction() -> FieldExtraction:
    """FieldExtraction with weapon=handgun and full metadata."""
    return FieldExtraction(
        field_name="weapon",
        value="handgun",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com"],
        source_quotes=["the victim use a handgun to shoot the officer Martinez"],
        llm_reasoning="The type of the weapon is listed in the extracted content.",
    )


@pytest.fixture
def base_field_extraction_none() -> FieldExtraction:
    """FieldExtraction with weapon=None (no value found)."""
    return FieldExtraction(
        field_name="weapon", value=None, confidence=ConfidenceLevel.PENDING
    )


@pytest.fixture
def base_field_extraction_minor_diff() -> FieldExtraction:
    """FieldExtraction with weapon=handguns (fuzzy match to handgun)."""
    return FieldExtraction(
        field_name="weapon",
        value="handguns",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example_minor_diff.com"],
        source_quotes=[
            "the person use handguns to attack Martinez, the police officer"
        ],
        llm_reasoning="The type of the weapon is listed in the extracted content.",
    )


@pytest.fixture()
def base_field_extraction_conflict() -> FieldExtraction:
    """FieldExtraction with weapon=knife (conflicts with handgun)."""
    return FieldExtraction(
        field_name="weapon",
        value="knife",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example_conflict.com"],
        source_quotes=[
            "the assailant wielded a knife to stab the officers Smith and Chen"
        ],
        llm_reasoning="The type of the weapon is listed in the extracted content.",
    )


@pytest.fixture
def base_state() -> EnrichmentState:
    """State with all incident fields with search & validation results (after Validate)."""
    return EnrichmentState(
        incident_id="142",
        dataset_type=DatasetType.CIVILIANS_SHOT,
        location="Houston",
        incident_date=date(2018, 3, 15),
        officer_name="James Rodriguez",
        civilian_name="John Doe",
        severity="fatal",
        current_stage=PipelineStage.SEARCH,
        next_strategy=SearchStrategyType.EXACT_MATCH,
        retrieved_articles=[
            Article(
                url="https://example.com/article1",
                title="Houston officer James Rodriguez involved in shooting of John Doe",
                snippet="A Houston police officer fatally shot John Doe during a traffic stop on March 15.",
                content="A Houston police officer identified as James Rodriguez fatally shot John Doe, 34, during a traffic stop on the city's east side on March 15, 2018. Witnesses say the encounter escalated quickly after Doe exited his vehicle.",
                source_name="CBS",
                relevance_score=0.9,
                published_date=date(2018, 3, 15),
            ),
            Article(
                url="https://example.com/article2",
                title="Houston fatal police shooting, victim is John Doe",
                snippet="Police in Houston, TX confirmed a fatal officer-involved shooting on March 14.",
                content="Police in Houston, TX confirmed a fatal officer-involved shooting near downtown on Wednesday. The victim was identified as John Doe. Officials have not yet released the name of the officer involved.",
                source_name="NBC",
                relevance_score=0.7,
                published_date=date(2018, 3, 14),
            ),
        ],
        validation_results=[
            ValidationResult(
                article=Article(
                    url="https://example.com/article1",
                    title="Houston officer James Rodriguez involved in shooting of John Doe",
                    snippet="A Houston police officer fatally shot John Doe during a traffic stop on March 15.",
                    content="A Houston police officer identified as James Rodriguez fatally shot John Doe, 34, during a traffic stop on the city's east side on March 15, 2018. Witnesses say the encounter escalated quickly after Doe exited his vehicle.",
                    source_name="CBS",
                    relevance_score=0.9,
                    published_date=date(2018, 3, 15),
                ),
                date_match=True,
                location_match=True,
                victim_name_match=True,
                passed=True,
            ),
            ValidationResult(
                article=Article(
                    url="https://example.com/article2",
                    title="Houston fatal police shooting, victim is John Doe",
                    snippet="Police in Houston, TX confirmed a fatal officer-involved shooting on March 14.",
                    content="Police in Houston, TX confirmed a fatal officer-involved shooting near downtown on Wednesday. The victim was identified as John Doe. Officials have not yet released the name of the officer involved.",
                    source_name="NBC",
                    relevance_score=0.7,
                    published_date=date(2018, 3, 14),
                ),
                date_match=True,
                location_match=True,
                victim_name_match=True,
                passed=True,
            ),
        ],
    )


@pytest.fixture()
def base_article() -> Article:
    """Single article for extract_fields tests."""
    return Article(
        url="https://example.com/article",
        title="Houston fatal police shooting, victim is John Doe, officer name is Martinez",
        snippet="Police in Houston, TX confirmed a fatal shooting by officer police involved handgun on March 14.",
        content="Police in Houston, TX confirmed a fatal shooting by officer police involved handgun on March 14 near downtown on Wednesday. The victim was identified as John Doe.",
        source_name="NBC",
        relevance_score=0.7,
        published_date=date(2018, 3, 14),
    )


@pytest.fixture
def base_field_extraction_officer_name() -> FieldExtraction:
    """FieldExtraction for officer_name field."""
    return FieldExtraction(
        field_name="officer_name", value="Martinez", confidence=ConfidenceLevel.PENDING
    )


@pytest.fixture
def base_field_extraction_location_detail() -> FieldExtraction:
    """FieldExtraction for location_detail field."""
    return FieldExtraction(
        field_name="location_detail",
        value="Houston",
        confidence=ConfidenceLevel.PENDING,
    )


# --- check_reference_match tests ---


def test_check_reference_match(base_field_extraction: FieldExtraction) -> None:
    """Test reference matching: None ref, fuzzy match, mismatch, and non-string ref."""
    # reference is none
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), None
    )
    assert result[0] is True
    assert result[1].value == "handgun"

    # match
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), "handguns"
    )
    assert result[0] is True
    assert result[1].value == "handguns"

    # no match
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), "hammer"
    )
    assert result[0] is False
    assert result[1] is None

    # reference is not string
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), date(2025, 3, 18)
    )
    assert result[0] is False
    assert result[1] is None


# --- check_articles_match tests ---


def test_check_articles_match_no_articles(
    base_field_extraction_none: FieldExtraction,
) -> None:
    """Empty list and all-None values both return (False, None)."""
    result = check_articles_match(MediaFeatureField.WEAPON, [])
    assert result[0] is False
    assert result[1] is None

    result = check_articles_match(
        MediaFeatureField.WEAPON,
        [base_field_extraction_none, base_field_extraction_none],
    )
    assert result[0] is False
    assert result[1] is None


def test_check_articles_match_single_article(
    base_field_extraction: FieldExtraction,
    base_field_extraction_none: FieldExtraction,
) -> None:
    """Single non-null extraction returns (True, result) with MEDIUM confidence."""
    result = check_articles_match(
        MediaFeatureField.WEAPON,
        [
            base_field_extraction_none.model_copy(),
            base_field_extraction.model_copy(),
            base_field_extraction_none.model_copy(),
        ],
    )
    assert result[0] is True
    assert result[1].confidence == ConfidenceLevel.MEDIUM
    assert result[1].value == "handgun"


def test_check_articles_match_all_agree(
    base_field_extraction: FieldExtraction,
    base_field_extraction_none: FieldExtraction,
) -> None:
    """Multiple articles with identical values return HIGH confidence."""
    result = check_articles_match(
        MediaFeatureField.WEAPON,
        [
            base_field_extraction.model_copy(),
            base_field_extraction.model_copy(),
            base_field_extraction_none.model_copy(),
        ],
    )
    assert result[0] is True
    assert result[1].confidence == ConfidenceLevel.HIGH
    assert result[1].value == "handgun"


def test_check_articles_match_minor_diff(
    base_field_extraction: FieldExtraction,
    base_field_extraction_minor_diff: FieldExtraction,
    base_field_extraction_none: FieldExtraction,
) -> None:
    """Fuzzy-similar values resolve to most common with MEDIUM confidence."""
    result = check_articles_match(
        MediaFeatureField.WEAPON,
        [
            base_field_extraction_minor_diff.model_copy(),
            base_field_extraction_minor_diff.model_copy(),
            base_field_extraction.model_copy(),
            base_field_extraction_none.model_copy(),
        ],
    )
    assert result[0] is True
    assert result[1].confidence == ConfidenceLevel.MEDIUM
    assert result[1].value == "handguns"


def test_check_articles_match_conflict(
    base_field_extraction: FieldExtraction,
    base_field_extraction_conflict: FieldExtraction,
    base_field_extraction_none: FieldExtraction,
) -> None:
    """Completely different values return (False, None)."""
    result = check_articles_match(
        MediaFeatureField.WEAPON,
        [
            base_field_extraction_conflict.model_copy(),
            base_field_extraction.model_copy(),
            base_field_extraction_none.model_copy(),
        ],
    )
    assert result[0] is False
    assert result[1] is None


# --- extract_fields tests ---


def test_extract_fields_errors(
    base_article: Article, base_field_extraction: FieldExtraction
) -> None:
    """LLM API error returns empty dict instead of raising."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = Exception(
        "API error"
    )
    result = extract_fields(base_article, mock_llm, list(MediaFeatureField))
    assert result == {}


def test_extract_fields_empty_results() -> None:
    """Article with None content returns empty dict without calling LLM."""
    article = Article(
        url="...",
        title="...",
        content=None,
        snippet="",
        published_date=None,
        source_name="",
        relevance_score=0,
    )
    result = extract_fields(article, MagicMock(), list(MediaFeatureField))
    assert result == {}


def test_extract_fields_happy_path(
    base_article: Article,
    base_field_extraction: FieldExtraction,
    base_field_extraction_officer_name: FieldExtraction,
    base_field_extraction_location_detail: FieldExtraction,
) -> None:
    """Successful extraction maps field_name to FieldExtraction and sets metadata."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        MergeExtractionResponse(
            extractions=[
                base_field_extraction,
                base_field_extraction_officer_name,
                base_field_extraction_location_detail,
            ]
        )
    )
    result = extract_fields(
        base_article,
        mock_llm,
        [
            MediaFeatureField.WEAPON,
            MediaFeatureField.OFFICER_NAME,
            MediaFeatureField.LOCATION_DETAIL,
        ],
    )
    assert result["weapon"].field_name == "weapon"
    assert result["officer_name"].field_name == "officer_name"
    assert result["location_detail"].field_name == "location_detail"
    assert result["weapon"].value == "handgun"
    assert result["officer_name"].value == "Martinez"
    assert result["location_detail"].value == "Houston"
    assert result["weapon"].sources == ["https://example.com/article"]
    assert result["weapon"].confidence == ConfidenceLevel.PENDING


# --- merge_node tests ---


def _make_extraction(field_name: str, value: str | None) -> FieldExtraction:
    """Helper to build a FieldExtraction with minimal boilerplate."""
    return FieldExtraction(
        field_name=field_name,
        value=value,
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/article"],
        source_quotes=[f"Quote about {field_name}"],
        llm_reasoning=f"Reasoning for {field_name}",
    )


def _build_mock_llm(extractions_per_article: list[list[FieldExtraction]]) -> MagicMock:
    """Build a mock LLM that returns different extractions per article.

    Args:
        extractions_per_article: List of extraction lists, one per article.
            Each inner list becomes a MergeExtractionResponse.
    """
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        MergeExtractionResponse(extractions=exts) for exts in extractions_per_article
    ]
    return mock_llm


class TestMergeNode:
    """Tests for the merge_node orchestrator."""

    def test_happy_path_articles_agree(self, base_state: EnrichmentState) -> None:
        """Both articles return same values, names match DB reference."""
        shared_extractions = [
            _make_extraction("officer_name", "James Rodriguez"),
            _make_extraction("civilian_name", "John Doe"),
            _make_extraction("weapon", "handgun"),
            _make_extraction("civilian_age", "34"),
        ]
        mock_llm = _build_mock_llm([shared_extractions, shared_extractions])

        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = merge_node(base_state, config)

        assert result.current_stage == PipelineStage.MERGE
        assert result.error_message is None
        assert result.conflicting_fields == []
        # All 4 fields should be in extracted_fields
        extracted_names = [e.field_name for e in result.extracted_fields]
        assert "weapon" in extracted_names
        assert "officer_name" in extracted_names
        assert "civilian_name" in extracted_names
        assert "civilian_age" in extracted_names
        # Names should be overwritten with DB reference
        officer = next(
            e for e in result.extracted_fields if e.field_name == "officer_name"
        )
        assert officer.value == "James Rodriguez"
        civilian = next(
            e for e in result.extracted_fields if e.field_name == "civilian_name"
        )
        assert civilian.value == "John Doe"
        # Confidence should be HIGH (both articles agree)
        weapon = next(e for e in result.extracted_fields if e.field_name == "weapon")
        assert weapon.confidence == ConfidenceLevel.HIGH

    def test_reference_conflict(self, base_state: EnrichmentState) -> None:
        """Articles agree with each other but disagree with DB reference."""
        shared_extractions = [
            _make_extraction("officer_name", "Mike Thompson"),
            _make_extraction("weapon", "handgun"),
        ]
        mock_llm = _build_mock_llm([shared_extractions, shared_extractions])

        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = merge_node(base_state, config)

        assert result.current_stage == PipelineStage.MERGE
        # officer_name should be in conflicting_fields (doesn't match DB)
        assert MediaFeatureField.OFFICER_NAME in result.conflicting_fields
        # officer_name should still be in extracted_fields
        extracted_names = [e.field_name for e in result.extracted_fields]
        assert "officer_name" in extracted_names

    def test_articles_conflict(self, base_state: EnrichmentState) -> None:
        """Articles disagree on a field value."""
        article1_extractions = [
            _make_extraction("weapon", "handgun"),
            _make_extraction("civilian_name", "John Doe"),
        ]
        article2_extractions = [
            _make_extraction("weapon", "rifle"),
            _make_extraction("civilian_name", "John Doe"),
        ]
        mock_llm = _build_mock_llm([article1_extractions, article2_extractions])

        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = merge_node(base_state, config)

        assert result.current_stage == PipelineStage.MERGE
        assert MediaFeatureField.WEAPON in result.conflicting_fields
        # weapon should NOT be in extracted_fields (conflict)
        extracted_names = [e.field_name for e in result.extracted_fields]
        assert "weapon" not in extracted_names
        # civilian_name should still work
        assert "civilian_name" in extracted_names

    def test_llm_error_gracefully_skips(self, base_state: EnrichmentState) -> None:
        """LLM failure in extract_fields returns empty dict, merge continues."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception(
            "API error"
        )

        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = merge_node(base_state, config)

        # Helpers catch the error -- orchestrator completes normally
        assert result.current_stage == PipelineStage.MERGE
        assert result.error_message is None
        assert result.extracted_fields == []
        assert result.conflicting_fields == []
