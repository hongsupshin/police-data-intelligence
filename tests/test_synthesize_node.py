"""Tests for the Synthesize Node.

Tests cover three helper functions (check_reference_match,
check_articles_match, extract_fields) and the synthesize_node orchestrator.
LLM calls are mocked via MagicMock.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from src.agents.state import (
    Article,
    ConfidenceLevel,
    ConflictAnnotation,
    ConflictType,
    DatasetType,
    EnrichmentState,
    FieldConflict,
    FieldExtraction,
    MediaFeatureField,
    MergeExtractionResponse,
    PipelineStage,
    SearchStrategyType,
    ValidationResult,
)
from src.config import Settings
from src.synthesize.race_verifier import RaceVerificationVerdict
from src.synthesize.relevance_judge import RelevanceVerdict
from src.synthesize.synthesize_node import (
    check_articles_match,
    check_reference_match,
    extract_fields,
    normalize_name,
    normalize_race,
    synthesize_node,
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


@patch("src.synthesize.synthesize_node.weapons_match")
def test_check_reference_match(
    mock_weapons_match: MagicMock, base_field_extraction: FieldExtraction
) -> None:
    """Test reference matching: None ref, weapon match, weapon mismatch, and non-string ref."""
    # reference is none (weapon branch not reached)
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), None
    )
    assert result[0] is True
    assert result[1].value == "handgun"

    # weapon match
    mock_weapons_match.return_value = True
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), "handguns"
    )
    assert result[0] is True
    assert result[1].value == "handguns"

    # weapon mismatch
    mock_weapons_match.return_value = False
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), "hammer"
    )
    assert result[0] is False
    assert result[1] is None

    # weapon case-insensitive match (handled by weapons_match)
    mock_weapons_match.return_value = True
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), "HANDGUN"
    )
    assert result[0] is True
    assert result[1].value == "HANDGUN"

    # weapon reference is not string (weapons_match still applies)
    mock_weapons_match.return_value = False
    result = check_reference_match(
        MediaFeatureField.WEAPON, base_field_extraction.model_copy(), date(2025, 3, 18)
    )
    assert result[0] is False
    assert result[1] is None


# --- check_articles_match tests ---


def test_check_articles_match_no_articles(
    base_field_extraction_none: FieldExtraction,
) -> None:
    """Empty list and all-None values both return (True, None) — no data, not conflict."""
    result = check_articles_match(MediaFeatureField.WEAPON, [])
    assert result[0] is True
    assert result[1] is None

    result = check_articles_match(
        MediaFeatureField.WEAPON,
        [base_field_extraction_none, base_field_extraction_none],
    )
    assert result[0] is True
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


@patch("src.synthesize.synthesize_node.weapons_match", return_value=True)
def test_check_articles_match_minor_diff(
    mock_weapons_match: MagicMock,
    base_field_extraction: FieldExtraction,
    base_field_extraction_minor_diff: FieldExtraction,
    base_field_extraction_none: FieldExtraction,
) -> None:
    """Semantically similar weapons resolve to most common with MEDIUM confidence."""
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


@patch("src.synthesize.synthesize_node.weapons_match", return_value=False)
def test_check_articles_match_conflict(
    mock_weapons_match: MagicMock,
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


# --- normalize_name tests ---


def test_normalize_name_strips_honorific() -> None:
    """Military rank prefix is removed."""
    assert normalize_name("Master Sgt. Alva Joe Gwinn") == "alva joe gwinn"


def test_normalize_name_strips_quotes() -> None:
    """Nickname quotes are removed."""
    assert normalize_name("Alva 'Joe' Gwinn") == "alva joe gwinn"


def test_normalize_name_strips_title_and_quotes() -> None:
    """Both title and quotes are removed together."""
    assert normalize_name("Officer James 'Jim' Smith") == "james jim smith"


def test_normalize_name_plain_name() -> None:
    """Plain name is just lowercased."""
    assert normalize_name("John Doe") == "john doe"


# --- check_articles_match with name normalization ---


def test_check_articles_match_name_with_honorific() -> None:
    """Name variants with honorifics converge after normalization."""
    plain = FieldExtraction(
        field_name="civilian_name",
        value="Alva Joe Gwinn",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    with_rank = FieldExtraction(
        field_name="civilian_name",
        value="Master Sgt. Alva Joe Gwinn",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.CIVILIAN_NAME, [plain, with_rank]
    )
    assert matched is True
    assert winner is not None
    assert winner.confidence == ConfidenceLevel.MEDIUM


def test_check_articles_match_name_with_quotes() -> None:
    """Name variants with nickname quotes converge after normalization."""
    plain = FieldExtraction(
        field_name="civilian_name",
        value="Alva Joe Gwinn",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    quoted = FieldExtraction(
        field_name="civilian_name",
        value="Alva 'Joe' Gwinn",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.CIVILIAN_NAME, [plain, quoted]
    )
    assert matched is True
    assert winner is not None


def test_check_articles_match_genuinely_different_names() -> None:
    """Genuinely different names still conflict even for name fields."""
    name_a = FieldExtraction(
        field_name="civilian_name",
        value="John Doe",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    name_b = FieldExtraction(
        field_name="civilian_name",
        value="Jane Smith",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.CIVILIAN_NAME, [name_a, name_b]
    )
    assert matched is False
    assert winner is None


# --- check_articles_match with outcome / time normalization ---


def test_check_articles_match_outcome_consensus() -> None:
    """Different fatal phrasings converge on the fatal category."""
    killed = FieldExtraction(
        field_name="outcome",
        value="The suspect was shot and killed.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    died = FieldExtraction(
        field_name="outcome",
        value="The man died at the scene.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.OUTCOME, [killed, died]
    )
    assert matched is True
    assert winner is not None
    assert winner.confidence == ConfidenceLevel.MEDIUM


def test_check_articles_match_outcome_conflict() -> None:
    """Fatal vs non-fatal stays a conflict (no spurious consensus)."""
    killed = FieldExtraction(
        field_name="outcome",
        value="The suspect was killed.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    survived = FieldExtraction(
        field_name="outcome",
        value="The victim survived after being shot.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.OUTCOME, [killed, survived]
    )
    assert matched is False
    assert winner is None


def test_check_articles_match_outcome_unparseable_breaks_consensus() -> None:
    """An unparseable outcome (canonical None) prevents a commit."""
    killed = FieldExtraction(
        field_name="outcome",
        value="The suspect was killed.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    unclear = FieldExtraction(
        field_name="outcome",
        value="The outcome was not reported.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.OUTCOME, [killed, unclear]
    )
    assert matched is False
    assert winner is None


def test_check_articles_match_time_consensus() -> None:
    """A clock time and a period word converge on the same bucket."""
    clock = FieldExtraction(
        field_name="time_of_day",
        value="around 1 p.m.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    period = FieldExtraction(
        field_name="time_of_day",
        value="in the afternoon",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.TIME_OF_DAY, [clock, period]
    )
    assert matched is True
    assert winner is not None
    assert winner.confidence == ConfidenceLevel.MEDIUM


def test_check_articles_match_time_conflict() -> None:
    """Different time buckets stay a conflict."""
    morning = FieldExtraction(
        field_name="time_of_day",
        value="early in the morning",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    night = FieldExtraction(
        field_name="time_of_day",
        value="around 11 p.m.",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.TIME_OF_DAY, [morning, night]
    )
    assert matched is False
    assert winner is None


# --- extract_fields tests ---


def test_extract_fields_errors(
    base_article: Article, base_field_extraction: FieldExtraction
) -> None:
    """LLM API error returns empty dict instead of raising."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = Exception(
        "API error"
    )
    result = extract_fields(
        base_article, mock_llm, list(MediaFeatureField), DatasetType.CIVILIANS_SHOT
    )
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
    result = extract_fields(
        article, MagicMock(), list(MediaFeatureField), DatasetType.CIVILIANS_SHOT
    )
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
        DatasetType.CIVILIANS_SHOT,
    )
    assert result["weapon"].field_name == "weapon"
    assert result["officer_name"].field_name == "officer_name"
    assert result["location_detail"].field_name == "location_detail"
    assert result["weapon"].value == "handgun"
    assert result["officer_name"].value == "Martinez"
    assert result["location_detail"].value == "Houston"
    assert result["weapon"].sources == ["https://example.com/article"]
    assert result["weapon"].confidence == ConfidenceLevel.PENDING


def test_extract_fields_officers_prompt(base_article: Article) -> None:
    """officers_shot uses the officer-framed prompt (suspect vs officer-victim)."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        MergeExtractionResponse(extractions=[])
    )
    extract_fields(
        base_article, mock_llm, list(MediaFeatureField), DatasetType.OFFICERS_SHOT
    )
    prompt = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "officer-involved shooting" in prompt
    assert "SUSPECT/shooter" in prompt
    assert "the victim who was shot" in prompt  # officer outcome override


def test_extract_fields_civilians_prompt_unchanged(base_article: Article) -> None:
    """civilians_shot keeps the civilian-centric prompt (no officer framing)."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        MergeExtractionResponse(extractions=[])
    )
    extract_fields(
        base_article, mock_llm, list(MediaFeatureField), DatasetType.CIVILIANS_SHOT
    )
    prompt = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "police shooting incident article" in prompt
    assert "SUSPECT/shooter" not in prompt
    assert "officer-involved shooting" not in prompt


def test_extract_fields_anchored_civilians(base_article: Article) -> None:
    """civilians_shot with a target civilian prepends the TARGET CIVILIAN anchor."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        MergeExtractionResponse(extractions=[])
    )
    extract_fields(
        base_article,
        mock_llm,
        list(MediaFeatureField),
        DatasetType.CIVILIANS_SHOT,
        {"name": "Jane Roe", "age": 41, "gender": "FEMALE"},
    )
    prompt = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "TARGET CIVILIAN" in prompt
    assert "Jane Roe" in prompt
    assert "41" in prompt
    assert "FEMALE" in prompt
    assert "NEVER combine multiple people" in prompt
    # single-subject path must stay dominant (don't over-null named-less records)
    assert "extract theirs" in prompt


def test_extract_fields_anchor_name_absent(base_article: Article) -> None:
    """When the record victim has no name (~25%), the anchor falls back to age+gender."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        MergeExtractionResponse(extractions=[])
    )
    extract_fields(
        base_article,
        mock_llm,
        list(MediaFeatureField),
        DatasetType.CIVILIANS_SHOT,
        {"name": None, "age": 29, "gender": "MALE"},
    )
    prompt = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "unknown / not recorded" in prompt
    assert "29" in prompt
    assert "match on age + gender" in prompt


def test_extract_fields_civilians_no_anchor_when_none(base_article: Article) -> None:
    """No TARGET CIVILIAN block when no anchor is supplied (backward compatible)."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        MergeExtractionResponse(extractions=[])
    )
    extract_fields(
        base_article, mock_llm, list(MediaFeatureField), DatasetType.CIVILIANS_SHOT
    )
    prompt = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "TARGET CIVILIAN" not in prompt


def test_extract_fields_officers_ignores_anchor(base_article: Article) -> None:
    """officers_shot never gets the civilian anchor block, even if one is passed."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        MergeExtractionResponse(extractions=[])
    )
    extract_fields(
        base_article,
        mock_llm,
        list(MediaFeatureField),
        DatasetType.OFFICERS_SHOT,
        {"name": "Jane Roe", "age": 41, "gender": "FEMALE"},
    )
    prompt = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "TARGET CIVILIAN" not in prompt


# --- synthesize_node tests ---


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


class TestSynthesizeNode:
    """Tests for the synthesize_node orchestrator."""

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
        result = synthesize_node(base_state, config)

        assert result.current_stage == PipelineStage.SYNTHESIZE
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

    def test_civilian_anchor_reaches_extraction_prompt(
        self, base_state: EnrichmentState
    ) -> None:
        """synthesize_node anchors civilian extraction on the record's victim."""
        state = base_state.model_copy(
            update={"civilian_age": 34, "civilian_gender": "MALE"}
        )
        mock_llm = _build_mock_llm([[_make_extraction("civilian_age", "34")]] * 2)
        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        synthesize_node(state, config)
        prompt = (
            mock_llm.with_structured_output.return_value.invoke.call_args_list[0][0][0]
        )
        assert "TARGET CIVILIAN" in prompt
        assert "John Doe" in prompt  # base_state.civilian_name
        assert "34" in prompt
        assert "MALE" in prompt

    def test_reference_conflict(self, base_state: EnrichmentState) -> None:
        """Articles agree with each other but disagree with DB reference."""
        shared_extractions = [
            _make_extraction("officer_name", "Mike Thompson"),
            _make_extraction("weapon", "handgun"),
        ]
        mock_llm = _build_mock_llm([shared_extractions, shared_extractions])

        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = synthesize_node(base_state, config)

        assert result.current_stage == PipelineStage.SYNTHESIZE
        # officer_name should be in conflicting_fields (doesn't match DB)
        assert len(result.conflicting_fields) == 1
        conflict = result.conflicting_fields[0]
        assert isinstance(conflict, FieldConflict)
        assert conflict.field_name == MediaFeatureField.OFFICER_NAME
        assert conflict.conflict_type == ConflictType.REFERENCE_MISMATCH
        assert conflict.values == ["Mike Thompson"]
        assert conflict.reference_value == "James Rodriguez"
        # officer_name should still be in extracted_fields
        extracted_names = [e.field_name for e in result.extracted_fields]
        assert "officer_name" in extracted_names

    @patch("src.synthesize.synthesize_node.weapons_match", return_value=False)
    def test_articles_conflict(
        self, mock_weapons_match: MagicMock, base_state: EnrichmentState
    ) -> None:
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
        result = synthesize_node(base_state, config)

        assert result.current_stage == PipelineStage.SYNTHESIZE
        conflict_names = [c.field_name for c in result.conflicting_fields]
        assert MediaFeatureField.WEAPON in conflict_names
        weapon_conflict = next(
            c for c in result.conflicting_fields
            if c.field_name == MediaFeatureField.WEAPON
        )
        assert weapon_conflict.conflict_type == ConflictType.ARTICLES_DISAGREE
        assert set(weapon_conflict.values) == {"handgun", "rifle"}
        assert weapon_conflict.reference_value is None
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
        result = synthesize_node(base_state, config)

        # Helpers catch the error -- orchestrator completes normally
        assert result.current_stage == PipelineStage.SYNTHESIZE
        assert result.error_message is None
        assert result.extracted_fields == []
        assert result.conflicting_fields == []

    def test_only_validated_articles_processed(
        self, base_state: EnrichmentState
    ) -> None:
        """Merge skips articles that did not pass validation."""
        # Add a third unvalidated article to retrieved_articles
        unvalidated_article = Article(
            url="https://example.com/unvalidated",
            title="Completely unrelated article",
            snippet="This article is about a different incident.",
            content="This article is about a completely different incident in another state.",
            source_name="FOX",
            relevance_score=0.3,
            published_date=date(2018, 3, 20),
        )
        base_state.retrieved_articles.append(unvalidated_article)
        # validation_results still only has 2 passed entries (article1, article2)

        shared_extractions = [
            _make_extraction("weapon", "handgun"),
            _make_extraction("civilian_name", "John Doe"),
        ]
        # Only 2 LLM calls expected (validated articles), NOT 3
        mock_llm = _build_mock_llm([shared_extractions, shared_extractions])

        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = synthesize_node(base_state, config)

        assert result.current_stage == PipelineStage.SYNTHESIZE
        assert result.error_message is None
        # LLM was called exactly 2 times (once per validated article)
        assert mock_llm.with_structured_output.return_value.invoke.call_count == 2

    def test_no_validated_articles_produces_empty(
        self, base_state: EnrichmentState
    ) -> None:
        """When no articles pass validation, merge produces empty results."""
        # Mark all validation results as failed
        for vr in base_state.validation_results:
            vr.passed = False

        mock_llm = MagicMock()
        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = synthesize_node(base_state, config)

        assert result.current_stage == PipelineStage.SYNTHESIZE
        assert result.extracted_fields == []
        assert result.conflicting_fields == []
        # LLM should never be called
        mock_llm.with_structured_output.return_value.invoke.assert_not_called()

    def test_all_null_extractions_with_db_reference(
        self, base_state: EnrichmentState
    ) -> None:
        """All-null extractions with non-null DB reference: no crash, no conflict."""
        null_extractions = [
            _make_extraction("officer_name", None),
            _make_extraction("civilian_name", None),
            _make_extraction("weapon", None),
        ]
        mock_llm = _build_mock_llm([null_extractions, null_extractions])

        config = RunnableConfig({"configurable": {"llm_client": mock_llm}})
        result = synthesize_node(base_state, config)

        assert result.current_stage == PipelineStage.SYNTHESIZE
        assert result.error_message is None
        # No conflicts — nothing to compare against DB reference
        assert result.conflicting_fields == []
        # officer_name should NOT be in extracted_fields (all null)
        extracted_names = [e.field_name for e in result.extracted_fields]
        assert "officer_name" not in extracted_names
        assert "civilian_name" not in extracted_names


# --- normalize_race tests ---


def test_normalize_race_african_american() -> None:
    """'African American' normalizes to 'black'."""
    assert normalize_race("African American") == "black"


def test_normalize_race_caucasian() -> None:
    """'Caucasian' normalizes to 'white'."""
    assert normalize_race("Caucasian") == "white"


def test_normalize_race_latino() -> None:
    """'Latino' normalizes to 'hispanic'."""
    assert normalize_race("Latino") == "hispanic"


def test_normalize_race_asian_maps_to_other() -> None:
    """'Asian' maps to TJI's 'other' bucket (TJI has no Asian category)."""
    assert normalize_race("Asian") == "other"


def test_normalize_race_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped."""
    assert normalize_race("  Black  ") == "black"


def test_normalize_race_hispanic_latino_male() -> None:
    """'Hispanic/Latino male' strips gender and matches hispanic."""
    assert normalize_race("Hispanic/Latino male") == "hispanic"


def test_normalize_race_hispanic_or_latino() -> None:
    """'Hispanic or Latino' matches hispanic."""
    assert normalize_race("Hispanic or Latino") == "hispanic"


def test_normalize_race_african_american_slash_black() -> None:
    """'African-American/Black' matches black."""
    assert normalize_race("African-American/Black") == "black"


def test_normalize_race_nationality_iranian() -> None:
    """Unrecognized nationality defaults to 'other'."""
    assert normalize_race("Iranian") == "other"


def test_normalize_race_nationality_egyptian() -> None:
    """Unrecognized nationality defaults to 'other'."""
    assert normalize_race("Egyptian") == "other"


def test_normalize_race_unknown_defaults_to_other() -> None:
    """Completely unrecognized values default to 'other'."""
    assert normalize_race("Martian") == "other"


# --- check_articles_match with race normalization ---


def test_check_articles_match_race_synonyms() -> None:
    """Race synonyms ('African American' vs 'Black') converge after normalization."""
    val_a = FieldExtraction(
        field_name="civilian_race",
        value="African American",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    val_b = FieldExtraction(
        field_name="civilian_race",
        value="Black",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.CIVILIAN_RACE, [val_a, val_b]
    )
    assert matched is True
    assert winner is not None
    assert winner.confidence == ConfidenceLevel.MEDIUM


def test_check_articles_match_race_genuinely_different() -> None:
    """Genuinely different races still conflict."""
    val_a = FieldExtraction(
        field_name="civilian_race",
        value="White",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    val_b = FieldExtraction(
        field_name="civilian_race",
        value="Black",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.CIVILIAN_RACE, [val_a, val_b]
    )
    assert matched is False
    assert winner is None


# --- check_articles_match with partial_ratio and field thresholds ---


@patch("src.synthesize.synthesize_node.weapons_match", return_value=True)
def test_check_articles_match_weapon_embedding_similarity(
    mock_weapons_match: MagicMock,
) -> None:
    """Weapon field uses embedding-based similarity instead of rapidfuzz."""
    val_a = FieldExtraction(
        field_name="weapon",
        value="9mm handgun",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a1"],
    )
    val_b = FieldExtraction(
        field_name="weapon",
        value="handgun",
        confidence=ConfidenceLevel.PENDING,
        sources=["https://example.com/a2"],
    )
    matched, winner = check_articles_match(
        MediaFeatureField.WEAPON, [val_a, val_b]
    )
    assert matched is True
    assert winner is not None


class TestRelevanceGate:
    """The officer relevance judge: flag-gated, officers-only, fail-open."""

    @staticmethod
    def _exts() -> list[FieldExtraction]:
        return [
            _make_extraction("officer_name", "James Rodriguez"),
            _make_extraction("weapon", "handgun"),
        ]

    def _mock_llm(self, verdict=None, raise_on_judge: bool = False) -> MagicMock:
        """Mock LLM: an extraction per article, then optionally a judge verdict."""
        side = [
            MergeExtractionResponse(extractions=self._exts()),
            MergeExtractionResponse(extractions=self._exts()),
        ]
        if raise_on_judge:
            side.append(RuntimeError("judge boom"))
        elif verdict is not None:
            side.append(verdict)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = side
        return mock_llm

    @staticmethod
    def _config(mock_llm: MagicMock, flag: bool) -> RunnableConfig:
        return RunnableConfig(
            {
                "configurable": {
                    "llm_client": mock_llm,
                    "settings": Settings(enable_relevance_gate=flag),
                }
            }
        )

    @staticmethod
    def _officers(base_state: EnrichmentState) -> EnrichmentState:
        state = base_state.model_copy(deep=True)
        state.dataset_type = DatasetType.OFFICERS_SHOT
        return state

    @staticmethod
    def _judge_called(mock_llm: MagicMock) -> bool:
        models = [c.args[0] for c in mock_llm.with_structured_output.call_args_list]
        return RelevanceVerdict in models

    def test_veto_sets_flag(self, base_state: EnrichmentState) -> None:
        """Officers + flag on + not-relevant verdict -> relevance_vetoed True."""
        mock_llm = self._mock_llm(
            RelevanceVerdict(relevant_any=False, reasoning="off-topic")
        )
        result = synthesize_node(self._officers(base_state), self._config(mock_llm, True))
        assert result.relevance_vetoed is True
        assert result.extracted_fields  # a real (would-be) completion was judged

    def test_relevant_no_veto(self, base_state: EnrichmentState) -> None:
        """Officers + flag on + relevant verdict -> no veto."""
        mock_llm = self._mock_llm(
            RelevanceVerdict(relevant_any=True, reasoning="match")
        )
        result = synthesize_node(self._officers(base_state), self._config(mock_llm, True))
        assert result.relevance_vetoed is False
        assert self._judge_called(mock_llm)

    def test_flag_off_skips_judge(self, base_state: EnrichmentState) -> None:
        """Flag off -> the judge never runs."""
        mock_llm = self._mock_llm()
        result = synthesize_node(self._officers(base_state), self._config(mock_llm, False))
        assert result.relevance_vetoed is False
        assert not self._judge_called(mock_llm)

    def test_civilians_skip_judge(self, base_state: EnrichmentState) -> None:
        """Civilians + flag on -> officers-only gate skips the judge."""
        mock_llm = self._mock_llm()
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert result.relevance_vetoed is False
        assert not self._judge_called(mock_llm)

    def test_fail_open_on_judge_error(self, base_state: EnrichmentState) -> None:
        """A judge error must not block the completion (fail-open, no veto)."""
        mock_llm = self._mock_llm(raise_on_judge=True)
        result = synthesize_node(self._officers(base_state), self._config(mock_llm, True))
        assert result.relevance_vetoed is False
        assert result.extracted_fields


class TestRaceVerification:
    """The civilian race verifier: flag-gated, civilians-only, fail-open, nulls one field."""

    @staticmethod
    def _exts(with_race: bool = True) -> list[FieldExtraction]:
        exts = [_make_extraction("weapon", "handgun")]
        if with_race:
            exts.insert(0, _make_extraction("civilian_race", "Black"))
        return exts

    def _mock_llm(
        self,
        verdict: RaceVerificationVerdict | None = None,
        raise_on_verify: bool = False,
        with_race: bool = True,
    ) -> MagicMock:
        """Mock LLM: an extraction per article (2), then optionally a verify verdict."""
        exts = self._exts(with_race=with_race)
        side: list = [
            MergeExtractionResponse(extractions=exts),
            MergeExtractionResponse(extractions=exts),
        ]
        if raise_on_verify:
            side.append(RuntimeError("verify boom"))
        elif verdict is not None:
            side.append(verdict)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = side
        return mock_llm

    @staticmethod
    def _config(mock_llm: MagicMock, flag: bool) -> RunnableConfig:
        # relevance gate off so the race verifier is isolated (officers test).
        return RunnableConfig(
            {
                "configurable": {
                    "llm_client": mock_llm,
                    "settings": Settings(
                        enable_race_verification=flag, enable_relevance_gate=False
                    ),
                }
            }
        )

    @staticmethod
    def _officers(base_state: EnrichmentState) -> EnrichmentState:
        state = base_state.model_copy(deep=True)
        state.dataset_type = DatasetType.OFFICERS_SHOT
        return state

    @staticmethod
    def _verify_called(mock_llm: MagicMock) -> bool:
        models = [c.args[0] for c in mock_llm.with_structured_output.call_args_list]
        return RaceVerificationVerdict in models

    @staticmethod
    def _race(state: EnrichmentState) -> FieldExtraction | None:
        return next(
            (f for f in state.extracted_fields if f.field_name == "civilian_race"), None
        )

    def test_unsupported_nulls_race(self, base_state: EnrichmentState) -> None:
        """Civilians + flag on + unsupported -> civilian_race removed, rest survive."""
        mock_llm = self._mock_llm(
            RaceVerificationVerdict(supported=False, reasoning="not stated in source")
        )
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert self._verify_called(mock_llm)
        assert self._race(result) is None  # nulled
        assert result.extracted_fields  # other fields (weapon) survive

    def test_supported_keeps_race(self, base_state: EnrichmentState) -> None:
        """Civilians + flag on + supported -> civilian_race kept."""
        mock_llm = self._mock_llm(
            RaceVerificationVerdict(
                supported=True, quote="a Black man", reasoning="explicitly stated"
            )
        )
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert self._verify_called(mock_llm)
        race = self._race(result)
        assert race is not None and race.value == "Black"

    def test_flag_off_skips_verifier(self, base_state: EnrichmentState) -> None:
        """Flag off -> the verifier never runs; race kept."""
        mock_llm = self._mock_llm()
        result = synthesize_node(base_state, self._config(mock_llm, False))
        assert not self._verify_called(mock_llm)
        assert self._race(result) is not None

    def test_officers_skip_verifier(self, base_state: EnrichmentState) -> None:
        """Officers + flag on -> civilians-only gate skips the verifier."""
        mock_llm = self._mock_llm()
        synthesize_node(self._officers(base_state), self._config(mock_llm, True))
        assert not self._verify_called(mock_llm)

    def test_no_race_extraction_skips_verifier(self, base_state: EnrichmentState) -> None:
        """No civilian_race extracted -> the verifier is not called."""
        mock_llm = self._mock_llm(with_race=False)
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert not self._verify_called(mock_llm)
        assert self._race(result) is None

    def test_fail_open_on_error(self, base_state: EnrichmentState) -> None:
        """A verifier error must not null the race (fail-open)."""
        mock_llm = self._mock_llm(raise_on_verify=True)
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert self._verify_called(mock_llm)
        assert self._race(result) is not None  # kept on error


class TestConflictAnnotation:
    """Advisory conflict annotator: flag-gated, deep-conflicts-only, dataset-agnostic, fail-open."""

    def _mock_llm(
        self,
        art1_exts: list[FieldExtraction],
        art2_exts: list[FieldExtraction],
        annotation: ConflictAnnotation | None = None,
        raise_on_annotate: bool = False,
    ) -> MagicMock:
        """Mock LLM: an extraction per article (2), then optionally an annotation."""
        side: list = [
            MergeExtractionResponse(extractions=art1_exts),
            MergeExtractionResponse(extractions=art2_exts),
        ]
        if raise_on_annotate:
            side.append(RuntimeError("annotate boom"))
        elif annotation is not None:
            side.append(annotation)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = side
        return mock_llm

    @staticmethod
    def _config(mock_llm: MagicMock, flag: bool) -> RunnableConfig:
        # other gates off so the annotator is isolated.
        return RunnableConfig(
            {
                "configurable": {
                    "llm_client": mock_llm,
                    "settings": Settings(
                        enable_conflict_annotation=flag,
                        enable_relevance_gate=False,
                        enable_race_verification=False,
                    ),
                }
            }
        )

    @staticmethod
    def _annotate_called(mock_llm: MagicMock) -> bool:
        models = [c.args[0] for c in mock_llm.with_structured_output.call_args_list]
        return ConflictAnnotation in models

    def test_deep_conflict_sets_annotation(self, base_state: EnrichmentState) -> None:
        """A deep (civilian_age) conflict -> the annotator runs and sets the note."""
        a1 = [_make_extraction("weapon", "handgun"), _make_extraction("civilian_age", "51")]
        a2 = [_make_extraction("weapon", "handgun"), _make_extraction("civilian_age", "57")]
        mock_llm = self._mock_llm(a1, a2, ConflictAnnotation(note="ages differ: 51 vs 57"))
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert self._annotate_called(mock_llm)
        assert result.conflict_annotation is not None
        assert result.conflict_annotation.note

    def test_dataset_agnostic_officers(self, base_state: EnrichmentState) -> None:
        """The annotator is dataset-agnostic -> also fires on officers_shot."""
        state = base_state.model_copy(deep=True)
        state.dataset_type = DatasetType.OFFICERS_SHOT
        a1 = [_make_extraction("civilian_age", "51")]
        a2 = [_make_extraction("civilian_age", "57")]
        mock_llm = self._mock_llm(a1, a2, ConflictAnnotation(note="x"))
        result = synthesize_node(state, self._config(mock_llm, True))
        assert self._annotate_called(mock_llm)
        assert result.conflict_annotation is not None

    def test_flag_off_skips(self, base_state: EnrichmentState) -> None:
        """Flag off -> the annotator never runs."""
        a1 = [_make_extraction("civilian_age", "51")]
        a2 = [_make_extraction("civilian_age", "57")]
        mock_llm = self._mock_llm(a1, a2)
        result = synthesize_node(base_state, self._config(mock_llm, False))
        assert not self._annotate_called(mock_llm)
        assert result.conflict_annotation is None

    def test_shallow_only_conflict_skips(self, base_state: EnrichmentState) -> None:
        """A conflict only on a non-deep field (circumstance) -> annotator skipped."""
        a1 = [
            _make_extraction("weapon", "handgun"),
            _make_extraction("circumstance", "Officer responded to a domestic call."),
        ]
        a2 = [
            _make_extraction("weapon", "handgun"),
            _make_extraction("circumstance", "A robbery suspect fled on foot downtown."),
        ]
        mock_llm = self._mock_llm(a1, a2)
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert not self._annotate_called(mock_llm)
        assert result.conflict_annotation is None

    def test_fail_open_on_error(self, base_state: EnrichmentState) -> None:
        """An annotator error must not break the node (fail-open, no annotation)."""
        a1 = [_make_extraction("civilian_age", "51")]
        a2 = [_make_extraction("civilian_age", "57")]
        mock_llm = self._mock_llm(a1, a2, raise_on_annotate=True)
        result = synthesize_node(base_state, self._config(mock_llm, True))
        assert self._annotate_called(mock_llm)
        assert result.conflict_annotation is None  # fail-open


class TestRaceTaxonomyFlag:
    """civilian_race_taxonomy is set for any committed race (civilians + officers)."""

    @staticmethod
    def _config(
        mock_llm: MagicMock, *, race_verification: bool = False
    ) -> RunnableConfig:
        return RunnableConfig(
            {
                "configurable": {
                    "llm_client": mock_llm,
                    "settings": Settings(
                        enable_race_verification=race_verification,
                        enable_relevance_gate=False,
                    ),
                }
            }
        )

    @staticmethod
    def _mock_llm(
        race_value: str, verdict: RaceVerificationVerdict | None = None
    ) -> MagicMock:
        exts = [
            _make_extraction("civilian_race", race_value),
            _make_extraction("weapon", "handgun"),
        ]
        side: list = [
            MergeExtractionResponse(extractions=exts),
            MergeExtractionResponse(extractions=exts),
        ]
        if verdict is not None:
            side.append(verdict)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = side
        return mock_llm

    def test_civilians_committed_race_sets_flag(
        self, base_state: EnrichmentState
    ) -> None:
        """Verify on + supported -> race kept -> taxonomy flag computed."""
        mock_llm = self._mock_llm(
            "Black", RaceVerificationVerdict(supported=True, reasoning="stated")
        )
        result = synthesize_node(base_state, self._config(mock_llm, race_verification=True))
        assert result.civilian_race_taxonomy is not None
        assert result.civilian_race_taxonomy.tji_bucket == "black"
        assert result.civilian_race_taxonomy.diverges is False

    def test_officers_divergent_race_flagged(
        self, base_state: EnrichmentState
    ) -> None:
        """Any dataset: a divergent committed race is flagged (officers, no verify)."""
        state = base_state.model_copy(deep=True)
        state.dataset_type = DatasetType.OFFICERS_SHOT
        result = synthesize_node(state, self._config(self._mock_llm("Asian")))
        assert result.civilian_race_taxonomy is not None
        assert result.civilian_race_taxonomy.tji_bucket == "other"
        assert result.civilian_race_taxonomy.divergence_type == "race_absent_from_scheme"
        assert result.civilian_race_taxonomy.diverges is True

    def test_nulled_race_no_flag(self, base_state: EnrichmentState) -> None:
        """Verify on + unsupported -> race nulled -> no taxonomy flag."""
        mock_llm = self._mock_llm(
            "Black", RaceVerificationVerdict(supported=False, reasoning="unsourced")
        )
        result = synthesize_node(base_state, self._config(mock_llm, race_verification=True))
        assert result.civilian_race_taxonomy is None
