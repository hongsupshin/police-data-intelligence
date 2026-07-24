"""Field comparison layer shared by holdout evaluation and the discrepancy audit.

Extracted from ``src.eval.holdout`` so that comparison primitives can be
reused outside the holdout-evaluation context (where DB values are ground
truth). The discrepancy audit reuses the same comparators with inverted
semantics: the DB value is the audit object, not the reference.

Modules:
    Comparison primitives (EvalError, PipelineOutcome, MatchResult), the
    per-field comparison functions, field-to-comparator mappings for the
    holdout eval (``FIELD_COMPARATORS``) and the audit
    (``AUDIT_FIELD_COMPARATORS``), and the DB value fetch
    (``fetch_ground_truth``).

Note:
    ``FIELD_COMPARATORS`` is frozen for the holdout eval — adding fields to
    it would silently change holdout reports and invalidate saved baselines.
    Audit-only comparators (person names) are registered in the separate
    ``AUDIT_FIELD_COMPARATORS`` mapping instead.
"""

import unicodedata
from collections.abc import Callable
from datetime import time
from enum import StrEnum

from psycopg2.extensions import connection
from pydantic import BaseModel
from rapidfuzz import fuzz

from src.agents.state import (
    ConfidenceLevel,
    DatasetType,
    MediaFeatureField,
)
from src.field_normalizers import (
    detect_period as _detect_period,
)
from src.field_normalizers import (
    hour_in_bucket as _hour_in_bucket,
)
from src.field_normalizers import (
    normalize_outcome as _normalize_outcome_str,
)
from src.field_normalizers import (
    parse_hour as _parse_hour,
)
from src.race_taxonomy import normalize_race
from src.synthesize.synthesize_node import RAPIDFUZZ_THRESHOLD

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvalError(StrEnum):
    """Error categories for field evaluation."""

    NO_EXTRACTION = "no_extraction"
    PARSE_ERROR = "parse_error"
    NO_GROUND_TRUTH = "no_ground_truth"


class PipelineOutcome(StrEnum):
    """Pipeline terminal outcome."""

    COMPLETE = "complete"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MatchResult(BaseModel):
    """Single field comparison result.

    Attributes:
        field_name: MediaFeatureField name being compared.
        extracted_value: Value extracted by the pipeline (None if missing).
        ground_truth_value: Database value (None if missing). Named for the
            holdout eval; in the discrepancy audit the same slot holds the
            official record value under audit.
        exact_match: Whether values match exactly after normalization.
        fuzzy_match: Whether values match within fuzzy threshold.
        fuzzy_score: RapidFuzz score (None for exact-match-only fields).
        confidence: Extraction confidence level from the pipeline.
        error: Error category if comparison could not be performed.
    """

    field_name: str
    extracted_value: str | None = None
    ground_truth_value: str | None = None
    exact_match: bool = False
    fuzzy_match: bool = False
    fuzzy_score: float | None = None
    confidence: ConfidenceLevel | None = None
    error: EvalError | None = None


# ---------------------------------------------------------------------------
# Comparison Helpers
# ---------------------------------------------------------------------------

# Outcome / time-of-day normalizers (and TIME_PERIOD_BUCKETS) live in
# src.field_normalizers, shared with the pipeline's conflict resolver. They are
# imported at module top under their historical private names so the comparators
# below keep resolving unchanged.

# Race normalization lives in src.race_taxonomy (shared with the pipeline).
# Kept as a module-level alias so existing imports (e.g. gate.py) keep resolving.
# Note: after the taxonomy fix this maps to TJI's coarse 4-bucket scheme, so the
# civilian_race metric measures *fidelity to TJI's scheme*, not "correct race"
# (it cannot reward granularity — "Asian" scores as OTHER); the granular value
# and any divergence are carried by classify_race / RaceTaxonomyFlag.
_normalize_race = normalize_race


# ---------------------------------------------------------------------------
# Comparison Functions
# ---------------------------------------------------------------------------


def compare_age(
    extracted: str | None, ground_truth: int | None, field_name: str
) -> MatchResult:
    """Compare extracted age string against integer ground truth.

    Args:
        extracted: Age string from pipeline (e.g., "25").
        ground_truth: Age integer from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match only (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=str(ground_truth) if ground_truth is not None else None,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    try:
        extracted_int = int(extracted)
    except (ValueError, TypeError):
        result.error = EvalError.PARSE_ERROR
        return result
    result.exact_match = extracted_int == ground_truth
    result.fuzzy_match = result.exact_match
    return result


def compare_race(
    extracted: str | None, ground_truth: str | None, field_name: str
) -> MatchResult:
    """Compare extracted race against ground truth with alias normalization.

    Args:
        extracted: Race string from pipeline.
        ground_truth: Race string from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match after normalization (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    norm_extracted = _normalize_race(extracted)
    norm_truth = _normalize_race(ground_truth)
    result.exact_match = norm_extracted == norm_truth
    result.fuzzy_match = result.exact_match
    return result


def compare_weapon(
    extracted: str | None, ground_truth: str | None, field_name: str
) -> MatchResult:
    """Compare extracted weapon against ground truth using category normalization.

    Both values are normalized to canonical categories (HANDGUN, RIFLE,
    SHOTGUN, KNIFE, VEHICLE, OTHER, UNKNOWN) before comparison. Ground
    truth that normalizes to ``None`` (missing/empty) is treated as
    ``NO_GROUND_TRUTH``.

    Args:
        extracted: Weapon category from pipeline.
        ground_truth: Weapon description from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match and fuzzy_match set identically.
    """
    from src.synthesize.weapon_similarity import normalize_weapon

    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )
    gt_normalized = normalize_weapon(ground_truth)
    if gt_normalized is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    ext_normalized = normalize_weapon(extracted)
    if ext_normalized is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    matched = ext_normalized == gt_normalized
    result.exact_match = matched
    result.fuzzy_match = matched
    return result


def compare_location(
    extracted: str | None, ground_truth: str | None, field_name: str
) -> MatchResult:
    """Compare extracted location against ground truth using partial fuzzy matching.

    Uses fuzz.partial_ratio with RAPIDFUZZ_THRESHOLD, since extracted
    locations may have different granularity than database addresses.

    Args:
        extracted: Location string from pipeline.
        ground_truth: Address from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with fuzzy_score set.
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result
    score = fuzz.partial_ratio(extracted.lower(), ground_truth.lower())
    result.fuzzy_score = score
    result.exact_match = extracted.lower().strip() == ground_truth.lower().strip()
    result.fuzzy_match = score >= RAPIDFUZZ_THRESHOLD
    return result


def compare_time(
    extracted: str | None, ground_truth: time | None, field_name: str
) -> MatchResult:
    """Compare extracted time string against database TIME value.

    Primary: parse hour from extracted string, compare +/-2h to ground truth.
    Fallback: detect period keyword and check if ground truth hour falls
    within that period bucket.

    Args:
        extracted: Time description from pipeline (e.g., "2:30 PM", "evening").
        ground_truth: Time object from database.
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=str(ground_truth) if ground_truth is not None else None,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result

    gt_hour = ground_truth.hour

    # Primary: parse hour from extracted string
    extracted_hour = _parse_hour(extracted)
    if extracted_hour is not None:
        diff = abs(extracted_hour - gt_hour)
        # Handle wrap-around midnight
        diff = min(diff, 24 - diff)
        result.exact_match = diff <= 2
        result.fuzzy_match = result.exact_match
        return result

    # Fallback: period bucket matching
    period = _detect_period(extracted)
    if period is not None:
        result.exact_match = _hour_in_bucket(gt_hour, period)
        result.fuzzy_match = result.exact_match
        return result

    result.error = EvalError.PARSE_ERROR
    return result


def compare_outcome(
    extracted: str | None, ground_truth: bool | str | None, field_name: str
) -> MatchResult:
    """Compare extracted outcome against ground truth.

    Normalizes both sides to 'fatal' or 'non-fatal'.
    DB values: civilian_died=True -> fatal, officer_harm='DEATH' -> fatal.

    Args:
        extracted: Outcome description from pipeline.
        ground_truth: Boolean (civilian_died) or string (officer_harm).
        field_name: Field name for the result.

    Returns:
        MatchResult with exact_match (fuzzy_score=None).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=str(ground_truth) if ground_truth is not None else None,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result

    # Normalize ground truth
    if isinstance(ground_truth, bool):
        gt_normalized = "fatal" if ground_truth else "non-fatal"
    elif isinstance(ground_truth, str):
        gt_normalized = "fatal" if ground_truth.upper() == "DEATH" else "non-fatal"
    else:
        result.error = EvalError.PARSE_ERROR
        return result

    # Normalize extracted
    ext_normalized = _normalize_outcome_str(extracted)
    if ext_normalized is None:
        result.error = EvalError.PARSE_ERROR
        return result

    result.exact_match = ext_normalized == gt_normalized
    result.fuzzy_match = result.exact_match
    return result


# ---------------------------------------------------------------------------
# Person-Name Comparison (audit-only)
# ---------------------------------------------------------------------------

# Rank/title prefixes commonly attached to officer names in news coverage.
NAME_RANK_PREFIXES: frozenset[str] = frozenset(
    {
        "officer",
        "ofc",
        "off",
        "sgt",
        "sergeant",
        "deputy",
        "dep",
        "cpl",
        "corporal",
        "lt",
        "lieutenant",
        "det",
        "detective",
        "trooper",
        "capt",
        "captain",
        "chief",
        "constable",
        "sheriff",
        "agent",
        "ranger",
        "mr",
        "mrs",
        "ms",
    }
)

# Generational suffixes that carry no identity signal for comparison.
NAME_GENERATIONAL_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv"})


def normalize_person_name(name: str) -> list[str]:
    """Tokenize a person name for comparison.

    Folds diacritics (NFKD), lowercases, splits hyphens, strips punctuation,
    and drops rank/title prefixes and generational suffixes.

    Args:
        name: Raw person name (e.g., "Sgt. Armando Juarez Jr.").

    Returns:
        Ordered list of identity-bearing name tokens (may be empty if the
        input contains only titles/punctuation).

    Examples:
        >>> normalize_person_name("Sgt. Armando L. Juarez Jr.")
        ['armando', 'l', 'juarez']
        >>> normalize_person_name("IBARRA-RUIZ, José")
        ['ibarra', 'ruiz', 'jose']
    """
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower().replace("-", " ")
    cleaned = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in folded)
    tokens = cleaned.split()
    return [
        token
        for token in tokens
        if token not in NAME_RANK_PREFIXES and token not in NAME_GENERATIONAL_SUFFIXES
    ]


def _unmatched_tokens(tokens: list[str], other: list[str]) -> list[str]:
    """Return tokens with no counterpart in the other name.

    A token is matched if it appears in the other name verbatim, if it is a
    single-letter initial matching the first letter of one of the other
    name's tokens, or if the other name carries it as a matching initial.

    Args:
        tokens: Normalized tokens of one name.
        other: Normalized tokens of the other name.

    Returns:
        Subset of ``tokens`` with no counterpart in ``other``.
    """
    other_set = set(other)
    other_first_letters = {t[0] for t in other}
    unmatched = []
    for token in tokens:
        if token in other_set:
            continue
        if len(token) == 1 and token in other_first_letters:
            continue
        if any(len(o) == 1 and o == token[0] for o in other):
            continue
        unmatched.append(token)
    return unmatched


def _initials_contradict(tokens_a: list[str], tokens_b: list[str]) -> bool:
    """Detect a contradicting initial between two names.

    ``token_set_ratio`` under-weights single-letter tokens, so "B. Juarez"
    vs "Armando Juarez" scores above threshold despite the initial
    contradicting the full given name. A dangling initial on one side is a
    contradiction only when the other side has an unmatched token it
    disagrees with — a bare middle initial ("Armando L. Juarez" vs
    "Armando Juarez") is extra detail, not a contradiction.

    Args:
        tokens_a: Normalized tokens of one name.
        tokens_b: Normalized tokens of the other name.

    Returns:
        True if either side carries an initial contradicting the other.
    """
    unmatched_a = _unmatched_tokens(tokens_a, tokens_b)
    unmatched_b = _unmatched_tokens(tokens_b, tokens_a)
    a_has_dangling_initial = any(len(t) == 1 for t in unmatched_a)
    b_has_dangling_initial = any(len(t) == 1 for t in unmatched_b)
    return (a_has_dangling_initial and bool(unmatched_b)) or (
        b_has_dangling_initial and bool(unmatched_a)
    )


def _names_compatible(tokens_a: list[str], tokens_b: list[str]) -> bool:
    """Check whether the shorter name is consistent with the longer one.

    A less-specific name (surname only, initials, missing middle name) is
    *compatible* with a fuller name — it agrees without contradicting. Every
    token of the shorter name must either appear in the longer name or be a
    single-letter initial matching the first letter of one of its tokens.

    Args:
        tokens_a: Normalized tokens of one name.
        tokens_b: Normalized tokens of the other name.

    Returns:
        True if the shorter token list is compatible with the longer one.
    """
    shorter, longer = sorted((tokens_a, tokens_b), key=len)
    longer_set = set(longer)
    for token in shorter:
        if token in longer_set:
            continue
        if len(token) == 1 and any(other[0] == token for other in longer):
            continue
        return False
    return True


def compare_name(
    extracted: str | None, ground_truth: str | None, field_name: str
) -> MatchResult:
    """Compare an extracted person name against the database name.

    Accountability-oriented semantics: a mismatch is reported only on
    *contradiction*, not on lower specificity. A surname-only or
    initials-only news name that is consistent with the DB name counts as a
    fuzzy match (no flag); names whose identity-bearing tokens disagree
    beyond ``RAPIDFUZZ_THRESHOLD`` do not match.

    Args:
        extracted: Person name from pipeline (e.g., "Sgt. Duran").
        ground_truth: Person name from database (e.g., "John Duran").
        field_name: Field name for the result (officer_name/civilian_name).

    Returns:
        MatchResult with fuzzy_score set (token_set_ratio).
    """
    result = MatchResult(
        field_name=field_name,
        extracted_value=extracted,
        ground_truth_value=ground_truth,
    )
    if ground_truth is None:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if extracted is None:
        result.error = EvalError.NO_EXTRACTION
        return result

    ext_tokens = normalize_person_name(extracted)
    gt_tokens = normalize_person_name(ground_truth)
    if not gt_tokens:
        result.error = EvalError.NO_GROUND_TRUTH
        return result
    if not ext_tokens:
        result.error = EvalError.PARSE_ERROR
        return result

    result.exact_match = ext_tokens == gt_tokens
    score = fuzz.token_set_ratio(" ".join(ext_tokens), " ".join(gt_tokens))
    result.fuzzy_score = score
    consistent = score >= RAPIDFUZZ_THRESHOLD or _names_compatible(
        ext_tokens, gt_tokens
    )
    result.fuzzy_match = result.exact_match or (
        consistent and not _initials_contradict(ext_tokens, gt_tokens)
    )
    return result


# ---------------------------------------------------------------------------
# Field-to-Comparator Mappings
# ---------------------------------------------------------------------------

FIELD_COMPARATORS: dict[MediaFeatureField, Callable] = {
    MediaFeatureField.CIVILIAN_AGE: compare_age,
    MediaFeatureField.CIVILIAN_RACE: compare_race,
    MediaFeatureField.WEAPON: compare_weapon,
    MediaFeatureField.LOCATION_DETAIL: compare_location,
    MediaFeatureField.TIME_OF_DAY: compare_time,
    MediaFeatureField.OUTCOME: compare_outcome,
}

EVAL_FIELDS: set[MediaFeatureField] = set(FIELD_COMPARATORS.keys())

# Discrepancy-audit comparators: the holdout set plus person names. Kept
# separate from FIELD_COMPARATORS so holdout reports and frozen baselines
# are unaffected. Circumstance is deliberately absent: it is free text on
# the news side and only semi-structured in the DB, so a deterministic
# comparator would be noise (documented extension point).
AUDIT_FIELD_COMPARATORS: dict[MediaFeatureField, Callable] = {
    **FIELD_COMPARATORS,
    MediaFeatureField.OFFICER_NAME: compare_name,
    MediaFeatureField.CIVILIAN_NAME: compare_name,
}

# Ground truth DB column -> MediaFeatureField mapping per dataset
_CIVILIANS_GT_MAPPING: dict[str, MediaFeatureField] = {
    "age": MediaFeatureField.CIVILIAN_AGE,
    "race": MediaFeatureField.CIVILIAN_RACE,
    "weapon_reported_by_media": MediaFeatureField.WEAPON,
    "location_city": MediaFeatureField.LOCATION_DETAIL,
    "time_incident": MediaFeatureField.TIME_OF_DAY,
    "civilian_died": MediaFeatureField.OUTCOME,
}

_OFFICERS_GT_MAPPING: dict[str, MediaFeatureField] = {
    "age": MediaFeatureField.CIVILIAN_AGE,
    "race": MediaFeatureField.CIVILIAN_RACE,
    "location_city": MediaFeatureField.LOCATION_DETAIL,
    "officer_harm": MediaFeatureField.OUTCOME,
}

_DATASET_GT_MAPPINGS: dict[DatasetType, dict[str, MediaFeatureField]] = {
    DatasetType.CIVILIANS_SHOT: _CIVILIANS_GT_MAPPING,
    DatasetType.OFFICERS_SHOT: _OFFICERS_GT_MAPPING,
}


def comparators_for_dataset(
    dataset_type: DatasetType,
) -> dict[MediaFeatureField, Callable]:
    """Return the comparators evaluable for a dataset.

    Restricts FIELD_COMPARATORS to fields that have ground truth for the
    given dataset (e.g., officers_shot has no weapon/time_of_day columns),
    preserving FIELD_COMPARATORS ordering. This keeps officers reports from
    listing phantom civilian-only fields with zero evaluable incidents.

    Args:
        dataset_type: Which dataset is being evaluated.

    Returns:
        Subset of FIELD_COMPARATORS whose fields are mapped for the dataset.
    """
    mapped_fields = set(_DATASET_GT_MAPPINGS[dataset_type].values())
    return {
        field: comparator
        for field, comparator in FIELD_COMPARATORS.items()
        if field in mapped_fields
    }


# ---------------------------------------------------------------------------
# DB Queries
# ---------------------------------------------------------------------------


def fetch_ground_truth(
    conn: connection, incident_id: int, dataset_type: DatasetType
) -> dict[str, object]:
    """Fetch ground truth values for evaluable fields from the database.

    Uses dataset-specific queries to retrieve the 6 (civilians) or 4
    (officers) holdout fields that the pipeline never sees during
    extraction.

    Args:
        conn: Active PostgreSQL connection.
        incident_id: TJI incident identifier.
        dataset_type: Which dataset to query.

    Returns:
        Dict mapping MediaFeatureField values to ground truth values.
        Values may be None if the database column is NULL.

    Raises:
        KeyError: If incident_id not found in database.
    """
    cursor = conn.cursor()

    if dataset_type == DatasetType.CIVILIANS_SHOT:
        query = """
            SELECT c.age, c.race,
                   i.weapon_reported_by_media,
                   COALESCE(i.incident_city, i.incident_county) AS location_city,
                   i.time_incident, v.civilian_died
            FROM incidents_civilians_shot i
            LEFT JOIN incident_civilians_shot_victims v
                ON i.incident_id = v.incident_id
            LEFT JOIN civilians c ON v.civilian_id = c.civilian_id
            WHERE i.incident_id = %s LIMIT 1;
        """
        cursor.execute(query, (incident_id,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Incident {incident_id} not found in civilians_shot")
        columns = [
            "age", "race", "weapon_reported_by_media",
            "location_city", "time_incident", "civilian_died",
        ]
        raw = dict(zip(columns, row))
        return {
            _CIVILIANS_GT_MAPPING[col].value: val
            for col, val in raw.items()
        }
    else:
        query = """
            SELECT c.age, c.race,
                   COALESCE(i.incident_city, i.incident_county) AS location_city,
                   v.officer_harm
            FROM incidents_officers_shot i
            LEFT JOIN incident_officers_shot_shooters s
                ON i.incident_id = s.incident_id
            LEFT JOIN civilians c ON s.civilian_id = c.civilian_id
            LEFT JOIN incident_officers_shot_victims v
                ON i.incident_id = v.incident_id
            WHERE i.incident_id = %s LIMIT 1;
        """
        cursor.execute(query, (incident_id,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Incident {incident_id} not found in officers_shot")
        columns = ["age", "race", "location_city", "officer_harm"]
        raw = dict(zip(columns, row))
        return {
            _OFFICERS_GT_MAPPING[col].value: val
            for col, val in raw.items()
        }
