"""Race-string taxonomy: map free-text race to TJI's coarse buckets + flag divergence.

TJI's ``civilians.race`` ground truth has only four buckets — WHITE / BLACK /
HISPANIC / OTHER — with no Asian category and no nationalities. This module is the
single source of truth for collapsing a free-text race string to one of those
buckets (used **for comparison only**: eval scoring, fairness grouping, and
cross-article agreement — never to overwrite the stored value), and for
classifying *where the string diverges* from that coarse scheme.

``normalize_race`` answers "what would TJI code this?" (the matching key).
``classify_race`` additionally records the divergence (a deterministic precursor
to the step-3 LLM taxonomy-audit typer). The two are kept consistent:
``normalize_race(v) == classify_race(v).tji_bucket``.
"""

import re

from src.agents.state import RaceTaxonomyFlag

_GENDER_WORDS = re.compile(r"\b(male|female|man|woman)\b")

# Priority-ordered: (pattern, tji_bucket, divergence_type). First match wins.
# Synonyms first (no divergence), then the collapses, then default uncategorized.
_RACE_TAXONOMY: list[tuple[re.Pattern[str], str, str]] = [
    # 1. Synonyms of a TJI bucket -> no divergence.
    (re.compile(r"\b(black|african[- ]?american|african)\b"), "black", "none"),
    (re.compile(r"\b(hispanic|latino|latina|latinx)\b"), "hispanic", "none"),
    (re.compile(r"\b(white|caucasian|european[- ]?american)\b"), "white", "none"),
    # 2. Latin-American nationality -> hispanic (ethnicity-as-nationality collapse).
    (
        re.compile(
            r"\b(mexican|honduran|salvadoran|el salvadoran|guatemalan|cuban|"
            r"colombian|venezuelan|dominican|puerto rican|nicaraguan|peruvian|"
            r"ecuadorian|bolivian|chilean|argentin(?:e|ian)|costa rican|"
            r"panamanian|paraguayan|uruguayan)\b"
        ),
        "hispanic",
        "nationality_as_ethnicity",
    ),
    # 3. Races absent from TJI's scheme -> other.
    (
        re.compile(
            r"\b(asian|chinese|japanese|korean|vietnamese|filipin[oa]|hmong|"
            r"cambodian|laotian|thai|south asian|east asian|southeast asian|"
            r"pakistani|bangladeshi|pacific islander|native hawaiian|samoan)\b"
        ),
        "other",
        "race_absent_from_scheme",
    ),
    (
        re.compile(
            r"\b(middle eastern|arab|iranian|persian|egyptian|lebanese|syrian|"
            r"turkish|iraqi|saudi|moroccan)\b"
        ),
        "other",
        "race_absent_from_scheme",
    ),
    (
        re.compile(
            r"\b(native american|american indian|indigenous|alaska native|"
            r"first nations)\b"
        ),
        "other",
        "race_absent_from_scheme",
    ),
    (
        re.compile(r"\b(multiracial|biracial|mixed[- ]?race|two or more)\b"),
        "other",
        "race_absent_from_scheme",
    ),
]


def classify_race(value: str) -> RaceTaxonomyFlag:
    """Classify a free-text race string against TJI's coarse 4-bucket scheme.

    Strips gender words, then matches the curated taxonomy in priority order.
    Unrecognized strings map to ``other`` with ``uncategorized`` divergence.
    Ambiguous bare terms (e.g. "Indian": South-Asian vs American-Indian) are
    intentionally not in the table, so they land in ``uncategorized`` for human
    disambiguation rather than being mis-bucketed.

    Args:
        value: Raw race/ethnicity string (e.g. "Asian", "Honduran", "Black male").

    Returns:
        A RaceTaxonomyFlag with the TJI bucket, divergence type, and whether the
        source is more specific than the bucket.
    """
    lowered = value.strip().lower()
    lowered = _GENDER_WORDS.sub("", lowered).strip()
    lowered = re.sub(r"\s+", " ", lowered)
    for pattern, bucket, divergence_type in _RACE_TAXONOMY:
        if pattern.search(lowered):
            return RaceTaxonomyFlag(
                extracted=value,
                tji_bucket=bucket,
                divergence_type=divergence_type,
                diverges=divergence_type != "none",
            )
    return RaceTaxonomyFlag(
        extracted=value,
        tji_bucket="other",
        divergence_type="uncategorized",
        diverges=True,
    )


def normalize_race(value: str) -> str:
    """Collapse a free-text race string to its TJI bucket (the matching key).

    Returns one of ``white``, ``black``, ``hispanic``, ``other``. This is the
    comparison key for scoring/agreement; it deliberately mirrors TJI's coarse
    scheme (so "Asian" -> "other", "Honduran" -> "hispanic"). Granularity is
    carried by :func:`classify_race`, not this value.

    Examples:
        >>> normalize_race("African-American/Black")
        'black'
        >>> normalize_race("Honduran")
        'hispanic'
        >>> normalize_race("Asian")
        'other'
        >>> normalize_race("Iranian")
        'other'
    """
    return classify_race(value).tji_bucket
