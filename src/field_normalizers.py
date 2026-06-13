"""Shared canonical normalizers for outcome and time-of-day fields.

These helpers map free-text field values to a coarse canonical form so that
both the enrichment pipeline (``src.synthesize``) and the evaluation
(``src.eval.holdout``) can reason about agreement at the granularity that
matters, rather than at the raw-string level. This module is a leaf (it imports
only the standard library), mirroring how ``src.race_taxonomy`` and
``src.synthesize.weapon_similarity`` are shared between those two layers.

- Outcome is canonicalized to ``"fatal"`` / ``"non-fatal"``.
- Time-of-day is canonicalized to a coarse period bucket
  (``morning`` / ``afternoon`` / ``evening`` / ``night``).
"""

import re

# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------

FATAL_KEYWORDS = {"killed", "died", "death", "fatal", "fatally"}
NON_FATAL_KEYWORDS = {"injured", "survived", "wounded", "non-fatal", "nonfatal"}


def normalize_outcome(value: str) -> str | None:
    """Normalize an outcome string to 'fatal' or 'non-fatal'.

    Args:
        value: Free-text outcome description (e.g. "shot and killed").

    Returns:
        'fatal', 'non-fatal', or None if no outcome keyword is recognized.

    Examples:
        >>> normalize_outcome("the suspect was killed")
        'fatal'
        >>> normalize_outcome("the officer was wounded but survived")
        'non-fatal'
    """
    lowered = value.strip().lower()
    # Check non-fatal first since "non-fatal" contains "fatal"
    if any(kw in lowered for kw in NON_FATAL_KEYWORDS):
        return "non-fatal"
    if any(kw in lowered for kw in FATAL_KEYWORDS):
        return "fatal"
    return None


# ---------------------------------------------------------------------------
# Time of day
# ---------------------------------------------------------------------------

TIME_PERIOD_BUCKETS: dict[str, tuple[int, int]] = {
    "morning": (6, 11),
    "afternoon": (12, 17),
    "evening": (18, 21),
    "night": (22, 5),
}


def parse_hour(time_str: str) -> int | None:
    """Extract hour from a time string (e.g., '2:30 PM', '14:30', '2 a.m.').

    Args:
        time_str: Free-text time description.

    Returns:
        Hour as a 0-23 integer, or None if unparseable.
    """
    # Try HH:MM patterns (24h or 12h)
    match = re.search(
        r"(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?", time_str, re.IGNORECASE
    )
    if match:
        hour = int(match.group(1))
        period = match.group(3)
        if period:
            period = period.lower().replace(".", "")
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
        return hour

    # Try bare hour with AM/PM (e.g., "2 PM", "11 a.m.")
    match = re.search(
        r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)", time_str, re.IGNORECASE
    )
    if match:
        hour = int(match.group(1))
        period = match.group(2).lower().replace(".", "")
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        return hour

    return None


def hour_in_bucket(hour: int, bucket_name: str) -> bool:
    """Check if an hour falls within a named time period bucket."""
    start, end = TIME_PERIOD_BUCKETS[bucket_name]
    if start <= end:
        return start <= hour <= end
    # Wraps around midnight (night: 22-5)
    return hour >= start or hour <= end


def detect_period(text: str) -> str | None:
    """Detect a time period keyword in text."""
    lowered = text.strip().lower()
    for period in TIME_PERIOD_BUCKETS:
        if period in lowered:
            return period
    return None


def time_period_bucket(value: str) -> str | None:
    """Canonical coarse time bucket for a free-text time-of-day string.

    Maps a parsed clock hour to its ``TIME_PERIOD_BUCKETS`` name; if no hour
    parses, falls back to a period keyword found in the text. Used to decide
    whether two article time-of-day extractions agree at a coarse granularity
    (e.g. "around 1 p.m." and "this afternoon" both → "afternoon").

    Args:
        value: Free-text time description.

    Returns:
        A ``TIME_PERIOD_BUCKETS`` key, or None if neither an hour nor a period
        keyword is present.

    Examples:
        >>> time_period_bucket("around 1 p.m.")
        'afternoon'
        >>> time_period_bucket("Tuesday")  # no time signal
    """
    hour = parse_hour(value)
    if hour is not None:
        for name in TIME_PERIOD_BUCKETS:
            if hour_in_bucket(hour, name):
                return name
    return detect_period(value)
