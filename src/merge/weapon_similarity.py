"""Weapon category normalization for merge-node matching.

Maps raw weapon descriptions (from DB and LLM extractions) to 7 canonical
categories: HANDGUN, RIFLE, SHOTGUN, KNIFE, VEHICLE, OTHER, UNKNOWN.
Replaces the previous embedding-based similarity approach which suffered
from low accuracy due to verbose LLM descriptions vs. short DB labels.
"""

WEAPON_CATEGORIES: list[str] = [
    "HANDGUN",
    "RIFLE",
    "SHOTGUN",
    "KNIFE",
    "VEHICLE",
    "OTHER",
    "UNKNOWN",
]

WEAPON_CATEGORY_MAP: dict[str, str] = {
    # Handguns
    "GUN": "HANDGUN",
    "HANDGUN": "HANDGUN",
    "FIREARM": "HANDGUN",
    "FIREARM - HANDGUN": "HANDGUN",
    "GLOCK 40": "HANDGUN",
    "PISTOL": "HANDGUN",
    "REVOLVER": "HANDGUN",
    "SEMI-AUTOMATIC HANDGUN": "HANDGUN",
    "SEMI-AUTOMATIC PISTOL": "HANDGUN",
    # Rifles
    "RIFLE": "RIFLE",
    "FIREARM - RIFLE": "RIFLE",
    "AR-15": "RIFLE",
    "ASSAULT RIFLE": "RIFLE",
    "SEMI-AUTOMATIC RIFLE": "RIFLE",
    # Shotguns
    "SHOTGUN": "SHOTGUN",
    "FIREARM - SHOTGUN": "SHOTGUN",
    # Knives / edged weapons
    "KNIFE": "KNIFE",
    "MACHETE": "KNIFE",
    "SWORD": "KNIFE",
    "AXE": "KNIFE",
    "BOX CUTTER": "KNIFE",
    # Vehicles
    "VEHICLE": "VEHICLE",
    "CAR": "VEHICLE",
    "TRUCK": "VEHICLE",
    # Other / unknown
    "OTHER": "OTHER",
    "UNKNOWN": "UNKNOWN",
    "UNDETERMINED": "UNKNOWN",
}

_DETAILS_MISSING = {"(DETAILS MISSING)", "DETAILS MISSING"}


def normalize_weapon(value: str | None) -> str | None:
    """Normalize a weapon string to a canonical category.

    Looks up the uppercased, stripped value in ``WEAPON_CATEGORY_MAP``.
    Unmapped non-null values fall back to ``OTHER``. Null, empty, or
    ``(DETAILS MISSING)`` values return ``None``.

    Args:
        value: Raw weapon description from DB or LLM extraction.

    Returns:
        Canonical category string, or ``None`` for null/empty/missing values.

    Examples:
        >>> normalize_weapon("GLOCK 40")
        'HANDGUN'
        >>> normalize_weapon("machete")
        'KNIFE'
        >>> normalize_weapon(None) is None
        True
        >>> normalize_weapon("taser")
        'OTHER'
    """
    if value is None:
        return None
    cleaned = value.strip().upper()
    if not cleaned or cleaned in _DETAILS_MISSING:
        return None
    return WEAPON_CATEGORY_MAP.get(cleaned, "OTHER")


def weapons_match(text_a: str, text_b: str) -> bool:
    """Check if two weapon descriptions map to the same canonical category.

    Args:
        text_a: First weapon description.
        text_b: Second weapon description.

    Returns:
        True if both normalize to the same non-None category.

    Examples:
        >>> weapons_match("GLOCK 40", "handgun")
        True
        >>> weapons_match("knife", "rifle")
        False
    """
    cat_a = normalize_weapon(text_a)
    cat_b = normalize_weapon(text_b)
    if cat_a is None or cat_b is None:
        return False
    return cat_a == cat_b
