"""Tests for src/merge/weapon_similarity.py.

Tests normalize_weapon() category mapping and weapons_match() equality logic.
No external dependencies (sentence-transformers, numpy) needed.
"""


from src.merge.weapon_similarity import (
    WEAPON_CATEGORIES,
    WEAPON_CATEGORY_MAP,
    normalize_weapon,
    weapons_match,
)

# ---------------------------------------------------------------------------
# normalize_weapon
# ---------------------------------------------------------------------------


class TestNormalizeWeapon:
    """Test normalize_weapon category mapping."""

    def test_mapped_value_handgun(self) -> None:
        """Test mapped value returns correct category."""
        assert normalize_weapon("GUN") == "HANDGUN"

    def test_mapped_value_glock(self) -> None:
        """Test specific firearm model maps to HANDGUN."""
        assert normalize_weapon("GLOCK 40") == "HANDGUN"

    def test_mapped_value_rifle(self) -> None:
        """Test RIFLE maps to RIFLE."""
        assert normalize_weapon("RIFLE") == "RIFLE"

    def test_mapped_value_machete(self) -> None:
        """Test MACHETE maps to KNIFE."""
        assert normalize_weapon("MACHETE") == "KNIFE"

    def test_mapped_value_vehicle(self) -> None:
        """Test VEHICLE maps to VEHICLE."""
        assert normalize_weapon("VEHICLE") == "VEHICLE"

    def test_case_insensitive(self) -> None:
        """Test lookup is case-insensitive."""
        assert normalize_weapon("handgun") == "HANDGUN"
        assert normalize_weapon("Knife") == "KNIFE"

    def test_whitespace_stripped(self) -> None:
        """Test leading/trailing whitespace is stripped."""
        assert normalize_weapon("  RIFLE  ") == "RIFLE"

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert normalize_weapon(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Test empty string returns None."""
        assert normalize_weapon("") is None

    def test_whitespace_only_returns_none(self) -> None:
        """Test whitespace-only string returns None."""
        assert normalize_weapon("   ") is None

    def test_details_missing_returns_none(self) -> None:
        """Test '(DETAILS MISSING)' returns None."""
        assert normalize_weapon("(DETAILS MISSING)") is None

    def test_details_missing_no_parens_returns_none(self) -> None:
        """Test 'DETAILS MISSING' without parens returns None."""
        assert normalize_weapon("DETAILS MISSING") is None

    def test_unmapped_value_returns_other(self) -> None:
        """Test unmapped non-null value falls back to OTHER."""
        assert normalize_weapon("taser") == "OTHER"

    def test_unmapped_value_baseball_bat(self) -> None:
        """Test another unmapped value returns OTHER."""
        assert normalize_weapon("baseball bat") == "OTHER"


class TestWeaponCategoryMapCompleteness:
    """Verify all mapped values produce valid categories."""

    def test_all_map_values_are_valid_categories(self) -> None:
        """Every value in WEAPON_CATEGORY_MAP is in WEAPON_CATEGORIES."""
        for raw, category in WEAPON_CATEGORY_MAP.items():
            assert category in WEAPON_CATEGORIES, (
                f"'{raw}' maps to '{category}' which is not a valid category"
            )


# ---------------------------------------------------------------------------
# weapons_match
# ---------------------------------------------------------------------------


class TestWeaponsMatch:
    """Test weapons_match equality logic."""

    def test_same_category_match(self) -> None:
        """Two values mapping to same category return True."""
        assert weapons_match("GUN", "HANDGUN") is True

    def test_same_category_case_insensitive(self) -> None:
        """Case-insensitive match works."""
        assert weapons_match("handgun", "GLOCK 40") is True

    def test_cross_category_no_match(self) -> None:
        """Values in different categories return False."""
        assert weapons_match("KNIFE", "RIFLE") is False

    def test_unmapped_values_both_other(self) -> None:
        """Two unmapped values both normalize to OTHER and match."""
        assert weapons_match("taser", "pepper spray") is True

    def test_none_input_returns_false(self) -> None:
        """None on either side returns False."""
        assert weapons_match("", "HANDGUN") is False

    def test_vehicle_match(self) -> None:
        """Vehicle synonyms match."""
        assert weapons_match("CAR", "VEHICLE") is True

    def test_edged_weapons_match(self) -> None:
        """Edged weapon synonyms match."""
        assert weapons_match("MACHETE", "KNIFE") is True
