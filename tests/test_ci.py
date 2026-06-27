"""Unit tests for src/eval/ci.py — Wilson confidence intervals.

Covers wilson_ci against externally verified values, its clamping and validation
behaviour at the boundaries, and cis_from_report's aggregation over a small
synthetic holdout report.
"""

import pytest

from src.agents.state import DatasetType
from src.eval.ci import cis_from_report, wilson_ci
from src.eval.holdout import FieldMetrics, HoldoutReport


def _field(
    name: str, n_extracted: int, n_exact: int, n_fuzzy: int | None = None
) -> FieldMetrics:
    """Build a FieldMetrics carrying only the counts the CI helper reads."""
    n_fuzzy = n_exact if n_fuzzy is None else n_fuzzy
    return FieldMetrics(
        field_name=name,
        n_evaluable=n_extracted,
        n_extracted=n_extracted,
        n_exact_match=n_exact,
        n_fuzzy_match=n_fuzzy,
        coverage=1.0,
        exact_accuracy=n_exact / n_extracted if n_extracted else 0.0,
        fuzzy_accuracy=n_fuzzy / n_extracted if n_extracted else 0.0,
    )


def _report(n_incidents: int, n_completed: int, fields: list[FieldMetrics]) -> HoldoutReport:
    """Build a minimal HoldoutReport for CI aggregation tests."""
    return HoldoutReport(
        dataset_type=DatasetType.CIVILIANS_SHOT,
        n_incidents=n_incidents,
        n_completed=n_completed,
        n_escalated=n_incidents - n_completed,
        completion_rate=n_completed / n_incidents if n_incidents else 0.0,
        field_metrics=fields,
    )


class TestWilsonCi:
    """wilson_ci point behaviour, boundaries, and validation."""

    @pytest.mark.parametrize(
        ("successes", "total", "lower", "upper"),
        [
            (70, 100, 0.604, 0.781),
            (92, 100, 0.850, 0.959),
            (210, 272, 0.719, 0.818),
            (147, 207, 0.645, 0.768),
            (10, 11, 0.623, 0.984),
            (6, 9, 0.354, 0.879),
        ],
    )
    def test_known_values(self, successes, total, lower, upper):
        """Match values verified against the manuscript's reported counts."""
        lo, hi = wilson_ci(successes, total)
        assert lo == pytest.approx(lower, abs=1e-3)
        assert hi == pytest.approx(upper, abs=1e-3)

    def test_total_zero_returns_full_interval(self):
        """An undefined proportion yields the whole [0, 1] interval."""
        assert wilson_ci(0, 0) == (0.0, 1.0)

    def test_p_zero_clamps_lower_bound(self):
        """All-failure stays pinned at 0 below and informative above."""
        lo, hi = wilson_ci(0, 5)
        assert lo == 0.0
        assert 0.0 < hi < 1.0

    def test_p_one_clamps_upper_bound(self):
        """All-success stays pinned at 1 above and informative below."""
        lo, hi = wilson_ci(5, 5)
        assert hi == 1.0
        assert 0.0 < lo < 1.0

    @pytest.mark.parametrize(
        ("successes", "total"), [(0, 1), (1, 1), (3, 9), (50, 50), (0, 0)]
    )
    def test_bounds_within_unit_interval(self, successes, total):
        """Lower and upper bounds are ordered and inside [0, 1]."""
        lo, hi = wilson_ci(successes, total)
        assert 0.0 <= lo <= hi <= 1.0

    def test_higher_confidence_widens_interval(self):
        """A 99% interval contains the 95% interval."""
        lo95, hi95 = wilson_ci(70, 100, confidence=0.95)
        lo99, hi99 = wilson_ci(70, 100, confidence=0.99)
        assert lo99 < lo95
        assert hi99 > hi95

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_confidence_raises(self, confidence):
        """Confidence must lie strictly inside (0, 1)."""
        with pytest.raises(ValueError):
            wilson_ci(7, 10, confidence=confidence)

    @pytest.mark.parametrize(("successes", "total"), [(6, 5), (-1, 5), (5, -1)])
    def test_invalid_counts_raise(self, successes, total):
        """Successes must satisfy 0 <= successes <= total."""
        with pytest.raises(ValueError):
            wilson_ci(successes, total)


class TestCisFromReport:
    """cis_from_report aggregation over a holdout report's counts."""

    def test_keys_and_completion_interval(self):
        """Returns completion, aggregate, and per-field intervals consistently."""
        fields = [
            _field("outcome", n_extracted=74, n_exact=68, n_fuzzy=68),
            _field("civilian_race", n_extracted=11, n_exact=10, n_fuzzy=10),
        ]
        report = _report(n_incidents=100, n_completed=70, fields=fields)

        cis = cis_from_report(report)

        assert cis["completion_rate"] == wilson_ci(70, 100)
        # Aggregate is over the summed extracted values across fields (85 = 74 + 11).
        assert cis["aggregate_exact"] == wilson_ci(78, 85)
        assert cis["aggregate_fuzzy"] == wilson_ci(78, 85)
        assert cis["civilian_race_exact"] == wilson_ci(10, 11)
        assert cis["outcome_exact"] == wilson_ci(68, 74)

    def test_empty_fields_give_defined_completion_only(self):
        """No fields -> aggregate over zero trials is the full interval."""
        report = _report(n_incidents=10, n_completed=4, fields=[])
        cis = cis_from_report(report)
        assert cis["completion_rate"] == wilson_ci(4, 10)
        assert cis["aggregate_exact"] == (0.0, 1.0)
