"""Wilson score confidence intervals for the holdout report's proportions.

A pure, offline helper that turns the raw success/total counts already stored in
a :class:`~src.eval.holdout.HoldoutReport` into confidence intervals, with no new
inference. It mirrors the "pure function over saved reports" shape of
:mod:`src.eval.gate`: every headline metric (completion rate, aggregate exact and
fuzzy accuracy, per-field accuracy) is a binomial proportion whose sampling
uncertainty the point estimate alone does not convey.

The Wilson score interval is used rather than the textbook Wald interval because
it stays inside ``[0, 1]`` and remains informative both at the extremes (e.g.
92/100) and for the small per-field cells (e.g. 10/11) where Wald breaks down.
The critical value is derived from :class:`statistics.NormalDist`, so the module
needs only the standard library — no SciPy dependency.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from src.eval.holdout import HoldoutReport


def wilson_ci(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    The Wilson interval inverts the score test; unlike the Wald interval it
    cannot fall outside ``[0, 1]`` and stays well calibrated when the proportion
    is near 0 or 1 or the sample is small.

    Args:
        successes: Number of successes (e.g. exact matches);
            ``0 <= successes <= total``.
        total: Number of trials (e.g. extracted values). When 0 the proportion
            is undefined and the full ``(0.0, 1.0)`` interval is returned.
        confidence: Two-sided confidence level, e.g. ``0.95`` for a 95% interval.

    Returns:
        The ``(lower, upper)`` bounds as fractions in ``[0, 1]``.

    Raises:
        ValueError: If ``confidence`` is not in ``(0, 1)``, or ``successes`` is
            negative or exceeds ``total``.

    Examples:
        >>> lo, hi = wilson_ci(70, 100)
        >>> round(lo, 3), round(hi, 3)
        (0.604, 0.781)
        >>> wilson_ci(0, 0)
        (0.0, 1.0)
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(
            f"require 0 <= successes <= total, got {successes}/{total}"
        )
    if total == 0:
        return (0.0, 1.0)

    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def cis_from_report(
    report: HoldoutReport, confidence: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Compute Wilson intervals for a holdout report's headline proportions.

    Reads only counts already saved in the report, so it can be applied to any
    archived report with zero new inference (the same property
    :func:`src.eval.gate.gate` relies on).

    Args:
        report: A holdout report carrying the raw success/total counts.
        confidence: Two-sided confidence level applied to every interval.

    Returns:
        A mapping from metric key to its ``(lower, upper)`` interval:

        * ``"completion_rate"`` — over ``n_incidents``;
        * ``"aggregate_exact"`` / ``"aggregate_fuzzy"`` — over the summed
          extracted-value count across all fields;
        * ``"<field>_exact"`` — one entry per field, over that field's
          extracted count.
    """
    n_extracted = sum(fm.n_extracted for fm in report.field_metrics)
    n_exact = sum(fm.n_exact_match for fm in report.field_metrics)
    n_fuzzy = sum(fm.n_fuzzy_match for fm in report.field_metrics)

    cis: dict[str, tuple[float, float]] = {
        "completion_rate": wilson_ci(
            report.n_completed, report.n_incidents, confidence
        ),
        "aggregate_exact": wilson_ci(n_exact, n_extracted, confidence),
        "aggregate_fuzzy": wilson_ci(n_fuzzy, n_extracted, confidence),
    }
    for fm in report.field_metrics:
        cis[f"{fm.field_name}_exact"] = wilson_ci(
            fm.n_exact_match, fm.n_extracted, confidence
        )
    return cis
