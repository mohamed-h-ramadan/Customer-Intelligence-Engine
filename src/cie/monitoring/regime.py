"""Pure horizon-overlap regime assessment for CLV monitoring.

The Champion (Tweedie GLM) and Calibrated Run-Rate multiplier (1.42746205196053)
were validated strictly on the 90-day horizon [Sep 10, Dec 9) with a 270-day lookback
window (cutoffs 2010-09-10 and 2011-09-10).

This module defines the proposed horizon-overlap regime rule without altering
existing runtime validation (src/cie/core/validation.py) until approved.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

VALIDATED_CUTOFF_MONTH = 9
VALIDATED_CUTOFF_DAY = 10
VALIDATED_LOOKBACK_DAYS = 270
VALIDATED_HORIZON_DAYS = 90
DEFAULT_MIN_OVERLAP_FRACTION = 0.80


@dataclass(frozen=True)
class RegimeAssessment:
    """Structured outcome of horizon-overlap and lookback regime check."""

    as_of: str
    lookback_days: int
    horizon_days: int
    validated_horizon_start: str
    validated_horizon_end: str
    overlap_days: int
    overlap_fraction: float
    min_overlap_fraction: float
    lookback_in_regime: bool
    seasonal_in_regime: bool
    in_regime: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_date(value: str | date | datetime) -> date:
    """Convert ISO date string, datetime, or pd.Timestamp-like object to datetime.date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"Unsupported as_of type: {type(value).__name__}")


def assess_regime(
    as_of: str | date | datetime,
    lookback_days: int = VALIDATED_LOOKBACK_DAYS,
    horizon_days: int = VALIDATED_HORIZON_DAYS,
    min_overlap_fraction: float = DEFAULT_MIN_OVERLAP_FRACTION,
) -> RegimeAssessment:
    """Evaluate whether [as_of, as_of + horizon_days) is within the validated [Sep 10, Dec 9) regime.

    Args:
        as_of: Scoring cutoff date.
        lookback_days: Lookback window used to compute customer features (must equal 270 to pass).
        horizon_days: Forward forecasting horizon in days (must be > 0).
        min_overlap_fraction: Minimum fraction of [as_of, as_of + horizon_days) overlapping
            with the annual validated horizon [Sep 10, Dec 9) (default: 0.80 -> >= 72/90 days,
            equivalent to |as_of - Sep 10| <= 18 days for a 90-day horizon).

    Returns:
        RegimeAssessment with overlap metrics and boolean verdict.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if not (0.0 <= min_overlap_fraction <= 1.0):
        raise ValueError(f"min_overlap_fraction must be in [0.0, 1.0], got {min_overlap_fraction}")

    cutoff = _coerce_date(as_of)
    scoring_end = cutoff + timedelta(days=horizon_days)

    best_overlap = -1
    best_dist = float("inf")
    best_val_start = date(cutoff.year, VALIDATED_CUTOFF_MONTH, VALIDATED_CUTOFF_DAY)
    best_val_end = best_val_start + timedelta(days=VALIDATED_HORIZON_DAYS)

    # Check adjacent years so Dec/Jan or Aug cutoffs resolve cleanly; break zero-overlap ties by nearest cutoff
    for year in (cutoff.year - 1, cutoff.year, cutoff.year + 1):
        val_start = date(year, VALIDATED_CUTOFF_MONTH, VALIDATED_CUTOFF_DAY)
        val_end = val_start + timedelta(days=VALIDATED_HORIZON_DAYS)

        inter_start = max(cutoff, val_start)
        inter_end = min(scoring_end, val_end)
        overlap = max(0, (inter_end - inter_start).days)
        dist = abs((cutoff - val_start).days)

        if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
            best_overlap = overlap
            best_dist = dist
            best_val_start = val_start
            best_val_end = val_end

    overlap_fraction = best_overlap / float(horizon_days)
    lookback_ok = lookback_days == VALIDATED_LOOKBACK_DAYS
    horizon_ok = horizon_days == VALIDATED_HORIZON_DAYS
    seasonal_ok = overlap_fraction >= min_overlap_fraction
    in_regime = bool(lookback_ok and horizon_ok and seasonal_ok)

    reasons = []
    if not seasonal_ok:
        reasons.append(
            f"Horizon [{cutoff.isoformat()}, {scoring_end.isoformat()}) overlaps "
            f"{best_overlap}/{horizon_days} days ({overlap_fraction:.1%}) with validated window "
            f"[{best_val_start.isoformat()}, {best_val_end.isoformat()}), below threshold {min_overlap_fraction:.0%}."
        )
    if not lookback_ok:
        reasons.append(f"lookback_days={lookback_days} != validated {VALIDATED_LOOKBACK_DAYS}d.")
    if not horizon_ok:
        reasons.append(f"horizon_days={horizon_days} != validated {VALIDATED_HORIZON_DAYS}d.")

    reason_str = (
        " ".join(reasons)
        if reasons
        else (
            f"In-regime: {best_overlap}/{horizon_days} days ({overlap_fraction:.1%}) overlap "
            f"with [{best_val_start.isoformat()}, {best_val_end.isoformat()}), lookback={lookback_days}d."
        )
    )

    return RegimeAssessment(
        as_of=cutoff.isoformat(),
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        validated_horizon_start=best_val_start.isoformat(),
        validated_horizon_end=best_val_end.isoformat(),
        overlap_days=best_overlap,
        overlap_fraction=overlap_fraction,
        min_overlap_fraction=min_overlap_fraction,
        lookback_in_regime=lookback_ok,
        seasonal_in_regime=seasonal_ok,
        in_regime=in_regime,
        reason=reason_str,
    )
