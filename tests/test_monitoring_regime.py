"""Known-answer tests for cie.monitoring.regime horizon-overlap logic."""

from datetime import date

import pandas as pd
import pytest

from cie.monitoring.regime import assess_regime


def test_validated_cutoffs_exact_100_percent_overlap():
    """Exact Sep 10 cutoffs must yield 90/90 overlap days and in_regime=True."""
    for dt in ["2010-09-10", "2011-09-10", date(2012, 9, 10), pd.Timestamp("2011-09-10")]:
        res = assess_regime(as_of=dt, lookback_days=270)
        assert res.overlap_days == 90
        assert res.overlap_fraction == pytest.approx(1.0)
        assert res.lookback_in_regime is True
        assert res.seasonal_in_regime is True
        assert res.in_regime is True


def test_out_of_regime_q4_and_winter_cutoffs():
    """Verify exact overlap days and out-of-regime verdict for 2011-11-01 and 2011-12-10."""
    # [2011-11-01, 2012-01-30) intersects [2011-09-10, 2011-12-09) on 38 days
    nov = assess_regime("2011-11-01")
    assert nov.overlap_days == 38
    assert nov.overlap_fraction == pytest.approx(38.0 / 90.0)
    assert nov.seasonal_in_regime is False
    assert nov.in_regime is False

    # [2011-12-10, 2012-03-09) has 0 overlap days with [2011-09-10, 2011-12-09)
    dec = assess_regime("2011-12-10")
    assert dec.overlap_days == 0
    assert dec.overlap_fraction == 0.0
    assert dec.seasonal_in_regime is False
    assert dec.in_regime is False

    # Spring/Summer cutoff also 0 overlap
    jun = assess_regime("2011-06-01")
    assert jun.overlap_days == 0
    assert jun.in_regime is False


def test_horizon_overlap_80_percent_boundary():
    """Boundary check: 72/90 days (0.80, +/- 18d from Sep 10) is in-regime; 71/90 is not."""
    # 18 days after Sep 10 -> Sep 28 -> 72 days overlap (0.80)
    assert assess_regime("2011-09-28").overlap_days == 72
    assert assess_regime("2011-09-28").in_regime is True

    # 19 days after Sep 10 -> Sep 29 -> 71 days overlap (0.7889)
    assert assess_regime("2011-09-29").overlap_days == 71
    assert assess_regime("2011-09-29").in_regime is False

    # 18 days before Sep 10 -> Aug 23 -> 72 days overlap (0.80)
    assert assess_regime("2011-08-23").overlap_days == 72
    assert assess_regime("2011-08-23").in_regime is True

    # 19 days before Sep 10 -> Aug 22 -> 71 days overlap
    assert assess_regime("2011-08-22").overlap_days == 71
    assert assess_regime("2011-08-22").in_regime is False


def test_wrong_lookback_and_invalid_inputs():
    """Incorrect lookback or invalid arguments behave explicitly."""
    bad_lb = assess_regime("2011-09-10", lookback_days=180)
    assert bad_lb.seasonal_in_regime is True
    assert bad_lb.lookback_in_regime is False
    assert bad_lb.in_regime is False

    with pytest.raises(ValueError, match="horizon_days"):
        assess_regime("2011-09-10", horizon_days=0)

    with pytest.raises(ValueError, match="min_overlap_fraction"):
        assess_regime("2011-09-10", min_overlap_fraction=1.5)

    with pytest.raises(TypeError, match="Unsupported as_of"):
        assess_regime(12345)  # type: ignore[arg-type]
