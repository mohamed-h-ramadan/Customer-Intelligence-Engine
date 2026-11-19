"""Known-answer tests for cie.monitoring.regime integer-day horizon-overlap logic."""

from datetime import date

import pandas as pd
import pytest

from cie.monitoring.regime import assess_regime


def test_validated_cutoffs_exact_100_percent_overlap():
    """Exact Sep 10 cutoffs must yield 90/90 overlap days and in_regime=True."""
    for dt in ["2010-09-10", "2011-09-10", date(2012, 9, 10), pd.Timestamp("2011-09-10")]:
        res = assess_regime(as_of=dt, lookback_days=270, horizon_days=90)
        assert res.overlap_days == 90
        assert res.overlap_fraction == pytest.approx(1.0)
        assert res.lookback_in_regime is True
        assert res.horizon_in_regime is True
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


def test_integer_day_boundary_76_vs_75_days():
    """Boundary check: offset <= 14d (76/90 overlap) is True; offset >= 15d (75/90) is False."""
    # +14 days after Sep 10 -> Sep 24 -> 76 days overlap -> in_regime=True
    plus_14 = assess_regime("2011-09-24")
    assert plus_14.overlap_days == 76
    assert plus_14.in_regime is True

    # +15 days after Sep 10 -> Sep 25 -> 75 days overlap -> in_regime=False
    plus_15 = assess_regime("2011-09-25")
    assert plus_15.overlap_days == 75
    assert plus_15.in_regime is False

    # -14 days before Sep 10 -> Aug 27 -> 76 days overlap -> in_regime=True
    minus_14 = assess_regime("2011-08-27")
    assert minus_14.overlap_days == 76
    assert minus_14.in_regime is True

    # -15 days before Sep 10 -> Aug 26 -> 75 days overlap -> in_regime=False
    minus_15 = assess_regime("2011-08-26")
    assert minus_15.overlap_days == 75
    assert minus_15.in_regime is False


def test_wrong_lookback_horizon_and_invalid_inputs():
    """Incorrect lookback/horizon or invalid arguments behave explicitly."""
    bad_lb = assess_regime("2011-09-10", lookback_days=180)
    assert bad_lb.seasonal_in_regime is True
    assert bad_lb.lookback_in_regime is False
    assert bad_lb.in_regime is False

    bad_hz = assess_regime("2011-09-10", horizon_days=60, min_overlap_days=50)
    assert bad_hz.horizon_in_regime is False
    assert bad_hz.in_regime is False

    with pytest.raises(ValueError, match="horizon_days"):
        assess_regime("2011-09-10", horizon_days=0)

    with pytest.raises(ValueError, match="min_overlap_days"):
        assess_regime("2011-09-10", horizon_days=90, min_overlap_days=91)

    with pytest.raises(TypeError, match="Unsupported as_of"):
        assess_regime(12345)  # type: ignore[arg-type]
