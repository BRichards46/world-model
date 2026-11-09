from __future__ import annotations

import pandas as pd

import numpy as np

from world_model.features.economic import (
    build_realtime_indicator_from_vintage,
    historical_percentile_rank,
    rolling_z_score,
    score_indicator,
    weighted_average_signed,
)


def test_realtime_indicator_applies_revisions_only_after_realtime_start() -> None:
    vintage = pd.DataFrame(
        {
            "series_id": ["TEST", "TEST", "TEST"],
            "observation_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"]),
            "value": [100.0, 105.0, 110.0],
            "realtime_start": pd.to_datetime(["2020-02-01", "2020-03-01", "2020-03-15"]),
            "realtime_end": ["2020-02-29", "9999-12-31", "9999-12-31"],
        }
    )

    result = build_realtime_indicator_from_vintage(vintage, transform="level")

    assert result.value.loc["2020-02-01"] == 100.0
    assert result.value.loc["2020-03-01"] == 105.0
    assert result.value.loc["2020-03-15"] == 110.0


def test_realtime_indicator_yoy_uses_known_prior_revision() -> None:
    vintage = pd.DataFrame(
        {
            "series_id": ["TEST", "TEST", "TEST"],
            "observation_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-01-01"]),
            "value": [100.0, 105.0, 115.5],
            "realtime_start": pd.to_datetime(["2020-02-01", "2020-03-01", "2021-02-01"]),
            "realtime_end": ["2020-02-29", "9999-12-31", "9999-12-31"],
        }
    )

    result = build_realtime_indicator_from_vintage(vintage, transform="yoy_pct")

    assert round(result.value.loc["2021-02-01"], 4) == 0.10


def test_score_indicator_squashes_to_unit_range_and_can_invert() -> None:
    values = pd.Series(range(20), index=pd.date_range("2020-01-01", periods=20))

    score = score_indicator(values, window=10, min_periods=5, scale=2.0)
    inverted = score_indicator(values, window=10, min_periods=5, scale=2.0, invert=True)

    assert score.dropna().between(-1.0, 1.0).all()
    assert round(score.dropna().iloc[-1], 6) == round(-inverted.dropna().iloc[-1], 6)


def test_weighted_average_signed_uses_absolute_weights_as_denominator() -> None:
    frame = pd.DataFrame({"growth_score": [0.5], "inflation_score": [0.5]})

    result = weighted_average_signed(frame, {"growth_score": 1.0, "inflation_score": -1.0})

    assert result.iloc[0] == 0.0


def test_calendar_z_score_uses_trailing_years_and_excludes_current_value() -> None:
    index = pd.date_range("2008-01-01", "2020-12-01", freq="MS")
    values = pd.Series(np.sin(np.arange(len(index)) / 6.0), index=index)
    values.loc["2009-01-01"] = 100.0
    values.iloc[-1] = 2.0

    result = rolling_z_score(
        values,
        window=None,
        window_years=10,
        min_periods=60,
        min_history_years=5,
    )

    timestamp = values.index[-1]
    history = values.loc[
        (values.index >= timestamp - pd.DateOffset(years=10)) & (values.index < timestamp)
    ]
    expected = (values.iloc[-1] - history.mean()) / history.std(ddof=0)

    assert np.isclose(result.iloc[-1], expected)
    assert "2009-01-01" not in history.index


def test_calendar_percentile_uses_prior_native_frequency_history() -> None:
    index = pd.date_range("2010-01-01", "2021-12-01", freq="MS")
    values = pd.Series(np.arange(len(index), dtype=float), index=index)

    percentile = historical_percentile_rank(
        values,
        window_years=10,
        min_periods=60,
        min_history_years=5,
    )

    assert percentile.iloc[-1] == 100.0
    assert pd.isna(percentile.loc["2014-12-01"])


def test_expanding_percentile_keeps_long_history() -> None:
    index = pd.date_range("2000-01-01", periods=30, freq="YS")
    values = pd.Series([100.0, *range(1, 29), 50.0], index=index)

    trailing = historical_percentile_rank(
        values,
        window_years=10,
        min_periods=5,
        min_history_years=5,
    )
    expanding = historical_percentile_rank(
        values,
        window_years=None,
        min_periods=5,
        min_history_years=5,
    )

    assert trailing.iloc[-1] == 100.0
    assert expanding.iloc[-1] < 100.0
