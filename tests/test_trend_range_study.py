from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.trend_range_study import (
    aligned_levels,
    daily_volatility,
    quad_split_returns,
    range_position,
    range_timing_returns,
    risk_portfolio,
    rotation_p,
    trend_score,
    weekly_dates,
)


def _calendar(n: int = 400) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=n)


def test_trend_score_needs_every_lookback_and_has_the_trend_sign() -> None:
    calendar = _calendar()
    rng = np.random.default_rng(1)
    up = pd.Series(np.cumsum(0.2 + rng.normal(0, 1, len(calendar))), index=calendar)
    levels = pd.DataFrame({"up": up, "down": -up})
    vol = daily_volatility(levels, 63)
    score = trend_score(levels, vol, (63, 252))
    assert score.iloc[:252].isna().all().all()
    assert score["up"].iloc[-1] > 0 > score["down"].iloc[-1]


def test_range_position_is_negative_after_a_dip() -> None:
    calendar = _calendar(100)
    rng = np.random.default_rng(2)
    level = pd.Series(np.cumsum(rng.normal(0, 1, len(calendar))), index=calendar)
    level.iloc[-1] = level.iloc[-2] - 5.0
    levels = level.to_frame("a")
    position = range_position(levels, daily_volatility(levels, 63), 15)
    assert position["a"].iloc[-1] < 0


def test_risk_portfolio_is_unit_gross_and_empty_is_nan() -> None:
    index = pd.DatetimeIndex(["2020-01-03", "2020-01-10"])
    positions = pd.DataFrame({"a": [1.0, 0.0], "b": [-1.0, 0.0]}, index=index)
    volatility = pd.DataFrame({"a": [1.0, 1.0], "b": [3.0, 1.0]}, index=index)
    forward = pd.DataFrame({"a": [4.0, 1.0], "b": [-4.0, 1.0]}, index=index)
    returns = risk_portfolio(positions, volatility, forward)
    # Weights 1 and -1/3 → normalized 0.75 and -0.25; both legs earn 4.
    assert np.isclose(returns.iloc[0], 4.0)
    assert np.isnan(returns.iloc[1])


def test_range_timing_splits_dips_in_uptrends_from_chasing() -> None:
    index = pd.DatetimeIndex(["2020-01-03"])
    trend = pd.DataFrame({"dip": [1.0], "chase": [1.0], "rally_in_downtrend": [-1.0]}, index=index)
    band = pd.DataFrame({"dip": [-1.0], "chase": [1.0], "rally_in_downtrend": [1.0]}, index=index)
    volatility = pd.DataFrame(1.0, index=index, columns=trend.columns)
    forward = pd.DataFrame({"dip": [2.0], "chase": [-2.0], "rally_in_downtrend": [-6.0]}, index=index)
    frame = range_timing_returns(trend, band, volatility, forward, threshold=0.5)
    assert np.isclose(frame["trend_favorable"].iloc[0], (2.0 + 6.0) / 2)
    assert np.isclose(frame["trend_unfavorable"].iloc[0], -2.0)


def test_quad_split_uses_the_playbook_sign() -> None:
    index = pd.DatetimeIndex(["2020-01-31"])
    trend = pd.DataFrame({"bonds": [1.0], "stocks": [1.0]}, index=index)
    volatility = pd.DataFrame(1.0, index=index, columns=trend.columns)
    forward = pd.DataFrame({"bonds": [3.0], "stocks": [-1.0]}, index=index)
    scores = {quad: pd.Series({"bonds": 0.0, "stocks": 0.0}) for quad in (1, 2, 3)}
    scores[4] = pd.Series({"bonds": 1.0, "stocks": -1.0})
    split = quad_split_returns(trend, volatility, forward, scores)
    assert np.isclose(split[4]["agree"].iloc[0], 3.0)
    assert np.isclose(split[4]["disagree"].iloc[0], -1.0)
    assert np.isnan(split[1]["agree"].iloc[0])


def test_rotation_detects_a_planted_quad_effect() -> None:
    rng = np.random.default_rng(4)
    dates = pd.date_range("2000-01-31", periods=200, freq="ME")
    labels = pd.Series(rng.integers(1, 5, len(dates)), index=dates)
    differences = pd.DataFrame(rng.normal(0, 1, (len(dates), 4)), index=dates, columns=[1, 2, 3, 4])
    for date, label in labels.items():
        differences.at[date, label] += 1.5
    observed, p_value = rotation_p(differences, labels, minimum_gap=12)
    assert observed > 1.0 and p_value < 0.05


def test_alignment_carries_short_gaps_only() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=10)
    series = pd.Series([1.0, 2.0], index=[calendar[0], calendar[5]])
    levels = aligned_levels({"a": series}, calendar, max_fill_days=2)
    assert levels["a"].iloc[2] == 1.0 and np.isnan(levels["a"].iloc[4])
    # Wed 1 Jan to Tue 14 Jan spans three calendar weeks; the last one is partial.
    assert list(weekly_dates(calendar).strftime("%m-%d")) == ["01-03", "01-10", "01-14"]
