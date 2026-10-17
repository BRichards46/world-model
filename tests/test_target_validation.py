from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.target_validation import (
    conditional_target_estimate,
    future_realized_vol,
    relative_forward_return,
    summarize_target_forecasts,
)


def test_relative_forward_return_uses_long_minus_short_baskets() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "ES": [100, 101, 102, 103, 104],
            "ZN": [100, 100, 100, 100, 100],
        },
        index=index,
    )

    result = relative_forward_return(prices, ["ES"], ["ZN"], days=1)

    assert result.iloc[0] > 0
    assert np.isclose(result.iloc[0], 0.01)


def test_future_realized_vol_returns_forward_measure() -> None:
    index = pd.bdate_range("2020-01-01", periods=20)
    prices = pd.Series(np.linspace(100, 110, len(index)), index=index)

    vol = future_realized_vol(prices, days=5)

    assert vol.dropna().ge(0).all()


def test_conditional_target_estimate_binary_probability() -> None:
    scores = pd.Series([-2, -1, 0, 1, 2] * 40, dtype=float)
    target = pd.Series([0, 0, 0, 1, 1] * 40, dtype=float)

    estimate = conditional_target_estimate(scores, target, current_score=2.0, kind="binary")

    assert estimate == 1.0


def test_summarize_target_forecasts_binary_brier() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_date": pd.bdate_range("2020-01-01", periods=5),
            "target": "drawdown",
            "target_label": "Drawdown",
            "target_kind": "binary",
            "horizon": "one_month",
            "trading_days": 21,
            "forecast_score": [-1, -0.5, 0.5, 1.0, 0.75],
            "forecast_value": [0.1, 0.2, 0.8, 0.9, 0.7],
            "baseline_value": [0.4] * 5,
            "realized_value": [0, 0, 1, 1, 1],
        }
    )

    summary = summarize_target_forecasts(forecasts)

    assert summary.loc[0, "model_brier"] < summary.loc[0, "baseline_brier"]
    assert summary.loc[0, "q5_q1_realized_spread"] > 0
