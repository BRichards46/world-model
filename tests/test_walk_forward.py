from __future__ import annotations

import pandas as pd

from world_model.states.walk_forward import (
    summarize_probability_calibration,
    summarize_quintiles,
    summarize_walk_forward,
)


def _sample_forecasts() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=10)
    expected = [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    realized = [-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    return pd.DataFrame(
        {
            "forecast_date": dates,
            "horizon": "one_month",
            "trading_days": 21,
            "weighted_mean_return": expected,
            "realized_return": realized,
            "baseline_mean_return": 0.01,
            "random_mean_return": 0.01,
            "p25_return": [-0.04] * 10,
            "p75_return": [0.07] * 10,
            "p10_return": [-0.05] * 10,
            "p90_return": [0.08] * 10,
            "probability_positive": [0.3, 0.35, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            "probability_drawdown_gt_10pct": [0.1] * 10,
            "baseline_probability_drawdown_gt_10pct": [0.2] * 10,
            "realized_max_drawdown": [-0.02] * 10,
        }
    )


def test_summarize_walk_forward_includes_quintile_spread() -> None:
    summary = summarize_walk_forward(_sample_forecasts())

    assert summary.loc[0, "horizon"] == "one_month"
    assert summary.loc[0, "observations"] == 10
    assert summary.loc[0, "top_bottom_realized_spread"] > 0
    assert summary.loc[0, "forecast_realized_corr"] > 0.99


def test_summarize_quintiles_orders_buckets() -> None:
    quintiles = summarize_quintiles(_sample_forecasts())

    assert quintiles["forecast_quintile"].tolist() == [1, 2, 3, 4, 5]
    assert quintiles.iloc[-1]["mean_realized_return"] > quintiles.iloc[0]["mean_realized_return"]


def test_probability_calibration_buckets() -> None:
    calibration = summarize_probability_calibration(_sample_forecasts())

    assert not calibration.empty
    assert "probability_positive_bucket" in calibration.columns
