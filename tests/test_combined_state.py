from __future__ import annotations

import pandas as pd

from world_model.states.combined import (
    CombinedTarget,
    build_combined_state_panel,
    joint_bucket_forecast,
    single_bucket_forecast,
    summarize_combined_forecasts,
    target_contexts,
)


def test_single_bucket_forecast_uses_current_feature_bucket() -> None:
    train_feature = pd.Series(range(10), dtype=float)
    train_target = pd.Series([0.0] * 5 + [1.0] * 5)

    forecast, count = single_bucket_forecast(
        train_feature=train_feature,
        train_target=train_target,
        current_value=9.0,
        bucket_count=2,
        min_bucket_observations=2,
    )

    assert forecast == 1.0
    assert count == 5


def test_joint_bucket_forecast_requires_both_current_buckets() -> None:
    train_left = pd.Series([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    train_right = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    train_target = pd.Series([0, 0, 0, 1, 0, 0, 0, 1], dtype=float)

    forecast, count = joint_bucket_forecast(
        train_left=train_left,
        train_right=train_right,
        train_target=train_target,
        current_left=1.0,
        current_right=1.0,
        bucket_count=2,
        min_bucket_observations=2,
    )

    assert forecast == 1.0
    assert count == 2


def test_target_contexts_enforce_horizon_gap() -> None:
    eval_dates = pd.bdate_range("2020-01-31", periods=8, freq="ME")
    target_values = pd.Series(range(8), index=eval_dates, dtype=float)
    target = CombinedTarget(
        name="target",
        label="Target",
        kind="continuous",
        horizon="one_month",
        trading_days=21,
        target=target_values,
    )

    contexts = target_contexts(eval_dates=eval_dates, target_values=target_values, target=target, min_train=2)

    assert contexts
    assert contexts[0]["train_end_pos"] < contexts[0]["eval_pos"]


def test_summarize_combined_forecasts_compares_to_market_bucket() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_date": list(pd.date_range("2020-01-31", periods=4, freq="ME")) * 3,
            "target": ["t"] * 12,
            "target_label": ["Target"] * 12,
            "target_kind": ["binary"] * 12,
            "horizon": ["one_month"] * 12,
            "trading_days": [21] * 12,
            "model": ["unconditional"] * 4 + ["market_bucket"] * 4 + ["market_x_economic"] * 4,
            "model_label": ["Unconditional"] * 4 + ["Market"] * 4 + ["Joint"] * 4,
            "model_kind": ["unconditional"] * 4 + ["single_bucket"] * 4 + ["joint_bucket"] * 4,
            "forecast_value": [0.5] * 4 + [0.4, 0.4, 0.6, 0.6] + [0.1, 0.1, 0.9, 0.9],
            "realized_value": [0.0, 0.0, 1.0, 1.0] * 3,
            "train_observations": [10] * 12,
            "bucket_observations": [10] * 12,
        }
    )

    summary = summarize_combined_forecasts(forecasts)
    joint = summary[summary["model"].eq("market_x_economic")].iloc[0]

    assert joint["improvement_vs_market_bucket"] > 0.0


def test_build_combined_state_panel_exposes_named_axes() -> None:
    index = pd.date_range("2020-01-01", periods=2)
    market = pd.DataFrame(
        {
            "composite_score": [0.1, 0.2],
            "component_coverage": [1.0, 1.0],
        },
        index=index,
    )
    economic = pd.DataFrame(
        {
            "composite_score": [-0.1, -0.2],
            "component_coverage": [1.0, 1.0],
            "activity_score": [0.0, 0.1],
            "inflation_pressure_score": [0.2, 0.3],
            "liquidity_credit_score": [-0.2, -0.3],
        },
        index=index,
    )

    panel = build_combined_state_panel(market, economic)

    assert "market_composite_score" in panel
    assert "economic_composite_score" in panel
    assert "activity_score" in panel
