from __future__ import annotations

import pandas as pd

from world_model.states.economic_asset_drivers import (
    asset_return_proxy,
    conditional_mean_return,
    forward_log_returns,
    walk_forward_contexts,
)


def test_forward_log_returns_use_next_period_only() -> None:
    returns = pd.DataFrame({"asset": [0.01, 0.02, -0.01, 0.03]}, index=pd.date_range("2020-01-01", periods=4))

    forward = forward_log_returns(returns, horizon_days=2)

    assert round(forward.loc["2020-01-01", "asset"], 4) == 0.01
    assert round(forward.loc["2020-01-02", "asset"], 4) == 0.02
    assert pd.isna(forward.loc["2020-01-03", "asset"])


def test_asset_return_proxy_handles_non_positive_back_adjusted_prices() -> None:
    price = pd.Series(
        [10.0, 11.0, -9.0, -8.0],
        index=pd.date_range("2020-01-01", periods=4),
    )

    returns = asset_return_proxy(
        price,
        method="normalized_price_change",
        normalization_window=2,
        normalization_min_periods=1,
    )

    assert round(returns.iloc[1], 4) == 0.1
    assert pd.notna(returns.iloc[2])
    assert pd.notna(returns.iloc[3])


def test_walk_forward_contexts_exclude_incomplete_prior_outcomes() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    target = pd.Series(range(6), index=dates, dtype=float)

    contexts = walk_forward_contexts(eval_dates=dates, target_values=target, horizon_days=21, min_train=2)

    assert contexts[0]["forecast_date"] == dates[2]
    assert contexts[0]["train_end_pos"] == 1


def test_conditional_mean_return_uses_matching_binary_bucket() -> None:
    feature = pd.Series([0, 0, 1, 1], dtype=float)
    target = pd.Series([-0.01, 0.00, 0.02, 0.04], dtype=float)

    forecast = conditional_mean_return(feature, target, current_feature=1.0, kind="binary", min_bucket_observations=1)

    assert round(forecast, 4) == 0.03
