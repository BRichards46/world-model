import numpy as np
import pandas as pd

from world_model.data.benchmarks import chain_price_total_return
from world_model.states.gtt_risk_validation import (
    _below_average_flag,
    _decline_flag,
    _latest_real_retail_yoy,
    _nonpositive_growth_flag,
    _unemployment_signal,
    block_bootstrap_incremental_lift,
    build_forward_log_risk_targets,
    build_gtt_episode_details,
    build_price_trends,
    summarize_gtt_risk,
)


def test_unemployment_signal_is_above_trailing_twelve_month_average() -> None:
    values = pd.Series(np.arange(12, dtype=float), index=pd.date_range("2020-01-01", periods=12, freq="MS"))
    current, average, danger = _unemployment_signal(values)
    assert current == 11.0
    assert average == 5.5
    assert danger == 1.0


def test_real_retail_growth_uses_only_common_observation_months() -> None:
    dates = pd.date_range("2019-01-01", periods=25, freq="MS")
    retail = pd.Series(np.linspace(100, 120, 25), index=dates)
    cpi = pd.Series(np.linspace(100, 110, 25), index=dates)
    cpi = cpi.drop(dates[-1])
    result = _latest_real_retail_yoy(retail, cpi)
    expected = 100 * ((retail.iloc[-2] / cpi.iloc[-1]) / (retail.iloc[-14] / cpi.iloc[-13]) - 1)
    assert np.isclose(result, expected)


def test_production_signal_transforms_use_trailing_values() -> None:
    values = pd.Series(
        [100.0, 101.0, 102.0, 99.0],
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    assert _nonpositive_growth_flag(values, 3) == 1.0
    assert _below_average_flag(values, 3) == 1.0
    assert _decline_flag(values, 3) == 1.0


def test_price_trend_uses_trailing_month_end_information() -> None:
    dates = pd.date_range("2020-01-01", "2020-06-30", freq="B")
    returns = pd.DataFrame({"equity": np.r_[np.repeat(0.1, len(dates) - 20), np.repeat(-1.0, 20)]}, index=dates)
    trends = build_price_trends(returns, ten_month_window=3, daily_window=20)
    monthly = trends["equity__ten_month"].dropna()
    assert monthly.index.to_period("M").is_unique
    assert monthly.iloc[-1] == 1.0


def test_hybrid_index_uses_total_return_changes_when_available() -> None:
    dates = pd.date_range("1987-12-30", periods=6, freq="B")
    price = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=dates)
    total_return = pd.Series([200.0, 204.0, 208.0], index=dates[-3:])

    history = chain_price_total_return(price, total_return)

    assert history.loc[dates[-3], "return_source"] == "price_return"
    assert history.loc[dates[-2], "return_source"] == "total_return"
    assert np.isclose(history.loc[dates[-2], "log_return"], np.log(204.0 / 200.0))
    expected_path_change = (204.0 / 200.0) * (208.0 / 204.0)
    actual_path_change = history.loc[dates[-1], "normalized_index"] / history.loc[dates[-3], "normalized_index"]
    assert np.isclose(actual_path_change, expected_path_change)


def test_log_return_drawdown_is_compounded_percentage() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    log_returns = pd.DataFrame({"sp500": np.log([1.0, 1.10, 0.80, 1.05])}, index=dates)
    targets = build_forward_log_risk_targets(log_returns, horizon_days=3)
    assert np.isclose(targets["max_drawdown"].iloc[0, 0], -0.20)


def test_summary_requires_increment_beyond_price_trend_only() -> None:
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    frame = pd.DataFrame(
        {
            "forecast_date": dates,
            "asset_group": "equity",
            "asset_group_label": "Equity",
            "macro_signal": "unemployment_trend",
            "price_trend": "ten_month",
            "horizon": "one_month",
            "target": "future_volatility",
            "macro_danger": [1] * 4 + [0] * 4 + [0] * 4,
            "price_downtrend": [1] * 8 + [0] * 4,
            "double_confirmation": [1] * 4 + [0] * 8,
            "realized_target": [3, 4, 5, 6, 1, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    summary = summarize_gtt_risk(frame, minimum_per_state=4).iloc[0]
    assert summary["risk_lift"] > 0
    assert summary["incremental_lift_vs_trend_only"] > 0


def test_block_bootstrap_detects_consistently_positive_incremental_lift() -> None:
    risk = pd.Series(np.tile([4.0, 1.0], 60))
    active = pd.Series(np.tile([True, False], 60))
    trend_only = ~active
    low, high, p_nonpositive = block_bootstrap_incremental_lift(
        risk, active, trend_only, block_length=12, iterations=200, confidence=0.90, seed=7
    )
    assert low > 0
    assert high >= low
    assert p_nonpositive == 0.0


def test_episode_details_separate_noncontiguous_active_runs() -> None:
    dates = pd.to_datetime(["2000-01-31", "2000-02-29", "2000-06-30", "2000-07-31"])
    frame = pd.DataFrame(
        {
            "forecast_date": dates,
            "asset_group": "equity",
            "asset_group_label": "Equity",
            "macro_signal": "dual_growth",
            "price_trend": "ten_month",
            "horizon": "one_month",
            "target": "max_drawdown",
            "macro_danger": [1, 1, 1, 1],
            "price_downtrend": [1, 1, 1, 1],
            "double_confirmation": [1, 1, 1, 1],
            "realized_target": [-0.1, -0.2, -0.3, -0.4],
        }
    )
    episodes = build_gtt_episode_details(frame)
    assert len(episodes) == 2
    assert episodes["episode_observations"].tolist() == [2, 2]
    assert episodes["stress_era_id"].tolist() == [1, 1]
