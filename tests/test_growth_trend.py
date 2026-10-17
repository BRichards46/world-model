from __future__ import annotations

import pandas as pd

from world_model.states.growth_trend import (
    build_growth_filter,
    build_growth_trend_panel,
    build_strategy_exposure,
    build_trend_signal,
    strategy_returns,
    summarize_annual_returns,
    summarize_timing_strategies,
)


def test_growth_trend_exposure_stays_long_when_growth_is_healthy() -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    panel = pd.DataFrame(
        {
            "growth_test": [1.0, 1.0, 0.0, 0.0],
            "trend_test": [0.0, 1.0, 0.0, 1.0],
        },
        index=index,
    )
    strategy = {"id": "gtt", "kind": "growth_trend", "growth_filter": "test", "trend_rule": "test"}

    exposure = build_strategy_exposure(panel, strategy)

    assert exposure.tolist() == [1.0, 1.0, 0.0, 1.0]


def test_strategy_returns_use_prior_day_exposure() -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    target_return = pd.Series([0.0, 0.10, -0.10, 0.20], index=index)
    defensive_return = pd.Series(0.0, index=index)
    exposure = pd.Series([0.0, 0.0, 1.0, 1.0], index=index)

    returns = strategy_returns(target_return, defensive_return, exposure, one_way_transaction_cost=0.0)

    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == 0.0
    assert returns.iloc[2] == 0.0
    assert returns.iloc[3] == 0.20


def test_daily_trend_signal_uses_price_above_average() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    price = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0], index=index)

    signal = build_trend_signal(price, {"frequency": "B", "window": 3, "min_periods": 3})

    assert pd.isna(signal.iloc[1])
    assert signal.iloc[2] == 1.0
    assert signal.iloc[-1] == 0.0


def test_growth_filter_supports_moving_average_rules() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    economic = pd.DataFrame({"unemployment_rate_value": [5.0, 5.0, 5.0, 6.0, 7.0]}, index=index)
    growth_filter = {
        "healthy_when": "all",
        "rules": [
            {
                "column": "unemployment_rate_value",
                "operator": "<=",
                "moving_average_window": 3,
                "min_periods": 3,
            }
        ],
    }

    healthy = build_growth_filter(economic, growth_filter)

    assert healthy.iloc[2] == 1.0
    assert healthy.iloc[-1] == 0.0


def test_growth_trend_panel_builds_expected_strategy_columns() -> None:
    index = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.DataFrame({"ES": range(100, 360)}, index=index, dtype=float)
    economic = pd.DataFrame(
        {
            "real_retail_sales_yoy_value": [0.01] * 130 + [-0.01] * 130,
            "industrial_production_yoy_value": [0.01] * 260,
        },
        index=index,
    )
    config = {
        "target_asset": "ES",
        "trend_rules": {"daily": {"frequency": "B", "window": 20, "min_periods": 20}},
        "growth_filters": {
            "growth": {
                "healthy_when": "all",
                "rules": [
                    {"column": "real_retail_sales_yoy_value", "operator": ">", "threshold": 0.0},
                    {"column": "industrial_production_yoy_value", "operator": ">", "threshold": 0.0},
                ],
            }
        },
        "strategies": [
            {"id": "buy_hold", "kind": "buy_hold", "label": "Buy and hold"},
            {
                "id": "gtt",
                "kind": "growth_trend",
                "label": "GTT",
                "growth_filter": "growth",
                "trend_rule": "daily",
            },
        ],
    }

    panel = build_growth_trend_panel(prices, economic, config, {"target_asset": "ES"})

    assert "exposure_gtt" in panel
    assert "return_gtt" in panel
    assert "equity_gtt" in panel


def test_summary_compares_against_buy_hold() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    panel = pd.DataFrame(
        {
            "target_price": [100, 110, 100, 120, 130],
            "return_buy_hold": [None, 0.10, -0.09, 0.20, 0.08],
            "return_test": [None, 0.10, 0.00, 0.20, 0.08],
            "exposure_buy_hold": [1, 1, 1, 1, 1],
            "exposure_test": [1, 1, 0, 1, 1],
        },
        index=index,
        dtype=float,
    )
    strategies = [
        {"id": "buy_hold", "kind": "buy_hold", "label": "Buy and hold"},
        {"id": "test", "kind": "trend", "label": "Test"},
    ]

    summary = summarize_timing_strategies(panel, strategies)
    test_row = summary[summary["strategy"].eq("test")].iloc[0]

    assert test_row["excess_cagr_vs_buy_hold"] > 0.0


def test_annual_returns_use_common_strategy_window() -> None:
    index = pd.bdate_range("2020-12-30", periods=5)
    panel = pd.DataFrame(
        {
            "return_buy_hold": [None, 0.10, 0.10, 0.10, 0.10],
            "return_test": [None, 0.00, 0.20, 0.00, 0.20],
        },
        index=index,
        dtype=float,
    )
    strategies = [
        {"id": "buy_hold", "label": "Buy and hold"},
        {"id": "test", "label": "Test"},
    ]

    annual = summarize_annual_returns(panel, strategies)
    test_2021 = annual[(annual["strategy"].eq("test")) & (annual["year"].eq(2021))].iloc[0]

    assert round(test_2021["return"], 6) == 0.44
    assert bool(test_2021["is_partial_year"])
