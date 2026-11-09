import numpy as np
import pandas as pd

from world_model.config import default_config_path, load_yaml
from world_model.states.macro_market_validation import (
    benjamini_hochberg,
    build_forward_market_targets,
    build_macro_feature_panel,
    cumulative_linear_forecasts,
    expanding_rank,
    forward_max_drawdown,
    individual_asset_config,
    rank_linear_forecast,
)


def test_configured_equity_universe_is_es_and_nq() -> None:
    config = load_yaml(default_config_path("economic_asset_drivers.yml"))

    assert config["asset_groups"]["equity"]["assets"] == {
        "ES": "ES__B.parquet",
        "NQ": "NQ__B.parquet",
    }


def test_individual_asset_config_expands_each_contract_without_changing_source() -> None:
    source = {
        "return_method": "log",
        "asset_groups": {
            "equity": {
                "label": "Equity futures",
                "assets": {"ES": "es.parquet", "NQ": "nq.parquet"},
            }
        },
    }

    expanded = individual_asset_config(source)

    assert set(expanded["asset_groups"]) == {"equity__es", "equity__nq"}
    assert expanded["asset_groups"]["equity__es"]["assets"] == {"ES": "es.parquet"}
    assert expanded["asset_groups"]["equity__es"]["label"] == "ES · Equity futures"
    assert source["asset_groups"]["equity"]["assets"] == {"ES": "es.parquet", "NQ": "nq.parquet"}


def _macro_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "realtime_start": pd.to_datetime(["2020-01-10", "2020-02-10"]),
            "indicator_id": ["series_a", "series_a"],
            "label": ["Series A", "Series A"],
            "component": ["activity_labor", "activity_labor"],
            "polarity": ["higher_strength", "higher_strength"],
            "is_context": [False, False],
            "level_value": [1.0, 2.0],
            "velocity_value": [0.1, 0.2],
            "level_percentile_10y": [40.0, 60.0],
            "velocity_percentile_10y": [30.0, 70.0],
        }
    )


def test_macro_feature_panel_uses_release_date_without_lookahead() -> None:
    index = pd.date_range("2020-01-01", "2020-02-20", freq="B")
    panel, metadata = build_macro_feature_panel(
        _macro_history(),
        index,
        ["level_value", "level_percentile_10y", "velocity_percentile_10y", "level_velocity_interaction"],
    )
    assert pd.isna(panel.loc["2020-01-09", "series_a__level_value"])
    assert panel.loc["2020-01-10", "series_a__level_value"] == 1.0
    assert panel.loc["2020-02-07", "series_a__level_value"] == 1.0
    assert panel.loc["2020-02-10", "series_a__level_value"] == 2.0
    assert np.isclose(panel.loc["2020-02-10", "series_a__level_velocity_interaction"], 0.08)
    assert metadata.loc["series_a__level_value", "indicator_label"] == "Series A"


def test_forward_max_drawdown_uses_only_future_returns() -> None:
    returns = pd.DataFrame({"asset": [0.5, 0.1, -0.3, 0.1, 0.2]}, index=pd.date_range("2020-01-01", periods=5))
    result = forward_max_drawdown(returns, horizon_days=3)
    assert np.isclose(result.iloc[0, 0], -0.3)
    assert np.isclose(result.iloc[1, 0], -0.3)
    assert pd.isna(result.iloc[2, 0])


def test_forward_targets_use_each_assets_own_valid_trading_calendar() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    returns = pd.DataFrame(
        {
            "asset_a": [0.1, 0.2, np.nan, 0.3, 0.4, 0.5],
            "asset_b": [1.0, np.nan, 2.0, 3.0, 4.0, 5.0],
        },
        index=index,
    )
    targets = build_forward_market_targets(returns, horizon_days=2, targets=["forward_return"])
    forward = targets["forward_return"]
    assert np.isclose(forward.loc["2020-01-02", "asset_a"], 0.7)
    assert np.isclose(forward.loc["2020-01-01", "asset_b"], 5.0)


def test_rank_linear_forecast_adds_incremental_feature() -> None:
    index = pd.RangeIndex(20)
    features = pd.DataFrame({"market": np.linspace(0, 1, 20), "macro": np.linspace(1, 2, 20)}, index=index)
    target = features["market"] + 2 * features["macro"]
    forecast = rank_linear_forecast(features, target, pd.Series({"market": 1.0, "macro": 2.0}))
    assert forecast > target.mean()


def test_benjamini_hochberg_is_monotone_in_sorted_p_values() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, np.nan])
    adjusted = benjamini_hochberg(p_values)
    ordered = adjusted.dropna().sort_values()
    assert ordered.is_monotonic_increasing
    assert adjusted.iloc[0] <= adjusted.iloc[1]


def test_expanding_rank_does_not_use_future_values() -> None:
    values = pd.Series([10.0, 20.0, 0.0])
    ranks = expanding_rank(values)
    assert ranks.iloc[0] == 0.5
    assert ranks.iloc[1] == 0.75
    assert np.isclose(ranks.iloc[2], 1 / 6)


def test_cumulative_forecast_respects_train_end_position() -> None:
    predictors = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0.0, 1.0, 2.0, 100.0])
    contexts = [{"eval_pos": 3, "train_end_pos": 2}]
    forecasts = cumulative_linear_forecasts(predictors, target, contexts, min_train=3)
    assert np.isclose(forecasts[3][0], 3.0)
