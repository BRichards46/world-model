from __future__ import annotations

import pandas as pd

from world_model.states.combined import CombinedTarget
from world_model.states.simple_market import (
    SimpleMarketRegimeEngine,
    build_simple_validation_panel,
    build_simple_validation_targets,
    label_simple_regime,
    summarize_simple_forecasts,
)


def test_label_simple_regime_uses_four_block_directions() -> None:
    row = pd.Series(
        {
            "simple_composite_score": 0.1,
            "equity_score": 0.3,
            "rates_score": -0.4,
            "commodities_score": 0.0,
            "fx_score": 0.5,
        }
    )

    assert label_simple_regime(row) == "equity_up__rates_pressure__mixed__dollar_weak"


def test_simple_engine_builds_four_blocks_from_prices() -> None:
    index = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "ES": range(100, 360),
            "ZN": range(200, 460),
            "CL": range(50, 310),
            "DX": range(300, 560),
        },
        index=index,
        dtype=float,
    )
    config = {
        "blocks": {
            "equity": {"assets": ["ES"], "weight": 1.0},
            "rates": {"assets": ["ZN"], "weight": 1.0},
            "commodities": {"assets": ["CL"], "weight": 1.0},
            "fx": {"assets": ["DX"], "weight": 1.0, "invert": True},
        }
    }
    engine = SimpleMarketRegimeEngine({"csi_processed": ""}, config)
    engine.load_prices = lambda: prices

    state = engine.build_state_history()

    assert {"equity_score", "rates_score", "commodities_score", "fx_score"}.issubset(state.columns)
    assert state["fx_score"].dropna().iloc[-1] < 0.0


def test_build_simple_validation_panel_aliases_required_forecast_features() -> None:
    index = pd.date_range("2020-01-01", periods=2)
    simple = pd.DataFrame(
        {
            "simple_composite_score": [0.1, 0.2],
            "equity_score": [0.2, 0.3],
            "rates_score": [0.0, 0.1],
            "commodities_score": [0.1, 0.2],
            "fx_score": [-0.1, -0.2],
        },
        index=index,
    )
    detailed = pd.DataFrame({"composite_score": [-0.2, -0.1]}, index=index)

    panel = build_simple_validation_panel(simple, detailed)

    assert "detailed_market_composite_score" in panel
    assert "market_composite_score" in panel
    assert "economic_composite_score" in panel


def test_build_simple_validation_targets_includes_deterioration() -> None:
    index = pd.bdate_range("2020-01-01", periods=40)
    simple = pd.DataFrame({"simple_composite_score": [0.5] * 20 + [0.0] * 20}, index=index)
    prices = pd.DataFrame(
        {
            "ES": range(100, 140),
            "NQ": range(100, 140),
            "CL": range(100, 140),
            "HG": range(100, 140),
            "ZN": range(100, 140),
            "ZB": range(100, 140),
            "GC": range(100, 140),
        },
        index=index,
        dtype=float,
    )

    targets = build_simple_validation_targets(
        simple_state=simple,
        prices=prices,
        horizons={"one_month": 21},
        config={"simple_deterioration_threshold": -0.25},
    )

    names = {target.name for target in targets}
    assert "simple_regime_deterioration_one_month" in names
    assert all(isinstance(target, CombinedTarget) for target in targets)


def test_summarize_simple_forecasts_compares_to_detailed_market() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_date": list(pd.date_range("2020-01-31", periods=4, freq="ME")) * 3,
            "target": ["t"] * 12,
            "target_label": ["Target"] * 12,
            "target_kind": ["binary"] * 12,
            "horizon": ["one_month"] * 12,
            "trading_days": [21] * 12,
            "model": ["unconditional"] * 4 + ["detailed_market_bucket"] * 4 + ["simple_composite_bucket"] * 4,
            "model_label": ["Unconditional"] * 4 + ["Detailed"] * 4 + ["Simple"] * 4,
            "model_kind": ["unconditional"] * 4 + ["single_bucket"] * 4 + ["single_bucket"] * 4,
            "forecast_value": [0.5] * 4 + [0.4, 0.4, 0.6, 0.6] + [0.1, 0.1, 0.9, 0.9],
            "realized_value": [0.0, 0.0, 1.0, 1.0] * 3,
            "train_observations": [10] * 12,
            "bucket_observations": [10] * 12,
        }
    )

    summary = summarize_simple_forecasts(forecasts)
    simple = summary[summary["model"].eq("simple_composite_bucket")].iloc[0]

    assert simple["improvement_vs_detailed_market"] > 0.0
