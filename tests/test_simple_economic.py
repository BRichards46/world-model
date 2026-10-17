from __future__ import annotations

import pandas as pd

from world_model.states.combined import CombinedTarget
from world_model.states.simple_economic import (
    SimpleEconomicRegimeEngine,
    build_simple_economic_validation_panel,
    build_simple_economic_validation_targets,
    label_simple_economic_regime,
    summarize_simple_economic_forecasts,
)


def test_label_simple_economic_regime_uses_activity_inflation_and_liquidity() -> None:
    row = pd.Series(
        {
            "activity_score": -0.4,
            "inflation_pressure_score": 0.3,
            "liquidity_credit_score": -0.5,
        }
    )

    assert label_simple_economic_regime(row) == "stagflation_pressure_liquidity_tight"


def test_simple_engine_builds_three_axes_and_inverts_inflation_in_composite() -> None:
    index = pd.date_range("2020-01-01", periods=2)
    detailed = pd.DataFrame(
        {
            "activity_score": [0.4, 0.4],
            "inflation_pressure_score": [-0.5, 0.5],
            "liquidity_credit_score": [0.2, 0.2],
            "composite_score": [0.3, 0.1],
        },
        index=index,
    )
    engine = SimpleEconomicRegimeEngine({}, {}, {})

    state = engine.build_state_history(detailed)

    assert {"activity_score", "inflation_pressure_score", "liquidity_credit_score"}.issubset(state.columns)
    assert state["simple_economic_composite_score"].iloc[0] > state["simple_economic_composite_score"].iloc[1]
    assert state["axis_coverage"].iloc[-1] == 1.0


def test_build_simple_economic_validation_panel_aliases_required_forecast_features() -> None:
    index = pd.date_range("2020-01-01", periods=2)
    simple = pd.DataFrame(
        {
            "simple_economic_composite_score": [0.1, 0.2],
            "activity_score": [0.2, 0.3],
            "inflation_pressure_score": [0.0, 0.1],
            "liquidity_credit_score": [-0.1, -0.2],
        },
        index=index,
    )
    detailed = pd.DataFrame({"composite_score": [-0.2, -0.1]}, index=index)
    market = pd.DataFrame({"composite_score": [0.4, 0.5]}, index=index)

    panel = build_simple_economic_validation_panel(simple, detailed, market)

    assert "detailed_economic_composite_score" in panel
    assert "detailed_market_composite_score" in panel
    assert "market_composite_score" in panel
    assert "economic_composite_score" in panel


def test_build_simple_economic_validation_targets_includes_economic_deterioration() -> None:
    index = pd.bdate_range("2020-01-01", periods=40)
    simple = pd.DataFrame({"simple_economic_composite_score": [0.5] * 20 + [0.0] * 20}, index=index)
    detailed = pd.DataFrame({"composite_score": [0.4] * 20 + [0.1] * 20}, index=index)

    targets = build_simple_economic_validation_targets(
        simple_state=simple,
        detailed_state=detailed,
        market_state=None,
        prices=None,
        horizons={"one_month": 21},
        config={"simple_deterioration_threshold": -0.25, "detailed_deterioration_threshold": -0.25},
    )

    names = {target.name for target in targets}
    assert "simple_economic_deterioration_one_month" in names
    assert "detailed_economic_deterioration_one_month" in names
    assert all(isinstance(target, CombinedTarget) for target in targets)


def test_summarize_simple_economic_forecasts_compares_to_detailed_economic() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_date": list(pd.date_range("2020-01-31", periods=4, freq="ME")) * 3,
            "target": ["t"] * 12,
            "target_label": ["Target"] * 12,
            "target_kind": ["binary"] * 12,
            "horizon": ["one_month"] * 12,
            "trading_days": [21] * 12,
            "model": ["unconditional"] * 4 + ["detailed_economic_bucket"] * 4 + ["simple_economic_bucket"] * 4,
            "model_label": ["Unconditional"] * 4 + ["Detailed"] * 4 + ["Simple"] * 4,
            "model_kind": ["unconditional"] * 4 + ["single_bucket"] * 4 + ["single_bucket"] * 4,
            "forecast_value": [0.5] * 4 + [0.4, 0.4, 0.6, 0.6] + [0.1, 0.1, 0.9, 0.9],
            "realized_value": [0.0, 0.0, 1.0, 1.0] * 3,
            "train_observations": [10] * 12,
            "bucket_observations": [10] * 12,
        }
    )

    summary = summarize_simple_economic_forecasts(forecasts)
    simple = summary[summary["model"].eq("simple_economic_bucket")].iloc[0]

    assert simple["improvement_vs_detailed_economic"] > 0.0
