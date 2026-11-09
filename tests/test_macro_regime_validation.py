from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.macro_regime_validation import (
    build_combination_panel,
    summarize_macro_regime_forecasts,
)


def test_combination_panel_only_activates_adverse_corner() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    history = pd.DataFrame(
        {
            "realtime_start": list(dates) * 2,
            "indicator_id": ["industrial_production"] * 4 + ["core_pce_inflation"] * 4,
            "label": ["Industrial production"] * 4 + ["Core PCE inflation"] * 4,
            "component": ["activity_labor"] * 4 + ["inflation"] * 4,
            "polarity": ["higher_strength"] * 4 + ["higher_heat"] * 4,
            "is_context": [False] * 8,
            "level_percentile_10y": [50, 50, 50, 50, 20, 80, 80, 20],
            "velocity_percentile_10y": [20, 20, 80, 80, 50, 50, 50, 50],
            "velocity_value": [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    definitions = [
        {
            "id": "test",
            "label": "Test",
            "growth_indicator": "industrial_production",
            "growth_metric": "velocity_value",
            "inflation_indicator": "core_pce_inflation",
            "inflation_metric": "level_percentile_10y",
            "inflation_condition": "elevated",
        }
    ]

    panel = build_combination_panel(history, dates, definitions)["test"]

    assert panel["interaction_score"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_regime_summary_requires_incremental_oos_value() -> None:
    dates = pd.date_range("2000-01-31", periods=80, freq="ME")
    interaction = np.tile([0.0, 0.2, 0.6, 1.0], 20)
    realized = interaction + np.linspace(0.0, 0.1, len(dates))
    frame = pd.DataFrame(
        {
            "forecast_date": dates,
            "asset_group": "equity",
            "asset_group_label": "U.S. equity index futures (ES + NQ)",
            "target": "future_volatility",
            "horizon": "one_month",
            "trading_days": 21,
            "combination_id": "test",
            "combination_label": "Test combination",
            "growth_indicator": "industrial_production",
            "inflation_indicator": "core_pce_inflation",
            "inflation_condition": "elevated",
            "growth_weakness": np.tile([0.0, 0.4, 0.8, 1.0], 20),
            "inflation_pressure": np.tile([0.0, 0.5, 0.75, 1.0], 20),
            "interaction_score": interaction,
            "realized_target": realized,
            "market_forecast": np.zeros(len(dates)),
            "component_forecast": np.zeros(len(dates)),
            "combination_forecast": realized,
        }
    )

    result = summarize_macro_regime_forecasts(
        frame,
        {"minimum_oos_observations": 60, "fdr_alpha": 0.10},
    )

    assert result.loc[0, "combination_improvement_vs_components"] > 0
    assert bool(result.loc[0, "robust_candidate"])
