from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.macro_regime_performance import (
    analyze_macro_regime_performance,
    build_regime_tape,
)


def test_regime_tape_uses_next_day_and_inverts_unemployment() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    rows = []
    values = {
        "payroll_growth": ([80, 80, 20, 20], [80, 80, 20, 20], "higher_strength"),
        "unemployment_rate": ([20, 20, 80, 80], [20, 20, 80, 80], "higher_slack"),
        "core_pce_inflation": ([30, 30, 80, 80], [30, 30, 80, 80], "higher_heat"),
        "core_cpi_inflation": ([30, 30, 80, 80], [30, 30, 80, 80], "higher_heat"),
        "real_policy_rate": ([50] * 4, [50] * 4, "higher_tightness"),
        "high_yield_spread": ([50] * 4, [50] * 4, "higher_stress"),
    }
    for indicator_id, (levels, velocities, polarity) in values.items():
        for date, level, velocity in zip(dates, levels, velocities):
            rows.append(
                {
                    "realtime_start": date,
                    "indicator_id": indicator_id,
                    "label": indicator_id,
                    "component": "test",
                    "polarity": polarity,
                    "is_context": False,
                    "level_percentile_10y": level,
                    "velocity_percentile_10y": velocity,
                }
            )
    config = {
        "availability_lag_trading_days": 1,
        "low_percentile": 40,
        "high_percentile": 60,
        "minimum_components": 2,
        "dimensions": {
            "growth": {"indicators": {"payroll_growth": 1, "unemployment_rate": -1}},
            "inflation": {"indicators": {"core_pce_inflation": 1, "core_cpi_inflation": 1}},
            "financial_conditions": {"indicators": {"real_policy_rate": 1, "high_yield_spread": 1}},
        },
    }

    tape = build_regime_tape(pd.DataFrame(rows), dates, config)

    assert pd.isna(tape.loc[dates[0], "macro_regime"])
    assert tape.loc[dates[1], "growth_level_score"] == 80.0
    assert tape.loc[dates[1], "macro_regime"] == "goldilocks_recovery"
    assert tape.loc[dates[3], "macro_regime"] == "stagflation_pressure"


def test_regime_performance_tracks_episodes_and_transitions() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="B")
    tape = pd.DataFrame(
        {
            "macro_regime": ["a"] * 6 + ["b"] * 6,
            "growth_state": ["strong__improving"] * 12,
            "inflation_state": ["contained__stable"] * 12,
            "financial_conditions_state": ["neutral"] * 12,
        },
        index=dates,
    )
    returns = pd.DataFrame({"equity": [0.01] * 6 + [-0.01] * 6}, index=dates)
    metadata = pd.DataFrame(
        {
            "asset_label": ["Equity"],
            "asset_type": ["class"],
            "parent_asset_group": ["equity"],
        },
        index=pd.Index(["equity"], name="asset_id"),
    )

    episodes, performance, transitions = analyze_macro_regime_performance(
        tape,
        returns,
        metadata,
        {"minimum_episode_days": 5},
    )

    macro_episodes = episodes.loc[episodes["regime_dimension"].eq("macro_regime")]
    assert len(macro_episodes) == 2
    a = performance.loc[
        performance["regime_dimension"].eq("macro_regime") & performance["regime_id"].eq("a")
    ].iloc[0]
    b = performance.loc[
        performance["regime_dimension"].eq("macro_regime") & performance["regime_id"].eq("b")
    ].iloc[0]
    assert a["annualized_performance"] > 0
    assert b["annualized_performance"] < 0
    transition = transitions.loc[
        transitions["regime_dimension"].eq("macro_regime")
        & transitions["from_regime"].eq("a")
        & transitions["to_regime"].eq("b")
    ].iloc[0]
    assert transition["transition_probability"] == 1.0
