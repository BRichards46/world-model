from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.equity_risk_context import (
    _episodes,
    build_risk_context,
    classify_states,
    current_state,
    forward_outcomes,
    production_inflation_read,
    tier_for,
)

TRIGGERS = {
    "indpro_yoy": ("<=", 1.0),
    "payems_yoy": ("<=", 1.0),
    "unemployment_gap": (">", 0.1),
    "housing_lf_yoy": ("<=", -10.0),
    "cpi_yoy": (">=", 2.5),
}


def _values(index: pd.DatetimeIndex, **overrides: float) -> pd.DataFrame:
    base = {"indpro_yoy": 3.0, "payems_yoy": 2.0, "unemployment_gap": -0.2, "housing_lf_yoy": 0.0, "cpi_yoy": 2.0}
    base.update(overrides)
    return pd.DataFrame(base, index=index)


def test_states_split_invested_months_by_legacy_count() -> None:
    month_ends = pd.Series(np.linspace(100, 130, 24), index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    values = _values(pd.DatetimeIndex(["2019-12-15"]))
    clear = classify_states(month_ends, values, TRIGGERS, 2)
    assert set(clear["state"].dropna()) == {"normal"}
    legacy = classify_states(month_ends, _values(values.index, payems_yoy=0.5, cpi_yoy=3.7), TRIGGERS, 2)
    assert set(legacy["state"].dropna()) == {"normal_legacy"}
    assert legacy["legacy_count"].iloc[-1] == 2

    falling = pd.Series(np.linspace(130, 100, 24), index=month_ends.index)
    warned = classify_states(falling, _values(values.index, indpro_yoy=-1.0), TRIGGERS, 2)
    assert warned["state"].iloc[-1] == "defensive"
    assert classify_states(falling, values, TRIGGERS, 2)["state"].iloc[-1] == "market_warning"
    assert classify_states(month_ends, _values(values.index, unemployment_gap=0.3), TRIGGERS, 2)["state"].iloc[-1] == "macro_watch"


def test_forward_outcomes_measure_drawdown_and_volatility() -> None:
    days = pd.bdate_range("2020-01-01", periods=80)
    levels = pd.Series(100.0, index=days)
    levels.iloc[10:] = 90.0
    out = forward_outcomes(levels, [days[0], days[30]], horizon_days=20)
    assert np.isclose(out.loc[days[0], "max_drawdown"], -0.10)
    assert out.loc[days[30], "max_drawdown"] == 0.0 and out.loc[days[30], "volatility"] == 0.0


def test_tiers_and_episodes() -> None:
    assert tier_for(0.03, 0.10) == "below average"
    assert tier_for(0.12, 0.10) == "average"
    assert tier_for(0.20, 0.10) == "elevated"
    assert tier_for(0.36, 0.10) == "high"
    dates = pd.to_datetime(["1987-07-31", "1987-08-31", "1987-09-30", "1990-07-31"])
    assert _episodes(dates, 3) == ["1987-07", "1990-07"]


def test_context_flags_risk_above_the_usual_for_the_invested_stance() -> None:
    index = pd.date_range("2000-01-31", periods=40, freq="ME")
    states = pd.DataFrame({"state": ["normal"] * 20 + ["normal_legacy"] * 20}, index=index)
    drawdowns = [-0.02] * 20 + [-0.30] * 4 + [-0.02] * 16
    outcomes = pd.DataFrame({"max_drawdown": drawdowns, "volatility": [0.12] * 40}, index=index)
    context = build_risk_context(states, outcomes, {"state": "normal_legacy", "legacy_count": 2, "legacy_on": ["payrolls", "CPI"]}, None)
    assert context["tier"] == "elevated" and context["above_usual_for_stance"] is True  # 20% vs a 10% base
    assert context["history"]["states"]["normal_legacy"]["worst_drawdown_rate"] == 0.2
    calm = build_risk_context(states, outcomes, {"state": "normal"}, None)
    assert calm["tier"] == "below average" and calm["above_usual_for_stance"] is False


def test_current_state_uses_latest_close_and_macro_values() -> None:
    month_ends = pd.Series([100.0] * 9 + [110.0], index=pd.date_range("2026-02-28", periods=10, freq="ME"))
    values = _values(pd.DatetimeIndex(["2026-11-06"]), payems_yoy=0.56, cpi_yoy=3.7)
    now = current_state(month_ends, 110.0, values, TRIGGERS, 2)
    assert now["state"] == "normal_legacy" and now["legacy_on"] == ["payrolls", "CPI"] and now["legacy_total"] == 5


def test_production_inflation_read_needs_slowing_production_and_elevated_inflation() -> None:
    summary = pd.DataFrame(
        {
            "asset_group": ["equity"], "inflation_indicator": ["core_pce_inflation"], "inflation_condition": ["elevated"],
            "additive_robust_candidate": [True], "target": ["future_volatility"], "horizon": ["three_months"],
            "active_regime_mean_outcome": [0.127], "inactive_regime_mean_outcome": [0.114],
            "sample_start": ["2005-10-31"], "active_regime_observations": [63],
        }
    )
    snapshot = [
        {"indicator_id": "industrial_production", "velocity_value": -1.96, "observation_date": "2026-09-01T00:00:00"},
        {"indicator_id": "core_pce_inflation", "level_percentile_10y": 70.0},
        {"indicator_id": "core_cpi_inflation", "level_percentile_10y": None},
    ]
    read = production_inflation_read(snapshot, summary)
    assert read["active"] is True and read["active_with"] == ["core PCE"]
    assert read["measures"][0]["effects"][0]["active"] == 0.127
    snapshot[0]["velocity_value"] = 0.5
    assert production_inflation_read(snapshot, summary)["active"] is False
