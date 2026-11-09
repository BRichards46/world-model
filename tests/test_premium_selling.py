from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.data.tanuki import select_delta_expiry
from world_model.features.premium_selling import conditional_path_snapshot, touch_statistics
from world_model.states.premium_selling import _direction_state, _strategy_state, _volatility_state


def _prices(rows: int = 900) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    returns = 0.0004 + np.sin(np.arange(rows) / 31) * 0.0015
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.008, "low": close * 0.992, "close": close}, index=index)


def test_conditional_path_snapshot_keeps_forward_outcomes_out_of_current_state() -> None:
    horizons = [
        {"id": "one_month", "days": 21, "weight": 0.25},
        {"id": "three_months", "days": 63, "weight": 0.25},
        {"id": "six_months", "days": 126, "weight": 0.25},
        {"id": "twelve_months", "days": 252, "weight": 0.25},
    ]
    result = conditional_path_snapshot(_prices(), horizons, forward_days=21, analog_count=40)
    assert result["analog_count"] == 40
    assert result["sample_end"] < result["observation_date"]
    assert set(result["components"]) == {item["id"] for item in horizons}
    assert result["conditional"]["median_downside_mae"] <= result["conditional"]["median_upside_mfe"]


def test_touch_statistics_uses_intraperiod_extremes_and_terminal_separately() -> None:
    paths = pd.DataFrame({
        "downside_mae": [-0.08, -0.02],
        "upside_mfe": [0.03, 0.09],
        "terminal_return": [-0.01, 0.02],
    })
    result = touch_statistics(paths, put_distance_pct=5, call_distance_pct=5)
    assert result["put_touch_rate"] == 0.5
    assert result["call_touch_rate"] == 0.5
    assert result["put_terminal_breach_rate"] == 0.0
    assert result["call_terminal_breach_rate"] == 0.0


def test_select_delta_expiry_uses_latest_snapshot_and_nearest_target() -> None:
    frame = pd.DataFrame({
        "symbol": ["SPY", "SPY", "SPY"],
        "snapshot_at": ["2026-10-26T20:00:00Z", "2026-10-27T20:00:00Z", "2026-10-27T20:00:00Z"],
        "dte": [30, 28, 31],
        "is_monthly": [True, True, False],
    })
    result = select_delta_expiry(frame, target_dte=30, minimum_dte=20, maximum_dte=45)
    assert len(result) == 1
    assert int(result.iloc[0]["dte"]) == 31


def test_strategy_requires_rich_iv_and_improved_directional_touch() -> None:
    config = {
        "direction_return_pct": 0.5,
        "direction_positive_rate_pct": 55,
        "rich_iv_edge_vol_pts": 2,
        "cheap_iv_edge_vol_pts": -1,
        "minimum_touch_improvement_pct": 3,
        "strangle_touch_ceiling_pct": 35,
    }
    direction = _direction_state({"median_return": 0.012, "positive_rate": 0.60}, config)
    volatility = _volatility_state(4.0, config)
    strategy = _strategy_state(
        direction,
        volatility,
        {"put_touch_rate": 0.18, "call_touch_rate": 0.32},
        {"put_touch_rate": 0.25, "call_touch_rate": 0.28},
        config,
    )
    assert strategy["id"] == "short_put"
    assert _strategy_state(direction, _volatility_state(0.5, config), {}, {}, config)["id"] == "wait"
