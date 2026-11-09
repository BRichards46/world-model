from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.quad_asset_performance import (
    analyze,
    build_quad_signals,
    forward_returns_at,
)


def test_signals_pivot_walk_forward_by_horizon() -> None:
    walk_forward = pd.DataFrame(
        [
            {"forecast_date": "2020-01-31", "horizon": 0, "projected_quad": 2, "persistence_quad": 4, "realized_quad": 1},
            {"forecast_date": "2020-01-31", "horizon": 1, "projected_quad": 3, "persistence_quad": 4, "realized_quad": None},
        ]
    )
    signals = build_quad_signals(walk_forward)
    row = signals.loc[pd.Timestamp("2020-01-31")]
    assert row["realized_last"] == 4
    assert row["projected_current"] == 2
    assert row["hindsight_current"] == 1
    assert row["projected_next"] == 3


def test_forward_returns_start_after_the_availability_lag() -> None:
    calendar = pd.bdate_range("2020-01-27", periods=10)
    cumulative = {"asset": pd.Series(np.arange(10, dtype=float), index=calendar)}
    # Signal on Friday 2020-01-31 (position 4); lag 1 starts Monday (5); 3 sessions later is position 8.
    forward = forward_returns_at(pd.DatetimeIndex(["2020-01-31"]), calendar, cumulative, horizon_days=3, lag_days=1)
    assert forward.loc["2020-01-31", "asset"] == 3.0
    late = forward_returns_at(pd.DatetimeIndex(["2020-02-06"]), calendar, cumulative, horizon_days=3, lag_days=1)
    assert np.isnan(late.loc["2020-02-06", "asset"])


def test_analyze_detects_a_planted_quad_effect_and_not_noise() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("1998-01-31", periods=240, freq="ME")
    quads = pd.Series(np.repeat(rng.integers(1, 5, size=80), 3), index=dates, dtype=float)
    effect = np.where(quads.to_numpy() == 3, -3.0, 0.5)
    forward = pd.DataFrame(
        {"planted": effect + rng.normal(0, 1.0, len(dates)), "noise": rng.normal(0, 1.0, len(dates))},
        index=dates,
    )
    metadata = pd.DataFrame(
        {"asset_label": ["Planted", "Noise"], "asset_group": ["test", "test"], "units": ["%", "%"]},
        index=["planted", "noise"],
    )
    quad_summary, joint = analyze(
        pd.DataFrame({"projected_current": quads}),
        {"one_month": forward},
        metadata,
        {"minimum_months_per_quad": 24, "minimum_rotation_gap_months": 12},
        {"one_month": 21},
    )
    planted = quad_summary[(quad_summary["asset_id"] == "planted") & (quad_summary["quad"] == 3)].iloc[0]
    assert planted["annualized_spread"] < 0
    assert planted["p_value"] < 0.05
    assert joint.set_index("asset_id").loc["planted", "p_value"] < joint.set_index("asset_id").loc["noise", "p_value"]
