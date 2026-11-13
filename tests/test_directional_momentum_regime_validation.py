from __future__ import annotations

import pandas as pd

from world_model.states.directional_momentum_regime_validation import (
    build_matching_horizon_regimes,
    component_state,
)


def test_component_state_requires_direction_agreement_and_threshold() -> None:
    assert component_state(0.10, 0.50, 0.25) == "positive"
    assert component_state(-0.10, -0.50, 0.25) == "negative"
    assert component_state(0.10, 0.20, 0.25) == "neutral"
    assert component_state(0.10, -0.50, 0.25) == "neutral"


def test_matching_horizon_regime_uses_all_four_assets() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    records = []
    values = {"ES": 0.5, "NQ": 0.6, "DX": -0.7, "ZB": 0.1}
    for asset, adjusted in values.items():
        for date in dates:
            records.append(
                {
                    "date": date,
                    "asset_id": asset,
                    "horizon_id": "one_month",
                    "horizon_label": "1m",
                    "horizon_days": 21,
                    "raw_momentum": 0.1 if adjusted > 0 else -0.1,
                    "vol_adjusted_momentum": adjusted,
                }
            )
    result = build_matching_horizon_regimes(
        pd.DataFrame(records),
        {
            "assets": ["ES", "NQ", "DX", "ZB"],
            "evaluation_frequency": "W-FRI",
            "component_signal_threshold": 0.25,
        },
    )

    assert result.iloc[-1]["regime_id"] == "ES+ / NQ+ / DX- / ZB0"
