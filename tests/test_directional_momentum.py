from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.features.directional_momentum import (
    classify_bias,
    equity_momentum_history,
    futures_momentum_history,
    normalized_futures_change,
)


HORIZONS = [
    {"id": "one_month", "label": "1m", "days": 21, "weight": 0.10},
    {"id": "three_months", "label": "3m", "days": 63, "weight": 0.20},
    {"id": "six_months", "label": "6m", "days": 126, "weight": 0.25},
    {"id": "ten_months", "label": "10m", "days": 210, "weight": 0.25},
    {"id": "twelve_months", "label": "12m", "days": 252, "weight": 0.20},
]


def test_normalized_futures_change_survives_nonpositive_adjusted_prices() -> None:
    index = pd.bdate_range("2020-01-01", periods=400)
    price = pd.Series(np.linspace(-30.0, 90.0, len(index)), index=index)

    change = normalized_futures_change(price)
    history = futures_momentum_history(price, HORIZONS)

    assert np.isfinite(change.dropna()).all()
    assert np.isfinite(history["raw_momentum"].dropna()).all()
    assert set(history.index.get_level_values("horizon_id")) == {item["id"] for item in HORIZONS}


def test_equity_history_includes_absolute_and_relative_components() -> None:
    index = pd.bdate_range("2022-01-03", periods=400)
    benchmark = pd.Series(100.0 * np.exp(np.arange(len(index)) * 0.0005), index=index)
    sector = pd.Series(100.0 * np.exp(np.arange(len(index)) * 0.0010), index=index)

    history = equity_momentum_history(sector, benchmark, HORIZONS)
    latest = history.xs(index[-1], level="date")

    assert (latest["raw_momentum"] > 0).all()
    assert (latest["relative_raw_momentum"] > 0).all()
    assert latest["relative_vol_adjusted_momentum"].notna().all()


def test_bias_exposes_alignment_and_downgrades_disagreement() -> None:
    rows = pd.DataFrame(
        {
            "raw_momentum": [0.1, 0.1, 0.1, 0.1, -0.1],
            "vol_adjusted_momentum": [1.5, 1.2, 1.1, 1.0, -0.2],
            "weight": [0.1, 0.2, 0.25, 0.25, 0.2],
        }
    )
    config = {
        "positive_threshold": 0.25,
        "strong_threshold": 1.0,
        "minimum_components": 3,
        "strong_alignment": 4,
        "directional_alignment": 3,
    }

    result = classify_bias(rows, config)
    assert result["id"] == "positive"
    assert result["alignment"] == "4/5 positive"
    assert result["available_components"] == 5

    rows["raw_momentum"] *= -1
    disagreement = classify_bias(rows, config)
    assert disagreement["id"] == "mixed"
