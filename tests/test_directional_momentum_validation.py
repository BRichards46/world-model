from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from world_model.states.directional_momentum_validation import (
    calculate_excursion,
    summarize_excursion_value,
)


def _price_input(values: list[float], mode: str = "equity") -> dict[str, object]:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    price = pd.Series(values, index=index, dtype=float)
    volatility = pd.Series(0.01, index=index)
    result: dict[str, object] = {
        "price": price,
        "volatility": volatility,
        "mode": mode,
    }
    if mode == "futures":
        result["scale"] = pd.Series(100.0, index=index)
    return result


def test_excursion_is_oriented_to_the_signal_not_terminal_accuracy() -> None:
    prices = _price_input([100.0, 104.0, 98.0, 102.0])
    date = pd.Timestamp("2020-01-01")

    positive = calculate_excursion(prices, date, direction=1, forward_days=3)
    negative = calculate_excursion(prices, date, direction=-1, forward_days=3)

    assert positive is not None
    assert negative is not None
    assert positive["mfe"] == pytest.approx(0.04)
    assert positive["mae"] == pytest.approx(0.02)
    assert positive["terminal"] == pytest.approx(0.02)
    assert negative["mfe"] == pytest.approx(0.02)
    assert negative["mae"] == pytest.approx(0.04)
    assert negative["terminal"] == pytest.approx(-0.02)


def test_futures_excursion_uses_price_scale_instead_of_percentage_return() -> None:
    prices = _price_input([-20.0, -10.0, -30.0], mode="futures")

    result = calculate_excursion(prices, pd.Timestamp("2020-01-01"), direction=1, forward_days=2)

    assert result is not None
    assert result["mfe"] == 0.10
    assert result["mae"] == 0.10


def test_summary_flags_only_joint_mfe_and_mae_improvement() -> None:
    dates = pd.bdate_range("2020-01-01", periods=4)
    common = {
        "asset_id": "ES",
        "asset_class": "future",
        "lens": "absolute",
        "signal_type": "composite",
        "lookback": "composite",
        "direction": 1,
        "forward_id": "one_month",
        "forward_label": "1m",
        "forward_days": 21,
    }
    events = pd.DataFrame(
        [
            {
                **common,
                "signal_date": date,
                "mfe": 2.0,
                "mae": 0.5,
                "edge": 1.5,
                "mfe_risk": 2.0,
                "mae_risk": 0.5,
                "edge_risk": 1.5,
                "terminal": 1.0,
            }
            for date in dates
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "asset_id": "ES",
                "asset_class": "future",
                "lens": "absolute",
                "signal_date": date,
                "direction": 1,
                "forward_id": "one_month",
                "forward_label": "1m",
                "forward_days": 21,
                "mfe": 1.0,
                "mae": 1.0,
                "edge": 0.0,
                "mfe_risk": 1.0,
                "mae_risk": 1.0,
                "edge_risk": 0.0,
                "terminal": -0.5 if index % 2 else 0.5,
            }
            for index, date in enumerate(dates)
        ]
    )
    config = {
        "minimum_sample_size": 4,
        "periods": [{"id": "full", "label": "Full history", "start": None}],
    }

    summary = summarize_excursion_value(events, baselines, config)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["mfe_uplift_risk"] == 1.0
    assert row["mae_reduction_risk"] == 0.5
    assert row["edge_uplift_risk"] == 1.5
    assert bool(row["robust_excursion_improvement"])
    assert bool(row["stable_across_periods"])

