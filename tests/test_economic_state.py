from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.economic import EconomicStateEngine


def test_economic_state_engine_builds_pit_state(tmp_path: Path) -> None:
    fred_root = tmp_path / "fred"
    vintage_root = fred_root / "vintage"
    vintage_root.mkdir(parents=True)
    _write_monthly_vintage(vintage_root / "GROWTH.parquet", "GROWTH", start=100.0, step=1.0)
    _write_monthly_vintage(vintage_root / "INFL.parquet", "INFL", start=200.0, step=2.0)

    config = {
        "history_frequency": "B",
        "scoring": {"window": 12, "min_periods": 6, "scale": 2.0},
        "components": {
            "growth": {
                "weight": 0.60,
                "composite_sign": 1.0,
                "indicators": [
                    {
                        "id": "growth_yoy",
                        "label": "Growth YoY",
                        "series_id": "GROWTH",
                        "transform": "yoy_pct",
                    }
                ],
            },
            "inflation": {
                "weight": 0.40,
                "composite_sign": -1.0,
                "indicators": [
                    {
                        "id": "inflation_yoy",
                        "label": "Inflation YoY",
                        "series_id": "INFL",
                        "transform": "yoy_pct",
                    }
                ],
            },
        },
    }

    engine = EconomicStateEngine({"fred": str(fred_root)}, config)
    state = engine.build_state_history()
    snapshot = engine.snapshot()

    assert "growth_score" in state
    assert "inflation_score" in state
    assert "composite_score" in state
    assert snapshot.component_coverage > 0.0
    assert set(snapshot.indicators["series_id"]) == {"GROWTH", "INFL"}


def _write_monthly_vintage(path: Path, series_id: str, start: float, step: float) -> None:
    observations = pd.date_range("2018-01-01", periods=42, freq="MS")
    rows = []
    for idx, observation_date in enumerate(observations):
        rows.append(
            {
                "series_id": series_id,
                "observation_date": observation_date,
                "value": start + idx * step,
                "realtime_start": observation_date + pd.DateOffset(months=1),
                "realtime_end": "9999-12-31",
            }
        )
    pd.DataFrame(rows).to_parquet(path)
