from __future__ import annotations

import pandas as pd

from world_model.data.benchmarks import chain_price_total_return
from world_model.data.norgate import read_norgate_index_series
from world_model.features.equity_risk_overlay import compute_equity_risk_overlay


def _macro_series(dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {
        "UNRATE": pd.Series([4.0] * 15 + [5.0], index=dates),
        "INDPRO": pd.Series([100.0] * 13 + [99.0, 98.0, 97.0], index=dates),
        "AWHMAN": pd.Series([40.0] * 13 + [39.9, 39.8, 39.7], index=dates),
        "DGORDER": pd.Series([200.0] * 13 + [195.0, 190.0, 185.0], index=dates),
        "CPIAUCSL": pd.Series([100.0] * 16, index=dates),
    }


def test_overlay_identifies_high_conviction_defensive_state() -> None:
    market_dates = pd.date_range("2025-01-31", periods=16, freq="ME")
    price = pd.Series([100.0 + index for index in range(15)] + [80.0], index=market_dates)
    history = chain_price_total_return(price)
    series = _macro_series(pd.date_range("2025-01-01", periods=16, freq="MS"))

    result = compute_equity_risk_overlay(series, history, "2026-04-30", {"moving_average_months": 10})

    assert result["state_id"] == "high_conviction"
    assert result["data_warnings"] == []
    inputs = {item["id"]: item for item in result["inputs"]}
    assert inputs["market_trend"]["status"] == "warning"
    assert inputs["unemployment_trend"]["status"] == "warning"
    assert inputs["industrial_production"]["status"] == "warning"
    assert inputs["real_durable_orders"]["status"] == "warning"


def test_overlay_flags_stale_market_history() -> None:
    market_dates = pd.date_range("2025-01-31", periods=16, freq="ME")
    price = pd.Series([100.0 + index for index in range(16)], index=market_dates)
    history = chain_price_total_return(price)
    series = _macro_series(pd.date_range("2025-01-01", periods=16, freq="MS"))

    result = compute_equity_risk_overlay(
        series, history, "2026-05-20", {"moving_average_months": 10, "max_market_lag_days": 5}
    )

    assert len(result["data_warnings"]) == 1
    assert "stale" in result["data_warnings"][0]


def test_norgate_index_reader_uses_catalog_dataset(tmp_path) -> None:
    norgate = tmp_path / "norgate"
    metadata = norgate / "metadata"
    index_dir = norgate / "indices" / "us" / "adjustment=totalreturn"
    metadata.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    dates = pd.date_range("2026-01-02", periods=3, freq="B")
    pd.DataFrame({"date": dates, "close": [10.0, 11.0, 12.0]}).to_parquet(
        index_dir / "assetid=1_symbol=SPX.parquet"
    )
    pd.DataFrame(
        [
            {
                "symbol": "$SPX",
                "dataset": "us_indices",
                "price_adjustment": "TOTALRETURN",
                "last_date": dates[-1],
                "row_count": 3,
                "relative_path": "indices/us/adjustment=totalreturn/assetid=1_symbol=SPX.parquet",
            }
        ]
    ).to_parquet(metadata / "price_file_catalog.parquet")

    series = read_norgate_index_series(metadata, "$SPX")

    assert series.name == "$SPX"
    assert series.tolist() == [10.0, 11.0, 12.0]
