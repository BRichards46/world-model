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


def test_unemployment_warns_only_beyond_the_buffer() -> None:
    market_dates = pd.date_range("2025-01-31", periods=16, freq="ME")
    history = chain_price_total_return(pd.Series([100.0 + index for index in range(16)], index=market_dates))
    dates = pd.date_range("2025-01-01", periods=16, freq="MS")

    def unemployment_status(latest: float, config: dict) -> str:
        series = _macro_series(dates)
        series["UNRATE"] = pd.Series([4.0] * 15 + [latest], index=dates)
        result = compute_equity_risk_overlay(series, history, "2026-04-30", {"moving_average_months": 10, **config})
        return {item["id"]: item for item in result["inputs"]}["unemployment_trend"]["status"]

    # Trailing 12-month average includes the latest month, so a jump of x sits about 11x/12 above it.
    assert unemployment_status(4.1, {}) == "healthy"  # about +0.09 pp: inside the default 0.1 buffer
    assert unemployment_status(4.2, {}) == "warning"  # about +0.18 pp
    assert unemployment_status(4.1, {"unemployment_buffer_pp": 0.0}) == "warning"  # zero line


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


def test_triggers_give_the_levels_that_would_flip_each_signal() -> None:
    from world_model.features.equity_risk_overlay import (
        industrial_production_trigger,
        market_trigger,
        unemployment_trigger,
    )

    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    unemployment = pd.Series([4.0] * 12, index=months)
    trigger = unemployment_trigger(unemployment, 0.1)
    assert trigger["month"] == "2026-01"
    # 11 readings of 4.0 stay in the window: X - (44 + X) / 12 > 0.1  =>  X > 4.109...
    assert abs(trigger["threshold_rate"] - (0.1 * 12 + 44.0) / 11) < 1e-12

    production = pd.Series([100.0] * 11 + [104.0], index=months)
    ip = industrial_production_trigger(production)
    assert ip["month"] == "2026-01" and abs(ip["threshold_change_pct"] - (100.0 / 104.0 - 1.0) * 100.0) < 1e-9

    # Ten month-ends; the last is the month in progress, so the other nine set the line.
    closes = pd.Series([100.0] * 9 + [110.0], index=pd.date_range("2025-01-31", periods=10, freq="ME"))
    market = market_trigger(closes, 10, pd.Timestamp("2025-10-15"))
    assert market["month"] == "2025-10" and abs(market["threshold_change_pct"] - (100.0 / 110.0 - 1.0) * 100.0) < 1e-9


def test_overlay_attaches_first_print_odds_when_vintages_are_given() -> None:
    market_dates = pd.date_range("2025-01-31", periods=16, freq="ME")
    history = chain_price_total_return(pd.Series([100.0 + index for index in range(16)], index=market_dates))
    dates = pd.date_range("2025-01-01", periods=16, freq="MS")
    series = _macro_series(dates)
    series["UNRATE"] = pd.Series([4.0] * 16, index=dates)
    releases = [date + pd.DateOffset(months=1, days=4) for date in dates]
    unrate = pd.DataFrame(
        {
            "observation_date": dates,
            "value": [4.0] * 15 + [4.3],
            "realtime_start": releases,
            "realtime_end": pd.Timestamp("2099-12-31"),
        }
    )
    result = compute_equity_risk_overlay(
        series, history, "2026-04-30", {"moving_average_months": 10, "trigger_odds_start": "2025-01-01"},
        vintages={"UNRATE": unrate},
    )
    trigger = result["triggers"]["unemployment_trend"]
    assert trigger["published_threshold"] == 4.2  # 4.0 flat: X > (1.2 + 44) / 11 = 4.109
    odds = trigger["odds"]
    # Only the 5 months with a full 12-month history before them can be classified by starting state;
    # one of them (the last, 4.0 -> 4.3) jumped at least the 0.2 pp needed.
    assert odds["hits"] == 1 and odds["months"] == 5 and abs(odds["required_jump"] - 0.2) < 1e-9
