import numpy as np
import pandas as pd
import pytest

from world_model.data.cleveland_nowcast import parse_daily, release_nowcasts
from world_model.data.kalshi import candle_price, reference_month, strike_of
from world_model.states.release_benchmarks import (
    accuracy,
    build_comparison,
    edge_over,
    kalshi_expectations,
    ladder_distribution,
)


def test_ladder_distribution_shifts_strikes_by_half_a_rounding_step():
    implied = ladder_distribution([0.1, 0.2, 0.3, 0.4], [1.0, 0.8, 0.3, 0.0], rounding=0.1)
    assert implied["mean"] == pytest.approx(0.31)
    assert implied["median"] == pytest.approx(0.31)
    assert implied["tail_mass"] == pytest.approx(0.0)


def test_ladder_distribution_repairs_prices_that_rise_with_the_strike():
    implied = ladder_distribution([0.1, 0.2, 0.3], [0.9, 0.95, 0.2], rounding=0.1)
    clean = ladder_distribution([0.1, 0.2, 0.3], [0.9, 0.9, 0.2], rounding=0.1)
    assert implied["mean"] == pytest.approx(clean["mean"])


def test_kalshi_tickers_and_strikes():
    assert strike_of({"ticker": "CPI-21DEC-T0.8"}) == 0.8
    assert strike_of({"ticker": "CPI-22JUN-TN0.1"}) == -0.1
    assert strike_of({"ticker": "KXCPI-26MAY-T-0.1", "strike_type": "greater", "floor_strike": -0.1}) == -0.1
    assert strike_of({"ticker": "X-26MAY-B0.3", "strike_type": "between", "floor_strike": 0.3}) is None
    assert reference_month("KXPAYROLLS-24DEC") == pd.Timestamp("2024-12-01")
    assert reference_month("GDP-21DEC31") == pd.Timestamp("2021-12-01")


def test_candle_price_prefers_a_tight_mid_then_the_last_trade():
    tight = {"yes_bid": {"close": "0.40"}, "yes_ask": {"close": "0.44"}, "price": {"close": "0.10"}}
    wide = {"yes_bid": {"close": "0.10"}, "yes_ask": {"close": "0.60"}, "price": {"close": None, "previous": "0.35"}}
    assert candle_price(tight) == pytest.approx(0.42)
    assert candle_price(wide) == pytest.approx(0.35)
    live = {"yes_bid": {"close_dollars": "0.08"}, "yes_ask": {"close_dollars": "0.11"}, "price": {"close_dollars": "0.10"}}
    assert candle_price(live) == pytest.approx(0.095)


def test_kalshi_expectations_keep_the_busier_duplicate_event():
    ladders = pd.DataFrame(
        {
            "series": ["KXCPI"] * 4,
            "event_ticker": ["CPI-23JAN", "CPI-23JAN", "KXCPI-23JAN", "KXCPI-23JAN"],
            "reference_month": [pd.Timestamp("2023-01-01")] * 4,
            "strike": [0.2, 0.3, 0.2, 0.3],
            "price_at_close": [0.9, 0.1, 0.5, 0.1],
            "volume": [10.0, 10.0, 500.0, 500.0],
            "close_time": [pd.Timestamp("2023-02-14 13:25", tz="UTC")] * 4,
        }
    )
    table = kalshi_expectations(ladders)
    assert len(table) == 1
    assert table["event_ticker"].iloc[0] == "KXCPI-23JAN"


def _chart(subcaption, labels, nowcasts, actual_index, actual):
    categories = [{"label": label} for label in labels] + [{"label": "CPI Oct", "vline": "true"}]
    actual_data = [{"value": ""} for _ in labels]
    actual_data[actual_index] = {"value": str(actual)}
    return {
        "chart": {"subcaption": subcaption},
        "categories": [{"category": categories}],
        "dataset": [
            {"seriesname": "CPI Inflation", "data": [{"value": "" if v is None else str(v)} for v in nowcasts]},
            {"seriesname": "Actual CPI Inflation", "data": actual_data},
        ],
    }


def test_cleveland_release_nowcast_is_the_last_value_before_the_release():
    chart = _chart("2026-10", ["11/06", "11/09", "11/10"], [0.20, 0.24, None], 2, 0.35)
    table = release_nowcasts(parse_daily([chart]))
    row = table.iloc[0]
    assert row["nowcast"] == pytest.approx(0.24)
    assert row["release_date"] == pd.Timestamp("2026-11-10")
    assert row["cleveland_actual"] == pytest.approx(0.35)


def test_cleveland_dates_roll_into_the_next_year_for_december():
    chart = _chart("2025-12", ["12/30", "01/12", "01/13"], [0.2, 0.3, None], 2, 0.31)
    table = release_nowcasts(parse_daily([chart]))
    assert table["release_date"].iloc[0] == pd.Timestamp("2026-01-13")


def test_edge_over_finds_a_forecast_that_knows_the_surprise():
    rng = np.random.default_rng(0)
    market = rng.normal(0.3, 0.05, 60)
    surprise = rng.normal(0.0, 0.1, 60)
    table = pd.DataFrame({"actual": market + surprise, "kalshi": market, "ours": market + surprise * 0.8})
    useful = edge_over(table, "kalshi", "ours", 0.0)
    assert useful["slope"] == pytest.approx(1.25, rel=1e-6)
    assert useful["same_side_rate"] == 1.0
    noise = edge_over(table.assign(ours=market + rng.normal(0, 0.1, 60)), "kalshi", "ours", 0.0)
    assert abs(noise["t_stat"]) < 3


def test_accuracy_uses_only_releases_every_source_covers():
    ours = pd.DataFrame(
        {
            "target_id": ["core_cpi_mom"] * 3,
            "reference_month": pd.to_datetime(["2026-07-01", "2026-08-01", "2026-09-01"]),
            "release_date": pd.to_datetime(["2026-08-12", "2026-09-11", "2026-10-14"]),
            "actual": [0.3, 0.2, 0.4],
            "ours": [0.25, 0.25, 0.3],
            "naive": [0.3, 0.3, 0.3],
        }
    )
    kalshi = pd.DataFrame({"target_id": ["core_cpi_mom"], "reference_month": [pd.Timestamp("2026-09-01")], "kalshi": [0.35]})
    table = build_comparison(ours, kalshi, pd.DataFrame())
    result = accuracy(table, ["ours", "kalshi"])
    assert result["releases"] == 1
    assert result["ours"]["mae"] == pytest.approx(0.1)
    assert result["kalshi"]["mae"] == pytest.approx(0.05)


def test_ladder_range_is_flagged_open_when_it_runs_past_the_last_strike():
    implied = ladder_distribution([0.0, 0.1, 0.2], [0.95, 0.5, 0.3], rounding=0.1)
    assert implied["low_open"] is False
    assert implied["high_open"] is True
    assert implied["high"] == pytest.approx(0.25)


def test_our_forecasts_score_bottom_up_targets_on_the_bottom_up_nowcast():
    from world_model.states.release_benchmarks import our_forecasts
    months = pd.date_range("2026-01-01", periods=3, freq="MS")
    forecasts = pd.DataFrame({
        "target_id": ["core_cpi_mom"] * 3, "variant_id": ["pipeline"] * 3, "observation_date": months,
        "release_date": months + pd.DateOffset(days=40), "actual": [0.2, 0.3, 0.25],
        "forecast": [0.5, 0.5, 0.5], "baseline_forecast": [0.1, 0.1, 0.1],
    })
    bottom_up = {"core_cpi_mom": pd.Series([0.21, 0.29], index=months[:2])}
    scored = our_forecasts(forecasts, None, bottom_up)
    assert list(scored["ours"]) == [0.21, 0.29]
    assert list(scored["naive"]) == [0.1, 0.1]
