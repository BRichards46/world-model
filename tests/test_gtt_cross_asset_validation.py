from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.gtt_cross_asset_validation import build_logic_filters, cash_yield, synthetic_treasury_log_returns


def test_flat_yield_earns_its_coupon_and_rising_yield_loses_about_duration() -> None:
    days = pd.bdate_range("2020-01-01", periods=253)
    flat = synthetic_treasury_log_returns(pd.Series(4.0, index=days))
    annual = np.expm1(flat.sum())
    assert abs(annual - 0.04 * (days[-1] - days[0]).days / 365.0) < 0.002

    jump = pd.Series([4.0, 5.0], index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]))
    one_day = np.expm1(synthetic_treasury_log_returns(jump).iloc[0])
    assert -0.085 < one_day < -0.075  # a 10-year par bond's modified duration is about 8


def test_cash_prefers_primary_and_falls_back() -> None:
    primary = pd.Series([1.0], index=[pd.Timestamp("1990-01-02")])
    fallback = pd.Series([5.0, 6.0], index=pd.DatetimeIndex(["1970-01-02", "1990-01-02"]))
    combined = cash_yield(primary, fallback)
    assert combined.loc["1970-01-02"] == 5.0 and combined.loc["1990-01-02"] == 1.0


def test_logic_filters_combine_growth_and_inflation() -> None:
    values = pd.DataFrame(
        {
            "indpro_yoy": [1.0, -1.0, 1.0, np.nan],
            "unemployment_gap": [0.0, 0.0, 0.2, 0.0],
            "real_retail_yoy": [1.0, 1.0, 1.0, 1.0],
            "cpi_yoy": [3.0, 1.0, 5.0, 3.0],
        }
    )
    flags = build_logic_filters(values)
    assert flags["growth_weak"].tolist()[:3] == [0.0, 1.0, 1.0]
    assert flags["growth_weak_strict"].tolist()[:3] == [0.0, 0.0, 0.0]
    assert flags["growth_healthy"].tolist()[:3] == [1.0, 0.0, 0.0]
    assert np.isnan(flags["growth_healthy"].iloc[3])
    assert flags["growth_healthy_and_cpi_above_2_5"].tolist()[:3] == [1.0, 0.0, 0.0]
    assert flags["cpi_above_4"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert flags["cpi_below_2_5"].tolist() == [0.0, 1.0, 0.0, 0.0]
