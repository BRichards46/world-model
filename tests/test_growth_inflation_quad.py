from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.features.growth_inflation_quad import (
    QuadInputs,
    QuadSettings,
    build_quad_outlook,
    classify_quad,
    cpi_quarterly_yoy,
    gdp_yoy,
    inflation_hurdle,
)
from world_model.features.monthly_windows import (
    fill_missing_months,
    pct_change_months,
    trailing_average,
)


def test_classify_quad_follows_growth_and_inflation_directions() -> None:
    assert classify_quad(0.1, -0.1) == 1
    assert classify_quad(0.1, 0.1) == 2
    assert classify_quad(-0.1, 0.1) == 3
    assert classify_quad(-0.1, -0.1) == 4
    assert classify_quad(None, 0.1) is None


def test_missing_month_does_not_stretch_yoy_to_thirteen_months() -> None:
    dates = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    values = pd.Series(np.arange(len(dates), dtype=float) + 100.0, index=dates).drop(pd.Timestamp("2024-12-01"))
    # The comparison month is unpublished, so no YoY is reported rather than a 13-month change.
    assert pct_change_months(values, 12) is None

    filled, missing = fill_missing_months(values)
    assert missing == [pd.Timestamp("2024-12-01")]
    interpolated_dec = float(filled.loc["2024-12-01"])  # time-weighted log interpolation
    assert 110.0 < interpolated_dec < 112.0
    assert np.isclose(pct_change_months(filled, 12), 100.0 * (123.0 / interpolated_dec - 1.0))


def test_trailing_average_uses_calendar_window_and_tolerates_one_gap() -> None:
    dates = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    values = pd.Series(1.0, index=dates)
    values.iloc[-1] = 13.0
    gapped = values.drop(pd.Timestamp("2025-06-01"))
    assert np.isclose(trailing_average(gapped, 12), (10 * 1.0 + 13.0) / 11)
    assert trailing_average(gapped.drop(pd.Timestamp("2025-07-01")), 12) is None


def _inputs(as_of: str, gdp_saar: list[float], cpi_mom: list[float]) -> QuadInputs:
    gdp_dates = pd.date_range("2020-01-01", periods=len(gdp_saar) + 1, freq="QS")
    gdp = [100.0]
    for rate in gdp_saar:
        gdp.append(gdp[-1] * (1.0 + rate / 100.0) ** 0.25)
    cpi_dates = pd.date_range("2020-01-01", periods=len(cpi_mom) + 1, freq="MS")
    cpi = [100.0]
    for rate in cpi_mom:
        cpi.append(cpi[-1] * (1.0 + rate / 100.0))
    cpi_series = pd.Series(cpi, index=cpi_dates)
    return QuadInputs(
        as_of=pd.Timestamp(as_of),
        gdp=pd.Series(gdp, index=gdp_dates),
        cpi=cpi_series,
        core_cpi=cpi_series,
        nowcast_saar=pd.Series(dtype=float),
    )


def test_growth_hurdle_is_the_comparison_quarter_growth() -> None:
    # Quarters 2020Q2..2022Q4 are reported; 2022Q1 (index 8) grew 0.5% annualized.
    saar = [2.0] * 7 + [0.5, 2.0, 2.0, 2.0]
    inputs = _inputs("2023-02-15", saar, [0.2] * 36)
    outlook = build_quad_outlook(inputs, QuadSettings(forward_quarters=1))
    projected = {row["quarter"]: row for row in outlook["quarters"] if row["status"] != "realized"}
    # 2023Q1 compares with 2022Q1, which grew 0.5% annualized; trend is ~2%, so growth accelerates.
    assert np.isclose(projected["2023Q1"]["growth_hurdle_saar"], 0.5, atol=1e-6)
    assert projected["2023Q1"]["growth_change"] > 0


def test_inflation_hurdle_holds_quarterly_average_flat() -> None:
    inputs = _inputs("2023-02-15", [2.0] * 12, [0.2] * 30 + [0.6, 0.6, 0.6, 0.2, 0.2, 0.2])
    quarter = pd.Timestamp("2023-01-01")
    hurdle = inflation_hurdle(inputs, quarter, 0.2)
    assert hurdle is not None
    from world_model.features.growth_inflation_quad import extend_monthly

    path = extend_monthly(inputs.cpi, pd.Timestamp("2023-03-01"), hurdle)
    averages = cpi_quarterly_yoy(path)
    assert np.isclose(averages.loc[quarter], averages.loc[pd.Timestamp("2022-10-01")], atol=1e-6)


def test_projection_reports_scenario_grid_and_realized_history() -> None:
    inputs = _inputs("2023-02-15", [2.0] * 12, [0.2] * 36)
    outlook = build_quad_outlook(inputs, QuadSettings(forward_quarters=2, history_quarters=4))
    statuses = [row["status"] for row in outlook["quarters"]]
    assert "realized" in statuses and "projected" in statuses
    projected = [row for row in outlook["quarters"] if row["status"] != "realized"]
    assert all(row["scenario_total"] == 9 for row in projected)
    assert len(outlook["cpi_comps"]) == 12
    assert gdp_yoy(inputs.gdp).dropna().size > 0
