import numpy as np
import pandas as pd
import pytest

from world_model.states.energy_cpi import (
    _seasonally_adjust,
    known,
    monthly_average,
    monthly_changes,
    nowcast,
    release_events,
)


def _vintage(series_id, values, release_lag_days=14):
    """A vintage table where each monthly value is first published ``release_lag_days`` after its month ends."""
    rows = []
    for date, value in values.items():
        released = pd.Timestamp(date) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=release_lag_days)
        rows.append({"series_id": series_id, "observation_date": pd.Timestamp(date), "value": float(value),
                     "realtime_start": released, "realtime_end": pd.Timestamp("2262-01-01")})
    return pd.DataFrame(rows)


def test_known_ignores_values_published_after_the_cutoff():
    table = _vintage("X", {"2026-01-01": 100.0, "2026-02-01": 101.0})
    seen = known(table, pd.Timestamp("2026-03-01"))
    assert list(seen.index) == [pd.Timestamp("2026-01-01")]


def test_monthly_changes_skip_gaps():
    index = pd.Series([100.0, 101.0, 103.02], index=pd.to_datetime(["2026-01-01", "2026-02-01", "2026-04-01"]))
    changes = monthly_changes(index)
    assert changes[pd.Timestamp("2026-02-01")] == pytest.approx(1.0)
    assert np.isnan(changes[pd.Timestamp("2026-04-01")])


def test_monthly_average_holds_the_last_weekly_price_to_month_end():
    weekly = pd.Series([4.0, 4.0, 3.0, 3.0, 2.0], index=pd.to_datetime(["2026-10-05", "2026-10-12", "2026-10-19", "2026-10-26", "2026-11-02"]))
    averages = monthly_average(weekly, carry_to=pd.Timestamp("2026-11-01"))
    assert averages[pd.Timestamp("2026-10-01")] == pytest.approx(3.5)
    assert averages[pd.Timestamp("2026-11-01")] == pytest.approx(2.0)


def test_seasonal_adjustment_uses_last_years_step_between_the_same_months():
    months = pd.date_range("2025-01-01", "2026-02-01", freq="MS")
    unadjusted = pd.Series(100.0, index=months)
    adjusted = pd.Series(100.0, index=months)
    unadjusted[pd.Timestamp("2025-03-01")] = 102.0  # last March ran 2% above trend before adjustment
    change = _seasonally_adjust(2.0, unadjusted, adjusted, pd.Timestamp("2026-03-01"))
    assert change == pytest.approx(0.0, abs=1e-9)


def test_release_events_keep_only_monthly_first_prints():
    table = _vintage("CPIAUCSL", {"2026-01-01": 1.0, "2026-02-01": 2.0})
    table = pd.concat([table, table.assign(realtime_start=pd.Timestamp("2026-06-01"))])  # later revisions
    events = release_events(table, "2026-01-01")
    assert list(events["release_date"]) == [pd.Timestamp("2026-02-14"), pd.Timestamp("2026-03-14")]


def test_nowcast_passes_a_retail_gasoline_drop_through_to_headline():
    months = pd.date_range("2018-01-01", "2026-10-01", freq="MS")
    rng = np.random.default_rng(1)
    retail = 3.0 * np.exp(np.cumsum(rng.normal(0, 0.04, len(months))))
    gasoline = 100.0 * retail / retail[0]
    flat = 100.0 * np.cumprod(np.full(len(months), 1.002))
    energy = 0.5 * gasoline + 0.5 * flat
    headline = 0.9 * flat + 0.1 * energy
    tables = {
        "gasoline": _vintage("G", dict(zip(months, gasoline))),
        "gasoline_nsa": _vintage("GN", dict(zip(months, gasoline))),
        "electricity": _vintage("E", dict(zip(months, flat))),
        "utility_gas": _vintage("U", dict(zip(months, flat))),
        "fuel_oil": _vintage("F", dict(zip(months, flat))),
        "energy": _vintage("EN", dict(zip(months, energy))),
        "food": _vintage("FD", dict(zip(months, flat))),
        "core": _vintage("C", dict(zip(months, flat))),
        "headline": _vintage("H", dict(zip(months, headline))),
        "retail_gasoline": _vintage("R", {m + pd.Timedelta(days=d): r for m, r in zip(months, retail) for d in (0, 7, 14, 21)}, -20),
        "retail_diesel": _vintage("D", dict(zip(months, retail)), -20),
        "henry_hub": _vintage("HH", dict(zip(months, flat)), -20),
    }
    # November retail gasoline falls 10%.
    november = pd.Timestamp("2026-11-01")
    tables["retail_gasoline"] = pd.concat([tables["retail_gasoline"], _vintage("R", {november + pd.Timedelta(days=2): retail[-1] * 0.9}, -25)])
    result = nowcast(tables, november, pd.Timestamp("2026-11-12"), core_forecast=0.2)
    assert result["gasoline_nsa"] == pytest.approx(-10.0, abs=0.5)
    assert result["energy"] < -3.0
    assert result["headline"] < 0.2


def _results(n=40):
    months = pd.date_range("2022-01-01", periods=n, freq="MS")
    actual = np.linspace(0.1, 0.5, n)
    return pd.DataFrame({
        "month": months,
        "release_date": months + pd.DateOffset(days=44),
        "headline_actual": actual,
        "headline_forecast": actual + np.tile([0.05, -0.05], n // 2),
    })


def test_validation_requires_beating_the_benchmark_in_both_halves():
    from world_model.states.energy_cpi import validation
    results = _results()
    good = pd.Series(results["headline_actual"].to_numpy() + 0.2, index=results["month"])
    assert validation(results, good)["validated"]
    better_early = good.copy()
    better_early.iloc[:20] = results["headline_actual"].to_numpy()[:20]
    assert not validation(results, better_early)["validated"]


def test_site_row_replaces_the_headline_forecast_and_its_ranges():
    from world_model.states.energy_cpi import site_row
    live = {"month": pd.Timestamp("2026-11-01"), "headline": 0.14, "previous": {"headline": 0.35}, "gasoline": -3.4, "energy": -1.5}
    row = site_row({"target_id": "headline_cpi_mom", "direction_hit_rate": 0.7}, live, _results(), {"validated": True, "observations": 92})
    assert row["variant_id"] == "bottom_up" and row["forecast"] == 0.14 and row["validated"]
    assert row["direction_hit_rate"] is None
    assert row["range_80"][0] < 0.14 < row["range_80"][1]
    assert row["reference_month"] == "2026-11"
