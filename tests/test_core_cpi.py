import numpy as np
import pandas as pd
import pytest

from world_model.states.core_cpi import (
    BLOCKS,
    ITEM_CODES,
    SERIES,
    Settings,
    choose_window,
    contributions,
    nowcast,
    site_row,
    weights_known,
)


def _vintage(values, release_lag_days=14):
    """A vintage table where each monthly value is first published ``release_lag_days`` after its month ends."""
    rows = []
    for date, value in values.items():
        released = pd.Timestamp(date) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=release_lag_days)
        rows.append({"observation_date": pd.Timestamp(date), "value": float(value), "realtime_start": released})
    return pd.DataFrame(rows)


SHARES = {
    "core": 100.0, "goods": 25.0, "services": 75.0,
    "oer": 33.0, "rent": 10.0, "lodging": 2.0, "medical": 8.0, "transport": 8.0,
    "new_vehicles": 5.0, "used_vehicles": 3.0, "apparel": 3.0,
}


def _weights(years=(2020, 2021, 2022, 2023, 2024, 2025)):
    return pd.DataFrame([SHARES for _ in years], index=list(years))


def test_weights_wait_for_the_january_report_that_publishes_them():
    months = pd.date_range("2024-01-01", "2026-01-01", freq="MS")
    core = _vintage(dict(zip(months, np.arange(len(months)) + 100.0)))
    weights = _weights()
    weights.loc[2025, "oer"] = 40.0
    # The January 2026 print (with the December 2025 weights) comes out 2026-02-14.
    before = weights_known(weights, core, pd.Timestamp("2026-02-13"))
    after = weights_known(weights, core, pd.Timestamp("2026-02-14"))
    assert before["oer"] == pytest.approx(0.33)
    assert after["oer"] == pytest.approx(0.40)


def test_residual_blocks_are_what_their_parent_leaves():
    index = pd.date_range("2026-01-01", periods=2, freq="MS")
    changes = {name: pd.Series(0.2, index=index) for name in ITEM_CODES}
    changes["goods"] = pd.Series(0.4, index=index)
    shares = {name: value / 100.0 for name, value in SHARES.items()}
    parts = contributions(changes, shares)
    assert parts["other_goods"].iloc[0] == pytest.approx(0.25 * 0.4 - (0.05 + 0.03 + 0.03) * 0.2)
    total = sum(parts[b].iloc[0] for b in BLOCKS)
    assert total == pytest.approx(0.25 * 0.4 + 0.75 * 0.2)


def test_window_choice_follows_the_kind_of_series():
    index = pd.date_range("2018-01-01", periods=96, freq="MS")
    rng = np.random.default_rng(3)
    noisy = pd.Series(0.3 + rng.normal(0, 0.5, len(index)), index=index)
    drifting = pd.Series(np.cumsum(rng.normal(0, 0.1, len(index))), index=index)
    assert choose_window(noisy, index[-1]) == 12
    assert choose_window(drifting, index[-1]) == 1


def test_nowcast_adds_blocks_back_up_to_core():
    months = pd.date_range("2018-01-01", "2026-10-01", freq="MS")
    level = 100.0 * np.cumprod(np.full(len(months), 1.002))
    tables = {name: _vintage(dict(zip(months, level))) for name in SERIES}
    result = nowcast(tables, _weights(), pd.Timestamp("2026-11-01"), pd.Timestamp("2026-12-09"), Settings())
    assert result["core"] == pytest.approx(0.2, abs=1e-6)
    assert sum(result["groups"].values()) == pytest.approx(result["core"])
    assert result["blocks"]["oer"] == pytest.approx(0.2, abs=1e-6)


def test_nowcast_ignores_prints_after_the_cutoff():
    months = pd.date_range("2018-01-01", "2026-10-01", freq="MS")
    level = 100.0 * np.cumprod(np.full(len(months), 1.002))
    jumped = level.copy()
    jumped[-1] *= 1.05  # October jumps 5%, published 2026-11-14
    tables = {name: _vintage(dict(zip(months, jumped))) for name in SERIES}
    early = nowcast(tables, _weights(), pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-13"))
    assert early["core"] == pytest.approx(0.2, abs=1e-6)


def test_site_row_replaces_the_core_forecast_and_its_ranges():
    months = pd.date_range("2022-01-01", periods=40, freq="MS")
    actual = np.linspace(0.1, 0.5, 40)
    results = pd.DataFrame({
        "month": months,
        "release_date": months + pd.DateOffset(days=44),
        "core_actual": actual,
        "core_forecast": actual + np.tile([0.05, -0.05], 20),
    })
    live = {"month": pd.Timestamp("2026-11-01"), "core": 0.24, "previous": {"core": 0.3},
            "groups": {"shelter": 0.12, "core_goods": 0.02, "services_ex_shelter": 0.1}}
    row = site_row({"target_id": "core_cpi_mom", "direction_hit_rate": 0.6}, live, results, {"validated": True, "observations": 81})
    assert row["variant_id"] == "bottom_up" and row["forecast"] == 0.24 and row["validated"]
    assert row["direction_hit_rate"] is None
    assert row["range_80"][0] < 0.24 < row["range_80"][1]
    assert row["blocks"]["shelter"] == 0.12
