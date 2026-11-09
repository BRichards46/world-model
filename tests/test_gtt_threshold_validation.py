from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.gtt_threshold_validation import (
    Threshold,
    build_filters,
    build_point_in_time_values,
    combine_or,
    count_at_least,
    monthly_timing_backtest,
)


def _vintage(series_id: str, rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": series_id,
            "observation_date": pd.to_datetime([row[0] for row in rows]),
            "value": [row[1] for row in rows],
            "realtime_start": pd.to_datetime([row[2] for row in rows]),
            "realtime_end": pd.Timestamp("2099-12-31"),
        }
    )


def test_values_only_use_releases_known_at_each_date() -> None:
    months = pd.date_range("2020-01-01", periods=13, freq="MS")
    rows = [(f"{month:%Y-%m-%d}", 100.0, "2021-01-15") for month in months[:-1]]
    rows.append((f"{months[-1]:%Y-%m-%d}", 102.0, "2021-02-15"))
    rows.append((f"{months[-1]:%Y-%m-%d}", 99.0, "2021-03-15"))  # later revision
    metrics = {"indpro_yoy": (("INDPRO",), lambda s: (s["INDPRO"].iloc[-1] / s["INDPRO"].iloc[-13] - 1.0) * 100.0 if len(s["INDPRO"]) >= 13 else float("nan"))}
    values = build_point_in_time_values("unused", metrics, tables={"INDPRO": _vintage("INDPRO", rows)})
    assert np.isnan(values.loc["2021-01-15", "indpro_yoy"])
    assert np.isclose(values.loc["2021-02-15", "indpro_yoy"], 2.0)
    assert np.isclose(values.loc["2021-03-15", "indpro_yoy"], -1.0)


def test_threshold_flags_and_unknowns() -> None:
    values = pd.DataFrame({"x": [-1.0, 0.0, 0.5, np.nan]})
    assert Threshold("x", "<=", 0.0).flag(values).tolist()[:3] == [1.0, 1.0, 0.0]
    assert np.isnan(Threshold("x", "<=", 0.0).flag(values).iloc[3])
    assert Threshold("x", ">", 0.0).flag(values).tolist()[:3] == [0.0, 0.0, 1.0]


def test_or_and_count_need_every_input_known() -> None:
    a = pd.Series([1.0, 0.0, 0.0, np.nan])
    b = pd.Series([0.0, 0.0, 1.0, 1.0])
    c = pd.Series([1.0, 0.0, 1.0, 1.0])
    assert combine_or([a, b]).tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(combine_or([a, b]).iloc[3])
    assert count_at_least([a, b, c], 2).tolist()[:3] == [1.0, 0.0, 1.0]


def test_build_filters_names_singles_sweeps_and_combinations() -> None:
    values = pd.DataFrame({"g": [-1.0, 1.0, 2.0], "u": [0.2, -0.1, 0.0]})
    config = {
        "series": {
            "g": {"danger_operator": "<=", "zero_line": 0.0, "your_threshold": 1.0, "sweep": [2.0]},
            "u": {"danger_operator": ">", "zero_line": 0.0},
        },
        "or_combinations": {"g_or_u": ["g@zero_line", "u@zero_line"]},
        "counts": {"both": {"members": ["g@your_threshold", "u@zero_line"], "at_least": [2]}},
    }
    flags, catalog = build_filters(values, config)
    assert flags["g@your_threshold"].tolist() == [1.0, 1.0, 0.0]
    assert flags["g@sweep+2"].tolist() == [1.0, 1.0, 1.0]
    assert flags["g_or_u"].tolist() == [1.0, 0.0, 0.0]
    assert flags["both>=2"].tolist() == [1.0, 0.0, 0.0]
    assert set(catalog["family"]) == {"single", "sweep", "or", "count"}


def test_backtest_moves_to_cash_only_when_trend_and_macro_both_warn() -> None:
    days = pd.bdate_range("2000-01-03", "2003-12-31")
    returns = pd.Series(0.0004, index=days)
    returns.loc["2001-02-01":"2001-02-28"] = -0.01  # a bad February
    month_ends = returns.groupby(returns.index.to_period("M")).apply(lambda x: x.index[-1]).to_numpy()
    trend = pd.Series(0.0, index=month_ends)
    trend.loc[pd.Timestamp("2001-01-31")] = 1.0
    cash = pd.Series(0.0, index=days)
    warned = pd.Series(1.0, index=[pd.Timestamp("2000-01-01")])
    healthy = pd.Series(0.0, index=[pd.Timestamp("2000-01-01")])
    protected = monthly_timing_backtest(returns, cash, trend, warned, pd.Timestamp("2000-02-01"), cost_bps=0.0)
    ignored = monthly_timing_backtest(returns, cash, trend, healthy, pd.Timestamp("2000-02-01"), cost_bps=0.0)
    assert protected["time_in_cash"] > 0 and ignored["time_in_cash"] == 0
    assert protected["max_drawdown"] > ignored["max_drawdown"]  # less negative
