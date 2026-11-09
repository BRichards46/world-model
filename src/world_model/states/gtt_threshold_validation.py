"""GTT macro filters at alternative thresholds, OR combinations and trigger counts.

Every input is replayed release by release from the FRED vintage store, so a
value is known only from its ``realtime_start``. Continuous point-in-time
values are built once; each candidate filter is then a threshold applied to
those values (danger = 1, healthy = 0, unknown = NaN).

Filters are scored two ways:

* the GTT risk test (``run_gtt_risk_validation``): when the S&P 500 is below
  trend, do months with macro danger see more future volatility and drawdown
  than months where the macro filter stayed healthy?
* a simple monthly timing backtest: hold the S&P 500 unless price trend and
  macro danger both warn, in which case hold cash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table
from world_model.features.monthly_windows import fill_missing_months, pct_change_months, trailing_average


# --- point-in-time values ------------------------------------------------------------------


def _yoy(series: pd.Series) -> float:
    value = pct_change_months(series, 12)
    return float("nan") if value is None else float(value)


def _gap_vs_average(series: pd.Series) -> float:
    average = trailing_average(series, 12)
    clean = series.dropna()
    return float("nan") if average is None or clean.empty else float(clean.iloc[-1] - average)


def _ratio_yoy(numerator: pd.Series, denominator: pd.Series) -> float:
    common = numerator.index.intersection(denominator.index)
    if common.empty:
        return float("nan")
    return _yoy((numerator.reindex(common) / denominator.reindex(common)).dropna())


def _deflated_yoy(nominal: pd.Series, cpi: pd.Series) -> float:
    return _ratio_yoy(nominal, cpi)


# Each metric: (input series, function of the latest known series by id).
METRICS: dict[str, tuple[tuple[str, ...], Callable[[Mapping[str, pd.Series]], float]]] = {
    "indpro_yoy": (("INDPRO",), lambda s: _yoy(s["INDPRO"])),
    "payems_yoy": (("PAYEMS",), lambda s: _yoy(s["PAYEMS"])),
    "unemployment_gap": (("UNRATE",), lambda s: _gap_vs_average(s["UNRATE"])),
    "housing_lf_yoy": (("HOUST", "CLF16OV"), lambda s: _ratio_yoy(s["HOUST"], s["CLF16OV"])),
    "cpi_yoy": (("CPIAUCSL",), lambda s: _yoy(s["CPIAUCSL"])),
    "core_cpi_yoy": (("CPILFESL",), lambda s: _yoy(s["CPILFESL"])),
    "retail_yoy": (("RSAFS",), lambda s: _yoy(s["RSAFS"])),
    "real_retail_yoy": (("RSAFS", "CPIAUCSL"), lambda s: _deflated_yoy(s["RSAFS"], s["CPIAUCSL"])),
    "awhman_gap": (("AWHMAN",), lambda s: _gap_vs_average(s["AWHMAN"])),
    "real_dgorder_yoy": (("DGORDER", "CPIAUCSL"), lambda s: _deflated_yoy(s["DGORDER"], s["CPIAUCSL"])),
}


def build_point_in_time_values(
    fred_root: str | Path,
    metrics: Mapping[str, tuple[tuple[str, ...], Callable[[Mapping[str, pd.Series]], float]]] = METRICS,
    tables: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Continuous metric values as known on each release date (forward-filled between releases)."""
    series_ids = sorted({series_id for inputs, _ in metrics.values() for series_id in inputs})
    tables = dict(tables) if tables is not None else {series_id: load_vintage_table(fred_root, series_id) for series_id in series_ids}
    by_release = {
        series_id: {date: group for date, group in tables[series_id].groupby("realtime_start", sort=True)}
        for series_id in series_ids
    }
    releases = sorted(set().union(*(set(groups) for groups in by_release.values())))
    known: dict[str, dict[pd.Timestamp, float]] = {series_id: {} for series_id in series_ids}
    latest: dict[str, pd.Series] = {series_id: pd.Series(dtype=float) for series_id in series_ids}
    values = {name: float("nan") for name in metrics}
    rows = []
    for release in releases:
        updated = set()
        for series_id, groups in by_release.items():
            update = groups.get(release)
            if update is None:
                continue
            for record in update.itertuples(index=False):
                date = pd.Timestamp(record.observation_date)
                if pd.notna(record.value):
                    known[series_id][date] = float(record.value)
                else:
                    known[series_id].pop(date, None)
            series = pd.Series(known[series_id], dtype=float).sort_index()
            if series_id in {"CPIAUCSL"}:
                series, _ = fill_missing_months(series)  # deflator: interpolate unpublished months
            latest[series_id] = series
            updated.add(series_id)
        changed = False
        for name, (inputs, function) in metrics.items():
            if updated.intersection(inputs):
                value = function(latest)
                if not (pd.isna(value) and pd.isna(values[name])) and value != values[name]:
                    values[name] = value
                    changed = True
        if changed or not rows:
            rows.append({"realtime_start": pd.Timestamp(release), **values})
    return pd.DataFrame.from_records(rows).set_index("realtime_start").sort_index()


# --- filters ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Threshold:
    metric: str
    operator: str  # danger when value <op> level
    level: float

    def flag(self, values: pd.DataFrame) -> pd.Series:
        series = values[self.metric]
        if self.operator == "<=":
            danger = series.le(self.level)
        elif self.operator == ">=":
            danger = series.ge(self.level)
        elif self.operator == "<":
            danger = series.lt(self.level)
        elif self.operator == ">":
            danger = series.gt(self.level)
        else:
            raise ValueError(f"Unsupported operator {self.operator}")
        return danger.astype(float).where(series.notna())


def combine_or(flags: list[pd.Series]) -> pd.Series:
    """Danger if any input warns; unknown unless every input is known."""
    frame = pd.concat(flags, axis=1)
    known = frame.notna().all(axis=1)
    return frame.eq(1.0).any(axis=1).astype(float).where(known)


def count_at_least(flags: list[pd.Series], minimum: int) -> pd.Series:
    frame = pd.concat(flags, axis=1)
    known = frame.notna().all(axis=1)
    return frame.eq(1.0).sum(axis=1).ge(minimum).astype(float).where(known)


def build_filters(values: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Danger flags for every configured filter, plus a catalog describing each one."""
    flags: dict[str, pd.Series] = {}
    catalog: list[dict[str, Any]] = []
    definitions: dict[str, Threshold] = {}
    for metric, settings in config.get("series", {}).items():
        operator = str(settings["danger_operator"])
        levels = {"zero_line": settings.get("zero_line"), "your_threshold": settings.get("your_threshold")}
        for kind, level in levels.items():
            if level is None:
                continue
            name = f"{metric}@{kind}"
            definitions[name] = Threshold(metric, operator, float(level))
            flags[name] = definitions[name].flag(values)
            catalog.append({"filter": name, "family": "single", "metric": metric, "kind": kind, "level": float(level), "operator": operator})
        for level in settings.get("sweep", []):
            name = f"{metric}@sweep{float(level):+g}"
            flags[name] = Threshold(metric, operator, float(level)).flag(values)
            catalog.append({"filter": name, "family": "sweep", "metric": metric, "kind": "sweep", "level": float(level), "operator": operator})
    for name, members in config.get("or_combinations", {}).items():
        flags[name] = combine_or([flags[member] for member in members])
        catalog.append({"filter": name, "family": "or", "members": " OR ".join(members)})
    for name, settings in config.get("counts", {}).items():
        members = list(settings["members"])
        for minimum in settings.get("at_least", [1, 2, 3]):
            filter_name = f"{name}>={minimum}"
            flags[filter_name] = count_at_least([flags[member] for member in members], int(minimum))
            catalog.append({"filter": filter_name, "family": "count", "members": f"{minimum}+ of " + ", ".join(members)})
    return pd.DataFrame(flags), pd.DataFrame.from_records(catalog)


# --- timing backtest ----------------------------------------------------------------------


def monthly_timing_backtest(
    log_returns: pd.Series,
    cash_yield: pd.Series,
    trend_down: pd.Series,
    danger: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    cost_bps: float = 5.0,
) -> dict[str, float]:
    """Hold the index unless the month-end trend (and, if given, macro danger) warns; then hold cash.

    ``trend_down`` and ``danger`` are known at each month-end; the position applies to the next month.
    ``cash_yield`` is an annualized percentage (missing = 0).
    """
    monthly_index = log_returns.groupby(log_returns.index.to_period("M")).sum()
    month_end_dates = log_returns.groupby(log_returns.index.to_period("M")).apply(lambda x: x.index[-1])
    asset = np.exp(monthly_index) - 1.0
    cash = (cash_yield.groupby(cash_yield.index.to_period("M")).mean().reindex(asset.index).fillna(0.0) / 100.0) / 12.0
    trend = trend_down.reindex(month_end_dates.to_numpy(), method="ffill")
    trend.index = asset.index
    if danger is None:
        defensive = trend.eq(1.0)
        known = trend.notna()
    else:
        macro = danger.reindex(danger.index.union(month_end_dates.to_numpy())).sort_index().ffill().reindex(month_end_dates.to_numpy())
        macro.index = asset.index
        defensive = trend.eq(1.0) & macro.eq(1.0)
        known = trend.notna() & macro.notna()
    exposure = (~defensive).astype(float).where(known).shift(1)  # decided at prior month-end
    period_start = pd.Period(start, "M")
    period_end = pd.Period(end, "M") if end is not None else asset.index[-1]
    window = (asset.index >= period_start) & (asset.index <= period_end) & exposure.notna().to_numpy()
    exposure, asset, cash = exposure[window], asset[window], cash[window]
    if len(exposure) < 24:
        return {"months": int(len(exposure))}
    switches = exposure.diff().abs().fillna(0.0)
    returns = exposure * asset + (1.0 - exposure) * cash - switches * cost_bps / 10_000.0
    wealth = (1.0 + returns).cumprod()
    years = len(returns) / 12.0
    excess = returns - cash
    return {
        "months": int(len(returns)),
        "start": str(returns.index[0]),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) * 100.0,
        "volatility": float(returns.std(ddof=1) * np.sqrt(12.0)) * 100.0,
        "sharpe": float(excess.mean() / excess.std(ddof=1) * np.sqrt(12.0)) if excess.std(ddof=1) > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()) * 100.0,
        "time_in_cash": float((exposure == 0.0).mean()) * 100.0,
        "switches_per_year": float(switches.sum() / years),
    }
