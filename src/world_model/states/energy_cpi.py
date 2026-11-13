"""Bottom-up energy and headline CPI nowcast, built only from data known the day before each CPI release.

Headline CPI is assembled from blocks, each forecast from its own driver:

- **Gasoline**: the month's average weekly retail price (``GASREGW``) against the previous month's, calibrated
  to the unadjusted gasoline CPI. BLS's seasonal adjustment is then applied using last year's seasonal step
  for the same pair of months, taken from the adjusted and unadjusted indexes as published at the time.
- **Electricity**: the average of its last three monthly changes (it moves slowly).
- **Utility gas**: Henry Hub spot prices this month and the two before, through a trailing regression.
- **Fuel oil**: retail diesel prices this month and last, through a trailing regression.
- **Energy**: the four blocks combined with weights fitted on the trailing three years.
- **Headline**: core, food and energy combined the same way. Core comes from the caller (our core CPI
  forecast); food is the average of its last three monthly changes.

Market prices (retail gasoline, diesel, Henry Hub) are effectively never revised, so their latest values up to
the cutoff stand in for vintages. Every CPI series is read as published on the cutoff date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table

SERIES = {
    "headline": "CPIAUCSL",
    "core": "CPILFESL",
    "food": "CPIUFDSL",
    "energy": "CPIENGSL",
    "gasoline": "CUSR0000SETB01",
    "gasoline_nsa": "CUUR0000SETB01",
    "electricity": "CUSR0000SEHF01",
    "utility_gas": "CUSR0000SEHF02",
    "fuel_oil": "CUSR0000SEHE",
    "retail_gasoline": "GASREGW",
    "retail_diesel": "GASDESW",
    "henry_hub": "DHHNGSP",
}
ENERGY_BLOCKS = ("gasoline", "electricity", "utility_gas", "fuel_oil")
HEADLINE_BLOCKS = ("core", "food", "energy")


@dataclass(frozen=True)
class Settings:
    calibration_months: int = 60
    weight_months: int = 36
    trailing_months: int = 3


def load_tables(fred_root: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, series_id in SERIES.items():
        table = load_vintage_table(fred_root, series_id).copy()
        table["observation_date"] = pd.to_datetime(table["observation_date"])
        table["realtime_start"] = pd.to_datetime(table["realtime_start"])
        tables[name] = table
    return tables


def known(vintage: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Each observation's value as published on ``cutoff`` (the latest vintage that had started by then)."""
    frame = vintage.loc[vintage["realtime_start"] <= cutoff]
    if frame.empty:
        return pd.Series(dtype=float)
    frame = frame.sort_values(["realtime_start", "observation_date"]).drop_duplicates("observation_date", keep="last")
    series = frame.set_index("observation_date")["value"].astype(float).dropna().sort_index()
    return series[series.index <= cutoff]


def monthly_changes(index: pd.Series) -> pd.Series:
    """Percent change on the previous month, keyed by month start."""
    monthly = index.copy()
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp()
    monthly = monthly.groupby(level=0).last()
    expected = monthly.index.to_series().shift(1) + pd.DateOffset(months=1)
    change = 100.0 * (monthly / monthly.shift(1) - 1.0)
    return change.where(expected.eq(monthly.index.to_series()))


def monthly_average(prices: pd.Series, carry_to: pd.Timestamp | None = None) -> pd.Series:
    """Average price per month. With ``carry_to``, the last price is held flat to that month's end."""
    prices = prices.dropna().sort_index()
    if prices.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    prices.index = pd.DatetimeIndex(prices.index)
    if carry_to is not None:
        last_date = prices.index[-1]
        month_end = carry_to + pd.offsets.MonthEnd(0)
        if last_date < month_end:
            step = pd.Timedelta(days=7) if len(prices) > 1 and (prices.index[-1] - prices.index[-2]).days >= 5 else pd.Timedelta(days=1)
            filler = pd.date_range(last_date + step, month_end, freq=step)
            prices = pd.concat([prices, pd.Series(prices.iloc[-1], index=filler)])
    return prices.groupby(prices.index.to_period("M").to_timestamp()).mean()


def release_events(headline: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """First release date of each monthly CPI print (lag 20-85 days, as in the release forecast module)."""
    first = headline.groupby("observation_date")["realtime_start"].min().rename("release_date").reset_index()
    lag = (first["release_date"] - first["observation_date"]).dt.days
    first = first[lag.between(20, 85) & (first["observation_date"] >= pd.Timestamp(start))]
    if end is not None:
        first = first[first["observation_date"] <= pd.Timestamp(end)]
    return first.sort_values("observation_date").reset_index(drop=True)


def _fit(y: pd.Series, x: pd.DataFrame, intercept: bool, months: int) -> np.ndarray | None:
    data = pd.concat([y.rename("_y"), x], axis=1).dropna().tail(months)
    if len(data) < max(12, x.shape[1] * 4):
        return None
    design = data[x.columns].to_numpy(float)
    if intercept:
        design = np.column_stack([np.ones(len(design)), design])
    coef, *_ = np.linalg.lstsq(design, data["_y"].to_numpy(float), rcond=None)
    return coef


def _predict(coef: np.ndarray | None, values: list[float], intercept: bool) -> float | None:
    if coef is None or any(v is None or not np.isfinite(v) for v in values):
        return None
    row = np.array(([1.0] if intercept else []) + list(values))
    return float(row @ coef)


def _trailing_mean(changes: pd.Series, before: pd.Timestamp, months: int) -> float | None:
    window = changes[changes.index < before].dropna().tail(months)
    return float(window.mean()) if len(window) == months else None


def nowcast(
    tables: Mapping[str, pd.DataFrame],
    month: pd.Timestamp,
    cutoff: pd.Timestamp,
    core_forecast: float | None = None,
    settings: Settings = Settings(),
) -> dict[str, Any]:
    """Forecast each block's adjusted month-over-month change (percent) for ``month`` using data known at ``cutoff``."""
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    previous = month - pd.DateOffset(months=1)
    seen = {name: known(table, cutoff) for name, table in tables.items()}
    changes = {name: monthly_changes(seen[name]) for name in SERIES if not name.startswith(("retail", "henry"))}
    out: dict[str, Any] = {"month": month, "cutoff": cutoff}

    # Gasoline: retail price change, calibrated to the unadjusted CPI, then seasonally adjusted.
    retail = monthly_average(seen["retail_gasoline"], carry_to=month)
    retail_change = 100.0 * (retail / retail.shift(1) - 1.0)
    out["retail_gasoline_change"] = float(retail_change.get(month, np.nan))
    unadjusted = changes["gasoline_nsa"]
    coef = _fit(unadjusted[unadjusted.index < month], retail_change.to_frame("x"), True, settings.calibration_months)
    gasoline_nsa = _predict(coef, [out["retail_gasoline_change"]], True)
    out["gasoline_nsa"] = gasoline_nsa
    out["gasoline"] = _seasonally_adjust(gasoline_nsa, seen["gasoline_nsa"], seen["gasoline"], month)

    # Electricity: slow-moving, so its recent average.
    out["electricity"] = _trailing_mean(changes["electricity"], month, settings.trailing_months)

    # Utility gas: Henry Hub this month and the two before.
    hub = monthly_average(seen["henry_hub"], carry_to=month)
    hub_change = 100.0 * (hub / hub.shift(1) - 1.0)
    lags = pd.concat({f"lag{k}": hub_change.shift(k) for k in range(3)}, axis=1)
    target = changes["utility_gas"]
    coef = _fit(target[target.index < month], lags, True, settings.calibration_months)
    out["utility_gas"] = _predict(coef, [float(lags[c].get(month, np.nan)) for c in lags.columns], True)
    if out["utility_gas"] is None:
        out["utility_gas"] = _trailing_mean(target, month, settings.trailing_months)

    # Fuel oil: retail diesel this month and last.
    diesel = monthly_average(seen["retail_diesel"], carry_to=month)
    diesel_change = 100.0 * (diesel / diesel.shift(1) - 1.0)
    lags = pd.concat({f"lag{k}": diesel_change.shift(k) for k in range(2)}, axis=1)
    target = changes["fuel_oil"]
    coef = _fit(target[target.index < month], lags, True, settings.calibration_months)
    out["fuel_oil"] = _predict(coef, [float(lags[c].get(month, np.nan)) for c in lags.columns], True)

    # Energy and headline: fitted weights on the trailing three years of published changes.
    out["energy_weights"] = _weights(changes["energy"], changes, ENERGY_BLOCKS, month, settings)
    out["energy"] = _combine(out["energy_weights"], out, ENERGY_BLOCKS)
    out["food"] = _trailing_mean(changes["food"], month, settings.trailing_months)
    out["core"] = core_forecast if core_forecast is not None else _trailing_mean(changes["core"], month, settings.trailing_months)
    out["headline_weights"] = _weights(changes["headline"], changes, HEADLINE_BLOCKS, month, settings)
    out["headline"] = _combine(out["headline_weights"], out, HEADLINE_BLOCKS)
    out["previous"] = {name: _value(changes[name], previous) for name in ("gasoline", "energy", "food", "core", "headline")}
    return out


def _seasonally_adjust(change_nsa: float | None, unadjusted: pd.Series, adjusted: pd.Series, month: pd.Timestamp) -> float | None:
    """Turn an unadjusted change into an adjusted one using last year's seasonal step between the same months."""
    if change_nsa is None:
        return None
    ratio = (unadjusted / adjusted).dropna()
    ratio.index = pd.DatetimeIndex(ratio.index).to_period("M").to_timestamp()
    year_ago, year_ago_previous = month - pd.DateOffset(months=12), month - pd.DateOffset(months=13)
    if year_ago not in ratio.index or year_ago_previous not in ratio.index:
        return None
    seasonal_step = ratio[year_ago] / ratio[year_ago_previous]
    return 100.0 * ((1.0 + change_nsa / 100.0) / seasonal_step - 1.0)


def _weights(target: pd.Series, changes: Mapping[str, pd.Series], blocks: tuple[str, ...], month: pd.Timestamp, settings: Settings) -> list[float] | None:
    x = pd.concat({b: changes[b] for b in blocks}, axis=1)
    coef = _fit(target[target.index < month], x, False, settings.weight_months)
    return None if coef is None else [float(c) for c in coef]


def _combine(weights: list[float] | None, values: Mapping[str, Any], blocks: tuple[str, ...]) -> float | None:
    parts = [values.get(b) for b in blocks]
    if weights is None or any(p is None or not np.isfinite(p) for p in parts):
        return None
    return float(np.dot(weights, parts))


def _value(series: pd.Series, month: pd.Timestamp) -> float | None:
    value = series.get(month)
    return None if value is None or not np.isfinite(value) else float(value)


def first_prints(tables: Mapping[str, pd.DataFrame], month: pd.Timestamp, release: pd.Timestamp) -> dict[str, float | None]:
    """Each block's adjusted change for ``month`` as first published on ``release``."""
    out = {}
    for name in ("headline", "core", "food", "energy", *ENERGY_BLOCKS, "gasoline_nsa"):
        out[name] = _value(monthly_changes(known(tables[name], release)), month)
    return out


def backtest(
    tables: Mapping[str, pd.DataFrame],
    start: str = "2013-06-01",
    end: str | None = None,
    core_forecasts: Mapping[pd.Timestamp, float] | None = None,
    settings: Settings = Settings(),
) -> pd.DataFrame:
    """One row per CPI release: each block's forecast (made the day before) and its first print."""
    rows = []
    for event in release_events(tables["headline"], start, end).itertuples(index=False):
        month, release = pd.Timestamp(event.observation_date), pd.Timestamp(event.release_date)
        cutoff = release - pd.Timedelta(days=1)
        ours_core = (core_forecasts or {}).get(month)
        forecast = nowcast(tables, month, cutoff, ours_core, settings)
        oracle = nowcast(tables, month, cutoff, None, settings)
        actual = first_prints(tables, month, release)
        row = {"month": month, "release_date": release, "core_source": "ours" if ours_core is not None else "trailing"}
        for name in ("headline", "core", "food", "energy", *ENERGY_BLOCKS):
            row[f"{name}_forecast"] = forecast.get(name)
            row[f"{name}_actual"] = actual.get(name)
            row[f"{name}_previous"] = forecast["previous"].get(name) if name in forecast["previous"] else None
        row["gasoline_nsa_forecast"] = forecast.get("gasoline_nsa")
        row["gasoline_nsa_actual"] = actual.get("gasoline_nsa")
        row["retail_gasoline_change"] = forecast.get("retail_gasoline_change")
        # With core and food known exactly, how good is the energy block alone at headline?
        weights = oracle.get("headline_weights")
        if weights and all(actual.get(b) is not None for b in ("core", "food")) and forecast.get("energy") is not None:
            row["headline_given_core_food"] = float(np.dot(weights, [actual["core"], actual["food"], forecast["energy"]]))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def validation(results: pd.DataFrame, baseline: pd.Series) -> dict[str, Any]:
    """Headline accuracy against the release module's autoregressive benchmark, in full and in each half.

    ``baseline`` is that benchmark's forecast keyed by month. Validated means lower mean absolute error than
    the benchmark in the full sample and in both halves, the same bar the release forecast module uses.
    """
    sample = results.assign(baseline=results["month"].map(baseline)).dropna(
        subset=["headline_forecast", "headline_actual", "baseline"]
    ).sort_values("month")
    if len(sample) < 24:
        return {"validated": False, "observations": int(len(sample))}
    model = (sample["headline_actual"] - sample["headline_forecast"]).abs()
    bench = (sample["headline_actual"] - sample["baseline"]).abs()
    half = len(sample) // 2
    improvements = [float(bench.mean() - model.mean()), float(bench.iloc[:half].mean() - model.iloc[:half].mean()), float(bench.iloc[half:].mean() - model.iloc[half:].mean())]
    return {
        "validated": all(value > 0 for value in improvements),
        "observations": int(len(sample)),
        "model_mae": float(model.mean()),
        "baseline_mae": float(bench.mean()),
        "improvement_full_early_late": improvements,
    }


def site_row(
    existing: Mapping[str, Any] | None,
    live: Mapping[str, Any],
    results: pd.DataFrame,
    checks: Mapping[str, Any],
    window: int = 60,
    minimum: int = 24,
) -> dict[str, Any] | None:
    """The dashboard's headline CPI row, rebuilt from the bottom-up nowcast and its own walk-forward misses."""
    forecast = live.get("headline")
    if forecast is None:
        return None
    misses = results.dropna(subset=["headline_forecast", "headline_actual"]).sort_values("release_date").tail(window)
    row = dict(existing or {})
    row.update(
        {
            "target_id": "headline_cpi_mom",
            "target_label": "Headline CPI MoM",
            "variant_id": "bottom_up",
            "model": "bottom-up: core, food and energy blocks",
            "reference_month": f"{pd.Timestamp(live['month']):%Y-%m}",
            "forecast": float(forecast),
            "prior": live.get("previous", {}).get("headline", row.get("prior")),
            "unit": "percent",
            "validated": bool(checks.get("validated")),
            "model_mae": checks.get("model_mae"),
            "baseline_mae": checks.get("baseline_mae"),
            "observations": checks.get("observations"),
            "direction_hit_rate": None,
            "blocks": {name: live.get(name) for name in ("gasoline", "energy", "food", "core")},
        }
    )
    if len(misses) >= minimum:
        errors = misses["headline_actual"] - misses["headline_forecast"]
        q = errors.quantile([0.10, 0.25, 0.75, 0.90])
        row["range_50"] = [float(forecast + q.loc[0.25]), float(forecast + q.loc[0.75])]
        row["range_80"] = [float(forecast + q.loc[0.10]), float(forecast + q.loc[0.90])]
        row["range_observations"] = int(len(misses))
        row["range_since"] = f"{pd.Timestamp(misses['release_date'].iloc[0]):%Y-%m}"
    return row
