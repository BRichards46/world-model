"""Bottom-up core CPI nowcast, built only from data known the day before each CPI release.

Core CPI is split into blocks that together make up all of it:

- **Shelter**: owners' equivalent rent, rent, and lodging away from home.
- **Core goods**: new vehicles, used cars and trucks, apparel, and everything else in core goods.
- **Core services excluding shelter**: medical care services, transportation services, and everything else.

Each block's adjusted monthly change is forecast as an equal mix of two averages of its own recent changes:

- **Adaptive**: the average over the window (1, 3, 6 or 12 months) that would have forecast that block best over
  the previous three years. Shelter usually picks a short window; noisy blocks such as lodging pick a long one.
- **Anchor**: the 12-month average. The adaptive part catches turns early; the anchor keeps it from chasing noise
  in calm periods. Either alone does worse than the mix in the backtest (docs/core_cpi.md).

Blocks are combined with BLS relative importance weights (each block's share of core in the December weights
that had been published by the cutoff). The two "everything else" blocks are what is left of core goods and core
services after their named parts, so the blocks add up to core.

Every CPI series is read as published on the cutoff date (FRED vintages). Relative importance comes from the
BLS store and is treated as known once the January CPI report that publishes it is out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table
from world_model.states.energy_cpi import known, monthly_changes, release_events

SERIES = {
    "core": "CPILFESL",
    "oer": "CUSR0000SEHC",
    "rent": "CUSR0000SEHA",
    "lodging": "CUSR0000SEHB",
    "goods": "CUSR0000SACL1E",
    "services": "CUSR0000SASLE",
    "new_vehicles": "CUSR0000SETA01",
    "used_vehicles": "CUSR0000SETA02",
    "apparel": "CPIAPPSL",
    "medical": "CUSR0000SAM2",
    "transport": "CUSR0000SAS4",
}
# BLS item codes for relative importance.
ITEM_CODES = {
    "core": "SA0L1E",
    "oer": "SEHC",
    "rent": "SEHA",
    "lodging": "SEHB",
    "goods": "SACL1E",
    "services": "SASLE",
    "new_vehicles": "SETA01",
    "used_vehicles": "SETA02",
    "apparel": "SAA",
    "medical": "SAM2",
    "transport": "SAS4",
}
NAMED_BLOCKS = ("oer", "rent", "lodging", "new_vehicles", "used_vehicles", "apparel", "medical", "transport")
RESIDUALS = {
    "other_goods": ("goods", ("new_vehicles", "used_vehicles", "apparel")),
    "other_services": ("services", ("oer", "rent", "lodging", "medical", "transport")),
}
BLOCKS = NAMED_BLOCKS + tuple(RESIDUALS)
GROUPS = {
    "shelter": ("oer", "rent", "lodging"),
    "core_goods": ("new_vehicles", "used_vehicles", "apparel", "other_goods"),
    "services_ex_shelter": ("medical", "transport", "other_services"),
}


@dataclass(frozen=True)
class Settings:
    windows: tuple[int, ...] = (1, 3, 6, 12)
    selection_months: int = 36
    anchor_window: int = 12
    anchor_weight: float = 0.5


def load_tables(fred_root: str | Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, series_id in SERIES.items():
        table = load_vintage_table(fred_root, series_id).copy()
        table["observation_date"] = pd.to_datetime(table["observation_date"])
        table["realtime_start"] = pd.to_datetime(table["realtime_start"])
        tables[name] = table
    return tables


def load_relative_importance(bls_root: str | Path) -> pd.DataFrame:
    """December relative importance (CPI-U, percent of all items) by year and block."""
    path = Path(bls_root).expanduser() / "relative_importance" / "relative_importance.parquet"
    table = pd.read_parquet(path)
    names = {code: name for name, code in ITEM_CODES.items()}
    table = table[table["item_code"].isin(names)]
    weights = table.pivot_table(index="year", columns="item_code", values="cpi_u", aggfunc="first").rename(columns=names)
    return weights.astype(float).sort_index()


def weights_known(relative_importance: pd.DataFrame, core: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float] | None:
    """Each block's share of core in the latest December weights published by ``cutoff``.

    December weights come out with the following January's CPI report, so year ``y`` counts once the first
    print of January ``y + 1`` core CPI is out.
    """
    first_prints = core.groupby("observation_date")["realtime_start"].min()
    for year in sorted(relative_importance.index, reverse=True):
        january = pd.Timestamp(year=int(year) + 1, month=1, day=1)
        published = first_prints.get(january)
        if published is None or published > cutoff:
            continue
        row = relative_importance.loc[year]
        if not np.isfinite(row.get("core", np.nan)) or any(not np.isfinite(row.get(b, np.nan)) for b in ITEM_CODES):
            continue
        return {name: float(row[name] / row["core"]) for name in ITEM_CODES}
    return None


def contributions(changes: Mapping[str, pd.Series], shares: Mapping[str, float]) -> dict[str, pd.Series]:
    """Each block's contribution to core (share x change); residual blocks are what their parent leaves over."""
    out = {name: shares[name] * changes[name] for name in NAMED_BLOCKS}
    for name, (parent, parts) in RESIDUALS.items():
        out[name] = shares[parent] * changes[parent] - sum(out[p] for p in parts)
    return out


def choose_window(series: pd.Series, before: pd.Timestamp, settings: Settings = Settings()) -> int | None:
    """The averaging window that would have forecast ``series`` best over the ``selection_months`` before ``before``."""
    history = series[series.index < before].dropna()
    if len(history) < max(settings.windows) + 12:
        return None
    errors = {}
    targets = history.index[-settings.selection_months :]
    for window in settings.windows:
        averages = history.rolling(window, min_periods=window).mean().shift(1)
        errors[window] = float((history.loc[targets] - averages.loc[targets]).abs().mean())
    errors = {w: e for w, e in errors.items() if np.isfinite(e)}
    return min(errors, key=errors.get) if errors else None


def nowcast(
    tables: Mapping[str, pd.DataFrame],
    relative_importance: pd.DataFrame,
    month: pd.Timestamp,
    cutoff: pd.Timestamp,
    settings: Settings = Settings(),
) -> dict[str, Any]:
    """Forecast core CPI's adjusted month-over-month change (percent) for ``month`` from data known at ``cutoff``."""
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    out: dict[str, Any] = {"month": month, "cutoff": cutoff, "core": None}
    shares = weights_known(relative_importance, tables["core"], cutoff)
    if shares is None:
        return out
    changes = {name: monthly_changes(known(table, cutoff)) for name, table in tables.items()}
    parts = contributions(changes, shares)
    out["shares"] = {name: shares[name] for name in NAMED_BLOCKS}
    out["windows"], out["contributions"], out["blocks"] = {}, {}, {}
    for name in BLOCKS:
        series = parts[name]
        window = choose_window(series, month, settings)
        history = series[series.index < month].dropna()
        longest = max(window or 0, settings.anchor_window)
        if window is None or len(history) < longest:
            return out
        adaptive, anchor = float(history.tail(window).mean()), float(history.tail(settings.anchor_window).mean())
        value = (1.0 - settings.anchor_weight) * adaptive + settings.anchor_weight * anchor
        out["windows"][name] = window
        out["contributions"][name] = value
        if name in NAMED_BLOCKS:
            out["blocks"][name] = value / shares[name]
    out["core"] = float(sum(out["contributions"].values()))
    out["groups"] = {group: float(sum(out["contributions"][b] for b in members)) for group, members in GROUPS.items()}
    previous = month - pd.DateOffset(months=1)
    value = changes["core"].get(previous)
    out["previous"] = {"core": None if value is None or not np.isfinite(value) else float(value)}
    return out


def backtest(
    tables: Mapping[str, pd.DataFrame],
    relative_importance: pd.DataFrame,
    start: str = "2011-06-01",
    end: str | None = None,
    settings: Settings = Settings(),
) -> pd.DataFrame:
    """One row per CPI release: the core forecast made the day before, its first print, and each group's share."""
    rows = []
    for event in release_events(tables["core"], start, end).itertuples(index=False):
        month, release = pd.Timestamp(event.observation_date), pd.Timestamp(event.release_date)
        forecast = nowcast(tables, relative_importance, month, release - pd.Timedelta(days=1), settings)
        if forecast.get("core") is None:
            continue
        actual = monthly_changes(known(tables["core"], release)).get(month)
        row = {
            "month": month,
            "release_date": release,
            "core_forecast": forecast["core"],
            "core_actual": None if actual is None or not np.isfinite(actual) else float(actual),
            "core_previous": forecast["previous"]["core"],
        }
        row.update({f"{group}_contribution": value for group, value in forecast["groups"].items()})
        row.update({f"{name}_window": value for name, value in forecast["windows"].items()})
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def validation(results: pd.DataFrame, baseline: pd.Series, column: str = "core") -> dict[str, Any]:
    """Accuracy against the release module's autoregressive benchmark, in full and in each half.

    Validated means lower mean absolute error than the benchmark in the full sample and in both halves, the same
    bar the release forecast module and the headline build use.
    """
    sample = results.assign(baseline=results["month"].map(baseline)).dropna(
        subset=[f"{column}_forecast", f"{column}_actual", "baseline"]
    ).sort_values("month")
    if len(sample) < 24:
        return {"validated": False, "observations": int(len(sample))}
    model = (sample[f"{column}_actual"] - sample[f"{column}_forecast"]).abs()
    bench = (sample[f"{column}_actual"] - sample["baseline"]).abs()
    half = len(sample) // 2
    improvements = [
        float(bench.mean() - model.mean()),
        float(bench.iloc[:half].mean() - model.iloc[:half].mean()),
        float(bench.iloc[half:].mean() - model.iloc[half:].mean()),
    ]
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
    """The dashboard's core CPI row, rebuilt from the bottom-up nowcast and its own walk-forward misses."""
    forecast = live.get("core")
    if forecast is None:
        return None
    misses = results.dropna(subset=["core_forecast", "core_actual"]).sort_values("release_date").tail(window)
    row = dict(existing or {})
    row.update(
        {
            "target_id": "core_cpi_mom",
            "target_label": "Core CPI MoM",
            "variant_id": "bottom_up",
            "model": "bottom-up: shelter, core goods and services blocks",
            "reference_month": f"{pd.Timestamp(live['month']):%Y-%m}",
            "forecast": float(forecast),
            "prior": live.get("previous", {}).get("core", row.get("prior")),
            "unit": "percent",
            "validated": bool(checks.get("validated")),
            "model_mae": checks.get("model_mae"),
            "baseline_mae": checks.get("baseline_mae"),
            "observations": checks.get("observations"),
            "direction_hit_rate": None,
            "blocks": dict(live.get("groups") or {}),
        }
    )
    if len(misses) >= minimum:
        errors = misses["core_actual"] - misses["core_forecast"]
        q = errors.quantile([0.10, 0.25, 0.75, 0.90])
        row["range_50"] = [float(forecast + q.loc[0.25]), float(forecast + q.loc[0.75])]
        row["range_80"] = [float(forecast + q.loc[0.10]), float(forecast + q.loc[0.90])]
        row["range_observations"] = int(len(misses))
        row["range_since"] = f"{pd.Timestamp(misses['release_date'].iloc[0]):%Y-%m}"
    return row
