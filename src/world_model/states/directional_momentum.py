from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from world_model.data.csi import read_price_series
from world_model.data.norgate import read_catalog_price_series
from world_model.features.directional_momentum import (
    classify_bias,
    equity_momentum_history,
    futures_momentum_history,
)


@dataclass(frozen=True)
class DirectionalMomentumResult:
    history: pd.DataFrame
    snapshot: pd.DataFrame
    payload: dict[str, Any]
    as_of_date: pd.Timestamp


class DirectionalMomentumEngine:
    def __init__(self, data_roots: dict[str, str], config: dict[str, Any]):
        self.data_roots = data_roots
        self.config = config

    def build(self, as_of: str | pd.Timestamp | None = None) -> DirectionalMomentumResult:
        as_of_date = pd.Timestamp(as_of).normalize() if as_of else pd.Timestamp.today().normalize()
        horizons = self.config["horizons"]
        histories: list[pd.DataFrame] = []

        for asset in self.config["futures"]:
            price = read_price_series(self.data_roots["csi_processed"], asset["file"])
            price = price.loc[price.index <= as_of_date]
            component = futures_momentum_history(price, horizons)
            histories.append(_identify(component, asset, "future", "Normalized move"))

        equity_config = self.config["equities"]
        benchmark_id = str(equity_config["benchmark"]["id"])
        benchmark = self._read_norgate_series(benchmark_id).loc[:as_of_date]
        for asset in equity_config["assets"]:
            price = self._read_norgate_series(str(asset["id"])).loc[:as_of_date]
            component = equity_momentum_history(price, benchmark, horizons)
            equity_asset = {
                **asset,
                "group": "sector_etfs",
                "group_label": "Select Sector SPDR ETFs",
            }
            histories.append(_identify(component, equity_asset, "sector_etf", "Total return"))

        history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
        history = history.sort_values(["asset_class", "display_order", "date", "horizon_days"])
        snapshot = _latest_snapshot(history)
        payload = build_directional_momentum_payload(snapshot, self.config, as_of_date)
        return DirectionalMomentumResult(history, snapshot, payload, as_of_date)

    def _read_norgate_series(self, symbol: str) -> pd.Series:
        return read_catalog_price_series(
            self.data_roots["norgate_prices"],
            self.data_roots["norgate_metadata"],
            symbol,
        )


def build_directional_momentum_payload(
    snapshot: pd.DataFrame,
    config: dict[str, Any],
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    classification = config["classification"]
    freshness = config.get("freshness", {})
    for asset_id, rows in snapshot.groupby("asset_id", sort=False):
        rows = rows.sort_values("horizon_days")
        first = rows.iloc[0]
        absolute = classify_bias(rows, classification)
        relative = (
            classify_bias(rows, classification, relative=True)
            if str(first["asset_class"]) == "sector_etf"
            else None
        )
        observation_date = pd.Timestamp(rows["date"].max()).normalize()
        age_days = int((as_of_date - observation_date).days)
        due_after = int(freshness.get("due_after_days", 3))
        stale_after = int(freshness.get("stale_after_days", 7))
        freshness_status = "stale" if age_days > stale_after else "due" if age_days > due_after else "fresh"
        components = []
        for row in rows.itertuples(index=False):
            components.append(
                {
                    "id": row.horizon_id,
                    "label": row.horizon_label,
                    "days": int(row.horizon_days),
                    "weight": float(row.weight),
                    "raw": _number(row.raw_momentum),
                    "adjusted": _number(row.vol_adjusted_momentum),
                    "relative_raw": _number(row.relative_raw_momentum),
                    "relative_adjusted": _number(row.relative_vol_adjusted_momentum),
                }
            )
        assets.append(
            {
                "id": str(asset_id),
                "label": str(first["label"]),
                "asset_class": str(first["asset_class"]),
                "group": str(first["group"]),
                "group_label": str(first["group_label"]),
                "order": int(first["display_order"]),
                "raw_label": str(first["raw_label"]),
                "observation_date": observation_date.date().isoformat(),
                "age_days": age_days,
                "freshness": freshness_status,
                "absolute_bias": absolute,
                "relative_bias": relative,
                "components": components,
            }
        )
    assets.sort(key=lambda item: (item["asset_class"], item["order"]))
    return {
        "as_of_date": as_of_date.date().isoformat(),
        "benchmark": config["equities"]["benchmark"],
        "horizons": config["horizons"],
        "assets": assets,
        "methodology": {
            "futures_raw": "Cumulative daily point change normalized by the prior one-year median absolute adjusted price.",
            "equity_raw": "Norgate total return over the horizon.",
            "adjusted": "Cumulative move divided by realized daily volatility times the square root of the horizon.",
            "relative": "Sector total return relative to SPY; adjusted by realized tracking error.",
        },
    }


def _identify(
    component: pd.DataFrame,
    asset: dict[str, Any],
    asset_class: str,
    raw_label: str,
) -> pd.DataFrame:
    if component.empty:
        return component
    frame = component.reset_index()
    frame["asset_id"] = str(asset["id"])
    frame["label"] = str(asset["label"])
    frame["asset_class"] = asset_class
    frame["group"] = str(asset["group"])
    frame["group_label"] = str(asset["group_label"])
    frame["display_order"] = int(asset["order"])
    frame["raw_label"] = raw_label
    for column in [
        "relative_raw_momentum",
        "relative_vol_adjusted_momentum",
        "relative_realized_volatility",
    ]:
        if column not in frame:
            frame[column] = float("nan")
    return frame


def _latest_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    latest_dates = history.groupby("asset_id")["date"].transform("max")
    return history.loc[history["date"].eq(latest_dates)].copy().reset_index(drop=True)


def _number(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
