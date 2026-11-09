from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from world_model.data.benchmarks import load_equity_benchmark_history
from world_model.data.fred import has_vintage_series, load_vintage_table
from world_model.features.macro_dashboard import (
    compute_point_in_time_indicator_history,
    indicator_status_as_of,
)
from world_model.features.macro_dashboard_details import (
    compute_curve_move_classification,
    compute_drilldown_metrics,
    point_in_time_series,
)
from world_model.features.equity_risk_overlay import compute_equity_risk_overlay
from world_model.features.market_relevance import load_market_relevance
from world_model.features.asset_indications import build_asset_indications
from world_model.config import project_root


@dataclass(frozen=True)
class MacroDashboardResult:
    as_of_date: pd.Timestamp
    history: pd.DataFrame
    snapshot: pd.DataFrame
    risk_overlay: dict[str, Any]
    asset_indications: dict[str, Any]


class MacroDashboardEngine:
    """Build normalized dashboard rows from the standard FRED vintage store."""

    def __init__(self, data_roots: Mapping[str, str], config: Mapping[str, Any]):
        self.data_roots = dict(data_roots)
        self.config = dict(config)

    @property
    def fred_root(self) -> str:
        return self.data_roots["fred"]

    def build(
        self,
        as_of: str | pd.Timestamp | None = None,
        start_date: str | pd.Timestamp | None = None,
    ) -> MacroDashboardResult:
        as_of_date = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
        if start_date is None:
            years = int(self.config.get("build", {}).get("bootstrap_history_years", 2))
            start = as_of_date - pd.DateOffset(years=years)
        else:
            start = pd.Timestamp(start_date).normalize()
        if start > as_of_date:
            raise ValueError("start_date must be on or before as_of")

        definitions = self._definitions()
        relevance_by_indicator = self._market_relevance()
        vintage_by_id = self._load_vintages(definitions, as_of_date)
        current_series = {
            series_id: point_in_time_series(vintage)
            for series_id, vintage in vintage_by_id.items()
        }
        histories: list[pd.DataFrame] = []
        snapshots: list[dict[str, Any]] = []

        for definition in definitions:
            sources = {
                series_id: vintage_by_id[series_id]
                for series_id in definition["series_ids"]
            }
            release_history = compute_point_in_time_indicator_history(
                definition,
                sources,
                self.config["history"],
                start_date=start,
            )
            if release_history.empty:
                raise ValueError(
                    f"No calculated releases for {definition['id']} between "
                    f"{start.date()} and {as_of_date.date()}"
                )

            normalized = release_history.reset_index()
            for key, value in self._dimensions(definition).items():
                normalized[key] = value
            histories.append(normalized)

            current = indicator_status_as_of(
                release_history,
                definition,
                as_of=as_of_date,
                freshness_config=self.config["freshness"],
            )
            snapshot_record = current.to_dict()
            snapshot_record.update(self._dimensions(definition))
            if str(definition["id"]) == "treasury_curve":
                snapshot_record.update(compute_curve_move_classification(current_series))
            snapshot_record["detail_metrics"] = json.dumps(
                compute_drilldown_metrics(str(definition["id"]), current_series),
                separators=(",", ":"),
            )
            snapshot_record["market_relevance"] = json.dumps(
                relevance_by_indicator.get(str(definition["id"]), {}),
                separators=(",", ":"),
            )
            snapshots.append(snapshot_record)

        history = pd.concat(histories, ignore_index=True, sort=False)
        history = history.sort_values(["realtime_start", "display_order", "indicator_id"]).reset_index(drop=True)
        snapshot = pd.DataFrame.from_records(snapshots)
        snapshot = snapshot.sort_values(["display_order", "is_context", "indicator_id"]).reset_index(drop=True)
        overlay_config = self.config.get("equity_risk_overlay", {})
        risk_overlay = (
            compute_equity_risk_overlay(
                current_series,
                load_equity_benchmark_history(self.data_roots, overlay_config),
                as_of_date,
                overlay_config,
            )
            if overlay_config.get("enabled", False)
            else {}
        )
        asset_indications = self._asset_indications(snapshot, risk_overlay)
        return MacroDashboardResult(
            as_of_date=as_of_date,
            history=history,
            snapshot=snapshot,
            risk_overlay=risk_overlay,
            asset_indications=asset_indications,
        )

    def _market_relevance(self) -> dict[str, dict[str, Any]]:
        settings = self.config.get("market_relevance", {})
        if not settings.get("enabled", False):
            return {}
        configured_path = Path(str(settings.get("summary_path", ""))).expanduser()
        path = configured_path if configured_path.is_absolute() else project_root() / configured_path
        return load_market_relevance(
            path,
            max_relationships=int(settings.get("max_relationships_per_indicator", 4)),
        )

    def _asset_indications(
        self,
        snapshot: pd.DataFrame,
        risk_overlay: dict[str, Any],
    ) -> dict[str, Any]:
        settings = self.config.get("asset_indications", {})
        if not settings.get("enabled", False):
            return {"assets": []}
        configured_path = Path(str(settings.get("summary_path", ""))).expanduser()
        path = configured_path if configured_path.is_absolute() else project_root() / configured_path
        regime_setting = settings.get("regime_summary_path")
        regime_path = None
        if regime_setting:
            configured_regime = Path(str(regime_setting)).expanduser()
            regime_path = configured_regime if configured_regime.is_absolute() else project_root() / configured_regime
        return build_asset_indications(
            snapshot,
            risk_overlay,
            path,
            deadband_percentile_points=float(settings.get("deadband_percentile_points", 10.0)),
            regime_summary_path=regime_path,
        )

    def _load_vintages(
        self,
        definitions: list[dict[str, Any]],
        as_of_date: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        series_ids = {
            str(series_id)
            for definition in definitions
            for series_id in definition["series_ids"]
        }
        series_ids.update(str(value) for value in self.config.get("drilldown_fred_series_ids", []))
        missing = sorted(
            series_id
            for series_id in series_ids
            if not has_vintage_series(self.fred_root, series_id)
        )
        if missing:
            raise FileNotFoundError(f"Missing required FRED vintage series: {', '.join(missing)}")
        return {
            series_id: load_vintage_table(self.fred_root, series_id).loc[
                lambda frame: frame["realtime_start"] <= as_of_date
            ]
            for series_id in sorted(series_ids)
        }

    def _definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for row in self.config["rows"]:
            primary = dict(row)
            primary["is_context"] = False
            primary["parent_indicator_id"] = None
            primary["display_order"] = float(row["order"])
            definitions.append(primary)
            for context_position, context in enumerate(row.get("context", []), start=1):
                item = dict(context)
                item["component"] = row["component"]
                item["is_context"] = True
                item["parent_indicator_id"] = row["id"]
                item["display_order"] = float(row["order"]) + context_position / 100.0
                definitions.append(item)
        return definitions

    @staticmethod
    def _dimensions(definition: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "indicator_id": definition["id"],
            "label": definition["label"],
            "component": definition["component"],
            "display_order": definition["display_order"],
            "frequency": definition["frequency"],
            "polarity": definition["polarity"],
            "recipe": definition["recipe"],
            "series_ids": ",".join(str(value) for value in definition["series_ids"]),
            "is_context": bool(definition["is_context"]),
            "parent_indicator_id": definition["parent_indicator_id"],
        }


def merge_macro_dashboard_history(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    replace_from: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    required = {"indicator_id", "realtime_start"}
    if not required <= set(existing.columns):
        raise ValueError("Existing dashboard history is missing indicator_id or realtime_start")
    retained = existing.copy()
    retained["realtime_start"] = pd.to_datetime(retained["realtime_start"], errors="coerce")
    if replace_from is not None:
        cutoff = pd.Timestamp(replace_from).normalize()
        replaced_indicators = set(new["indicator_id"].astype(str))
        retained = retained.loc[
            ~(
                retained["indicator_id"].astype(str).isin(replaced_indicators)
                & (retained["realtime_start"] >= cutoff)
            )
        ]
    combined = pd.concat([retained, new], ignore_index=True, sort=False)
    combined["realtime_start"] = pd.to_datetime(combined["realtime_start"], errors="coerce")
    combined = combined.drop_duplicates(["indicator_id", "realtime_start"], keep="last")
    return combined.sort_values(["realtime_start", "display_order", "indicator_id"]).reset_index(drop=True)


def read_macro_dashboard_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return pd.DataFrame()
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    if resolved.suffix == ".csv":
        frame = pd.read_csv(resolved)
        for column in [
            "realtime_start",
            "observation_date",
            "release_date",
            "observation_release_date",
            "expected_next_release",
            "as_of_date",
        ]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame
    raise ValueError("Dashboard table path must end in .parquet or .csv")


def write_macro_dashboard_table(frame: pd.DataFrame, path: str | Path) -> None:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.suffix == ".parquet":
        frame.to_parquet(resolved, index=False)
    elif resolved.suffix == ".csv":
        frame.to_csv(resolved, index=False)
    else:
        raise ValueError("Dashboard table path must end in .parquet or .csv")
