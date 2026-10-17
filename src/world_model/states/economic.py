from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from world_model.data.fred import available_vintage_series, has_vintage_series, load_vintage_table
from world_model.features.economic import (
    RealtimeIndicator,
    build_realtime_indicator_from_vintage,
    score_indicator,
    weighted_average_signed,
)


ECONOMIC_COMPONENT_COLUMNS = [
    "growth_score",
    "labor_score",
    "inflation_score",
    "liquidity_score",
    "credit_score",
    "housing_score",
]


RECOMMENDED_SERIES = [
    {"series_id": "FEDFUNDS", "reason": "monthly effective fed funds rate for policy stance"},
    {"series_id": "DFF", "reason": "daily effective fed funds rate for high-frequency policy history"},
    {"series_id": "GS10", "reason": "10-year Treasury yield for nominal rate backdrop"},
    {"series_id": "T10Y2Y", "reason": "2s10s Treasury curve slope"},
    {"series_id": "T10Y3M", "reason": "3m10y Treasury curve slope"},
    {"series_id": "BAMLH0A0HYM2", "reason": "high-yield option-adjusted spread"},
    {"series_id": "BAMLC0A0CM", "reason": "investment-grade corporate option-adjusted spread"},
    {"series_id": "BAMLC0A4CBBB", "reason": "BBB corporate option-adjusted spread"},
    {"series_id": "NFCI", "reason": "Chicago Fed national financial conditions index"},
    {"series_id": "ANFCI", "reason": "adjusted national financial conditions index"},
    {"series_id": "STLFSI4", "reason": "St. Louis Fed financial stress index"},
    {"series_id": "MORTGAGE30US", "reason": "30-year mortgage rate for housing affordability pressure"},
    {"series_id": "BUSLOANS", "reason": "commercial and industrial loans at banks"},
    {"series_id": "TDSP", "reason": "household debt service burden"},
    {"series_id": "PSAVERT", "reason": "personal saving rate"},
]


@dataclass(frozen=True)
class EconomicStateSnapshot:
    as_of_date: pd.Timestamp
    regime_label: str
    composite_score: float
    component_scores: dict[str, float]
    component_coverage: float
    activity_score: float
    inflation_pressure_score: float
    liquidity_credit_score: float
    indicators: pd.DataFrame


class EconomicStateEngine:
    def __init__(self, data_roots: dict[str, str], config: dict[str, Any]):
        self.data_roots = data_roots
        self.config = config

    @property
    def fred_root(self) -> str:
        return self.data_roots["fred"]

    def build_state_history(self) -> pd.DataFrame:
        indicator_results = self._load_indicators()
        if not indicator_results:
            raise ValueError("No economic indicators were loaded.")

        release_index = _combined_release_index(indicator_results)
        history_index = pd.date_range(
            release_index.min(),
            release_index.max(),
            freq=self.config.get("history_frequency", "B"),
            name="date",
        )
        state = pd.DataFrame(index=history_index)

        indicator_lookup = {indicator["id"]: indicator for indicator in self._indicator_definitions()}
        score_defaults = self.config.get("scoring", {})

        for indicator_id, result in indicator_results.items():
            settings = indicator_lookup[indicator_id]
            score = score_indicator(
                result.value,
                window=int(settings.get("score_window", score_defaults.get("window", 120))),
                min_periods=int(settings.get("score_min_periods", score_defaults.get("min_periods", 36))),
                scale=float(settings.get("score_scale", score_defaults.get("scale", 2.0))),
                invert=bool(settings.get("invert", False)),
            )
            state[f"{indicator_id}_value"] = result.value.reindex(history_index).ffill()
            state[f"{indicator_id}_score"] = score.reindex(history_index).ffill()
            state[f"{indicator_id}_observation_date"] = result.latest_observation_date.reindex(history_index).ffill()

        for component, component_config in self.config["components"].items():
            indicator_ids = [indicator["id"] for indicator in self._component_indicators(component, component_config)]
            score_columns = [f"{indicator_id}_score" for indicator_id in indicator_ids if f"{indicator_id}_score" in state]
            state[f"{component}_score"] = state[score_columns].mean(axis=1) if score_columns else pd.NA

        available_components = [column for column in ECONOMIC_COMPONENT_COLUMNS if column in state]
        state["component_coverage"] = state[available_components].notna().mean(axis=1)
        state["activity_score"] = weighted_average_signed(
            state[[column for column in ["growth_score", "labor_score", "housing_score"] if column in state]],
            {"growth_score": 0.45, "labor_score": 0.35, "housing_score": 0.20},
        )
        state["inflation_pressure_score"] = state.get("inflation_score", pd.Series(index=state.index, dtype=float))
        state["liquidity_credit_score"] = weighted_average_signed(
            state[[column for column in ["liquidity_score", "credit_score"] if column in state]],
            {"liquidity_score": 0.60, "credit_score": 0.40},
        )
        state["composite_score"] = weighted_average_signed(state[available_components], self._composite_weights())
        state["regime_label"] = state.apply(self._label_row, axis=1)
        return state

    def snapshot(
        self,
        as_of: str | pd.Timestamp | None = None,
        state: pd.DataFrame | None = None,
    ) -> EconomicStateSnapshot:
        state = self.build_state_history() if state is None else state
        row = self._row_as_of(state, as_of)
        indicators = self.indicator_snapshot(state=state, as_of=row.name)
        return EconomicStateSnapshot(
            as_of_date=pd.Timestamp(row.name),
            regime_label=str(row["regime_label"]),
            composite_score=_clean_float(row["composite_score"]),
            component_scores={column: _clean_float(row.get(column)) for column in ECONOMIC_COMPONENT_COLUMNS},
            component_coverage=_clean_float(row["component_coverage"]),
            activity_score=_clean_float(row["activity_score"]),
            inflation_pressure_score=_clean_float(row["inflation_pressure_score"]),
            liquidity_credit_score=_clean_float(row["liquidity_credit_score"]),
            indicators=indicators,
        )

    def indicator_snapshot(self, state: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
        row = self._row_as_of(state, as_of)
        records: list[dict[str, Any]] = []
        for indicator in self._indicator_definitions():
            indicator_id = indicator["id"]
            value_column = f"{indicator_id}_value"
            score_column = f"{indicator_id}_score"
            observation_column = f"{indicator_id}_observation_date"
            if value_column not in state:
                continue
            records.append(
                {
                    "component": indicator["component"],
                    "indicator_id": indicator_id,
                    "series_id": indicator["series_id"],
                    "label": indicator.get("label", indicator_id),
                    "transform": indicator.get("transform", "level"),
                    "value": _clean_float(row.get(value_column)),
                    "score": _clean_float(row.get(score_column)),
                    "observation_date": row.get(observation_column),
                    "invert": bool(indicator.get("invert", False)),
                }
            )
        return pd.DataFrame(records)

    def recommended_missing_series(self) -> pd.DataFrame:
        available = available_vintage_series(self.fred_root)
        missing = [row for row in RECOMMENDED_SERIES if row["series_id"] not in available]
        return pd.DataFrame(missing)

    def _load_indicators(self) -> dict[str, RealtimeIndicator]:
        results: dict[str, RealtimeIndicator] = {}
        for indicator in self._indicator_definitions():
            series_id = indicator["series_id"]
            if not has_vintage_series(self.fred_root, series_id):
                if indicator.get("required", True):
                    raise FileNotFoundError(f"Missing required FRED vintage series: {series_id}")
                continue
            vintage = load_vintage_table(self.fred_root, series_id)
            params = dict(indicator.get("params", {}))
            results[indicator["id"]] = build_realtime_indicator_from_vintage(
                vintage,
                transform=indicator.get("transform", "level"),
                params=params,
            )
        return results

    def _indicator_definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for component, component_config in self.config["components"].items():
            definitions.extend(self._component_indicators(component, component_config))
        return definitions

    def _component_indicators(self, component: str, component_config: dict[str, Any]) -> list[dict[str, Any]]:
        indicators = []
        for indicator in component_config.get("indicators", []):
            item = dict(indicator)
            item["component"] = component
            item["id"] = item.get("id", item["series_id"].lower())
            indicators.append(item)
        return indicators

    def _composite_weights(self) -> dict[str, float]:
        weights = {}
        for component, settings in self.config["components"].items():
            column = f"{component}_score"
            weights[column] = float(settings.get("weight", 1.0)) * float(settings.get("composite_sign", 1.0))
        return weights

    def _row_as_of(self, state: pd.DataFrame, as_of: str | pd.Timestamp | None) -> pd.Series:
        usable = state.dropna(subset=["composite_score"])
        if usable.empty:
            raise ValueError("No usable economic state rows.")
        if as_of is None:
            return usable.iloc[-1]
        as_of_ts = pd.Timestamp(as_of).normalize()
        prior = usable.loc[usable.index <= as_of_ts]
        if prior.empty:
            raise ValueError(f"No economic state available on or before {as_of_ts.date()}.")
        return prior.iloc[-1]

    def _label_row(self, row: pd.Series) -> str:
        activity = row.get("activity_score")
        inflation = row.get("inflation_pressure_score")
        liquidity = row.get("liquidity_credit_score")
        if pd.isna(activity) or pd.isna(inflation):
            return "insufficient_data"

        if activity >= 0.20 and inflation <= -0.10:
            base = "goldilocks_expansion"
        elif activity >= 0.20 and inflation > 0.20:
            base = "reflationary_expansion"
        elif activity <= -0.20 and inflation > 0.20:
            base = "stagflation_pressure"
        elif activity <= -0.20 and inflation <= 0.10:
            base = "disinflationary_slowdown"
        else:
            base = "mixed_cycle"

        qualifiers: list[str] = []
        if pd.notna(liquidity):
            if liquidity >= 0.25:
                qualifiers.append("liquidity_easy")
            elif liquidity <= -0.25:
                qualifiers.append("liquidity_tight")
        if row.get("housing_score", 0.0) <= -0.35:
            qualifiers.append("housing_weak")
        if row.get("labor_score", 0.0) <= -0.35:
            qualifiers.append("labor_weak")
        return "_".join([base, *qualifiers])


def write_economic_state_outputs(state: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        state.to_parquet(path)
    elif path.suffix == ".csv":
        state.to_csv(path)
    else:
        raise ValueError("State output path must end in .parquet or .csv")


def _combined_release_index(results: dict[str, RealtimeIndicator]) -> pd.DatetimeIndex:
    indexes = [result.value.index for result in results.values() if not result.value.empty]
    if not indexes:
        raise ValueError("No release dates found in economic indicators.")
    combined = indexes[0]
    for index in indexes[1:]:
        combined = combined.union(index)
    return pd.DatetimeIndex(combined).sort_values()


def _clean_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
