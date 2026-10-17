from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.config import project_root
from world_model.data.csi import load_cot_panel, load_price_panel
from world_model.features.market import (
    breadth_score,
    cot_crowding_score,
    momentum_score,
    trend_score,
    volatility_score,
    weighted_average,
)


COMPONENT_COLUMNS = [
    "trend_score",
    "momentum_score",
    "breadth_score",
    "volatility_score",
    "rates_score",
    "dollar_score",
    "commodities_score",
    "positioning_score",
]


@dataclass(frozen=True)
class MarketStateSnapshot:
    as_of_date: pd.Timestamp
    regime_label: str
    composite_score: float
    component_scores: dict[str, float]
    component_coverage: float


@dataclass(frozen=True)
class AnalogResult:
    as_of_date: pd.Timestamp
    current_vector: pd.Series
    analogs: pd.DataFrame
    distribution: pd.DataFrame


class MarketStateEngine:
    def __init__(self, data_roots: dict[str, str], config: dict[str, Any]):
        self.data_roots = data_roots
        self.config = config

    def load_prices(self) -> pd.DataFrame:
        return load_price_panel(self.data_roots["csi_processed"], self.config["instruments"])

    def load_cot(self) -> pd.DataFrame:
        return load_cot_panel(self.data_roots["csi_cot_processed"], self.config.get("cot", {}))

    def load_breadth(self) -> pd.DataFrame:
        breadth_config = self.config.get("breadth", {})
        if not breadth_config.get("enabled", False):
            return pd.DataFrame()
        configured_path = Path(breadth_config["file"]).expanduser()
        path = configured_path if configured_path.is_absolute() else project_root() / configured_path
        if not path.exists():
            return pd.DataFrame()
        breadth = pd.read_parquet(path)
        if not isinstance(breadth.index, pd.DatetimeIndex):
            if "date" not in breadth.columns:
                raise ValueError(f"Breadth file must have a DatetimeIndex or date column: {path}")
            breadth["date"] = pd.to_datetime(breadth["date"])
            breadth = breadth.set_index("date")
        breadth.index = pd.to_datetime(breadth.index).normalize()
        return breadth.sort_index()

    def build_asset_scores(self, prices: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        prices = self.load_prices() if prices is None else prices
        trend = prices.apply(trend_score)
        momentum = prices.apply(momentum_score)
        return {"trend": trend, "momentum": momentum}

    def build_state_history(self) -> pd.DataFrame:
        prices = self.load_prices()
        cots = self.load_cot()
        breadth = self.load_breadth()
        asset_scores = self.build_asset_scores(prices)
        components_config = self.config["components"]

        state = pd.DataFrame(index=prices.index)

        trend_assets = components_config["trend"]["assets"]
        state["trend_score"] = asset_scores["trend"][trend_assets].mean(axis=1)

        momentum_assets = components_config["momentum"]["assets"]
        state["momentum_score"] = asset_scores["momentum"][momentum_assets].mean(axis=1)

        if not breadth.empty and "breadth" in components_config:
            state["breadth_score"] = breadth_score(breadth).reindex(prices.index).ffill()
        else:
            state["breadth_score"] = np.nan

        vol_config = components_config["volatility"]
        state["volatility_score"] = volatility_score(
            prices[vol_config["equity_asset"]],
            prices.get(vol_config.get("volatility_asset")),
        )

        rates_assets = components_config["rates"]["assets"]
        state["rates_score"] = asset_scores["trend"][rates_assets].mean(axis=1)

        dollar_config = components_config["dollar"]
        dollar = asset_scores["trend"][dollar_config["assets"]].mean(axis=1)
        state["dollar_score"] = -dollar if dollar_config.get("invert", False) else dollar

        commodity_assets = components_config["commodities"]["assets"]
        state["commodities_score"] = asset_scores["momentum"][commodity_assets].mean(axis=1)

        positioning_assets = components_config.get("positioning", {}).get("assets", [])
        if not cots.empty and positioning_assets:
            cot_daily = cots.reindex(prices.index).ffill()
            cot_scores = cot_daily[positioning_assets].apply(cot_crowding_score)
            state["positioning_score"] = cot_scores.mean(axis=1)
        else:
            state["positioning_score"] = np.nan

        component_weights = self._component_weights()
        state["component_coverage"] = state[COMPONENT_COLUMNS].notna().mean(axis=1)
        state["composite_score"] = weighted_average(state[COMPONENT_COLUMNS], component_weights)
        state["regime_label"] = state.apply(self._label_row, axis=1)
        return state

    def snapshot(self, as_of: str | pd.Timestamp | None = None) -> MarketStateSnapshot:
        state = self.build_state_history()
        row = self._row_as_of(state, as_of)
        components = {col: _clean_float(row[col]) for col in COMPONENT_COLUMNS}
        return MarketStateSnapshot(
            as_of_date=pd.Timestamp(row.name),
            regime_label=str(row["regime_label"]),
            composite_score=float(row["composite_score"]),
            component_scores=components,
            component_coverage=float(row["component_coverage"]),
        )

    def analog_forecast(
        self,
        as_of: str | pd.Timestamp | None = None,
        top_n: int | None = None,
    ) -> AnalogResult:
        state = self.build_state_history()
        prices = self.load_prices()
        target = self.config.get("target_asset", "ES")
        target_close = prices[target].reindex(state.index).ffill()
        horizons: dict[str, int] = self.config.get("horizons", {})
        analog_config = self.config.get("analog", {})
        top_n = int(top_n or analog_config.get("top_n", 40))
        min_coverage = float(analog_config.get("min_component_coverage", 0.70))
        max_horizon = int(max(horizons.values())) if horizons else 252

        current_row = self._row_as_of(state, as_of)
        as_of_date = pd.Timestamp(current_row.name)
        current_vector = current_row[COMPONENT_COLUMNS].astype(float)

        candidate_state = state.loc[state.index < as_of_date].copy()
        candidate_state = candidate_state[candidate_state["component_coverage"] >= min_coverage]
        candidate_state = candidate_state.loc[candidate_state.index <= as_of_date - pd.tseries.offsets.BDay(max_horizon)]
        vectors = candidate_state[COMPONENT_COLUMNS].dropna(thresh=max(3, int(len(COMPONENT_COLUMNS) * min_coverage)))
        vectors = vectors.dropna(axis=0, subset=current_vector.dropna().index)

        comparable_cols = current_vector.dropna().index.tolist()
        if not comparable_cols:
            raise ValueError("Current state vector has no comparable component scores.")
        if vectors.empty:
            raise ValueError("No historical analog candidates after coverage and forward-return filters.")

        weights = pd.Series(self._component_weights(), dtype=float).reindex(comparable_cols).fillna(1.0)
        diffs = vectors[comparable_cols].sub(current_vector[comparable_cols], axis=1)
        distances = np.sqrt((diffs.pow(2).multiply(weights, axis=1)).sum(axis=1) / weights.sum())
        min_gap = int(analog_config.get("min_trading_days_between_analogs", 0))
        closest = select_diverse_analogs(distances, top_n=top_n, min_trading_days=min_gap)

        analogs = pd.DataFrame(index=closest.index)
        analogs["distance"] = closest
        analogs["similarity"] = 1.0 / (1.0 + analogs["distance"])
        analogs["probability_weight"] = analogs["similarity"] / analogs["similarity"].sum()
        analogs["regime_label"] = state.loc[analogs.index, "regime_label"]
        analogs["composite_score"] = state.loc[analogs.index, "composite_score"]

        for label, days in horizons.items():
            analogs[f"{label}_return"] = _forward_return(target_close, days).reindex(analogs.index)
            analogs[f"{label}_max_drawdown"] = _forward_drawdown(target_close, days).reindex(analogs.index)

        distribution = summarize_forward_distribution(analogs, horizons)
        return AnalogResult(
            as_of_date=as_of_date,
            current_vector=current_vector,
            analogs=analogs.sort_values("distance"),
            distribution=distribution,
        )

    def _component_weights(self) -> dict[str, float]:
        weights = {}
        for name, settings in self.config["components"].items():
            column = f"{name}_score"
            if name == "commodities":
                column = "commodities_score"
            weights[column] = float(settings.get("weight", 1.0))
        return weights

    def _row_as_of(self, state: pd.DataFrame, as_of: str | pd.Timestamp | None) -> pd.Series:
        usable = state.dropna(subset=["composite_score"])
        if usable.empty:
            raise ValueError("No usable market state rows.")
        if as_of is None:
            return usable.iloc[-1]
        as_of_ts = pd.Timestamp(as_of).normalize()
        prior = usable.loc[usable.index <= as_of_ts]
        if prior.empty:
            raise ValueError(f"No market state available on or before {as_of_ts.date()}.")
        return prior.iloc[-1]

    def _label_row(self, row: pd.Series) -> str:
        score = row.get("composite_score")
        if pd.isna(score):
            return "insufficient_data"
        if score >= 0.45:
            base = "risk_on"
        elif score >= 0.15:
            base = "constructive"
        elif score > -0.15:
            base = "neutral"
        elif score > -0.45:
            base = "fragile"
        else:
            base = "risk_off"

        qualifiers: list[str] = []
        if row.get("positioning_score", 0) < -0.45 and score > 0.0:
            qualifiers.append("crowded")
        if row.get("breadth_score", 0) < -0.25 and score > 0.0:
            qualifiers.append("weak_breadth")
        if row.get("volatility_score", 0) < -0.35:
            qualifiers.append("volatile")
        if row.get("trend_score", 0) > 0.35 and row.get("momentum_score", 0) < -0.10:
            qualifiers.append("waning_momentum")
        return "_".join([base, *qualifiers])


def summarize_forward_distribution(analogs: pd.DataFrame, horizons: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    weights = analogs["probability_weight"]
    for label in horizons:
        returns = analogs[f"{label}_return"].dropna()
        drawdowns = analogs[f"{label}_max_drawdown"].dropna()
        valid_weights = weights.reindex(returns.index)
        valid_weights = valid_weights / valid_weights.sum()
        drawdown_weights = weights.reindex(drawdowns.index)
        drawdown_weights = drawdown_weights / drawdown_weights.sum()
        rows.append(
            {
                "horizon": label,
                "trading_days": horizons[label],
                "analog_count": int(returns.count()),
                "weighted_mean_return": float((returns * valid_weights).sum()),
                "median_return": float(returns.median()),
                "p10_return": float(returns.quantile(0.10)),
                "p25_return": float(returns.quantile(0.25)),
                "p75_return": float(returns.quantile(0.75)),
                "p90_return": float(returns.quantile(0.90)),
                "probability_positive": float(((returns > 0).astype(float) * valid_weights).sum()),
                "probability_drawdown_gt_5pct": float(((drawdowns <= -0.05).astype(float) * drawdown_weights).sum()),
                "probability_drawdown_gt_10pct": float(((drawdowns <= -0.10).astype(float) * drawdown_weights).sum()),
            }
        )
    return pd.DataFrame(rows)


def select_diverse_analogs(
    distances: pd.Series,
    top_n: int,
    min_trading_days: int = 0,
) -> pd.Series:
    sorted_distances = distances.sort_values()
    if min_trading_days <= 0:
        return sorted_distances.head(top_n)

    ordered_index = pd.DatetimeIndex(sorted_distances.index).sort_values()
    positions = {date: position for position, date in enumerate(ordered_index)}
    selected: list[pd.Timestamp] = []
    for date in sorted_distances.index:
        date = pd.Timestamp(date)
        position = positions[date]
        if all(abs(position - positions[pd.Timestamp(existing)]) >= min_trading_days for existing in selected):
            selected.append(date)
        if len(selected) >= top_n:
            break
    return sorted_distances.reindex(selected)


def _forward_return(close: pd.Series, days: int) -> pd.Series:
    return close.shift(-days) / close - 1.0


def _forward_drawdown(close: pd.Series, days: int) -> pd.Series:
    path_returns = [pd.Series(0.0, index=close.index)]
    path_returns.extend(close.shift(-step) / close - 1.0 for step in range(1, days + 1))
    return pd.concat(path_returns, axis=1).min(axis=1)


def _clean_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def write_state_outputs(
    state: pd.DataFrame,
    output_path: str | Path,
) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        state.to_parquet(path)
    elif path.suffix == ".csv":
        state.to_csv(path)
    else:
        raise ValueError("State output path must end in .parquet or .csv")
