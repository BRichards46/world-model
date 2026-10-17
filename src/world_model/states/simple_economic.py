from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from world_model.features.economic import weighted_average_signed
from world_model.states.combined import (
    CombinedTarget,
    CombinedValidationResult,
    run_combined_forecasts,
    summarize_combined_forecasts,
)
from world_model.states.economic import EconomicStateEngine
from world_model.states.market import MarketStateEngine, _forward_drawdown, _forward_return
from world_model.states.target_validation import relative_forward_return


SIMPLE_ECONOMIC_AXIS_COLUMNS = ["activity_score", "inflation_pressure_score", "liquidity_credit_score"]

DEFAULT_AXIS_CONFIG = {
    "activity": {"column": "activity_score", "weight": 0.45, "composite_sign": 1.0},
    "inflation_pressure": {"column": "inflation_pressure_score", "weight": 0.25, "composite_sign": -1.0},
    "liquidity_credit": {"column": "liquidity_credit_score", "weight": 0.30, "composite_sign": 1.0},
}

DEFAULT_REGIME_THRESHOLDS = {
    "strong_activity": 0.20,
    "weak_activity": -0.20,
    "high_inflation": 0.20,
    "low_inflation": 0.10,
    "disinflation": -0.10,
    "easy_liquidity": 0.25,
    "tight_liquidity": -0.25,
}


@dataclass(frozen=True)
class SimpleEconomicSnapshot:
    as_of_date: pd.Timestamp
    regime_label: str
    simple_economic_composite_score: float
    detailed_economic_composite_score: float
    axis_scores: dict[str, float]
    axis_coverage: float


class SimpleEconomicRegimeEngine:
    def __init__(self, data_roots: dict[str, str], config: dict[str, Any], economic_config: dict[str, Any]):
        self.data_roots = data_roots
        self.config = config
        self.economic_config = economic_config

    def build_detailed_state_history(self) -> pd.DataFrame:
        return EconomicStateEngine(self.data_roots, self.economic_config).build_state_history()

    def build_state_history(self, detailed_state: pd.DataFrame | None = None) -> pd.DataFrame:
        detailed = self.build_detailed_state_history() if detailed_state is None else detailed_state
        state = pd.DataFrame(index=detailed.index)
        for axis in self._axis_config().values():
            column = axis["column"]
            source = axis.get("source_column", column)
            if source not in detailed:
                raise KeyError(f"Detailed economic state is missing required axis column: {source}")
            state[column] = detailed[source].astype(float)

        state["detailed_economic_composite_score"] = detailed["composite_score"].astype(float)
        state["axis_coverage"] = state[SIMPLE_ECONOMIC_AXIS_COLUMNS].notna().mean(axis=1)
        state["simple_economic_composite_score"] = weighted_average_signed(
            state[SIMPLE_ECONOMIC_AXIS_COLUMNS],
            self._axis_weights(),
        )
        thresholds = self.config.get("regime_thresholds", {})
        state["regime_label"] = state.apply(lambda row: label_simple_economic_regime(row, thresholds), axis=1)
        return state

    def snapshot(
        self,
        as_of: str | pd.Timestamp | None = None,
        state: pd.DataFrame | None = None,
    ) -> SimpleEconomicSnapshot:
        state = self.build_state_history() if state is None else state
        row = row_as_of(state, as_of)
        return SimpleEconomicSnapshot(
            as_of_date=pd.Timestamp(row.name),
            regime_label=str(row["regime_label"]),
            simple_economic_composite_score=_clean_float(row["simple_economic_composite_score"]),
            detailed_economic_composite_score=_clean_float(row["detailed_economic_composite_score"]),
            axis_scores={column: _clean_float(row[column]) for column in SIMPLE_ECONOMIC_AXIS_COLUMNS},
            axis_coverage=_clean_float(row["axis_coverage"]),
        )

    def validate(
        self,
        market_config: dict[str, Any] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        simple_state: pd.DataFrame | None = None,
        detailed_state: pd.DataFrame | None = None,
    ) -> CombinedValidationResult:
        detailed = self.build_detailed_state_history() if detailed_state is None else detailed_state
        simple = self.build_state_history(detailed) if simple_state is None else simple_state
        market_state = None
        prices = None
        if market_config is not None:
            market_engine = MarketStateEngine(self.data_roots, market_config)
            market_state = market_engine.build_state_history()
            prices = market_engine.load_prices().ffill()

        panel = build_simple_economic_validation_panel(
            simple_state=simple,
            detailed_state=detailed,
            market_state=market_state,
        )
        targets = build_simple_economic_validation_targets(
            simple_state=simple,
            detailed_state=detailed,
            market_state=market_state,
            prices=prices,
            horizons=self.config.get("horizons", {}),
            config=self.config.get("validation", {}),
            target_asset=market_config.get("target_asset", "ES") if market_config is not None else "ES",
        )
        validation_config = dict(self.config.get("validation", {}))
        validation_config["models"] = validation_config.get("models", [])
        forecasts = run_combined_forecasts(
            state_panel=panel,
            targets=targets,
            models=validation_config["models"],
            config=validation_config,
            start=start,
            end=end,
        )
        return CombinedValidationResult(
            state_panel=panel,
            forecasts=forecasts,
            summary=summarize_simple_economic_forecasts(forecasts),
        )

    def _axis_config(self) -> dict[str, dict[str, Any]]:
        axes = {name: dict(settings) for name, settings in DEFAULT_AXIS_CONFIG.items()}
        for name, settings in self.config.get("axes", {}).items():
            merged = dict(axes.get(name, {}))
            merged.update(settings)
            axes[name] = merged
        return axes

    def _axis_weights(self) -> dict[str, float]:
        weights = {}
        for axis in self._axis_config().values():
            weights[axis["column"]] = float(axis.get("weight", 1.0)) * float(axis.get("composite_sign", 1.0))
        return weights


def build_simple_economic_validation_panel(
    simple_state: pd.DataFrame,
    detailed_state: pd.DataFrame,
    market_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    panel = simple_state.copy()
    panel["detailed_economic_composite_score"] = detailed_state["composite_score"].reindex(panel.index).ffill()
    if market_state is not None:
        panel["detailed_market_composite_score"] = market_state["composite_score"].reindex(panel.index).ffill()
        panel["market_composite_score"] = panel["detailed_market_composite_score"]
    else:
        panel["market_composite_score"] = panel["detailed_economic_composite_score"]
    panel["economic_composite_score"] = panel["simple_economic_composite_score"]
    return panel


def build_simple_economic_validation_targets(
    simple_state: pd.DataFrame,
    detailed_state: pd.DataFrame,
    market_state: pd.DataFrame | None,
    prices: pd.DataFrame | None,
    horizons: dict[str, int],
    config: dict[str, Any],
    target_asset: str = "ES",
) -> list[CombinedTarget]:
    if not horizons:
        raise ValueError("Simple economic regime config must define horizons.")
    targets: list[CombinedTarget] = []
    simple_composite = simple_state["simple_economic_composite_score"]
    detailed_composite = detailed_state["composite_score"].reindex(simple_state.index).ffill()
    market_composite = (
        market_state["composite_score"].reindex(simple_state.index).ffill() if market_state is not None else None
    )
    target_close = prices[target_asset].reindex(simple_state.index).ffill() if prices is not None else None
    risk_on_defensive = {}
    if prices is not None:
        aligned_prices = prices.reindex(simple_state.index).ffill()
        risk_on_defensive = {
            label: relative_forward_return(
                aligned_prices,
                ["ES", "NQ", "CL", "HG"],
                ["ZN", "ZB", "GC"],
                int(days),
            )
            for label, days in horizons.items()
        }

    for label, days in horizons.items():
        days = int(days)
        future_simple = simple_composite.shift(-days)
        simple_drop = future_simple - simple_composite <= float(config.get("simple_deterioration_threshold", -0.25))
        targets.append(
            CombinedTarget(
                name=f"simple_economic_deterioration_{label}",
                label=f"Simple economic deterioration {label}",
                kind="binary",
                horizon=label,
                trading_days=days,
                target=simple_drop.astype(float).where(future_simple.notna() & simple_composite.notna()),
            )
        )

        future_detailed = detailed_composite.shift(-days)
        detailed_drop = future_detailed - detailed_composite <= float(config.get("detailed_deterioration_threshold", -0.25))
        targets.append(
            CombinedTarget(
                name=f"detailed_economic_deterioration_{label}",
                label=f"Detailed economic deterioration {label}",
                kind="binary",
                horizon=label,
                trading_days=days,
                target=detailed_drop.astype(float).where(future_detailed.notna() & detailed_composite.notna()),
            )
        )

        if market_composite is not None:
            future_market = market_composite.shift(-days)
            market_drop = future_market - market_composite <= float(config.get("market_deterioration_threshold", -0.25))
            targets.append(
                CombinedTarget(
                    name=f"market_state_deterioration_{label}",
                    label=f"Market state deterioration {label}",
                    kind="binary",
                    horizon=label,
                    trading_days=days,
                    target=market_drop.astype(float).where(future_market.notna() & market_composite.notna()),
                )
            )

        if target_close is not None:
            forward_return = _forward_return(target_close, days)
            forward_drawdown = _forward_drawdown(target_close, days)
            targets.append(
                CombinedTarget(
                    name=f"es_return_{label}",
                    label=f"ES forward return {label}",
                    kind="continuous",
                    horizon=label,
                    trading_days=days,
                    target=forward_return,
                )
            )
            targets.append(
                CombinedTarget(
                    name=f"es_drawdown_10pct_{label}",
                    label=f"ES 10% drawdown risk {label}",
                    kind="binary",
                    horizon=label,
                    trading_days=days,
                    target=(forward_drawdown <= float(config.get("drawdown_threshold", -0.10))).astype(float).where(forward_drawdown.notna()),
                )
            )
            targets.append(
                CombinedTarget(
                    name=f"risk_on_vs_defensive_{label}",
                    label=f"Risk-on vs defensive {label}",
                    kind="continuous",
                    horizon=label,
                    trading_days=days,
                    target=risk_on_defensive[label],
                )
            )
    return targets


def summarize_simple_economic_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_combined_forecasts(forecasts)
    if summary.empty:
        return summary
    detailed = summary[summary["model"].eq("detailed_economic_bucket")][
        ["target", "horizon", "error_metric"]
    ].rename(columns={"error_metric": "detailed_economic_error_metric"})
    simple = summary[summary["model"].eq("simple_economic_bucket")][
        ["target", "horizon", "error_metric"]
    ].rename(columns={"error_metric": "simple_economic_error_metric"})
    summary = summary.merge(detailed, on=["target", "horizon"], how="left")
    summary = summary.merge(simple, on=["target", "horizon"], how="left")
    summary["improvement_vs_detailed_economic"] = summary["detailed_economic_error_metric"] - summary["error_metric"]
    summary["improvement_vs_simple_economic"] = summary["simple_economic_error_metric"] - summary["error_metric"]
    return summary.sort_values(["target", "trading_days", "model"]).reset_index(drop=True)


def label_simple_economic_regime(row: pd.Series, thresholds: dict[str, Any] | None = None) -> str:
    levels = dict(DEFAULT_REGIME_THRESHOLDS)
    levels.update(thresholds or {})
    activity = row.get("activity_score")
    inflation = row.get("inflation_pressure_score")
    liquidity = row.get("liquidity_credit_score")
    if pd.isna(activity) or pd.isna(inflation):
        return "insufficient_data"

    if activity >= levels["strong_activity"] and inflation <= levels["disinflation"]:
        base = "growth_disinflation"
    elif activity >= levels["strong_activity"] and inflation > levels["high_inflation"]:
        base = "reflationary_growth"
    elif activity <= levels["weak_activity"] and inflation > levels["high_inflation"]:
        base = "stagflation_pressure"
    elif activity <= levels["weak_activity"] and inflation <= levels["low_inflation"]:
        base = "slowdown_disinflation"
    else:
        base = "mixed_cycle"

    qualifiers: list[str] = []
    if pd.notna(liquidity):
        if liquidity >= levels["easy_liquidity"]:
            qualifiers.append("liquidity_easy")
        elif liquidity <= levels["tight_liquidity"]:
            qualifiers.append("liquidity_tight")
    return "_".join([base, *qualifiers])


def row_as_of(state: pd.DataFrame, as_of: str | pd.Timestamp | None) -> pd.Series:
    usable = state.dropna(subset=["simple_economic_composite_score"])
    if usable.empty:
        raise ValueError("No usable simple economic regime rows.")
    if as_of is None:
        return usable.iloc[-1]
    as_of_ts = pd.Timestamp(as_of).normalize()
    prior = usable.loc[usable.index <= as_of_ts]
    if prior.empty:
        raise ValueError(f"No simple economic regime available on or before {as_of_ts.date()}.")
    return prior.iloc[-1]


def write_simple_economic_state_outputs(state: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        state.to_parquet(path)
    elif path.suffix == ".csv":
        state.to_csv(path)
    else:
        raise ValueError("State output path must end in .parquet or .csv")


def write_simple_economic_validation_outputs(result: CombinedValidationResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_panel": root / "simple_economic_validation_panel.parquet",
        "forecasts": root / "simple_economic_validation_forecasts.parquet",
        "summary": root / "simple_economic_validation_summary.csv",
    }
    result.state_panel.to_parquet(paths["state_panel"])
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    return paths


def _clean_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
