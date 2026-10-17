from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from world_model.data.csi import load_price_panel
from world_model.features.market import momentum_score, trend_score, weighted_average
from world_model.states.combined import (
    CombinedTarget,
    CombinedValidationResult,
    run_combined_forecasts,
    summarize_combined_forecasts,
)
from world_model.states.market import MarketStateEngine, _forward_drawdown, _forward_return
from world_model.states.target_validation import relative_forward_return


SIMPLE_BLOCK_COLUMNS = ["equity_score", "rates_score", "commodities_score", "fx_score"]


@dataclass(frozen=True)
class SimpleMarketSnapshot:
    as_of_date: pd.Timestamp
    regime_label: str
    simple_composite_score: float
    block_scores: dict[str, float]
    block_coverage: float


class SimpleMarketRegimeEngine:
    def __init__(self, data_roots: dict[str, str], config: dict[str, Any]):
        self.data_roots = data_roots
        self.config = config

    def load_prices(self) -> pd.DataFrame:
        return load_price_panel(self.data_roots["csi_processed"], self.config["instruments"])

    def build_state_history(self) -> pd.DataFrame:
        prices = self.load_prices()
        asset_scores = build_asset_direction_scores(prices)
        state = pd.DataFrame(index=prices.index)

        for block, settings in self.config["blocks"].items():
            assets = settings.get("assets", [])
            column = f"{block}_score"
            block_score = asset_scores[assets].mean(axis=1)
            if settings.get("invert", False):
                block_score = -block_score
            state[column] = block_score

        weights = self._block_weights()
        state["block_coverage"] = state[SIMPLE_BLOCK_COLUMNS].notna().mean(axis=1)
        state["simple_composite_score"] = weighted_average(state[SIMPLE_BLOCK_COLUMNS], weights)
        state["regime_label"] = state.apply(label_simple_regime, axis=1)
        return state

    def snapshot(
        self,
        as_of: str | pd.Timestamp | None = None,
        state: pd.DataFrame | None = None,
    ) -> SimpleMarketSnapshot:
        state = self.build_state_history() if state is None else state
        row = row_as_of(state, as_of)
        return SimpleMarketSnapshot(
            as_of_date=pd.Timestamp(row.name),
            regime_label=str(row["regime_label"]),
            simple_composite_score=_clean_float(row["simple_composite_score"]),
            block_scores={column: _clean_float(row[column]) for column in SIMPLE_BLOCK_COLUMNS},
            block_coverage=_clean_float(row["block_coverage"]),
        )

    def validate_against_market_state(
        self,
        market_config: dict[str, Any],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> CombinedValidationResult:
        simple_state = self.build_state_history()
        prices = self.load_prices().ffill()
        detailed_state = MarketStateEngine(self.data_roots, market_config).build_state_history()
        panel = build_simple_validation_panel(simple_state=simple_state, detailed_state=detailed_state)
        targets = build_simple_validation_targets(
            simple_state=simple_state,
            prices=prices,
            horizons=self.config.get("horizons", {}),
            config=self.config.get("validation", {}),
            target_asset=self.config.get("target_asset", "ES"),
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
            summary=summarize_simple_forecasts(forecasts),
        )

    def _block_weights(self) -> dict[str, float]:
        weights = {}
        for block, settings in self.config["blocks"].items():
            weights[f"{block}_score"] = float(settings.get("weight", 1.0))
        return weights


def build_asset_direction_scores(prices: pd.DataFrame) -> pd.DataFrame:
    trend = prices.apply(trend_score)
    momentum = prices.apply(momentum_score)
    return pd.concat([trend.stack(), momentum.stack()], axis=1).mean(axis=1).unstack()


def build_simple_validation_panel(simple_state: pd.DataFrame, detailed_state: pd.DataFrame) -> pd.DataFrame:
    panel = simple_state.copy()
    panel["detailed_market_composite_score"] = detailed_state["composite_score"].reindex(panel.index).ffill()
    panel["market_composite_score"] = panel["detailed_market_composite_score"]
    panel["economic_composite_score"] = panel["simple_composite_score"]
    return panel


def build_simple_validation_targets(
    simple_state: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: dict[str, int],
    config: dict[str, Any],
    target_asset: str = "ES",
) -> list[CombinedTarget]:
    targets: list[CombinedTarget] = []
    if not horizons:
        raise ValueError("Simple market regime config must define horizons.")
    target_close = prices[target_asset].reindex(simple_state.index).ffill()
    simple_composite = simple_state["simple_composite_score"]
    risk_on_defensive = {
        label: relative_forward_return(
            prices.reindex(simple_state.index).ffill(),
            ["ES", "NQ", "CL", "HG"],
            ["ZN", "ZB", "GC"],
            int(days),
        )
        for label, days in horizons.items()
    }

    for label, days in horizons.items():
        days = int(days)
        forward_return = _forward_return(target_close, days)
        forward_drawdown = _forward_drawdown(target_close, days)
        future_simple = simple_composite.shift(-days)
        deterioration = future_simple - simple_composite <= float(config.get("simple_deterioration_threshold", -0.25))

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
                name=f"simple_regime_deterioration_{label}",
                label=f"Simple regime deterioration {label}",
                kind="binary",
                horizon=label,
                trading_days=days,
                target=deterioration.astype(float).where(future_simple.notna() & simple_composite.notna()),
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


def summarize_simple_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_combined_forecasts(forecasts)
    if summary.empty:
        return summary
    detailed = summary[summary["model"].eq("detailed_market_bucket")][
        ["target", "horizon", "error_metric"]
    ].rename(columns={"error_metric": "detailed_market_error_metric"})
    simple = summary[summary["model"].eq("simple_composite_bucket")][
        ["target", "horizon", "error_metric"]
    ].rename(columns={"error_metric": "simple_composite_error_metric"})
    summary = summary.merge(detailed, on=["target", "horizon"], how="left")
    summary = summary.merge(simple, on=["target", "horizon"], how="left")
    summary["improvement_vs_detailed_market"] = summary["detailed_market_error_metric"] - summary["error_metric"]
    summary["improvement_vs_simple_composite"] = summary["simple_composite_error_metric"] - summary["error_metric"]
    return summary.sort_values(["target", "trading_days", "model"]).reset_index(drop=True)


def label_simple_regime(row: pd.Series) -> str:
    if pd.isna(row.get("simple_composite_score")):
        return "insufficient_data"
    parts = [
        _label_block(row.get("equity_score"), "equity_up", "equity_down"),
        _label_block(row.get("rates_score"), "rates_bid", "rates_pressure"),
        _label_block(row.get("commodities_score"), "commodity_bid", "commodity_soft"),
        _label_block(row.get("fx_score"), "dollar_weak", "dollar_strong"),
    ]
    return "__".join(parts)


def row_as_of(state: pd.DataFrame, as_of: str | pd.Timestamp | None) -> pd.Series:
    usable = state.dropna(subset=["simple_composite_score"])
    if usable.empty:
        raise ValueError("No usable simple market regime rows.")
    if as_of is None:
        return usable.iloc[-1]
    as_of_ts = pd.Timestamp(as_of).normalize()
    prior = usable.loc[usable.index <= as_of_ts]
    if prior.empty:
        raise ValueError(f"No simple market regime available on or before {as_of_ts.date()}.")
    return prior.iloc[-1]


def write_simple_state_outputs(state: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        state.to_parquet(path)
    elif path.suffix == ".csv":
        state.to_csv(path)
    else:
        raise ValueError("State output path must end in .parquet or .csv")


def write_simple_validation_outputs(result: CombinedValidationResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_panel": root / "simple_market_validation_panel.parquet",
        "forecasts": root / "simple_market_validation_forecasts.parquet",
        "summary": root / "simple_market_validation_summary.csv",
    }
    result.state_panel.to_parquet(paths["state_panel"])
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    return paths


def _label_block(value: float, positive_label: str, negative_label: str) -> str:
    if pd.isna(value):
        return "unknown"
    if value >= 0.20:
        return positive_label
    if value <= -0.20:
        return negative_label
    return "mixed"


def _clean_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
