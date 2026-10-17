from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.states.economic import EconomicStateEngine
from world_model.states.market import MarketStateEngine, _forward_drawdown, _forward_return
from world_model.states.target_validation import relative_forward_return


@dataclass(frozen=True)
class CombinedTarget:
    name: str
    label: str
    kind: str
    horizon: str
    trading_days: int
    target: pd.Series


@dataclass(frozen=True)
class CombinedValidationResult:
    state_panel: pd.DataFrame
    forecasts: pd.DataFrame
    summary: pd.DataFrame


class CombinedStateValidator:
    def __init__(
        self,
        data_roots: dict[str, str],
        market_config: dict[str, Any],
        economic_config: dict[str, Any],
        combined_config: dict[str, Any],
    ):
        self.data_roots = data_roots
        self.market_config = market_config
        self.economic_config = economic_config
        self.combined_config = combined_config

    def run(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> CombinedValidationResult:
        market_engine = MarketStateEngine(self.data_roots, self.market_config)
        economic_engine = EconomicStateEngine(self.data_roots, self.economic_config)
        market_state = market_engine.build_state_history()
        economic_state = economic_engine.build_state_history()
        prices = market_engine.load_prices().ffill()
        state_panel = build_combined_state_panel(market_state=market_state, economic_state=economic_state)
        targets = build_combined_targets(
            state_panel=state_panel,
            prices=prices,
            horizons=self.market_config.get("horizons", {}),
            config=self.combined_config,
            target_asset=self.market_config.get("target_asset", "ES"),
        )
        forecasts = run_combined_forecasts(
            state_panel=state_panel,
            targets=targets,
            models=self.combined_config.get("models", []),
            config=self.combined_config,
            start=start,
            end=end,
        )
        summary = summarize_combined_forecasts(forecasts)
        return CombinedValidationResult(state_panel=state_panel, forecasts=forecasts, summary=summary)


def build_combined_state_panel(market_state: pd.DataFrame, economic_state: pd.DataFrame) -> pd.DataFrame:
    market = market_state.add_prefix("market_")
    economic = economic_state.add_prefix("economic_")
    panel = pd.concat([market, economic], axis=1).sort_index()
    panel["market_composite_score"] = panel["market_composite_score"].astype(float)
    panel["economic_composite_score"] = panel["economic_composite_score"].astype(float)
    panel["activity_score"] = panel["economic_activity_score"].astype(float)
    panel["inflation_pressure_score"] = panel["economic_inflation_pressure_score"].astype(float)
    panel["liquidity_credit_score"] = panel["economic_liquidity_credit_score"].astype(float)
    panel["economic_component_coverage"] = panel["economic_component_coverage"].astype(float)
    panel["market_component_coverage"] = panel["market_component_coverage"].astype(float)
    return panel


def build_combined_targets(
    state_panel: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: dict[str, int],
    config: dict[str, Any],
    target_asset: str = "ES",
) -> list[CombinedTarget]:
    if not horizons:
        raise ValueError("Market config must define horizons for combined validation.")
    targets: list[CombinedTarget] = []
    settings = config.get("targets", {})
    es = prices[target_asset].reindex(state_panel.index).ffill()
    market_composite = state_panel["market_composite_score"]
    economic_composite = state_panel["economic_composite_score"]
    risk_on_defensive = {
        label: relative_forward_return(
            prices.reindex(state_panel.index).ffill(),
            ["ES", "NQ", "CL", "HG"],
            ["ZN", "ZB", "GC"],
            int(days),
        )
        for label, days in horizons.items()
    }

    for label, days in horizons.items():
        days = int(days)
        forward_return = _forward_return(es, days)
        forward_drawdown = _forward_drawdown(es, days)
        future_market = market_composite.shift(-days)
        future_economic = economic_composite.shift(-days)

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
                target=(forward_drawdown <= float(settings.get("drawdown_threshold", -0.10))).astype(float).where(forward_drawdown.notna()),
            )
        )
        market_drop = future_market - market_composite <= float(settings.get("market_deterioration_threshold", -0.25))
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
        economic_drop = future_economic - economic_composite <= float(settings.get("economic_deterioration_threshold", -0.25))
        targets.append(
            CombinedTarget(
                name=f"economic_state_deterioration_{label}",
                label=f"Economic state deterioration {label}",
                kind="binary",
                horizon=label,
                trading_days=days,
                target=economic_drop.astype(float).where(future_economic.notna() & economic_composite.notna()),
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


def run_combined_forecasts(
    state_panel: pd.DataFrame,
    targets: list[CombinedTarget],
    models: list[dict[str, Any]],
    config: dict[str, Any],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not models:
        raise ValueError("Combined validation config must define at least one model.")
    max_horizon = max(target.trading_days for target in targets)
    common_index = state_panel.dropna(subset=["market_composite_score", "economic_composite_score"]).index
    for target in targets:
        common_index = common_index.intersection(target.target.dropna().index)
    eval_dates = evaluation_dates(
        common_index=pd.DatetimeIndex(common_index),
        max_horizon=max_horizon,
        frequency=config.get("evaluation_frequency", "ME"),
        start=start,
        end=end,
    )
    eval_frame = state_panel.reindex(eval_dates)
    min_train = int(config.get("min_train_observations", 84))
    rows: list[dict[str, Any]] = []

    for target in targets:
        target_values = target.target.reindex(eval_dates)
        contexts = target_contexts(eval_dates=eval_dates, target_values=target_values, target=target, min_train=min_train)
        for context in contexts:
            train_target = target_values.iloc[: context["train_end_pos"] + 1]
            train_features = eval_frame.iloc[: context["train_end_pos"] + 1]
            current_features = eval_frame.iloc[context["eval_pos"]]
            for model in models:
                forecast, bucket_count = model_forecast(
                    model=model,
                    train_features=train_features,
                    train_target=train_target,
                    current_features=current_features,
                    config=config,
                )
                if pd.isna(forecast):
                    continue
                rows.append(
                    {
                        "forecast_date": context["forecast_date"],
                        "target": target.name,
                        "target_label": target.label,
                        "target_kind": target.kind,
                        "horizon": target.horizon,
                        "trading_days": target.trading_days,
                        "model": model["id"],
                        "model_label": model.get("label", model["id"]),
                        "model_kind": model.get("kind", "unknown"),
                        "forecast_value": float(forecast),
                        "realized_value": context["realized"],
                        "train_observations": context["train_observations"],
                        "bucket_observations": bucket_count,
                    }
                )
    forecasts = pd.DataFrame(rows)
    if forecasts.empty:
        raise ValueError("Combined validation produced no forecasts.")
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
    return forecasts.sort_values(["target", "trading_days", "model", "forecast_date"]).reset_index(drop=True)


def target_contexts(
    eval_dates: pd.DatetimeIndex,
    target_values: pd.Series,
    target: CombinedTarget,
    min_train: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for eval_pos, forecast_date in enumerate(eval_dates):
        realized = target_values.iloc[eval_pos]
        if pd.isna(realized):
            continue
        train_cutoff = forecast_date - pd.tseries.offsets.BDay(target.trading_days)
        train_end_pos = int(eval_dates.searchsorted(train_cutoff, side="right")) - 1
        if train_end_pos < 0:
            continue
        train_observations = int(target_values.iloc[: train_end_pos + 1].notna().sum())
        if train_observations < min_train:
            continue
        contexts.append(
            {
                "forecast_date": forecast_date,
                "eval_pos": eval_pos,
                "train_end_pos": train_end_pos,
                "realized": float(realized),
                "train_observations": train_observations,
            }
        )
    return contexts


def model_forecast(
    model: dict[str, Any],
    train_features: pd.DataFrame,
    train_target: pd.Series,
    current_features: pd.Series,
    config: dict[str, Any],
) -> tuple[float, int]:
    sample = pd.concat([train_features, train_target.rename("target")], axis=1).dropna(subset=["target"])
    if sample.empty:
        return float("nan"), 0
    kind = model.get("kind")
    if kind == "unconditional":
        return float(sample["target"].mean()), int(len(sample))
    if kind == "single_bucket":
        feature = model["feature"]
        return single_bucket_forecast(
            train_feature=sample[feature],
            train_target=sample["target"],
            current_value=current_features.get(feature),
            bucket_count=int(config.get("bucket_count", 5)),
            min_bucket_observations=int(config.get("min_bucket_observations", 12)),
        )
    if kind == "joint_bucket":
        left = model["left_feature"]
        right = model["right_feature"]
        return joint_bucket_forecast(
            train_left=sample[left],
            train_right=sample[right],
            train_target=sample["target"],
            current_left=current_features.get(left),
            current_right=current_features.get(right),
            bucket_count=int(config.get("joint_bucket_count", 3)),
            min_bucket_observations=int(config.get("min_bucket_observations", 12)),
        )
    raise ValueError(f"Unsupported combined model kind: {kind}")


def single_bucket_forecast(
    train_feature: pd.Series,
    train_target: pd.Series,
    current_value: float,
    bucket_count: int,
    min_bucket_observations: int,
) -> tuple[float, int]:
    sample = pd.concat([train_feature.rename("feature"), train_target.rename("target")], axis=1).dropna()
    fallback = float(sample["target"].mean()) if not sample.empty else float("nan")
    if sample.empty or pd.isna(current_value) or sample["feature"].nunique() < 2:
        return fallback, int(len(sample))
    sample["bucket"] = pd.qcut(sample["feature"], bucket_count, labels=False, duplicates="drop")
    bucket = bucket_for_value(sample.groupby("bucket", observed=True)["feature"].agg(["min", "max"]), float(current_value))
    bucket_sample = sample[sample["bucket"].eq(bucket)]
    if len(bucket_sample) < min_bucket_observations:
        return fallback, int(len(bucket_sample))
    return float(bucket_sample["target"].mean()), int(len(bucket_sample))


def joint_bucket_forecast(
    train_left: pd.Series,
    train_right: pd.Series,
    train_target: pd.Series,
    current_left: float,
    current_right: float,
    bucket_count: int,
    min_bucket_observations: int,
) -> tuple[float, int]:
    sample = pd.concat(
        [train_left.rename("left"), train_right.rename("right"), train_target.rename("target")],
        axis=1,
    ).dropna()
    fallback = float(sample["target"].mean()) if not sample.empty else float("nan")
    if sample.empty or pd.isna(current_left) or pd.isna(current_right):
        return fallback, int(len(sample))
    if sample["left"].nunique() < 2 or sample["right"].nunique() < 2:
        return fallback, int(len(sample))

    sample["left_bucket"] = pd.qcut(sample["left"], bucket_count, labels=False, duplicates="drop")
    sample["right_bucket"] = pd.qcut(sample["right"], bucket_count, labels=False, duplicates="drop")
    left_bucket = bucket_for_value(sample.groupby("left_bucket", observed=True)["left"].agg(["min", "max"]), float(current_left))
    right_bucket = bucket_for_value(sample.groupby("right_bucket", observed=True)["right"].agg(["min", "max"]), float(current_right))
    bucket_sample = sample[sample["left_bucket"].eq(left_bucket) & sample["right_bucket"].eq(right_bucket)]
    if len(bucket_sample) < min_bucket_observations:
        return fallback, int(len(bucket_sample))
    return float(bucket_sample["target"].mean()), int(len(bucket_sample))


def summarize_combined_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, horizon, model), group in forecasts.groupby(["target", "horizon", "model"], sort=False):
        clean = group.dropna(subset=["forecast_value", "realized_value"])
        if clean.empty:
            continue
        errors = clean["forecast_value"] - clean["realized_value"]
        rows.append(
            {
                "target": target,
                "target_label": clean["target_label"].iloc[0],
                "target_kind": clean["target_kind"].iloc[0],
                "horizon": horizon,
                "trading_days": int(clean["trading_days"].iloc[0]),
                "model": model,
                "model_label": clean["model_label"].iloc[0],
                "model_kind": clean["model_kind"].iloc[0],
                "observations": int(len(clean)),
                "realized_mean": float(clean["realized_value"].mean()),
                "forecast_mean": float(clean["forecast_value"].mean()),
                "error_metric": float(np.sqrt(np.mean(errors.pow(2)))) if clean["target_kind"].iloc[0] == "continuous" else float(np.mean(errors.pow(2))),
                "forecast_realized_corr": _safe_corr(clean["forecast_value"], clean["realized_value"]),
                "q5_q1_realized_spread": _top_bottom_spread(clean),
                "avg_bucket_observations": float(clean["bucket_observations"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    baselines = summary[["target", "horizon", "model", "error_metric"]].copy()
    unconditional = baselines[baselines["model"].eq("unconditional")].rename(columns={"error_metric": "unconditional_error_metric"})
    market = baselines[baselines["model"].eq("market_bucket")].rename(columns={"error_metric": "market_bucket_error_metric"})
    economic = baselines[baselines["model"].eq("economic_bucket")].rename(columns={"error_metric": "economic_bucket_error_metric"})
    summary = summary.merge(unconditional[["target", "horizon", "unconditional_error_metric"]], on=["target", "horizon"], how="left")
    summary = summary.merge(market[["target", "horizon", "market_bucket_error_metric"]], on=["target", "horizon"], how="left")
    summary = summary.merge(economic[["target", "horizon", "economic_bucket_error_metric"]], on=["target", "horizon"], how="left")
    summary["improvement_vs_unconditional"] = summary["unconditional_error_metric"] - summary["error_metric"]
    summary["improvement_vs_market_bucket"] = summary["market_bucket_error_metric"] - summary["error_metric"]
    summary["improvement_vs_economic_bucket"] = summary["economic_bucket_error_metric"] - summary["error_metric"]
    return summary.sort_values(["target", "trading_days", "model"]).reset_index(drop=True)


def evaluation_dates(
    common_index: pd.DatetimeIndex,
    max_horizon: int,
    frequency: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    usable_index = pd.DatetimeIndex(common_index).sort_values()
    if usable_index.empty:
        raise ValueError("No common index available for combined validation.")
    start_ts = pd.Timestamp(start).normalize() if start is not None else usable_index.min()
    end_ts = pd.Timestamp(end).normalize() if end is not None else usable_index.max() - pd.tseries.offsets.BDay(max_horizon)
    candidate_dates = pd.date_range(start_ts, end_ts, freq=frequency)
    selected: list[pd.Timestamp] = []
    for candidate in candidate_dates:
        prior = usable_index[usable_index <= candidate]
        if not prior.empty:
            selected.append(pd.Timestamp(prior[-1]))
    return pd.DatetimeIndex(sorted(set(selected)))


def write_combined_validation_outputs(result: CombinedValidationResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_panel": root / "combined_state_panel.parquet",
        "forecasts": root / "combined_state_validation_forecasts.parquet",
        "summary": root / "combined_state_validation_summary.csv",
    }
    result.state_panel.to_parquet(paths["state_panel"])
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    return paths


def bucket_for_value(bucket_edges: pd.DataFrame, value: float) -> int:
    containing = bucket_edges[(bucket_edges["min"] <= value) & (value <= bucket_edges["max"])]
    if not containing.empty:
        return int(containing.index[0])
    distances = (bucket_edges.mean(axis=1) - value).abs()
    return int(distances.idxmin())


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1]))


def _top_bottom_spread(clean: pd.DataFrame) -> float:
    if clean["forecast_value"].nunique() < 5:
        return float("nan")
    bucketed = clean.copy()
    bucketed["_quintile"] = pd.qcut(bucketed["forecast_value"], 5, labels=False, duplicates="drop")
    realized_by_quintile = bucketed.groupby("_quintile")["realized_value"].mean()
    if len(realized_by_quintile) < 2:
        return float("nan")
    return float(realized_by_quintile.iloc[-1] - realized_by_quintile.iloc[0])
