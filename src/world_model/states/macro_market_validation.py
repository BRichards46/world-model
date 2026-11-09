from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.states.economic_asset_drivers import (
    evaluation_dates,
    forward_cumulative_returns,
    load_asset_group_returns,
    walk_forward_contexts,
)


@dataclass(frozen=True)
class MacroMarketValidationResult:
    macro_feature_panel: pd.DataFrame
    feature_metadata: pd.DataFrame
    asset_group_returns: pd.DataFrame
    market_target_panel: pd.DataFrame
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    return_method: str


class MacroMarketValidator:
    def __init__(
        self,
        data_roots: dict[str, str],
        asset_config: dict[str, Any],
        validation_config: dict[str, Any],
    ):
        self.data_roots = data_roots
        self.asset_config = asset_config
        self.validation_config = validation_config

    def run(
        self,
        macro_history: pd.DataFrame,
        market_state_history: pd.DataFrame,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> MacroMarketValidationResult:
        return_method = str(self.asset_config.get("return_method", "normalized_price_change"))
        asset_returns = load_asset_group_returns(
            root=self.data_roots["csi_processed"],
            asset_groups=self.asset_config.get("asset_groups", {}),
            return_method=return_method,
            normalization_window=int(self.asset_config.get("normalization_window", 252)),
            normalization_min_periods=int(self.asset_config.get("normalization_min_periods", 63)),
        )
        macro_panel, metadata = build_macro_feature_panel(
            macro_history,
            daily_index=asset_returns.index,
            metrics=list(self.validation_config.get("features", [])),
            include_context=bool(self.validation_config.get("include_context", True)),
        )
        forecasts, target_panel = run_macro_market_forecasts(
            macro_panel=macro_panel,
            feature_metadata=metadata,
            asset_returns=asset_returns,
            market_state_history=market_state_history,
            config=self.validation_config,
            asset_labels={
                key: str(value.get("label", key))
                for key, value in self.asset_config.get("asset_groups", {}).items()
            },
            start=start,
            end=end,
        )
        summary = summarize_macro_market_forecasts(forecasts, self.validation_config)
        return MacroMarketValidationResult(
            macro_feature_panel=macro_panel,
            feature_metadata=metadata,
            asset_group_returns=asset_returns,
            market_target_panel=target_panel,
            forecasts=forecasts,
            summary=summary,
            return_method=return_method,
        )


def individual_asset_config(asset_config: dict[str, Any]) -> dict[str, Any]:
    """Expand configured asset classes into one validation target per contract."""
    expanded = copy.deepcopy(asset_config)
    individual: dict[str, Any] = {}
    for parent_id, settings in asset_config.get("asset_groups", {}).items():
        parent_label = str(settings.get("label", parent_id))
        for symbol, filename in settings.get("assets", {}).items():
            asset_id = f"{parent_id}__{str(symbol).lower()}"
            individual[asset_id] = {
                "label": f"{symbol} · {parent_label}",
                "parent_asset_group": str(parent_id),
                "assets": {str(symbol): filename},
            }
    if not individual:
        raise ValueError("No individual assets are configured.")
    expanded["asset_groups"] = individual
    return expanded


def build_macro_feature_panel(
    macro_history: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    metrics: list[str],
    include_context: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "realtime_start",
        "indicator_id",
        "label",
        "component",
        "polarity",
        "is_context",
        "level_percentile_10y",
        "velocity_percentile_10y",
    }
    missing = sorted(required - set(macro_history.columns))
    if missing:
        raise ValueError(f"Macro history is missing required columns: {', '.join(missing)}")
    frame = macro_history.copy()
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
    frame = frame.dropna(subset=["realtime_start", "indicator_id"])
    if not include_context:
        frame = frame.loc[~frame["is_context"].fillna(False).astype(bool)]
    requested = [metric for metric in metrics if metric != "level_velocity_interaction"]
    missing_metrics = sorted(set(requested) - set(frame.columns))
    if missing_metrics:
        raise ValueError(f"Macro history is missing requested metrics: {', '.join(missing_metrics)}")

    index = pd.DatetimeIndex(daily_index).sort_values().unique()
    columns: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []
    for indicator_id, group in frame.groupby("indicator_id", sort=False):
        group = group.sort_values("realtime_start").drop_duplicates("realtime_start", keep="last")
        identity = group.iloc[-1]
        values_by_metric: dict[str, pd.Series] = {}
        for metric in requested:
            series = pd.Series(
                pd.to_numeric(group[metric], errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(group["realtime_start"]),
                dtype=float,
            )
            expanded = series.reindex(index.union(series.index)).sort_index().ffill().reindex(index)
            values_by_metric[metric] = expanded
        if "level_velocity_interaction" in metrics:
            level = values_by_metric.get("level_percentile_10y")
            velocity = values_by_metric.get("velocity_percentile_10y")
            if level is None or velocity is None:
                raise ValueError("level_velocity_interaction requires both 10-year percentile features")
            values_by_metric["level_velocity_interaction"] = ((level - 50.0) / 50.0) * ((velocity - 50.0) / 50.0)

        for metric, series in values_by_metric.items():
            feature = f"{indicator_id}__{metric}"
            columns[feature] = series.rename(feature)
            metadata_rows.append(
                {
                    "feature": feature,
                    "indicator_id": str(indicator_id),
                    "indicator_label": str(identity["label"]),
                    "component": str(identity["component"]),
                    "polarity": str(identity["polarity"]),
                    "is_context": bool(identity["is_context"]),
                    "metric": metric,
                    "feature_label": f"{identity['label']} · {metric.replace('_', ' ')}",
                }
            )
    if not columns:
        raise ValueError("No macro validation features could be built.")
    panel = pd.concat(columns.values(), axis=1).sort_index()
    metadata = pd.DataFrame.from_records(metadata_rows).drop_duplicates("feature").set_index("feature")
    return panel, metadata


def run_macro_market_forecasts(
    macro_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    asset_returns: pd.DataFrame,
    market_state_history: pd.DataFrame,
    config: dict[str, Any],
    asset_labels: dict[str, str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizons = {key: int(value) for key, value in config.get("horizons", {}).items()}
    target_names = [str(value) for value in config.get("targets", ["forward_return"])]
    if not horizons:
        raise ValueError("Macro market validation requires at least one horizon.")
    unsupported = sorted(set(target_names) - {"forward_return", "future_volatility", "max_drawdown"})
    if unsupported:
        raise ValueError(f"Unsupported market targets: {', '.join(unsupported)}")

    market_column = str(config.get("market_baseline_column", "composite_score"))
    if market_column not in market_state_history:
        raise ValueError(f"Market state history is missing baseline column: {market_column}")
    market_last = pd.to_numeric(market_state_history[market_column], errors="coerce").dropna().index.max()
    requested_end = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp(market_last).normalize()
    effective_end = min(requested_end, pd.Timestamp(market_last).normalize())
    min_train = int(config.get("min_train_observations", 24))
    rows: list[dict[str, Any]] = []
    target_columns: dict[str, pd.Series] = {}

    for asset_group in asset_returns.columns:
        group_returns = asset_returns[[asset_group]].dropna()
        common_index = macro_panel.dropna(how="all").index.intersection(group_returns.index)
        eval_dates = evaluation_dates(
            common_index,
            max_horizon=max(horizons.values()),
            frequency=str(config.get("evaluation_frequency", "ME")),
            start=start,
            end=effective_end,
        )
        eval_features = macro_panel.reindex(eval_dates)
        market_series = _market_baseline_series(market_state_history, market_column, eval_dates)
        for horizon, trading_days in horizons.items():
            targets = build_forward_market_targets(group_returns, trading_days, target_names)
            for target_name, target_frame in targets.items():
                target_values = target_frame[asset_group].reindex(eval_dates)
                target_key = f"{asset_group}__{horizon}__{target_name}"
                target_columns[target_key] = target_values.rename(target_key)
                contexts = walk_forward_contexts(eval_dates, target_values, trading_days, min_train)
                for feature in eval_features.columns:
                    meta = feature_metadata.loc[feature]
                    feature_values = eval_features[feature]
                    feature_rank = expanding_rank(feature_values)
                    market_rank = expanding_rank(market_series)
                    macro_predictions = cumulative_linear_forecasts(
                        pd.DataFrame({"feature": feature_rank}),
                        target_values,
                        contexts,
                        min_train=min_train,
                    )
                    comparison_mask = feature_rank.notna()
                    market_predictions = cumulative_linear_forecasts(
                        pd.DataFrame({"market": market_rank.where(comparison_mask)}),
                        target_values,
                        contexts,
                        min_train=min_train,
                    )
                    joint_predictions = cumulative_linear_forecasts(
                        pd.DataFrame({"market": market_rank, "feature": feature_rank}),
                        target_values,
                        contexts,
                        min_train=min_train,
                    )
                    for context in contexts:
                        eval_pos = int(context["eval_pos"])
                        current_feature = feature_values.iloc[eval_pos]
                        macro_forecast, macro_count = macro_predictions.get(eval_pos, (float("nan"), 0))
                        if pd.isna(current_feature) or pd.isna(macro_forecast):
                            continue
                        market_forecast, market_count = market_predictions.get(eval_pos, (float("nan"), 0))
                        joint_forecast, joint_count = joint_predictions.get(eval_pos, (float("nan"), 0))
                        rows.append(
                            {
                                "forecast_date": context["forecast_date"],
                                "asset_group": asset_group,
                                "asset_group_label": asset_labels.get(asset_group, asset_group),
                                "horizon": horizon,
                                "trading_days": trading_days,
                                "target": target_name,
                                "feature": feature,
                                "feature_value": float(current_feature),
                                "indicator_id": meta["indicator_id"],
                                "indicator_label": meta["indicator_label"],
                                "component": meta["component"],
                                "metric": meta["metric"],
                                "polarity": meta["polarity"],
                                "is_context": bool(meta["is_context"]),
                                "train_observations": int(macro_count),
                                "market_train_observations": int(min(market_count, joint_count)),
                                "realized_target": float(context["realized"]),
                                "unconditional_forecast": float(context["baseline"]),
                                "macro_forecast": macro_forecast,
                                "market_forecast": market_forecast,
                                "joint_forecast": joint_forecast,
                            }
                        )
    forecasts = pd.DataFrame.from_records(rows)
    if forecasts.empty:
        raise ValueError("Macro market validation produced no out-of-sample forecasts.")
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
    target_panel = pd.concat(target_columns.values(), axis=1).sort_index() if target_columns else pd.DataFrame()
    return forecasts.sort_values(["asset_group", "target", "horizon", "feature", "forecast_date"]).reset_index(drop=True), target_panel


def build_forward_market_targets(
    returns: pd.DataFrame,
    horizon_days: int,
    targets: list[str],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    if "forward_return" in targets:
        result["forward_return"] = _columnwise_forward_target(
            returns,
            lambda frame: forward_cumulative_returns(frame, horizon_days),
        )
    if "future_volatility" in targets:
        result["future_volatility"] = _columnwise_forward_target(
            returns,
            lambda frame: frame.shift(-1).rolling(horizon_days, min_periods=horizon_days).std().shift(-(horizon_days - 1)) * math.sqrt(252.0),
        )
    if "max_drawdown" in targets:
        result["max_drawdown"] = _columnwise_forward_target(
            returns,
            lambda frame: forward_max_drawdown(frame, horizon_days),
        )
    return result


def _columnwise_forward_target(
    returns: pd.DataFrame,
    builder: Any,
) -> pd.DataFrame:
    columns: list[pd.Series] = []
    for column in returns.columns:
        valid = returns[[column]].dropna()
        if valid.empty:
            continue
        columns.append(builder(valid)[column].rename(column))
    return pd.concat(columns, axis=1).sort_index() if columns else pd.DataFrame(index=returns.index)


def forward_max_drawdown(returns: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    values = returns.to_numpy(dtype=float)
    output = np.full_like(values, np.nan, dtype=float)
    for position in range(len(returns) - horizon_days):
        window = values[position + 1 : position + horizon_days + 1]
        for column in range(values.shape[1]):
            series = window[:, column]
            if np.isnan(series).any():
                continue
            cumulative = np.concatenate([[0.0], np.cumsum(series)])
            peaks = np.maximum.accumulate(cumulative)
            output[position, column] = float(np.min(cumulative - peaks))
    return pd.DataFrame(output, index=returns.index, columns=returns.columns)


def rank_linear_forecast(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    current_features: pd.Series,
) -> float:
    sample = pd.concat([train_features, train_target.rename("target")], axis=1).dropna()
    if len(sample) < 3:
        return float("nan")
    ranked = sample[train_features.columns].rank(method="average", pct=True)
    current_rank = []
    for column in train_features.columns:
        values = sample[column]
        current_rank.append(float((values.le(float(current_features[column])).sum() + 0.5) / (len(values) + 1.0)))
    design = np.column_stack([np.ones(len(ranked)), ranked.to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, sample["target"].to_numpy(dtype=float), rcond=None)
    return float(np.dot(np.array([1.0, *current_rank]), coefficients))


def expanding_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    observed: list[float] = []
    for position, value in enumerate(numeric.to_numpy(dtype=float)):
        if np.isnan(value):
            continue
        observed.append(float(value))
        array = np.asarray(observed, dtype=float)
        less = float(np.sum(array < value))
        equal = float(np.sum(array == value))
        output.iloc[position] = (less + 0.5 * equal) / len(array)
    return output


def cumulative_linear_forecasts(
    predictors: pd.DataFrame,
    target: pd.Series,
    contexts: list[dict[str, Any]],
    min_train: int,
) -> dict[int, tuple[float, int]]:
    aligned = predictors.copy()
    aligned["_target"] = target
    predictor_columns = list(predictors.columns)
    valid = aligned[predictor_columns + ["_target"]].notna().all(axis=1).to_numpy()
    x_values = aligned[predictor_columns].to_numpy(dtype=float)
    y_values = aligned["_target"].to_numpy(dtype=float)
    width = len(predictor_columns) + 1
    cumulative_xx = np.zeros((len(aligned), width, width), dtype=float)
    cumulative_xy = np.zeros((len(aligned), width), dtype=float)
    cumulative_count = np.zeros(len(aligned), dtype=int)
    xx = np.zeros((width, width), dtype=float)
    xy = np.zeros(width, dtype=float)
    count = 0
    for position in range(len(aligned)):
        if valid[position]:
            row = np.array([1.0, *x_values[position]], dtype=float)
            xx += np.outer(row, row)
            xy += row * y_values[position]
            count += 1
        cumulative_xx[position] = xx
        cumulative_xy[position] = xy
        cumulative_count[position] = count

    forecasts: dict[int, tuple[float, int]] = {}
    for context in contexts:
        eval_pos = int(context["eval_pos"])
        train_end = int(context["train_end_pos"])
        train_count = int(cumulative_count[train_end])
        if train_count < min_train or not predictors.iloc[eval_pos].notna().all():
            continue
        coefficients = np.linalg.pinv(cumulative_xx[train_end]) @ cumulative_xy[train_end]
        current = np.array([1.0, *x_values[eval_pos]], dtype=float)
        forecasts[eval_pos] = (float(current @ coefficients), train_count)
    return forecasts


def summarize_macro_market_forecasts(
    forecasts: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["asset_group", "target", "horizon", "feature"]
    for _, group in forecasts.groupby(keys, sort=False):
        clean = group.dropna(subset=["feature_value", "realized_target", "macro_forecast", "unconditional_forecast"])
        if clean.empty:
            continue
        market_clean = clean.dropna(subset=["market_forecast", "joint_forecast"])
        midpoint = max(1, len(clean) // 2)
        early = clean.iloc[:midpoint]
        late = clean.iloc[midpoint:]
        corr = _safe_corr(clean["feature_value"], clean["realized_target"])
        early_corr = _safe_corr(early["feature_value"], early["realized_target"])
        late_corr = _safe_corr(late["feature_value"], late["realized_target"])
        overlap = max(1.0, float(clean["trading_days"].iloc[0]) / 21.0)
        p_value = approximate_correlation_p_value(corr, len(clean), overlap)
        macro_error = clean["macro_forecast"] - clean["realized_target"]
        unconditional_error = clean["unconditional_forecast"] - clean["realized_target"]
        market_mse = float("nan")
        joint_mse = float("nan")
        incremental = float("nan")
        if not market_clean.empty:
            market_error = market_clean["market_forecast"] - market_clean["realized_target"]
            joint_error = market_clean["joint_forecast"] - market_clean["realized_target"]
            market_mse = float(np.mean(market_error.pow(2)))
            joint_mse = float(np.mean(joint_error.pow(2)))
            incremental = market_mse - joint_mse
        rows.append(
            {
                "asset_group": clean["asset_group"].iloc[0],
                "asset_group_label": clean["asset_group_label"].iloc[0],
                "target": clean["target"].iloc[0],
                "horizon": clean["horizon"].iloc[0],
                "trading_days": int(clean["trading_days"].iloc[0]),
                "feature": clean["feature"].iloc[0],
                "indicator_id": clean["indicator_id"].iloc[0],
                "indicator_label": clean["indicator_label"].iloc[0],
                "component": clean["component"].iloc[0],
                "metric": clean["metric"].iloc[0],
                "polarity": clean["polarity"].iloc[0],
                "is_context": bool(clean["is_context"].iloc[0]),
                "observations": int(len(clean)),
                "sample_start": clean["forecast_date"].min(),
                "sample_end": clean["forecast_date"].max(),
                "feature_realized_corr": corr,
                "early_corr": early_corr,
                "late_corr": late_corr,
                "stable_sign": bool(pd.notna(early_corr) and pd.notna(late_corr) and np.sign(early_corr) == np.sign(late_corr)),
                "correlation_p_value_approx": p_value,
                "q5_q1_realized_spread": _top_bottom_spread(clean),
                "quintile_monotonicity": _quintile_monotonicity(clean),
                "macro_forecast_realized_corr": _safe_corr(clean["macro_forecast"], clean["realized_target"]),
                "unconditional_mse": float(np.mean(unconditional_error.pow(2))),
                "macro_mse": float(np.mean(macro_error.pow(2))),
                "macro_improvement_vs_unconditional": float(np.mean(unconditional_error.pow(2)) - np.mean(macro_error.pow(2))),
                "market_observations": int(len(market_clean)),
                "market_mse": market_mse,
                "joint_mse": joint_mse,
                "joint_improvement_vs_market": incremental,
            }
        )
    summary = pd.DataFrame.from_records(rows)
    if summary.empty:
        return summary
    summary["fdr_q_value_approx"] = benjamini_hochberg(summary["correlation_p_value_approx"])
    minimum = int(config.get("minimum_oos_observations", 24))
    alpha = float(config.get("fdr_alpha", 0.10))
    summary["robust_candidate"] = (
        summary["observations"].ge(minimum)
        & summary["stable_sign"]
        & summary["fdr_q_value_approx"].le(alpha)
        & summary["joint_improvement_vs_market"].gt(0.0)
    )
    return summary.sort_values(
        ["robust_candidate", "joint_improvement_vs_market", "feature_realized_corr"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def approximate_correlation_p_value(correlation: float, observations: int, overlap_factor: float = 1.0) -> float:
    if pd.isna(correlation) or observations < 4:
        return float("nan")
    effective = max(4.0, observations / max(1.0, overlap_factor))
    clipped = float(np.clip(correlation, -0.999999, 0.999999))
    z = abs(np.arctanh(clipped)) * math.sqrt(max(1.0, effective - 3.0))
    return float(math.erfc(z / math.sqrt(2.0)))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def write_macro_market_validation_outputs(
    result: MacroMarketValidationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": root / "macro_market_feature_panel.parquet",
        "feature_metadata": root / "macro_market_feature_metadata.csv",
        "asset_returns": root / "macro_market_asset_group_returns.parquet",
        "targets": root / "macro_market_target_panel.parquet",
        "forecasts": root / "macro_market_forecasts.parquet",
        "summary": root / "macro_market_summary.csv",
    }
    result.macro_feature_panel.to_parquet(paths["features"])
    result.feature_metadata.reset_index().to_csv(paths["feature_metadata"], index=False)
    result.asset_group_returns.to_parquet(paths["asset_returns"])
    result.market_target_panel.to_parquet(paths["targets"])
    result.forecasts.to_parquet(paths["forecasts"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    return paths


def _market_baseline_series(history: pd.DataFrame, column: str, eval_dates: pd.DatetimeIndex) -> pd.Series:
    if column not in history:
        raise ValueError(f"Market state history is missing baseline column: {column}")
    values = pd.to_numeric(history[column], errors="coerce")
    values.index = pd.to_datetime(history.index, errors="coerce")
    values = values.loc[~values.index.isna()].sort_index()
    return values.reindex(values.index.union(eval_dates)).sort_index().ffill().reindex(eval_dates).rename(column)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1], method="spearman"))


def _top_bottom_spread(clean: pd.DataFrame) -> float:
    if clean["feature_value"].nunique() < 5:
        return float("nan")
    bucketed = clean.copy()
    bucketed["_quintile"] = pd.qcut(bucketed["feature_value"], 5, labels=False, duplicates="drop")
    means = bucketed.groupby("_quintile", observed=True)["realized_target"].mean()
    return float(means.iloc[-1] - means.iloc[0]) if len(means) >= 2 else float("nan")


def _quintile_monotonicity(clean: pd.DataFrame) -> float:
    if clean["feature_value"].nunique() < 5:
        return float("nan")
    bucketed = clean.copy()
    bucketed["_quintile"] = pd.qcut(bucketed["feature_value"], 5, labels=False, duplicates="drop")
    means = bucketed.groupby("_quintile", observed=True)["realized_target"].mean()
    if len(means) < 3:
        return float("nan")
    return float(pd.Series(means.index, dtype=float).corr(pd.Series(means.to_numpy()), method="spearman"))
