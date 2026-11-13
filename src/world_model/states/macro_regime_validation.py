from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.states.economic_asset_drivers import (
    evaluation_dates,
    load_asset_group_returns,
    walk_forward_contexts,
)
from world_model.states.macro_market_validation import (
    approximate_correlation_p_value,
    benjamini_hochberg,
    build_forward_market_targets,
    build_macro_feature_panel,
    cumulative_linear_forecasts,
    expanding_rank,
)


@dataclass(frozen=True)
class MacroRegimeValidationResult:
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    combination_panel: pd.DataFrame


def build_combination_panel(
    macro_history: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    definitions: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build pre-specified binary regimes from point-in-time macro histories."""
    requested_metrics = sorted(
        {
            str(definition[metric])
            for definition in definitions
            for metric in ("growth_metric", "inflation_metric")
        }
        | {"level_percentile_10y", "velocity_percentile_10y"}
    )
    panel, _ = build_macro_feature_panel(
        macro_history,
        daily_index=daily_index,
        metrics=requested_metrics,
        include_context=True,
    )
    output: dict[tuple[str, str], pd.Series] = {}
    for definition in definitions:
        combination_id = str(definition["id"])
        growth_column = f"{definition['growth_indicator']}__{definition['growth_metric']}"
        inflation_column = f"{definition['inflation_indicator']}__{definition['inflation_metric']}"
        missing = [column for column in (growth_column, inflation_column) if column not in panel]
        if missing:
            raise ValueError(f"Combination {combination_id} is missing features: {', '.join(missing)}")

        # Use actual direction for slowing/rising, not a historical-median
        # proxy. Elevated inflation is deliberately a stricter 60th-percentile
        # condition. The product is therefore a true two-condition regime.
        growth_weakness = panel[growth_column].lt(0.0).where(panel[growth_column].notna()).astype(float)
        condition = str(definition["inflation_condition"])
        if condition == "rising":
            inflation_pressure = panel[inflation_column].gt(0.0).where(panel[inflation_column].notna()).astype(float)
        elif condition == "elevated":
            inflation_pressure = panel[inflation_column].ge(60.0).where(panel[inflation_column].notna()).astype(float)
        else:
            raise ValueError(f"Unsupported inflation condition for {combination_id}: {condition}")
        output[(combination_id, "growth_weakness")] = growth_weakness
        output[(combination_id, "inflation_pressure")] = inflation_pressure
        output[(combination_id, "interaction_score")] = growth_weakness * inflation_pressure

    combined = pd.concat(output, axis=1).sort_index()
    combined.columns = pd.MultiIndex.from_tuples(combined.columns, names=["combination_id", "metric"])
    return combined


def run_macro_regime_validation(
    macro_history: pd.DataFrame,
    market_state_history: pd.DataFrame,
    data_roots: dict[str, str],
    asset_config: dict[str, Any],
    config: dict[str, Any],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> MacroRegimeValidationResult:
    asset_group = str(config.get("asset_group", "equity"))
    configured_groups = asset_config.get("asset_groups", {})
    if asset_group not in configured_groups:
        raise ValueError(f"Asset group is not configured: {asset_group}")
    selected_asset_config = {asset_group: configured_groups[asset_group]}
    returns = load_asset_group_returns(
        root=data_roots["csi_processed"],
        asset_groups=selected_asset_config,
        return_method=str(asset_config.get("return_method", "normalized_price_change")),
        normalization_window=int(asset_config.get("normalization_window", 252)),
        normalization_min_periods=int(asset_config.get("normalization_min_periods", 63)),
    )
    definitions = list(config.get("combinations", []))
    if not definitions:
        raise ValueError("Macro regime validation requires at least one combination.")
    panel = build_combination_panel(macro_history, returns.index, definitions)

    market_column = str(config.get("market_baseline_column", "composite_score"))
    if market_column not in market_state_history:
        raise ValueError(f"Market state history is missing baseline column: {market_column}")
    market = pd.to_numeric(market_state_history[market_column], errors="coerce")
    market.index = pd.to_datetime(market_state_history.index, errors="coerce")
    market = market.loc[~market.index.isna()].sort_index()

    horizons = {key: int(value) for key, value in config.get("horizons", {}).items()}
    targets = [str(value) for value in config.get("targets", [])]
    min_train = int(config.get("min_train_observations", 60))
    market_last = market.dropna().index.max()
    requested_end = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp(market_last).normalize()
    effective_end = min(requested_end, pd.Timestamp(market_last).normalize())
    rows: list[dict[str, Any]] = []
    definition_by_id = {str(item["id"]): item for item in definitions}

    for group_id in returns.columns:
        group_returns = returns[[group_id]].dropna()
        common_index = panel.dropna(how="all").index.intersection(group_returns.index)
        eval_dates = evaluation_dates(
            common_index,
            max_horizon=max(horizons.values()),
            frequency=str(config.get("evaluation_frequency", "ME")),
            start=start,
            end=effective_end,
        )
        market_eval = market.reindex(market.index.union(eval_dates)).sort_index().ffill().reindex(eval_dates)
        market_rank = expanding_rank(market_eval)
        for horizon, trading_days in horizons.items():
            forward_targets = build_forward_market_targets(group_returns, trading_days, targets)
            for target_name, target_frame in forward_targets.items():
                realized = target_frame[group_id].reindex(eval_dates)
                contexts = walk_forward_contexts(eval_dates, realized, trading_days, min_train)
                for combination_id in panel.columns.get_level_values("combination_id").unique():
                    definition = definition_by_id[str(combination_id)]
                    values = panel[str(combination_id)].reindex(eval_dates)
                    ranked = values.apply(expanding_rank)
                    controls = pd.DataFrame(
                        {
                            "market": market_rank,
                            "growth": ranked["growth_weakness"],
                            "inflation": ranked["inflation_pressure"],
                        },
                        index=eval_dates,
                    )
                    combined = controls.assign(interaction=ranked["interaction_score"])
                    market_forecasts = cumulative_linear_forecasts(
                        controls[["market"]], realized, contexts, min_train
                    )
                    control_forecasts = cumulative_linear_forecasts(controls, realized, contexts, min_train)
                    combination_forecasts = cumulative_linear_forecasts(combined, realized, contexts, min_train)
                    for context in contexts:
                        eval_pos = int(context["eval_pos"])
                        market_forecast, market_count = market_forecasts.get(eval_pos, (float("nan"), 0))
                        control_forecast, control_count = control_forecasts.get(eval_pos, (float("nan"), 0))
                        combination_forecast, combined_count = combination_forecasts.get(eval_pos, (float("nan"), 0))
                        interaction_value = values["interaction_score"].iloc[eval_pos]
                        if (
                            pd.isna(interaction_value)
                            or pd.isna(market_forecast)
                            or pd.isna(control_forecast)
                            or pd.isna(combination_forecast)
                        ):
                            continue
                        rows.append(
                            {
                                "forecast_date": context["forecast_date"],
                                "asset_group": group_id,
                                "asset_group_label": str(selected_asset_config[group_id].get("label", group_id)),
                                "target": target_name,
                                "horizon": horizon,
                                "trading_days": trading_days,
                                "combination_id": combination_id,
                                "combination_label": str(definition["label"]),
                                "growth_indicator": str(definition["growth_indicator"]),
                                "inflation_indicator": str(definition["inflation_indicator"]),
                                "inflation_condition": str(definition["inflation_condition"]),
                                "growth_weakness": float(values["growth_weakness"].iloc[eval_pos]),
                                "inflation_pressure": float(values["inflation_pressure"].iloc[eval_pos]),
                                "interaction_score": float(interaction_value),
                                "train_observations": int(min(market_count, control_count, combined_count)),
                                "realized_target": float(context["realized"]),
                                "market_forecast": float(market_forecast),
                                "component_forecast": float(control_forecast),
                                "combination_forecast": float(combination_forecast),
                            }
                        )
    forecasts = pd.DataFrame.from_records(rows)
    if forecasts.empty:
        raise ValueError("Macro regime validation produced no out-of-sample forecasts.")
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
    summary = summarize_macro_regime_forecasts(forecasts, config)
    return MacroRegimeValidationResult(forecasts=forecasts, summary=summary, combination_panel=panel)


def summarize_macro_regime_forecasts(forecasts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["asset_group", "target", "horizon", "combination_id"]
    for _, group in forecasts.groupby(keys, sort=False):
        clean = group.dropna(
            subset=[
                "interaction_score",
                "realized_target",
                "market_forecast",
                "component_forecast",
                "combination_forecast",
            ]
        )
        if clean.empty:
            continue
        midpoint = max(1, len(clean) // 2)
        corr = _partial_correlation(clean)
        early_corr = _partial_correlation(clean.iloc[:midpoint])
        late_corr = _partial_correlation(clean.iloc[midpoint:])
        overlap = max(1.0, float(clean["trading_days"].iloc[0]) / 21.0)
        component_error = clean["component_forecast"] - clean["realized_target"]
        combination_error = clean["combination_forecast"] - clean["realized_target"]
        market_error = clean["market_forecast"] - clean["realized_target"]
        component_corr = _safe_spearman(clean["component_forecast"], clean["realized_target"])
        early_component_corr = _safe_spearman(
            clean["component_forecast"].iloc[:midpoint], clean["realized_target"].iloc[:midpoint]
        )
        late_component_corr = _safe_spearman(
            clean["component_forecast"].iloc[midpoint:], clean["realized_target"].iloc[midpoint:]
        )
        rows.append(
            {
                "asset_group": clean["asset_group"].iloc[0],
                "asset_group_label": clean["asset_group_label"].iloc[0],
                "target": clean["target"].iloc[0],
                "horizon": clean["horizon"].iloc[0],
                "trading_days": int(clean["trading_days"].iloc[0]),
                "combination_id": clean["combination_id"].iloc[0],
                "combination_label": clean["combination_label"].iloc[0],
                "growth_indicator": clean["growth_indicator"].iloc[0],
                "inflation_indicator": clean["inflation_indicator"].iloc[0],
                "inflation_condition": clean["inflation_condition"].iloc[0],
                "observations": int(len(clean)),
                "sample_start": clean["forecast_date"].min(),
                "sample_end": clean["forecast_date"].max(),
                "partial_corr_beyond_components": corr,
                "early_partial_corr": early_corr,
                "late_partial_corr": late_corr,
                "stable_sign": bool(
                    pd.notna(early_corr)
                    and pd.notna(late_corr)
                    and np.sign(early_corr) == np.sign(late_corr)
                ),
                "correlation_p_value_approx": approximate_correlation_p_value(corr, len(clean), overlap),
                "component_forecast_realized_corr": component_corr,
                "early_component_forecast_corr": early_component_corr,
                "late_component_forecast_corr": late_component_corr,
                "component_forecast_stable_sign": bool(
                    pd.notna(early_component_corr)
                    and pd.notna(late_component_corr)
                    and early_component_corr > 0.0
                    and late_component_corr > 0.0
                ),
                "component_forecast_p_value_approx": approximate_correlation_p_value(
                    component_corr, len(clean), overlap
                ),
                "market_mse": float(np.mean(market_error.pow(2))),
                "component_mse": float(np.mean(component_error.pow(2))),
                "components_improvement_vs_market": float(
                    np.mean(market_error.pow(2)) - np.mean(component_error.pow(2))
                ),
                "combination_mse": float(np.mean(combination_error.pow(2))),
                "combination_improvement_vs_components": float(
                    np.mean(component_error.pow(2)) - np.mean(combination_error.pow(2))
                ),
                "active_regime_observations": int(clean["interaction_score"].gt(0.0).sum()),
                "active_regime_mean_outcome": float(
                    clean.loc[clean["interaction_score"].gt(0.0), "realized_target"].mean()
                ),
                "inactive_regime_mean_outcome": float(
                    clean.loc[clean["interaction_score"].eq(0.0), "realized_target"].mean()
                ),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    if summary.empty:
        return summary
    summary["fdr_q_value_approx"] = benjamini_hochberg(summary["correlation_p_value_approx"])
    summary["component_forecast_fdr_q_value_approx"] = benjamini_hochberg(
        summary["component_forecast_p_value_approx"]
    )
    minimum = int(config.get("minimum_oos_observations", 60))
    alpha = float(config.get("fdr_alpha", 0.10))
    summary["robust_candidate"] = (
        summary["observations"].ge(minimum)
        & summary["active_regime_observations"].ge(15)
        & summary["stable_sign"]
        & summary["fdr_q_value_approx"].le(alpha)
        & summary["combination_improvement_vs_components"].gt(0.0)
    )
    summary["additive_robust_candidate"] = (
        summary["observations"].ge(minimum)
        & summary["component_forecast_stable_sign"]
        & summary["component_forecast_fdr_q_value_approx"].le(alpha)
        & summary["components_improvement_vs_market"].gt(0.0)
    )
    return summary.sort_values(
        [
            "robust_candidate",
            "additive_robust_candidate",
            "combination_improvement_vs_components",
            "partial_corr_beyond_components",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def write_macro_regime_validation_outputs(
    result: MacroRegimeValidationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": root / "macro_regime_forecasts.parquet",
        "summary": root / "macro_regime_summary.csv",
        "panel": root / "macro_regime_combination_panel.parquet",
    }
    result.forecasts.to_parquet(paths["forecasts"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.combination_panel.to_parquet(paths["panel"])
    return paths


def _partial_correlation(frame: pd.DataFrame) -> float:
    columns = ["growth_weakness", "inflation_pressure"]
    sample = frame[[*columns, "interaction_score", "realized_target"]].dropna()
    if len(sample) < 5 or sample["interaction_score"].nunique() < 2:
        return float("nan")
    controls = np.column_stack([np.ones(len(sample)), sample[columns].to_numpy(dtype=float)])
    interaction = sample["interaction_score"].to_numpy(dtype=float)
    target = sample["realized_target"].to_numpy(dtype=float)
    interaction_residual = interaction - controls @ np.linalg.lstsq(controls, interaction, rcond=None)[0]
    target_residual = target - controls @ np.linalg.lstsq(controls, target, rcond=None)[0]
    if np.std(interaction_residual) == 0.0 or np.std(target_residual) == 0.0:
        return float("nan")
    return float(pd.Series(interaction_residual).corr(pd.Series(target_residual), method="spearman"))


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1], method="spearman"))
