from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.forecast import ForecastRankerResult


def write_forecast_ranker_report(
    result: ForecastRankerResult,
    output_path: str | Path,
    frequency: str,
    min_train_observations: int,
    train_window_years: int | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Market Forecast Ranker Validation")
    lines.append("")
    lines.append(f"Frequency: `{frequency}`")
    lines.append(f"Minimum train observations: `{min_train_observations}`")
    if train_window_years is not None:
        lines.append(f"Training window: `{train_window_years} years`")
    lines.append(
        "Forecast period: `{start}` to `{end}`".format(
            start=result.forecasts["forecast_date"].min().date(),
            end=result.forecasts["forecast_date"].max().date(),
        )
    )
    lines.append(f"Forecast rows: `{len(result.forecasts):,}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Horizon | Obs | Realized mean | Forecast mean | Baseline mean | Q5-Q1 spread | Corr | Model RMSE | Baseline RMSE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in result.summary.to_dict("records"):
        lines.append(
            "| {horizon} | {obs:,} | {realized} | {forecast} | {baseline} | {spread} | {corr} | {model_rmse} | {baseline_rmse} |".format(
                horizon=row["horizon"],
                obs=row["observations"],
                realized=_pct(row["realized_mean_return"]),
                forecast=_pct(row["forecast_mean_return"]),
                baseline=_pct(row["baseline_mean_return"]),
                spread=_pct(row["top_bottom_realized_spread"]),
                corr=_num(row["forecast_realized_corr"]),
                model_rmse=_pct(row["model_rmse"]),
                baseline_rmse=_pct(row["baseline_rmse"]),
            )
        )
    lines.append("")
    lines.append("## Forecast Quintiles")
    lines.append("")
    lines.append("Quintile 1 is the lowest forecast-return bucket. Quintile 5 is the highest.")
    lines.append("")
    lines.append("| Horizon | Quintile | Obs | Mean forecast | Mean realized | Median realized | Positive rate | Avg max drawdown |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in result.quintiles.to_dict("records"):
        lines.append(
            "| {horizon} | {quintile} | {obs:,} | {forecast} | {mean_realized} | {median_realized} | {positive} | {drawdown} |".format(
                horizon=row["horizon"],
                quintile=row["forecast_quintile"],
                obs=row["observations"],
                forecast=_pct(row["mean_forecast_return"]),
                mean_realized=_pct(row["mean_realized_return"]),
                median_realized=_pct(row["median_realized_return"]),
                positive=_pct(row["positive_rate"]),
                drawdown=_pct(row["mean_realized_max_drawdown"]),
            )
        )
    lines.append("")
    lines.append("## Component Diagnostics")
    lines.append("")
    lines.append("Full-sample component diagnostics are descriptive, not training inputs for any forecast date.")
    lines.append("")
    lines.append("| Horizon | Component | Obs | Rank IC | Q5-Q1 spread | Q1 positive | Q5 positive |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    diagnostics = result.component_diagnostics.copy()
    diagnostics["_abs_rank_ic"] = diagnostics["rank_ic"].abs()
    diagnostics = diagnostics.sort_values(["horizon", "_abs_rank_ic"], ascending=[True, False])
    for row in diagnostics.to_dict("records"):
        lines.append(
            "| {horizon} | {component} | {obs:,} | {rank_ic} | {spread} | {q1_positive} | {q5_positive} |".format(
                horizon=row["horizon"],
                component=row["component"].replace("_score", ""),
                obs=row["observations"],
                rank_ic=_num(row["rank_ic"]),
                spread=_pct(row["q5_q1_spread"]),
                q1_positive=_pct(row["q1_positive_rate"]),
                q5_positive=_pct(row["q5_positive_rate"]),
            )
        )
    lines.append("")
    lines.append("## Average Learned Weights")
    lines.append("")
    lines.append("| Horizon | Component | Mean weight | Mean rank IC | Selection count |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    weights = summarize_weights(result.learned_weights)
    for row in weights.to_dict("records"):
        lines.append(
            "| {horizon} | {component} | {weight} | {rank_ic} | {count:,} |".format(
                horizon=row["horizon"],
                component=row["component"].replace("_score", ""),
                weight=_num(row["mean_weight"]),
                rank_ic=_num(row["mean_rank_ic"]),
                count=row["selection_count"],
            )
        )
    lines.append("")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return weights
    grouped = (
        weights.groupby(["horizon", "trading_days", "component"], as_index=False)
        .agg(
            mean_weight=("weight", "mean"),
            mean_rank_ic=("rank_ic", "mean"),
            selection_count=("component", "size"),
        )
        .sort_values(["trading_days", "mean_weight"], ascending=[True, False])
    )
    return grouped


def _pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _num(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"
