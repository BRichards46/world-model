from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.target_validation import TargetValidationResult


def write_target_validation_report(
    result: TargetValidationResult,
    output_path: str | Path,
    frequency: str,
    min_train_observations: int,
) -> None:
    lines: list[str] = []
    lines.append("# Market Alternative Target Validation")
    lines.append("")
    lines.append(f"Frequency: `{frequency}`")
    lines.append(f"Minimum train observations: `{min_train_observations}`")
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
    lines.append("| Target | Horizon | Kind | Obs | Realized mean | Forecast mean | Baseline mean | Q5-Q1 spread | Corr | Model RMSE/Brier | Baseline RMSE/Brier |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in result.summary.to_dict("records"):
        metric_name = "model_brier" if row["target_kind"] == "binary" else "model_rmse"
        baseline_metric_name = "baseline_brier" if row["target_kind"] == "binary" else "baseline_rmse"
        lines.append(
            "| {target} | {horizon} | {kind} | {obs:,} | {realized} | {forecast} | {baseline} | {spread} | {corr} | {model_metric} | {baseline_metric} |".format(
                target=row["target_label"],
                horizon=row["horizon"],
                kind=row["target_kind"],
                obs=row["observations"],
                realized=_fmt(row["realized_mean"], row["target_kind"]),
                forecast=_fmt(row["forecast_mean"], row["target_kind"]),
                baseline=_fmt(row["baseline_mean"], row["target_kind"]),
                spread=_fmt(row["q5_q1_realized_spread"], row["target_kind"]),
                corr=_num(row["forecast_realized_corr"]),
                model_metric=_fmt(row[metric_name], row["target_kind"]),
                baseline_metric=_fmt(row[baseline_metric_name], row["target_kind"]),
            )
        )
    lines.append("")
    lines.append("## Quintiles")
    lines.append("")
    lines.append("Quintile 1 is the lowest forecast-value bucket. Quintile 5 is the highest.")
    lines.append("")
    lines.append("| Target | Horizon | Quintile | Obs | Mean forecast | Mean realized | Median realized |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in result.quintiles.to_dict("records"):
        lines.append(
            "| {target} | {horizon} | {quintile} | {obs:,} | {forecast} | {realized} | {median} |".format(
                target=row["target_label"],
                horizon=row["horizon"],
                quintile=row["forecast_quintile"],
                obs=row["observations"],
                forecast=_fmt(row["mean_forecast_value"], row["target_kind"]),
                realized=_fmt(row["mean_realized_value"], row["target_kind"]),
                median=_fmt(row["median_realized_value"], row["target_kind"]),
            )
        )
    lines.append("")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}" if kind in {"binary", "continuous"} else f"{value:.3f}"


def _num(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"
