from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.walk_forward import WalkForwardResult


def write_walk_forward_report(
    result: WalkForwardResult,
    output_path: str | Path,
    frequency: str,
    top_n: int | None,
) -> None:
    lines: list[str] = []
    lines.append("# Market State Walk-Forward Validation")
    lines.append("")
    lines.append(f"Frequency: `{frequency}`")
    lines.append(
        "Forecast period: `{start}` to `{end}`".format(
            start=result.forecasts["forecast_date"].min().date(),
            end=result.forecasts["forecast_date"].max().date(),
        )
    )
    if top_n is not None:
        lines.append(f"Analog count: `{top_n}`")
    lines.append(f"Forecast rows: `{len(result.forecasts):,}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Horizon | Obs | Realized mean | Forecast mean | Baseline mean | Random mean | Q5-Q1 spread | Corr | Model RMSE | Baseline RMSE | P10/P90 coverage |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in result.summary.to_dict("records"):
        lines.append(
            "| {horizon} | {obs:,} | {realized} | {forecast} | {baseline} | {random} | {spread} | {corr} | {model_rmse} | {baseline_rmse} | {coverage} |".format(
                horizon=row["horizon"],
                obs=row["observations"],
                realized=_pct(row["realized_mean_return"]),
                forecast=_pct(row["forecast_mean_return"]),
                baseline=_pct(row["baseline_mean_return"]),
                random=_pct(row["random_mean_return"]),
                spread=_pct(row["top_bottom_realized_spread"]),
                corr=_num(row["forecast_realized_corr"]),
                model_rmse=_pct(row["model_rmse"]),
                baseline_rmse=_pct(row["baseline_rmse"]),
                coverage=_pct(row["p10_p90_coverage"]),
            )
        )
    lines.append("")
    lines.append("## Forecast Quintiles")
    lines.append("")
    lines.append("Quintile 1 is the lowest expected-return bucket. Quintile 5 is the highest.")
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
    lines.append("## Positive Probability Calibration")
    lines.append("")
    lines.append("| Horizon | Forecast probability bucket | Obs | Mean forecast probability | Observed positive rate |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in result.calibration.to_dict("records"):
        lines.append(
            "| {horizon} | {bucket} | {obs:,} | {forecast} | {observed} |".format(
                horizon=row["horizon"],
                bucket=row["probability_positive_bucket"],
                obs=row["observations"],
                forecast=_pct(row["mean_forecast_probability"]),
                observed=_pct(row["observed_rate"]),
            )
        )
    lines.append("")
    lines.append("## Drawdown Probability Calibration")
    lines.append("")
    lines.append("| Horizon | Forecast probability bucket | Obs | Mean forecast probability | Observed 10% drawdown rate |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in result.drawdown_calibration.to_dict("records"):
        lines.append(
            "| {horizon} | {bucket} | {obs:,} | {forecast} | {observed} |".format(
                horizon=row["horizon"],
                bucket=row["drawdown_10pct_probability_bucket"],
                obs=row["observations"],
                forecast=_pct(row["mean_forecast_probability"]),
                observed=_pct(row["observed_rate"]),
            )
        )
    lines.append("")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _num(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"
