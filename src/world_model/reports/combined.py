from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.combined import CombinedValidationResult


def format_metric(value: float, kind: str) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}" if kind == "binary" else f"{value:.2%}"


def format_float(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def write_markdown_report(result: CombinedValidationResult, output_path: str | Path) -> None:
    lines: list[str] = []
    lines.append("# Combined Market/Economic State Validation")
    lines.append("")
    lines.append(f"Forecast rows: `{len(result.forecasts):,}`")
    lines.append(f"Summary rows: `{len(result.summary):,}`")
    lines.append("")
    _add_top_joint_table(lines, result.summary)
    _add_model_family_summary(lines, result.summary)
    _add_target_tables(lines, result.summary)

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _add_top_joint_table(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Best Joint Improvements")
    lines.append("")
    if summary.empty:
        lines.append("No summary rows available.")
        lines.append("")
        return
    joint = summary[summary["model_kind"].eq("joint_bucket")].copy()
    joint = joint.sort_values("improvement_vs_market_bucket", ascending=False).head(25)
    lines.append("Sorted by improvement versus the market-composite bucket baseline.")
    lines.append("")
    lines.append("| Target | Horizon | Model | Obs | Error | Market error | Improvement | Corr | Spread |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in joint.to_dict("records"):
        kind = row["target_kind"]
        lines.append(
            "| {target} | {horizon} | {model} | {obs} | {error} | {market_error} | {improvement} | {corr} | {spread} |".format(
                target=row["target_label"],
                horizon=row["horizon"],
                model=row["model_label"],
                obs=int(row["observations"]),
                error=format_metric(row["error_metric"], kind),
                market_error=format_metric(row["market_bucket_error_metric"], kind),
                improvement=format_metric(row["improvement_vs_market_bucket"], kind),
                corr=format_float(row["forecast_realized_corr"]),
                spread=format_metric(row["q5_q1_realized_spread"], kind),
            )
        )
    lines.append("")


def _add_model_family_summary(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Model Family Summary")
    lines.append("")
    family = (
        summary.groupby(["model", "model_label", "model_kind"], sort=False)
        .agg(
            rows=("target", "size"),
            avg_improvement_vs_market=("improvement_vs_market_bucket", "mean"),
            median_improvement_vs_market=("improvement_vs_market_bucket", "median"),
            positive_vs_market=("improvement_vs_market_bucket", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values("avg_improvement_vs_market", ascending=False)
    )
    lines.append("| Model | Kind | Rows | Avg improvement vs market | Median improvement | Positive share |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in family.to_dict("records"):
        lines.append(
            "| {model} | {kind} | {rows} | {avg} | {median} | {share} |".format(
                model=row["model_label"],
                kind=row["model_kind"],
                rows=int(row["rows"]),
                avg=f"{row['avg_improvement_vs_market']:.3f}",
                median=f"{row['median_improvement_vs_market']:.3f}",
                share=f"{row['positive_vs_market']:.0%}",
            )
        )
    lines.append("")


def _add_target_tables(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Target Details")
    lines.append("")
    for target_label, target_group in summary.groupby("target_label", sort=False):
        lines.append(f"### {target_label}")
        lines.append("")
        display = target_group.sort_values("improvement_vs_market_bucket", ascending=False).head(10)
        lines.append("| Horizon | Model | Error | Uncond | Market | Economic | Improvement vs market |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in display.to_dict("records"):
            kind = row["target_kind"]
            lines.append(
                "| {horizon} | {model} | {error} | {uncond} | {market} | {econ} | {improvement} |".format(
                    horizon=row["horizon"],
                    model=row["model_label"],
                    error=format_metric(row["error_metric"], kind),
                    uncond=format_metric(row["unconditional_error_metric"], kind),
                    market=format_metric(row["market_bucket_error_metric"], kind),
                    econ=format_metric(row["economic_bucket_error_metric"], kind),
                    improvement=format_metric(row["improvement_vs_market_bucket"], kind),
                )
            )
        lines.append("")
