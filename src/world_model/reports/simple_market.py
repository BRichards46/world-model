from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.combined import CombinedValidationResult
from world_model.states.simple_market import SimpleMarketSnapshot


def format_score(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.2f}"


def format_metric(value: float, kind: str) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}" if kind == "binary" else f"{value:.2%}"


def write_simple_market_report(snapshot: SimpleMarketSnapshot, output_path: str | Path) -> None:
    lines: list[str] = []
    lines.append(f"# Simple Market Regime - {snapshot.as_of_date.date()}")
    lines.append("")
    lines.append(f"Regime: `{snapshot.regime_label}`")
    lines.append(f"Simple composite score: `{format_score(snapshot.simple_composite_score)}`")
    lines.append(f"Block coverage: `{snapshot.block_coverage:.0%}`")
    lines.append("")
    lines.append("## Block Scores")
    lines.append("")
    lines.append("| Block | Score |")
    lines.append("| --- | ---: |")
    for block, score in snapshot.block_scores.items():
        lines.append(f"| {block.replace('_score', '')} | {format_score(score)} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Equity, rates, commodities, and FX are each scored from trend and momentum only.")
    lines.append("- FX is inverted, so positive means weaker dollar / easier dollar-liquidity pressure.")
    lines.append("- This is intentionally a coarse regime layer. Detail should live underneath it as diagnostics.")
    lines.append("")
    _write_lines(lines, output_path)


def write_simple_validation_report(result: CombinedValidationResult, output_path: str | Path) -> None:
    lines: list[str] = []
    lines.append("# Simple Market Regime Validation")
    lines.append("")
    lines.append(f"Forecast rows: `{len(result.forecasts):,}`")
    lines.append(f"Summary rows: `{len(result.summary):,}`")
    lines.append("")
    _add_best_simple_table(lines, result.summary)
    _add_family_table(lines, result.summary)
    _add_target_tables(lines, result.summary)
    _write_lines(lines, output_path)


def _add_best_simple_table(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Best Improvements Versus Detailed Market Composite")
    lines.append("")
    if summary.empty:
        lines.append("No validation summary available.")
        lines.append("")
        return
    ranked = summary.sort_values("improvement_vs_detailed_market", ascending=False).head(25)
    lines.append("| Target | Horizon | Model | Obs | Error | Detailed error | Improvement | Corr | Spread |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in ranked.to_dict("records"):
        kind = row["target_kind"]
        lines.append(
            "| {target} | {horizon} | {model} | {obs} | {error} | {detailed} | {improvement} | {corr} | {spread} |".format(
                target=row["target_label"],
                horizon=row["horizon"],
                model=row["model_label"],
                obs=int(row["observations"]),
                error=format_metric(row["error_metric"], kind),
                detailed=format_metric(row["detailed_market_error_metric"], kind),
                improvement=format_metric(row["improvement_vs_detailed_market"], kind),
                corr=format_score(row["forecast_realized_corr"]),
                spread=format_metric(row["q5_q1_realized_spread"], kind),
            )
        )
    lines.append("")


def _add_family_table(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Model Family Summary")
    lines.append("")
    family = (
        summary.groupby(["model", "model_label", "model_kind"], sort=False)
        .agg(
            rows=("target", "size"),
            avg_improvement_vs_detailed=("improvement_vs_detailed_market", "mean"),
            median_improvement_vs_detailed=("improvement_vs_detailed_market", "median"),
            positive_vs_detailed=("improvement_vs_detailed_market", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values("avg_improvement_vs_detailed", ascending=False)
    )
    lines.append("| Model | Kind | Rows | Avg improvement | Median improvement | Positive share |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in family.to_dict("records"):
        lines.append(
            "| {model} | {kind} | {rows} | {avg:.3f} | {median:.3f} | {share:.0%} |".format(
                model=row["model_label"],
                kind=row["model_kind"],
                rows=int(row["rows"]),
                avg=row["avg_improvement_vs_detailed"],
                median=row["median_improvement_vs_detailed"],
                share=row["positive_vs_detailed"],
            )
        )
    lines.append("")


def _add_target_tables(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Target Details")
    lines.append("")
    for target_label, target_group in summary.groupby("target_label", sort=False):
        lines.append(f"### {target_label}")
        lines.append("")
        display = target_group.sort_values("improvement_vs_detailed_market", ascending=False).head(8)
        lines.append("| Horizon | Model | Error | Unconditional | Detailed | Simple | Improvement vs detailed |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in display.to_dict("records"):
            kind = row["target_kind"]
            lines.append(
                "| {horizon} | {model} | {error} | {uncond} | {detailed} | {simple} | {improvement} |".format(
                    horizon=row["horizon"],
                    model=row["model_label"],
                    error=format_metric(row["error_metric"], kind),
                    uncond=format_metric(row["unconditional_error_metric"], kind),
                    detailed=format_metric(row["detailed_market_error_metric"], kind),
                    simple=format_metric(row["simple_composite_error_metric"], kind),
                    improvement=format_metric(row["improvement_vs_detailed_market"], kind),
                )
            )
        lines.append("")


def _write_lines(lines: list[str], output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
