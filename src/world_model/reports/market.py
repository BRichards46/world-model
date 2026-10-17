from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.market import AnalogResult, MarketStateSnapshot


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def format_score(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.2f}"


def write_markdown_report(
    snapshot: MarketStateSnapshot,
    analog_result: AnalogResult,
    output_path: str | Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# Market State - {snapshot.as_of_date.date()}")
    lines.append("")
    lines.append(f"Regime: `{snapshot.regime_label}`")
    lines.append(f"Composite score: `{format_score(snapshot.composite_score)}`")
    lines.append(f"Component coverage: `{snapshot.component_coverage:.0%}`")
    lines.append("")
    lines.append("## Component Scores")
    lines.append("")
    lines.append("| Component | Score |")
    lines.append("| --- | ---: |")
    for component, score in snapshot.component_scores.items():
        lines.append(f"| {component.replace('_score', '')} | {format_score(score)} |")
    lines.append("")
    lines.append("## Forward Analog Distribution")
    lines.append("")
    lines.append("| Horizon | Weighted mean | Median | 25th | 75th | P(positive) | P(drawdown > 10%) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in analog_result.distribution.to_dict("records"):
        lines.append(
            "| {horizon} | {mean} | {median} | {p25} | {p75} | {positive} | {dd10} |".format(
                horizon=row["horizon"],
                mean=format_pct(row["weighted_mean_return"]),
                median=format_pct(row["median_return"]),
                p25=format_pct(row["p25_return"]),
                p75=format_pct(row["p75_return"]),
                positive=format_pct(row["probability_positive"]),
                dd10=format_pct(row["probability_drawdown_gt_10pct"]),
            )
        )
    lines.append("")
    lines.append("## Closest Historical Analogs")
    lines.append("")
    lines.append("| Date | Regime | Similarity | Composite | 3m return | 6m return | 12m return |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    analogs = analog_result.analogs.head(15)
    for date, row in analogs.iterrows():
        lines.append(
            "| {date} | `{regime}` | {similarity:.2f} | {composite} | {r3m} | {r6m} | {r12m} |".format(
                date=pd.Timestamp(date).date(),
                regime=row["regime_label"],
                similarity=row["similarity"],
                composite=format_score(row["composite_score"]),
                r3m=format_pct(row.get("three_months_return")),
                r6m=format_pct(row.get("six_months_return")),
                r12m=format_pct(row.get("twelve_months_return")),
            )
        )
    lines.append("")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
