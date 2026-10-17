from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.economic import EconomicStateSnapshot


def format_score(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.2f}"


def format_value(value: float, transform: str) -> str:
    if pd.isna(value):
        return "n/a"
    if "pct" in transform:
        return f"{value:.1%}"
    return f"{value:,.2f}"


def write_markdown_report(
    snapshot: EconomicStateSnapshot,
    missing_series: pd.DataFrame,
    output_path: str | Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# Economic State - {snapshot.as_of_date.date()}")
    lines.append("")
    lines.append(f"Regime: `{snapshot.regime_label}`")
    lines.append(f"Composite score: `{format_score(snapshot.composite_score)}`")
    lines.append(f"Component coverage: `{snapshot.component_coverage:.0%}`")
    lines.append("")
    lines.append("## Core Axes")
    lines.append("")
    lines.append("| Axis | Score |")
    lines.append("| --- | ---: |")
    lines.append(f"| Activity | {format_score(snapshot.activity_score)} |")
    lines.append(f"| Inflation pressure | {format_score(snapshot.inflation_pressure_score)} |")
    lines.append(f"| Liquidity / credit | {format_score(snapshot.liquidity_credit_score)} |")
    lines.append("")
    lines.append("## Component Scores")
    lines.append("")
    lines.append("| Component | Score |")
    lines.append("| --- | ---: |")
    for component, score in snapshot.component_scores.items():
        lines.append(f"| {component.replace('_score', '')} | {format_score(score)} |")
    lines.append("")
    lines.append("## Indicator Snapshot")
    lines.append("")
    lines.append("| Component | Indicator | Series | Transform | Observation | Value | Score |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for row in snapshot.indicators.to_dict("records"):
        observation_date = row.get("observation_date")
        if pd.isna(observation_date):
            observation = "n/a"
        else:
            observation = str(pd.Timestamp(observation_date).date())
        lines.append(
            "| {component} | {label} | `{series}` | `{transform}` | {observation} | {value} | {score} |".format(
                component=row["component"],
                label=row["label"],
                series=row["series_id"],
                transform=row["transform"],
                observation=observation,
                value=format_value(row["value"], row["transform"]),
                score=format_score(row["score"]),
            )
        )
    lines.append("")
    lines.append("## Missing Candidate Series")
    lines.append("")
    if missing_series.empty:
        lines.append("All candidate expansion series are present in the point-in-time FRED store.")
    else:
        lines.append("| Series | Why it matters |")
        lines.append("| --- | --- |")
        for row in missing_series.to_dict("records"):
            lines.append(f"| `{row['series_id']}` | {row['reason']} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Scores are rolling z-scores squashed to roughly `-1` to `+1`.")
    lines.append("- Inflation is a pressure score. Positive inflation is not treated as good; it subtracts from the composite.")
    lines.append("- The history uses FRED `vintage/` files and release dates, so prior states are based on data available at the time.")
    lines.append("")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
