from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.economic_triggers import EconomicTriggerValidationResult


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def format_float(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def write_markdown_report(result: EconomicTriggerValidationResult, output_path: str | Path) -> None:
    lines: list[str] = []
    lines.append("# Economic Trigger Validation")
    lines.append("")
    lines.append(f"Forecast rows: `{len(result.forecasts):,}`")
    lines.append(f"Trigger panel rows: `{len(result.trigger_panel):,}`")
    lines.append("")
    _add_current_snapshot(lines, result.current_snapshot)
    _add_top_tables(lines, result.summary)
    _add_target_feature_tables(lines, result.summary)

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _add_current_snapshot(lines: list[str], snapshot: pd.DataFrame) -> None:
    lines.append("## Current Trigger Snapshot")
    lines.append("")
    if snapshot.empty:
        lines.append("No current trigger snapshot available.")
        lines.append("")
        return
    as_of = pd.Timestamp(snapshot["date"].iloc[0]).date()
    failed = snapshot[snapshot["failed"].eq(1.0)].copy()
    healthy = snapshot[snapshot["healthy"].eq(1.0)].copy()
    lines.append(f"As of `{as_of}`, failed triggers: `{len(failed)}` / `{len(snapshot)}`.")
    lines.append("")
    lines.append("| Status | Trigger | Value | Rule | Margin | Observation |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: |")
    display = pd.concat([failed, healthy]).head(20)
    for row in display.to_dict("records"):
        status = "failed" if row["failed"] == 1.0 else "healthy"
        observation_date = row.get("observation_date")
        observation = "n/a" if pd.isna(observation_date) else str(pd.Timestamp(observation_date).date())
        lines.append(
            "| {status} | {label} | {value} | `{op} {threshold}` | {margin} | {observation} |".format(
                status=status,
                label=row["label"],
                value=format_float(row["value"]),
                op=row["healthy_operator"],
                threshold=format_float(row["threshold"]),
                margin=format_float(row["margin"]),
                observation=observation,
            )
        )
    lines.append("")


def _add_top_tables(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Strongest Results")
    lines.append("")
    if summary.empty:
        lines.append("No validation summary available.")
        lines.append("")
        return

    ranked = summary.sort_values("brier_improvement_vs_state_bucket", ascending=False).head(20)
    lines.append("Sorted by improvement versus a baseline that already knows the current state-score bucket.")
    lines.append("")
    lines.append("| Target | Horizon | Feature | Obs | Event rate | Feature spread | Corr | Brier improvement vs state bucket |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in ranked.to_dict("records"):
        spread = row["active_inactive_spread"] if row["feature_kind"] == "binary" else row["q5_q1_realized_spread"]
        lines.append(
            "| {target} | {horizon} | {feature} | {obs} | {event_rate} | {spread} | {corr} | {improvement} |".format(
                target=row["target_label"],
                horizon=row["horizon"],
                feature=row["feature_label"],
                obs=int(row["observations"]),
                event_rate=format_pct(row["realized_mean"]),
                spread=format_pct(spread),
                corr=format_float(row["feature_realized_corr"]),
                improvement=format_pct(row["brier_improvement_vs_state_bucket"]),
            )
        )
    lines.append("")


def _add_target_feature_tables(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Target Summaries")
    lines.append("")
    if summary.empty:
        return
    for target_label, target_group in summary.groupby("target_label", sort=False):
        lines.append(f"### {target_label}")
        lines.append("")
        ranked = target_group.sort_values("brier_improvement_vs_state_bucket", ascending=False).head(8)
        lines.append("| Horizon | Feature | Kind | Event rate | Active rate | Active event | Inactive event | Model Brier | State-bucket Brier |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in ranked.to_dict("records"):
            lines.append(
                "| {horizon} | {feature} | {kind} | {event_rate} | {active_rate} | {active_event} | {inactive_event} | {model} | {state} |".format(
                    horizon=row["horizon"],
                    feature=row["feature_label"],
                    kind=row["feature_kind"],
                    event_rate=format_pct(row["realized_mean"]),
                    active_rate=format_pct(row["active_rate"]),
                    active_event=format_pct(row["active_event_rate"]),
                    inactive_event=format_pct(row["inactive_event_rate"]),
                    model=format_pct(row["model_brier"]),
                    state=format_pct(row["state_bucket_brier"]),
                )
            )
        lines.append("")
