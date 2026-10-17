from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.economic_asset_drivers import EconomicAssetDriverResult


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def format_float(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def write_markdown_report(result: EconomicAssetDriverResult, output_path: str | Path) -> None:
    lines: list[str] = []
    lines.append("# Economic Asset Driver Validation")
    lines.append("")
    lines.append(f"Forecast rows: `{len(result.forecasts):,}`")
    lines.append(f"Asset-group return rows: `{len(result.asset_group_returns):,}`")
    lines.append(f"Trigger panel rows: `{len(result.trigger_panel):,}`")
    lines.append(f"Asset return proxy: `{result.return_method}`")
    lines.append("")
    lines.append(
        "This is a walk-forward conditional-return study. It learns the trigger/asset relationship only from prior completed forward-return windows."
    )
    lines.append(
        "For CSI futures, the default return proxy uses normalized point changes instead of log returns because additively back-adjusted futures can cross zero."
    )
    lines.append("")
    _add_top_results(lines, result.summary)
    _add_group_results(lines, result.summary)

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _add_top_results(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Strongest Robust Mapped Driver Candidates")
    lines.append("")
    if summary.empty:
        lines.append("No summary rows available.")
        lines.append("")
        return
    robust = summary[summary["observations"] >= 60]
    mapped = robust[robust["matches_asset_prior"]]
    ranked = mapped.sort_values(
        ["mse_improvement_vs_baseline", "forecast_realized_corr"],
        ascending=[False, False],
    ).head(25)
    if ranked.empty:
        lines.append("No economically mapped rows have at least 60 out-of-sample observations.")
        lines.append("")
        return
    lines.append("| Asset group | Horizon | Driver | Feature | Obs | Forecast/realized corr | Spread | MSE improvement | Hit rate |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in ranked.to_dict("records"):
        spread = row["active_inactive_spread"] if row["feature_kind"] == "binary" else row["q5_q1_realized_spread"]
        lines.append(
            "| {group} | {horizon} | {driver} | {feature} | {obs} | {corr} | {spread} | {mse} | {hit} |".format(
                group=row["asset_group_label"],
                horizon=row["horizon"],
                driver=row["driver_group"],
                feature=row["feature_label"],
                obs=int(row["observations"]),
                corr=format_float(row["forecast_realized_corr"]),
                spread=format_pct(spread),
                mse=format_pct(row["mse_improvement_vs_baseline"]),
                hit=format_pct(row["directional_hit_rate"]),
            )
        )
    lines.append("")
    thin_sample = summary[summary["observations"] < 60]
    if not thin_sample.empty:
        lines.append(
            "Rows with fewer than 60 out-of-sample observations, and rows outside the configured asset-driver priors, are excluded from the robust ranking."
        )
        lines.append("")


def _add_group_results(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Best Result By Asset Group")
    lines.append("")
    if summary.empty:
        return
    lines.append("Best means highest-ranked available row under the reporting rules; it does not mean the row clears a usefulness threshold.")
    lines.append("")
    lines.append("| Asset group | Horizon | Driver | Feature | Obs | Prior | Corr | Spread | MSE improvement | Baseline return | Forecast return |")
    lines.append("| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    rows: list[pd.Series] = []
    for _, group in summary.groupby("asset_group", sort=True):
        candidates = group[group["observations"].ge(60) & group["matches_asset_prior"]]
        if candidates.empty:
            candidates = group[group["observations"] >= 60]
        if candidates.empty:
            candidates = group
        best = candidates.sort_values(
            ["mse_improvement_vs_baseline", "forecast_realized_corr"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(best)
    for row in rows:
        spread = row["active_inactive_spread"] if row["feature_kind"] == "binary" else row["q5_q1_realized_spread"]
        lines.append(
            "| {group} | {horizon} | {driver} | {feature} | {obs} | {prior} | {corr} | {spread} | {mse} | {baseline} | {forecast} |".format(
                group=row["asset_group_label"],
                horizon=row["horizon"],
                driver=row["driver_group"],
                feature=row["feature_label"],
                obs=int(row["observations"]),
                prior="yes" if row["matches_asset_prior"] else "no",
                corr=format_float(row["forecast_realized_corr"]),
                spread=format_pct(spread),
                mse=format_pct(row["mse_improvement_vs_baseline"]),
                baseline=format_pct(row["baseline_mean_return"]),
                forecast=format_pct(row["forecast_mean_return"]),
            )
        )
    lines.append("")
