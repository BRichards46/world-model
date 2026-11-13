from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.macro_market_validation import MacroMarketValidationResult


def write_markdown_report(result: MacroMarketValidationResult, output_path: str | Path) -> None:
    summary = result.summary
    feature_start = result.macro_feature_panel.dropna(how="all").index.min()
    feature_end = result.macro_feature_panel.dropna(how="all").index.max()
    lines = [
        "# Macro / Market Relevance Validation",
        "",
        f"Macro feature coverage: `{feature_start.date()} to {feature_end.date()}`",
        f"Daily asset-return rows: `{len(result.asset_group_returns):,}`",
        f"Walk-forward forecast rows: `{len(result.forecasts):,}`",
        f"Summary tests: `{len(summary):,}`",
        f"Asset return proxy: `{result.return_method}`",
        "",
        "This study uses only macro states available by each forecast date. Forward market windows begin after that date. Conditional macro forecasts and nested market-plus-macro regressions are trained only on prior outcomes whose own forward windows had completed.",
        "",
        "Daily market data determine the complete path of each forward return, volatility, and drawdown outcome. Forecasts are sampled at month-end so highly overlapping daily forecast dates are not counted as independent evidence.",
        "",
        "The approximate p-values reduce effective sample size for overlapping horizons, and the q-values apply a Benjamini-Hochberg correction across the full test grid. They remain screening statistics—not proof. Correlated feature variants, horizons, targets, and asset groups can cause several rows to describe the same underlying relationship.",
        "",
        "## Screening summary",
        "",
        *_screening_summary(summary),
        "## Robust candidates",
        "",
    ]
    robust = summary.loc[summary["robust_candidate"]] if not summary.empty else summary
    if robust.empty:
        lines.extend([
            "No row currently clears all robustness gates: minimum out-of-sample observations, stable correlation sign across sample halves, approximate 10% false-discovery control, and positive incremental MSE improvement beyond the market-state baseline.",
            "",
        ])
    else:
        lines.extend(_table(robust.head(30)))
    lines.extend(["## Best exploratory rows", ""])
    eligible = summary.loc[summary["observations"].ge(12)].head(40) if not summary.empty else summary
    if eligible.empty:
        lines.extend(["No rows have enough observations for an exploratory ranking.", ""])
    else:
        lines.extend(_table(eligible))
    lines.extend([
        "## Interpretation rules",
        "",
        "- Positive `incremental vs market` means the macro feature reduced walk-forward MSE beyond the existing market composite.",
        "- Stable sign requires the feature/target rank correlation to have the same direction in both sample halves.",
        "- A strong in-sample-looking spread without incremental out-of-sample improvement is not a regime input.",
        "- Forecasting a macro series is a later step, undertaken only for features that survive this market-relevance screen.",
        "",
    ])
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _screening_summary(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["No completed tests were available.", ""]
    robust = summary.loc[summary["robust_candidate"]]
    family_keys = ["indicator_id", "asset_group", "target", "horizon"]
    families = robust.drop_duplicates(family_keys)
    directional = families.loc[families["target"].eq("forward_return")]
    risk = families.loc[families["target"].isin(["future_volatility", "max_drawdown"])]
    lines = [
        f"- Robust candidate rows: **{len(robust):,}**. These include alternate representations of the same macro signal.",
        f"- Distinct indicator / asset / target / horizon families: **{len(families):,}**.",
        f"- Directional-return families: **{len(directional):,}**; volatility or drawdown families: **{len(risk):,}**.",
    ]
    if not robust.empty:
        target_counts = robust.groupby("target").size().sort_values(ascending=False)
        rendered = ", ".join(f"{name.replace('_', ' ')} {count:,}" for name, count in target_counts.items())
        lines.append(f"- Candidate-row concentration by target: {rendered}.")
    lines.extend([
        "",
        "The family count is the more honest discovery count. The row count is useful for checking whether a relationship survives alternate feature definitions and horizons, but should not be read as independent confirmation.",
        "",
    ])
    return lines


def _table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Asset group | Target | Horizon | Indicator | Feature | Obs | Rank corr | Q5-Q1 | Stable | q-value | Incremental vs market |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            "| {asset} | {target} | {horizon} | {indicator} | {metric} | {obs} | {corr} | {spread} | {stable} | {q} | {incremental} |".format(
                asset=row["asset_group_label"],
                target=row["target"].replace("_", " "),
                horizon=row["horizon"].replace("_", " "),
                indicator=row["indicator_label"],
                metric=row["metric"].replace("_", " "),
                obs=int(row["observations"]),
                corr=_fmt(row["feature_realized_corr"]),
                spread=_fmt(row["q5_q1_realized_spread"]),
                stable="yes" if row["stable_sign"] else "no",
                q=_fmt(row["fdr_q_value_approx"]),
                incremental=_fmt(row["joint_improvement_vs_market"]),
            )
        )
    lines.append("")
    return lines


def _fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"
