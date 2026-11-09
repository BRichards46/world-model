from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.gtt_risk_validation import GTTRiskValidationResult


def write_gtt_risk_report(result: GTTRiskValidationResult, output_path: str | Path) -> None:
    summary = result.summary
    candidates = summary.loc[summary["research_candidate"]]
    family_count = len(
        candidates.drop_duplicates(["asset_group", "target", "horizon", "macro_signal"])
    )
    lines = [
        "# GTT Drawdown and Volatility Validation",
        "",
        "This study tests the Growth–Trend Timing double-confirmation concept as a risk indicator. It does not assume that GTT is a return-forecasting model or recommend a trading allocation.",
        "",
        "Macro danger is measured four ways: unemployment above its point-in-time trailing 12-month average; non-positive industrial-production year-over-year growth; non-positive industrial production or real retail-sales growth; and any of those available warnings. Price danger is measured with 10-month and 200-trading-day moving-average rules.",
        "",
        "A double confirmation occurs only when macro danger and a negative price trend are both present. Daily market data determine the complete future volatility and drawdown paths; states are sampled at month-end.",
        "",
        "The S&P 500 benchmark uses price-index returns through January 4, 1988 and total-index returns from January 5, 1988 onward. Returns, rather than raw index levels, are chained into one continuous path.",
        "",
        f"Observation rows: `{len(result.observations):,}`  ",
        f"Summary tests: `{len(summary):,}`  ",
        f"Research candidate rows: `{len(candidates):,}`  ",
        f"Distinct relationship families: `{family_count:,}`",
        "",
        "## Candidate relationships",
        "",
    ]
    if candidates.empty:
        lines.extend(["No relationship passed the minimum state-count and split-sample stability gates.", ""])
    else:
        lines.extend(_table(candidates.head(40)))
    lines.extend([
        "## How to read the results",
        "",
        "- `Risk lift` is double-confirmation risk minus all other states. Positive is supportive.",
        "- `Incremental vs trend-only` compares double confirmation with negative-price-trend observations where the macro signal remained healthy. This is the direct test of whether the macro filter distinguishes a sustained danger state from a possible whipsaw.",
        "- Drawdown is sign-inverted for comparison, so a larger positive lift always means a worse subsequent risk outcome.",
        "- `Severe capture` is the share of the worst outcome decile that was already in double confirmation. `Precision` is the share of double-confirmation observations that entered that worst decile.",
        "- Candidate status requires adequate state counts, positive total and incremental lift, split-sample stability, above-base-rate severe-risk precision, a positive 90% time-block-bootstrap lower bound, and positive incremental lift after removing any single episode or any 12-month-clustered stress era.",
        "",
        "## Source concepts",
        "",
        "- https://www.philosophicaleconomics.com/2016/02/uetrend/",
        "- https://www.philosophicaleconomics.com/2016/01/gtt/",
        "- https://www.philosophicaleconomics.com/2016/01/movingaverage/",
        "",
    ])
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Asset | Outcome | Horizon | Macro filter | Trend | Active obs | Stress eras | Incremental vs trend-only | 90% CI low | Worst leave-era-out | Precision |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            "| {asset} | {target} | {horizon} | {macro} | {trend} | {obs} | {episodes} | {incremental} | {ci_low} | {leave_one_out} | {precision} |".format(
                asset=row["asset_group_label"],
                target=row["target"].replace("_", " "),
                horizon=row["horizon"].replace("_", " "),
                macro=row["macro_signal"].replace("_", " "),
                trend=row["price_trend"].replace("_", " "),
                obs=int(row["active_observations"]),
                episodes=int(row["active_stress_era_count"]),
                incremental=_fmt(row["incremental_lift_vs_trend_only"]),
                ci_low=_fmt(row["incremental_lift_ci_low"]),
                leave_one_out=_fmt(row["minimum_leave_one_stress_era_out_lift"]),
                precision=_pct(row["signal_precision"]),
            )
        )
    lines.append("")
    return lines


def _fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * value:.1f}%"
