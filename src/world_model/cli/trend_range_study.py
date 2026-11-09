from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.cli.quad_forecast_research import markdown_table
from world_model.config import default_config_path, load_yaml, project_root
from world_model.states.quad_asset_performance import load_cumulative_assets
from world_model.states.quad_playbook_test import textbook_scores
from world_model.states.trend_range_study import run_trend_range_study


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test TREND, TRADE-range timing, and the quad as a filter on trend.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--config", default=str(default_config_path("trend_range_study.yml")))
    parser.add_argument("--asset-config", default=str(default_config_path("quad_asset_performance.yml")))
    parser.add_argument("--groups-config", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--quad-signals", default=str(root / "outputs" / "quad_asset_performance" / "quad_signals.parquet"))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "trend_range_study"))
    return parser


def _pct(value: object, decimals: int = 1) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:+.{decimals}f}"


def _range_table(summary: pd.DataFrame) -> str:
    frame = summary[summary["study"] == "trend_range"].copy()
    frame["threshold"] = frame["threshold"].map(lambda x: f"{x:.1f}")
    columns = ["horizon", "trend", "threshold", "series", "start", "annualized", "sharpe", "t_stat_hac", "q_value", "first_half", "second_half", "since_2010", "significant"]
    return markdown_table(frame[columns])


def _quad_table(summary: pd.DataFrame) -> str:
    frame = summary[summary["study"] == "quad_filter"]
    columns = ["horizon", "signal", "series", "start", "annualized", "sharpe", "t_stat_hac", "rotation_p", "q_value", "first_half", "second_half", "since_2010", "significant"]
    return markdown_table(frame[[c for c in columns if c in frame]])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    config = load_yaml(args.config)
    asset_config = load_yaml(args.asset_config)
    groups = load_yaml(args.groups_config).get("asset_groups", {})
    cumulative, metadata = load_cumulative_assets(data_roots, asset_config, groups)
    calendar = cumulative["sp500"].dropna().index
    signals = pd.read_parquet(args.quad_signals)
    assets = list(cumulative)
    scores = textbook_scores(asset_config.get("playbook", {}), assets)
    result = run_trend_range_study(cumulative, calendar, signals, scores, config)

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output / "trend_range_summary.csv", index=False)
    result.range_returns.to_parquet(output / "trend_range_returns.parquet")
    result.quad_returns.to_parquet(output / "quad_filter_returns.parquet")
    snapshot = result.snapshot.copy()
    snapshot.insert(1, "label", snapshot["asset_id"].map(metadata["asset_label"]))
    snapshot.to_csv(output / "trend_range_snapshot.csv", index=False)

    lines = [
        "# TREND, TRADE range and the quad filter",
        "",
        f"Signals from {config.get('start')} to {calendar.max():%Y-%m-%d}. Assets: {', '.join(metadata['asset_label'])}.",
        "",
        "Portfolios are inverse-volatility weighted with unit gross exposure. Returns are % per year "
        "(CSI futures use normalized price change). t-stats are Newey-West; q-values are Benjamini-Hochberg within "
        "study and horizon; *significant* also requires the same sign in both halves.",
        "",
        "## TREND and TRADE (weekly signals)",
        "",
        "- `trend`: hold every asset in the direction of its trend.",
        "- `trend_favorable` / `trend_unfavorable`: only trend positions entered at the favorable end of the 15-day band "
        "(dip in an uptrend, rally in a downtrend) or the unfavorable end.",
        "- `favorable_minus_unfavorable`: the range-timing effect.",
        "- `trade_fade`: fade the 15-day band with no trend filter (pure short-term reversal).",
        "",
        _range_table(result.summary),
        "",
        "## Quad as a filter on TREND (month-end signals, textbook playbook)",
        "",
        "- `trend_only`: trend positions in the playbook assets.",
        "- `agree` / `disagree`: trend positions the quad playbook agrees or disagrees with.",
        "- `agree_minus_disagree`: the quad's added value; `rotation_p` rotates the quad sequence against dates (one-sided).",
        "- `agree_only`: hold only agreeing positions, cash otherwise.",
        "",
        _quad_table(result.summary),
        "",
        "## Current state",
        "",
        markdown_table(snapshot),
        "",
    ]
    report = output / "trend_range_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    keep = ["study", "horizon", "trend", "threshold", "signal", "series", "annualized", "sharpe", "t_stat_hac", "test_p", "q_value", "significant"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(result.summary[keep].round(3).to_string(index=False))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
