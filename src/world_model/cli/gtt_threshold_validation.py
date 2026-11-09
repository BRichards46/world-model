from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.cli.quad_forecast_research import markdown_table
from world_model.config import default_config_path, load_yaml, project_root
from world_model.data.benchmarks import load_equity_benchmark_history
from world_model.states.gtt_risk_validation import build_price_trends, run_gtt_risk_validation
from world_model.states.gtt_threshold_validation import (
    build_filters,
    build_point_in_time_values,
    monthly_timing_backtest,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test GTT macro filters at alternative thresholds, OR combinations and counts.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--gtt-config", default=str(default_config_path("gtt_risk_validation.yml")))
    parser.add_argument("--config", default=str(default_config_path("gtt_threshold_validation.yml")))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "gtt_threshold_validation"))
    parser.add_argument("--from-outputs", action="store_true", help="Rebuild the report from saved CSVs.")
    return parser


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("macro_signal")
        .agg(
            passes=("research_candidate", "sum"),
            tests=("research_candidate", "size"),
            median_incremental_lift=("incremental_lift_vs_trend_only", "median"),
            median_precision=("signal_precision", "median"),
            median_base_rate=("severe_base_rate", "median"),
            defensive_share=("double_confirmation_share", "median"),
            stress_eras=("active_stress_era_count", "median"),
        )
        .reset_index()
        .rename(columns={"macro_signal": "filter"})
    )


def _load_cash(fred_root: str) -> pd.Series:
    path = Path(fred_root).expanduser() / "latest" / "DGS3MO.parquet"
    frame = pd.read_parquet(path)
    series = pd.Series(frame["latest_value"].to_numpy(dtype=float), index=pd.to_datetime(frame["observation_date"]))
    return series.dropna().sort_index()


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_roots = load_yaml(args.data_roots)
    gtt_config = load_yaml(args.gtt_config)
    config = load_yaml(args.config)
    history = load_equity_benchmark_history(data_roots, gtt_config["equity_benchmark"])
    returns = history[["log_return"]].rename(columns={"log_return": "sp500"})
    trends = build_price_trends(
        returns,
        ten_month_window=int(gtt_config["price_trends"]["ten_month"]),
        daily_window=int(gtt_config["price_trends"]["two_hundred_day"]),
        return_conventions={"sp500": "log"},
    )
    values = build_point_in_time_values(data_roots["fred"])
    flags, catalog = build_filters(values, config)
    cash = _load_cash(data_roots["fred"])
    study_config = dict(gtt_config)
    study_config["bootstrap_iterations"] = int(config.get("bootstrap_iterations", 500))

    risk_rows, backtest_rows = [], []
    for window, settings in config["windows"].items():
        start = pd.Timestamp(settings["start"])
        usable = [name for name in flags.columns if flags[name].notna().any()]
        study_config["macro_signals"] = usable
        result = run_gtt_risk_validation(flags, trends, returns, study_config, {"sp500": "S&P 500"}, start, None, return_conventions={"sp500": "log"})
        summary = aggregate(result.summary)
        summary.insert(0, "window", window)
        risk_rows.append(summary)
        trend = trends["sp500__ten_month"]
        log_returns = returns["sp500"]
        cost = float(config.get("backtest_cost_bps", 5))
        baselines = {
            "buy_and_hold": monthly_timing_backtest(log_returns, cash, trend * 0.0, None, start, cost_bps=cost),
            "trend_only": monthly_timing_backtest(log_returns, cash, trend, None, start, cost_bps=cost),
        }
        for name, stats in baselines.items():
            backtest_rows.append({"window": window, "filter": name, **stats})
        for name in usable:
            stats = monthly_timing_backtest(log_returns, cash, trend, flags[name], start, cost_bps=cost)
            backtest_rows.append({"window": window, "filter": name, **stats})
    return values, catalog, pd.concat(risk_rows, ignore_index=True), pd.DataFrame.from_records(backtest_rows)


def write_report(catalog: pd.DataFrame, risk: pd.DataFrame, backtest: pd.DataFrame, path: Path, windows: dict) -> None:
    merged = risk.merge(backtest, on=["window", "filter"], how="left").merge(catalog, on="filter", how="left")
    merged["passes"] = merged["passes"].astype(int).astype(str) + "/" + merged["tests"].astype(int).astype(str)
    columns = ["filter", "start", "passes", "median_incremental_lift", "median_precision", "defensive_share", "cagr", "max_drawdown", "sharpe", "time_in_cash", "switches_per_year"]

    def table(frame: pd.DataFrame) -> str:
        shown = frame[columns].copy()
        for column in ("defensive_share", "median_precision"):
            shown[column] = shown[column] * 100.0
        return markdown_table(shown)

    lines = [
        "# GTT macro filters: thresholds, OR combinations and counts",
        "",
        "Each filter is a macro danger flag built release by release from FRED vintages. The GTT risk test asks whether, "
        "when the S&P 500 is below trend (10-month or 200-day), months with macro danger saw more future volatility and "
        "drawdown than months where the filter stayed healthy. `passes` counts the 16 trend × horizon × target tests that "
        "clear every robustness gate (bootstrap lower bound > 0, stable halves, 3+ stress eras, leave-one-era-out). "
        "`median_incremental_lift` is the extra future risk captured versus trend alone; `median_precision` is the share "
        "of defensive months followed by a worst-decile outcome (base rate about 10%); `defensive_share` is the share of "
        "months the filter would be defensive (%).",
        "",
        "The timing backtest holds the S&P 500 total return unless the month-end 10-month trend and the filter both warn, "
        "then holds 3-month Treasury bills (0% before September 1981), 5 bp per switch. CAGR, drawdown, time in cash in %.",
        "",
    ]
    for window, settings in windows.items():
        label = settings["label"]
        frame = merged[merged["window"] == window]
        base = backtest[(backtest["window"] == window) & backtest["filter"].isin(["buy_and_hold", "trend_only"])]
        lines += [f"## {label}", "", "### Baselines", "", markdown_table(base[["filter", "cagr", "max_drawdown", "sharpe", "time_in_cash", "switches_per_year"]]), ""]
        lines += ["### Zero lines vs your thresholds", "", table(frame[frame["family"] == "single"].sort_values(["metric", "kind"])), ""]
        lines += ["### OR combinations", "", table(frame[frame["family"] == "or"]), ""]
        lines += ["### Trigger counts", "", table(frame[frame["family"] == "count"]), ""]
        sweep = frame[frame["family"].isin(["single", "sweep"])].copy()
        if not sweep.empty:
            lines += ["### Threshold sweep (passes out of 16 by level)", ""]
            for metric, group in sweep.groupby("metric"):
                group = group.sort_values("level")
                cells = " · ".join(f"{level:g}: {passes.split('/')[0]}" for level, passes in zip(group["level"], group["passes"]))
                lines.append(f"- `{metric}` — {cells}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if args.from_outputs:
        catalog = pd.read_csv(output / "filter_catalog.csv")
        risk = pd.read_csv(output / "gtt_threshold_risk_summary.csv")
        backtest = pd.read_csv(output / "gtt_threshold_backtest.csv")
    else:
        values, catalog, risk, backtest = run(args)
        values.to_parquet(output / "point_in_time_values.parquet")
        catalog.to_csv(output / "filter_catalog.csv", index=False)
        risk.to_csv(output / "gtt_threshold_risk_summary.csv", index=False)
        backtest.to_csv(output / "gtt_threshold_backtest.csv", index=False)
    report = output / "gtt_threshold_validation_report.md"
    write_report(catalog, risk, backtest, report, load_yaml(args.config)["windows"])
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
