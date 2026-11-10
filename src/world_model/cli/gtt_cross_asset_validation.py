from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.cli.gtt_threshold_validation import aggregate
from world_model.cli.quad_forecast_research import markdown_table
from world_model.config import default_config_path, load_yaml, project_root
from world_model.data.benchmarks import load_equity_benchmark_history
from world_model.data.csi import read_price_series
from world_model.data.norgate import read_norgate_index_series
from world_model.states.gtt_cross_asset_validation import build_logic_filters, cash_yield, synthetic_treasury_log_returns
from world_model.states.gtt_risk_validation import build_price_trends, run_gtt_risk_validation
from world_model.states.gtt_threshold_validation import build_point_in_time_values, monthly_timing_backtest


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test GTT on the Russell 2000, Treasuries and the dollar.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--gtt-config", default=str(default_config_path("gtt_risk_validation.yml")))
    parser.add_argument("--config", default=str(default_config_path("gtt_cross_asset_validation.yml")))
    parser.add_argument("--values", default=str(root / "outputs" / "gtt_threshold_validation" / "point_in_time_values.parquet"),
                        help="Point-in-time macro values from the threshold study; rebuilt if missing.")
    parser.add_argument("--output-dir", default=str(root / "outputs" / "gtt_cross_asset_validation"))
    return parser


def _latest(fred_root: str, series_id: str) -> pd.Series:
    frame = pd.read_parquet(Path(fred_root).expanduser() / "latest" / f"{series_id}.parquet")
    return pd.Series(frame["latest_value"].to_numpy(dtype=float), index=pd.to_datetime(frame["observation_date"])).dropna().sort_index()


def load_asset_log_returns(source: dict, data_roots: dict, gtt_config: dict) -> pd.Series:
    kind = source["kind"]
    if kind == "benchmark":
        return load_equity_benchmark_history(data_roots, gtt_config["equity_benchmark"])["log_return"].dropna()
    if kind == "norgate_index":
        price = read_norgate_index_series(data_roots["norgate_metadata"], source["symbol"])
        return np.log(price[price > 0]).diff().dropna()
    if kind == "synthetic_treasury":
        return synthetic_treasury_log_returns(_latest(data_roots["fred"], source["yield_series"]), float(source.get("maturity_years", 10)))
    if kind == "csi":
        price = read_price_series(data_roots["csi_processed"], source["file"])
        return np.log(price[price > 0]).diff().dropna()
    raise ValueError(f"Unknown source kind {kind}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    gtt_config = load_yaml(args.gtt_config)
    config = load_yaml(args.config)
    values_path = Path(args.values)
    values = pd.read_parquet(values_path) if values_path.exists() else build_point_in_time_values(data_roots["fred"])
    filters = build_logic_filters(values)
    cash = cash_yield(_latest(data_roots["fred"], "DGS3MO"), _latest(data_roots["fred"], "DGS1"))
    no_cash = pd.Series(0.0, index=cash.index)
    cost = float(config.get("backtest_cost_bps", 5))

    risk_rows, backtest_rows = [], []
    for asset_id, settings in config["assets"].items():
        log_returns = load_asset_log_returns(settings["source"], data_roots, gtt_config)
        returns = log_returns.to_frame(asset_id)
        trends = build_price_trends(returns, int(gtt_config["price_trends"]["ten_month"]), int(gtt_config["price_trends"]["two_hundred_day"]), {asset_id: "log"})
        start = pd.Timestamp(settings["start"])
        names = list(config["logic_filters"][settings["logic"]])
        study_config = dict(gtt_config, macro_signals=names, bootstrap_iterations=int(config.get("bootstrap_iterations", 500)))
        result = run_gtt_risk_validation(filters, trends, returns, study_config, {asset_id: settings["label"]}, start, None, return_conventions={asset_id: "log"})
        summary = aggregate(result.summary)
        summary.insert(0, "asset", asset_id)
        risk_rows.append(summary)
        trend = trends[f"{asset_id}__ten_month"]
        carry = no_cash if settings.get("excess_returns") else cash
        backtest_rows.append({"asset": asset_id, "filter": "buy_and_hold", **monthly_timing_backtest(log_returns, carry, trend * 0.0, None, start, cost_bps=cost)})
        backtest_rows.append({"asset": asset_id, "filter": "trend_only", **monthly_timing_backtest(log_returns, carry, trend, None, start, cost_bps=cost)})
        for name in names:
            backtest_rows.append({"asset": asset_id, "filter": name, **monthly_timing_backtest(log_returns, carry, trend, filters[name], start, cost_bps=cost)})
        print(f"{asset_id}: {len(log_returns):,} days from {log_returns.index.min():%Y-%m-%d}")

    risk = pd.concat(risk_rows, ignore_index=True)
    backtest = pd.DataFrame.from_records(backtest_rows)
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    risk.to_csv(output / "gtt_cross_asset_risk_summary.csv", index=False)
    backtest.to_csv(output / "gtt_cross_asset_backtest.csv", index=False)
    merged = backtest.merge(risk, on=["asset", "filter"], how="left")
    merged["passes"] = merged.apply(lambda row: "" if pd.isna(row.get("passes")) else f"{int(row['passes'])}/{int(row['tests'])}", axis=1)
    merged["median_precision"] = merged["median_precision"] * 100.0
    columns = ["filter", "start", "passes", "median_incremental_lift", "median_precision", "cagr", "max_drawdown", "sharpe", "time_in_cash", "switches_per_year"]
    lines = [
        "# GTT beyond the S&P 500",
        "",
        "Macro logic is fixed per asset class in `config/gtt_cross_asset_validation.yml` before any results. The strategy "
        "holds the asset unless its month-end 10-month trend is down and the danger filter is on, then holds cash "
        "(T-bills; 1-year Treasury before 1981). Futures rows are excess returns with cash at 0. `passes` counts the 16 "
        "GTT risk tests (two trends × four horizons × volatility and drawdown) that clear every robustness gate. "
        "Returns, drawdown and time in cash are in %.",
        "",
    ]
    for asset_id, settings in config["assets"].items():
        frame = merged[merged["asset"] == asset_id]
        lines += [f"## {settings['label']}", "", f"Logic: `{settings['logic']}`.", "", markdown_table(frame[columns]), ""]
    report = output / "gtt_cross_asset_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
