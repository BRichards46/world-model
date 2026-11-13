from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.data.benchmarks import load_equity_benchmark_history
from world_model.reports.gtt_risk_validation import write_gtt_risk_report
from world_model.states.economic_asset_drivers import load_asset_group_returns
from world_model.states.gtt_risk_validation import (
    build_point_in_time_gtt_macro_signals,
    build_price_trends,
    run_gtt_risk_validation,
    write_gtt_risk_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test GTT double-confirmation states against future market risk.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--asset-config", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--gtt-config", default=str(default_config_path("gtt_risk_validation.yml")))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(root / "outputs" / "gtt_risk_validation"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    asset_config = load_yaml(args.asset_config)
    config = load_yaml(args.gtt_config)
    returns = load_asset_group_returns(
        data_roots["csi_processed"],
        asset_config.get("asset_groups", {}),
        return_method=str(asset_config.get("return_method", "normalized_price_change")),
        normalization_window=int(asset_config.get("normalization_window", 252)),
        normalization_min_periods=int(asset_config.get("normalization_min_periods", 63)),
    )
    labels = {
        name: str(settings.get("label", name))
        for name, settings in asset_config.get("asset_groups", {}).items()
    }
    return_conventions: dict[str, str] = {}
    benchmark_history: pd.DataFrame | None = None
    benchmark = config.get("equity_benchmark", {})
    if benchmark.get("enabled", False):
        asset_id = str(benchmark.get("id", "sp500"))
        history = load_equity_benchmark_history(data_roots, benchmark)
        benchmark_history = history
        returns = pd.concat([returns, history["log_return"].rename(asset_id)], axis=1).sort_index()
        return_conventions[asset_id] = "log"
        labels[asset_id] = str(benchmark.get("label", "S&P 500"))
    macro = build_point_in_time_gtt_macro_signals(data_roots["fred"])
    trends = build_price_trends(
        returns,
        ten_month_window=int(config.get("price_trends", {}).get("ten_month", 10)),
        daily_window=int(config.get("price_trends", {}).get("two_hundred_day", 200)),
        return_conventions=return_conventions,
    )
    result = run_gtt_risk_validation(
        macro,
        trends,
        returns,
        config,
        labels,
        args.start,
        args.end,
        return_conventions=return_conventions,
    )
    paths = write_gtt_risk_outputs(result, args.output_dir)
    if benchmark_history is not None:
        benchmark_path = Path(args.output_dir).expanduser() / "gtt_sp500_hybrid_history.parquet"
        benchmark_history.to_parquet(benchmark_path)
        paths["sp500_history"] = benchmark_path
    report_path = Path(args.output_dir).expanduser() / "gtt_risk_validation_report.md"
    write_gtt_risk_report(result, report_path)
    print("GTT risk validation complete")
    print(f"Observation rows: {len(result.observations):,}")
    print(f"Summary rows: {len(result.summary):,}")
    print(f"Research candidates: {int(result.summary['research_candidate'].sum()):,}")
    print(f"Summary: {paths['summary']}")
    if "sp500_history" in paths:
        print(f"S&P history: {paths['sp500_history']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
