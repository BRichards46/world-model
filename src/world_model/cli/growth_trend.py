from __future__ import annotations

import argparse
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.growth_trend import write_growth_trend_report
from world_model.states.growth_trend import GrowthTrendTimingEngine, write_growth_trend_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run growth-trend timing strategy tests.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--gtt-config", default=str(default_config_path("growth_trend_timing.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--economic-config", default=str(default_config_path("economic_state.yml")))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(project_root() / "outputs"))
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "growth_trend_timing_report.md"),
    )
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = GrowthTrendTimingEngine(
        data_roots=load_yaml(args.data_roots),
        config=load_yaml(args.gtt_config),
        market_config=load_yaml(args.market_config),
        economic_config=load_yaml(args.economic_config),
    )
    result = engine.run(start=args.start, end=args.end)
    paths = write_growth_trend_outputs(result, args.output_dir)
    if not args.no_report:
        write_growth_trend_report(result, args.report_out)

    print("Growth-trend timing complete")
    print(f"Strategies: {len(result.summary):,}")
    print(f"Panel rows: {len(result.panel):,}")
    print(f"Summary: {paths['summary']}")
    print(f"Annual returns: {paths['annual_returns']}")
    print(f"Trades: {paths['trades']}")
    if not args.no_report:
        print(f"Report written: {Path(args.report_out).expanduser()}")
    if not result.summary.empty:
        top = result.summary.sort_values("cagr", ascending=False).iloc[0]
        print(f"Top CAGR strategy: {top['strategy_label']} ({top['cagr']:.1%})")


if __name__ == "__main__":
    main()
