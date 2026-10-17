from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.simple_market import write_simple_market_report, write_simple_validation_report
from world_model.states.simple_market import (
    SimpleMarketRegimeEngine,
    write_simple_state_outputs,
    write_simple_validation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate simplified four-block market regimes.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--simple-config", default=str(default_config_path("simple_market_regime.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--history-out", default=None)
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "simple_market_regime_report.md"),
    )
    parser.add_argument(
        "--validation-report-out",
        default=str(project_root() / "outputs" / "simple_market_validation_report.md"),
    )
    parser.add_argument("--output-dir", default=str(project_root() / "outputs"))
    parser.add_argument("--validate", action="store_true", help="Run validation against target outcomes.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    simple_config = load_yaml(args.simple_config)
    engine = SimpleMarketRegimeEngine(data_roots, simple_config)

    state = engine.build_state_history()
    snapshot = engine.snapshot(args.as_of, state=state)
    if args.history_out:
        write_simple_state_outputs(state, args.history_out)
    if not args.no_report:
        write_simple_market_report(snapshot, args.report_out)

    print(f"Simple market regime as of {snapshot.as_of_date.date()}: {snapshot.regime_label}")
    print(f"Simple composite score: {snapshot.simple_composite_score:+.2f}")
    for block, score in snapshot.block_scores.items():
        print(f"  {block}: {score:+.2f}" if not pd.isna(score) else f"  {block}: n/a")
    if not args.no_report:
        print(f"Report written: {args.report_out}")

    if args.validate:
        result = engine.validate_against_market_state(
            market_config=load_yaml(args.market_config),
            start=args.start,
            end=args.end,
        )
        paths = write_simple_validation_outputs(result, args.output_dir)
        if not args.no_report:
            write_simple_validation_report(result, args.validation_report_out)
        print("")
        print("Simple market validation complete")
        print(f"Forecast rows: {len(result.forecasts):,}")
        print(f"Summary rows: {len(result.summary):,}")
        print(f"Summary: {paths['summary']}")
        if not args.no_report:
            print(f"Validation report written: {Path(args.validation_report_out).expanduser()}")


if __name__ == "__main__":
    main()
