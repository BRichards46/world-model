from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.simple_economic import (
    write_simple_economic_report,
    write_simple_economic_validation_report,
)
from world_model.states.simple_economic import (
    SimpleEconomicRegimeEngine,
    write_simple_economic_state_outputs,
    write_simple_economic_validation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate simplified three-axis economic regimes.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--simple-config", default=str(default_config_path("simple_economic_regime.yml")))
    parser.add_argument("--economic-config", default=str(default_config_path("economic_state.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--history-out", default=None)
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "simple_economic_regime_report.md"),
    )
    parser.add_argument(
        "--validation-report-out",
        default=str(project_root() / "outputs" / "simple_economic_validation_report.md"),
    )
    parser.add_argument("--output-dir", default=str(project_root() / "outputs"))
    parser.add_argument("--validate", action="store_true", help="Run validation against economic and market outcomes.")
    parser.add_argument(
        "--no-market-targets",
        action="store_true",
        help="Validate only economic deterioration targets, without market-state or price targets.",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    simple_config = load_yaml(args.simple_config)
    economic_config = load_yaml(args.economic_config)
    engine = SimpleEconomicRegimeEngine(data_roots, simple_config, economic_config)

    detailed_state = engine.build_detailed_state_history()
    state = engine.build_state_history(detailed_state)
    snapshot = engine.snapshot(args.as_of, state=state)
    if args.history_out:
        write_simple_economic_state_outputs(state, args.history_out)
    if not args.no_report:
        write_simple_economic_report(snapshot, args.report_out)

    print(f"Simple economic regime as of {snapshot.as_of_date.date()}: {snapshot.regime_label}")
    print(f"Simple economic composite score: {snapshot.simple_economic_composite_score:+.2f}")
    print(f"Detailed economic composite score: {snapshot.detailed_economic_composite_score:+.2f}")
    for axis, score in snapshot.axis_scores.items():
        print(f"  {axis}: {score:+.2f}" if not pd.isna(score) else f"  {axis}: n/a")
    if not args.no_report:
        print(f"Report written: {args.report_out}")

    if args.validate:
        market_config = None if args.no_market_targets else load_yaml(args.market_config)
        result = engine.validate(
            market_config=market_config,
            start=args.start,
            end=args.end,
            simple_state=state,
            detailed_state=detailed_state,
        )
        paths = write_simple_economic_validation_outputs(result, args.output_dir)
        if not args.no_report:
            write_simple_economic_validation_report(result, args.validation_report_out)
        print("")
        print("Simple economic validation complete")
        print(f"Forecast rows: {len(result.forecasts):,}")
        print(f"Summary rows: {len(result.summary):,}")
        print(f"Summary: {paths['summary']}")
        if not args.no_report:
            print(f"Validation report written: {Path(args.validation_report_out).expanduser()}")


if __name__ == "__main__":
    main()
