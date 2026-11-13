from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.macro_market_validation import write_markdown_report
from world_model.states.macro_market_validation import (
    MacroMarketValidator,
    individual_asset_config,
    write_macro_market_validation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test point-in-time macro dashboard features against future market outcomes.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--asset-config", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--validation-config", default=str(default_config_path("macro_market_validation.yml")))
    parser.add_argument("--macro-history", default=str(root / "outputs" / "macro_dashboard_history.parquet"))
    parser.add_argument("--market-history", default=str(root / "outputs" / "market_state_history.parquet"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(root / "outputs" / "macro_market_validation"))
    parser.add_argument("--report-out", default=str(root / "outputs" / "macro_market_validation" / "macro_market_validation_report.md"))
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--individual-assets",
        action="store_true",
        help="Validate every configured futures contract separately instead of class-level equal-weight returns.",
    )
    parser.add_argument(
        "--indicator-id",
        action="append",
        default=[],
        help="Limit validation to one dashboard indicator; repeat for multiple indicators.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    macro_history = pd.read_parquet(Path(args.macro_history).expanduser())
    if args.indicator_id:
        macro_history = macro_history.loc[macro_history["indicator_id"].isin(args.indicator_id)].copy()
        if macro_history.empty:
            raise ValueError("No macro history rows match the requested indicator IDs.")
    market_history = pd.read_parquet(Path(args.market_history).expanduser())
    asset_config = load_yaml(args.asset_config)
    if args.individual_assets:
        asset_config = individual_asset_config(asset_config)
    validator = MacroMarketValidator(
        data_roots=load_yaml(args.data_roots),
        asset_config=asset_config,
        validation_config=load_yaml(args.validation_config),
    )
    result = validator.run(macro_history, market_history, start=args.start, end=args.end)
    paths = write_macro_market_validation_outputs(result, args.output_dir)
    if not args.no_report:
        write_markdown_report(result, args.report_out)
    robust = int(result.summary["robust_candidate"].sum()) if not result.summary.empty else 0
    print("Macro / market validation complete")
    print(f"Forecast rows: {len(result.forecasts):,}")
    print(f"Summary rows: {len(result.summary):,}")
    print(f"Robust candidates: {robust}")
    print(f"Summary: {paths['summary']}")
    if not args.no_report:
        print(f"Report: {Path(args.report_out).expanduser()}")


if __name__ == "__main__":
    main()
