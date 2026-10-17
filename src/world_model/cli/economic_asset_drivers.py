from __future__ import annotations

import argparse
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.economic_asset_drivers import write_markdown_report
from world_model.states.economic_asset_drivers import (
    EconomicAssetDriverValidator,
    write_economic_asset_driver_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate point-in-time economic triggers against asset-group returns.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--trigger-config", default=str(default_config_path("economic_triggers.yml")))
    parser.add_argument("--asset-driver-config", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(project_root() / "outputs" / "economic_asset_drivers"))
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "economic_asset_drivers" / "economic_asset_driver_report.md"),
    )
    parser.add_argument("--no-report", action="store_true", help="Do not write the markdown report.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validator = EconomicAssetDriverValidator(
        data_roots=load_yaml(args.data_roots),
        trigger_config=load_yaml(args.trigger_config),
        asset_driver_config=load_yaml(args.asset_driver_config),
    )
    result = validator.run(start=args.start, end=args.end)
    paths = write_economic_asset_driver_outputs(result, args.output_dir)
    if not args.no_report:
        write_markdown_report(result, args.report_out)

    print("Economic asset driver validation complete")
    print(f"Forecast rows: {len(result.forecasts):,}")
    print(f"Summary rows: {len(result.summary):,}")
    print(f"Summary: {paths['summary']}")
    if not args.no_report:
        print(f"Report written: {Path(args.report_out).expanduser()}")


if __name__ == "__main__":
    main()
