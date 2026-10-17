from __future__ import annotations

import argparse
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.combined import write_markdown_report
from world_model.states.combined import CombinedStateValidator, write_combined_validation_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate combined market/economic state conditioning.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--economic-config", default=str(default_config_path("economic_state.yml")))
    parser.add_argument("--combined-config", default=str(default_config_path("combined_state_validation.yml")))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(project_root() / "outputs"))
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "combined_state_validation_report.md"),
        help="Markdown report path. Defaults to outputs/combined_state_validation_report.md.",
    )
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validator = CombinedStateValidator(
        data_roots=load_yaml(args.data_roots),
        market_config=load_yaml(args.market_config),
        economic_config=load_yaml(args.economic_config),
        combined_config=load_yaml(args.combined_config),
    )
    result = validator.run(start=args.start, end=args.end)
    paths = write_combined_validation_outputs(result, args.output_dir)
    if not args.no_report:
        write_markdown_report(result, args.report_out)

    print("Combined state validation complete")
    print(f"Forecast rows: {len(result.forecasts):,}")
    print(f"Summary rows: {len(result.summary):,}")
    print(f"Summary: {paths['summary']}")
    if not args.no_report:
        print(f"Report written: {Path(args.report_out).expanduser()}")


if __name__ == "__main__":
    main()
