from __future__ import annotations

import argparse
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.economic_triggers import write_markdown_report
from world_model.states.economic_triggers import EconomicTriggerValidator, write_economic_trigger_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate economic trigger thresholds against future state outcomes.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--economic-config", default=str(default_config_path("economic_state.yml")))
    parser.add_argument("--trigger-config", default=str(default_config_path("economic_triggers.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(project_root() / "outputs"))
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "economic_trigger_validation_report.md"),
        help="Markdown report path. Defaults to outputs/economic_trigger_validation_report.md.",
    )
    parser.add_argument("--no-market-targets", action="store_true", help="Skip market-state deterioration targets.")
    parser.add_argument("--no-report", action="store_true", help="Do not write the markdown report.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    economic_config = load_yaml(args.economic_config)
    trigger_config = load_yaml(args.trigger_config)
    market_config = None if args.no_market_targets else load_yaml(args.market_config)
    if args.no_market_targets:
        trigger_config = dict(trigger_config)
        trigger_config["targets"] = dict(trigger_config.get("targets", {}))
        trigger_config["targets"]["include_market_state"] = False

    validator = EconomicTriggerValidator(
        data_roots=data_roots,
        economic_config=economic_config,
        trigger_config=trigger_config,
        market_config=market_config,
    )
    result = validator.run(start=args.start, end=args.end)
    paths = write_economic_trigger_outputs(result, args.output_dir)
    if not args.no_report:
        write_markdown_report(result, args.report_out)

    print("Economic trigger validation complete")
    print(f"Forecast rows: {len(result.forecasts):,}")
    print(f"Summary rows: {len(result.summary):,}")
    print(f"Trigger panel: {paths['trigger_panel']}")
    print(f"Summary: {paths['summary']}")
    if not args.no_report:
        print(f"Report written: {Path(args.report_out).expanduser()}")


if __name__ == "__main__":
    main()
