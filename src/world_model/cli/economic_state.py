from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.economic import write_markdown_report
from world_model.states.economic import EconomicStateEngine, write_economic_state_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build point-in-time economic state vectors from FRED data.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--economic-config", default=str(default_config_path("economic_state.yml")))
    parser.add_argument("--as-of", default=None, help="Snapshot date. Defaults to latest available date.")
    parser.add_argument("--history-out", default=None, help="Optional .parquet or .csv path for state history.")
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "economic_state_report.md"),
        help="Markdown report path. Defaults to outputs/economic_state_report.md inside the project.",
    )
    parser.add_argument("--no-report", action="store_true", help="Do not write the default markdown report.")
    parser.add_argument("--json", action="store_true", help="Print snapshot as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = EconomicStateEngine(load_yaml(args.data_roots), load_yaml(args.economic_config))

    state = engine.build_state_history()
    snapshot = engine.snapshot(args.as_of, state=state)
    if args.history_out:
        write_economic_state_outputs(state, args.history_out)

    wrote_report = False
    if args.report_out and not args.no_report:
        write_markdown_report(snapshot, engine.recommended_missing_series(), args.report_out)
        wrote_report = True

    if args.json:
        payload = {
            "as_of_date": str(snapshot.as_of_date.date()),
            "regime_label": snapshot.regime_label,
            "composite_score": snapshot.composite_score,
            "component_coverage": snapshot.component_coverage,
            "activity_score": snapshot.activity_score,
            "inflation_pressure_score": snapshot.inflation_pressure_score,
            "liquidity_credit_score": snapshot.liquidity_credit_score,
            "component_scores": snapshot.component_scores,
            "indicators": snapshot.indicators.to_dict("records"),
        }
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        print(f"Economic state as of {snapshot.as_of_date.date()}: {snapshot.regime_label}")
        print(f"Composite score: {snapshot.composite_score:+.2f}")
        print(f"Activity: {snapshot.activity_score:+.2f}")
        print(f"Inflation pressure: {snapshot.inflation_pressure_score:+.2f}")
        print(f"Liquidity / credit: {snapshot.liquidity_credit_score:+.2f}")
        print("Component scores:")
        for component, score in snapshot.component_scores.items():
            print(f"  {component}: {score:+.2f}" if not pd.isna(score) else f"  {component}: n/a")
        if wrote_report:
            print("")
            print(f"Report written: {args.report_out}")


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
