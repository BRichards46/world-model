from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.market import write_markdown_report
from world_model.states.market import MarketStateEngine, write_state_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build market state vectors and analog forecasts.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--as-of", default=None, help="Snapshot date. Defaults to latest available date.")
    parser.add_argument("--top-n", type=int, default=None, help="Number of closest historical analogs.")
    parser.add_argument("--history-out", default=None, help="Optional .parquet or .csv path for state history.")
    parser.add_argument("--analogs-out", default=None, help="Optional .parquet or .csv path for analog rows.")
    parser.add_argument("--distribution-out", default=None, help="Optional .parquet or .csv path for distribution rows.")
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "market_state_report.md"),
        help="Markdown report path. Defaults to outputs/market_state_report.md inside the project.",
    )
    parser.add_argument("--no-report", action="store_true", help="Do not write the default markdown report.")
    parser.add_argument("--json", action="store_true", help="Print snapshot and distribution as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = MarketStateEngine(load_yaml(args.data_roots), load_yaml(args.market_config))

    state = engine.build_state_history()
    snapshot = engine.snapshot(args.as_of)
    analog_result = engine.analog_forecast(args.as_of, top_n=args.top_n)

    if args.history_out:
        write_state_outputs(state, args.history_out)
    if args.analogs_out:
        _write_frame(analog_result.analogs, args.analogs_out)
    if args.distribution_out:
        _write_frame(analog_result.distribution, args.distribution_out)
    wrote_report = False
    if args.report_out and not args.no_report:
        write_markdown_report(snapshot, analog_result, args.report_out)
        wrote_report = True

    if args.json:
        closest_analogs = []
        for date, row in analog_result.analogs.head(10).iterrows():
            record = row.to_dict()
            record["date"] = str(pd.Timestamp(date).date())
            closest_analogs.append(record)
        payload = {
            "as_of_date": str(snapshot.as_of_date.date()),
            "regime_label": snapshot.regime_label,
            "composite_score": snapshot.composite_score,
            "component_coverage": snapshot.component_coverage,
            "component_scores": snapshot.component_scores,
            "distribution": analog_result.distribution.to_dict("records"),
            "closest_analogs": closest_analogs,
        }
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        print(f"Market state as of {snapshot.as_of_date.date()}: {snapshot.regime_label}")
        print(f"Composite score: {snapshot.composite_score:+.2f}")
        print("Component scores:")
        for component, score in snapshot.component_scores.items():
            print(f"  {component}: {score:+.2f}" if not pd.isna(score) else f"  {component}: n/a")
        print("")
        print(analog_result.distribution.to_string(index=False))
        if wrote_report:
            print("")
            print(f"Report written: {args.report_out}")


def _write_frame(frame: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path)
    elif path.suffix == ".csv":
        frame.to_csv(path)
    else:
        raise ValueError("Output path must end in .parquet or .csv")


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
