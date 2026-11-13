from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.macro_regime_forecast import write_macro_regime_forecast_report
from world_model.states.macro_regime_forecast import (
    run_macro_regime_forecast,
    write_macro_regime_forecast_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Forecast entry into and persistence of market-relevant macro risk regimes."
    )
    parser.add_argument(
        "--regime-tape",
        default=str(root / "outputs" / "macro_regime_performance" / "macro_regime_tape.parquet"),
    )
    parser.add_argument("--config", default=str(default_config_path("macro_regime_forecast.yml")))
    parser.add_argument(
        "--output-dir", default=str(root / "outputs" / "macro_regime_forecast")
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_macro_regime_forecast(
        pd.read_parquet(Path(args.regime_tape).expanduser()), load_yaml(args.config)
    )
    paths = write_macro_regime_forecast_outputs(result, args.output_dir)
    report_path = write_macro_regime_forecast_report(
        result, Path(args.output_dir).expanduser() / "macro_regime_forecast_report.md"
    )
    candidates = int(result.summary["research_candidate"].sum()) if not result.summary.empty else 0
    print("Macro regime forecast research complete")
    print(f"Out-of-sample forecasts: {len(result.forecasts):,}")
    print(f"Research candidates: {candidates:,}")
    print(f"Summary: {paths['summary']}")
    print(f"Current: {paths['current']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
