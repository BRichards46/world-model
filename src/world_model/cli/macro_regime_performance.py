from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.macro_regime_performance import write_macro_regime_performance_report
from world_model.states.macro_regime_performance import (
    run_macro_regime_performance,
    write_macro_regime_performance_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Measure asset performance while point-in-time macro regimes are active."
    )
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--asset-config", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--regime-config", default=str(default_config_path("macro_regime_performance.yml")))
    parser.add_argument("--macro-history", default=str(root / "outputs" / "macro_dashboard_history.parquet"))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "macro_regime_performance"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_macro_regime_performance(
        macro_history=pd.read_parquet(Path(args.macro_history).expanduser()),
        data_roots=load_yaml(args.data_roots),
        asset_config=load_yaml(args.asset_config),
        config=load_yaml(args.regime_config),
    )
    paths = write_macro_regime_performance_outputs(result, args.output_dir)
    report_path = write_macro_regime_performance_report(
        result, Path(args.output_dir).expanduser() / "macro_regime_performance_report.md"
    )
    current = result.regime_tape["macro_regime"].dropna()
    print("Macro regime performance analysis complete")
    print(f"Current regime: {current.iloc[-1] if not current.empty else 'unavailable'}")
    print(f"Daily tape rows: {len(result.regime_tape):,}")
    print(f"Episode rows: {len(result.episodes):,}")
    print(f"Performance rows: {len(result.performance):,}")
    print(f"Performance: {paths['performance']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
