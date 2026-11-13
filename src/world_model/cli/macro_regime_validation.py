from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.states.macro_regime_validation import (
    run_macro_regime_validation,
    write_macro_regime_validation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Test pre-specified macro combinations against future market outcomes."
    )
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--asset-config", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--validation-config", default=str(default_config_path("macro_regime_validation.yml")))
    parser.add_argument("--macro-history", default=str(root / "outputs" / "macro_dashboard_history.parquet"))
    parser.add_argument("--market-history", default=str(root / "outputs" / "market_state_history.parquet"))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "macro_regime_validation"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_macro_regime_validation(
        macro_history=pd.read_parquet(Path(args.macro_history).expanduser()),
        market_state_history=pd.read_parquet(Path(args.market_history).expanduser()),
        data_roots=load_yaml(args.data_roots),
        asset_config=load_yaml(args.asset_config),
        config=load_yaml(args.validation_config),
        start=args.start,
        end=args.end,
    )
    paths = write_macro_regime_validation_outputs(result, args.output_dir)
    robust = int(result.summary["robust_candidate"].sum()) if not result.summary.empty else 0
    additive = int(result.summary["additive_robust_candidate"].sum()) if not result.summary.empty else 0
    print("Macro regime validation complete")
    print(f"Forecast rows: {len(result.forecasts):,}")
    print(f"Summary rows: {len(result.summary):,}")
    print(f"Robust combinations: {robust}")
    print(f"Robust additive pairs: {additive}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
