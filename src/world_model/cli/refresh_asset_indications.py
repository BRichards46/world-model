from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from world_model.config import project_root
from world_model.features.asset_indications import build_asset_indications


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Refresh browser-ready asset indications from current research outputs."
    )
    parser.add_argument("--snapshot", default=str(root / "outputs" / "macro_dashboard_snapshot.parquet"))
    parser.add_argument(
        "--risk-overlay",
        default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "risk_overlay.json"),
    )
    parser.add_argument(
        "--market-summary",
        default=str(root / "outputs" / "macro_market_validation" / "macro_market_summary.csv"),
    )
    parser.add_argument(
        "--regime-summary",
        default=str(root / "outputs" / "macro_regime_validation" / "macro_regime_summary.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "asset_indications.json"),
    )
    parser.add_argument("--deadband-percentile-points", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    snapshot = pd.read_parquet(Path(args.snapshot).expanduser())
    risk_overlay = json.loads(Path(args.risk_overlay).expanduser().read_text(encoding="utf-8"))
    indications = build_asset_indications(
        snapshot,
        risk_overlay,
        args.market_summary,
        deadband_percentile_points=args.deadband_percentile_points,
        regime_summary_path=args.regime_summary,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(indications, indent=2) + "\n", encoding="utf-8")
    print(f"Asset indications refreshed: {output}")


if __name__ == "__main__":
    main()
