from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.states.growth_inflation_quad import GrowthInflationQuadEngine


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build growth/inflation quads with a base-effect look-ahead.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--config", default=str(default_config_path("growth_inflation_quad.yml")))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--no-backtest", action="store_true", help="Skip the month-end walk-forward check.")
    parser.add_argument("--snapshot-json-out", default=str(root / "outputs" / "growth_inflation_quad_snapshot.json"))
    parser.add_argument("--backtest-out", default=str(root / "outputs" / "growth_inflation_quad_backtest.parquet"))
    parser.add_argument("--site-json-out", default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "growth_inflation_quad.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = GrowthInflationQuadEngine(load_yaml(args.data_roots), load_yaml(args.config))
    result = engine.build(as_of=args.as_of, run_backtest=not args.no_backtest)
    text = json.dumps(result.payload, indent=2) + "\n"
    for raw_path in [args.snapshot_json_out, args.site_json_out]:
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if not result.backtest.empty:
        path = Path(args.backtest_out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        result.backtest.to_parquet(path)

    print(f"Growth/inflation quad as of {result.as_of_date.date()} (current quarter {result.payload['current_quarter']})")
    for row in result.payload["quarters"]:
        if row["status"] == "realized" and row is not result.payload["quarters"][-1]:
            continue
        probabilities = row.get("probabilities")
        detail = (
            " · " + " ".join(f"Q{quad} {value:.0%}" for quad, value in probabilities.items())
            if probabilities
            else ""
        )
        print(f"  {row['quarter']} [{row['status']}] base case Quad {row['quad']}{detail}")
    print(f"Site data: {Path(args.site_json_out).expanduser()}")


if __name__ == "__main__":
    main()
