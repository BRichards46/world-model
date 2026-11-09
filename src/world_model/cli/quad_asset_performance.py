from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.quad_asset_performance import build_site_payload, write_report
from world_model.states.growth_inflation_quad import GrowthInflationQuadEngine
from world_model.states.quad_asset_performance import load_cumulative_assets, run_quad_asset_performance, write_outputs
from world_model.states.quad_playbook_test import run_playbook_test


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test forward asset performance by point-in-time growth/inflation quad.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--quad-config", default=str(default_config_path("growth_inflation_quad.yml")))
    parser.add_argument("--config", default=str(default_config_path("quad_asset_performance.yml")))
    parser.add_argument("--asset-config", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--output-dir", default=str(root / "outputs" / "quad_asset_performance"))
    parser.add_argument("--site-json-out", default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "quad_asset_performance.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    config = load_yaml(args.config)
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else pd.Timestamp.today().normalize()
    engine = GrowthInflationQuadEngine(data_roots, load_yaml(args.quad_config))
    walk_forward = engine.walk_forward(as_of)
    asset_groups = load_yaml(args.asset_config).get("asset_groups", {})
    result = run_quad_asset_performance(walk_forward, data_roots, config, asset_groups, as_of)
    paths = write_outputs(result, args.output_dir)
    cumulative, _ = load_cumulative_assets(data_roots, config, {})
    playbook = run_playbook_test(
        engine,
        result.signals,
        cumulative,
        cumulative[str(config.get("excess_benchmark", "sp500"))].dropna().index,
        config,
        {str(name): int(days) for name, days in config.get("horizons", {}).items()},
        int(config.get("availability_lag_trading_days", 1)),
        float(config.get("false_discovery_rate", 0.10)),
        int(config.get("minimum_rotation_gap_months", 12)),
    )
    playbook_path = Path(args.output_dir).expanduser() / "quad_playbook_summary.csv"
    playbook.summary.to_csv(playbook_path, index=False)
    report = write_report(result, config, Path(args.output_dir) / "quad_asset_performance_report.md", playbook.summary)
    site_path = Path(args.site_json_out).expanduser()
    site_path.parent.mkdir(parents=True, exist_ok=True)
    site_path.write_text(build_site_payload(result, config, playbook.summary), encoding="utf-8")
    robust = int(result.quad_summary["robust"].sum()) if not result.quad_summary.empty else 0
    print(f"Quad asset performance as of {as_of.date()}: {len(result.quad_summary)} quad × asset tests, {robust} robust")
    for name, path in {**paths, "playbook": playbook_path, "report": report, "site": site_path}.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
