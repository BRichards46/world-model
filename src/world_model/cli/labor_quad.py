"""Build the payroll growth × inflation quad, its walk-forward, Fed-path context and asset tests.

Runs after ``growth_inflation_quad``: the inflation leg is that model's forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.quad_asset_performance import build_site_payload
from world_model.states.labor_quad import (
    LaborQuadEngine,
    fed_path_by_quad,
    known_quads_from,
    summarize_walk_forward,
)
from world_model.states.quad_asset_performance import build_quad_signals, load_cumulative_assets, run_quad_asset_performance, write_outputs
from world_model.states.quad_playbook_test import run_playbook_test

EVIDENCE_HORIZON = "three_months"


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--config", default=str(default_config_path("labor_quad.yml")))
    parser.add_argument("--asset-config", default=str(default_config_path("quad_asset_performance.yml")))
    parser.add_argument("--asset-groups", default=str(default_config_path("economic_asset_drivers.yml")))
    parser.add_argument("--gdp-snapshot", default=str(root / "outputs" / "growth_inflation_quad_snapshot.json"))
    parser.add_argument("--gdp-backtest", default=str(root / "outputs" / "growth_inflation_quad_backtest.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--no-assets", action="store_true", help="Skip the asset and playbook tests.")
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    parser.add_argument("--site-json-out", default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "labor_quad.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = load_yaml(args.data_roots)
    config = load_yaml(args.config)
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else pd.Timestamp.today().normalize()
    gdp_snapshot = json.loads(Path(args.gdp_snapshot).expanduser().read_text(encoding="utf-8"))
    gdp_backtest = pd.read_parquet(Path(args.gdp_backtest).expanduser())

    engine = LaborQuadEngine(roots, config)
    walk = engine.walk_forward(as_of, gdp_backtest)
    payload = engine.outlook(as_of, gdp_snapshot, walk)
    payload["backtest"] = summarize_walk_forward(walk, gdp_backtest)

    signals = build_quad_signals(walk)
    fed_settings = config.get("fed", {})
    fed_funds = _latest(roots["fred"], str(fed_settings.get("series", "DFF")))
    payload["fed"] = {
        "series": str(fed_settings.get("series", "DFF")),
        "signal": "projected_current",
        "paths": [fed_path_by_quad(signals["projected_current"], fed_funds, int(months)) for months in fed_settings.get("months", [6, 12])],
        "note": "Average change in the effective fed funds rate after month-ends with each projected current-quarter payroll quad. "
        "p = circular-rotation test of the spread between quads. Context for the Fed's leaning, not a rates forecast: "
        "no quad predicted the 2-year yield's next move, which the market already prices.",
    }

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    walk.to_parquet(output / "labor_quad_backtest.parquet")
    if not args.no_assets:
        payload["assets"] = _asset_tests(walk, signals, roots, config, args, as_of, output / "quad_asset_performance_labor")

    payload["method"] = {
        "growth": "Nonfarm payrolls (PAYEMS): year-over-year growth of each quarter's average level. Unreported months grow at the "
        "average monthly change of the last three months.",
        "inflation": "Same as the GDP quad: quarterly average of headline CPI year-over-year rates, with that model's forecast for "
        "unreported months, so the two quads differ only in growth.",
        "probabilities": "Growth odds come from this model's own past forecast errors at the same horizon; inflation odds from the "
        "GDP quad. The two are treated as independent.",
        "why": "GDP is quarterly, late and heavily revised: even a perfect forecast of its first print gets the growth direction "
        "right only 86% of the time against revised data. Payrolls are monthly and their direction carries on from quarter "
        "to quarter (76% vs 55% for GDP).",
        "warning": "Descriptive rate-of-change regime, not a return forecast. Asset results are historical conditional averages.",
    }
    text = json.dumps(payload, indent=2, default=_json_default) + "\n"
    for path in (output / "labor_quad_snapshot.json", Path(args.site_json_out).expanduser()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    current = next((row for row in payload["quarters"] if row["is_current_quarter"]), None)
    print(f"Payroll quad as of {as_of.date()} (payrolls through {payload['last_payroll_month']})")
    for row in payload["quarters"]:
        if row["status"] == "realized" and row is not current:
            continue
        odds = row.get("probabilities")
        detail = " · " + " ".join(f"Q{q} {v:.0%}" for q, v in odds.items()) if odds else ""
        print(f"  {row['quarter']} [{row['status']}] Quad {row['quad']}{detail}")
    for horizon in payload["backtest"]["horizons"]:
        print(f"  Walk-forward {horizon['label']}: quad {horizon['quad_hit_rate']:.0%} (GDP quad {horizon.get('gdp_quad_hit_rate', float('nan')):.0%}), "
              f"growth direction {horizon['growth_direction_hit_rate']:.0%} (GDP {horizon.get('gdp_growth_direction_hit_rate', float('nan')):.0%})")
    print(f"Site data: {Path(args.site_json_out).expanduser()}")
    return 0


def _asset_tests(walk, signals, roots, config, args, as_of, directory: Path) -> dict:
    """The existing quad asset and playbook tests, run on the payroll quad's signals."""
    asset_config = dict(load_yaml(args.asset_config))
    asset_config["signals"] = config.get("signals", asset_config.get("signals", {}))
    groups = load_yaml(args.asset_groups).get("asset_groups", {})
    result = run_quad_asset_performance(walk, roots, asset_config, groups, as_of)
    write_outputs(result, directory)
    cumulative, _ = load_cumulative_assets(roots, asset_config, {})
    benchmark = str(asset_config.get("excess_benchmark", "sp500"))
    playbook = run_playbook_test(
        None,
        result.signals,
        cumulative,
        cumulative[benchmark].dropna().index,
        asset_config,
        {str(name): int(days) for name, days in asset_config.get("horizons", {}).items()},
        int(asset_config.get("availability_lag_trading_days", 1)),
        float(asset_config.get("false_discovery_rate", 0.10)),
        int(asset_config.get("minimum_rotation_gap_months", 12)),
        known_quads=known_quads_from(walk, int(config.get("probabilities", {}).get("knowledge_lag_days", 120))),
    )
    playbook.summary.to_csv(directory / "quad_playbook_summary.csv", index=False)
    assets = json.loads(build_site_payload(result, asset_config, playbook.summary))
    summary = result.quad_summary
    evidence = summary[
        (summary["signal"] == "projected_current")
        & (summary["horizon"] == EVIDENCE_HORIZON)
        & summary["evidence"].isin(["robust", "suggestive"])
    ]
    assets["evidence_by_quad"] = {
        str(int(quad)): [
            {
                "asset_label": row.asset_label,
                "annualized_spread": float(row.annualized_spread),
                "early_annualized_spread": float(row.early_annualized_spread),
                "late_annualized_spread": float(row.late_annualized_spread),
                "q_value": float(row.q_value),
                "evidence": row.evidence,
            }
            for row in group.reindex(group["annualized_spread"].abs().sort_values(ascending=False).index).itertuples()
        ]
        for quad, group in evidence.groupby("quad")
    }
    return assets


def _latest(fred_root: str, series_id: str) -> pd.Series:
    frame = pd.read_parquet(Path(fred_root).expanduser() / "latest" / f"{series_id}.parquet")
    series = frame.set_index(pd.to_datetime(frame["observation_date"]))["latest_value"].astype(float)
    return series.dropna().sort_index()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return f"{value:%Y-%m-%d}"
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
