from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.data.benchmarks import load_equity_benchmark_history
from world_model.states.equity_risk_context import (
    build_risk_context,
    classify_states,
    current_state,
    forward_outcomes,
    production_inflation_read,
)
from world_model.states.gtt_threshold_validation import METRICS, build_point_in_time_values


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Describe how much risk the current equity-overlay state has carried.")
    parser.add_argument("--config", default=str(default_config_path("macro_dashboard.yml")))
    parser.add_argument("--data-roots", default=str(root / "config" / "data_roots.yml"))
    parser.add_argument("--data-dir", default=str(root / "sites" / "macro-dashboard" / "dist" / "data"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    config = load_yaml(args.config)
    settings = config.get("equity_risk_context", {})
    if not settings.get("enabled", False):
        print("Equity risk context is disabled.")
        return 0
    overlay_settings = config.get("equity_risk_overlay", {})
    roots = load_yaml(args.data_roots)
    triggers = {metric: (str(rule[0]), float(rule[1])) for metric, rule in settings["legacy_triggers"].items()}
    at_least = int(settings.get("legacy_at_least", 2))
    trend_months = int(overlay_settings.get("moving_average_months", 10))
    buffer = float(overlay_settings.get("unemployment_buffer_pp", 0.1))

    values = build_point_in_time_values(roots["fred"], {metric: METRICS[metric] for metric in triggers})
    daily = load_equity_benchmark_history(roots, overlay_settings)["normalized_index"].dropna()
    month_ends = daily.groupby(daily.index.to_period("M")).tail(1)

    states = classify_states(month_ends, values, triggers, at_least, trend_months, buffer)
    states = states.loc[pd.Timestamp(settings.get("start", "1972-09-01")) :]
    outcomes = forward_outcomes(daily, states.index, int(settings.get("horizon_days", 63)))
    current = current_state(month_ends, float(daily.iloc[-1]), values, triggers, at_least, trend_months, buffer)

    data_dir = Path(args.data_dir).expanduser()
    snapshot_path = data_dir / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else []
    summary_path = root / str(settings.get("regime_summary", ""))
    summary = pd.read_csv(summary_path) if summary_path.is_file() else None
    production = production_inflation_read(snapshot, summary, float(settings.get("elevated_percentile", 60)))

    context = build_risk_context(states, outcomes, current, production, float(settings.get("worst_share", 0.10)))
    context["as_of_date"] = f"{daily.index[-1]:%Y-%m-%d}"
    target = data_dir / "risk_context.json"
    target.write_text(json.dumps(context, indent=2) + "\n")
    print(f"Risk context: {context['state']} ({context['tier']} risk) -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
