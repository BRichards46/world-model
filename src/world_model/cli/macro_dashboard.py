from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.states.macro_dashboard import (
    MacroDashboardEngine,
    merge_macro_dashboard_history,
    read_macro_dashboard_table,
    write_macro_dashboard_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the normalized point-in-time macro dashboard dataset."
    )
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--dashboard-config", default=str(default_config_path("macro_dashboard.yml")))
    parser.add_argument("--as-of", default=None, help="Dashboard date. Defaults to today.")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Recalculate releases on or after this date. Defaults to an incremental overlap or bootstrap window.",
    )
    parser.add_argument(
        "--history-out",
        default=str(project_root() / "outputs" / "macro_dashboard_history.parquet"),
    )
    parser.add_argument(
        "--snapshot-out",
        default=str(project_root() / "outputs" / "macro_dashboard_snapshot.parquet"),
    )
    parser.add_argument(
        "--snapshot-json-out",
        default=str(project_root() / "outputs" / "macro_dashboard_snapshot.json"),
    )
    parser.add_argument(
        "--site-data-dir",
        default=str(project_root() / "sites" / "macro-dashboard" / "dist" / "data"),
        help="Directory receiving browser-ready snapshot and compact history JSON.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and replace an existing history file using the full available vintage archive.",
    )
    parser.add_argument("--json", action="store_true", help="Print the current snapshot as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_yaml(args.dashboard_config)
    existing = pd.DataFrame() if args.no_cache else read_macro_dashboard_table(args.history_out)
    start_date = args.start_date or (pd.Timestamp("1900-01-01") if args.no_cache else _incremental_start(existing, config))

    engine = MacroDashboardEngine(load_yaml(args.data_roots), config)
    result = engine.build(as_of=args.as_of, start_date=start_date)
    history = (
        result.history
        if args.no_cache
        else merge_macro_dashboard_history(existing, result.history, replace_from=start_date)
    )

    write_macro_dashboard_table(history, args.history_out)
    write_macro_dashboard_table(result.snapshot, args.snapshot_out)
    json_text = result.snapshot.to_json(orient="records", date_format="iso", indent=2)
    json_path = Path(args.snapshot_json_out).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    _write_site_data(
        result.snapshot,
        history,
        result.risk_overlay,
        result.asset_indications,
        args.site_data_dir,
    )

    if args.json:
        print(json_text)
    else:
        stale_count = int(result.snapshot["freshness_status"].eq("stale").sum())
        warning_count = int(result.snapshot["history_status"].ne("available").sum())
        print(f"Macro dashboard as of {result.as_of_date.date()}: {len(result.snapshot)} rows")
        print(f"Stale rows: {stale_count}; history warnings: {warning_count}")
        print(f"History: {args.history_out}")
        print(f"Snapshot: {args.snapshot_out}")
        print(f"Snapshot JSON: {args.snapshot_json_out}")
        print(f"Site data: {args.site_data_dir}")


def _incremental_start(existing: pd.DataFrame, config: dict) -> pd.Timestamp | None:
    if existing.empty or "realtime_start" not in existing:
        return None
    release_dates = pd.to_datetime(existing["realtime_start"], errors="coerce")
    frame = existing.assign(_release_date=release_dates).dropna(subset=["_release_date"])
    if frame.empty:
        return None
    latest_by_indicator = frame.groupby("indicator_id")["_release_date"].max()
    overlap_days = int(config.get("build", {}).get("incremental_overlap_days", 14))
    return pd.Timestamp(latest_by_indicator.min()).normalize() - pd.Timedelta(days=overlap_days)


def _write_site_data(
    snapshot: pd.DataFrame,
    history: pd.DataFrame,
    risk_overlay: dict,
    asset_indications: dict,
    directory: str | Path,
) -> None:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "snapshot.json").write_text(
        snapshot.to_json(orient="records", date_format="iso", indent=2) + "\n",
        encoding="utf-8",
    )
    history_fields = [
        "indicator_id",
        "realtime_start",
        "observation_date",
        "revision_status",
        "is_revision",
        "level_percentile_10y",
        "level_percentile_expanding",
        "velocity_percentile_10y",
        "velocity_percentile_expanding",
        "level_value",
        "velocity_value",
    ]
    compact_history = history[[field for field in history_fields if field in history]].copy()
    compact_history = compact_history.sort_values(["indicator_id", "realtime_start"])
    (destination / "history.json").write_text(
        compact_history.to_json(orient="records", date_format="iso", indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "risk_overlay.json").write_text(
        json.dumps(risk_overlay, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "asset_indications.json").write_text(
        json.dumps(asset_indications, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
