from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from world_model.config import load_yaml, project_root
from world_model.data import cleveland_nowcast, kalshi
from world_model.data.fred import load_vintage_table
from world_model.states.release_benchmarks import (
    KALSHI_SERIES,
    build_comparison,
    build_site_payload,
    current_expectations,
    kalshi_expectations,
    our_forecasts,
    prior_payroll_levels,
    summarize,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Score release forecasts against Kalshi and the Cleveland Fed nowcast.")
    parser.add_argument("--data-roots", default=str(root / "config" / "data_roots.yml"))
    parser.add_argument(
        "--forecasts",
        default=str(root / "outputs" / "macro_release_forecast" / "macro_release_forecasts.parquet"),
    )
    parser.add_argument(
        "--current",
        default=str(root / "outputs" / "macro_release_forecast" / "macro_release_current_forecasts.csv"),
    )
    parser.add_argument("--output-dir", default=str(root / "outputs" / "release_benchmarks"))
    parser.add_argument(
        "--headline-bottom-up",
        default=str(root / "outputs" / "energy_cpi" / "energy_cpi_backtest.parquet"),
        help="Score headline CPI on the bottom-up nowcast when this backtest exists.",
    )
    parser.add_argument(
        "--core-bottom-up",
        default=str(root / "outputs" / "core_cpi" / "core_cpi_backtest.parquet"),
        help="Score core CPI on the bottom-up nowcast when this backtest exists.",
    )
    parser.add_argument(
        "--site-json-out",
        default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "release_benchmarks.json"),
    )
    parser.add_argument("--offline", action="store_true", help="Use cached Kalshi and Cleveland Fed data only.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    roots = load_yaml(args.data_roots)
    data_root = Path(roots["fred"]).expanduser().parent
    kalshi_dir = Path(roots.get("kalshi", data_root / "kalshi")).expanduser() / "release_ladders"
    nowcast_path = Path(roots.get("cleveland_nowcast", data_root / "cleveland_nowcast")).expanduser() / "nowcast_month.json"

    if not args.offline:
        cleveland_nowcast.download(nowcast_path)
    cleveland = cleveland_nowcast.load_release_nowcasts(nowcast_path)

    ladders = []
    for series in KALSHI_SERIES:
        cached = kalshi_dir / f"{series}.parquet"
        if args.offline:
            if cached.exists():
                ladders.append(pd.read_parquet(cached))
        else:
            ladders.append(kalshi.update_ladders(series, kalshi_dir))
    market = kalshi_expectations(pd.concat(ladders, ignore_index=True)) if ladders else pd.DataFrame()

    forecasts = pd.read_parquet(args.forecasts)
    payroll_rows = forecasts[forecasts["target_id"] == "payroll_change"].assign(
        reference_month=lambda f: pd.to_datetime(f["observation_date"]).dt.to_period("M").dt.to_timestamp()
    )[["reference_month", "release_date"]].drop_duplicates("reference_month")
    levels = prior_payroll_levels(load_vintage_table(roots["fred"], "PAYEMS"), payroll_rows)
    bottom_up_paths = {
        "headline_cpi_mom": (Path(args.headline_bottom_up).expanduser(), "headline", "energy_cpi_summary.json"),
        "core_cpi_mom": (Path(args.core_bottom_up).expanduser(), "core", "core_cpi_summary.json"),
    }
    bottom_up = {}
    for target_id, (path, column, _) in bottom_up_paths.items():
        if path.exists():
            table = pd.read_parquet(path)
            bottom_up[target_id] = pd.Series(table[f"{column}_forecast"].to_numpy(float), index=pd.to_datetime(table["month"]))
    comparison = build_comparison(our_forecasts(forecasts, levels, bottom_up), market, cleveland)
    summary = summarize(comparison)

    upcoming = []
    if not args.offline:
        open_ladders = kalshi.open_ladders(KALSHI_SERIES)
        open_market = kalshi_expectations(open_ladders) if not open_ladders.empty else pd.DataFrame()
        current_ours = _current_ours(Path(args.current))
        for target_id, (path, column, summary_name) in bottom_up_paths.items():
            current_ours.update(_current_bottom_up(path.with_name(summary_name), target_id, column))
        upcoming = current_expectations(open_market, cleveland, current_ours)

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(output / "release_comparison.parquet", index=False)
    comparison.to_csv(output / "release_comparison.csv", index=False)
    (output / "release_benchmark_summary.json").write_text(
        json.dumps({"summary": summary, "upcoming": upcoming}, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not args.offline:
        site = Path(args.site_json_out).expanduser()
        site.parent.mkdir(parents=True, exist_ok=True)
        payload = build_site_payload(summary, upcoming, f"{pd.Timestamp.now():%Y-%m-%d}")
        site.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Wrote {site}")
    print(f"Wrote {output}")
    for target_id, record in summary.items():
        common = record["all_sources"]
        parts = [f"{k} {v['mae']:.3f}" for k, v in common.items() if isinstance(v, dict) and "mae" in v]
        print(f"  {record['label']}: {common.get('releases', 0)} common releases, MAE " + ", ".join(parts))


def _current_bottom_up(path: Path, target_id: str, column: str) -> dict[str, float]:
    if not path.exists():
        return {}
    current = json.loads(path.read_text(encoding="utf-8")).get("current", {})
    if current.get(column) is None:
        return {}
    return {f"{target_id}:{current['month']}": float(current[column])}


def _current_ours(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    current = pd.read_csv(path)
    keep = {"core_cpi_mom": "pipeline", "core_pce_mom": "pipeline", "payroll_change": "core"}
    values = {}
    for row in current.itertuples(index=False):
        if keep.get(row.target_id) != row.variant_id:
            continue
        month = pd.Timestamp(row.forecast_observation_date).to_period("M").to_timestamp()
        values[f"{row.target_id}:{month:%Y-%m}"] = float(row.display_forecast)
    return values


if __name__ == "__main__":
    main()
