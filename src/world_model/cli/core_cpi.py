"""Backtest and run the bottom-up core CPI nowcast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.config import load_yaml, project_root
from world_model.states.core_cpi import GROUPS, backtest, load_relative_importance, load_tables, nowcast, site_row, validation


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", default=str(root / "config" / "data_roots.yml"))
    parser.add_argument("--forecasts", default=str(root / "outputs" / "macro_release_forecast" / "macro_release_forecasts.parquet"))
    parser.add_argument("--benchmarks", default=str(root / "outputs" / "release_benchmarks" / "release_comparison.parquet"))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "core_cpi"))
    parser.add_argument("--start", default="2011-06-01")
    parser.add_argument(
        "--site-json",
        default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "release_forecasts.json"),
        help="Release forecast payload whose core CPI row is replaced by the bottom-up nowcast.",
    )
    parser.add_argument("--as-of", help="Cutoff for the live nowcast (default: today).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = load_yaml(args.data_roots)
    tables = load_tables(roots["fred"])
    weights = load_relative_importance(roots["bls"])
    results = backtest(tables, weights, start=args.start)

    summary: dict = {"releases": int(len(results)), "core_vs_benchmarks": None}
    sample = results.dropna(subset=["core_forecast", "core_actual"])
    summary["mae"] = float((sample["core_actual"] - sample["core_forecast"]).abs().mean())
    bench_path = Path(args.benchmarks)
    if bench_path.exists():
        bench = pd.read_parquet(bench_path)
        bench = bench[bench["target_id"] == "core_cpi_mom"][["reference_month", "kalshi", "cleveland", "naive"]]
        # The benchmark file now scores the bottom-up nowcast as ours, so the old model is read from its own output.
        bench = bench.merge(_pipeline(Path(args.forecasts), "core_cpi_mom").rename("ours"), left_on="reference_month", right_index=True, how="left")
        joined = results.rename(columns={"month": "reference_month"}).merge(bench, on="reference_month", how="left")
        joined = joined.rename(columns={"core_forecast": "bottom_up", "core_actual": "actual", "ours": "previous_model"})
        summary["core_vs_benchmarks"] = {
            "with_cleveland": _scores(joined, ["bottom_up", "previous_model", "cleveland", "naive"]),
            "with_market": _scores(joined, ["bottom_up", "previous_model", "kalshi", "cleveland"]),
        }

    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().normalize()
    month = (cutoff - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    released = results["month"].max() if len(results) else None
    if released is not None and month <= released:
        month = released + pd.DateOffset(months=1)
    live = nowcast(tables, weights, month, cutoff)
    summary["current"] = {
        "month": f"{month:%Y-%m}",
        "cutoff": f"{cutoff:%Y-%m-%d}",
        "core": live.get("core"),
        "previous": (live.get("previous") or {}).get("core"),
        "groups": live.get("groups"),
        "blocks": live.get("blocks"),
        "windows": live.get("windows"),
        "shares": live.get("shares"),
    }

    checks = validation(results, _baseline(Path(args.forecasts)))
    summary["validation"] = checks
    site = Path(args.site_json).expanduser()
    if site.exists():
        payload = json.loads(site.read_text(encoding="utf-8"))
        forecasts = payload.get("forecasts", [])
        index = next((i for i, f in enumerate(forecasts) if f.get("target_id") == "core_cpi_mom"), None)
        row = site_row(forecasts[index] if index is not None else None, live, results, checks)
        if row is not None:
            if index is None:
                forecasts.append(row)
            else:
                forecasts[index] = row
            payload["forecasts"] = forecasts
            site.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
            print(f"  Core CPI on the dashboard now uses the bottom-up nowcast ({'validated' if checks['validated'] else 'preview'}).")

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output / "core_cpi_backtest.parquet", index=False)
    (output / "core_cpi_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Core CPI backtest: {len(results)} releases, MAE {summary['mae']:.3f} -> {output}")
    current = summary["current"]
    if current["core"] is not None:
        groups = ", ".join(f"{name.replace('_', ' ')} {current['groups'][name]:+.3f}" for name in GROUPS)
        print(f"  Nowcast {current['month']}: core {current['core']:+.2f}% (contributions: {groups})")
    return 0


def _scores(table: pd.DataFrame, columns: list[str]) -> dict:
    sample = table.dropna(subset=["actual", *columns])
    out = {"releases": int(len(sample))}
    if len(sample):
        out["start"], out["end"] = f"{sample['reference_month'].min():%Y-%m}", f"{sample['reference_month'].max():%Y-%m}"
    for column in columns:
        error = sample["actual"] - sample[column]
        out[column] = {"mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean()))}
    return out


def _pipeline(path: Path, target_id: str) -> pd.Series:
    """The release module's single-equation forecast for ``target_id``, keyed by month."""
    if not path.exists():
        return pd.Series(dtype=float)
    table = pd.read_parquet(path)
    table = table[(table["target_id"] == target_id) & (table["variant_id"] == "pipeline")]
    months = pd.to_datetime(table["observation_date"]).dt.to_period("M").dt.to_timestamp()
    return pd.Series(table["forecast"].to_numpy(float), index=months)


def _baseline(path: Path) -> pd.Series:
    """The release module's autoregressive benchmark for core CPI, keyed by month."""
    if not path.exists():
        return pd.Series(dtype=float)
    table = pd.read_parquet(path)
    table = table[(table["target_id"] == "core_cpi_mom") & (table["variant_id"] == "pipeline")]
    months = pd.to_datetime(table["observation_date"]).dt.to_period("M").dt.to_timestamp()
    return pd.Series(table["baseline_forecast"].to_numpy(float), index=months)


if __name__ == "__main__":
    raise SystemExit(main())
