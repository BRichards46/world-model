"""Backtest and run the bottom-up energy and headline CPI nowcast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.config import load_yaml, project_root
from world_model.states.energy_cpi import ENERGY_BLOCKS, backtest, load_tables, nowcast, site_row, validation

BLOCKS = ("gasoline", *ENERGY_BLOCKS[1:], "energy", "food", "core", "headline")


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", default=str(root / "config" / "data_roots.yml"))
    parser.add_argument("--forecasts", default=str(root / "outputs" / "macro_release_forecast" / "macro_release_forecasts.parquet"))
    parser.add_argument("--current", default=str(root / "outputs" / "macro_release_forecast" / "macro_release_current_forecasts.csv"))
    parser.add_argument(
        "--core-bottom-up",
        default=str(root / "outputs" / "core_cpi" / "core_cpi_backtest.parquet"),
        help="Use the bottom-up core CPI nowcast for the core block when this backtest exists.",
    )
    parser.add_argument("--benchmarks", default=str(root / "outputs" / "release_benchmarks" / "release_comparison.parquet"))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "energy_cpi"))
    parser.add_argument("--start", default="2013-06-01")
    parser.add_argument(
        "--site-json",
        default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "release_forecasts.json"),
        help="Release forecast payload whose headline CPI row is replaced by the bottom-up nowcast.",
    )
    parser.add_argument("--as-of", help="Cutoff for the live nowcast (default: today).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = load_yaml(args.data_roots)
    tables = load_tables(roots["fred"])
    core_path = Path(args.core_bottom_up).expanduser()
    ours = _bottom_up_core(core_path) or _core_forecasts(Path(args.forecasts))
    results = backtest(tables, start=args.start, core_forecasts=ours)

    summary = {"blocks": {}, "headline_vs_benchmarks": None}
    for name in BLOCKS:
        sample = results.dropna(subset=[f"{name}_forecast", f"{name}_actual"])
        error = sample[f"{name}_actual"] - sample[f"{name}_forecast"]
        summary["blocks"][name] = {
            "releases": int(len(sample)),
            "mae": float(error.abs().mean()),
            "actual_sd": float(sample[f"{name}_actual"].std()),
            "correlation": float(sample[f"{name}_actual"].corr(sample[f"{name}_forecast"])),
        }
    sample = results.dropna(subset=["headline_given_core_food", "headline_actual"])
    summary["headline_given_actual_core_and_food_mae"] = float((sample["headline_actual"] - sample["headline_given_core_food"]).abs().mean())

    bench_path = Path(args.benchmarks)
    if bench_path.exists():
        bench = pd.read_parquet(bench_path)
        bench = bench[bench["target_id"] == "headline_cpi_mom"][["reference_month", "kalshi", "cleveland", "naive"]]
        # The benchmark file now scores the bottom-up nowcast as ours, so the old model is read from its own output.
        bench = bench.merge(_pipeline(Path(args.forecasts), "headline_cpi_mom").rename("ours"), left_on="reference_month", right_index=True, how="left")
        joined = results[results["core_source"] == "ours"].rename(columns={"month": "reference_month"}).merge(bench, on="reference_month", how="left")
        joined = joined.rename(columns={"headline_forecast": "bottom_up", "headline_actual": "actual", "ours": "previous_model"})
        summary["headline_vs_benchmarks"] = {
            "with_cleveland": _scores(joined, ["bottom_up", "previous_model", "cleveland", "naive"]),
            "with_market": _scores(joined, ["bottom_up", "previous_model", "kalshi", "cleveland"]),
        }

    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().normalize()
    month = (cutoff - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    released = results["month"].max() if len(results) else None
    if released is not None and month <= released:
        month = released + pd.DateOffset(months=1)
    current_core = _current_bottom_up_core(core_path.with_name("core_cpi_summary.json"), month)
    if current_core is None:
        current_core = _current_core(Path(args.current), month)
    live = nowcast(tables, month, cutoff, current_core)
    summary["current"] = {
        "month": f"{month:%Y-%m}",
        "cutoff": f"{cutoff:%Y-%m-%d}",
        "core_source": "ours" if current_core is not None else "trailing average",
        **{name: live.get(name) for name in ("retail_gasoline_change", "gasoline_nsa", *BLOCKS)},
        "energy_weights": dict(zip(ENERGY_BLOCKS, live.get("energy_weights") or [])),
        "headline_weights": dict(zip(("core", "food", "energy"), live.get("headline_weights") or [])),
    }

    checks = validation(results, _baseline(Path(args.forecasts)))
    summary["validation"] = checks
    site = Path(args.site_json).expanduser()
    if site.exists():
        payload = json.loads(site.read_text(encoding="utf-8"))
        forecasts = payload.get("forecasts", [])
        index = next((i for i, f in enumerate(forecasts) if f.get("target_id") == "headline_cpi_mom"), None)
        row = site_row(forecasts[index] if index is not None else None, live, results, checks)
        if row is not None:
            if index is None:
                forecasts.append(row)
            else:
                forecasts[index] = row
            payload["forecasts"] = forecasts
            site.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
            print(f"  Headline CPI on the dashboard now uses the bottom-up nowcast ({'validated' if checks['validated'] else 'preview'}).")

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output / "energy_cpi_backtest.parquet", index=False)
    (output / "energy_cpi_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Energy CPI backtest: {len(results)} releases -> {output}")
    for name in BLOCKS:
        block = summary["blocks"][name]
        print(f"  {name:12s} MAE {block['mae']:.3f} (actual sd {block['actual_sd']:.3f}, n {block['releases']})")
    current = summary["current"]
    print(f"  Nowcast {current['month']}: gasoline {current['gasoline']:+.2f}%, energy {current['energy']:+.2f}%, headline {current['headline']:+.2f}%")
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


def _core_forecasts(path: Path) -> dict[pd.Timestamp, float]:
    if not path.exists():
        return {}
    table = pd.read_parquet(path)
    table = table[(table["target_id"] == "core_cpi_mom") & (table["variant_id"] == "pipeline")]
    return {pd.Timestamp(r.observation_date).to_period("M").to_timestamp(): float(r.forecast) for r in table.itertuples()}


def _bottom_up_core(path: Path) -> dict[pd.Timestamp, float]:
    if not path.exists():
        return {}
    table = pd.read_parquet(path).dropna(subset=["core_forecast"])
    return {pd.Timestamp(r.month): float(r.core_forecast) for r in table.itertuples()}


def _current_bottom_up_core(path: Path, month: pd.Timestamp) -> float | None:
    if not path.exists():
        return None
    current = json.loads(path.read_text(encoding="utf-8")).get("current", {})
    if current.get("core") is None or current.get("month") != f"{month:%Y-%m}":
        return None
    return float(current["core"])


def _pipeline(path: Path, target_id: str) -> pd.Series:
    """The release module's single-equation forecast for ``target_id``, keyed by month."""
    if not path.exists():
        return pd.Series(dtype=float)
    table = pd.read_parquet(path)
    table = table[(table["target_id"] == target_id) & (table["variant_id"] == "pipeline")]
    months = pd.to_datetime(table["observation_date"]).dt.to_period("M").dt.to_timestamp()
    return pd.Series(table["forecast"].to_numpy(float), index=months)


def _baseline(path: Path) -> pd.Series:
    """The release module's autoregressive benchmark for headline CPI, keyed by month."""
    if not path.exists():
        return pd.Series(dtype=float)
    table = pd.read_parquet(path)
    table = table[(table["target_id"] == "headline_cpi_mom") & (table["variant_id"] == "pipeline")]
    months = pd.to_datetime(table["observation_date"]).dt.to_period("M").dt.to_timestamp()
    return pd.Series(table["baseline_forecast"].to_numpy(float), index=months)


def _current_core(path: Path, month: pd.Timestamp) -> float | None:
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table[(table["target_id"] == "core_cpi_mom") & (table["variant_id"] == "pipeline")]
    rows = rows[pd.to_datetime(rows["forecast_observation_date"]).dt.to_period("M").dt.to_timestamp() == month]
    return float(rows["forecast"].iloc[0]) if len(rows) else None


if __name__ == "__main__":
    raise SystemExit(main())
