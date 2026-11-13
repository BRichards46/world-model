from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.states.quad_forecast_research import (
    calibrated_probabilities,
    run_forecast_research,
    score_probabilities,
    score_variants,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Compare quad forecasting variants walk-forward.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--quad-config", default=str(default_config_path("growth_inflation_quad.yml")))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--start", default="1998-01-31")
    parser.add_argument("--output-dir", default=str(root / "outputs" / "quad_forecast_research"))
    parser.add_argument("--from-records", action="store_true", help="Rebuild the report from saved records instead of rerunning.")
    return parser


def paired_gain(records: pd.DataFrame, candidate: tuple[str, str], baseline: tuple[str, str], repetitions: int = 4000) -> pd.DataFrame:
    """Quad hit-rate gain of ``candidate`` over ``baseline`` with a bootstrap clustered by target quarter."""
    scored = records.dropna(subset=["realized_quad", "forecast_quad"]).copy()
    scored["hit"] = (scored["forecast_quad"] == scored["realized_quad"]).astype(float)
    keys = ["forecast_date", "quarter", "horizon"]

    def pick(variant: tuple[str, str]) -> pd.Series:
        rows = scored[(scored["growth_variant"] == variant[0]) & (scored["inflation_variant"] == variant[1])]
        return rows.set_index(keys)["hit"]

    joined = pd.concat([pick(baseline).rename("base"), pick(candidate).rename("new")], axis=1).dropna().reset_index()
    rng = np.random.default_rng(3)
    rows = []
    for horizon, group in joined.groupby("horizon"):
        per = group.groupby("quarter")[["base", "new"]].agg(["sum", "count"])
        draws = []
        for _ in range(repetitions):
            sample = per.iloc[rng.integers(0, len(per), len(per))]
            draws.append(sample[("new", "sum")].sum() / sample[("new", "count")].sum() - sample[("base", "sum")].sum() / sample[("base", "count")].sum())
        draws = np.asarray(draws)
        rows.append(
            {
                "horizon": int(horizon),
                "baseline_hit": float(group["base"].mean()),
                "candidate_hit": float(group["new"].mean()),
                "gain": float(group["new"].mean() - group["base"].mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "p_no_gain": float(np.mean(draws <= 0)),
                "target_quarters": int(len(per)),
            }
        )
    return pd.DataFrame.from_records(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Plain markdown table (avoids pandas' optional tabulate dependency)."""
    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else pd.Timestamp.today().normalize()
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if args.from_records:
        records = pd.read_parquet(output / "quad_forecast_records.parquet")
    else:
        records = run_forecast_research(load_yaml(args.data_roots), load_yaml(args.quad_config), as_of, start=args.start)
        records.to_parquet(output / "quad_forecast_records.parquet")
    summary = score_variants(records)
    summary.to_csv(output / "quad_forecast_variants.csv", index=False)
    gain = paired_gain(records, ("leading", "energy"), ("baseline", "baseline"))
    gain.to_csv(output / "quad_forecast_gain.csv", index=False)
    probability_rows = []
    for growth, inflation in (("baseline", "baseline"), ("leading", "energy")):
        subset = records[(records["growth_variant"] == growth) & (records["inflation_variant"] == inflation)]
        scores = score_probabilities(calibrated_probabilities(subset))
        scores.insert(0, "variant", f"{growth}+{inflation}")
        probability_rows.append(scores)
    probabilities = pd.concat(probability_rows, ignore_index=True)
    probabilities.to_csv(output / "quad_forecast_probability_scores.csv", index=False)

    lines = [
        "# Quad forecast research",
        "",
        f"Walk-forward from {args.start} to {as_of:%Y-%m-%d}, month-end forecasts scored against latest revised quads.",
        "",
        "## Variants (quad, growth-direction and inflation-direction hit rates)",
        "",
        markdown_table(summary),
        "",
        "## Production model (leading growth + energy inflation) vs baseline",
        "",
        "Bootstrap clustered by target quarter.",
        "",
        markdown_table(gain),
        "",
        "## Probability scoring (Brier, lower is better; uniform 25% = 0.75)",
        "",
        markdown_table(probabilities),
        "",
    ]
    report = output / "quad_forecast_research_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(gain.round(3).to_string(index=False))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
