from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.data.benchmarks import load_equity_benchmark_history
from world_model.states.gtt_risk_validation import (
    build_point_in_time_gtt_macro_signals,
    build_point_in_time_production_signals,
    build_price_trends,
    run_gtt_risk_validation,
    write_gtt_risk_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Compare point-in-time production signals for S&P risk.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--gtt-config", default=str(default_config_path("gtt_risk_validation.yml")))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(root / "outputs" / "gtt_production_validation"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_roots = load_yaml(args.data_roots)
    config = load_yaml(args.gtt_config)
    benchmark = config["equity_benchmark"]
    history = load_equity_benchmark_history(data_roots, benchmark)
    asset_id = "sp500"
    returns = history[["log_return"]].rename(columns={"log_return": asset_id})
    trends = build_price_trends(
        returns,
        ten_month_window=int(config["price_trends"]["ten_month"]),
        daily_window=int(config["price_trends"]["two_hundred_day"]),
        return_conventions={asset_id: "log"},
    )
    base = build_point_in_time_gtt_macro_signals(data_roots["fred"])[["industrial_production_growth"]]
    alternatives = build_point_in_time_production_signals(data_roots["fred"])
    macro = pd.concat([base, alternatives], axis=1).sort_index().ffill()
    for short_name, alternative in {
        "awhman": "awhman__below_12m_average",
        "dgorder": "dgorder__yoy_nonpositive",
    }.items():
        available = macro["industrial_production_growth"].notna() & macro[alternative].notna()
        macro[f"indpro_or_{short_name}"] = (
            macro["industrial_production_growth"].eq(1.0) | macro[alternative].eq(1.0)
        ).astype(float).where(available)
        macro[f"indpro_and_{short_name}"] = (
            macro["industrial_production_growth"].eq(1.0) & macro[alternative].eq(1.0)
        ).astype(float).where(available)
    study_config = dict(config)
    study_config["macro_signals"] = [
        "industrial_production_growth",
        *alternatives.columns,
        "indpro_or_awhman",
        "indpro_and_awhman",
        "indpro_or_dgorder",
        "indpro_and_dgorder",
    ]
    result = run_gtt_risk_validation(
        macro,
        trends,
        returns,
        study_config,
        {asset_id: "S&P 500"},
        args.start,
        args.end,
        return_conventions={asset_id: "log"},
    )
    paths = write_gtt_risk_outputs(result, args.output_dir)
    output_root = Path(args.output_dir).expanduser()
    history.to_parquet(output_root / "gtt_sp500_hybrid_history.parquet")
    _write_report(result.summary, output_root / "gtt_production_validation_report.md")
    print("GTT production validation complete")
    print(f"Summary rows: {len(result.summary):,}")
    print(f"Research candidates: {int(result.summary['research_candidate'].sum()):,}")
    print(f"Summary: {paths['summary']}")
    print(f"Report: {output_root / 'gtt_production_validation_report.md'}")


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    candidates = summary.loc[summary["research_candidate"]]
    aggregates = (
        summary.groupby("macro_signal")
        .agg(
            tests=("research_candidate", "size"),
            candidates=("research_candidate", "sum"),
            median_incremental_lift=("incremental_lift_vs_trend_only", "median"),
        )
        .sort_values(["candidates", "median_incremental_lift"], ascending=False)
    )
    lines = [
        "# GTT Production-Signal Validation",
        "",
        "This exploratory study compares predefined production decompositions and complements with the existing point-in-time INDPRO contraction rule. It uses the chained S&P 500 price/total-return history and the same robustness gates as the core GTT study.",
        "",
        "A pass means the signal separated negative-price-trend observations with worse future risk from negative-trend observations where that macro signal remained healthy. OR/AND combinations with the strongest alternatives provide an exploratory coverage-versus-confirmation comparison; they are not an independent holdout test.",
        "",
        "| Signal | Passed tests | Total tests | Median incremental lift |",
        "| --- | ---: | ---: | ---: |",
    ]
    for signal, row in aggregates.iterrows():
        lines.append(
            f"| {signal} | {int(row['candidates'])} | {int(row['tests'])} | {row['median_incremental_lift']:.4f} |"
        )
    lines.extend(["", f"Candidate rows: `{len(candidates)}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
