from __future__ import annotations

import argparse

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.forecast import write_forecast_ranker_report
from world_model.states.forecast import MarketForecastRanker, write_forecast_ranker_outputs
from world_model.states.market import MarketStateEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward validation for horizon-specific market forecast ranks.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--start", default=None, help="First forecast date.")
    parser.add_argument("--end", default=None, help="Last forecast date.")
    parser.add_argument("--frequency", default="ME", help="Evaluation frequency. Use ME, W-FRI, or B.")
    parser.add_argument("--min-train-observations", type=int, default=756)
    parser.add_argument("--min-abs-rank-ic", type=float, default=0.02)
    parser.add_argument("--max-components", type=int, default=5)
    parser.add_argument("--train-window-years", type=int, default=None)
    parser.add_argument("--output-dir", default=str(project_root() / "outputs"))
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "market_forecast_ranker_report.md"),
    )
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = MarketStateEngine(load_yaml(args.data_roots), load_yaml(args.market_config))
    ranker = MarketForecastRanker(engine)
    result = ranker.run(
        start=args.start,
        end=args.end,
        frequency=args.frequency,
        min_train_observations=args.min_train_observations,
        min_abs_rank_ic=args.min_abs_rank_ic,
        max_components=args.max_components,
        train_window_years=args.train_window_years,
    )
    output_paths = write_forecast_ranker_outputs(result, args.output_dir)
    if not args.no_report:
        write_forecast_ranker_report(
            result,
            args.report_out,
            frequency=args.frequency,
            min_train_observations=args.min_train_observations,
            train_window_years=args.train_window_years,
        )
        output_paths["report"] = args.report_out

    print("Market forecast ranker validation complete.")
    print(result.summary.to_string(index=False))
    print("")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
