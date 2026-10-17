from __future__ import annotations

import argparse

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.walk_forward import write_walk_forward_report
from world_model.states.market import MarketStateEngine
from world_model.states.walk_forward import MarketStateWalkForward, write_walk_forward_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward validation for the market state analog model.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--market-config", default=str(default_config_path("market_state.yml")))
    parser.add_argument("--start", default=None, help="First forecast date.")
    parser.add_argument("--end", default=None, help="Last forecast date.")
    parser.add_argument("--frequency", default="ME", help="Evaluation frequency. Use ME, W-FRI, or B.")
    parser.add_argument("--top-n", type=int, default=None, help="Number of analogs per forecast date.")
    parser.add_argument("--min-analogs", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(project_root() / "outputs"))
    parser.add_argument(
        "--report-out",
        default=str(project_root() / "outputs" / "market_walk_forward_report.md"),
    )
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = MarketStateEngine(load_yaml(args.data_roots), load_yaml(args.market_config))
    evaluator = MarketStateWalkForward(engine)
    result = evaluator.run(
        start=args.start,
        end=args.end,
        frequency=args.frequency,
        top_n=args.top_n,
        min_analogs=args.min_analogs,
        random_seed=args.random_seed,
    )
    output_paths = write_walk_forward_outputs(result, args.output_dir)
    if not args.no_report:
        write_walk_forward_report(result, args.report_out, frequency=args.frequency, top_n=args.top_n)
        output_paths["report"] = args.report_out

    print("Walk-forward validation complete.")
    print(result.summary.to_string(index=False))
    print("")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
