from __future__ import annotations

import argparse

from world_model.config import default_config_path, load_yaml, project_root
from world_model.data.norgate import build_market_breadth, write_market_breadth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Norgate-derived market breadth data.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--start-date", default="1990-01-01")
    parser.add_argument("--max-assets", type=int, default=None, help="Optional smoke-test asset limit.")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument(
        "--output",
        default=str(project_root() / "outputs" / "norgate_market_breadth.parquet"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    roots = load_yaml(args.data_roots)
    breadth, stats = build_market_breadth(
        price_root=roots["norgate_prices"],
        metadata_root=roots["norgate_metadata"],
        start_date=args.start_date,
        max_assets=args.max_assets,
        batch_size=args.batch_size,
    )
    write_market_breadth(breadth, args.output)
    print("Market breadth build complete.")
    print(f"Selected assets: {stats.selected_assets:,}")
    print(f"Processed assets: {stats.processed_assets:,}")
    print(f"Skipped assets: {stats.skipped_assets:,}")
    if stats.first_date is not None and stats.last_date is not None:
        print(f"Date range: {stats.first_date.date()} to {stats.last_date.date()}")
    print(f"Rows: {len(breadth):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
