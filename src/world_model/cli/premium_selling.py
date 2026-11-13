from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.states.premium_selling import PremiumSellingEngine


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build the premium-selling suitability monitor.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--config", default=str(default_config_path("premium_selling.yml")))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--snapshot-json-out", default=str(root / "outputs" / "premium_selling_snapshot.json"))
    parser.add_argument("--site-json-out", default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "premium_selling.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = PremiumSellingEngine(load_yaml(args.data_roots), load_yaml(args.config)).build(as_of=args.as_of)
    text = json.dumps(result.payload, indent=2) + "\n"
    for raw_path in [args.snapshot_json_out, args.site_json_out]:
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"Premium-selling monitor as of {result.as_of_date.date()}: {len(result.payload['assets'])} covered assets")
    print(f"Site data: {Path(args.site_json_out).expanduser()}")


if __name__ == "__main__":
    main()
