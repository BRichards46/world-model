from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.states.directional_momentum import DirectionalMomentumEngine


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build the cross-asset directional momentum monitor.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--config", default=str(default_config_path("directional_momentum.yml")))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--history-out", default=str(root / "outputs" / "directional_momentum_history.parquet"))
    parser.add_argument("--snapshot-out", default=str(root / "outputs" / "directional_momentum_snapshot.parquet"))
    parser.add_argument("--snapshot-json-out", default=str(root / "outputs" / "directional_momentum_snapshot.json"))
    parser.add_argument(
        "--site-json-out",
        default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "directional_momentum.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = DirectionalMomentumEngine(load_yaml(args.data_roots), load_yaml(args.config))
    result = engine.build(as_of=args.as_of)
    history_path = Path(args.history_out).expanduser()
    snapshot_path = Path(args.snapshot_out).expanduser()
    json_path = Path(args.snapshot_json_out).expanduser()
    site_path = Path(args.site_json_out).expanduser()
    for path in [history_path, snapshot_path, json_path, site_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.history.to_parquet(history_path, index=False)
    result.snapshot.to_parquet(snapshot_path, index=False)
    json_text = json.dumps(result.payload, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    site_path.write_text(json_text, encoding="utf-8")
    print(f"Directional momentum as of {result.as_of_date.date()}: {result.snapshot['asset_id'].nunique()} assets")
    print(f"History: {history_path}")
    print(f"Snapshot: {snapshot_path}")
    print(f"Site data: {site_path}")


if __name__ == "__main__":
    main()
