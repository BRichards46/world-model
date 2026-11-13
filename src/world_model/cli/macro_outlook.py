from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.config import project_root
from world_model.states.macro_outlook import build_macro_outlook


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build the dashboard's plain-language outlook from its other payloads.")
    parser.add_argument("--data-dir", default=str(root / "sites" / "macro-dashboard" / "dist" / "data"))
    return parser


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = Path(args.data_dir).expanduser()
    outlook = build_macro_outlook(
        _load(data / "growth_inflation_quad.json"),
        _load(data / "risk_overlay.json"),
        _load(data / "release_forecasts.json"),
        _load(data / "snapshot.json") or [],
        _load(data / "risk_context.json"),
        _load(data / "release_benchmarks.json"),
    )
    target = data / "outlook.json"
    target.write_text(json.dumps(outlook, indent=2) + "\n")
    print(f"Outlook: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
