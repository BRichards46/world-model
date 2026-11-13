from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.macro_release_forecast import write_macro_release_forecast_report
from world_model.states.macro_release_forecast import (
    build_release_site_payload,
    run_macro_release_forecast,
    write_macro_release_forecast_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run release-aware payroll and inflation forecast research.")
    parser.add_argument("--config", default=str(default_config_path("macro_release_forecast.yml")))
    parser.add_argument("--data-roots", default=str(root / "config" / "data_roots.yml"))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "macro_release_forecast"))
    parser.add_argument("--as-of")
    parser.add_argument(
        "--site-json-out",
        default=str(root / "sites" / "macro-dashboard" / "dist" / "data" / "release_forecasts.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    roots = load_yaml(args.data_roots)
    result = run_macro_release_forecast(roots["fred"], load_yaml(args.config), as_of=args.as_of)
    paths = write_macro_release_forecast_outputs(result, args.output_dir)
    report = write_macro_release_forecast_report(
        result, Path(args.output_dir).expanduser() / "macro_release_forecast_report.md"
    )
    site_path = Path(args.site_json_out).expanduser()
    site_path.parent.mkdir(parents=True, exist_ok=True)
    site_path.write_text(json.dumps(build_release_site_payload(result), indent=2) + "\n", encoding="utf-8")
    candidates = int(result.summary["research_candidate"].sum()) if not result.summary.empty else 0
    print("Macro release forecast research complete")
    print(f"Out-of-sample forecasts: {len(result.forecasts):,}")
    print(f"Research candidates: {candidates:,}")
    print(f"Current: {paths['current']}")
    print(f"Summary: {paths['summary']}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
