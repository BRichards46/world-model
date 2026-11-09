from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.directional_momentum_regime_validation import (
    render_directional_momentum_regime_report,
)
from world_model.states.directional_momentum_regime_validation import (
    DirectionalMomentumRegimeValidator,
)


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Test matching-horizon ES/NQ/DX/ZB momentum regimes.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--momentum-config", default=str(default_config_path("directional_momentum.yml")))
    parser.add_argument(
        "--regime-config",
        default=str(default_config_path("directional_momentum_regime_validation.yml")),
    )
    parser.add_argument("--history", default=str(root / "outputs" / "directional_momentum_history.parquet"))
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs" / "directional_momentum_regime_validation"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validator = DirectionalMomentumRegimeValidator(
        load_yaml(args.data_roots),
        load_yaml(args.momentum_config),
        load_yaml(args.regime_config),
    )
    result = validator.run(pd.read_parquet(Path(args.history).expanduser()))
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    result.regimes.to_parquet(output / "regimes.parquet", index=False)
    result.events.to_parquet(output / "excursion_events.parquet", index=False)
    result.summary.to_csv(output / "excursion_summary.csv", index=False)
    (output / "report.md").write_text(
        render_directional_momentum_regime_report(result.summary) + "\n",
        encoding="utf-8",
    )
    full = result.summary.loc[result.summary["period_id"].eq("full")]
    print(f"Weekly matching-horizon regimes: {len(result.regimes):,}")
    print(f"Tested relationships: {len(full):,}")
    print(f"Full-history incremental improvements: {int(full['incremental_regime_value'].sum()):,}")
    print(f"Stable across both periods: {int(full['stable_across_periods'].sum()):,}")
    print(f"Stable adverse modifiers: {int(full['stable_degradation_across_periods'].sum()):,}")
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
