from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from world_model.config import default_config_path, load_yaml, project_root
from world_model.reports.directional_momentum_validation import render_directional_momentum_validation_report
from world_model.states.directional_momentum_validation import DirectionalMomentumValidator


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    output = root / "outputs" / "directional_momentum_validation"
    parser = argparse.ArgumentParser(description="Test whether directional momentum improves MFE/MAE.")
    parser.add_argument("--data-roots", default=str(default_config_path("data_roots.yml")))
    parser.add_argument("--momentum-config", default=str(default_config_path("directional_momentum.yml")))
    parser.add_argument("--validation-config", default=str(default_config_path("directional_momentum_validation.yml")))
    parser.add_argument("--history", default=str(root / "outputs" / "directional_momentum_history.parquet"))
    parser.add_argument("--output-dir", default=str(output))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    history = pd.read_parquet(Path(args.history).expanduser())
    validator = DirectionalMomentumValidator(
        load_yaml(args.data_roots),
        load_yaml(args.momentum_config),
        load_yaml(args.validation_config),
    )
    result = validator.run(history)
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    result.signals.to_parquet(output / "signals.parquet", index=False)
    result.events.to_parquet(output / "excursion_events.parquet", index=False)
    result.baselines.to_parquet(output / "baseline_events.parquet", index=False)
    result.summary.to_csv(output / "excursion_summary.csv", index=False)
    report = render_directional_momentum_validation_report(result.summary)
    (output / "report.md").write_text(report + "\n", encoding="utf-8")
    full = result.summary.loc[result.summary["period_id"].eq("full")] if not result.summary.empty else result.summary
    recent = result.summary.loc[result.summary["period_id"].eq("post_2015")] if not result.summary.empty else result.summary
    stable = int(full["stable_across_periods"].sum()) if not full.empty else 0
    print(f"Signals: {len(result.signals):,}")
    print(f"Excursion events: {len(result.events):,}")
    print(f"Summary relationships: {len(result.summary):,}")
    print(f"Full-history excursion improvements: {int(full['robust_excursion_improvement'].sum()):,}")
    print(f"2015-onward excursion improvements: {int(recent['robust_excursion_improvement'].sum()):,}")
    print(f"Stable across both periods: {stable:,}")
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
