from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.macro_release_forecast import MacroReleaseForecastResult


def write_macro_release_forecast_report(
    result: MacroReleaseForecastResult,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Payroll and inflation release forecast research",
        "",
        "All historical features are reconstructed from FRED vintages available before each target release. "
        "The benchmark is an expanding autoregression using only the prior target reading.",
        "",
        "## Current estimates",
        "",
        "| Target | Variant | Reference month | Forecast | AR benchmark | Prior reading | Research candidate? |",
        "| --- | --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for row in result.current.sort_values(["target_id", "variant_id"]).itertuples():
        lines.append(
            f"| {row.target_label} | {row.variant_id} | {pd.Timestamp(row.forecast_observation_date).strftime('%Y-%m')} | "
            f"{row.display_forecast:.2f} {row.display_unit} | {row.display_baseline_forecast:.2f} | "
            f"{row.display_prior_target:.2f} | "
            f"{'yes' if bool(row.research_candidate) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Walk-forward validation",
            "",
            "A candidate must beat the autoregressive benchmark on MAE in the full, early, and late "
            "out-of-sample periods, satisfy the target's absolute-error ceiling, and achieve at least "
            "a 52% directional hit rate.",
            "",
            "| Target | Variant | OOS | Model MAE | AR MAE | Improvement | Early | Late | Direction | Candidate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for row in result.summary.sort_values(["target_id", "variant_id"]).itertuples():
        lines.append(
            f"| {row.target_label} | {row.variant_id} | {int(row.observations)} | {row.model_mae:.3f} | "
            f"{row.baseline_mae:.3f} | {row.mae_improvement:+.3f} | {row.early_mae_improvement:+.3f} | "
            f"{row.late_mae_improvement:+.3f} | {row.direction_hit_rate:.1%} | "
            f"{'yes' if row.research_candidate else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A failed candidate remains useful as a transparent preview, but it must not be presented as a "
            "validated forecast. ADP has a shorter genuine point-in-time history than the other payroll inputs, "
            "so its variant is evaluated separately.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
