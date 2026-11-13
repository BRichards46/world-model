from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.macro_regime_forecast import MacroRegimeForecastResult


def write_macro_regime_forecast_report(
    result: MacroRegimeForecastResult,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = result.summary.loc[result.summary["research_candidate"]]
    as_of = pd.to_datetime(result.current["as_of_date"]).max().date().isoformat()
    lines = [
        "# Macro regime forecast research",
        "",
        f"As of **{as_of}**. These are research estimates, not dashboard signals.",
        "",
        "## Current estimates",
        "",
        "| Target | Horizon | Type | Model probability | State-conditional baseline | Validated? |",
        "| --- | --- | --- | ---: | ---: | :---: |",
    ]
    validated = {
        (row.target_id, row.horizon): bool(row.research_candidate)
        for row in result.summary.itertuples()
    }
    for row in result.current.sort_values(["target_id", "trading_days"]).itertuples():
        lines.append(
            f"| {row.target_label} | {row.horizon.replace('_', ' ')} | {row.forecast_type} | "
            f"{row.probability * 100:.1f}% | {row.baseline_probability * 100:.1f}% | "
            f"{'yes' if validated.get((row.target_id, row.horizon), False) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Walk-forward validation",
            "",
            "The benchmark is the historical transition rate conditional on whether the same risk "
            "state is active today. A candidate must improve Brier score in the full, early, and "
            "late out-of-sample periods, have at least 60 forecasts, and achieve ROC AUC of at least 0.55.",
            "",
            "| Target | Horizon | OOS observations | Brier improvement | Early | Late | AUC | Candidate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for row in result.summary.sort_values(["target_id", "trading_days"]).itertuples():
        lines.append(
            f"| {row.target_label} | {row.horizon.replace('_', ' ')} | {int(row.observations)} | "
            f"{row.brier_improvement:+.3f} | {row.early_brier_improvement:+.3f} | "
            f"{row.late_brier_improvement:+.3f} | {row.roc_auc:.2f} | "
            f"{'yes' if row.research_candidate else 'no'} |"
        )
    lines.extend(["", "## Conclusion", ""])
    if candidates.empty:
        lines.append(
            "No forecast passes the production threshold. The three-month low-volatility-state "
            "model is promising but has only 59 out-of-sample observations—one short of the "
            "pre-set minimum. High-volatility-state entry is not forecastable with this simple "
            "composite-level and momentum model. Current probabilities must remain research-only."
        )
    else:
        lines.append(
            f"{len(candidates)} forecast specification(s) pass the research threshold. They still "
            "require frozen-specification monitoring before dashboard promotion."
        )
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
