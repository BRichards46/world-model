from __future__ import annotations

import pandas as pd


def render_directional_momentum_validation_report(summary: pd.DataFrame) -> str:
    lines = [
        "# Directional Momentum Excursion Validation",
        "",
        "This study asks whether a measured directional bias improves the subsequent path: more maximum favorable excursion (MFE), less maximum adverse excursion (MAE), and a better MFE-minus-MAE balance than the same asset's unconditional direction-specific baseline.",
        "",
        "It does not require the terminal return to have the predicted sign and does not treat overlapping observations as independent trading results.",
        "",
    ]
    if summary.empty:
        return "\n".join(lines + ["No valid validation rows were produced.", ""])
    full = summary.loc[summary["period_id"].eq("full")].copy()
    recent = summary.loc[summary["period_id"].eq("post_2015")].copy()
    robust = full.loc[full["robust_excursion_improvement"]].sort_values("edge_uplift_risk", ascending=False)
    stable = full.loc[full.get("stable_across_periods", False)].sort_values("edge_uplift_risk", ascending=False)
    composite_full = full.loc[full["signal_type"].eq("composite")]
    composite_recent = recent.loc[recent["signal_type"].eq("composite")]
    composite_stable = stable.loc[stable["signal_type"].eq("composite")]
    lines.extend(
        [
            "## Coverage",
            "",
            f"- Tested relationships: {len(full):,}",
            f"- Full-history relationships improving median MFE, MAE, and excursion balance: {len(robust):,}",
            f"- Composite relationships passing in full history: {int(composite_full['robust_excursion_improvement'].sum()):,} of {len(composite_full):,}",
            f"- Composite relationships passing from 2015 onward: {int(composite_recent['robust_excursion_improvement'].sum()):,} of {len(composite_recent):,}",
            f"- Composite relationships passing in both periods: {len(composite_stable):,} of {len(composite_full):,}",
            f"- Assets represented: {full['asset_id'].nunique():,}",
            "",
            "## Strongest stable excursion improvements",
            "",
        ]
    )
    if stable.empty:
        lines.append("No relationship passed all three excursion-improvement conditions in both periods.")
    else:
        lines.append("| Asset | Lens | Signal | Direction | Forward | N | MFE uplift | MAE reduction | Edge uplift |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in stable.head(30).itertuples(index=False):
            signal = "Composite" if row.signal_type == "composite" else f"{row.lookback} component"
            direction = "Positive" if row.direction > 0 else "Negative"
            lines.append(
                f"| {row.asset_id} | {row.lens} | {signal} | {direction} | {row.forward_label} | {row.sample_size} | {row.mfe_uplift_risk:+.2f} | {row.mae_reduction_risk:+.2f} | {row.edge_uplift_risk:+.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All uplifts are expressed in ex-ante volatility units relative to the asset's unconditional baseline for the same positive or negative orientation. A positive MFE uplift means the signal created more favorable opportunity; a positive MAE reduction means it reduced adverse excursion. The strict robustness flag requires both, plus a better median MFE-minus-MAE balance.",
            "",
            "The stable flag additionally requires the relationship to pass in both the full history and the 2015-onward sample. It is a descriptive stability screen, not a claim of statistical independence: weekly observations overlap for multi-week outcomes, and the many tested variants are correlated.",
            "",
        ]
    )
    return "\n".join(lines)
