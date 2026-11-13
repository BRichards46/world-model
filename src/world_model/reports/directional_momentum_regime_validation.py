from __future__ import annotations

import pandas as pd


def render_directional_momentum_regime_report(summary: pd.DataFrame) -> str:
    lines = [
        "# Matching-Horizon Cross-Asset Momentum Regimes",
        "",
        "This study tests whether the joint ES/NQ/DX/ZB state adds MFE/MAE information beyond the target asset's own momentum state at the same lookback.",
        "",
    ]
    if summary.empty:
        return "\n".join(lines + ["No valid regime results were produced.", ""])
    full = summary.loc[summary["period_id"].eq("full")]
    stable = full.loc[full["stable_across_periods"]].sort_values(
        ["edge_uplift_risk", "sample_size"], ascending=[False, False]
    )
    degrading = full.loc[full["stable_degradation_across_periods"]].sort_values(
        ["edge_uplift_risk", "sample_size"], ascending=[True, False]
    )
    lines.extend(
        [
            "## Coverage",
            "",
            f"- Tested regime/target/outcome relationships: {len(full):,}",
            f"- Full-history incremental improvements: {int(full['incremental_regime_value'].sum()):,}",
            f"- Improvements stable in full and 2015-onward samples: {len(stable):,}",
            f"- Degradations stable in full and 2015-onward samples: {len(degrading):,}",
            "",
            "## Strongest stable incremental relationships",
            "",
        ]
    )
    if stable.empty:
        lines.append("No regime added stable excursion value beyond the target asset's own signal.")
    else:
        lines.extend(
            [
                "| Regime | Signal horizon | Target | Forward | N | MFE uplift | MAE reduction | Edge uplift |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stable.head(40).itertuples(index=False):
            lines.append(
                f"| {row.regime_id} | {row.signal_horizon_label} | {row.target_asset} | {row.forward_label} | {row.sample_size} | {row.mfe_uplift_risk:+.2f} | {row.mae_reduction_risk:+.2f} | {row.edge_uplift_risk:+.2f} |"
            )
    lines.extend(
        [
            "",
            "## Strongest stable adverse modifiers",
            "",
        ]
    )
    if degrading.empty:
        lines.append("No regime consistently degraded all three excursion measures.")
    else:
        lines.extend(
            [
                "| Regime | Signal horizon | Target | Forward | N | MFE uplift | MAE reduction | Edge uplift |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in degrading.head(40).itertuples(index=False):
            lines.append(
                f"| {row.regime_id} | {row.signal_horizon_label} | {row.target_asset} | {row.forward_label} | {row.sample_size} | {row.mfe_uplift_risk:+.2f} | {row.mae_reduction_risk:+.2f} | {row.edge_uplift_risk:+.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Positive uplifts are incremental to the target asset's own positive or negative matching-horizon state. This prevents a joint regime from receiving credit for information already present in the target's momentum alone. An adverse modifier has less MFE, more MAE, and a worse excursion balance in both sample periods.",
            "",
            "The stability screen is descriptive. Weekly observations overlap for longer forward windows, regime counts can be sparse, and the tested relationships are correlated.",
            "",
        ]
    )
    return "\n".join(lines)
