from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.macro_regime_performance import MacroRegimePerformanceResult


def write_macro_regime_performance_report(
    result: MacroRegimePerformanceResult,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tape = result.regime_tape.dropna(subset=["macro_regime"])
    current = tape.iloc[-1]
    as_of = pd.Timestamp(tape.index[-1]).date().isoformat()
    performance = result.performance
    benchmark = performance.loc[
        performance["regime_dimension"].eq("macro_regime")
        & performance["asset_id"].eq("sp500_total_return")
    ].copy()
    current_classes = performance.loc[
        performance["regime_dimension"].eq("macro_regime")
        & performance["asset_type"].eq("class")
        & performance["is_current_regime"]
    ].copy()
    repeatable = performance.loc[
        performance["asset_type"].eq("class")
        & (
            performance["repeatable_performance_shift"]
            | (
                performance["regime_dimension"].eq("macro_regime")
                & performance["repeatable_volatility_shift"]
            )
        )
    ].copy()
    transitions = result.transitions.loc[
        result.transitions["regime_dimension"].eq("macro_regime")
        & result.transitions["from_regime"].eq(str(current["macro_regime"]))
    ].copy()

    lines = [
        "# Macro regime performance",
        "",
        f"As of **{as_of}**, the point-in-time regime is **{_label(current['macro_regime'])}**.",
        "",
        "## Current macro tape",
        "",
        f"- Growth: {_score(current['growth_level_score'])} level / "
        f"{_score(current['growth_momentum_score'])} momentum "
        f"({_label(current['growth_state'])})",
        f"- Inflation: {_score(current['inflation_level_score'])} level / "
        f"{_score(current['inflation_momentum_score'])} momentum "
        f"({_label(current['inflation_state'])})",
        f"- Financial conditions: {_score(current['financial_conditions_level_score'])} "
        f"({_label(current['financial_conditions_state'])})",
        "",
        "The tape is lagged by one trading day after each point-in-time macro release. "
        "Neutral bands are retained; a stable reading is not forced into improving or slowing.",
        "",
        "## S&P 500 total-return history during named regimes",
        "",
        "| Regime | Days | Episodes | Ann. log return | Ann. vol | Vol vs baseline | Repeatable return? | Repeatable vol? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for row in benchmark.sort_values("regime_id").itertuples():
        lines.append(
            f"| {_label(row.regime_id)} | {int(row.observations):,} | {int(row.episodes):,} | "
            f"{_pct(row.annualized_performance)} | {_pct(row.annualized_volatility)} | "
            f"{row.volatility_ratio_vs_baseline:.2f}× | "
            f"{_yes(row.repeatable_performance_shift)} | {_yes(row.repeatable_volatility_shift)} |"
        )

    lines.extend(
        [
            "",
            "## Current regime across asset classes",
            "",
            "Futures use a roll-safe directional performance score (daily price change divided by "
            "trailing median absolute price). It supports sign and relative-risk comparisons, not a cash-return claim.",
            "",
            "| Asset class | Directional score | Difference vs baseline | Vol vs baseline | Positive episodes |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in current_classes.sort_values("asset_label").itertuples():
        lines.append(
            f"| {row.asset_label} | {row.annualized_performance:.3f} | "
            f"{row.performance_spread_vs_baseline:+.3f} | {row.volatility_ratio_vs_baseline:.2f}× | "
            f"{_pct(row.positive_episode_rate, digits=0)} |"
        )

    lines.extend(["", "## Retained relationships", ""])
    if repeatable.empty:
        lines.append("No class-level relationships passed the repeatability filters.")
    else:
        lines.extend(
            [
                "A relationship is retained only when it is material, has the same direction in the "
                "early and late samples, and—where directional performance is concerned—survives "
                "episode resampling and false-discovery control.",
                "",
                "| Dimension | State | Asset | Return shift | Vol shift |",
                "| --- | --- | --- | :---: | :---: |",
            ]
        )
        for row in repeatable.sort_values(
            ["regime_dimension", "regime_id", "asset_label"]
        ).itertuples():
            lines.append(
                f"| {_label(row.regime_dimension)} | {_label(row.regime_id)} | {row.asset_label} | "
                f"{_yes(row.repeatable_performance_shift)} | {_yes(row.repeatable_volatility_shift)} |"
            )

    lines.extend(["", "## Historical exits from the current regime", ""])
    if transitions.empty:
        lines.append("No completed transition is available.")
    else:
        lines.extend(["| Next regime | Share of exits | Count |", "| --- | ---: | ---: |"])
        for row in transitions.sort_values("transition_probability", ascending=False).itertuples():
            lines.append(
                f"| {_label(row.to_regime)} | {_pct(row.transition_probability, digits=0)} | "
                f"{int(row.transitions)} |"
            )

    lines.extend(
        [
            "",
            "## Research conclusion",
            "",
            "The regime tape currently has more reliable information about **risk/volatility** than "
            "about market direction. The next forecasting target should therefore be entry into, "
            "persistence of, and exit from the repeatable high- and low-volatility macro states. "
            "Directional return forecasts should remain secondary until they pass the same "
            "out-of-sample and multiple-testing controls.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _label(value: object) -> str:
    return str(value).replace("__", " · ").replace("_", " ").title()


def _score(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.1f}th percentile"


def _pct(value: object, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{float(value) * 100:.{digits}f}%"


def _yes(value: object) -> str:
    return "yes" if bool(value) else "no"
