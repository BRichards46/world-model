from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.states.quad_asset_performance import QuadAssetPerformanceResult


TRADEABLE_SIGNALS = ("projected_current", "projected_next", "realized_last")
REPORT_HORIZON = "three_months"


def _fmt(value: Any, decimals: int = 1, signed: bool = True) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"


def _current_quads(result: QuadAssetPerformanceResult) -> dict[str, int | None]:
    latest = result.signals.ffill().iloc[-1] if not result.signals.empty else pd.Series(dtype=float)
    return {signal: (int(latest[signal]) if signal in latest and pd.notna(latest[signal]) else None) for signal in result.signals.columns}


def _matrix(summary: pd.DataFrame, signal: str, horizon: str, assets: list[str]) -> str:
    frame = summary[(summary["signal"] == signal) & (summary["horizon"] == horizon) & summary["asset_id"].isin(assets)]
    if frame.empty:
        return "_No results._\n"
    labels = frame.drop_duplicates("asset_id").set_index("asset_id")["asset_label"]
    lines = ["| Asset | Quad 1 | Quad 2 | Quad 3 | Quad 4 |", "| --- | ---: | ---: | ---: | ---: |"]
    marks = {"robust": "**", "suggestive": "†", "nominal only": "·"}
    for asset in assets:
        rows = frame[frame["asset_id"] == asset].set_index("quad")
        if rows.empty:
            continue
        cells = []
        for quad in (1, 2, 3, 4):
            if quad not in rows.index:
                cells.append("—")
                continue
            row = rows.loc[quad]
            mark = marks.get(str(row["evidence"]), "")
            text = _fmt(row["annualized_spread"])
            cells.append(f"**{text}**" if mark == "**" else f"{text}{mark}")
        lines.append(f"| {labels.get(asset, asset)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _playbook_markdown(playbook: pd.DataFrame | None, signals_meta: Mapping[str, Any]) -> list[str]:
    if playbook is None or playbook.empty:
        return []
    lines = [
        "",
        "## Follow vs fade the quad playbook",
        "",
        "A risk-balanced long/short portfolio of each quad's leaders minus laggards, held for the horizon after each "
        "month-end. *Follow* holds it; *fade* (the stale-quad recovery idea) holds the opposite. The textbook playbook is "
        "fixed in config; the learned playbook uses only in-quarter returns from quarters whose quad was known at the time. "
        "p-values rotate the quad sequence against returns; q-values adjust within horizon and direction.",
        "",
        "| Horizon | Playbook | Signal | Follow %/yr | t (HAC) | Follow q | Fade q | 1st half | 2nd half | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    follow = playbook[playbook["direction"] == "follow"].set_index(["horizon", "playbook", "signal"])
    fade = playbook[playbook["direction"] == "fade"].set_index(["horizon", "playbook", "signal"])
    for key, row in follow.iterrows():
        faded = fade.loc[key]
        verdict = "follow works" if row["significant"] else "fade works" if faded["significant"] else "no edge"
        lines.append(
            f"| {key[0].replace('_', ' ')} | {key[1]} | {signals_meta.get(key[2], {}).get('label', key[2])} | "
            f"{_fmt(row['annualized_return'], 2)} | {_fmt(row['t_stat_hac'], 2)} | {row['q_value']:.3f} | {faded['q_value']:.3f} | "
            f"{_fmt(row['first_half_annualized'], 2)} | {_fmt(row['second_half_annualized'], 2)} | {verdict} |"
        )
    return lines


def write_report(
    result: QuadAssetPerformanceResult,
    config: Mapping[str, Any],
    path: str | Path,
    playbook: pd.DataFrame | None = None,
) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.quad_summary
    joint = result.joint_summary
    signals_meta = config.get("signals", {})
    current = _current_quads(result)
    assets = list(dict.fromkeys(summary["asset_id"])) if not summary.empty else []
    absolute = [asset for asset in assets if not asset.endswith("_vs_sp500")]
    relative = [asset for asset in assets if asset.endswith("_vs_sp500")]

    lines = [
        "# Asset performance by growth/inflation quad",
        "",
        f"As of **{result.as_of_date:%Y-%m-%d}**. Monthly signals from "
        f"{result.signals.index.min():%Y-%m} to {result.signals.index.max():%Y-%m}, built only from FRED vintages known at each month-end.",
        "",
        "Forward returns start one trading day after the signal date. Spreads are annualized differences between the "
        "average forward return when the signal shows a quad and the average over all months. Significance uses a circular "
        "rotation test of the quad sequence against returns, with Benjamini-Hochberg adjustment within each signal and horizon.",
        "",
        "Evidence tiers: **bold** = robust (q ≤ 0.10, same sign in both halves, on medians, and excluding 2008–09 and 2020); "
        "† = suggestive (q ≤ 0.25 with the same consistency checks); · = nominal p ≤ 0.05 only.",
        "",
        "## Current signals",
        "",
        "| Signal | Current quad |",
        "| --- | --- |",
    ]
    for signal, quad in current.items():
        lines.append(f"| {signals_meta.get(signal, {}).get('label', signal)} | {'Quad ' + str(quad) if quad else 'n/a'} |")
    lines += ["", "## Do quads separate forward returns at all?", "", "Joint test across all four quads, by asset (count with q ≤ 0.10).", "", "| Signal | 1-month | 3-month |", "| --- | ---: | ---: |"]
    for signal in result.signals.columns:
        counts = []
        for horizon in ("one_month", "three_months"):
            group = joint[(joint["signal"] == signal) & (joint["horizon"] == horizon)]
            counts.append(f"{int(group['significant'].sum())} / {len(group)}")
        lines.append(f"| {signals_meta.get(signal, {}).get('label', signal)} | {counts[0]} | {counts[1]} |")

    for signal in ("hindsight_current", *TRADEABLE_SIGNALS):
        label = signals_meta.get(signal, {}).get("label", signal)
        lines += [
            "",
            f"## {label} — 3-month forward, annualized spread vs all months",
            "",
            str(signals_meta.get(signal, {}).get("note", "")),
            "",
            "### Absolute",
            "",
            _matrix(summary, signal, REPORT_HORIZON, absolute),
            "### Relative to the S&P 500",
            "",
            _matrix(summary, signal, REPORT_HORIZON, relative),
        ]

    tiers = summary[summary["evidence"].isin(["robust", "suggestive"]) & summary["signal"].isin(TRADEABLE_SIGNALS)]
    lines += ["", "## Tradeable findings that pass the consistency checks", ""]
    if tiers.empty:
        lines.append("None.")
    else:
        lines += ["| Signal | Horizon | Quad | Asset | Spread | Median | Ex-crisis | p | q | Tier |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in tiers.sort_values(["evidence", "q_value"]).itertuples():
            lines.append(
                f"| {signals_meta.get(row.signal, {}).get('label', row.signal)} | {row.horizon.replace('_', ' ')} | {row.quad} | {row.asset_label} | "
                f"{_fmt(row.annualized_spread)} | {_fmt(row.median_annualized_spread)} | {_fmt(row.ex_crisis_annualized_spread)} | "
                f"{row.p_value:.3f} | {row.q_value:.3f} | {row.evidence} |"
            )
    lines += _playbook_markdown(playbook, signals_meta)
    lines += [
        "",
        "## Limits",
        "",
        "- About 28 years of monthly signals: each quad has roughly 70–115 months, and 3-month forward returns overlap.",
        "- Futures groups use normalized price change (direction and relative size), not cash returns.",
        "- Long Treasuries (TLT) start in 2002.",
        "- Quad labels are descriptive. These are historical conditional averages, not a validated trading rule.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_site_payload(
    result: QuadAssetPerformanceResult,
    config: Mapping[str, Any],
    playbook: pd.DataFrame | None = None,
) -> str:
    summary = result.quad_summary
    signals_meta = config.get("signals", {})
    current = _current_quads(result)
    blocks = []
    for signal in TRADEABLE_SIGNALS:
        quad = current.get(signal)
        if quad is None:
            continue
        frame = summary[
            (summary["signal"] == signal) & (summary["horizon"] == REPORT_HORIZON) & (summary["quad"] == quad)
        ].copy()
        frame = frame.reindex(frame["annualized_spread"].abs().sort_values(ascending=False).index)
        blocks.append(
            {
                "signal": signal,
                "signal_label": signals_meta.get(signal, {}).get("label", signal),
                "quad": quad,
                "assets": [
                    {
                        "asset_id": row.asset_id,
                        "asset_label": row.asset_label,
                        "asset_group": row.asset_group,
                        "quad_months": int(row.quad_months),
                        "annualized_spread": _clean(row.annualized_spread),
                        "median_annualized_spread": _clean(row.median_annualized_spread),
                        "ex_crisis_annualized_spread": _clean(row.ex_crisis_annualized_spread),
                        "p_value": _clean(row.p_value),
                        "q_value": _clean(row.q_value),
                        "evidence": row.evidence,
                    }
                    for row in frame.itertuples()
                ],
            }
        )
    joint = result.joint_summary
    payload = {
        "as_of_date": f"{result.as_of_date:%Y-%m-%d}",
        "horizon": REPORT_HORIZON,
        "sample_start": f"{result.signals.index.min():%Y-%m}",
        "sample_end": f"{result.signals.index.max():%Y-%m}",
        "current_signals": blocks,
        "joint_significant": {
            signal: int(joint[(joint["signal"] == signal) & (joint["horizon"] == REPORT_HORIZON)]["significant"].sum())
            for signal in result.signals.columns
        },
        "joint_tests": int(len(joint[(joint["signal"] == "projected_current") & (joint["horizon"] == REPORT_HORIZON)])),
        "playbook": []
        if playbook is None or playbook.empty
        else [
            {
                "playbook": row.playbook,
                "signal": row.signal,
                "signal_label": signals_meta.get(row.signal, {}).get("label", row.signal),
                "direction": row.direction,
                "annualized_return": _clean(row.annualized_return),
                "t_stat_hac": _clean(row.t_stat_hac),
                "q_value": _clean(row.q_value),
                "significant": bool(row.significant),
            }
            for row in playbook[playbook["horizon"] == REPORT_HORIZON].itertuples()
        ],
        "note": "3-month forward performance after month-ends with the same signal, annualized versus all months since 1998. "
        "Robust = q ≤ 0.10 and consistent; suggestive = q ≤ 0.25 and consistent (same sign in both halves, on medians, and excluding 2008–09 and 2020).",
    }
    return json.dumps(payload, indent=2) + "\n"


def _clean(value: Any) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None
