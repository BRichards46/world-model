"""How much risk the current equity-overlay state has historically carried.

The overlay answers "stay invested or go defensive". This adds "how risky is it here": for every month-end
since 1972 it records the overlay state, splits the invested state by the legacy five-trigger count, and
measures what followed over the next three months (worst drawdown and realized volatility). Each state's
risk tier comes from how often a worst-decile outcome followed it, compared with the 10% base rate.

A second, separately validated read (``docs/macro_regime_validation.md``) is attached when active: slowing
industrial production together with elevated core inflation was followed by higher volatility and deeper
drawdowns.

All of this is descriptive context. Only the overlay's stance has been validated as a timing rule.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

STATE_LABELS = {
    "normal": "Invested, legacy triggers clear",
    "normal_legacy": "Invested, legacy trigger count on",
    "macro_watch": "Invested, macro warning without a trend break",
    "market_warning": "Trend break without a macro warning",
    "defensive": "Defensive: trend break and macro warning",
}

TIERS = ((0.7, "below average"), (1.4, "average"), (2.5, "elevated"), (float("inf"), "high"))

TRIGGER_LABELS = {
    "indpro_yoy": "industrial production",
    "payems_yoy": "payrolls",
    "unemployment_gap": "unemployment",
    "housing_lf_yoy": "housing",
    "cpi_yoy": "CPI",
}


def legacy_flags(values: pd.DataFrame, triggers: Mapping[str, tuple[str, float]]) -> pd.DataFrame:
    """One column per legacy trigger: 1 when it warns, 0 when clear, NaN when unknown."""
    flags = {}
    for metric, (operator, level) in triggers.items():
        series = pd.to_numeric(values[metric], errors="coerce")
        compare = {"<=": series <= level, "<": series < level, ">=": series >= level, ">": series > level}[operator]
        flags[metric] = compare.astype(float).where(series.notna())
    return pd.DataFrame(flags, index=values.index)


def classify_states(
    month_end_levels: pd.Series,
    values: pd.DataFrame,
    triggers: Mapping[str, tuple[str, float]],
    at_least: int,
    trend_months: int = 10,
    unemployment_buffer: float = 0.1,
) -> pd.DataFrame:
    """Overlay state at each month-end, using macro values as known on that date."""
    levels = month_end_levels.dropna().sort_index()
    average = levels.rolling(trend_months, min_periods=trend_months).mean()
    known = values.sort_index().reindex(levels.index, method="ffill")
    flags = legacy_flags(known, triggers)
    frame = pd.DataFrame(index=levels.index)
    frame["trend_up"] = (levels >= average).where(average.notna())
    frame["macro_warning"] = ((known["indpro_yoy"] <= 0.0) | (known["unemployment_gap"] > unemployment_buffer)).where(
        known["indpro_yoy"].notna() & known["unemployment_gap"].notna()
    )
    frame["legacy_count"] = flags.sum(axis=1, min_count=len(triggers))
    frame["state"] = [
        _state(trend, macro, count, at_least)
        for trend, macro, count in zip(frame["trend_up"], frame["macro_warning"], frame["legacy_count"])
    ]
    return frame


def _state(trend: Any, macro: Any, count: Any, at_least: int) -> str | None:
    if pd.isna(trend) or pd.isna(macro):
        return None
    if not trend:
        return "defensive" if macro else "market_warning"
    if macro:
        return "macro_watch"
    if pd.isna(count):
        return None
    return "normal_legacy" if count >= at_least else "normal"


def forward_outcomes(daily_levels: pd.Series, dates: Iterable[pd.Timestamp], horizon_days: int = 63) -> pd.DataFrame:
    """Worst drawdown from each date's close and annualized realized volatility over the next ``horizon_days``."""
    levels = daily_levels.dropna().sort_index()
    log_returns = np.log(levels).diff()
    index = levels.index
    rows = []
    for date in dates:
        position = index.get_indexer([pd.Timestamp(date)])[0]
        if position < 0 or position + horizon_days >= len(index):
            continue
        window = levels.iloc[position : position + horizon_days + 1]
        drawdown = float((window / window.cummax() - 1.0).min())
        volatility = float(log_returns.iloc[position + 1 : position + horizon_days + 1].std() * np.sqrt(252.0))
        rows.append({"date": pd.Timestamp(date), "max_drawdown": drawdown, "volatility": volatility})
    return pd.DataFrame.from_records(rows, columns=["date", "max_drawdown", "volatility"]).set_index("date")


def tier_table(states: pd.DataFrame, outcomes: pd.DataFrame, worst_share: float = 0.10, gap_months: int = 3) -> dict[str, Any]:
    """Per state: months, how often a worst-decile drawdown or volatility followed, and in how many episodes."""
    joined = states.join(outcomes, how="inner").dropna(subset=["state", "max_drawdown", "volatility"])
    drawdown_line = float(joined["max_drawdown"].quantile(worst_share))
    volatility_line = float(joined["volatility"].quantile(1.0 - worst_share))
    joined["bad_drawdown"] = joined["max_drawdown"] <= drawdown_line
    joined["bad_volatility"] = joined["volatility"] >= volatility_line
    table = {}
    for state, group in joined.groupby("state"):
        drawdown_rate = float(group["bad_drawdown"].mean())
        table[state] = {
            "label": STATE_LABELS.get(state, state),
            "months": int(len(group)),
            "worst_drawdown_rate": drawdown_rate,
            "worst_volatility_rate": float(group["bad_volatility"].mean()),
            "median_drawdown": float(group["max_drawdown"].median()),
            "median_volatility": float(group["volatility"].median()),
            "drawdown_episodes": _episodes(group.index[group["bad_drawdown"]], gap_months),
            "tier": tier_for(drawdown_rate, worst_share),
        }
    return {
        "base_rate": worst_share,
        "drawdown_line": drawdown_line,
        "volatility_line": volatility_line,
        "months": int(len(joined)),
        "start": f"{joined.index.min():%Y-%m}" if len(joined) else None,
        "end": f"{joined.index.max():%Y-%m}" if len(joined) else None,
        "states": table,
    }


def tier_for(rate: float, base_rate: float) -> str:
    ratio = rate / base_rate if base_rate else float("nan")
    for limit, label in TIERS:
        if ratio < limit:
            return label
    return TIERS[-1][1]


def _episodes(dates: Iterable[pd.Timestamp], gap_months: int) -> list[str]:
    """Group bad months separated by no more than ``gap_months`` into episodes, labelled by their first month."""
    episodes: list[str] = []
    previous: pd.Timestamp | None = None
    for date in sorted(pd.Timestamp(d) for d in dates):
        if previous is None or (date.to_period("M") - previous.to_period("M")).n > gap_months:
            episodes.append(f"{date:%Y-%m}")
        previous = date
    return episodes


def production_inflation_read(
    snapshot: Iterable[Mapping[str, Any]],
    summary: pd.DataFrame | None,
    elevated_percentile: float = 60.0,
) -> dict[str, Any] | None:
    """Whether slowing production with elevated core inflation is active now, with the study's effect sizes."""
    rows = {str(row.get("indicator_id")): row for row in snapshot}
    production = rows.get("industrial_production", {})
    velocity = _number(production.get("velocity_value"))
    if velocity is None:
        return None
    measures = []
    for indicator, label in (("core_pce_inflation", "core PCE"), ("core_cpi_inflation", "core CPI")):
        percentile = _number(rows.get(indicator, {}).get("level_percentile_10y"))
        measures.append(
            {
                "indicator": indicator,
                "label": label,
                "percentile_10y": percentile,
                "elevated": None if percentile is None else percentile >= elevated_percentile,
                "effects": _effects(summary, indicator),
            }
        )
    elevated = [m for m in measures if m["elevated"]]
    return {
        "production_velocity": velocity,
        "production_slowing": velocity < 0.0,
        "production_observation_date": str(production.get("observation_date") or "")[:10] or None,
        "measures": measures,
        "active": velocity < 0.0 and bool(elevated),
        "active_with": [m["label"] for m in elevated] if velocity < 0.0 else [],
    }


def _effects(summary: pd.DataFrame | None, indicator: str) -> list[dict[str, Any]]:
    if summary is None or summary.empty:
        return []
    rows = summary[
        (summary["asset_group"] == "equity")
        & (summary["inflation_indicator"] == indicator)
        & (summary["inflation_condition"] == "elevated")
        & (summary["additive_robust_candidate"].astype(bool))
    ]
    return [
        {
            "target": row.target,
            "horizon": row.horizon,
            "active": float(row.active_regime_mean_outcome),
            "inactive": float(row.inactive_regime_mean_outcome),
            "sample_start": str(row.sample_start)[:7],
            "active_months": int(row.active_regime_observations),
        }
        for row in rows.itertuples(index=False)
    ]


def build_risk_context(
    states: pd.DataFrame,
    outcomes: pd.DataFrame,
    current: Mapping[str, Any],
    production_read: Mapping[str, Any] | None,
    worst_share: float = 0.10,
) -> dict[str, Any]:
    """Current state, its risk tier and history, and the comparison with the other states."""
    table = tier_table(states, outcomes, worst_share)
    state = current.get("state")
    record = table["states"].get(state) if state else None
    normal = table["states"].get("normal")
    above_usual = bool(
        record
        and normal
        and state != "normal"
        and record["worst_drawdown_rate"] > normal["worst_drawdown_rate"] * 1.5
    ) or bool(production_read and production_read.get("active"))
    return {
        "state": state,
        "state_label": STATE_LABELS.get(state, state) if state else None,
        "tier": record["tier"] if record else None,
        "above_usual_for_stance": above_usual if state in {"normal", "normal_legacy", "macro_watch"} else None,
        "legacy_count": current.get("legacy_count"),
        "legacy_needed": current.get("legacy_needed"),
        "legacy_total": current.get("legacy_total"),
        "legacy_on": current.get("legacy_on", []),
        "history": table,
        "production_inflation": production_read,
        "note": "Descriptive: how often a worst-decile 3-month outcome followed each state since "
        f"{table['start'][:4] if table['start'] else 'n/a'}. Only the overlay's invested/defensive stance is a "
        "validated timing rule.",
    }


def current_state(
    month_end_levels: pd.Series,
    latest_level: float,
    values: pd.DataFrame,
    triggers: Mapping[str, tuple[str, float]],
    at_least: int,
    trend_months: int = 10,
    unemployment_buffer: float = 0.1,
) -> dict[str, Any]:
    """State today: the latest close against the 10-month average (the month in progress counts as one month)."""
    levels = month_end_levels.dropna().sort_index()
    average = float(levels.iloc[-trend_months:].mean()) if len(levels) >= trend_months else float("nan")
    latest = values.sort_index().iloc[-1]
    flags = legacy_flags(values.sort_index().iloc[[-1]], triggers).iloc[0]
    trend_up = latest_level >= average if np.isfinite(average) else None
    macro = None
    if pd.notna(latest.get("indpro_yoy")) and pd.notna(latest.get("unemployment_gap")):
        macro = bool(latest["indpro_yoy"] <= 0.0 or latest["unemployment_gap"] > unemployment_buffer)
    count = int(flags.sum()) if flags.notna().all() else None
    return {
        "state": _state(trend_up, macro, count, at_least) if trend_up is not None and macro is not None else None,
        "legacy_count": count,
        "legacy_needed": at_least,
        "legacy_total": len(triggers),
        "legacy_on": [TRIGGER_LABELS.get(metric, metric) for metric, flag in flags.items() if flag == 1.0],
        "as_of": f"{pd.Timestamp(values.index.max()):%Y-%m-%d}",
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
