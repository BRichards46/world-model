"""Plain-language outlook for the top of the macro dashboard.

Built only from the dashboard's own payloads (quad model, risk overlay, release forecasts and monitor rows),
so every number on it also appears, with its method, elsewhere on the page. The outlook says where the
regime and risk signals stand, where the quad model projects them, and what each upcoming release would
need to show to change them.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping

import pandas as pd

# Regime names used on the dashboard in place of quad numbers.
QUAD_NAME = {1: "Goldilocks", 2: "Reflation", 3: "Stagflation", 4: "Deflation"}

QUAD_SHORT = {
    1: "growth up, inflation down",
    2: "growth up, inflation up",
    3: "growth down, inflation up",
    4: "growth down, inflation down",
}

# (release id, label, monitor row that carries its expected date, forecast rows to show)
RELEASES: tuple[tuple[str, str, str, tuple[tuple[str, str, str], ...]], ...] = (
    ("cpi", "CPI", "core_cpi_inflation", (("headline_cpi_mom", "pipeline", "Headline"), ("core_cpi_mom", "pipeline", "Core"))),
    ("jobs", "Jobs report", "payroll_growth", (("payroll_change", "core", "Payrolls"),)),
    ("pce", "PCE inflation", "core_pce_inflation", (("core_pce_mom", "pipeline", "Core PCE"),)),
    ("industrial_production", "Industrial production", "industrial_production", ()),
    ("gdp", "GDP", "real_gdp_growth", ()),
)


def build_macro_outlook(
    quad: Mapping[str, Any] | None,
    overlay: Mapping[str, Any] | None,
    releases: Mapping[str, Any] | None,
    snapshot: Iterable[Mapping[str, Any]],
    risk_context: Mapping[str, Any] | None = None,
    benchmarks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    quad = quad or {}
    risk_context = risk_context or {}
    overlay = overlay or {}
    rows = {str(row.get("indicator_id")): row for row in snapshot}
    forecasts = {(row["target_id"], row["variant_id"]): row for row in (releases or {}).get("forecasts", [])}
    for target_id in ("headline_cpi_mom", "core_cpi_mom"):
        if (target_id, "bottom_up") in forecasts:
            # The bottom-up nowcast replaces the single-equation model wherever the outlook reads it.
            forecasts[(target_id, "pipeline")] = forecasts.pop((target_id, "bottom_up"))
    forecasts = _with_outside_views(forecasts, benchmarks or {})
    quarters = list(quad.get("quarters", []))
    current_index = next((i for i, q in enumerate(quarters) if q.get("is_current_quarter")), None)
    realized = [q for q in quarters if q.get("status") == "realized"]
    horizons = {int(h["horizon"]): h for h in quad.get("backtest", {}).get("horizons", [])}
    inputs = {item["id"]: item for item in overlay.get("inputs", [])}
    triggers = overlay.get("triggers", {}) or {}

    ahead = []
    if current_index is not None:
        for offset, quarter in enumerate(quarters[current_index : current_index + 3]):
            probability = _number(quarter.get("probability_top"))
            hit_rate = _number(horizons.get(offset, {}).get("quad_hit_rate"))
            quad_id = int(quarter.get("probability_quad") or quarter.get("quad"))
            ahead.append(
                {
                    "quarter": quarter.get("quarter"),
                    "status": quarter.get("status"),
                    "quad": quad_id,
                    "name": QUAD_NAME.get(quad_id, ""),
                    "description": QUAD_SHORT.get(quad_id, ""),
                    "probability": probability,
                    "conviction": _conviction(probability),
                    "historical_hit_rate": hit_rate,
                }
            )

    now = []
    if realized and ahead:
        last = realized[-1]
        current = ahead[0]
        now.append(
            {
                "label": "Regime",
                "text": f"{_q(last['quarter'])} was {QUAD_NAME.get(int(last['quad']), '')} ({QUAD_SHORT.get(int(last['quad']), '')}). "
                f"{_q(current['quarter'])} is tracking {current['name']} ({current['description']}), "
                f"{_pct(current['probability'])} likely.",
            }
        )
    market = _number(inputs.get("market_trend", {}).get("value"))
    market_trigger = _number((triggers.get("market_trend") or {}).get("threshold_change_pct"))
    if overlay.get("state_label"):
        text = _stance_text(overlay, risk_context)
        if market is not None:
            text += f" The S&P is {abs(market):.1f}% {'above' if market >= 0 else 'below'} its 10-month average"
            if market_trigger is not None and market >= 0:
                text += f"; a month-end close about {abs(market_trigger):.1f}% below today's level would break that trend"
                odds = (triggers.get("market_trend") or {}).get("odds")
                if odds:
                    text += f" ({_odds(odds['probability'])} by month-end at today's volatility)"
            text += "."
        risk_off = _risk_off_text(triggers)
        if risk_off:
            text += " " + risk_off
        now.append({"label": "Equity risk", "text": text})
        detail = _risk_detail(risk_context)
        if detail:
            now.append({"label": "Why", "text": detail})
    growth = _growth_text(quarters[current_index] if current_index is not None else None, inputs)
    if growth:
        now.append({"label": "Growth", "text": growth})
    labor = _labor_text(rows)
    if labor:
        now.append({"label": "Jobs", "text": labor})
    inflation = _inflation_text(rows, quarters[current_index] if current_index is not None else None)
    if inflation:
        now.append({"label": "Inflation", "text": inflation})

    upcoming = []
    for release_id, label, row_id, forecast_keys in RELEASES:
        row = rows.get(row_id, {})
        date = row.get("expected_next_release")
        if not date:
            continue
        shown = [_forecast_view(forecasts[key[:2]], key[2]) for key in forecast_keys if key[:2] in forecasts]
        upcoming.append(
            {
                "id": release_id,
                "label": label,
                "date": f"{_weekday(pd.Timestamp(date)):%Y-%m-%d}",
                "date_is_estimate": bool(row.get("expected_next_release_is_estimate")),
                "reference": _reference(release_id, row, shown),
                "forecasts": shown,
                "meaning": _meaning(release_id, quad, quarters, current_index, overlay, rows, forecasts),
            }
        )
    upcoming.sort(key=lambda item: item["date"])

    return {
        "as_of_date": overlay.get("as_of_date") or quad.get("as_of_date"),
        "headline": _headline(ahead, overlay, risk_context),
        "now": now,
        "ahead": ahead,
        "releases": upcoming,
        "notes": [
            "Regimes describe the growth and inflation backdrop; they are not trading signals. The equity risk "
            "overlay is the only validated signal here.",
            "Forecast ranges come from the model's own out-of-sample misses over the last five years. "
            + _validated_note(forecasts),
            "Market expectations are the median of Kalshi's prices on each range of the print. The market has been "
            "more accurate than our models and the Cleveland Fed nowcast on every release we score, so a surprise "
            "against the market is the real news.",
            "Release dates marked ≈ are estimated from the last release and can be off by a few days.",
        ],
    }


def _headline(ahead: list[dict[str, Any]], overlay: Mapping[str, Any], risk_context: Mapping[str, Any] | None = None) -> str:
    parts = []
    if ahead:
        current = ahead[0]
        parts.append(f"This quarter is tracking {current['name']}: {current['description']} ({_pct(current['probability'])}).")
    if len(ahead) > 1:
        nxt = ahead[1]
        parts.append(
            f"Next quarter leans {nxt['name']}, {nxt['description']} ({_pct(nxt['probability'])}, {nxt['conviction']} conviction)."
        )
    if overlay.get("state_label"):
        parts.append(f"Equity risk: {_stance_text(overlay, risk_context or {}).rstrip('.').lower()}.")
    return " ".join(parts)


STANCE = {
    "normal": "Stay invested",
    "macro_watch": "Stay invested, macro on watch",
    "market_warning": "Watch: trend break without macro confirmation",
    "defensive": "Reduce risk",
    "high_conviction": "Reduce risk",
}


def _stance_text(overlay: Mapping[str, Any], risk_context: Mapping[str, Any]) -> str:
    stance = STANCE.get(str(overlay.get("state_id")), str(overlay.get("state_label", "")))
    tier = risk_context.get("tier")
    if not tier:
        return f"{stance}."
    if risk_context.get("above_usual_for_stance"):
        return f"{stance}, but risk is above normal for this stance ({tier} overall)."
    return f"{stance}; risk is {tier}."


def _risk_detail(risk_context: Mapping[str, Any]) -> str:
    history = risk_context.get("history") or {}
    states = history.get("states") or {}
    state = risk_context.get("state")
    current = states.get(state) if state else None
    parts = []
    if current:
        line = abs(float(history.get("drawdown_line", 0.0))) * 100.0
        since = str(history.get("start") or "")[:4]
        episodes = len(current.get("drawdown_episodes") or [])
        rate = 100.0 * current["worst_drawdown_rate"]
        lead = ""
        if risk_context.get("legacy_on"):
            count, total = risk_context.get("legacy_count"), risk_context.get("legacy_total")
            lead = f"{count} of your {total} legacy triggers are on ({', '.join(risk_context['legacy_on'])}). "
        comparison = []
        for other, label in (("normal", "when the legacy triggers were clear"), ("defensive", "in the defensive state")):
            if other != state and other in states:
                comparison.append(f"{100.0 * states[other]['worst_drawdown_rate']:.0f}% {label}")
        parts.append(
            f"{lead}Since {since}, a 3-month drawdown of {line:.1f}% or worse followed {rate:.0f}% of months like this "
            f"({episodes} separate episode{'s' if episodes != 1 else ''})"
            + (f", against {' and '.join(comparison)}." if comparison else ".")
        )
    production = risk_context.get("production_inflation") or {}
    if production.get("active"):
        measure = next((m for m in production.get("measures", []) if m.get("elevated") and m.get("effects")), None)
        if measure:
            effects = {(e["target"], e["horizon"]): e for e in measure["effects"]}
            volatility = effects.get(("future_volatility", "three_months")) or effects.get(("future_volatility", "one_month"))
            drawdown = effects.get(("max_drawdown", "one_month"))
            pieces = []
            if volatility:
                horizon = "3-month" if volatility["horizon"] == "three_months" else "1-month"
                pieces.append(f"{horizon} volatility averaged {100 * volatility['active']:.1f}% vs {100 * volatility['inactive']:.1f}%")
            if drawdown:
                pieces.append(f"1-month drawdowns {100 * drawdown['active']:.1f}% vs {100 * drawdown['inactive']:.1f}%")
            start = min(e["sample_start"] for e in measure["effects"])[:4]
            parts.append(
                f"Slowing industrial production with elevated {measure['label']} is also on: since {start}, "
                + " and ".join(pieces)
                + " when both were on."
            )
    return " ".join(parts)


def _growth_text(current: Mapping[str, Any] | None, inputs: Mapping[str, Any]) -> str:
    parts = []
    if current and current.get("growth_assumption_source") == "nowcast" and current.get("growth_assumption_saar") is not None:
        parts.append(f"The Atlanta Fed nowcast has {_q(current['quarter'])} GDP at {float(current['growth_assumption_saar']):.1f}% annualized.")
    production = _number(inputs.get("industrial_production", {}).get("value"))
    if production is not None:
        parts.append(f"Industrial production is {production:+.1f}% YoY.")
    return " ".join(parts)


def _labor_text(rows: Mapping[str, Mapping[str, Any]]) -> str:
    rate = _detail(rows, "unemployment_rate", "Latest unemployment rate")
    payroll = _detail(rows, "payroll_growth", "Three-month average change")
    parts = []
    if rate is not None:
        parts.append(f"Unemployment {rate:.1f}%")
    if payroll is not None:
        parts.append(f"payrolls averaging {payroll:+.0f}k a month over three months")
    return (", ".join(parts) + ".") if parts else ""


def _inflation_text(rows: Mapping[str, Mapping[str, Any]], current: Mapping[str, Any] | None) -> str:
    parts = []
    core_3m = _detail(rows, "core_cpi_inflation", "Core CPI", unit="% 3m annualized")
    pce = _detail(rows, "core_pce_inflation", "Core PCE inflation", unit="% YoY")
    if core_3m is not None:
        parts.append(f"Core CPI is running {core_3m:.1f}% annualized over three months")
    if pce is not None:
        parts.append(f"core PCE is {pce:.1f}% YoY")
    text = (", ".join(parts) + ".") if parts else ""
    if current and current.get("inflation_yoy") is not None:
        text += f" Headline CPI is tracking {float(current['inflation_yoy']):.1f}% YoY for {_q(current['quarter'])}."
    return text.strip()


def _forecast_view(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "reference_month": row.get("reference_month"),
        "unit": row.get("unit"),
        "forecast": row.get("forecast"),
        "range_50": row.get("range_50"),
        "range_80": row.get("range_80"),
        "prior": row.get("prior"),
        "validated": bool(row.get("validated")),
        "market": row.get("market"),
        "market_range_80": row.get("market_range_80"),
        "market_range_open": row.get("market_range_open"),
        "cleveland": row.get("cleveland"),
        "track_record": row.get("track_record"),
    }


def _validated_note(forecasts: Mapping[tuple[str, str], Mapping[str, Any]]) -> str:
    labels = {"headline_cpi_mom": "headline CPI", "core_cpi_mom": "core CPI", "core_pce_mom": "core PCE", "payroll_change": "payrolls"}
    validated = [labels.get(key[0], key[0]) for key, row in forecasts.items() if row.get("validated")]
    if not validated:
        return "None of the release forecasts beats a naive forecast yet; all are previews."
    names = " and ".join(validated) if len(validated) <= 2 else ", ".join(validated[:-1]) + " and " + validated[-1]
    return f"Only {names} beat{'s' if len(validated) == 1 else ''} a naive forecast; the others are previews."


def _with_outside_views(
    forecasts: Mapping[tuple[str, str], Mapping[str, Any]], benchmarks: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Attach the market (Kalshi) and Cleveland Fed expectations for the same month, plus each source's record."""
    upcoming = {(item["target_id"], item["reference_month"]): item for item in benchmarks.get("upcoming", [])}
    track = benchmarks.get("track_record", {})
    merged = {}
    for key, row in forecasts.items():
        row = dict(row)
        outside = upcoming.get((key[0], row.get("reference_month")), {})
        row["market"] = outside.get("kalshi")
        row["market_range_80"] = outside.get("kalshi_range_80")
        row["market_range_open"] = outside.get("kalshi_range_open")
        row["cleveland"] = outside.get("cleveland")
        record = track.get(key[0])
        if record and record.get("mae"):
            row["track_record"] = {"releases": record.get("releases"), "start": record.get("start"), "mae": record["mae"]}
        merged[key] = row
    return merged


def _reference(release_id: str, row: Mapping[str, Any], shown: list[dict[str, Any]]) -> str | None:
    if shown and shown[0].get("reference_month"):
        return f"{pd.Timestamp(shown[0]['reference_month'] + '-01'):%B}"
    observed = row.get("observation_date")
    if not observed:
        return None
    observed = pd.Timestamp(observed)
    if release_id == "gdp":
        following = observed + pd.DateOffset(months=3)
        return f"Q{following.quarter} {following.year}"
    return f"{observed + pd.DateOffset(months=1):%B}"


def _meaning(
    release_id: str,
    quad: Mapping[str, Any],
    quarters: list[Mapping[str, Any]],
    current_index: int | None,
    overlay: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    forecasts: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[str]:
    current = quarters[current_index] if current_index is not None else None
    following = quarters[current_index + 1] if current_index is not None and current_index + 1 < len(quarters) else None
    triggers = overlay.get("triggers", {}) or {}
    lines: list[str] = []

    if release_id == "cpi":
        headline = forecasts.get(("headline_cpi_mom", "pipeline"))
        month = headline.get("reference_month") if headline else None
        comp = next((c for c in quad.get("cpi_comps", []) if c.get("month") == month), None)
        if headline and comp:
            market = _number(headline.get("market"))
            expected, source = (market, "the market's expectation") if market is not None else (float(headline["forecast"]), "our forecast")
            change = expected - float(comp["comp_mom"])
            lines.append(
                f"A year ago this month CPI rose {float(comp['comp_mom']):.2f}%"
                f"{' (estimated: that month was never published)' if comp.get('comp_interpolated') else ''}, "
                f"so any print above that lifts headline inflation YoY; {source} would move it about {change:+.1f} pp."
            )
            ours = _number(headline.get("forecast"))
            if market is not None and ours is not None and abs(ours - market) >= 0.1:
                record = (headline.get("track_record") or {}).get("mae", {})
                misses = (
                    f" On headline CPI the market has missed by {record['kalshi']:.2f} pp on average, our model by "
                    f"{record['ours']:.2f}."
                    if record.get("kalshi") is not None and record.get("ours") is not None
                    else " The market has been the more accurate on headline CPI."
                )
                lines.append(f"Our model ({ours:+.2f}%) is well {'above' if ours > market else 'below'} the market ({market:+.2f}%).{misses}")
        hurdle = _number(current.get("inflation_hurdle_mom")) if current else None
        if hurdle is not None and hurdle < 0:
            lines.append(
                f"{_q(current['quarter'])} inflation is already rising: it would take monthly CPI averaging below "
                f"{hurdle:.2f}% for the rest of the quarter to stop it."
            )
        elif hurdle is not None:
            lines.append(f"{_q(current['quarter'])} inflation rises if the rest of the quarter averages above {hurdle:.2f}% a month.")
        next_hurdle = _number(following.get("inflation_hurdle_mom")) if following else None
        base = _number(quad.get("assumptions", {}).get("inflation_base_mom"))
        if next_hurdle is not None and base is not None:
            lines.append(
                f"{_q(following['quarter'])} inflation rises only if monthly CPI averages above {next_hurdle:.2f}% "
                f"(the model assumes {base:.2f}%). A run of prints at that pace would undercut the "
                f"{QUAD_NAME.get(int(following.get('probability_quad') or following.get('quad') or 0), 'current')} lean."
            )
        lines.append("Not a risk-overlay input: inflation failed every GTT risk test.")

    elif release_id == "jobs":
        trigger = triggers.get("unemployment_trend")
        if trigger:
            threshold = float(trigger["threshold_rate"])
            published = trigger.get("published_threshold") or math.floor(threshold * 10.0 + 1e-9) / 10.0 + 0.1
            line = (
                f"The overlay's unemployment warning needs the rate above {threshold:.2f}%, so a print of "
                f"{float(published):.1f}% or higher (last {float(trigger['latest_rate']):.1f}%)."
            )
            odds = trigger.get("odds")
            if odds:
                line += (
                    f" Since 1990, when unemployment started {odds.get('starting_state', 'where it is now')} its "
                    f"12-month average as it does now, the first print jumped {float(odds['required_jump']):.1f} pp or "
                    f"more in {odds['hits']} of {odds['months']} months ({_odds(odds['probability'])})."
                )
            lines.append(line)
            lines.append("That alone is a watch, not risk-off: the S&P must also close a month below its 10-month average.")
        payroll = forecasts.get(("payroll_change", "core"))
        market = _number(payroll.get("market")) if payroll else None
        if payroll and market is not None and abs(float(payroll["forecast"]) - market) >= 50.0:
            lines.append(
                f"The market expects about {market:+.0f}k against our {float(payroll['forecast']):+.0f}k. Our payroll "
                "forecasts have run high and the market has been the more accurate."
            )
        if payroll and payroll.get("range_80"):
            low, high = payroll["range_80"]
            if high < 0:
                lines.append("Our range sits entirely below zero, which would be a recession-style print.")
            elif low < 0:
                lines.append("The low end of our range is a negative payroll print.")

    elif release_id == "pce":
        row = forecasts.get(("core_pce_mom", "pipeline"))
        if row and row.get("validated"):
            lines.append("Beats a naive model in every test period.")
        lines.append("The Fed's preferred inflation gauge. Context only: not a regime or risk-overlay input.")

    elif release_id == "industrial_production":
        trigger = triggers.get("industrial_production")
        if trigger:
            drop = float(trigger["threshold_change_pct"])
            odds = trigger.get("odds")
            history = (
                f" First prints fell that much in {odds['hits']} of {odds['months']} months since 1990 ({_odds(odds['probability'])})."
                if odds
                else ""
            )
            if drop < 0:
                lines.append(
                    f"The overlay's production warning needs output to fall {abs(drop):.1f}% in a single month "
                    f"(YoY to zero).{history}"
                )
            else:
                lines.append(f"Production is already below a year ago unless it rises {drop:.1f}% this month.")

    elif release_id == "gdp":
        if current and current.get("growth_hurdle_saar") is not None:
            hurdle = float(current["growth_hurdle_saar"])
            line = f"{_q(current['quarter'])} growth counts as accelerating unless GDP comes in below {hurdle:.1f}% annualized"
            if current.get("growth_assumption_saar") is not None:
                line += f"; the current estimate is {float(current['growth_assumption_saar']):.1f}%"
            lines.append(line + ".")
        if following and following.get("growth_hurdle_saar") is not None and following.get("growth_assumption_saar") is not None:
            need = float(following["growth_hurdle_saar"])
            expected = float(following["growth_assumption_saar"])
            verdict = "a coin flip" if abs(expected - need) < 0.25 else "likely" if expected > need else "unlikely"
            lines.append(
                f"{_q(following['quarter'])} growth accelerates only above {need:.2f}% annualized; leading indicators "
                f"point to {expected:.2f}%, so {verdict}."
            )
    return lines


def _q(label: Any) -> str:
    """'2026Q4' -> 'Q4 2026'."""
    text = str(label or "")
    return f"Q{text[-1]} {text[:4]}" if len(text) == 6 and text[4] == "Q" else text


def _weekday(date: pd.Timestamp) -> pd.Timestamp:
    """Estimated dates are last release plus a month; US data are not released at weekends, so use the Friday."""
    return date - pd.Timedelta(days=date.weekday() - 4) if date.weekday() >= 5 else date


def _risk_off_text(triggers: Mapping[str, Any]) -> str:
    """Risk-off needs a trend break and a macro trigger; summarise the macro odds for the next releases."""
    macro = [
        (label, (triggers.get(key) or {}).get("odds"))
        for key, label in (("unemployment_trend", "unemployment"), ("industrial_production", "production"))
    ]
    macro = [(label, odds) for label, odds in macro if odds]
    if not macro:
        return ""
    parts = " and ".join(f"{label} {_odds(odds['probability'])}" for label, odds in macro)
    return f"Risk-off also needs a macro trigger; historical odds on the next releases: {parts}."


def _odds(probability: Any) -> str:
    value = _number(probability)
    if value is None:
        return "n/a"
    if value < 0.01:
        return "under 1%"
    return f"about {100.0 * value:.0f}%"


def _detail(rows: Mapping[str, Mapping[str, Any]], indicator: str, label: str, unit: str | None = None) -> float | None:
    metrics = rows.get(indicator, {}).get("detail_metrics") or []
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    for metric in metrics:
        if metric.get("label") == label and (unit is None or metric.get("unit") == unit):
            return _number(metric.get("value"))
    return None


def _conviction(probability: float | None) -> str:
    if probability is None:
        return "unknown"
    if probability >= 0.7:
        return "high"
    if probability >= 0.5:
        return "moderate"
    return "low"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct(probability: float | None) -> str:
    return "n/a" if probability is None else f"{100.0 * probability:.0f}%"
