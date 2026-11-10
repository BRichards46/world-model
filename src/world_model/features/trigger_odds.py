"""Historical odds that the next data point crosses a risk-overlay trigger.

Macro odds use first prints: for each month, the change between its first release and the prior month's
value as published in that same release. That is what the overlay sees on release day, and comparing values
within one vintage avoids jumps where an index was rebased between vintages.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_START = "1990-01-01"


def first_print_changes(
    vintage: pd.DataFrame,
    start: str | pd.Timestamp = DEFAULT_START,
    percent: bool = False,
    average_months: int = 12,
) -> pd.DataFrame:
    """Per month: the first-print change and the prior month's gap over its trailing average, both as published.

    ``vintage`` has ``observation_date``, ``value``, ``realtime_start`` and ``realtime_end`` columns.
    """
    table = vintage.copy()
    table["observation_date"] = pd.to_datetime(table["observation_date"])
    table["realtime_start"] = pd.to_datetime(table["realtime_start"])
    table["realtime_end"] = pd.to_datetime(table["realtime_end"], errors="coerce").fillna(pd.Timestamp.max.normalize())
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.dropna(subset=["value"])
    first_release = table.groupby("observation_date")["realtime_start"].min().sort_index()
    start = pd.Timestamp(start)
    rows = []
    for observation, released in first_release[first_release.index >= start].items():
        known = table[(table["realtime_start"] <= released) & (table["realtime_end"] >= released)]
        known = known.sort_values("realtime_start").drop_duplicates("observation_date", keep="last")
        values = known.set_index("observation_date")["value"].sort_index()
        previous = (observation - pd.DateOffset(months=1)).normalize()
        if observation not in values.index or previous not in values.index:
            continue
        current, prior = float(values.loc[observation]), float(values.loc[previous])
        change = 100.0 * (current / prior - 1.0) if percent else current - prior
        window = values[(values.index > previous - pd.DateOffset(months=average_months)) & (values.index <= previous)]
        gap = prior - float(window.mean()) if len(window) >= average_months - 1 else np.nan
        rows.append({"observation_date": observation, "change": change, "prior_gap": gap})
    return pd.DataFrame.from_records(rows, columns=["observation_date", "change", "prior_gap"])


def frequency_at_least(changes: pd.Series, threshold: float) -> dict[str, Any] | None:
    values = pd.to_numeric(changes, errors="coerce").dropna()
    if values.empty:
        return None
    hits = int((values >= threshold).sum())
    return {"probability": hits / len(values), "hits": hits, "months": int(len(values))}


def frequency_at_most(changes: pd.Series, threshold: float) -> dict[str, Any] | None:
    values = pd.to_numeric(changes, errors="coerce").dropna()
    if values.empty:
        return None
    hits = int((values <= threshold).sum())
    return {"probability": hits / len(values), "hits": hits, "months": int(len(values))}


def market_break_odds(
    index_level: pd.Series,
    threshold_change_pct: float,
    as_of: pd.Timestamp,
    start: str | pd.Timestamp = DEFAULT_START,
    vol_days: int = 63,
) -> dict[str, Any] | None:
    """Chance the index ends the month at or below ``threshold_change_pct`` from its latest close.

    Uses every historical stretch of the same number of trading days, each scaled by the volatility of the
    63 days before it, and compares with the threshold scaled by today's volatility. Holidays are ignored when
    counting the trading days left.
    """
    levels = pd.to_numeric(index_level, errors="coerce").dropna().sort_index()
    if len(levels) < vol_days + 2:
        return None
    last = pd.Timestamp(levels.index[-1]).normalize()
    month_end = (as_of.normalize() + pd.offsets.MonthEnd(0)).normalize()
    days = len(pd.bdate_range(last + pd.Timedelta(days=1), month_end))
    if days <= 0:
        return None
    log_levels = np.log(levels)
    daily = log_levels.diff()
    volatility = daily.rolling(vol_days, min_periods=vol_days).std()
    current_vol = float(volatility.iloc[-1])
    if not np.isfinite(current_vol) or current_vol <= 0.0:
        return None
    forward = log_levels.shift(-days) - log_levels
    scaled = (forward / (volatility * np.sqrt(days)))[levels.index >= pd.Timestamp(start)].dropna()
    if scaled.empty:
        return None
    threshold = np.log1p(threshold_change_pct / 100.0) / (current_vol * np.sqrt(days))
    hits = int((scaled <= threshold).sum())
    return {
        "probability": hits / len(scaled),
        "trading_days_left": days,
        "annualized_volatility": 100.0 * current_vol * np.sqrt(252.0),
        "samples": int(len(scaled)),
    }
