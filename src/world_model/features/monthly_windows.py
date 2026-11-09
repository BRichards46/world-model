"""Calendar-aligned windows for monthly macro series.

FRED occasionally has missing months (for example, October 2025 CPI and
unemployment were never published). Positional ``shift(12)`` or ``iloc[-12:]``
silently turns those into 13-month comparisons, so these helpers align by
calendar month instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna().sort_index()


def value_months_ago(series: pd.Series, months: int) -> float | None:
    """Value exactly ``months`` calendar months before the latest observation, if published."""
    values = _clean(series)
    if values.empty:
        return None
    key = (pd.Timestamp(values.index[-1]) - pd.DateOffset(months=months)).normalize()
    return float(values.loc[key]) if key in values.index else None


def pct_change_months(series: pd.Series, months: int) -> float | None:
    """Percent change of the latest value versus the same month ``months`` earlier."""
    values = _clean(series)
    prior = value_months_ago(values, months)
    if values.empty or prior is None or prior <= 0.0 or values.iloc[-1] <= 0.0:
        return None
    return 100.0 * (float(values.iloc[-1]) / prior - 1.0)


def trailing_months(series: pd.Series, months: int) -> pd.Series:
    """Observations within the trailing ``months`` calendar months, including the latest."""
    values = _clean(series)
    if values.empty:
        return values
    start = pd.Timestamp(values.index[-1]) - pd.DateOffset(months=months)
    return values[values.index > start]


def fill_missing_months(series: pd.Series) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Reindex a positive monthly series to month starts and log-interpolate interior gaps."""
    values = _clean(series)
    values = values[values > 0.0]
    if values.empty:
        return values.astype(float), []
    index = pd.DatetimeIndex(values.index).to_period("M").start_time.normalize()
    values = pd.Series(values.to_numpy(dtype=float), index=index)
    values = values[~values.index.duplicated(keep="last")].sort_index()
    full = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="MS"))
    missing = list(full.index[full.isna()])
    if missing:
        full = np.exp(np.log(full).interpolate(method="time"))
    return full.astype(float), missing


def trailing_average(series: pd.Series, months: int, max_missing: int = 1) -> float | None:
    """Average over the trailing calendar window, tolerating up to ``max_missing`` unpublished months."""
    window = trailing_months(series, months)
    if len(window) < months - max_missing:
        return None
    return float(window.mean())
