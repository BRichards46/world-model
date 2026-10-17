from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RealtimeIndicator:
    value: pd.Series
    latest_observation_date: pd.Series


def build_realtime_indicator_from_vintage(
    vintage: pd.DataFrame,
    transform: str,
    params: dict[str, Any] | None = None,
) -> RealtimeIndicator:
    params = params or {}
    known_values: dict[pd.Timestamp, float] = {}
    values: list[float] = []
    observation_dates: list[pd.Timestamp | pd.NaT] = []
    release_dates: list[pd.Timestamp] = []

    for release_date, release_rows in vintage.groupby("realtime_start", sort=True):
        for _, row in release_rows.iterrows():
            known_values[pd.Timestamp(row["observation_date"])] = float(row["value"]) if pd.notna(row["value"]) else np.nan

        known = pd.Series(known_values, dtype=float).sort_index().dropna()
        release_dates.append(pd.Timestamp(release_date))
        observation_dates.append(known.index[-1] if not known.empty else pd.NaT)
        values.append(_compute_metric(known, transform, params))

    index = pd.DatetimeIndex(release_dates, name="realtime_start")
    return RealtimeIndicator(
        value=pd.Series(values, index=index, dtype=float).sort_index(),
        latest_observation_date=pd.Series(observation_dates, index=index, name="latest_observation_date").sort_index(),
    )


def score_indicator(
    value: pd.Series,
    window: int = 120,
    min_periods: int = 36,
    scale: float = 2.0,
    invert: bool = False,
) -> pd.Series:
    z = rolling_z_score(value, window=window, min_periods=min_periods)
    score = np.tanh(z / scale)
    if invert:
        score = -score
    return pd.Series(score, index=value.index, name=value.name)


def rolling_z_score(value: pd.Series, window: int = 120, min_periods: int = 36) -> pd.Series:
    clean = value.astype(float)
    baseline = clean.rolling(window=window, min_periods=min_periods)
    mean = baseline.mean().shift(1)
    std = baseline.std(ddof=0).shift(1)
    return (clean - mean) / std.replace(0.0, np.nan)


def weighted_average_signed(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    aligned_weights = pd.Series(weights, dtype=float).reindex(frame.columns).fillna(0.0)
    available = frame.notna()
    numerator = frame.fillna(0.0).multiply(aligned_weights, axis=1).sum(axis=1)
    denominator = available.multiply(aligned_weights.abs(), axis=1).sum(axis=1)
    return numerator / denominator.replace(0.0, np.nan)


def _compute_metric(known: pd.Series, transform: str, params: dict[str, Any]) -> float:
    if known.empty:
        return float("nan")

    latest_date = pd.Timestamp(known.index[-1])
    latest_value = float(known.iloc[-1])
    transform = transform.lower()

    if transform == "level":
        return latest_value
    if transform == "yoy_pct":
        return _pct_change(known, latest_date, period_months=int(params.get("period_months", 12)), params=params)
    if transform == "pct_change":
        return _pct_change(known, latest_date, period_months=int(params["period_months"]), params=params)
    if transform == "annualized_pct_change":
        period_months = int(params["period_months"])
        change = _pct_change(known, latest_date, period_months=period_months, params=params)
        if pd.isna(change):
            return float("nan")
        prior = latest_value / (1.0 + change)
        if latest_value <= 0.0 or prior <= 0.0:
            return float("nan")
        return (latest_value / prior) ** (12.0 / period_months) - 1.0
    if transform == "diff":
        return _diff(known, latest_date, period_months=int(params["period_months"]), params=params)
    if transform == "yoy_diff":
        return _diff(known, latest_date, period_months=int(params.get("period_months", 12)), params=params)

    raise ValueError(f"Unsupported economic transform: {transform}")


def _pct_change(known: pd.Series, latest_date: pd.Timestamp, period_months: int, params: dict[str, Any]) -> float:
    latest_value = float(known.iloc[-1])
    prior = _prior_value(known, latest_date, period_months=period_months, params=params)
    if pd.isna(prior) or prior == 0.0:
        return float("nan")
    return latest_value / float(prior) - 1.0


def _diff(known: pd.Series, latest_date: pd.Timestamp, period_months: int, params: dict[str, Any]) -> float:
    latest_value = float(known.iloc[-1])
    prior = _prior_value(known, latest_date, period_months=period_months, params=params)
    if pd.isna(prior):
        return float("nan")
    return latest_value - float(prior)


def _prior_value(known: pd.Series, latest_date: pd.Timestamp, period_months: int, params: dict[str, Any]) -> float:
    prior_date = latest_date - pd.DateOffset(months=period_months)
    max_gap_days = params.get("max_prior_gap_days")
    if max_gap_days is None:
        max_gap_days = 45 if period_months >= 12 else max(45, int(period_months * 31 * 0.75))

    candidates = known.loc[known.index <= prior_date]
    if candidates.empty:
        return float("nan")
    candidate_date = pd.Timestamp(candidates.index[-1])
    if max_gap_days is not None and (prior_date - candidate_date).days > int(max_gap_days):
        return float("nan")
    return float(candidates.iloc[-1])
