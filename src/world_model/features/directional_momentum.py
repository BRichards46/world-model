from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


def normalized_futures_change(
    price: pd.Series,
    normalization_window: int = 252,
    normalization_min_periods: int = 63,
) -> pd.Series:
    """Roll-safe daily move for an additively back-adjusted futures series."""
    clean = _clean_series(price)
    scale = (
        clean.abs()
        .rolling(normalization_window, min_periods=normalization_min_periods)
        .median()
        .shift(1)
    )
    return (clean.diff() / scale.where(scale > 0.0)).replace([np.inf, -np.inf], np.nan)


def futures_momentum_history(
    price: pd.Series,
    horizons: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    daily_move = normalized_futures_change(price)
    return _build_history(daily_move, horizons, raw_mode="sum")


def equity_momentum_history(
    price: pd.Series,
    benchmark: pd.Series,
    horizons: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    aligned = pd.concat(
        [_clean_series(price).rename("asset"), _clean_series(benchmark).rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    asset_log_return = np.log(aligned["asset"] / aligned["asset"].shift(1))
    relative_log_return = asset_log_return - np.log(
        aligned["benchmark"] / aligned["benchmark"].shift(1)
    )
    history = _build_history(asset_log_return, horizons, raw_mode="compound")
    relative = _build_history(relative_log_return, horizons, raw_mode="compound")
    if history.empty:
        return history
    relative = relative.rename(
        columns={
            "raw_momentum": "relative_raw_momentum",
            "vol_adjusted_momentum": "relative_vol_adjusted_momentum",
            "realized_volatility": "relative_realized_volatility",
        }
    )
    return history.join(
        relative[
            [
                "relative_raw_momentum",
                "relative_vol_adjusted_momentum",
                "relative_realized_volatility",
            ]
        ],
        how="left",
    )


def classify_bias(
    rows: pd.DataFrame,
    classification: dict[str, Any],
    *,
    relative: bool = False,
) -> dict[str, Any]:
    raw_column = "relative_raw_momentum" if relative else "raw_momentum"
    adjusted_column = (
        "relative_vol_adjusted_momentum" if relative else "vol_adjusted_momentum"
    )
    available = rows.dropna(subset=[raw_column, adjusted_column, "weight"]).copy()
    minimum = int(classification.get("minimum_components", 3))
    if len(available) < minimum:
        return {
            "id": "unavailable",
            "label": "Unavailable",
            "tone": "neutral",
            "score": None,
            "raw_score": None,
            "alignment": f"{len(available)}/5 available",
            "confidence": "insufficient",
            "available_components": int(len(available)),
        }

    weights = available["weight"].astype(float)
    denominator = float(weights.abs().sum())
    score = float((available[adjusted_column].astype(float) * weights).sum() / denominator)
    raw_score = float((available[raw_column].astype(float) * weights).sum() / denominator)
    positive = int((available[raw_column] > 0).sum())
    negative = int((available[raw_column] < 0).sum())
    neutral = int(len(available) - positive - negative)
    direction_alignment = int(classification.get("directional_alignment", 3))
    strong_alignment = int(classification.get("strong_alignment", 4))
    positive_threshold = float(classification.get("positive_threshold", 0.25))
    strong_threshold = float(classification.get("strong_threshold", 1.0))
    raw_sign = 1 if raw_score > 0 else -1 if raw_score < 0 else 0
    adjusted_sign = 1 if score > 0 else -1 if score < 0 else 0
    agrees = raw_sign == adjusted_sign and raw_sign != 0

    if score >= strong_threshold and positive >= strong_alignment and agrees:
        bias_id, label, tone, confidence = "strong_positive", "Strong positive", "positive", "high"
    elif score >= positive_threshold and positive >= direction_alignment and agrees:
        bias_id, label, tone, confidence = "positive", "Positive", "positive", "moderate"
    elif score <= -strong_threshold and negative >= strong_alignment and agrees:
        bias_id, label, tone, confidence = "strong_negative", "Strong negative", "negative", "high"
    elif score <= -positive_threshold and negative >= direction_alignment and agrees:
        bias_id, label, tone, confidence = "negative", "Negative", "negative", "moderate"
    else:
        bias_id, label, tone, confidence = "mixed", "Mixed", "warning", "low"

    if positive == len(available):
        alignment = f"{positive}/{len(available)} positive"
    elif negative == len(available):
        alignment = f"{negative}/{len(available)} negative"
    elif positive > negative:
        alignment = f"{positive}/{len(available)} positive"
    elif negative > positive:
        alignment = f"{negative}/{len(available)} negative"
    else:
        alignment = "Split" if neutral == 0 else f"{neutral} neutral"

    return {
        "id": bias_id,
        "label": label,
        "tone": tone,
        "score": score,
        "raw_score": raw_score,
        "alignment": alignment,
        "confidence": confidence,
        "available_components": int(len(available)),
    }


def _build_history(
    daily_change: pd.Series,
    horizons: Iterable[dict[str, Any]],
    *,
    raw_mode: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    clean = daily_change.replace([np.inf, -np.inf], np.nan).sort_index()
    for horizon in horizons:
        days = int(horizon["days"])
        minimum = max(15, int(math.ceil(days * 0.8)))
        cumulative = clean.rolling(days, min_periods=days).sum()
        raw = np.expm1(cumulative) if raw_mode == "compound" else cumulative
        realized = clean.rolling(days, min_periods=minimum).std()
        adjusted = cumulative / (realized * math.sqrt(days)).where(realized > 0.0)
        part = pd.DataFrame(
            {
                "date": clean.index,
                "horizon_id": str(horizon["id"]),
                "horizon_label": str(horizon["label"]),
                "horizon_days": days,
                "weight": float(horizon["weight"]),
                "raw_momentum": raw.to_numpy(),
                "vol_adjusted_momentum": adjusted.replace([np.inf, -np.inf], np.nan).to_numpy(),
                "realized_volatility": realized.to_numpy(),
            }
        ).set_index(["date", "horizon_id"])
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def _clean_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float).sort_index()
    clean.index = pd.to_datetime(clean.index).normalize()
    return clean[~clean.index.duplicated(keep="last")]
