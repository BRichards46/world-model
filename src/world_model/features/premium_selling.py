from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def conditional_path_snapshot(
    prices: pd.DataFrame,
    horizons: list[dict[str, Any]],
    forward_days: int = 21,
    analog_count: int = 100,
) -> dict[str, Any]:
    """Estimate a one-month path from nearest historical multi-horizon states."""
    prices = prices[["open", "high", "low", "close"]].dropna().sort_index()
    close = prices["close"]
    daily_vol = close.pct_change(fill_method=None).rolling(63, min_periods=42).std().shift(1)
    features = pd.DataFrame(index=prices.index)
    weights: dict[str, float] = {}
    for horizon in horizons:
        days = int(horizon["days"])
        key = str(horizon["id"])
        features[key] = np.log(close / close.shift(days)) / (daily_vol * np.sqrt(days))
        weights[key] = float(horizon.get("weight", 1.0))
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError("Insufficient price history for conditional path features.")

    current_date = features.index[-1]
    current = features.iloc[-1]
    positions = pd.Series(np.arange(len(prices)), index=prices.index)
    weekly_dates = features.groupby(features.index.to_period("W-FRI")).tail(1).index
    rows: list[dict[str, Any]] = []
    for date in weekly_dates:
        position = int(positions.loc[date])
        if position + forward_days >= len(prices):
            continue
        entry = float(prices.iloc[position + 1]["open"])
        future = prices.iloc[position + 1 : position + forward_days + 1]
        path_returns = np.log(pd.concat([pd.Series([entry]), future["close"].reset_index(drop=True)]).astype(float)).diff().dropna()
        row = {
            "date": date,
            "terminal_return": float(future.iloc[-1]["close"] / entry - 1.0),
            "downside_mae": float(future["low"].min() / entry - 1.0),
            "upside_mfe": float(future["high"].max() / entry - 1.0),
            "forward_rv": float(path_returns.std(ddof=0) * np.sqrt(252) * 100),
        }
        row.update(features.loc[date].to_dict())
        rows.append(row)
    samples = pd.DataFrame(rows).set_index("date")
    if samples.empty:
        raise ValueError("Insufficient completed forward paths for conditional analogs.")

    feature_columns = list(weights)
    center = samples[feature_columns].median()
    scale = (samples[feature_columns] - center).abs().median().replace(0, np.nan)
    scale = scale.fillna(samples[feature_columns].std()).replace(0, 1.0).fillna(1.0)
    normalized = (samples[feature_columns] - center) / scale
    current_normalized = (current[feature_columns] - center) / scale
    weight_vector = pd.Series(weights).reindex(feature_columns)
    weight_vector = weight_vector / weight_vector.sum()
    distance = np.sqrt(((normalized - current_normalized) ** 2).mul(weight_vector, axis=1).sum(axis=1))
    count = min(int(analog_count), len(samples))
    analogs = samples.loc[distance.nsmallest(count).index].copy()
    return {
        "observation_date": current_date,
        "sample_start": samples.index.min(),
        "sample_end": samples.index.max(),
        "sample_count": int(len(samples)),
        "analog_count": int(len(analogs)),
        "components": {key: float(current[key]) for key in feature_columns},
        "conditional": summarize_paths(analogs),
        "unconditional": summarize_paths(samples),
        "analogs": analogs,
        "samples": samples,
    }


def summarize_paths(paths: pd.DataFrame) -> dict[str, float]:
    return {
        "median_return": float(paths["terminal_return"].median()),
        "positive_rate": float((paths["terminal_return"] > 0).mean()),
        "median_downside_mae": float(paths["downside_mae"].median()),
        "median_upside_mfe": float(paths["upside_mfe"].median()),
        "median_forward_rv": float(paths["forward_rv"].median()),
    }


def touch_statistics(paths: pd.DataFrame, put_distance_pct: float, call_distance_pct: float) -> dict[str, float]:
    put_threshold = -abs(float(put_distance_pct)) / 100.0
    call_threshold = abs(float(call_distance_pct)) / 100.0
    return {
        "put_touch_rate": float((paths["downside_mae"] <= put_threshold).mean()),
        "call_touch_rate": float((paths["upside_mfe"] >= call_threshold).mean()),
        "put_terminal_breach_rate": float((paths["terminal_return"] <= put_threshold).mean()),
        "call_terminal_breach_rate": float((paths["terminal_return"] >= call_threshold).mean()),
    }
