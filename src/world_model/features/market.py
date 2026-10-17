from __future__ import annotations

import numpy as np
import pandas as pd


def bounded(value: pd.Series | pd.DataFrame, lower: float = -1.0, upper: float = 1.0):
    return value.clip(lower=lower, upper=upper)


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    return close / close.shift(window) - 1.0


def realized_vol(close: pd.Series, window: int = 21, annualization: int = 252) -> pd.Series:
    returns = close.pct_change(fill_method=None)
    return returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(annualization)


def rolling_percentile_rank(series: pd.Series, window: int = 756, min_periods: int = 252) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        if np.isnan(values[-1]):
            return np.nan
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return np.nan
        return float((values <= values[-1]).sum() / len(values))

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def trend_score(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    sma50 = close.rolling(50, min_periods=40).mean()
    sma200 = close.rolling(200, min_periods=160).mean()

    score = pd.concat(
        [
            np.tanh(((close / sma50) - 1.0) / 0.05),
            np.tanh(((close / sma200) - 1.0) / 0.10),
            np.tanh(rolling_return(close, 63) / 0.12),
            np.tanh(rolling_return(close, 126) / 0.20),
        ],
        axis=1,
    ).mean(axis=1)
    return bounded(score)


def momentum_score(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    ret63 = rolling_return(close, 63)
    ret126 = rolling_return(close, 126)
    vol63 = realized_vol(close, 63).replace(0, np.nan)
    vol126 = realized_vol(close, 126).replace(0, np.nan)

    score = pd.concat(
        [
            np.tanh(ret63 / (vol63 / np.sqrt(4))),
            np.tanh(ret126 / (vol126 / np.sqrt(2))),
        ],
        axis=1,
    ).mean(axis=1)
    return bounded(score)


def volatility_score(equity_close: pd.Series, volatility_close: pd.Series | None = None) -> pd.Series:
    vol = realized_vol(equity_close, 21)
    vol_percentile = rolling_percentile_rank(vol)
    realized_component = 1.0 - (2.0 * vol_percentile)

    components = [realized_component.rename("realized_vol")]
    if volatility_close is not None:
        components.append((-trend_score(volatility_close)).rename("vol_trend"))
    return bounded(pd.concat(components, axis=1).mean(axis=1))


def breadth_score(breadth: pd.DataFrame) -> pd.Series:
    required = [
        "pct_above_50d",
        "pct_above_200d",
        "net_new_52w_high_low",
        "pct_advancing_10d",
    ]
    missing = [column for column in required if column not in breadth.columns]
    if missing:
        raise ValueError(f"Breadth data is missing required columns: {missing}")

    components = pd.concat(
        [
            ((breadth["pct_above_50d"] - 0.50) / 0.25).rename("above_50d"),
            ((breadth["pct_above_200d"] - 0.50) / 0.25).rename("above_200d"),
            (breadth["net_new_52w_high_low"] / 0.08).rename("new_high_low"),
            ((breadth["pct_advancing_10d"] - 0.50) / 0.15).rename("advancing"),
        ],
        axis=1,
    )
    return bounded(components.mean(axis=1))


def cot_crowding_score(cot_series: pd.Series, window: int = 260) -> pd.Series:
    percentile = rolling_percentile_rank(cot_series.astype(float), window=window, min_periods=104)
    score = -(2.0 * percentile - 1.0)
    return bounded(score)


def weighted_average(frame: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    frame = frame.astype(float)
    if frame.empty:
        return pd.Series(index=frame.index, dtype=float)
    if weights is None:
        return frame.mean(axis=1)

    aligned_weights = pd.Series({col: weights.get(col, 0.0) for col in frame.columns}, dtype=float)
    aligned_weights = aligned_weights[aligned_weights != 0]
    if aligned_weights.empty:
        return frame.mean(axis=1)

    selected = frame[aligned_weights.index]
    weighted = selected.multiply(aligned_weights, axis=1)
    denominator = selected.notna().multiply(aligned_weights.abs(), axis=1).sum(axis=1)
    return weighted.sum(axis=1).where(denominator > 0) / denominator
