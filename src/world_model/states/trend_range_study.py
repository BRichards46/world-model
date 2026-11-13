"""Hedgeye-style TREND / TRADE study, and whether the quad adds to trend.

Three questions, each tested on risk-balanced portfolios built only from data
known at the signal date:

1. TREND: does following intermediate-term trend (vol-adjusted 3-, 6- and
   12-month momentum) earn money across our assets?
2. TRADE / risk range: inside a trend, does entering at the favorable end of a
   short-term volatility band (buy dips in uptrends, sell rallies in
   downtrends) beat entering at the unfavorable end? Also: does fading the
   short-term band on its own (pure reversal) work?
3. Quad filter: among trend positions, do the ones the quad playbook agrees
   with beat the ones it disagrees with? The null is "the quad adds nothing
   beyond trend"; it is tested by rotating the quad sequence against dates.

Levels are cumulative log performance ×100 (Norgate total return, CSI
normalized price change), so level differences are percentage moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.states.quad_asset_performance import _benjamini_hochberg
from world_model.states.quad_playbook_test import QUADS, _newey_west_t


@dataclass(frozen=True)
class TrendRangeSettings:
    trend_lookbacks: tuple[int, ...] = (63, 126, 252)
    volatility_days: int = 63
    range_days: int = 15
    lag_days: int = 1
    max_fill_days: int = 3


@dataclass(frozen=True)
class TrendRangeResult:
    range_returns: pd.DataFrame
    quad_returns: pd.DataFrame
    summary: pd.DataFrame
    snapshot: pd.DataFrame


# --- signals -----------------------------------------------------------------


def aligned_levels(cumulative: Mapping[str, pd.Series], calendar: pd.DatetimeIndex, max_fill_days: int = 3) -> pd.DataFrame:
    """Cumulative levels on one trading calendar; short gaps (other holidays) are carried forward."""
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    columns = {}
    for asset, series in cumulative.items():
        clean = series.dropna().sort_index()
        clean = clean[~clean.index.duplicated(keep="last")]
        if clean.empty:
            continue
        union = clean.reindex(clean.index.union(calendar)).ffill(limit=max_fill_days)
        aligned = union.reindex(calendar)
        aligned[aligned.index < clean.index[0]] = np.nan
        aligned[aligned.index > clean.index[-1]] = np.nan
        columns[asset] = aligned
    return pd.DataFrame(columns, index=calendar)


def daily_volatility(levels: pd.DataFrame, days: int) -> pd.DataFrame:
    """Trailing standard deviation of daily moves (same units as level differences)."""
    return levels.diff().rolling(days, min_periods=max(2, days * 2 // 3)).std()


def trend_score(levels: pd.DataFrame, volatility: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    """Average vol-adjusted momentum over the lookbacks; NaN unless every lookback is available."""
    parts = [(levels - levels.shift(days)) / (volatility * np.sqrt(days)) for days in lookbacks]
    stacked = np.stack([part.to_numpy() for part in parts])
    values = np.where(np.isfinite(stacked).all(axis=0), stacked.mean(axis=0), np.nan)
    return pd.DataFrame(values, index=levels.index, columns=levels.columns)


def range_position(levels: pd.DataFrame, volatility: pd.DataFrame, days: int) -> pd.DataFrame:
    """Distance from the short moving average in units of horizon volatility (the TRADE band)."""
    average = levels.rolling(days, min_periods=days).mean()
    return (levels - average) / (volatility * np.sqrt(days))


def forward_moves(levels: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    """Move from ``lag`` to ``lag + horizon`` sessions after each date (positional on the calendar)."""
    return levels.shift(-(lag + horizon)) - levels.shift(-lag)


def weekly_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar week."""
    series = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(series.groupby(calendar.to_period("W-FRI")).max().to_numpy())


def to_trading_dates(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day on or before each date."""
    positions = calendar.searchsorted(dates, side="right") - 1
    return pd.DatetimeIndex([calendar[p] if p >= 0 else pd.NaT for p in positions])


# --- portfolios --------------------------------------------------------------


def risk_portfolio(positions: pd.DataFrame, volatility: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
    """Inverse-vol weighted, unit-gross portfolio of signed positions; NaN when nothing is held."""
    weights = positions / volatility.reindex_like(positions)
    weights = weights.where(np.isfinite(weights) & forward.reindex_like(positions).notna() & positions.ne(0), 0.0)
    gross = weights.abs().sum(axis=1)
    normalized = weights.div(gross.where(gross > 0), axis=0)
    returns = (normalized * forward.reindex_like(positions).fillna(0.0)).sum(axis=1)
    return returns.where(gross > 0)


def range_timing_returns(
    trend: pd.DataFrame,
    band: pd.DataFrame,
    volatility: pd.DataFrame,
    forward: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Trend positions split by where price sits in the short-term band."""
    side = np.sign(trend).fillna(0.0)
    location = side * band  # negative = favorable (dip in an uptrend, rally in a downtrend)
    favorable = side.where(location <= -threshold, 0.0)
    unfavorable = side.where(location >= threshold, 0.0)
    fade = -np.sign(band).fillna(0.0)
    return pd.DataFrame(
        {
            "trend": risk_portfolio(side, volatility, forward),
            "trend_favorable": risk_portfolio(favorable, volatility, forward),
            "trend_unfavorable": risk_portfolio(unfavorable, volatility, forward),
            "trade_fade": risk_portfolio(fade, volatility, forward),
        }
    )


def quad_split_returns(
    trend: pd.DataFrame,
    volatility: pd.DataFrame,
    forward: pd.DataFrame,
    scores: Mapping[int, pd.Series],
) -> dict[int, pd.DataFrame]:
    """For each quad, trend positions the playbook agrees with vs disagrees with."""
    side = np.sign(trend).fillna(0.0)
    result = {}
    for quad in QUADS:
        score = scores[quad].reindex(side.columns).fillna(0.0)
        alignment = side * score
        result[quad] = pd.DataFrame(
            {
                "agree": risk_portfolio(side.where(alignment > 0, 0.0), volatility, forward),
                "disagree": risk_portfolio(side.where(alignment < 0, 0.0), volatility, forward),
            }
        )
    return result


# --- statistics --------------------------------------------------------------


def performance_row(series: pd.Series, periods_per_year: float, overlap: int) -> dict[str, Any]:
    clean = series.dropna()
    values = clean.to_numpy(dtype=float)
    if len(values) < 30:
        return {"periods": int(len(values))}
    midpoint = len(values) // 2
    recent = clean[clean.index >= pd.Timestamp("2010-01-01")]
    deviation = values.std(ddof=1)
    t_stat = _newey_west_t(values, lags=max(1, overlap))
    return {
        "periods": int(len(values)),
        "start": f"{clean.index[0]:%Y-%m}",
        "annualized": float(values.mean() * periods_per_year),
        "sharpe": float(values.mean() / deviation * np.sqrt(periods_per_year)) if deviation > 0 else np.nan,
        "t_stat_hac": t_stat,
        "p_value": float(_two_sided_normal_p(t_stat)),
        "hit_rate": float((values > 0).mean()),
        "first_half": float(values[:midpoint].mean() * periods_per_year),
        "second_half": float(values[midpoint:].mean() * periods_per_year),
        "since_2010": float(recent.mean() * periods_per_year) if len(recent) >= 30 else np.nan,
    }


def _two_sided_normal_p(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    from math import erf, sqrt

    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))


def rotation_p(differences: pd.DataFrame, labels: pd.Series, minimum_gap: int) -> tuple[float, float]:
    """Observed mean agree-minus-disagree and one-sided rotation p (quad label sequence rotated vs dates)."""
    usable = labels.notna()
    matrix = differences[usable].reindex(columns=list(QUADS)).to_numpy()
    label_values = labels[usable].astype(int).to_numpy()
    index = np.arange(len(label_values))
    observed = float(np.nanmean(matrix[index, label_values - 1]))
    rotated = [
        np.nanmean(matrix[index, np.roll(label_values, shift) - 1])
        for shift in range(minimum_gap, len(label_values) - minimum_gap + 1)
    ]
    rotated = np.asarray(rotated, dtype=float)
    return observed, float((np.count_nonzero(rotated >= observed) + 1) / (len(rotated) + 1))


# --- study -------------------------------------------------------------------


def run_trend_range_study(
    cumulative: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
    quad_signals: pd.DataFrame,
    textbook_scores: Mapping[int, pd.Series],
    config: Mapping[str, Any],
) -> TrendRangeResult:
    trend_config = config.get("trend", {})
    range_config = config.get("range", {})
    settings = TrendRangeSettings(
        trend_lookbacks=tuple(int(x) for x in trend_config.get("lookbacks", (63, 126, 252))),
        volatility_days=int(trend_config.get("volatility_days", 63)),
        range_days=int(range_config.get("window_days", 15)),
        lag_days=int(config.get("availability_lag_trading_days", 1)),
    )
    start = pd.Timestamp(config.get("start", "1990-01-01"))
    levels = aligned_levels(cumulative, calendar, settings.max_fill_days)
    volatility = daily_volatility(levels, settings.volatility_days)
    trend_variants = {"composite": settings.trend_lookbacks}
    for name, lookbacks in trend_config.get("sensitivity", {}).items():
        trend_variants[str(name)] = tuple(int(x) for x in lookbacks)
    trends = {name: trend_score(levels, volatility, lookbacks) for name, lookbacks in trend_variants.items()}
    band = range_position(levels, volatility, settings.range_days)

    summary_rows: list[dict[str, Any]] = []
    range_frames = []

    # 1-2. TREND and TRADE on weekly signals.
    weeks = weekly_dates(levels.index)
    weeks = weeks[weeks >= start]
    for horizon_name, days in config.get("range_horizons", {"one_week": 5, "three_weeks": 15}).items():
        days = int(days)
        forward = forward_moves(levels, days, settings.lag_days).reindex(weeks)
        periods = 52.0 * 5.0 / days
        overlap = int(np.ceil(days / 5.0))
        for trend_name, trend in trends.items():
            for threshold in range_config.get("thresholds", (0.0, 0.5)):
                frame = range_timing_returns(trend.reindex(weeks), band.reindex(weeks), volatility.reindex(weeks), forward, float(threshold))
                frame["favorable_minus_unfavorable"] = frame["trend_favorable"] - frame["trend_unfavorable"]
                tagged = frame.add_prefix(f"{horizon_name}|{trend_name}|{threshold}|")
                range_frames.append(tagged)
                for column in ("trend", "trend_favorable", "trend_unfavorable", "favorable_minus_unfavorable", "trade_fade"):
                    if column in ("trend", "trade_fade") and float(threshold) != float(range_config.get("thresholds", (0.0,))[0]):
                        continue  # identical across thresholds
                    if column == "trade_fade" and trend_name != "composite":
                        continue  # does not depend on the trend definition
                    summary_rows.append(
                        {
                            "study": "trend_range",
                            "horizon": horizon_name,
                            "trend": trend_name,
                            "threshold": float(threshold),
                            "signal": "",
                            "series": column,
                            **performance_row(frame[column], periods, overlap),
                        }
                    )

    # 3. Quad filter on month-end signals.
    quad_assets = [asset for asset in config.get("quad_filter_assets", []) if asset in levels.columns]
    signal_dates = quad_signals.index[quad_signals.index >= start]
    trading = to_trading_dates(signal_dates, levels.index)
    valid = ~trading.isna()
    signal_dates, trading = signal_dates[valid], trading[valid]
    quad_frames = []
    minimum_gap = int(config.get("minimum_rotation_gap_months", 12))
    for horizon_name, days in config.get("quad_horizons", {"one_month": 21, "three_months": 63}).items():
        days = int(days)
        periods = 252.0 / days
        overlap = max(1, int(round(days / 21)))
        forward = forward_moves(levels[quad_assets], days, settings.lag_days).loc[trading]
        trend = trends["composite"][quad_assets].loc[trading]
        vol = volatility[quad_assets].loc[trading]
        for frame in (forward, trend, vol):
            frame.index = signal_dates
        split = quad_split_returns(trend, vol, forward, textbook_scores)
        differences = pd.DataFrame({quad: split[quad]["agree"] - split[quad]["disagree"] for quad in QUADS})
        trend_only = risk_portfolio(np.sign(trend).fillna(0.0), vol, forward)
        summary_rows.append(
            {"study": "quad_filter", "horizon": horizon_name, "trend": "composite", "threshold": np.nan, "signal": "", "series": "trend_only", **performance_row(trend_only, periods, overlap)}
        )
        quad_frames.append(trend_only.rename(f"{horizon_name}|trend_only"))
        for signal in quad_signals.columns:
            labels = quad_signals[signal].reindex(signal_dates)
            picked = {}
            for column in ("agree", "disagree"):
                table = pd.DataFrame({quad: split[quad][column] for quad in QUADS})
                values = [table.at[date, int(label)] if pd.notna(label) else np.nan for date, label in labels.items()]
                picked[column] = pd.Series(values, index=signal_dates, dtype=float)
            picked["agree_minus_disagree"] = picked["agree"] - picked["disagree"]
            # Practical version: hold only the agreeing trend positions, cash otherwise.
            picked["agree_only"] = picked["agree"].fillna(0.0).where(labels.notna())
            _, p_rotation = rotation_p(differences, labels, minimum_gap)
            for column, values in picked.items():
                quad_frames.append(values.rename(f"{horizon_name}|{signal}|{column}"))
                row = {
                    "study": "quad_filter",
                    "horizon": horizon_name,
                    "trend": "composite",
                    "threshold": np.nan,
                    "signal": signal,
                    "series": column,
                    **performance_row(values, periods, overlap),
                }
                if column == "agree_minus_disagree":
                    row["rotation_p"] = p_rotation
                summary_rows.append(row)

    summary = pd.DataFrame.from_records(summary_rows)
    tests = summary["series"].isin(["favorable_minus_unfavorable", "trade_fade", "trend", "agree_minus_disagree", "trend_only"])
    summary["test_p"] = np.where(summary["series"].eq("agree_minus_disagree"), summary.get("rotation_p"), summary["p_value"])
    summary["q_value"] = np.nan
    for (study, horizon), group in summary[tests].groupby(["study", "horizon"]):
        summary.loc[group.index, "q_value"] = _benjamini_hochberg(group["test_p"].fillna(1.0))
    summary["stable"] = np.sign(summary["first_half"]) == np.sign(summary["second_half"])
    summary["significant"] = summary["q_value"].le(float(config.get("false_discovery_rate", 0.10))) & summary["stable"]

    snapshot = current_snapshot(levels, trends["composite"], band, quad_signals, textbook_scores)
    return TrendRangeResult(
        range_returns=pd.concat(range_frames, axis=1),
        quad_returns=pd.concat(quad_frames, axis=1),
        summary=summary,
        snapshot=snapshot,
    )


def current_snapshot(
    levels: pd.DataFrame,
    trend: pd.DataFrame,
    band: pd.DataFrame,
    quad_signals: pd.DataFrame,
    textbook_scores: Mapping[int, pd.Series],
) -> pd.DataFrame:
    latest = quad_signals.ffill().iloc[-1] if not quad_signals.empty else pd.Series(dtype=float)
    rows = []
    for asset in levels.columns:
        series = trend[asset].dropna()
        if series.empty:
            continue
        date = series.index[-1]
        score = float(series.iloc[-1])
        position = float(band[asset].get(date, np.nan))
        side = np.sign(score)
        location = side * position
        row = {
            "asset_id": asset,
            "date": f"{date:%Y-%m-%d}",
            "trend_score": score,
            "trend": "bullish" if score > 0 else "bearish",
            "range_position": position,
            "range_entry": "favorable" if location <= -0.5 else "unfavorable" if location >= 0.5 else "middle",
        }
        for signal in ("projected_current", "projected_next"):
            quad = latest.get(signal)
            if quad is None or pd.isna(quad):
                continue
            stance = float(textbook_scores[int(quad)].get(asset, 0.0))
            row[f"{signal}_quad"] = int(quad)
            row[f"{signal}_agrees"] = "agrees" if stance * side > 0 else "disagrees" if stance * side < 0 else "neutral"
        rows.append(row)
    return pd.DataFrame.from_records(rows)
