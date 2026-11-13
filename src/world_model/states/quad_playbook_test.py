"""Follow-vs-fade test of quad playbooks.

At each month-end signal date, a quad playbook assigns +1 to expected leaders
and -1 to expected laggards. Weights are scaled by each asset's trailing
volatility and normalized to unit gross exposure, giving a risk-balanced
long/short portfolio. "Follow" holds that portfolio over the next horizon;
"fade" (the stale-quad recovery idea) holds the opposite.

Two playbooks:
* ``textbook``: a commonly published mapping, fixed in config in advance.
* ``learned``: each quad's leaders/laggards from vol-scaled, cross-sectionally
  demeaned in-quarter returns, using only quarters whose realized quad was
  known (from FRED vintages) on the signal date.

Significance: the quad label sequence is circularly rotated against dates. For
each rotation, the portfolio for the rotated quad is evaluated on the actual
returns, preserving autocorrelation and the playbook structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from world_model.features.growth_inflation_quad import quarter_start
from world_model.states.growth_inflation_quad import GrowthInflationQuadEngine, _realized_quads
from world_model.states.quad_asset_performance import _benjamini_hochberg, forward_returns_at


QUADS = (1, 2, 3, 4)


@dataclass(frozen=True)
class PlaybookTestResult:
    portfolio_returns: pd.DataFrame
    summary: pd.DataFrame


def textbook_scores(config: Mapping[str, Any], assets: list[str]) -> dict[int, pd.Series]:
    playbook = config.get("textbook", {})
    scores = {}
    for quad in QUADS:
        entry = playbook.get(quad, playbook.get(str(quad), {}))
        values = pd.Series(0.0, index=assets)
        values[[asset for asset in entry.get("long", []) if asset in values.index]] = 1.0
        values[[asset for asset in entry.get("short", []) if asset in values.index]] = -1.0
        scores[quad] = values
    return scores


def quarter_returns(cumulative: Mapping[str, pd.Series], assets: list[str], volatility: pd.DataFrame) -> pd.DataFrame:
    """Vol-scaled, cross-sectionally demeaned returns for each calendar quarter."""
    rows = {}
    start = max(pd.Timestamp("1990-01-01"), min(series.dropna().index.min() for series in cumulative.values()))
    for quarter in pd.date_range(quarter_start(start), pd.Timestamp.today(), freq="QS"):
        end = quarter + pd.offsets.QuarterEnd(0)
        values = {}
        for asset in assets:
            series = cumulative[asset].dropna()
            if series.empty or series.index[0] > quarter or series.index[-1] < end:
                continue
            vol = volatility[asset].asof(quarter) if asset in volatility else np.nan
            if not np.isfinite(vol) or vol <= 0:
                continue
            values[asset] = (series.asof(end) - series.asof(quarter)) / vol
        if len(values) >= 4:
            row = pd.Series(values)
            rows[quarter] = row - row.mean()
    return pd.DataFrame(rows).T.reindex(columns=assets)


def learned_scores(
    known_quads: Mapping[pd.Timestamp, int],
    in_quarter: pd.DataFrame,
    as_of: pd.Timestamp,
    minimum_quarters: int,
    leaders: int,
) -> dict[int, pd.Series]:
    """Leaders (+1) and laggards (-1) per quad from quarters known by ``as_of``."""
    labels = pd.Series(known_quads, dtype=float)
    if labels.empty:
        return {quad: pd.Series(0.0, index=in_quarter.columns) for quad in QUADS}
    labels.index = pd.DatetimeIndex(labels.index)
    labels = labels[labels.index + pd.offsets.QuarterEnd(0) < as_of]
    scores = {}
    for quad in QUADS:
        quarters = labels.index[labels == quad].intersection(in_quarter.index)
        values = pd.Series(0.0, index=in_quarter.columns)
        if len(quarters) >= minimum_quarters:
            average = in_quarter.loc[quarters].mean().dropna()
            if len(average) >= 2 * leaders:
                ranked = average.sort_values()
                values[ranked.index[-leaders:]] = 1.0
                values[ranked.index[:leaders]] = -1.0
        scores[quad] = values
    return scores


def portfolio_return(scores: pd.Series, volatility: pd.Series, forward: pd.Series) -> float:
    usable = scores[(scores != 0) & volatility.reindex(scores.index).gt(0) & forward.reindex(scores.index).notna()]
    if usable.empty or usable.gt(0).sum() == 0 or usable.lt(0).sum() == 0:
        return np.nan
    weights = usable / volatility.reindex(usable.index)
    long = weights[weights > 0] / weights[weights > 0].sum() * 0.5
    short = weights[weights < 0] / weights[weights < 0].abs().sum() * 0.5
    combined = pd.concat([long, short])
    return float((combined * forward.reindex(combined.index)).sum())


def run_playbook_test(
    engine: GrowthInflationQuadEngine | None,
    signals: pd.DataFrame,
    cumulative: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
    config: Mapping[str, Any],
    horizons: Mapping[str, int],
    lag_days: int,
    false_discovery_rate: float = 0.10,
    minimum_rotation_gap: int = 12,
    known_quads: Callable[[pd.Timestamp], Mapping[pd.Timestamp, int]] | None = None,
) -> PlaybookTestResult:
    """Follow/fade returns for each signal. ``known_quads(date)`` gives the realized quads known on a date for the
    learned playbook; by default they are the GDP quad's, rebuilt from ``engine``'s vintages."""
    settings = config.get("playbook", {})
    assets = [asset for asset in settings.get("assets", []) if asset in cumulative]
    window = int(settings.get("volatility_window_days", 252))
    daily = pd.DataFrame({asset: cumulative[asset] for asset in assets}).sort_index().diff()
    volatility = daily.rolling(window, min_periods=window // 2).std() * np.sqrt(252.0)
    in_quarter = quarter_returns(cumulative, assets, volatility)
    textbook = textbook_scores(settings, assets)

    known_by_date = {}
    for date in signals.index:
        if known_quads is not None:
            known_by_date[date] = dict(known_quads(date))
            continue
        realized = _realized_quads(engine.inputs_as_of(date))
        known_by_date[date] = {quarter: value["quad"] for quarter, value in realized.items()}

    records = []
    for horizon, days in horizons.items():
        forward = forward_returns_at(signals.index, calendar, {asset: cumulative[asset] for asset in assets}, days, lag_days)
        for date in signals.index:
            # Per-column last value on or before the date (DataFrame.asof would require every column).
            vol = volatility.loc[:date].ffill().iloc[-1] if not volatility.loc[:date].empty else pd.Series(dtype=float)
            learned = learned_scores(
                known_by_date[date],
                in_quarter,
                date,
                int(settings.get("learned_minimum_quarters", 6)),
                int(settings.get("learned_leaders", 3)),
            )
            for playbook_name, scores in (("textbook", textbook), ("learned", learned)):
                for quad in QUADS:
                    records.append(
                        {
                            "horizon": horizon,
                            "signal_date": date,
                            "playbook": playbook_name,
                            "quad": quad,
                            "follow_return": portfolio_return(scores[quad], vol, forward.loc[date]),
                        }
                    )
    by_quad = pd.DataFrame.from_records(records)
    summary_rows = []
    returns_rows = []
    for (horizon, playbook_name), group in by_quad.groupby(["horizon", "playbook"]):
        table = group.pivot(index="signal_date", columns="quad", values="follow_return").reindex(columns=list(QUADS))
        periods_per_year = 252.0 / float(horizons[horizon])
        overlap = max(1, int(round(horizons[horizon] / 21)))
        for signal in signals.columns:
            labels = signals[signal].reindex(table.index)
            usable = labels.notna() & table.notna().any(axis=1)
            label_values = labels[usable].astype(int).to_numpy()
            matrix = table[usable].to_numpy()
            if len(label_values) < 60:
                continue
            index = np.arange(len(label_values))
            realized = matrix[index, label_values - 1]
            valid = np.isfinite(realized)
            series = pd.Series(realized, index=table.index[usable])
            returns_rows.append(series.rename((horizon, playbook_name, signal)))
            mean = float(np.nanmean(realized))
            rotations = []
            for shift in range(minimum_rotation_gap, len(label_values) - minimum_rotation_gap + 1):
                rolled = np.roll(label_values, shift)
                rotated = matrix[index, rolled - 1]
                rotations.append(np.nanmean(rotated))
            rotations = np.asarray(rotations, dtype=float)
            p_follow = float((np.count_nonzero(rotations >= mean) + 1) / (len(rotations) + 1))
            p_fade = float((np.count_nonzero(rotations <= mean) + 1) / (len(rotations) + 1))
            clean = realized[valid]
            midpoint = len(clean) // 2
            for direction, sign, p_value in (("follow", 1.0, p_follow), ("fade", -1.0, p_fade)):
                values = sign * clean
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "playbook": playbook_name,
                        "signal": signal,
                        "direction": direction,
                        "months": int(len(values)),
                        "annualized_return": float(values.mean() * periods_per_year),
                        "annualized_volatility": float(values.std(ddof=1) * np.sqrt(periods_per_year)),
                        "sharpe": float(values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year)) if values.std(ddof=1) > 0 else np.nan,
                        "t_stat_hac": _newey_west_t(values, lags=overlap),
                        "hit_rate": float((values > 0).mean()),
                        "first_half_annualized": float(values[:midpoint].mean() * periods_per_year),
                        "second_half_annualized": float(values[midpoint:].mean() * periods_per_year),
                        "rotation_mean_annualized": float(sign * np.nanmean(rotations) * periods_per_year),
                        "p_value": p_value,
                    }
                )
    summary = pd.DataFrame.from_records(summary_rows)
    if not summary.empty:
        summary["q_value"] = summary.groupby(["horizon", "direction"])["p_value"].transform(_benjamini_hochberg)
        summary["stable"] = np.sign(summary["first_half_annualized"]) == np.sign(summary["second_half_annualized"])
        summary["significant"] = summary["q_value"].le(false_discovery_rate) & summary["stable"]
    portfolio_returns = pd.concat(returns_rows, axis=1) if returns_rows else pd.DataFrame()
    return PlaybookTestResult(portfolio_returns=portfolio_returns, summary=summary)


def _newey_west_t(values: np.ndarray, lags: int) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 10:
        return np.nan
    demeaned = values - values.mean()
    variance = np.dot(demeaned, demeaned) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * np.dot(demeaned[lag:], demeaned[:-lag]) / n
    if variance <= 0:
        return np.nan
    return float(values.mean() / np.sqrt(variance / n))
