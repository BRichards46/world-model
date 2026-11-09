"""Forward asset performance conditional on point-in-time growth/inflation quads.

Each month-end signal date uses only FRED vintages known on that date (via the
quad walk-forward). Forward returns start ``availability_lag_trading_days``
after the signal date's last trading session. Significance uses a circular
rotation test: the quad label sequence is rotated against the return sequence,
which preserves the autocorrelation of both and the overlap of multi-month
forward returns. P-values are adjusted with Benjamini-Hochberg within each
signal and horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.data.norgate import read_catalog_price_series, read_norgate_index_series
from world_model.states.economic_asset_drivers import load_asset_group_returns


QUADS = (1, 2, 3, 4)


@dataclass(frozen=True)
class QuadAssetPerformanceResult:
    as_of_date: pd.Timestamp
    signals: pd.DataFrame
    forward_returns: pd.DataFrame
    quad_summary: pd.DataFrame
    joint_summary: pd.DataFrame
    asset_metadata: pd.DataFrame


def build_quad_signals(walk_forward: pd.DataFrame) -> pd.DataFrame:
    """Pivot the quad walk-forward into one row of signals per month-end."""
    frame = walk_forward.copy()
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    dates = pd.DatetimeIndex(sorted(frame["forecast_date"].unique()), name="signal_date")
    signals = pd.DataFrame(index=dates)
    signals["realized_last"] = frame.groupby("forecast_date")["persistence_quad"].first()
    current = frame[frame["horizon"].eq(0)].drop_duplicates("forecast_date").set_index("forecast_date")
    following = frame[frame["horizon"].eq(1)].drop_duplicates("forecast_date").set_index("forecast_date")
    signals["projected_current"] = current["projected_quad"]
    signals["hindsight_current"] = current["realized_quad"]
    signals["projected_next"] = following["projected_quad"]
    return signals.apply(pd.to_numeric, errors="coerce")


def load_cumulative_assets(
    data_roots: Mapping[str, Any],
    config: Mapping[str, Any],
    csi_groups_config: Mapping[str, Any],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Cumulative performance series (×100) per asset; forward returns are differences."""
    cumulative: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for asset_id, settings in config.get("norgate_assets", {}).items():
        symbol = str(settings["symbol"])
        if settings.get("dataset", "us_indices") == "equities":
            price = read_catalog_price_series(data_roots["norgate_prices"], data_roots["norgate_metadata"], symbol)
        else:
            price = read_norgate_index_series(data_roots["norgate_metadata"], symbol)
        price = price[price > 0.0]
        cumulative[asset_id] = 100.0 * np.log(price)
        rows.append(
            {
                "asset_id": asset_id,
                "asset_label": str(settings.get("label", asset_id)),
                "asset_group": str(settings.get("group", "")),
                "source": f"Norgate {symbol}",
                "units": "log total return, %",
            }
        )
    groups = {name: csi_groups_config[name] for name in config.get("csi_asset_groups", []) if name in csi_groups_config}
    if groups:
        returns = load_asset_group_returns(data_roots["csi_processed"], groups, return_method="normalized_price_change")
        for group_name in returns.columns:
            daily = returns[group_name]
            first = daily.first_valid_index()
            series = 100.0 * daily.fillna(0.0).cumsum()
            cumulative[f"futures_{group_name}"] = series.loc[first:] if first is not None else series.iloc[0:0]
            rows.append(
                {
                    "asset_id": f"futures_{group_name}",
                    "asset_label": f"{groups[group_name].get('label', group_name)}",
                    "asset_group": "futures",
                    "source": "CSI continuous futures",
                    "units": "normalized price change, % of price scale",
                }
            )
    return cumulative, pd.DataFrame.from_records(rows).set_index("asset_id")


def forward_returns_at(
    signal_dates: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    cumulative: Mapping[str, pd.Series],
    horizon_days: int,
    lag_days: int,
) -> pd.DataFrame:
    """Forward performance from ``lag`` to ``lag + horizon`` sessions after each signal date."""
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    positions = calendar.searchsorted(signal_dates, side="right") - 1
    starts, ends = [], []
    for position in positions:
        start, end = position + lag_days, position + lag_days + horizon_days
        valid = position >= 0 and end < len(calendar)
        starts.append(calendar[start] if valid else pd.NaT)
        ends.append(calendar[end] if valid else pd.NaT)
    starts = pd.DatetimeIndex(starts)
    ends = pd.DatetimeIndex(ends)
    result = pd.DataFrame(index=signal_dates)
    for asset_id, series in cumulative.items():
        clean = series.dropna().sort_index()
        if clean.empty:
            result[asset_id] = np.nan
            continue
        first, last = clean.index[0], clean.index[-1]
        values = []
        for start, end in zip(starts, ends):
            if pd.isna(start) or start < first or end > last:
                values.append(np.nan)
                continue
            values.append(float(clean.asof(end) - clean.asof(start)))
        result[asset_id] = values
    return result


def add_excess_returns(forward: pd.DataFrame, metadata: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = str(config.get("excess_benchmark", "sp500"))
    if benchmark not in forward:
        return forward, metadata
    groups = set(config.get("excess_groups", []))
    extra = set(config.get("excess_assets", []))
    forward = forward.copy()
    rows = []
    for asset_id, meta in metadata.iterrows():
        if asset_id == benchmark or not (meta["asset_group"] in groups or asset_id in extra):
            continue
        excess_id = f"{asset_id}_vs_{benchmark}"
        forward[excess_id] = forward[asset_id] - forward[benchmark]
        rows.append(
            {
                "asset_id": excess_id,
                "asset_label": f"{meta['asset_label']} vs {metadata.loc[benchmark, 'asset_label']}",
                "asset_group": "excess",
                "source": meta["source"],
                "units": "excess log total return, %",
            }
        )
    return forward, pd.concat([metadata, pd.DataFrame.from_records(rows).set_index("asset_id")])


def _rotations(n: int, gap: int) -> list[int]:
    return list(range(gap, n - gap + 1)) if n > 2 * gap else []


def _quad_spreads(labels: np.ndarray, values: np.ndarray) -> dict[int, float]:
    overall = values.mean()
    return {quad: (values[labels == quad].mean() - overall) if np.any(labels == quad) else np.nan for quad in QUADS}


def _between_quad_dispersion(labels: np.ndarray, values: np.ndarray) -> float:
    overall = values.mean()
    total = 0.0
    for quad in QUADS:
        mask = labels == quad
        if mask.any():
            total += mask.sum() * (values[mask].mean() - overall) ** 2
    return total / len(values)


def analyze(
    signals: pd.DataFrame,
    forward_by_horizon: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    horizon_days: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap = int(config.get("minimum_rotation_gap_months", 12))
    minimum = int(config.get("minimum_months_per_quad", 24))
    signal_meta = config.get("signals", {})
    exclusions = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in config.get("robustness_exclusions", [])]
    current = signals.ffill().iloc[-1] if not signals.empty else pd.Series(dtype=float)
    quad_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for horizon, forward in forward_by_horizon.items():
        periods_per_year = 252.0 / float(horizon_days[horizon])
        for signal in signals.columns:
            for asset_id in forward.columns:
                frame = pd.concat([signals[signal].rename("quad"), forward[asset_id].rename("value")], axis=1).dropna()
                if len(frame) < 4 * minimum // 2:
                    continue
                labels = frame["quad"].to_numpy(dtype=int)
                values = frame["value"].to_numpy(dtype=float)
                spreads = _quad_spreads(labels, values)
                dispersion = _between_quad_dispersion(labels, values)
                rotated_spreads = {quad: [] for quad in QUADS}
                rotated_dispersion = []
                for shift in _rotations(len(labels), gap):
                    rolled = np.roll(labels, shift)
                    for quad, value in _quad_spreads(rolled, values).items():
                        rotated_spreads[quad].append(value)
                    rotated_dispersion.append(_between_quad_dispersion(rolled, values))
                excluded = np.zeros(len(frame), dtype=bool)
                for start, end in exclusions:
                    excluded |= (frame.index >= start) & (frame.index <= end)
                kept_labels, kept_values = labels[~excluded], values[~excluded]
                midpoint = frame.index[len(frame) // 2]
                early, late = frame.loc[frame.index < midpoint], frame.loc[frame.index >= midpoint]
                meta = metadata.loc[asset_id]
                base = {
                    "horizon": horizon,
                    "signal": signal,
                    "signal_label": str(signal_meta.get(signal, {}).get("label", signal)),
                    "asset_id": asset_id,
                    "asset_label": meta["asset_label"],
                    "asset_group": meta["asset_group"],
                    "units": meta["units"],
                    "months": int(len(frame)),
                    "sample_start": frame.index.min(),
                    "sample_end": frame.index.max(),
                }
                joint_rows.append(
                    {
                        **base,
                        "between_quad_dispersion": dispersion,
                        "p_value": _rotation_p(dispersion, rotated_dispersion, two_sided=False),
                        "rotations": len(rotated_dispersion),
                    }
                )
                for quad in QUADS:
                    mask = labels == quad
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    early_spread = _half_spread(early, quad)
                    late_spread = _half_spread(late, quad)
                    quad_rows.append(
                        {
                            **base,
                            "quad": quad,
                            "quad_months": count,
                            "quad_share": count / len(labels),
                            "mean_forward": float(values[mask].mean()),
                            "overall_mean_forward": float(values.mean()),
                            "annualized_mean": float(values[mask].mean() * periods_per_year),
                            "annualized_spread": float(spreads[quad] * periods_per_year),
                            "median_annualized_spread": float((np.median(values[mask]) - np.median(values)) * periods_per_year),
                            "ex_crisis_annualized_spread": float(
                                (kept_values[kept_labels == quad].mean() - kept_values.mean()) * periods_per_year
                            )
                            if np.any(kept_labels == quad)
                            else np.nan,
                            "positive_rate": float((values[mask] > 0).mean()),
                            "overall_positive_rate": float((values > 0).mean()),
                            "early_annualized_spread": early_spread * periods_per_year if early_spread is not None else np.nan,
                            "late_annualized_spread": late_spread * periods_per_year if late_spread is not None else np.nan,
                            "p_value": _rotation_p(spreads[quad], rotated_spreads[quad], two_sided=True) if count >= minimum else np.nan,
                            "is_current_signal": bool(signal in current and current[signal] == quad),
                        }
                    )
    quad_summary = pd.DataFrame.from_records(quad_rows)
    joint_summary = pd.DataFrame.from_records(joint_rows)
    alpha = float(config.get("false_discovery_rate", 0.10))
    if not quad_summary.empty:
        quad_summary["q_value"] = quad_summary.groupby(["horizon", "signal"])["p_value"].transform(_benjamini_hochberg)
        quad_summary["stable_sign"] = (
            np.sign(quad_summary["early_annualized_spread"]) == np.sign(quad_summary["late_annualized_spread"])
        ) & (np.sign(quad_summary["early_annualized_spread"]) == np.sign(quad_summary["annualized_spread"]))
        quad_summary["robust"] = quad_summary["q_value"].le(alpha) & quad_summary["stable_sign"]
        consistent = (
            np.sign(quad_summary["median_annualized_spread"]) == np.sign(quad_summary["annualized_spread"])
        ) & (np.sign(quad_summary["ex_crisis_annualized_spread"]) == np.sign(quad_summary["annualized_spread"]))
        suggestive_q = float(config.get("suggestive_q_value", 0.25))
        quad_summary["evidence"] = np.select(
            [
                quad_summary["robust"] & consistent,
                quad_summary["q_value"].le(suggestive_q) & quad_summary["stable_sign"] & consistent,
                quad_summary["p_value"].le(0.05),
            ],
            ["robust", "suggestive", "nominal only"],
            default="none",
        )
    if not joint_summary.empty:
        joint_summary["q_value"] = joint_summary.groupby(["horizon", "signal"])["p_value"].transform(_benjamini_hochberg)
        joint_summary["significant"] = joint_summary["q_value"].le(alpha)
    return quad_summary, joint_summary


def _half_spread(frame: pd.DataFrame, quad: int) -> float | None:
    if frame.empty or not frame["quad"].eq(quad).any():
        return None
    return float(frame.loc[frame["quad"].eq(quad), "value"].mean() - frame["value"].mean())


def _rotation_p(observed: float, rotated: list[float], two_sided: bool) -> float:
    clean = np.asarray([value for value in rotated if np.isfinite(value)], dtype=float)
    if clean.size == 0 or not np.isfinite(observed):
        return np.nan
    if two_sided:
        exceed = np.count_nonzero(np.abs(clean) >= abs(observed))
    else:
        exceed = np.count_nonzero(clean >= observed)
    return float((exceed + 1.0) / (clean.size + 1.0))


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    clean = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if clean.empty:
        return result
    count = len(clean)
    adjusted = (clean.to_numpy(dtype=float) * count / np.arange(1, count + 1, dtype=float))[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[clean.index] = np.minimum(adjusted, 1.0)
    return result


def run_quad_asset_performance(
    walk_forward: pd.DataFrame,
    data_roots: Mapping[str, Any],
    config: Mapping[str, Any],
    csi_groups_config: Mapping[str, Any],
    as_of: pd.Timestamp,
) -> QuadAssetPerformanceResult:
    signals = build_quad_signals(walk_forward)
    cumulative, metadata = load_cumulative_assets(data_roots, config, csi_groups_config)
    benchmark = str(config.get("excess_benchmark", "sp500"))
    calendar = cumulative[benchmark].dropna().index
    lag = int(config.get("availability_lag_trading_days", 1))
    horizons = {str(name): int(days) for name, days in config.get("horizons", {}).items()}
    forward_by_horizon: dict[str, pd.DataFrame] = {}
    for name, days in horizons.items():
        forward = forward_returns_at(signals.index, calendar, cumulative, days, lag)
        forward, horizon_metadata = add_excess_returns(forward, metadata, config)
        forward_by_horizon[name] = forward
    metadata = horizon_metadata
    quad_summary, joint_summary = analyze(signals, forward_by_horizon, metadata, config, horizons)
    forward_long = pd.concat(
        {name: frame for name, frame in forward_by_horizon.items()}, names=["horizon", "signal_date"]
    ).reset_index()
    return QuadAssetPerformanceResult(
        as_of_date=as_of,
        signals=signals,
        forward_returns=forward_long,
        quad_summary=quad_summary,
        joint_summary=joint_summary,
        asset_metadata=metadata,
    )


def write_outputs(result: QuadAssetPerformanceResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "signals": root / "quad_signals.parquet",
        "forward_returns": root / "quad_forward_returns.parquet",
        "quad_summary": root / "quad_asset_summary.csv",
        "joint_summary": root / "quad_joint_tests.csv",
    }
    result.signals.to_parquet(paths["signals"])
    result.forward_returns.to_parquet(paths["forward_returns"])
    result.quad_summary.to_csv(paths["quad_summary"], index=False)
    result.joint_summary.to_csv(paths["joint_summary"], index=False)
    return paths
