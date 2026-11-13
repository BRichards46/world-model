from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table
from world_model.features.monthly_windows import (
    fill_missing_months,
    pct_change_months,
    trailing_average,
    value_months_ago,
)
from world_model.states.economic_asset_drivers import evaluation_dates
from world_model.states.macro_market_validation import build_forward_market_targets


@dataclass(frozen=True)
class GTTRiskValidationResult:
    macro_signals: pd.DataFrame
    price_trends: pd.DataFrame
    observations: pd.DataFrame
    summary: pd.DataFrame
    episodes: pd.DataFrame


def build_point_in_time_gtt_macro_signals(fred_root: str | Path) -> pd.DataFrame:
    """Build GTT macro danger flags using only values known on each release date."""
    tables = {
        series_id: load_vintage_table(fred_root, series_id)
        for series_id in ["UNRATE", "INDPRO", "RSAFS", "CPIAUCSL"]
    }
    releases = sorted(set().union(*(set(frame["realtime_start"]) for frame in tables.values())))
    current: dict[str, dict[pd.Timestamp, float]] = {series_id: {} for series_id in tables}
    by_release = {
        series_id: {date: group for date, group in frame.groupby("realtime_start", sort=True)}
        for series_id, frame in tables.items()
    }
    rows: list[dict[str, Any]] = []
    prior: tuple[float, ...] | None = None
    for release in releases:
        for series_id, groups in by_release.items():
            update = groups.get(release)
            if update is None:
                continue
            for record in update.itertuples(index=False):
                if pd.notna(record.value):
                    current[series_id][pd.Timestamp(record.observation_date)] = float(record.value)

        unemployment = _latest_series(current["UNRATE"])
        industrial = _latest_series(current["INDPRO"])
        retail = _latest_series(current["RSAFS"])
        cpi, _ = fill_missing_months(_latest_series(current["CPIAUCSL"]))

        ue_value, ue_average, ue_danger = _unemployment_signal(unemployment)
        ip_growth = _latest_yoy(industrial)
        retail_growth = _latest_real_retail_yoy(retail, cpi)
        ip_danger = float(ip_growth <= 0.0) if pd.notna(ip_growth) else float("nan")
        retail_danger = float(retail_growth <= 0.0) if pd.notna(retail_growth) else float("nan")
        dual_growth = (
            float(bool(ip_danger) or bool(retail_danger))
            if pd.notna(ip_danger) and pd.notna(retail_danger)
            else float("nan")
        )
        available_dangers = [value for value in [ue_danger, ip_danger, retail_danger] if pd.notna(value)]
        any_danger = float(any(bool(value) for value in available_dangers)) if available_dangers else float("nan")
        state = (
            ue_value,
            ue_average,
            ue_danger,
            ip_growth,
            ip_danger,
            retail_growth,
            retail_danger,
            dual_growth,
            any_danger,
        )
        if prior is not None and _states_equal(prior, state):
            continue
        prior = state
        rows.append(
            {
                "realtime_start": pd.Timestamp(release),
                "unemployment_rate": ue_value,
                "unemployment_12m_average": ue_average,
                "unemployment_trend": ue_danger,
                "industrial_production_yoy": ip_growth,
                "industrial_production_growth": ip_danger,
                "real_retail_sales_yoy": retail_growth,
                "real_retail_sales_growth": retail_danger,
                "dual_growth": dual_growth,
                "any_macro_danger": any_danger,
            }
        )
    if not rows:
        raise ValueError("No GTT macro signals could be built from the FRED vintage store.")
    return pd.DataFrame.from_records(rows).set_index("realtime_start").sort_index()


def build_point_in_time_production_signals(fred_root: str | Path) -> pd.DataFrame:
    """Build a predefined set of production alternatives from release vintages."""
    specs = {
        "IPMAN": ("index", False),
        "IPFINAL": ("index", False),
        "IPBUSEQ": ("index", False),
        "IPMAT": ("index", False),
        "AWHMAN": ("level", False),
        "NEWORDER": ("index", True),
        "AMTMNO": ("index", True),
        "DGORDER": ("index", True),
        "CFNAI": ("centered", False),
        "CFNAIMA3": ("centered", False),
        "CUMFNS": ("level", False),
    }
    series_ids = [*specs, "CPIAUCSL"]
    tables = {series_id: load_vintage_table(fred_root, series_id) for series_id in series_ids}
    releases = sorted(set().union(*(set(frame["realtime_start"]) for frame in tables.values())))
    current: dict[str, dict[pd.Timestamp, float]] = {series_id: {} for series_id in tables}
    by_release = {
        series_id: {date: group for date, group in frame.groupby("realtime_start", sort=True)}
        for series_id, frame in tables.items()
    }
    rows: list[dict[str, Any]] = []
    prior: tuple[float, ...] | None = None
    for release in releases:
        for series_id, groups in by_release.items():
            update = groups.get(release)
            if update is None:
                continue
            for record in update.itertuples(index=False):
                date = pd.Timestamp(record.observation_date)
                if pd.notna(record.value):
                    current[series_id][date] = float(record.value)
                else:
                    current[series_id].pop(date, None)

        cpi, _ = fill_missing_months(_latest_series(current["CPIAUCSL"]))
        state: dict[str, float] = {}
        for series_id, (kind, deflate) in specs.items():
            values = _latest_series(current[series_id])
            if deflate:
                common = values.index.intersection(cpi.index)
                values = (values.reindex(common) / cpi.reindex(common)).dropna()
            prefix = series_id.lower()
            if kind == "index":
                state[f"{prefix}__yoy_nonpositive"] = _nonpositive_growth_flag(values, 12)
                state[f"{prefix}__three_month_nonpositive"] = _nonpositive_growth_flag(values, 3)
            elif kind == "level":
                state[f"{prefix}__below_12m_average"] = _below_average_flag(values, 12)
                state[f"{prefix}__three_month_decline"] = _decline_flag(values, 3)
            elif kind == "centered":
                state[f"{prefix}__below_zero"] = (
                    float(values.iloc[-1] < 0.0) if not values.empty else float("nan")
                )
            else:
                raise ValueError(f"Unsupported production signal kind: {kind}")
        values_tuple = tuple(state.values())
        if prior is not None and _states_equal(prior, values_tuple):
            continue
        prior = values_tuple
        rows.append({"realtime_start": pd.Timestamp(release), **state})
    if not rows:
        raise ValueError("No point-in-time production signals could be built.")
    return pd.DataFrame.from_records(rows).set_index("realtime_start").sort_index()


def _nonpositive_growth_flag(series: pd.Series, periods: int) -> float:
    """Calendar-aligned: compares with the same month ``periods`` months earlier."""
    change = pct_change_months(series, periods)
    return float("nan") if change is None else float(change <= 0.0)


def _below_average_flag(series: pd.Series, periods: int) -> float:
    average = trailing_average(series, periods)
    if average is None:
        return float("nan")
    return float(series.dropna().iloc[-1] < average)


def _decline_flag(series: pd.Series, periods: int) -> float:
    prior = value_months_ago(series, periods)
    if prior is None:
        return float("nan")
    return float(series.dropna().iloc[-1] < prior)


def build_price_trends(
    asset_returns: pd.DataFrame,
    ten_month_window: int = 10,
    daily_window: int = 200,
    return_conventions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create group-level price-trend flags from normalized daily return paths."""
    conventions = return_conventions or {}
    columns: dict[str, pd.Series] = {}
    for asset_group in asset_returns.columns:
        returns = pd.to_numeric(asset_returns[asset_group], errors="coerce").dropna()
        if returns.empty:
            continue
        convention = conventions.get(asset_group, "additive")
        if convention == "log":
            synthetic_price = 100.0 * np.exp(returns.cumsum())
        elif convention == "additive":
            synthetic_price = 100.0 + returns.cumsum()
        else:
            raise ValueError(f"Unsupported GTT return convention for {asset_group}: {convention}")
        month_frame = pd.DataFrame({"price": synthetic_price})
        month_frame["period"] = month_frame.index.to_period("M")
        monthly = month_frame.groupby("period", sort=True).tail(1)["price"]
        monthly_ma = monthly.rolling(ten_month_window, min_periods=ten_month_window).mean()
        monthly_down = monthly.lt(monthly_ma).astype(float).where(monthly_ma.notna())
        daily_ma = synthetic_price.rolling(daily_window, min_periods=daily_window).mean()
        daily_down = synthetic_price.lt(daily_ma).astype(float).where(daily_ma.notna())
        columns[f"{asset_group}__ten_month"] = monthly_down.rename(f"{asset_group}__ten_month")
        columns[f"{asset_group}__two_hundred_day"] = daily_down.rename(f"{asset_group}__two_hundred_day")
    return pd.concat(columns.values(), axis=1).sort_index() if columns else pd.DataFrame()


def run_gtt_risk_validation(
    macro_signals: pd.DataFrame,
    price_trends: pd.DataFrame,
    asset_returns: pd.DataFrame,
    config: dict[str, Any],
    asset_labels: dict[str, str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    return_conventions: dict[str, str] | None = None,
) -> GTTRiskValidationResult:
    labels = asset_labels or {}
    conventions = return_conventions or {}
    macro_names = [str(value) for value in config.get("macro_signals", [])]
    horizons = {str(key): int(value) for key, value in config.get("horizons", {}).items()}
    trend_names = list(config.get("price_trends", {}).keys())
    rows: list[dict[str, Any]] = []
    max_horizon = max(horizons.values())

    for asset_group in asset_returns.columns:
        returns = asset_returns[[asset_group]].dropna()
        eval_dates = evaluation_dates(
            returns.index,
            max_horizon=max_horizon,
            frequency=str(config.get("evaluation_frequency", "ME")),
            start=start,
            end=end,
        )
        macro = _asof_reindex(macro_signals, eval_dates)
        if conventions.get(asset_group, "additive") == "log":
            targets_by_horizon = {
                horizon: build_forward_log_risk_targets(returns, days)
                for horizon, days in horizons.items()
            }
        else:
            targets_by_horizon = {
                horizon: build_forward_market_targets(returns, days, ["future_volatility", "max_drawdown"])
                for horizon, days in horizons.items()
            }
        for trend_name in trend_names:
            trend_column = f"{asset_group}__{trend_name}"
            if trend_column not in price_trends:
                continue
            trend = _asof_reindex(price_trends[[trend_column]], eval_dates)[trend_column]
            for macro_name in macro_names:
                if macro_name not in macro:
                    continue
                macro_danger = macro[macro_name]
                double_confirmation = (macro_danger.eq(1.0) & trend.eq(1.0)).astype(float)
                double_confirmation = double_confirmation.where(macro_danger.notna() & trend.notna())
                for horizon, trading_days in horizons.items():
                    target_sets = targets_by_horizon[horizon]
                    for target_name, target_frame in target_sets.items():
                        realized = target_frame[asset_group].reindex(eval_dates)
                        frame = pd.DataFrame(
                            {
                                "forecast_date": eval_dates,
                                "macro_danger": macro_danger.to_numpy(),
                                "price_downtrend": trend.to_numpy(),
                                "double_confirmation": double_confirmation.to_numpy(),
                                "realized_target": realized.to_numpy(),
                            }
                        ).dropna()
                        for record in frame.itertuples(index=False):
                            rows.append(
                                {
                                    "forecast_date": record.forecast_date,
                                    "asset_group": asset_group,
                                    "asset_group_label": labels.get(asset_group, asset_group),
                                    "macro_signal": macro_name,
                                    "price_trend": trend_name,
                                    "horizon": horizon,
                                    "trading_days": trading_days,
                                    "target": target_name,
                                    "macro_danger": record.macro_danger,
                                    "price_downtrend": record.price_downtrend,
                                    "double_confirmation": record.double_confirmation,
                                    "realized_target": record.realized_target,
                                }
                            )
    observations = pd.DataFrame.from_records(rows)
    if observations.empty:
        raise ValueError("GTT risk validation produced no observations.")
    summary = summarize_gtt_risk(
        observations,
        int(config.get("minimum_observations_per_state", 24)),
        float(config.get("minimum_split_lift_fraction", 0.10)),
        int(config.get("bootstrap_iterations", 1000)),
        float(config.get("bootstrap_confidence", 0.90)),
        int(config.get("bootstrap_minimum_block_months", 12)),
    )
    episodes = build_gtt_episode_details(observations)
    return GTTRiskValidationResult(macro_signals, price_trends, observations, summary, episodes)


def build_forward_log_risk_targets(
    log_returns: pd.DataFrame,
    horizon_days: int,
) -> dict[str, pd.DataFrame]:
    """Build annualized volatility and percentage drawdown from daily log returns."""
    volatility = (
        log_returns.shift(-1)
        .rolling(horizon_days, min_periods=horizon_days)
        .std()
        .shift(-(horizon_days - 1))
        * np.sqrt(252.0)
    )
    values = log_returns.to_numpy(dtype=float)
    drawdowns = np.full_like(values, np.nan, dtype=float)
    for position in range(len(log_returns) - horizon_days):
        window = values[position + 1 : position + horizon_days + 1]
        for column in range(values.shape[1]):
            series = window[:, column]
            if np.isnan(series).any():
                continue
            wealth = np.exp(np.concatenate([[0.0], np.cumsum(series)]))
            peaks = np.maximum.accumulate(wealth)
            drawdowns[position, column] = float(np.min(wealth / peaks - 1.0))
    return {
        "future_volatility": volatility,
        "max_drawdown": pd.DataFrame(drawdowns, index=log_returns.index, columns=log_returns.columns),
    }


def summarize_gtt_risk(
    observations: pd.DataFrame,
    minimum_per_state: int = 24,
    minimum_split_lift_fraction: float = 0.10,
    bootstrap_iterations: int = 1000,
    bootstrap_confidence: float = 0.90,
    bootstrap_minimum_block_months: int = 12,
) -> pd.DataFrame:
    keys = ["asset_group", "asset_group_label", "macro_signal", "price_trend", "horizon", "target"]
    rows: list[dict[str, Any]] = []
    for values, group in observations.groupby(keys, sort=False):
        group = group.sort_values("forecast_date").copy()
        # More negative drawdown means more risk; invert it so every lift has
        # the same interpretation: positive means the state predicted more risk.
        risk = -group["realized_target"] if values[-1] == "max_drawdown" else group["realized_target"]
        active = group["double_confirmation"].eq(1.0)
        inactive = group["double_confirmation"].eq(0.0)
        trend_only = group["price_downtrend"].eq(1.0) & group["macro_danger"].eq(0.0)
        midpoint = max(1, len(group) // 2)
        early = group.iloc[:midpoint]
        late = group.iloc[midpoint:]
        severe_cutoff = float(risk.quantile(0.90))
        severe = risk.ge(severe_cutoff)
        severe_base_rate = float(severe.mean())
        block_months = max(bootstrap_minimum_block_months, int(np.ceil(float(group["trading_days"].iloc[0]) / 21.0))) if "trading_days" in group else bootstrap_minimum_block_months
        identity = "|".join(str(value) for value in values)
        bootstrap = block_bootstrap_incremental_lift(
            risk,
            active,
            trend_only,
            block_length=block_months,
            iterations=bootstrap_iterations,
            confidence=bootstrap_confidence,
            seed=zlib.crc32(identity.encode("utf-8")),
        )
        episode_stats = _episode_statistics(group, risk, active, trend_only)
        row = dict(zip(keys, values))
        row.update(
            {
                "observations": len(group),
                "active_observations": int(active.sum()),
                "inactive_observations": int(inactive.sum()),
                "trend_only_observations": int(trend_only.sum()),
                "active_risk_mean": _mean(risk[active]),
                "inactive_risk_mean": _mean(risk[inactive]),
                "risk_lift": _mean(risk[active]) - _mean(risk[inactive]),
                "trend_only_risk_mean": _mean(risk[trend_only]),
                "incremental_lift_vs_trend_only": _mean(risk[active]) - _mean(risk[trend_only]),
                "incremental_lift_ci_low": bootstrap[0],
                "incremental_lift_ci_high": bootstrap[1],
                "bootstrap_p_nonpositive": bootstrap[2],
                "bootstrap_block_months": block_months,
                **episode_stats,
                "severe_risk_cutoff": severe_cutoff,
                "severe_base_rate": severe_base_rate,
                "severe_capture_rate": float(active[severe].mean()) if severe.any() else float("nan"),
                "signal_precision": float(severe[active].mean()) if active.any() else float("nan"),
                "price_downtrend_share": float(group["price_downtrend"].mean()),
                "double_confirmation_share": float(active.mean()),
                "early_risk_lift": _state_lift(early),
                "late_risk_lift": _state_lift(late),
            }
        )
        minimum_split_lift = minimum_split_lift_fraction * abs(row["risk_lift"])
        row["stable_positive_lift"] = bool(
            row["early_risk_lift"] >= minimum_split_lift
            and row["late_risk_lift"] >= minimum_split_lift
        )
        row["research_candidate"] = bool(
            row["active_observations"] >= minimum_per_state
            and row["inactive_observations"] >= minimum_per_state
            and row["trend_only_observations"] >= minimum_per_state
            and row["risk_lift"] > 0
            and row["incremental_lift_vs_trend_only"] > 0
            and row["stable_positive_lift"]
            and row["signal_precision"] > row["severe_base_rate"]
            and row["incremental_lift_ci_low"] > 0.0
            and row["episode_robust"]
            and row["stress_era_robust"]
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(
        ["research_candidate", "incremental_lift_vs_trend_only", "risk_lift"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _latest_series(values: dict[pd.Timestamp, float]) -> pd.Series:
    return pd.Series(values, dtype=float).sort_index()


def _unemployment_signal(series: pd.Series) -> tuple[float, float, float]:
    average = trailing_average(series, 12)
    if average is None:
        return float("nan"), float("nan"), float("nan")
    value = float(series.dropna().iloc[-1])
    return value, average, float(value > average)


def _latest_yoy(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    date = series.index[-1]
    prior_date = date - pd.DateOffset(years=1)
    if prior_date not in series.index or series.loc[prior_date] <= 0 or series.loc[date] <= 0:
        return float("nan")
    return float(100.0 * (series.loc[date] / series.loc[prior_date] - 1.0))


def _latest_real_retail_yoy(retail: pd.Series, cpi: pd.Series) -> float:
    common = retail.index.intersection(cpi.index)
    if common.empty:
        return float("nan")
    real = (retail.reindex(common) / cpi.reindex(common)).dropna()
    return _latest_yoy(real)


def _asof_reindex(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return frame.reindex(frame.index.union(dates)).sort_index().ffill().reindex(dates)


def _states_equal(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all((pd.isna(a) and pd.isna(b)) or a == b for a, b in zip(left, right))


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else float("nan")


def _state_lift(group: pd.DataFrame) -> float:
    risk = -group["realized_target"] if group["target"].iloc[0] == "max_drawdown" else group["realized_target"]
    active = group["double_confirmation"].eq(1.0)
    inactive = group["double_confirmation"].eq(0.0)
    return _mean(risk[active]) - _mean(risk[inactive])


def block_bootstrap_incremental_lift(
    risk: pd.Series,
    active: pd.Series,
    trend_only: pd.Series,
    block_length: int,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    values = risk.to_numpy(dtype=float)
    active_values = active.to_numpy(dtype=bool)
    trend_values = trend_only.to_numpy(dtype=bool)
    count = len(values)
    length = min(max(1, block_length), count)
    starts = np.arange(max(1, count - length + 1))
    blocks_needed = int(np.ceil(count / length))
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate([np.arange(start, start + length) for start in chosen])[:count]
        sampled_active = active_values[indices]
        sampled_trend = trend_values[indices]
        if not sampled_active.any() or not sampled_trend.any():
            continue
        estimates.append(float(values[indices][sampled_active].mean() - values[indices][sampled_trend].mean()))
    if not estimates:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    array = np.asarray(estimates, dtype=float)
    return (
        float(np.quantile(array, alpha)),
        float(np.quantile(array, 1.0 - alpha)),
        float(np.mean(array <= 0.0)),
    )


def build_gtt_episode_details(observations: pd.DataFrame) -> pd.DataFrame:
    keys = ["asset_group", "asset_group_label", "macro_signal", "price_trend", "horizon", "target"]
    rows: list[dict[str, Any]] = []
    for values, group in observations.groupby(keys, sort=False):
        group = group.sort_values("forecast_date").copy()
        risk = -group["realized_target"] if values[-1] == "max_drawdown" else group["realized_target"]
        active = group["double_confirmation"].eq(1.0)
        trend_only = group["price_downtrend"].eq(1.0) & group["macro_danger"].eq(0.0)
        trend_mean = _mean(risk[trend_only])
        eras = _active_run_positions(group, active, gap_days=365)
        era_by_position = {
            position: era_id
            for era_id, era_positions in enumerate(eras, start=1)
            for position in era_positions
        }
        for episode_id, positions in enumerate(_active_run_positions(group, active, gap_days=45), start=1):
            episode_risk = risk.iloc[positions]
            row = dict(zip(keys, values))
            row.update(
                {
                    "episode_id": episode_id,
                    "stress_era_id": era_by_position[positions[0]],
                    "episode_start": group["forecast_date"].iloc[positions[0]],
                    "episode_end": group["forecast_date"].iloc[positions[-1]],
                    "episode_observations": len(positions),
                    "episode_risk_mean": _mean(episode_risk),
                    "trend_only_risk_mean": trend_mean,
                    "episode_excess_vs_trend_only": _mean(episode_risk) - trend_mean,
                    "active_observation_share": len(positions) / max(1, int(active.sum())),
                }
            )
            rows.append(row)
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result["episode_start"] = pd.to_datetime(result["episode_start"])
        result["episode_end"] = pd.to_datetime(result["episode_end"])
    return result


def _episode_statistics(
    group: pd.DataFrame,
    risk: pd.Series,
    active: pd.Series,
    trend_only: pd.Series,
) -> dict[str, Any]:
    episodes = _active_run_positions(group, active, gap_days=45)
    eras = _active_run_positions(group, active, gap_days=365)
    leave_one_out: list[float] = []
    for positions in episodes:
        retained_active = active.copy()
        retained_active.iloc[positions] = False
        if not retained_active.any() or not trend_only.any():
            continue
        leave_one_out.append(_mean(risk[retained_active]) - _mean(risk[trend_only]))
    episode_count = len(episodes)
    max_share = max((len(positions) for positions in episodes), default=0) / max(1, int(active.sum()))
    minimum = min(leave_one_out) if leave_one_out else float("nan")
    median = float(np.median(leave_one_out)) if leave_one_out else float("nan")
    leave_one_era_out = _leave_one_run_out_lifts(risk, active, trend_only, eras)
    era_minimum = min(leave_one_era_out) if leave_one_era_out else float("nan")
    return {
        "active_episode_count": episode_count,
        "largest_episode_active_share": float(max_share),
        "minimum_leave_one_episode_out_lift": minimum,
        "median_leave_one_episode_out_lift": median,
        "episode_robust": bool(episode_count >= 3 and pd.notna(minimum) and minimum > 0.0),
        "active_stress_era_count": len(eras),
        "largest_stress_era_active_share": float(
            max((len(positions) for positions in eras), default=0) / max(1, int(active.sum()))
        ),
        "minimum_leave_one_stress_era_out_lift": era_minimum,
        "stress_era_robust": bool(len(eras) >= 3 and pd.notna(era_minimum) and era_minimum > 0.0),
    }


def _leave_one_run_out_lifts(
    risk: pd.Series,
    active: pd.Series,
    trend_only: pd.Series,
    runs: list[list[int]],
) -> list[float]:
    results: list[float] = []
    for positions in runs:
        retained_active = active.copy()
        retained_active.iloc[positions] = False
        if retained_active.any() and trend_only.any():
            results.append(_mean(risk[retained_active]) - _mean(risk[trend_only]))
    return results


def _active_run_positions(group: pd.DataFrame, active: pd.Series, gap_days: int) -> list[list[int]]:
    active_positions = np.flatnonzero(active.to_numpy(dtype=bool))
    if len(active_positions) == 0:
        return []
    dates = pd.to_datetime(group["forecast_date"]).reset_index(drop=True)
    episodes: list[list[int]] = [[int(active_positions[0])]]
    for position in active_positions[1:]:
        position = int(position)
        prior = episodes[-1][-1]
        if (dates.iloc[position] - dates.iloc[prior]).days > gap_days:
            episodes.append([position])
        else:
            episodes[-1].append(position)
    return episodes


def write_gtt_risk_outputs(result: GTTRiskValidationResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "macro_signals": root / "gtt_macro_signals.parquet",
        "price_trends": root / "gtt_price_trends.parquet",
        "observations": root / "gtt_risk_observations.parquet",
        "summary": root / "gtt_risk_summary.csv",
        "episodes": root / "gtt_risk_episodes.csv",
    }
    result.macro_signals.to_parquet(paths["macro_signals"])
    result.price_trends.to_parquet(paths["price_trends"])
    result.observations.to_parquet(paths["observations"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.episodes.to_csv(paths["episodes"], index=False)
    return paths
