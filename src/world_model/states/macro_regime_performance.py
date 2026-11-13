from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.data.benchmarks import load_equity_benchmark_history
from world_model.states.economic_asset_drivers import load_asset_group_returns
from world_model.states.macro_market_validation import build_macro_feature_panel, individual_asset_config


REGIME_DIMENSIONS = {
    "macro_regime": "Growth momentum × inflation level",
    "growth_state": "Growth level × growth momentum",
    "inflation_state": "Inflation level × inflation momentum",
    "financial_conditions_state": "Financial conditions",
}


@dataclass(frozen=True)
class MacroRegimePerformanceResult:
    regime_tape: pd.DataFrame
    asset_returns: pd.DataFrame
    asset_metadata: pd.DataFrame
    episodes: pd.DataFrame
    performance: pd.DataFrame
    transitions: pd.DataFrame


def build_regime_tape(
    macro_history: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    config: dict[str, Any],
) -> pd.DataFrame:
    panel, _ = build_macro_feature_panel(
        macro_history,
        daily_index=daily_index,
        metrics=["level_percentile_10y", "velocity_percentile_10y"],
        include_context=False,
    )
    tape = pd.DataFrame(index=panel.index)
    dimensions = config.get("dimensions", {})
    for dimension in ("growth", "inflation", "financial_conditions"):
        settings = dimensions.get(dimension, {})
        indicators = settings.get("indicators", {})
        tape[f"{dimension}_level_score"] = _composite_score(
            panel,
            indicators,
            metric="level_percentile_10y",
            minimum_components=int(config.get("minimum_components", 2)),
        )
        tape[f"{dimension}_momentum_score"] = _composite_score(
            panel,
            indicators,
            metric="velocity_percentile_10y",
            minimum_components=int(config.get("minimum_components", 2)),
        )

    lag = int(config.get("availability_lag_trading_days", 1))
    if lag > 0:
        tape = tape.shift(lag)
    low = float(config.get("low_percentile", 40.0))
    high = float(config.get("high_percentile", 60.0))

    tape["growth_level_state"] = _three_state(
        tape["growth_level_score"], low, high, "weak", "neutral", "strong"
    )
    tape["growth_momentum_state"] = _three_state(
        tape["growth_momentum_score"], low, high, "slowing", "stable", "improving"
    )
    tape["inflation_level_state"] = _three_state(
        tape["inflation_level_score"], low, high, "low", "contained", "high"
    )
    tape["inflation_momentum_state"] = _three_state(
        tape["inflation_momentum_score"], low, high, "cooling", "stable", "heating"
    )
    tape["financial_conditions_state"] = _three_state(
        tape["financial_conditions_level_score"], low, high, "easy", "neutral", "tight"
    )
    tape["growth_state"] = _join_states(tape["growth_level_state"], tape["growth_momentum_state"])
    tape["inflation_state"] = _join_states(
        tape["inflation_level_state"], tape["inflation_momentum_state"]
    )
    tape["macro_regime"] = _macro_regime(
        tape["growth_momentum_score"], tape["inflation_level_score"], low, high
    )
    tape.index.name = "date"
    return tape


def load_performance_returns(
    data_roots: dict[str, str],
    asset_config: dict[str, Any],
    performance_config: dict[str, Any] | None = None,
    return_method: str = "pct_change",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = load_asset_group_returns(
        root=data_roots["csi_processed"],
        asset_groups=asset_config.get("asset_groups", {}),
        return_method=return_method,
    )
    individual_config = individual_asset_config(asset_config)
    contracts = load_asset_group_returns(
        root=data_roots["csi_processed"],
        asset_groups=individual_config.get("asset_groups", {}),
        return_method=return_method,
    )
    metadata_rows: list[dict[str, Any]] = []
    for asset_id, settings in asset_config.get("asset_groups", {}).items():
        metadata_rows.append(
            {
                "asset_id": asset_id,
                "asset_label": str(settings.get("label", asset_id)),
                "asset_type": "class",
                "parent_asset_group": asset_id,
            }
        )
    for asset_id, settings in individual_config.get("asset_groups", {}).items():
        metadata_rows.append(
            {
                "asset_id": asset_id,
                "asset_label": str(settings.get("label", asset_id)),
                "asset_type": "contract",
                "parent_asset_group": str(settings.get("parent_asset_group", "")),
            }
        )
    benchmark = (performance_config or {}).get("equity_benchmark", {})
    benchmark_return: pd.Series | None = None
    if benchmark.get("enabled", False):
        asset_id = str(benchmark.get("id", "sp500_total_return"))
        history = load_equity_benchmark_history(data_roots, benchmark)
        benchmark_return = history["log_return"].rename(asset_id)
        metadata_rows.append(
            {
                "asset_id": asset_id,
                "asset_label": str(benchmark.get("label", "S&P 500 total return")),
                "asset_type": "benchmark",
                "parent_asset_group": "equity",
            }
        )
    combined = pd.concat([classes, contracts], axis=1).sort_index()
    if benchmark_return is not None:
        combined = pd.concat([combined, benchmark_return], axis=1).sort_index()
    combined = combined.loc[:, ~combined.columns.duplicated()]
    metadata = pd.DataFrame.from_records(metadata_rows).drop_duplicates("asset_id").set_index("asset_id")
    return combined, metadata


def analyze_macro_regime_performance(
    regime_tape: pd.DataFrame,
    asset_returns: pd.DataFrame,
    asset_metadata: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_index = regime_tape.index.intersection(asset_returns.index)
    tape = regime_tape.reindex(common_index)
    returns = asset_returns.reindex(common_index)
    episode_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    minimum_episode_days = int(config.get("minimum_episode_days", 5))

    for dimension, dimension_label in REGIME_DIMENSIONS.items():
        states = tape[dimension].dropna().astype(str)
        if states.empty:
            continue
        episode_ids = states.ne(states.shift()).cumsum()
        episode_frame = pd.DataFrame({"regime_id": states, "episode_id": episode_ids}, index=states.index)
        ordered_episodes = []
        for episode_id, episode in episode_frame.groupby("episode_id", sort=True):
            regime_id = str(episode["regime_id"].iloc[0])
            ordered_episodes.append((int(episode_id), regime_id, episode.index.min(), episode.index.max()))
        for (_, from_regime, _, _), (_, to_regime, _, _) in zip(ordered_episodes, ordered_episodes[1:]):
            transition_rows.append(
                {
                    "regime_dimension": dimension,
                    "regime_dimension_label": dimension_label,
                    "from_regime": from_regime,
                    "to_regime": to_regime,
                }
            )

        for asset_id in returns.columns:
            meta = asset_metadata.loc[asset_id]
            asset = pd.to_numeric(returns[asset_id], errors="coerce")
            baseline = asset.loc[states.index].dropna()
            baseline_return = _annualized_performance(baseline)
            baseline_volatility = _annualized_volatility(baseline)
            asset_episode_rows: list[dict[str, Any]] = []
            for episode_id, episode in episode_frame.groupby("episode_id", sort=True):
                regime_id = str(episode["regime_id"].iloc[0])
                sample = asset.reindex(episode.index).dropna()
                if sample.empty:
                    continue
                record = {
                    "regime_dimension": dimension,
                    "regime_dimension_label": dimension_label,
                    "regime_id": regime_id,
                    "episode_id": int(episode_id),
                    "asset_id": asset_id,
                    "asset_label": meta["asset_label"],
                    "asset_type": meta["asset_type"],
                    "parent_asset_group": meta["parent_asset_group"],
                    "start_date": sample.index.min(),
                    "end_date": sample.index.max(),
                    "observations": int(len(sample)),
                    "cumulative_performance": _cumulative_performance(sample),
                    "mean_daily_performance": float(sample.mean()),
                    "annualized_volatility": _annualized_volatility(sample),
                    "max_drawdown": _max_drawdown(sample),
                }
                episode_rows.append(record)
                asset_episode_rows.append(record)

            episode_data = pd.DataFrame.from_records(asset_episode_rows)
            if episode_data.empty:
                continue
            for regime_id, group in episode_data.groupby("regime_id", sort=False):
                qualifying = group.loc[group["observations"].ge(minimum_episode_days)]
                dates = episode_frame.index[episode_frame["regime_id"].eq(regime_id)]
                sample = asset.reindex(dates).dropna()
                if sample.empty:
                    continue
                midpoint = sample.index.min() + (sample.index.max() - sample.index.min()) / 2
                early = sample.loc[sample.index <= midpoint]
                late = sample.loc[sample.index > midpoint]
                ann_return = _annualized_performance(sample)
                ann_volatility = _annualized_volatility(sample)
                return_spread = ann_return - baseline_return
                volatility_ratio = (
                    ann_volatility / baseline_volatility
                    if baseline_volatility and pd.notna(baseline_volatility)
                    else float("nan")
                )
                early_spread = _annualized_performance(early) - _annualized_performance(
                    baseline.loc[baseline.index <= midpoint]
                )
                late_spread = _annualized_performance(late) - _annualized_performance(
                    baseline.loc[baseline.index > midpoint]
                )
                early_volatility_ratio = _volatility_ratio(
                    early, baseline.loc[baseline.index <= midpoint]
                )
                late_volatility_ratio = _volatility_ratio(
                    late, baseline.loc[baseline.index > midpoint]
                )
                episode_spreads = (
                    qualifying["mean_daily_performance"] - baseline.mean()
                ) * 252.0
                spread_ci_low, spread_ci_high, spread_p_value = _bootstrap_mean_test(
                    episode_spreads,
                    repetitions=int(config.get("bootstrap_repetitions", 2000)),
                )
                stable_performance_sign = bool(
                    pd.notna(early_spread)
                    and pd.notna(late_spread)
                    and np.sign(early_spread) == np.sign(late_spread)
                )
                stable_volatility_direction = bool(
                    pd.notna(early_volatility_ratio)
                    and pd.notna(late_volatility_ratio)
                    and (
                        (early_volatility_ratio > 1.0 and late_volatility_ratio > 1.0)
                        or (early_volatility_ratio < 1.0 and late_volatility_ratio < 1.0)
                    )
                )
                meaningful_low = float(config.get("meaningful_volatility_ratio_low", 0.85))
                meaningful_high = float(config.get("meaningful_volatility_ratio_high", 1.15))
                performance_rows.append(
                    {
                        "regime_dimension": dimension,
                        "regime_dimension_label": dimension_label,
                        "regime_id": regime_id,
                        "asset_id": asset_id,
                        "asset_label": meta["asset_label"],
                        "asset_type": meta["asset_type"],
                        "parent_asset_group": meta["parent_asset_group"],
                        "observations": int(len(sample)),
                        "episodes": int(len(qualifying)),
                        "average_episode_days": float(qualifying["observations"].mean()),
                        "median_episode_days": float(qualifying["observations"].median()),
                        "annualized_performance": ann_return,
                        "annualized_volatility": ann_volatility,
                        "performance_to_volatility": ann_return / ann_volatility if ann_volatility else float("nan"),
                        "positive_day_rate": float(sample.gt(0.0).mean()),
                        "average_episode_performance": float(qualifying["cumulative_performance"].mean()),
                        "median_episode_performance": float(qualifying["cumulative_performance"].median()),
                        "positive_episode_rate": float(qualifying["cumulative_performance"].gt(0.0).mean()),
                        "worst_episode_drawdown": float(qualifying["max_drawdown"].min()),
                        "baseline_annualized_performance": baseline_return,
                        "baseline_annualized_volatility": baseline_volatility,
                        "performance_spread_vs_baseline": return_spread,
                        "episode_performance_spread_ci_low": spread_ci_low,
                        "episode_performance_spread_ci_high": spread_ci_high,
                        "episode_performance_p_value": spread_p_value,
                        "volatility_ratio_vs_baseline": volatility_ratio,
                        "early_performance_spread_vs_baseline": early_spread,
                        "late_performance_spread_vs_baseline": late_spread,
                        "performance_spread_stable_sign": stable_performance_sign,
                        "early_volatility_ratio_vs_baseline": early_volatility_ratio,
                        "late_volatility_ratio_vs_baseline": late_volatility_ratio,
                        "volatility_direction_stable": stable_volatility_direction,
                        "repeatable_volatility_shift": bool(
                            stable_volatility_direction
                            and (
                                volatility_ratio <= meaningful_low
                                or volatility_ratio >= meaningful_high
                            )
                        ),
                        "sample_start": sample.index.min(),
                        "sample_end": sample.index.max(),
                        "is_current_regime": bool(regime_id == str(states.iloc[-1])),
                    }
                )

    episodes = pd.DataFrame.from_records(episode_rows)
    performance = pd.DataFrame.from_records(performance_rows)
    if not performance.empty:
        performance["episode_performance_q_value"] = performance.groupby(
            ["regime_dimension", "asset_type"], dropna=False
        )["episode_performance_p_value"].transform(_benjamini_hochberg)
        performance["repeatable_performance_shift"] = (
            performance["performance_spread_stable_sign"]
            & performance["episode_performance_q_value"].le(
                float(config.get("false_discovery_rate", 0.10))
            )
        )
    transitions = _summarize_transitions(pd.DataFrame.from_records(transition_rows))
    return episodes, performance, transitions


def run_macro_regime_performance(
    macro_history: pd.DataFrame,
    data_roots: dict[str, str],
    asset_config: dict[str, Any],
    config: dict[str, Any],
) -> MacroRegimePerformanceResult:
    returns, metadata = load_performance_returns(
        data_roots,
        asset_config,
        performance_config=config,
        return_method=str(config.get("return_method", "pct_change")),
    )
    tape = build_regime_tape(macro_history, returns.index, config)
    episodes, performance, transitions = analyze_macro_regime_performance(
        tape, returns, metadata, config
    )
    return MacroRegimePerformanceResult(
        regime_tape=tape,
        asset_returns=returns,
        asset_metadata=metadata,
        episodes=episodes,
        performance=performance,
        transitions=transitions,
    )


def write_macro_regime_performance_outputs(
    result: MacroRegimePerformanceResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "tape": root / "macro_regime_tape.parquet",
        "returns": root / "macro_regime_asset_returns.parquet",
        "metadata": root / "macro_regime_asset_metadata.csv",
        "episodes": root / "macro_regime_episodes.csv",
        "performance": root / "macro_regime_performance.csv",
        "transitions": root / "macro_regime_transitions.csv",
    }
    result.regime_tape.to_parquet(paths["tape"])
    result.asset_returns.to_parquet(paths["returns"])
    result.asset_metadata.reset_index().to_csv(paths["metadata"], index=False)
    result.episodes.to_csv(paths["episodes"], index=False)
    result.performance.to_csv(paths["performance"], index=False)
    result.transitions.to_csv(paths["transitions"], index=False)
    return paths


def _composite_score(
    panel: pd.DataFrame,
    indicators: dict[str, Any],
    metric: str,
    minimum_components: int,
) -> pd.Series:
    columns: list[pd.Series] = []
    for indicator_id, direction in indicators.items():
        column = f"{indicator_id}__{metric}"
        if column not in panel:
            continue
        values = pd.to_numeric(panel[column], errors="coerce")
        aligned = values if float(direction) >= 0.0 else 100.0 - values
        columns.append(aligned.rename(str(indicator_id)))
    if not columns:
        return pd.Series(np.nan, index=panel.index, dtype=float)
    frame = pd.concat(columns, axis=1)
    return frame.median(axis=1, skipna=True).where(frame.notna().sum(axis=1).ge(minimum_components))


def _three_state(
    values: pd.Series,
    low: float,
    high: float,
    low_label: str,
    middle_label: str,
    high_label: str,
) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    result.loc[values.lt(low)] = low_label
    result.loc[values.ge(low) & values.lt(high)] = middle_label
    result.loc[values.ge(high)] = high_label
    return result


def _join_states(left: pd.Series, right: pd.Series) -> pd.Series:
    valid = left.notna() & right.notna()
    result = pd.Series(pd.NA, index=left.index, dtype="string")
    result.loc[valid] = left.loc[valid].astype(str) + "__" + right.loc[valid].astype(str)
    return result


def _macro_regime(
    growth_momentum: pd.Series,
    inflation_level: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    valid = growth_momentum.notna() & inflation_level.notna()
    result = pd.Series(pd.NA, index=growth_momentum.index, dtype="string")
    slowing = growth_momentum.lt(low_threshold)
    stable = growth_momentum.ge(low_threshold) & growth_momentum.lt(high_threshold)
    improving = growth_momentum.ge(high_threshold)
    inflation_high = inflation_level.ge(high_threshold)
    result.loc[valid & improving & ~inflation_high] = "goldilocks_recovery"
    result.loc[valid & improving & inflation_high] = "reflation"
    result.loc[valid & slowing & ~inflation_high] = "disinflationary_slowdown"
    result.loc[valid & slowing & inflation_high] = "stagflation_pressure"
    result.loc[valid & stable & ~inflation_high] = "balanced_transition"
    result.loc[valid & stable & inflation_high] = "high_inflation_plateau"
    return result


def _summarize_transitions(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    counts = (
        rows.groupby(
            ["regime_dimension", "regime_dimension_label", "from_regime", "to_regime"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "transitions"})
    )
    totals = counts.groupby(["regime_dimension", "from_regime"])["transitions"].transform("sum")
    counts["transition_probability"] = counts["transitions"] / totals
    return counts.sort_values(
        ["regime_dimension", "from_regime", "transition_probability"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _annualized_performance(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean() * 252.0)


def _annualized_volatility(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    return float(clean.std(ddof=1) * np.sqrt(252.0)) if len(clean) >= 2 else float("nan")


def _volatility_ratio(sample: pd.Series, baseline: pd.Series) -> float:
    sample_volatility = _annualized_volatility(sample)
    baseline_volatility = _annualized_volatility(baseline)
    if not baseline_volatility or pd.isna(baseline_volatility):
        return float("nan")
    return sample_volatility / baseline_volatility


def _bootstrap_mean_test(
    values: pd.Series,
    repetitions: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 5 or repetitions <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(1729)
    samples = rng.choice(clean, size=(repetitions, len(clean)), replace=True).mean(axis=1)
    centered = clean - clean.mean()
    null_samples = rng.choice(centered, size=(repetitions, len(clean)), replace=True).mean(axis=1)
    p_value = (np.count_nonzero(np.abs(null_samples) >= abs(clean.mean())) + 1.0) / (
        repetitions + 1.0
    )
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
        float(p_value),
    )


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    clean = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if clean.empty:
        return result
    count = len(clean)
    ranks = np.arange(1, count + 1, dtype=float)
    adjusted = (clean.to_numpy(dtype=float) * count / ranks)[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[clean.index] = np.minimum(adjusted, 1.0)
    return result


def _cumulative_performance(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    return float(clean.sum()) if not clean.empty else float("nan")


def _max_drawdown(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    wealth = clean.cumsum()
    drawdown = wealth - wealth.cummax()
    return float(drawdown.min())
