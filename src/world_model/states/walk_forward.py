from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.states.market import (
    COMPONENT_COLUMNS,
    MarketStateEngine,
    _forward_drawdown,
    _forward_return,
    select_diverse_analogs,
    summarize_forward_distribution,
)


@dataclass(frozen=True)
class WalkForwardResult:
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    quintiles: pd.DataFrame
    calibration: pd.DataFrame
    drawdown_calibration: pd.DataFrame


class MarketStateWalkForward:
    def __init__(self, engine: MarketStateEngine):
        self.engine = engine

    def run(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        frequency: str = "M",
        top_n: int | None = None,
        min_analogs: int = 20,
        random_seed: int = 42,
    ) -> WalkForwardResult:
        state = self.engine.build_state_history()
        prices = self.engine.load_prices()
        target = self.engine.config.get("target_asset", "ES")
        target_close = prices[target].reindex(state.index).ffill()

        horizons: dict[str, int] = self.engine.config.get("horizons", {})
        if not horizons:
            raise ValueError("Market config must define at least one horizon.")

        analog_config = self.engine.config.get("analog", {})
        top_n = int(top_n or analog_config.get("top_n", 40))
        min_coverage = float(analog_config.get("min_component_coverage", 0.70))
        max_horizon = int(max(horizons.values()))
        min_gap = int(analog_config.get("min_trading_days_between_analogs", 0))

        forward_returns = {
            label: _forward_return(target_close, days)
            for label, days in horizons.items()
        }
        forward_drawdowns = {
            label: _forward_drawdown(target_close, days)
            for label, days in horizons.items()
        }

        eval_dates = _evaluation_dates(
            state=state,
            target_close=target_close,
            max_horizon=max_horizon,
            start=start,
            end=end,
            frequency=frequency,
        )

        rng = np.random.default_rng(random_seed)
        rows: list[dict[str, Any]] = []

        component_weights = pd.Series(self.engine._component_weights(), dtype=float).reindex(COMPONENT_COLUMNS).fillna(1.0)
        usable_state = state[state["component_coverage"] >= min_coverage]

        for forecast_date in eval_dates:
            current_row = state.loc[forecast_date]
            current_vector = current_row[COMPONENT_COLUMNS].astype(float)
            comparable_cols = current_vector.dropna().index.tolist()
            if len(comparable_cols) < max(3, int(len(COMPONENT_COLUMNS) * min_coverage)):
                continue

            known_cutoff = forecast_date - pd.tseries.offsets.BDay(max_horizon)
            candidates = usable_state.loc[usable_state.index <= known_cutoff]
            vectors = candidates[COMPONENT_COLUMNS].dropna(thresh=max(3, int(len(COMPONENT_COLUMNS) * min_coverage)))
            vectors = vectors.dropna(axis=0, subset=comparable_cols)
            if len(vectors) < min_analogs:
                continue

            weights = component_weights.reindex(comparable_cols).fillna(1.0)
            diffs = vectors[comparable_cols].sub(current_vector[comparable_cols], axis=1)
            distances = np.sqrt((diffs.pow(2).multiply(weights, axis=1)).sum(axis=1) / weights.sum())
            closest = select_diverse_analogs(distances, top_n=top_n, min_trading_days=min_gap)
            if len(closest) < min_analogs:
                continue

            analogs = _build_analogs(
                state=state,
                closest=closest,
                horizons=horizons,
                forward_returns=forward_returns,
                forward_drawdowns=forward_drawdowns,
            )
            model_distribution = summarize_forward_distribution(analogs, horizons)

            random_index = rng.choice(vectors.index.to_numpy(), size=len(closest), replace=False)
            random_closest = pd.Series(1.0, index=pd.DatetimeIndex(random_index))
            random_analogs = _build_analogs(
                state=state,
                closest=random_closest,
                horizons=horizons,
                forward_returns=forward_returns,
                forward_drawdowns=forward_drawdowns,
            )
            random_distribution = summarize_forward_distribution(random_analogs, horizons)

            for horizon_label, days in horizons.items():
                model_row = model_distribution.loc[model_distribution["horizon"] == horizon_label].iloc[0]
                random_row = random_distribution.loc[random_distribution["horizon"] == horizon_label].iloc[0]
                baseline = _baseline_distribution(
                    forward_returns=forward_returns[horizon_label],
                    forward_drawdowns=forward_drawdowns[horizon_label],
                    known_cutoff=forecast_date - pd.tseries.offsets.BDay(days),
                )
                rows.append(
                    {
                        "forecast_date": forecast_date,
                        "horizon": horizon_label,
                        "trading_days": days,
                        "regime_label": current_row["regime_label"],
                        "composite_score": current_row["composite_score"],
                        "analog_count": int(model_row["analog_count"]),
                        "weighted_mean_return": model_row["weighted_mean_return"],
                        "median_return": model_row["median_return"],
                        "p10_return": model_row["p10_return"],
                        "p25_return": model_row["p25_return"],
                        "p75_return": model_row["p75_return"],
                        "p90_return": model_row["p90_return"],
                        "probability_positive": model_row["probability_positive"],
                        "probability_drawdown_gt_5pct": model_row["probability_drawdown_gt_5pct"],
                        "probability_drawdown_gt_10pct": model_row["probability_drawdown_gt_10pct"],
                        "realized_return": forward_returns[horizon_label].get(forecast_date, np.nan),
                        "realized_max_drawdown": forward_drawdowns[horizon_label].get(forecast_date, np.nan),
                        "baseline_mean_return": baseline["mean_return"],
                        "baseline_probability_positive": baseline["probability_positive"],
                        "baseline_probability_drawdown_gt_10pct": baseline["probability_drawdown_gt_10pct"],
                        "random_mean_return": random_row["weighted_mean_return"],
                        "random_probability_positive": random_row["probability_positive"],
                        "random_probability_drawdown_gt_10pct": random_row["probability_drawdown_gt_10pct"],
                    }
                )

        forecasts = pd.DataFrame(rows)
        if forecasts.empty:
            raise ValueError("Walk-forward produced no forecasts. Try an earlier start date, lower min_analogs, or lower frequency.")
        forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
        forecasts = forecasts.sort_values(["trading_days", "forecast_date"]).reset_index(drop=True)

        summary = summarize_walk_forward(forecasts)
        quintiles = summarize_quintiles(forecasts)
        calibration = summarize_probability_calibration(forecasts)
        drawdown_calibration = summarize_drawdown_calibration(forecasts)
        return WalkForwardResult(
            forecasts=forecasts,
            summary=summary,
            quintiles=quintiles,
            calibration=calibration,
            drawdown_calibration=drawdown_calibration,
        )


def summarize_walk_forward(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in _group_by_horizon(forecasts):
        clean = group.dropna(subset=["weighted_mean_return", "realized_return"])
        if clean.empty:
            continue
        model_error = clean["weighted_mean_return"] - clean["realized_return"]
        baseline_error = clean["baseline_mean_return"] - clean["realized_return"]
        random_error = clean["random_mean_return"] - clean["realized_return"]
        rows.append(
            {
                "horizon": horizon,
                "trading_days": int(clean["trading_days"].iloc[0]),
                "observations": int(len(clean)),
                "realized_mean_return": float(clean["realized_return"].mean()),
                "forecast_mean_return": float(clean["weighted_mean_return"].mean()),
                "baseline_mean_return": float(clean["baseline_mean_return"].mean()),
                "random_mean_return": float(clean["random_mean_return"].mean()),
                "top_bottom_realized_spread": _top_bottom_spread(clean),
                "forecast_realized_corr": _safe_corr(clean["weighted_mean_return"], clean["realized_return"]),
                "baseline_realized_corr": _safe_corr(clean["baseline_mean_return"], clean["realized_return"]),
                "random_realized_corr": _safe_corr(clean["random_mean_return"], clean["realized_return"]),
                "model_rmse": float(np.sqrt(np.mean(model_error.pow(2)))),
                "baseline_rmse": float(np.sqrt(np.mean(baseline_error.pow(2)))),
                "random_rmse": float(np.sqrt(np.mean(random_error.pow(2)))),
                "model_direction_hit_rate": float(((clean["weighted_mean_return"] > 0) == (clean["realized_return"] > 0)).mean()),
                "baseline_direction_hit_rate": float(((clean["baseline_mean_return"] > 0) == (clean["realized_return"] > 0)).mean()),
                "random_direction_hit_rate": float(((clean["random_mean_return"] > 0) == (clean["realized_return"] > 0)).mean()),
                "p25_p75_coverage": float(((clean["realized_return"] >= clean["p25_return"]) & (clean["realized_return"] <= clean["p75_return"])).mean()),
                "p10_p90_coverage": float(((clean["realized_return"] >= clean["p10_return"]) & (clean["realized_return"] <= clean["p90_return"])).mean()),
                "drawdown_10pct_brier": _brier_score(clean["probability_drawdown_gt_10pct"], clean["realized_max_drawdown"] <= -0.10),
                "baseline_drawdown_10pct_brier": _brier_score(
                    clean["baseline_probability_drawdown_gt_10pct"],
                    clean["realized_max_drawdown"] <= -0.10,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("trading_days").reset_index(drop=True)


def summarize_quintiles(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in _group_by_horizon(forecasts):
        clean = group.dropna(subset=["weighted_mean_return", "realized_return"]).copy()
        if clean["weighted_mean_return"].nunique() < 5:
            continue
        clean["forecast_quintile"] = pd.qcut(clean["weighted_mean_return"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        for quintile, quintile_group in clean.groupby("forecast_quintile", observed=True):
            rows.append(
                {
                    "horizon": horizon,
                    "trading_days": int(quintile_group["trading_days"].iloc[0]),
                    "forecast_quintile": int(quintile),
                    "observations": int(len(quintile_group)),
                    "mean_forecast_return": float(quintile_group["weighted_mean_return"].mean()),
                    "mean_realized_return": float(quintile_group["realized_return"].mean()),
                    "median_realized_return": float(quintile_group["realized_return"].median()),
                    "positive_rate": float((quintile_group["realized_return"] > 0).mean()),
                    "mean_realized_max_drawdown": float(quintile_group["realized_max_drawdown"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["trading_days", "forecast_quintile"]).reset_index(drop=True)


def summarize_probability_calibration(forecasts: pd.DataFrame) -> pd.DataFrame:
    return _bucket_summary(
        forecasts=forecasts,
        forecast_column="probability_positive",
        observed_column="realized_return",
        observed_transform=lambda series: series > 0,
        bucket_name="probability_positive_bucket",
    )


def summarize_drawdown_calibration(forecasts: pd.DataFrame) -> pd.DataFrame:
    return _bucket_summary(
        forecasts=forecasts,
        forecast_column="probability_drawdown_gt_10pct",
        observed_column="realized_max_drawdown",
        observed_transform=lambda series: series <= -0.10,
        bucket_name="drawdown_10pct_probability_bucket",
    )


def write_walk_forward_outputs(result: WalkForwardResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": root / "market_walk_forward_forecasts.parquet",
        "summary": root / "market_walk_forward_summary.csv",
        "quintiles": root / "market_walk_forward_quintiles.csv",
        "calibration": root / "market_walk_forward_calibration.csv",
        "drawdown_calibration": root / "market_walk_forward_drawdown_calibration.csv",
    }
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    result.quintiles.to_csv(paths["quintiles"], index=False)
    result.calibration.to_csv(paths["calibration"], index=False)
    result.drawdown_calibration.to_csv(paths["drawdown_calibration"], index=False)
    return paths


def _evaluation_dates(
    state: pd.DataFrame,
    target_close: pd.Series,
    max_horizon: int,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    frequency: str,
) -> pd.DatetimeIndex:
    usable = state.dropna(subset=["composite_score"]).copy()
    last_realizable_date = target_close.dropna().index.max() - pd.tseries.offsets.BDay(max_horizon)
    usable = usable.loc[usable.index <= last_realizable_date]
    if start is not None:
        usable = usable.loc[usable.index >= pd.Timestamp(start)]
    if end is not None:
        usable = usable.loc[usable.index <= pd.Timestamp(end)]
    if usable.empty:
        return pd.DatetimeIndex([])
    if frequency.upper() == "M":
        frequency = "ME"
    normalized_frequency = frequency.lower()
    if normalized_frequency in {"b", "d", "daily"}:
        return pd.DatetimeIndex(usable.index)
    sampled = usable.groupby(pd.Grouper(freq=frequency)).tail(1)
    return pd.DatetimeIndex(sampled.index)


def _group_by_horizon(forecasts: pd.DataFrame):
    ordered = forecasts.sort_values(["trading_days", "forecast_date"])
    return ordered.groupby("horizon", sort=False)


def _build_analogs(
    state: pd.DataFrame,
    closest: pd.Series,
    horizons: dict[str, int],
    forward_returns: dict[str, pd.Series],
    forward_drawdowns: dict[str, pd.Series],
) -> pd.DataFrame:
    analogs = pd.DataFrame(index=closest.index)
    analogs["distance"] = closest.astype(float)
    analogs["similarity"] = 1.0 / (1.0 + analogs["distance"])
    analogs["probability_weight"] = analogs["similarity"] / analogs["similarity"].sum()
    analogs["regime_label"] = state.loc[analogs.index, "regime_label"]
    analogs["composite_score"] = state.loc[analogs.index, "composite_score"]
    for horizon_label in horizons:
        analogs[f"{horizon_label}_return"] = forward_returns[horizon_label].reindex(analogs.index)
        analogs[f"{horizon_label}_max_drawdown"] = forward_drawdowns[horizon_label].reindex(analogs.index)
    return analogs


def _baseline_distribution(
    forward_returns: pd.Series,
    forward_drawdowns: pd.Series,
    known_cutoff: pd.Timestamp,
) -> dict[str, float]:
    returns = forward_returns.loc[forward_returns.index <= known_cutoff].dropna()
    drawdowns = forward_drawdowns.loc[forward_drawdowns.index <= known_cutoff].dropna()
    return {
        "mean_return": float(returns.mean()) if not returns.empty else np.nan,
        "probability_positive": float((returns > 0).mean()) if not returns.empty else np.nan,
        "probability_drawdown_gt_10pct": float((drawdowns <= -0.10).mean()) if not drawdowns.empty else np.nan,
    }


def _bucket_summary(
    forecasts: pd.DataFrame,
    forecast_column: str,
    observed_column: str,
    observed_transform,
    bucket_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bins = np.linspace(0.0, 1.0, 6)
    for horizon, group in _group_by_horizon(forecasts):
        clean = group.dropna(subset=[forecast_column, observed_column]).copy()
        if clean.empty:
            continue
        clean[bucket_name] = pd.cut(clean[forecast_column], bins=bins, include_lowest=True)
        observed = observed_transform(clean[observed_column]).astype(float)
        clean["_observed"] = observed
        for bucket, bucket_group in clean.groupby(bucket_name, observed=True):
            rows.append(
                {
                    "horizon": horizon,
                    "trading_days": int(bucket_group["trading_days"].iloc[0]),
                    bucket_name: str(bucket),
                    "observations": int(len(bucket_group)),
                    "mean_forecast_probability": float(bucket_group[forecast_column].mean()),
                    "observed_rate": float(bucket_group["_observed"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["trading_days", bucket_name]).reset_index(drop=True)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    clean = pd.concat([left, right], axis=1).dropna()
    if len(clean) < 3 or clean.iloc[:, 0].nunique() < 2 or clean.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1]))


def _top_bottom_spread(clean: pd.DataFrame) -> float:
    if clean["weighted_mean_return"].nunique() < 5:
        return float("nan")
    bucketed = clean.copy()
    bucketed["_quintile"] = pd.qcut(bucketed["weighted_mean_return"], 5, labels=False, duplicates="drop")
    realized_by_quintile = bucketed.groupby("_quintile")["realized_return"].mean()
    if realized_by_quintile.empty:
        return float("nan")
    return float(realized_by_quintile.iloc[-1] - realized_by_quintile.iloc[0])


def _brier_score(probability: pd.Series, observed: pd.Series) -> float:
    clean = pd.concat([probability.astype(float), observed.astype(float)], axis=1).dropna()
    if clean.empty:
        return float("nan")
    return float((clean.iloc[:, 0] - clean.iloc[:, 1]).pow(2).mean())
