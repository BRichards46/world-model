from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.states.market import COMPONENT_COLUMNS, MarketStateEngine, _forward_drawdown, _forward_return
from world_model.states.walk_forward import (
    _baseline_distribution,
    _evaluation_dates,
    summarize_drawdown_calibration,
    summarize_probability_calibration,
    summarize_quintiles,
    summarize_walk_forward,
)


@dataclass(frozen=True)
class ForecastRankerResult:
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    quintiles: pd.DataFrame
    calibration: pd.DataFrame
    drawdown_calibration: pd.DataFrame
    component_diagnostics: pd.DataFrame
    learned_weights: pd.DataFrame


class MarketForecastRanker:
    def __init__(self, engine: MarketStateEngine):
        self.engine = engine

    def run(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        frequency: str = "ME",
        min_train_observations: int = 756,
        min_abs_rank_ic: float = 0.02,
        max_components: int = 5,
        train_window_years: int | None = None,
    ) -> ForecastRankerResult:
        state = self.engine.build_state_history()
        prices = self.engine.load_prices()
        target = self.engine.config.get("target_asset", "ES")
        target_close = prices[target].reindex(state.index).ffill()
        horizons: dict[str, int] = self.engine.config.get("horizons", {})
        if not horizons:
            raise ValueError("Market config must define at least one horizon.")

        forward_returns = {
            label: _forward_return(target_close, days)
            for label, days in horizons.items()
        }
        forward_drawdowns = {
            label: _forward_drawdown(target_close, days)
            for label, days in horizons.items()
        }
        max_horizon = int(max(horizons.values()))
        eval_dates = _evaluation_dates(
            state=state,
            target_close=target_close,
            max_horizon=max_horizon,
            start=start,
            end=end,
            frequency=frequency,
        )

        forecast_rows: list[dict[str, Any]] = []
        weight_rows: list[dict[str, Any]] = []
        for forecast_date in eval_dates:
            current_features = state.loc[forecast_date, COMPONENT_COLUMNS].astype(float)
            if current_features.dropna().empty:
                continue

            for horizon_label, days in horizons.items():
                train_cutoff = forecast_date - pd.tseries.offsets.BDay(days)
                train = pd.concat(
                    [
                        state[COMPONENT_COLUMNS],
                        forward_returns[horizon_label].rename("forward_return"),
                        forward_drawdowns[horizon_label].rename("forward_drawdown"),
                    ],
                    axis=1,
                ).loc[:train_cutoff]
                if train_window_years is not None:
                    train_start = train_cutoff - pd.DateOffset(years=train_window_years)
                    train = train.loc[train.index >= train_start]
                train = train.dropna(subset=["forward_return"])
                if len(train) < min_train_observations:
                    continue

                model = fit_component_rank_model(
                    train=train,
                    min_abs_rank_ic=min_abs_rank_ic,
                    max_components=max_components,
                )
                if model.empty:
                    continue

                training_scores = score_components(train[COMPONENT_COLUMNS], model)
                current_score = score_components(current_features.to_frame().T, model).iloc[0]
                if pd.isna(current_score):
                    continue

                conditional = conditional_distribution(
                    training_scores=training_scores,
                    training_returns=train["forward_return"],
                    training_drawdowns=train["forward_drawdown"],
                    current_score=float(current_score),
                )
                forecast_percentile = percentile_of_value(training_scores, current_score)
                baseline = _baseline_distribution(
                    forward_returns=forward_returns[horizon_label],
                    forward_drawdowns=forward_drawdowns[horizon_label],
                    known_cutoff=train_cutoff,
                )
                realized_return = forward_returns[horizon_label].get(forecast_date, np.nan)
                realized_drawdown = forward_drawdowns[horizon_label].get(forecast_date, np.nan)

                forecast_rows.append(
                    {
                        "forecast_date": forecast_date,
                        "horizon": horizon_label,
                        "trading_days": days,
                        "forecast_score": float(current_score),
                        "forecast_percentile": forecast_percentile,
                        "weighted_mean_return": conditional["mean_return"],
                        "median_return": conditional["median_return"],
                        "p10_return": conditional["p10_return"],
                        "p25_return": conditional["p25_return"],
                        "p75_return": conditional["p75_return"],
                        "p90_return": conditional["p90_return"],
                        "probability_positive": conditional["probability_positive"],
                        "probability_drawdown_gt_5pct": conditional["probability_drawdown_gt_5pct"],
                        "probability_drawdown_gt_10pct": conditional["probability_drawdown_gt_10pct"],
                        "realized_return": realized_return,
                        "realized_max_drawdown": realized_drawdown,
                        "baseline_mean_return": baseline["mean_return"],
                        "baseline_probability_positive": baseline["probability_positive"],
                        "baseline_probability_drawdown_gt_10pct": baseline["probability_drawdown_gt_10pct"],
                        "random_mean_return": baseline["mean_return"],
                        "random_probability_positive": baseline["probability_positive"],
                        "random_probability_drawdown_gt_10pct": baseline["probability_drawdown_gt_10pct"],
                        "component_count": int(len(model)),
                        "train_observations": int(len(train)),
                    }
                )

                for component, row in model.iterrows():
                    weight_rows.append(
                        {
                            "forecast_date": forecast_date,
                            "horizon": horizon_label,
                            "trading_days": days,
                            "component": component,
                            "weight": row["weight"],
                            "rank_ic": row["rank_ic"],
                            "quintile_spread": row["quintile_spread"],
                            "train_observations": int(len(train)),
                        }
                    )

        forecasts = pd.DataFrame(forecast_rows)
        if forecasts.empty:
            raise ValueError("Forecast ranker produced no forecasts.")
        forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
        forecasts = forecasts.sort_values(["trading_days", "forecast_date"]).reset_index(drop=True)

        learned_weights = pd.DataFrame(weight_rows)
        component_diagnostics = component_forecast_diagnostics(
            state=state,
            forward_returns=forward_returns,
            horizons=horizons,
        )
        return ForecastRankerResult(
            forecasts=forecasts,
            summary=summarize_walk_forward(forecasts),
            quintiles=summarize_quintiles(forecasts),
            calibration=summarize_probability_calibration(forecasts),
            drawdown_calibration=summarize_drawdown_calibration(forecasts),
            component_diagnostics=component_diagnostics,
            learned_weights=learned_weights,
        )


def component_forecast_diagnostics(
    state: pd.DataFrame,
    forward_returns: dict[str, pd.Series],
    horizons: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon_label, returns in forward_returns.items():
        days = horizons[horizon_label]
        for component in COMPONENT_COLUMNS + ["composite_score"]:
            sample = pd.concat(
                [state[component].rename("component_score"), returns.rename("forward_return")],
                axis=1,
            ).dropna()
            if len(sample) < 100 or sample["component_score"].nunique() < 5:
                continue
            quintiles = component_quintiles(sample["component_score"], sample["forward_return"])
            q1 = quintiles.iloc[0]["mean_realized_return"]
            q5 = quintiles.iloc[-1]["mean_realized_return"]
            rows.append(
                {
                    "horizon": horizon_label,
                    "trading_days": days,
                    "component": component,
                    "observations": int(len(sample)),
                    "pearson_corr": _safe_corr(sample["component_score"], sample["forward_return"]),
                    "rank_ic": rank_ic(sample["component_score"], sample["forward_return"]),
                    "q5_q1_spread": float(q5 - q1),
                    "q1_mean_return": float(q1),
                    "q5_mean_return": float(q5),
                    "q1_positive_rate": float(quintiles.iloc[0]["positive_rate"]),
                    "q5_positive_rate": float(quintiles.iloc[-1]["positive_rate"]),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["horizon", "component"]).reset_index(drop=True)
    return result


def fit_component_rank_model(
    train: pd.DataFrame,
    min_abs_rank_ic: float,
    max_components: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for component in COMPONENT_COLUMNS:
        sample = train[[component, "forward_return"]].dropna()
        if len(sample) < 100 or sample[component].nunique() < 5:
            continue
        ic = rank_ic(sample[component], sample["forward_return"])
        if pd.isna(ic) or abs(ic) < min_abs_rank_ic:
            continue
        quintiles = component_quintiles(sample[component], sample["forward_return"])
        q_spread = float(quintiles.iloc[-1]["mean_realized_return"] - quintiles.iloc[0]["mean_realized_return"])
        rows.append(
            {
                "component": component,
                "rank_ic": float(ic),
                "quintile_spread": q_spread,
                "strength": float(abs(ic)),
            }
        )

    model = pd.DataFrame(rows)
    if model.empty:
        return model
    model = model.sort_values("strength", ascending=False).head(max_components)
    abs_sum = model["strength"].sum()
    model["weight"] = np.sign(model["rank_ic"]) * (model["strength"] / abs_sum)
    return model.set_index("component").sort_index()


def score_components(features: pd.DataFrame, model: pd.DataFrame) -> pd.Series:
    model_features = features.reindex(columns=model.index)
    weights = model["weight"].reindex(model_features.columns).astype(float)
    denominator = model_features.notna().multiply(weights.abs(), axis=1).sum(axis=1)
    score = model_features.multiply(weights, axis=1).sum(axis=1)
    return score.where(denominator > 0) / denominator


def conditional_distribution(
    training_scores: pd.Series,
    training_returns: pd.Series,
    training_drawdowns: pd.Series,
    current_score: float,
    buckets: int = 5,
) -> dict[str, float]:
    sample = pd.concat(
        [
            training_scores.rename("score"),
            training_returns.rename("forward_return"),
            training_drawdowns.rename("forward_drawdown"),
        ],
        axis=1,
    ).dropna(subset=["score", "forward_return"])
    if sample.empty:
        return _empty_distribution()

    sample["score_bucket"] = pd.qcut(sample["score"], buckets, labels=False, duplicates="drop")
    bucket_edges = sample.groupby("score_bucket")["score"].agg(["min", "max"])
    current_bucket = _bucket_for_score(bucket_edges, current_score)
    bucket_sample = sample[sample["score_bucket"].eq(current_bucket)]
    if len(bucket_sample) < 30:
        bucket_sample = sample
    returns = bucket_sample["forward_return"].dropna()
    drawdowns = bucket_sample["forward_drawdown"].dropna()
    if returns.empty:
        return _empty_distribution()
    return {
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "p10_return": float(returns.quantile(0.10)),
        "p25_return": float(returns.quantile(0.25)),
        "p75_return": float(returns.quantile(0.75)),
        "p90_return": float(returns.quantile(0.90)),
        "probability_positive": float((returns > 0).mean()),
        "probability_drawdown_gt_5pct": float((drawdowns <= -0.05).mean()) if not drawdowns.empty else np.nan,
        "probability_drawdown_gt_10pct": float((drawdowns <= -0.10).mean()) if not drawdowns.empty else np.nan,
    }


def component_quintiles(component_score: pd.Series, forward_return: pd.Series) -> pd.DataFrame:
    sample = pd.concat(
        [component_score.rename("component_score"), forward_return.rename("forward_return")],
        axis=1,
    ).dropna()
    sample["quintile"] = pd.qcut(sample["component_score"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rows = []
    for quintile, group in sample.groupby("quintile", observed=True):
        rows.append(
            {
                "quintile": int(quintile),
                "observations": int(len(group)),
                "mean_component_score": float(group["component_score"].mean()),
                "mean_realized_return": float(group["forward_return"].mean()),
                "median_realized_return": float(group["forward_return"].median()),
                "positive_rate": float((group["forward_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("quintile").reset_index(drop=True)


def rank_ic(component_score: pd.Series, forward_return: pd.Series) -> float:
    sample = pd.concat([component_score, forward_return], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(sample.iloc[:, 0].rank(pct=True).corr(sample.iloc[:, 1].rank(pct=True)))


def percentile_of_value(sample: pd.Series, value: float) -> float:
    clean = sample.dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= value).mean())


def write_forecast_ranker_outputs(result: ForecastRankerResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": root / "market_forecast_ranker_forecasts.parquet",
        "summary": root / "market_forecast_ranker_summary.csv",
        "quintiles": root / "market_forecast_ranker_quintiles.csv",
        "calibration": root / "market_forecast_ranker_calibration.csv",
        "drawdown_calibration": root / "market_forecast_ranker_drawdown_calibration.csv",
        "component_diagnostics": root / "market_component_diagnostics.csv",
        "learned_weights": root / "market_forecast_ranker_weights.parquet",
    }
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    result.quintiles.to_csv(paths["quintiles"], index=False)
    result.calibration.to_csv(paths["calibration"], index=False)
    result.drawdown_calibration.to_csv(paths["drawdown_calibration"], index=False)
    result.component_diagnostics.to_csv(paths["component_diagnostics"], index=False)
    result.learned_weights.to_parquet(paths["learned_weights"])
    return paths


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1]))


def _bucket_for_score(bucket_edges: pd.DataFrame, score: float) -> int:
    containing = bucket_edges[(bucket_edges["min"] <= score) & (score <= bucket_edges["max"])]
    if not containing.empty:
        return int(containing.index[0])
    distances = (bucket_edges.mean(axis=1) - score).abs()
    return int(distances.idxmin())


def _empty_distribution() -> dict[str, float]:
    return {
        "mean_return": float("nan"),
        "median_return": float("nan"),
        "p10_return": float("nan"),
        "p25_return": float("nan"),
        "p75_return": float("nan"),
        "p90_return": float("nan"),
        "probability_positive": float("nan"),
        "probability_drawdown_gt_5pct": float("nan"),
        "probability_drawdown_gt_10pct": float("nan"),
    }
