from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.states.forecast import fit_component_rank_model, rank_ic, score_components
from world_model.states.market import COMPONENT_COLUMNS, MarketStateEngine, _forward_drawdown, _forward_return
from world_model.states.walk_forward import _evaluation_dates


@dataclass(frozen=True)
class TargetSpec:
    name: str
    label: str
    kind: str
    target: pd.Series


@dataclass(frozen=True)
class TargetValidationResult:
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    quintiles: pd.DataFrame
    component_diagnostics: pd.DataFrame
    learned_weights: pd.DataFrame


class MarketTargetValidator:
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
    ) -> TargetValidationResult:
        state = self.engine.build_state_history()
        prices = self.engine.load_prices().ffill()
        horizons: dict[str, int] = self.engine.config.get("horizons", {})
        if not horizons:
            raise ValueError("Market config must define at least one horizon.")

        target_specs = build_target_specs(state=state, prices=prices, horizons=horizons)
        max_horizon = int(max(horizons.values()))
        eval_dates = _evaluation_dates(
            state=state,
            target_close=prices[self.engine.config.get("target_asset", "ES")],
            max_horizon=max_horizon,
            start=start,
            end=end,
            frequency=frequency,
        )

        forecast_rows: list[dict[str, Any]] = []
        weight_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []

        for spec in target_specs:
            horizon_label, days = _target_horizon(spec.name, horizons)
            diagnostic_rows.extend(component_target_diagnostics(state, spec, horizon_label, days))
            for forecast_date in eval_dates:
                current_features = state.loc[forecast_date, COMPONENT_COLUMNS].astype(float)
                if current_features.dropna().empty:
                    continue

                train_cutoff = forecast_date - pd.tseries.offsets.BDay(days)
                train = pd.concat(
                    [state[COMPONENT_COLUMNS], spec.target.rename("target_value")],
                    axis=1,
                ).loc[:train_cutoff]
                if train_window_years is not None:
                    train_start = train_cutoff - pd.DateOffset(years=train_window_years)
                    train = train.loc[train.index >= train_start]
                train = train.dropna(subset=["target_value"])
                if len(train) < min_train_observations:
                    continue

                model = fit_component_rank_model(
                    train=train.rename(columns={"target_value": "forward_return"}),
                    min_abs_rank_ic=min_abs_rank_ic,
                    max_components=max_components,
                )
                if model.empty:
                    continue

                training_scores = score_components(train[COMPONENT_COLUMNS], model)
                current_score = score_components(current_features.to_frame().T, model).iloc[0]
                if pd.isna(current_score):
                    continue

                forecast = conditional_target_estimate(
                    training_scores=training_scores,
                    training_target=train["target_value"],
                    current_score=float(current_score),
                    kind=spec.kind,
                )
                realized = spec.target.get(forecast_date, np.nan)
                baseline = float(train["target_value"].mean())

                forecast_rows.append(
                    {
                        "forecast_date": forecast_date,
                        "target": spec.name,
                        "target_label": spec.label,
                        "target_kind": spec.kind,
                        "horizon": horizon_label,
                        "trading_days": days,
                        "forecast_score": float(current_score),
                        "forecast_value": forecast,
                        "baseline_value": baseline,
                        "realized_value": realized,
                        "component_count": int(len(model)),
                        "train_observations": int(len(train)),
                    }
                )

                for component, row in model.iterrows():
                    weight_rows.append(
                        {
                            "forecast_date": forecast_date,
                            "target": spec.name,
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
            raise ValueError("Target validation produced no forecasts.")
        forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
        forecasts = forecasts.sort_values(["target", "trading_days", "forecast_date"]).reset_index(drop=True)
        learned_weights = pd.DataFrame(weight_rows)
        component_diagnostics = pd.DataFrame(diagnostic_rows)
        if not component_diagnostics.empty:
            component_diagnostics = component_diagnostics.sort_values(["target", "component"]).reset_index(drop=True)
        return TargetValidationResult(
            forecasts=forecasts,
            summary=summarize_target_forecasts(forecasts),
            quintiles=summarize_target_quintiles(forecasts),
            component_diagnostics=component_diagnostics,
            learned_weights=learned_weights,
        )


def build_target_specs(state: pd.DataFrame, prices: pd.DataFrame, horizons: dict[str, int]) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    es = prices["ES"]
    es_drawdowns = {label: _forward_drawdown(es, days) for label, days in horizons.items()}
    es_forward_vol = {label: future_realized_vol(es, days) for label, days in horizons.items()}
    current_vol = trailing_realized_vol(es, 63)

    for label, days in horizons.items():
        specs.append(
            TargetSpec(
                name=f"es_drawdown_10pct_{label}",
                label=f"ES 10% drawdown risk {label}",
                kind="binary",
                target=(es_drawdowns[label] <= -0.10).astype(float).where(es_drawdowns[label].notna()),
            )
        )
        specs.append(
            TargetSpec(
                name=f"es_vol_expansion_{label}",
                label=f"ES realized volatility expansion {label}",
                kind="binary",
                target=(es_forward_vol[label] > current_vol * 1.25).astype(float).where(es_forward_vol[label].notna() & current_vol.notna()),
            )
        )
        future_composite = state["composite_score"].shift(-days)
        deterioration = (future_composite - state["composite_score"]) <= -0.25
        specs.append(
            TargetSpec(
                name=f"state_deterioration_{label}",
                label=f"Market state deterioration {label}",
                kind="binary",
                target=deterioration.astype(float).where(future_composite.notna() & state["composite_score"].notna()),
            )
        )

    relative_targets = {
        "equity_vs_treasury": {
            "label": "ES relative to 10Y Treasury futures",
            "long": ["ES"],
            "short": ["ZN"],
        },
        "risk_on_vs_defensive": {
            "label": "Risk-on basket relative to defensive basket",
            "long": ["ES", "NQ", "CL", "HG"],
            "short": ["ZN", "ZB", "GC"],
        },
    }
    for base_name, settings in relative_targets.items():
        for label, days in horizons.items():
            specs.append(
                TargetSpec(
                    name=f"{base_name}_{label}",
                    label=f"{settings['label']} {label}",
                    kind="continuous",
                    target=relative_forward_return(prices, settings["long"], settings["short"], days),
                )
            )
    return specs


def summarize_target_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, horizon), group in forecasts.groupby(["target", "horizon"], sort=False):
        clean = group.dropna(subset=["forecast_value", "baseline_value", "realized_value"])
        if clean.empty:
            continue
        kind = str(clean["target_kind"].iloc[0])
        forecast_error = clean["forecast_value"] - clean["realized_value"]
        baseline_error = clean["baseline_value"] - clean["realized_value"]
        row = {
            "target": target,
            "target_label": clean["target_label"].iloc[0],
            "target_kind": kind,
            "horizon": horizon,
            "trading_days": int(clean["trading_days"].iloc[0]),
            "observations": int(len(clean)),
            "realized_mean": float(clean["realized_value"].mean()),
            "forecast_mean": float(clean["forecast_value"].mean()),
            "baseline_mean": float(clean["baseline_value"].mean()),
            "forecast_realized_corr": _safe_corr(clean["forecast_value"], clean["realized_value"]),
            "forecast_score_realized_corr": _safe_corr(clean["forecast_score"], clean["realized_value"]),
            "q5_q1_realized_spread": _top_bottom_spread(clean),
            "model_rmse": float(np.sqrt(np.mean(forecast_error.pow(2)))),
            "baseline_rmse": float(np.sqrt(np.mean(baseline_error.pow(2)))),
        }
        if kind == "binary":
            row["model_brier"] = float(np.mean(forecast_error.pow(2)))
            row["baseline_brier"] = float(np.mean(baseline_error.pow(2)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "trading_days"]).reset_index(drop=True)


def summarize_target_quintiles(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, horizon), group in forecasts.groupby(["target", "horizon"], sort=False):
        clean = group.dropna(subset=["forecast_value", "realized_value"]).copy()
        if clean["forecast_value"].nunique() < 5:
            continue
        clean["forecast_quintile"] = pd.qcut(clean["forecast_value"], 5, labels=False, duplicates="drop") + 1
        for quintile, q_group in clean.groupby("forecast_quintile", observed=True):
            rows.append(
                {
                    "target": target,
                    "target_label": clean["target_label"].iloc[0],
                    "target_kind": clean["target_kind"].iloc[0],
                    "horizon": horizon,
                    "trading_days": int(q_group["trading_days"].iloc[0]),
                    "forecast_quintile": int(quintile),
                    "observations": int(len(q_group)),
                    "mean_forecast_value": float(q_group["forecast_value"].mean()),
                    "mean_realized_value": float(q_group["realized_value"].mean()),
                    "median_realized_value": float(q_group["realized_value"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "trading_days", "forecast_quintile"]).reset_index(drop=True)


def component_target_diagnostics(
    state: pd.DataFrame,
    spec: TargetSpec,
    horizon_label: str,
    days: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in COMPONENT_COLUMNS + ["composite_score"]:
        sample = pd.concat(
            [state[component].rename("component_score"), spec.target.rename("target_value")],
            axis=1,
        ).dropna()
        if len(sample) < 100 or sample["component_score"].nunique() < 5:
            continue
        quintiles = component_target_quintiles(sample["component_score"], sample["target_value"])
        rows.append(
            {
                "target": spec.name,
                "target_label": spec.label,
                "target_kind": spec.kind,
                "horizon": horizon_label,
                "trading_days": days,
                "component": component,
                "observations": int(len(sample)),
                "rank_ic": rank_ic(sample["component_score"], sample["target_value"]),
                "q5_q1_spread": float(quintiles.iloc[-1]["mean_realized_value"] - quintiles.iloc[0]["mean_realized_value"]),
                "q1_mean": float(quintiles.iloc[0]["mean_realized_value"]),
                "q5_mean": float(quintiles.iloc[-1]["mean_realized_value"]),
            }
        )
    return rows


def component_target_quintiles(component_score: pd.Series, target: pd.Series) -> pd.DataFrame:
    sample = pd.concat([component_score.rename("component_score"), target.rename("target_value")], axis=1).dropna()
    sample["quintile"] = pd.qcut(sample["component_score"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rows = []
    for quintile, group in sample.groupby("quintile", observed=True):
        rows.append(
            {
                "quintile": int(quintile),
                "observations": int(len(group)),
                "mean_component_score": float(group["component_score"].mean()),
                "mean_realized_value": float(group["target_value"].mean()),
                "median_realized_value": float(group["target_value"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("quintile").reset_index(drop=True)


def conditional_target_estimate(
    training_scores: pd.Series,
    training_target: pd.Series,
    current_score: float,
    kind: str,
    buckets: int = 5,
) -> float:
    sample = pd.concat([training_scores.rename("score"), training_target.rename("target_value")], axis=1).dropna()
    if sample.empty:
        return float("nan")
    sample["score_bucket"] = pd.qcut(sample["score"], buckets, labels=False, duplicates="drop")
    bucket_edges = sample.groupby("score_bucket")["score"].agg(["min", "max"])
    current_bucket = _bucket_for_score(bucket_edges, current_score)
    bucket_sample = sample[sample["score_bucket"].eq(current_bucket)]
    if len(bucket_sample) < 30:
        bucket_sample = sample
    value = float(bucket_sample["target_value"].mean())
    if kind == "binary":
        return min(max(value, 0.0), 1.0)
    return value


def relative_forward_return(prices: pd.DataFrame, long_assets: list[str], short_assets: list[str], days: int) -> pd.Series:
    long_returns = [_forward_return(prices[asset].ffill(), days) for asset in long_assets if asset in prices.columns]
    short_returns = [_forward_return(prices[asset].ffill(), days) for asset in short_assets if asset in prices.columns]
    if not long_returns or not short_returns:
        return pd.Series(index=prices.index, dtype=float)
    long_basket = pd.concat(long_returns, axis=1).mean(axis=1)
    short_basket = pd.concat(short_returns, axis=1).mean(axis=1)
    return long_basket - short_basket


def future_realized_vol(close: pd.Series, days: int, annualization: int = 252) -> pd.Series:
    returns = close.pct_change(fill_method=None)
    paths = [returns.shift(-step) for step in range(1, days + 1)]
    return pd.concat(paths, axis=1).std(axis=1) * np.sqrt(annualization)


def trailing_realized_vol(close: pd.Series, window: int = 63, annualization: int = 252) -> pd.Series:
    return close.pct_change(fill_method=None).rolling(window, min_periods=max(20, window // 2)).std() * np.sqrt(annualization)


def write_target_validation_outputs(result: TargetValidationResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": root / "market_target_validation_forecasts.parquet",
        "summary": root / "market_target_validation_summary.csv",
        "quintiles": root / "market_target_validation_quintiles.csv",
        "component_diagnostics": root / "market_target_component_diagnostics.csv",
        "learned_weights": root / "market_target_validation_weights.parquet",
    }
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    result.quintiles.to_csv(paths["quintiles"], index=False)
    result.component_diagnostics.to_csv(paths["component_diagnostics"], index=False)
    result.learned_weights.to_parquet(paths["learned_weights"])
    return paths


def _target_horizon(target_name: str, horizons: dict[str, int]) -> tuple[str, int]:
    matches = [(label, days) for label, days in horizons.items() if target_name.endswith(f"_{label}")]
    if not matches:
        raise ValueError(f"Could not infer horizon from target: {target_name}")
    return matches[0]


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1]))


def _top_bottom_spread(clean: pd.DataFrame) -> float:
    if clean["forecast_value"].nunique() < 5:
        return float("nan")
    bucketed = clean.copy()
    bucketed["_quintile"] = pd.qcut(bucketed["forecast_value"], 5, labels=False, duplicates="drop")
    realized_by_quintile = bucketed.groupby("_quintile")["realized_value"].mean()
    if realized_by_quintile.empty:
        return float("nan")
    return float(realized_by_quintile.iloc[-1] - realized_by_quintile.iloc[0])


def _bucket_for_score(bucket_edges: pd.DataFrame, score: float) -> int:
    containing = bucket_edges[(bucket_edges["min"] <= score) & (score <= bucket_edges["max"])]
    if not containing.empty:
        return int(containing.index[0])
    distances = (bucket_edges.mean(axis=1) - score).abs()
    return int(distances.idxmin())
