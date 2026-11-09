from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroRegimeForecastResult:
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    current: pd.DataFrame


def run_macro_regime_forecast(
    regime_tape: pd.DataFrame,
    config: dict[str, Any],
) -> MacroRegimeForecastResult:
    daily = _feature_panel(regime_tape, config)
    monthly = daily.groupby(daily.index.to_period("M")).tail(1)
    feature_columns = [column for column in daily if column != "macro_regime"]
    rows: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    minimum_train = int(config.get("minimum_train_observations", 120))
    minimum_class = int(config.get("minimum_class_observations", 10))
    ridge = float(config.get("ridge_penalty", 1.0))

    for target_id, target_settings in config.get("targets", {}).items():
        regimes = {str(value) for value in target_settings.get("regimes", [])}
        current_active = str(daily["macro_regime"].dropna().iloc[-1]) in regimes
        for horizon, trading_days_value in config.get("horizons", {}).items():
            trading_days = int(trading_days_value)
            future_state = daily["macro_regime"].shift(-trading_days)
            target = future_state.isin(regimes).astype(float).where(future_state.notna())
            evaluation = monthly[feature_columns].copy()
            evaluation["target"] = target.reindex(evaluation.index)
            evaluation["current_active"] = monthly["macro_regime"].isin(regimes).astype(float)
            evaluation = evaluation.dropna()
            for position in range(minimum_train, len(evaluation)):
                train = evaluation.iloc[:position]
                if train["target"].value_counts().min() < minimum_class:
                    continue
                test = evaluation.iloc[position]
                probability = _fit_predict_logistic(
                    train[feature_columns], train["target"], test[feature_columns], ridge
                )
                comparable = train.loc[train["current_active"].eq(test["current_active"])]
                baseline_sample = comparable if len(comparable) >= minimum_class else train
                rows.append(
                    {
                        "forecast_date": evaluation.index[position],
                        "target_id": target_id,
                        "target_label": str(target_settings.get("label", target_id)),
                        "horizon": horizon,
                        "trading_days": trading_days,
                        "probability": probability,
                        "baseline_probability": float(baseline_sample["target"].mean()),
                        "current_target_active": bool(test["current_active"]),
                        "realized": float(test["target"]),
                        "train_observations": int(len(train)),
                    }
                )

            known_target = monthly[feature_columns].copy()
            known_target["target"] = target.reindex(known_target.index)
            known_target["current_active"] = monthly["macro_regime"].isin(regimes).astype(float)
            known_target = known_target.dropna()
            current_features = daily[feature_columns].dropna().iloc[-1]
            if (
                len(known_target) >= minimum_train
                and known_target["target"].nunique() == 2
                and known_target["target"].value_counts().min() >= minimum_class
            ):
                probability = _fit_predict_logistic(
                    known_target[feature_columns],
                    known_target["target"],
                    current_features,
                    ridge,
                )
                comparable = known_target.loc[
                    known_target["current_active"].eq(float(current_active))
                ]
                baseline_sample = comparable if len(comparable) >= minimum_class else known_target
                current_rows.append(
                    {
                        "as_of_date": current_features.name,
                        "target_id": target_id,
                        "target_label": str(target_settings.get("label", target_id)),
                        "horizon": horizon,
                        "trading_days": trading_days,
                        "probability": probability,
                        "baseline_probability": float(baseline_sample["target"].mean()),
                        "current_target_active": current_active,
                        "forecast_type": "persistence" if current_active else "entry",
                        "train_observations": int(len(known_target)),
                    }
                )

    forecasts = pd.DataFrame.from_records(rows)
    current = pd.DataFrame.from_records(current_rows)
    summary = summarize_macro_regime_forecasts(forecasts, config)
    return MacroRegimeForecastResult(forecasts=forecasts, summary=summary, current=current)


def summarize_macro_regime_forecasts(
    forecasts: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if forecasts.empty:
        return pd.DataFrame()
    for _, group in forecasts.groupby(["target_id", "horizon"], sort=False):
        group = group.sort_values("forecast_date")
        model_error = (group["probability"] - group["realized"]).pow(2)
        baseline_error = (group["baseline_probability"] - group["realized"]).pow(2)
        midpoint = max(1, len(group) // 2)
        early_improvement = float(
            baseline_error.iloc[:midpoint].mean() - model_error.iloc[:midpoint].mean()
        )
        late_improvement = float(
            baseline_error.iloc[midpoint:].mean() - model_error.iloc[midpoint:].mean()
        )
        brier_improvement = float(baseline_error.mean() - model_error.mean())
        auc = _roc_auc(group["realized"], group["probability"])
        observations = len(group)
        minimum_oos = int(config.get("minimum_oos_observations", 60))
        rows.append(
            {
                "target_id": group["target_id"].iloc[0],
                "target_label": group["target_label"].iloc[0],
                "horizon": group["horizon"].iloc[0],
                "trading_days": int(group["trading_days"].iloc[0]),
                "observations": observations,
                "sample_start": group["forecast_date"].min(),
                "sample_end": group["forecast_date"].max(),
                "event_rate": float(group["realized"].mean()),
                "model_brier_score": float(model_error.mean()),
                "baseline_brier_score": float(baseline_error.mean()),
                "brier_improvement": brier_improvement,
                "early_brier_improvement": early_improvement,
                "late_brier_improvement": late_improvement,
                "roc_auc": auc,
                "research_candidate": bool(
                    observations >= minimum_oos
                    and brier_improvement > 0.0
                    and early_improvement > 0.0
                    and late_improvement > 0.0
                    and pd.notna(auc)
                    and auc >= 0.55
                ),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["research_candidate", "brier_improvement"], ascending=[False, False]
    ).reset_index(drop=True)


def write_macro_regime_forecast_outputs(
    result: MacroRegimeForecastResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": root / "macro_regime_forecasts.parquet",
        "summary": root / "macro_regime_forecast_summary.csv",
        "current": root / "macro_regime_current_forecasts.csv",
    }
    result.forecasts.to_parquet(paths["forecasts"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.current.to_csv(paths["current"], index=False)
    return paths


def _feature_panel(tape: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    features = [str(value) for value in config.get("features", [])]
    panel = tape[["macro_regime", *features]].copy().sort_index()
    for window_value in config.get("feature_change_windows", []):
        window = int(window_value)
        for feature in features:
            panel[f"{feature}__change_{window}d"] = panel[feature] - panel[feature].shift(window)
    return panel


def _fit_predict_logistic(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.Series,
    ridge_penalty: float,
) -> float:
    x = train_x.to_numpy(dtype=float)
    y = train_y.to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    x = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1]) * ridge_penalty
    penalty[0, 0] = 0.0
    for _ in range(100):
        probability = _sigmoid(design @ coefficients)
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (probability - y) + penalty @ coefficients
        hessian = design.T @ (design * weights[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    test = (test_x.to_numpy(dtype=float) - mean) / scale
    return float(_sigmoid(np.array([1.0, *test]) @ coefficients))


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _roc_auc(actual: pd.Series, probability: pd.Series) -> float:
    frame = pd.DataFrame({"actual": actual, "probability": probability}).dropna()
    positives = int(frame["actual"].eq(1.0).sum())
    negatives = int(frame["actual"].eq(0.0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = frame["probability"].rank(method="average")
    positive_rank_sum = float(ranks.loc[frame["actual"].eq(1.0)].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
