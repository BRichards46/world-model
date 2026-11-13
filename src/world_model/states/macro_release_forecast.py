from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table


@dataclass(frozen=True)
class MacroReleaseForecastResult:
    datasets: pd.DataFrame
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    current: pd.DataFrame
    current_features: pd.DataFrame


def run_macro_release_forecast(
    fred_root: str | Path,
    config: dict[str, Any],
    as_of: str | pd.Timestamp | None = None,
) -> MacroReleaseForecastResult:
    as_of_date = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    model_settings = config.get("model", {})
    tables = _load_required_tables(fred_root, config)
    dataset_frames: list[pd.DataFrame] = []
    forecast_rows: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    current_feature_rows: list[dict[str, Any]] = []

    for target_id, target in config.get("targets", {}).items():
        base = _target_events(tables[str(target["series_id"])], target, as_of_date)
        if base.empty:
            continue
        for variant_id, variant in target.get("variants", {}).items():
            dataset = _build_dataset(base, tables, target_id, variant_id, target, variant)
            dataset_frames.append(dataset)
            minimum_train = int(variant.get("minimum_train_observations", model_settings.get("minimum_train_observations", 60)))
            minimum_coverage = float(
                variant.get("minimum_feature_coverage", model_settings.get("minimum_feature_coverage", 0.8))
            )
            ridge = float(variant.get("ridge_penalty", model_settings.get("ridge_penalty", 4.0)))
            features = [str(item["id"]) for item in variant.get("features", [])]
            usable = dataset.loc[dataset["feature_coverage"] >= minimum_coverage].dropna(subset=["target", "prior_target"])

            for position in range(minimum_train, len(usable)):
                train = usable.iloc[:position]
                test = usable.iloc[position]
                prediction, _ = _ridge_prediction(train[features], train["target"], test[features], ridge)
                baseline, _ = _ridge_prediction(
                    train[["prior_target"]], train["target"], test[["prior_target"]], ridge
                )
                forecast_rows.append(
                    _forecast_record(test, target_id, variant_id, target, prediction, baseline, len(train))
                )

            current_features = _feature_values(tables, variant.get("features", []), as_of_date)
            current_coverage = float(current_features.notna().mean()) if len(current_features) else 0.0
            if len(usable) < minimum_train or current_coverage < minimum_coverage:
                continue
            latest_target = _known_series(tables[str(target["series_id"])], as_of_date)
            prior_target = _metric(latest_target, str(target["target_transform"]), target)
            prediction, fit = _ridge_prediction(usable[features], usable["target"], current_features, ridge)
            baseline, _ = _ridge_prediction(
                usable[["prior_target"]], usable["target"], pd.Series({"prior_target": prior_target}), ridge
            )
            latest_observation = pd.Timestamp(latest_target.index[-1])
            forecast_observation = latest_observation + pd.DateOffset(months=1)
            current_rows.append(
                {
                    "as_of_date": as_of_date,
                    "target_id": target_id,
                    "target_label": str(target.get("label", target_id)),
                    "variant_id": variant_id,
                    "forecast_observation_date": forecast_observation,
                    "forecast": prediction,
                    "baseline_forecast": baseline,
                    "prior_target": prior_target,
                    "unit": str(target.get("unit", "")),
                    "display_forecast": _display_forecast(prediction, latest_target, target),
                    "display_baseline_forecast": _display_forecast(baseline, latest_target, target),
                    "display_prior_target": _display_forecast(prior_target, latest_target, target),
                    "display_unit": str(target.get("display_unit", target.get("unit", ""))),
                    "display_scale": _display_forecast(1.0, latest_target, target),
                    "feature_coverage": current_coverage,
                    "train_observations": len(usable),
                }
            )
            for feature_id in features:
                current_feature_rows.append(
                    {
                        "as_of_date": as_of_date,
                        "target_id": target_id,
                        "variant_id": variant_id,
                        "feature_id": feature_id,
                        "value": current_features.get(feature_id),
                        "standardized_coefficient": fit["coefficients"].get(feature_id),
                        "standardized_value": fit["standardized_test"].get(feature_id),
                        "contribution": fit["contributions"].get(feature_id),
                    }
                )

    datasets = pd.concat(dataset_frames, ignore_index=True, sort=False) if dataset_frames else pd.DataFrame()
    forecasts = pd.DataFrame.from_records(forecast_rows)
    current = pd.DataFrame.from_records(current_rows)
    current_features = pd.DataFrame.from_records(current_feature_rows)
    summary = summarize_macro_release_forecasts(forecasts, config)
    if not current.empty and not summary.empty:
        flags = summary[["target_id", "variant_id", "research_candidate"]]
        current = current.merge(flags, on=["target_id", "variant_id"], how="left")
    return MacroReleaseForecastResult(datasets, forecasts, summary, current, current_features)


def summarize_macro_release_forecasts(
    forecasts: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (target_id, variant_id), group in forecasts.groupby(["target_id", "variant_id"], sort=False):
        group = group.sort_values("release_date")
        model_error = (group["forecast"] - group["actual"]).abs()
        baseline_error = (group["baseline_forecast"] - group["actual"]).abs()
        split = max(1, len(group) // 2)
        early = float(baseline_error.iloc[:split].mean() - model_error.iloc[:split].mean())
        late = float(baseline_error.iloc[split:].mean() - model_error.iloc[split:].mean())
        improvement = float(baseline_error.mean() - model_error.mean())
        target_settings = config["targets"][target_id]
        variant_settings = target_settings["variants"][variant_id]
        minimum_oos = int(
            variant_settings.get(
                "minimum_oos_observations", config.get("model", {}).get("minimum_oos_observations", 36)
            )
        )
        maximum_mae = float(target_settings.get("max_model_mae", float("inf")))
        rows.append(
            {
                "target_id": target_id,
                "target_label": group["target_label"].iloc[0],
                "variant_id": variant_id,
                "observations": len(group),
                "sample_start": group["release_date"].min(),
                "sample_end": group["release_date"].max(),
                "model_mae": float(model_error.mean()),
                "baseline_mae": float(baseline_error.mean()),
                "mae_improvement": improvement,
                "early_mae_improvement": early,
                "late_mae_improvement": late,
                "model_rmse": float(np.sqrt(((group["forecast"] - group["actual"]) ** 2).mean())),
                "baseline_rmse": float(np.sqrt(((group["baseline_forecast"] - group["actual"]) ** 2).mean())),
                "direction_hit_rate": float(group["direction_correct"].mean()),
                "research_candidate": bool(
                    len(group) >= minimum_oos
                    and float(model_error.mean()) <= maximum_mae
                    and improvement > 0.0
                    and early > 0.0
                    and late > 0.0
                    and float(group["direction_correct"].mean()) >= 0.52
                ),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["research_candidate", "mae_improvement"], ascending=[False, False]
    ).reset_index(drop=True)


RANGE_WINDOW = 60
RANGE_MINIMUM = 24


def forecast_ranges(
    forecasts: pd.DataFrame,
    target_id: str,
    variant_id: str,
    window: int = RANGE_WINDOW,
    minimum: int = RANGE_MINIMUM,
) -> dict[str, Any] | None:
    """Empirical 50% and 80% error bands from the most recent walk-forward misses (actual minus forecast).

    Every walk-forward forecast used only vintages published before its release, so the misses are honest
    out-of-sample errors. The recent window keeps a persistent bias (for example payrolls coming in below the
    model) inside the band instead of averaging it away.
    """
    if forecasts.empty:
        return None
    rows = forecasts[(forecasts["target_id"] == target_id) & (forecasts["variant_id"] == variant_id)]
    rows = rows.dropna(subset=["actual", "forecast"]).sort_values("release_date").tail(window)
    if len(rows) < minimum:
        return None
    errors = rows["actual"] - rows["forecast"]
    quantiles = errors.quantile([0.10, 0.25, 0.75, 0.90])
    return {
        "error_50": [float(quantiles.loc[0.25]), float(quantiles.loc[0.75])],
        "error_80": [float(quantiles.loc[0.10]), float(quantiles.loc[0.90])],
        "observations": int(len(rows)),
        "since": f"{pd.Timestamp(rows['release_date'].iloc[0]):%Y-%m}",
    }


def build_release_site_payload(result: MacroReleaseForecastResult) -> dict[str, Any]:
    """Browser payload: current forecasts with their walk-forward record. Only validated rows count as signals."""
    summary = result.summary.set_index(["target_id", "variant_id"]) if not result.summary.empty else pd.DataFrame()
    rows = []
    for row in result.current.itertuples(index=False):
        record = summary.loc[(row.target_id, row.variant_id)] if (row.target_id, row.variant_id) in summary.index else None
        same_units = str(row.unit) == str(row.display_unit)

        def stat(name: str) -> float | None:
            if record is None or name not in record or pd.isna(record[name]):
                return None
            return float(record[name])

        ranges = forecast_ranges(result.forecasts, row.target_id, row.variant_id)
        scale = float(getattr(row, "display_scale", 1.0))
        forecast = float(row.display_forecast)
        rows.append(
            {
                "target_id": row.target_id,
                "target_label": row.target_label,
                "variant_id": row.variant_id,
                "reference_month": pd.Timestamp(row.forecast_observation_date).strftime("%Y-%m"),
                "forecast": float(row.display_forecast),
                "baseline_forecast": float(row.display_baseline_forecast),
                "prior": float(row.display_prior_target),
                "unit": row.display_unit,
                "validated": bool(pd.notna(getattr(row, "research_candidate", None)) and bool(getattr(row, "research_candidate"))),
                "model_mae": stat("model_mae") if same_units else None,
                "baseline_mae": stat("baseline_mae") if same_units else None,
                "direction_hit_rate": stat("direction_hit_rate"),
                "observations": int(record["observations"]) if record is not None and pd.notna(record["observations"]) else None,
                "range_50": [forecast + scale * error for error in ranges["error_50"]] if ranges else None,
                "range_80": [forecast + scale * error for error in ranges["error_80"]] if ranges else None,
                "range_observations": ranges["observations"] if ranges else None,
                "range_since": ranges["since"] if ranges else None,
            }
        )
    as_of = result.current["as_of_date"].iloc[0] if not result.current.empty else None
    return {
        "as_of_date": f"{pd.Timestamp(as_of):%Y-%m-%d}" if as_of is not None else None,
        "forecasts": rows,
        "note": "Validated = beats a prior-reading autoregression on mean absolute error in the full, early and late "
        "walk-forward samples, with at least a 52% directional hit rate. Other rows are previews, not signals. "
        f"Ranges come from the last {RANGE_WINDOW} out-of-sample misses: half of past releases landed inside the "
        "50% range and four in five inside the 80% range.",
    }


def write_macro_release_forecast_outputs(
    result: MacroReleaseForecastResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "datasets": root / "macro_release_datasets.parquet",
        "forecasts": root / "macro_release_forecasts.parquet",
        "summary": root / "macro_release_forecast_summary.csv",
        "current": root / "macro_release_current_forecasts.csv",
        "current_features": root / "macro_release_current_features.csv",
    }
    result.datasets.to_parquet(paths["datasets"], index=False)
    result.forecasts.to_parquet(paths["forecasts"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.current.to_csv(paths["current"], index=False)
    result.current_features.to_csv(paths["current_features"], index=False)
    return paths


def _load_required_tables(fred_root: str | Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    series_ids: set[str] = set()
    for target in config.get("targets", {}).values():
        series_ids.add(str(target["series_id"]))
        for variant in target.get("variants", {}).values():
            series_ids.update(str(feature["series_id"]) for feature in variant.get("features", []))
    return {series_id: load_vintage_table(fred_root, series_id) for series_id in sorted(series_ids)}


def _target_events(vintage: pd.DataFrame, settings: dict[str, Any], as_of: pd.Timestamp) -> pd.DataFrame:
    first_release = (
        vintage.loc[vintage["realtime_start"] <= as_of]
        .groupby("observation_date", as_index=False)["realtime_start"]
        .min()
        .rename(columns={"realtime_start": "release_date"})
    )
    lag = (first_release["release_date"] - first_release["observation_date"]).dt.days
    first_release = first_release.loc[
        lag.between(
            int(settings.get("min_release_lag_days", 20)),
            int(settings.get("max_release_lag_days", 85)),
        )
    ].sort_values("release_date")
    rows: list[dict[str, Any]] = []
    for row in first_release.itertuples(index=False):
        known = _known_series(vintage, pd.Timestamp(row.release_date))
        current = known.get(pd.Timestamp(row.observation_date), np.nan)
        prior_date = pd.Timestamp(row.observation_date) - pd.DateOffset(months=1)
        prior = known.get(prior_date, np.nan)
        if pd.isna(current) or pd.isna(prior):
            continue
        target = _pair_transform(float(current), float(prior), str(settings["target_transform"]))
        target *= float(settings.get("target_scale", 1.0))
        prior_target = _metric(
            known.loc[known.index <= prior_date], str(settings["target_transform"]), settings
        )
        rows.append(
            {
                "observation_date": pd.Timestamp(row.observation_date),
                "release_date": pd.Timestamp(row.release_date),
                "target": target,
                "prior_target": prior_target,
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_dataset(
    base: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    target_id: str,
    variant_id: str,
    target: dict[str, Any],
    variant: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in base.itertuples(index=False):
        cutoff = pd.Timestamp(event.release_date) - pd.Timedelta(days=1)
        values = _feature_values(tables, variant.get("features", []), cutoff)
        record = {
            "target_id": target_id,
            "target_label": str(target.get("label", target_id)),
            "variant_id": variant_id,
            "observation_date": pd.Timestamp(event.observation_date),
            "release_date": pd.Timestamp(event.release_date),
            "target": float(event.target),
            "prior_target": float(event.prior_target),
            "feature_coverage": float(values.notna().mean()) if len(values) else 0.0,
        }
        record.update(values.to_dict())
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _feature_values(
    tables: dict[str, pd.DataFrame],
    features: list[dict[str, Any]],
    cutoff: pd.Timestamp,
) -> pd.Series:
    values = {}
    for feature in features:
        known = _known_series(tables[str(feature["series_id"])], cutoff)
        values[str(feature["id"])] = _metric(known, str(feature["transform"]), feature)
    return pd.Series(values, dtype=float)


def _known_series(vintage: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    frame = vintage.loc[vintage["realtime_start"] <= cutoff]
    if frame.empty:
        return pd.Series(dtype=float)
    frame = frame.sort_values(["realtime_start", "observation_date"]).drop_duplicates(
        "observation_date", keep="last"
    )
    return frame.set_index("observation_date")["value"].dropna().astype(float).sort_index()


def _metric(series: pd.Series, transform: str, settings: dict[str, Any]) -> float:
    if series.empty:
        return float("nan")
    scale = float(settings.get("scale", settings.get("target_scale", 1.0)))
    lag = int(settings.get("lag", 1))
    window = int(settings.get("window", 1))
    if transform == "latest":
        value = float(series.iloc[-1])
    elif transform == "monthly_diff":
        monthly = _monthly(series)
        value = float(monthly.diff().iloc[-1]) if len(monthly) >= 2 else float("nan")
    elif transform == "monthly_diff_mean":
        monthly = _monthly(series)
        value = float(monthly.diff().tail(window).mean()) if len(monthly) >= window + 1 else float("nan")
    elif transform == "monthly_pct":
        monthly = _monthly(series)
        value = float(monthly.pct_change(periods=lag, fill_method=None).iloc[-1]) if len(monthly) > lag else float("nan")
    elif transform == "monthly_pct_mean":
        monthly = _monthly(series)
        changes = monthly.pct_change(fill_method=None)
        value = float(changes.tail(window).mean()) if len(changes.dropna()) >= window else float("nan")
    elif transform == "latest_pct":
        value = float(series.pct_change(periods=lag, fill_method=None).iloc[-1]) if len(series) > lag else float("nan")
    elif transform == "rolling_pct":
        value = float(series.iloc[-1] / series.iloc[-1 - lag] - 1.0) if len(series) > lag else float("nan")
    elif transform == "rolling_mean":
        value = float(series.tail(window).mean()) if len(series) >= window else float("nan")
    elif transform == "rolling_mean_change":
        if len(series) < window + lag:
            value = float("nan")
        else:
            value = float(series.tail(window).mean() - series.iloc[-window - lag : -lag].mean())
    elif transform == "monthly_mean_pct":
        monthly = series.groupby(series.index.to_period("M")).mean()
        value = float(monthly.pct_change(fill_method=None).iloc[-1]) if len(monthly) >= 2 else float("nan")
    else:
        raise ValueError(f"Unsupported release forecast transform: {transform}")
    return value * scale


def _monthly(series: pd.Series) -> pd.Series:
    monthly = series.copy()
    monthly.index = monthly.index.to_period("M")
    return monthly.groupby(level=0).last().sort_index()


def _pair_transform(current: float, prior: float, transform: str) -> float:
    if transform == "monthly_diff":
        return current - prior
    if transform == "monthly_pct":
        return current / prior - 1.0 if prior != 0.0 else float("nan")
    raise ValueError(f"Unsupported target transform: {transform}")


def _display_forecast(
    forecast: float,
    latest_target: pd.Series,
    settings: dict[str, Any],
) -> float:
    if not settings.get("display_as_level_change", False):
        return forecast
    scale = float(settings.get("target_scale", 1.0))
    return float(latest_target.iloc[-1]) * forecast / scale


def _ridge_prediction(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.Series | pd.DataFrame,
    ridge_penalty: float,
) -> tuple[float, dict[str, pd.Series]]:
    x = train_x.astype(float).copy()
    medians = x.median(axis=0)
    x = x.fillna(medians)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0).replace(0.0, 1.0)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized.to_numpy()])
    penalty = np.eye(design.shape[1]) * ridge_penalty
    penalty[0, 0] = 0.0
    y = train_y.to_numpy(dtype=float)
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    # Monthly payroll changes contain extraordinary but genuine shock observations.
    # Iteratively reweighted ridge prevents one episode from determining the live
    # forecast while retaining the observation in walk-forward scoring.
    for _ in range(12):
        residual = y - design @ coefficients
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        if mad <= 1e-12:
            break
        threshold = 1.5 * 1.4826 * mad
        weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual - center), 1e-12))
        weighted = design * weights[:, None]
        updated = np.linalg.solve(design.T @ weighted + penalty, design.T @ (weights * y))
        if np.max(np.abs(updated - coefficients)) < 1e-8:
            coefficients = updated
            break
        coefficients = updated
    test = test_x.iloc[0] if isinstance(test_x, pd.DataFrame) else test_x
    test = test.reindex(train_x.columns).astype(float).fillna(medians)
    standardized_test = (test - mean) / scale
    contributions = standardized_test * coefficients[1:]
    prediction = float(coefficients[0] + contributions.sum())
    return prediction, {
        "coefficients": pd.Series(coefficients[1:], index=train_x.columns, dtype=float),
        "standardized_test": standardized_test,
        "contributions": contributions,
    }


def _forecast_record(
    test: pd.Series,
    target_id: str,
    variant_id: str,
    target: dict[str, Any],
    prediction: float,
    baseline: float,
    train_observations: int,
) -> dict[str, Any]:
    reference = (
        0.0
        if str(target.get("direction_reference", "prior")) == "zero"
        else float(test["prior_target"])
    )
    predicted_direction = np.sign(prediction - reference)
    actual_direction = np.sign(float(test["target"]) - reference)
    return {
        "target_id": target_id,
        "target_label": str(target.get("label", target_id)),
        "variant_id": variant_id,
        "observation_date": test["observation_date"],
        "release_date": test["release_date"],
        "actual": float(test["target"]),
        "forecast": prediction,
        "baseline_forecast": baseline,
        "prior_target": float(test["prior_target"]),
        "direction_correct": bool(predicted_direction == actual_direction),
        "unit": str(target.get("unit", "")),
        "train_observations": train_observations,
    }
