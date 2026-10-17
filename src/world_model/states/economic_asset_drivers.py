from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.data.csi import read_price_series
from world_model.states.economic_triggers import EconomicTriggerValidator, _bucket_for_value


@dataclass(frozen=True)
class EconomicAssetDriverResult:
    trigger_panel: pd.DataFrame
    asset_group_returns: pd.DataFrame
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    return_method: str


class EconomicAssetDriverValidator:
    def __init__(
        self,
        data_roots: dict[str, str],
        trigger_config: dict[str, Any],
        asset_driver_config: dict[str, Any],
    ):
        self.data_roots = data_roots
        self.trigger_config = trigger_config
        self.asset_driver_config = asset_driver_config

    def run(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> EconomicAssetDriverResult:
        trigger_panel = EconomicTriggerValidator(
            data_roots=self.data_roots,
            economic_config={"components": {}},
            trigger_config=self.trigger_config,
        ).build_trigger_panel()
        asset_group_returns = load_asset_group_returns(
            root=self.data_roots["csi_processed"],
            asset_groups=self.asset_driver_config.get("asset_groups", {}),
            return_method=str(self.asset_driver_config.get("return_method", "normalized_price_change")),
            normalization_window=int(self.asset_driver_config.get("normalization_window", 252)),
            normalization_min_periods=int(self.asset_driver_config.get("normalization_min_periods", 63)),
        )
        forecasts = self._run_forecasts(trigger_panel, asset_group_returns, start=start, end=end)
        summary = summarize_asset_driver_forecasts(forecasts)
        return EconomicAssetDriverResult(
            trigger_panel=trigger_panel,
            asset_group_returns=asset_group_returns,
            forecasts=forecasts,
            summary=summary,
            return_method=str(self.asset_driver_config.get("return_method", "normalized_price_change")),
        )

    def _run_forecasts(
        self,
        trigger_panel: pd.DataFrame,
        asset_group_returns: pd.DataFrame,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        horizons = {key: int(value) for key, value in self.asset_driver_config.get("horizons", {}).items()}
        if not horizons:
            raise ValueError("Asset driver config must define at least one horizon.")

        common_index = trigger_panel.index.intersection(asset_group_returns.dropna(how="all").index)
        eval_dates = evaluation_dates(
            common_index=common_index,
            max_horizon=max(horizons.values()),
            frequency=str(self.asset_driver_config.get("evaluation_frequency", "ME")),
            start=start,
            end=end,
        )
        eval_features = trigger_panel.reindex(eval_dates).ffill()
        feature_specs = build_asset_driver_feature_specs(self.trigger_config, self.asset_driver_config)
        min_train = int(self.asset_driver_config.get("min_train_observations", 60))
        min_bucket = int(self.asset_driver_config.get("min_bucket_observations", 15))

        group_labels = {
            group: settings.get("label", group)
            for group, settings in self.asset_driver_config.get("asset_groups", {}).items()
        }
        asset_driver_groups = {
            group: set(settings.get("driver_groups", []))
            for group, settings in self.asset_driver_config.get("asset_groups", {}).items()
        }
        for horizon, trading_days in horizons.items():
            forward_returns = forward_cumulative_returns(asset_group_returns, trading_days).reindex(eval_dates)
            for group_name in forward_returns.columns:
                target_values = forward_returns[group_name]
                contexts = walk_forward_contexts(
                    eval_dates=eval_dates,
                    target_values=target_values,
                    horizon_days=trading_days,
                    min_train=min_train,
                )
                for feature_name, feature_settings in feature_specs.items():
                    if feature_name not in eval_features:
                        continue
                    driver_group = str(feature_settings.get("driver_group", "unknown"))
                    configured_driver_groups = asset_driver_groups.get(group_name, set())
                    matches_asset_prior = (
                        driver_group == "aggregate"
                        or not configured_driver_groups
                        or driver_group in configured_driver_groups
                    )
                    feature_values = eval_features[feature_name]
                    for context in contexts:
                        current_feature = feature_values.iloc[context["eval_pos"]]
                        if pd.isna(current_feature):
                            continue
                        train_feature = feature_values.iloc[: context["train_end_pos"] + 1]
                        train_target = target_values.iloc[: context["train_end_pos"] + 1]
                        train_sample = pd.concat(
                            [train_feature.rename("feature"), train_target.rename("target")],
                            axis=1,
                        ).dropna()
                        if len(train_sample) < min_train:
                            continue
                        forecast = conditional_mean_return(
                            train_feature=train_sample["feature"],
                            train_target=train_sample["target"],
                            current_feature=float(current_feature),
                            kind=feature_settings["kind"],
                            min_bucket_observations=min_bucket,
                        )
                        rows.append(
                            {
                                "forecast_date": context["forecast_date"],
                                "asset_group": group_name,
                                "asset_group_label": group_labels.get(group_name, group_name),
                                "horizon": horizon,
                                "trading_days": trading_days,
                                "feature": feature_name,
                                "feature_label": feature_settings["label"],
                                "feature_kind": feature_settings["kind"],
                                "driver_group": driver_group,
                                "matches_asset_prior": matches_asset_prior,
                                "feature_value": float(current_feature),
                                "forecast_return": forecast,
                                "baseline_return": context["baseline"],
                                "realized_return": context["realized"],
                                "train_observations": int(len(train_sample)),
                            }
                        )

        forecasts = pd.DataFrame(rows)
        if forecasts.empty:
            raise ValueError("Economic asset driver validation produced no forecasts.")
        forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
        return forecasts.sort_values(["asset_group", "horizon", "feature", "forecast_date"]).reset_index(drop=True)


def load_asset_group_returns(
    root: str | Path,
    asset_groups: dict[str, Any],
    return_method: str = "normalized_price_change",
    normalization_window: int = 252,
    normalization_min_periods: int = 63,
) -> pd.DataFrame:
    group_returns: dict[str, pd.Series] = {}
    for group_name, settings in asset_groups.items():
        asset_returns: list[pd.Series] = []
        for symbol, filename in settings.get("assets", {}).items():
            price = read_price_series(root, filename)
            returns = asset_return_proxy(
                price=price,
                method=return_method,
                normalization_window=normalization_window,
                normalization_min_periods=normalization_min_periods,
            )
            asset_returns.append(returns.rename(symbol))
        if not asset_returns:
            continue
        group_frame = pd.concat(asset_returns, axis=1).sort_index()
        group_returns[group_name] = group_frame.mean(axis=1, skipna=True).rename(group_name)
    if not group_returns:
        raise ValueError("No asset-group returns could be loaded.")
    return pd.concat(group_returns.values(), axis=1).sort_index()


def asset_return_proxy(
    price: pd.Series,
    method: str = "normalized_price_change",
    normalization_window: int = 252,
    normalization_min_periods: int = 63,
) -> pd.Series:
    price = price.sort_index().astype(float)
    if method == "normalized_price_change":
        scale = price.abs().rolling(normalization_window, min_periods=normalization_min_periods).median().shift(1)
        returns = price.diff() / scale.where(scale > 0.0)
        return returns.replace([np.inf, -np.inf], np.nan)
    if method == "log":
        positive_price = price.where(price > 0.0)
        returns = np.log(positive_price / positive_price.shift(1))
        return returns.replace([np.inf, -np.inf], np.nan)
    if method == "pct_change":
        positive_price = price.where(price > 0.0)
        return positive_price.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    raise ValueError(f"Unsupported asset return method: {method}")


def forward_cumulative_returns(returns: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    cumulative = returns.fillna(0.0).cumsum()
    forward = cumulative.shift(-horizon_days) - cumulative
    future_valid_count = (
        returns.notna()
        .shift(-1)
        .rolling(horizon_days, min_periods=horizon_days)
        .sum()
        .shift(-(horizon_days - 1))
    )
    return forward.where(future_valid_count.eq(horizon_days))


def forward_log_returns(returns: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    return forward_cumulative_returns(returns, horizon_days)


def build_asset_driver_feature_specs(
    trigger_config: dict[str, Any],
    asset_driver_config: dict[str, Any],
) -> dict[str, dict[str, str]]:
    settings = asset_driver_config.get("features", {})
    specs: dict[str, dict[str, str]] = {}
    if settings.get("include_aggregate_counts", True):
        specs.update(
            {
                "failed_count": {"label": "Failed trigger count", "kind": "continuous"},
                "failed_share": {"label": "Failed trigger share", "kind": "continuous"},
                "healthy_count": {"label": "Healthy trigger count", "kind": "continuous"},
                "healthy_share": {"label": "Healthy trigger share", "kind": "continuous"},
                "legacy_econ_score": {"label": "Legacy healthy trigger count", "kind": "continuous"},
                "legacy_econ_failed": {"label": "Legacy economic score below trigger", "kind": "binary"},
            }
        )
        for spec in specs.values():
            spec["driver_group"] = "aggregate"
    if settings.get("include_group_counts", True):
        groups = sorted({trigger.get("group", "ungrouped") for trigger in trigger_config.get("triggers", [])})
        for group in groups:
            specs[f"{group}_failed_count"] = {
                "label": f"{group} failed count",
                "kind": "continuous",
                "driver_group": group,
            }
            specs[f"{group}_healthy_count"] = {
                "label": f"{group} healthy count",
                "kind": "continuous",
                "driver_group": group,
            }
    for trigger in trigger_config.get("triggers", []):
        trigger_id = trigger["id"]
        label = trigger.get("label", trigger_id)
        driver_group = trigger.get("group", "ungrouped")
        if settings.get("include_trigger_failed", True):
            specs[f"{trigger_id}_failed"] = {
                "label": f"{label} failed",
                "kind": "binary",
                "driver_group": driver_group,
            }
        if settings.get("include_trigger_margins", True):
            specs[f"{trigger_id}_margin"] = {
                "label": f"{label} threshold margin",
                "kind": "continuous",
                "driver_group": driver_group,
            }
    return specs


def evaluation_dates(
    common_index: pd.DatetimeIndex,
    max_horizon: int,
    frequency: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    usable_index = pd.DatetimeIndex(common_index).sort_values()
    if usable_index.empty:
        raise ValueError("No common index available for economic asset driver validation.")
    start_ts = pd.Timestamp(start).normalize() if start is not None else usable_index.min()
    end_ts = pd.Timestamp(end).normalize() if end is not None else usable_index.max() - pd.tseries.offsets.BDay(max_horizon)
    candidate_dates = pd.date_range(start_ts, end_ts, freq=frequency)
    selected: list[pd.Timestamp] = []
    for candidate in candidate_dates:
        prior = usable_index[usable_index <= candidate]
        if not prior.empty:
            selected.append(pd.Timestamp(prior[-1]))
    return pd.DatetimeIndex(sorted(set(selected)))


def walk_forward_contexts(
    eval_dates: pd.DatetimeIndex,
    target_values: pd.Series,
    horizon_days: int,
    min_train: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for eval_pos, forecast_date in enumerate(eval_dates):
        realized = target_values.iloc[eval_pos]
        if pd.isna(realized):
            continue
        train_cutoff = forecast_date - pd.tseries.offsets.BDay(horizon_days)
        train_end_pos = int(eval_dates.searchsorted(train_cutoff, side="right")) - 1
        if train_end_pos < 0:
            continue
        train_target = target_values.iloc[: train_end_pos + 1].dropna()
        if len(train_target) < min_train:
            continue
        contexts.append(
            {
                "forecast_date": forecast_date,
                "eval_pos": eval_pos,
                "train_end_pos": train_end_pos,
                "baseline": float(train_target.mean()),
                "realized": float(realized),
            }
        )
    return contexts


def conditional_mean_return(
    train_feature: pd.Series,
    train_target: pd.Series,
    current_feature: float,
    kind: str,
    min_bucket_observations: int,
    buckets: int = 5,
) -> float:
    sample = pd.concat([train_feature.rename("feature"), train_target.rename("target")], axis=1).dropna()
    if sample.empty:
        return float("nan")
    baseline = float(sample["target"].mean())
    if kind == "binary":
        bucket = sample[sample["feature"].ge(0.5) == (current_feature >= 0.5)]
        return float(bucket["target"].mean()) if len(bucket) >= min_bucket_observations else baseline
    if sample["feature"].nunique() < 2:
        return baseline
    sample["bucket"] = pd.qcut(sample["feature"], buckets, labels=False, duplicates="drop")
    bucket_edges = sample.groupby("bucket", observed=True)["feature"].agg(["min", "max"])
    current_bucket = _bucket_for_value(bucket_edges, current_feature)
    bucket_sample = sample[sample["bucket"].eq(current_bucket)]
    return float(bucket_sample["target"].mean()) if len(bucket_sample) >= min_bucket_observations else baseline


def summarize_asset_driver_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["asset_group", "horizon", "feature"]
    for (asset_group, horizon, feature), group in forecasts.groupby(group_columns, sort=False):
        clean = group.dropna(subset=["forecast_return", "baseline_return", "realized_return", "feature_value"])
        if clean.empty:
            continue
        feature_kind = str(clean["feature_kind"].iloc[0])
        model_error = clean["forecast_return"] - clean["realized_return"]
        baseline_error = clean["baseline_return"] - clean["realized_return"]
        active_rate = np.nan
        active_forward_return = np.nan
        inactive_forward_return = np.nan
        active_inactive_spread = np.nan
        if feature_kind == "binary":
            active = clean[clean["feature_value"] >= 0.5]
            inactive = clean[clean["feature_value"] < 0.5]
            active_rate = float(len(active) / len(clean))
            if not active.empty:
                active_forward_return = float(active["realized_return"].mean())
            if not inactive.empty:
                inactive_forward_return = float(inactive["realized_return"].mean())
            if pd.notna(active_forward_return) and pd.notna(inactive_forward_return):
                active_inactive_spread = active_forward_return - inactive_forward_return
        rows.append(
            {
                "asset_group": asset_group,
                "asset_group_label": clean["asset_group_label"].iloc[0],
                "horizon": horizon,
                "trading_days": int(clean["trading_days"].iloc[0]),
                "feature": feature,
                "feature_label": clean["feature_label"].iloc[0],
                "feature_kind": feature_kind,
                "driver_group": clean["driver_group"].iloc[0],
                "matches_asset_prior": bool(clean["matches_asset_prior"].iloc[0]),
                "observations": int(len(clean)),
                "realized_mean_return": float(clean["realized_return"].mean()),
                "forecast_mean_return": float(clean["forecast_return"].mean()),
                "baseline_mean_return": float(clean["baseline_return"].mean()),
                "forecast_realized_corr": _safe_corr(clean["forecast_return"], clean["realized_return"]),
                "feature_realized_corr": _safe_corr(clean["feature_value"], clean["realized_return"]),
                "q5_q1_realized_spread": _top_bottom_spread(clean, "feature_value"),
                "active_rate": active_rate,
                "active_forward_return": active_forward_return,
                "inactive_forward_return": inactive_forward_return,
                "active_inactive_spread": active_inactive_spread,
                "model_mse": float(np.mean(model_error.pow(2))),
                "baseline_mse": float(np.mean(baseline_error.pow(2))),
                "mse_improvement_vs_baseline": float(np.mean(baseline_error.pow(2)) - np.mean(model_error.pow(2))),
                "directional_hit_rate": float((np.sign(clean["forecast_return"]) == np.sign(clean["realized_return"])).mean()),
                "baseline_directional_hit_rate": float(
                    (np.sign(clean["baseline_return"]) == np.sign(clean["realized_return"])).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["asset_group", "trading_days", "feature"]).reset_index(drop=True)


def write_economic_asset_driver_outputs(result: EconomicAssetDriverResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trigger_panel": root / "economic_asset_driver_trigger_panel.parquet",
        "asset_group_returns": root / "economic_asset_driver_group_returns.parquet",
        "forecasts": root / "economic_asset_driver_forecasts.parquet",
        "summary": root / "economic_asset_driver_summary.csv",
    }
    result.trigger_panel.to_parquet(paths["trigger_panel"])
    result.asset_group_returns.to_parquet(paths["asset_group_returns"])
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    return paths


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    sample = pd.concat([left, right], axis=1).dropna()
    if len(sample) < 3 or sample.iloc[:, 0].nunique() < 2 or sample.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(sample.iloc[:, 0].corr(sample.iloc[:, 1]))


def _top_bottom_spread(clean: pd.DataFrame, column: str) -> float:
    if clean[column].nunique() < 5:
        return float("nan")
    bucketed = clean.copy()
    bucketed["_quintile"] = pd.qcut(bucketed[column], 5, labels=False, duplicates="drop")
    realized_by_quintile = bucketed.groupby("_quintile", observed=True)["realized_return"].mean()
    if len(realized_by_quintile) < 2:
        return float("nan")
    return float(realized_by_quintile.iloc[-1] - realized_by_quintile.iloc[0])
