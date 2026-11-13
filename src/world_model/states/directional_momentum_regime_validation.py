from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from world_model.states.directional_momentum_validation import (
    DirectionalMomentumValidator,
    _metrics,
    calculate_excursion,
)


@dataclass(frozen=True)
class MomentumRegimeValidationResult:
    regimes: pd.DataFrame
    events: pd.DataFrame
    summary: pd.DataFrame


class DirectionalMomentumRegimeValidator:
    """Test the incremental excursion value of matching-horizon cross-asset states."""

    def __init__(
        self,
        data_roots: dict[str, str],
        momentum_config: dict[str, Any],
        regime_config: dict[str, Any],
    ):
        self.momentum_config = momentum_config
        self.regime_config = regime_config
        self.price_inputs = DirectionalMomentumValidator(
            data_roots,
            momentum_config,
            regime_config,
        )._load_price_inputs()

    def run(self, momentum_history: pd.DataFrame) -> MomentumRegimeValidationResult:
        regimes = build_matching_horizon_regimes(momentum_history, self.regime_config)
        events = build_regime_excursion_events(
            regimes,
            self.price_inputs,
            self.regime_config["forward_horizons"],
        )
        summary = summarize_regime_excursion_value(events, self.regime_config)
        return MomentumRegimeValidationResult(regimes, events, summary)


def build_matching_horizon_regimes(
    history: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    assets = [str(asset) for asset in config["assets"]]
    frame = history.loc[history["asset_id"].astype(str).isin(assets)].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["evaluation_period"] = frame["date"].dt.to_period(
        str(config.get("evaluation_frequency", "W-FRI"))
    )
    latest = frame.groupby(["asset_id", "horizon_id", "evaluation_period"])["date"].transform("max")
    frame = frame.loc[frame["date"].eq(latest)].copy()
    threshold = float(config.get("component_signal_threshold", 0.25))
    frame["state"] = [
        component_state(raw, adjusted, threshold)
        for raw, adjusted in zip(frame["raw_momentum"], frame["vol_adjusted_momentum"])
    ]
    states = frame.pivot_table(
        index=["evaluation_period", "horizon_id", "horizon_label", "horizon_days"],
        columns="asset_id",
        values="state",
        aggfunc="last",
    )
    dates = frame.groupby(
        ["evaluation_period", "horizon_id", "horizon_label", "horizon_days"]
    )["date"].max()
    result = states.join(dates.rename("signal_date")).reset_index()
    if any(asset not in result.columns for asset in assets):
        return pd.DataFrame()
    result = result.dropna(subset=assets)
    result["regime_id"] = result.apply(
        lambda row: " / ".join(f"{asset}{state_symbol(row[asset])}" for asset in assets),
        axis=1,
    )
    return result.drop(columns="evaluation_period").sort_values(["signal_date", "horizon_days"])


def component_state(raw: Any, adjusted: Any, threshold: float) -> str:
    if pd.isna(raw) or pd.isna(adjusted):
        return "neutral"
    raw_value = float(raw)
    adjusted_value = float(adjusted)
    if abs(adjusted_value) < threshold or np.sign(raw_value) != np.sign(adjusted_value):
        return "neutral"
    if adjusted_value > 0:
        return "positive"
    if adjusted_value < 0:
        return "negative"
    return "neutral"


def state_symbol(state: str) -> str:
    return {"positive": "+", "negative": "-", "neutral": "0"}[str(state)]


def build_regime_excursion_events(
    regimes: pd.DataFrame,
    price_inputs: dict[tuple[str, str], dict[str, Any]],
    forward_horizons: list[dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    asset_columns = [column for column in ("ES", "NQ", "DX", "ZB") if column in regimes.columns]
    for row in regimes.itertuples(index=False):
        for target in asset_columns:
            target_state = str(getattr(row, target))
            if target_state == "neutral":
                continue
            direction = 1 if target_state == "positive" else -1
            price_input = price_inputs.get((target, "absolute"))
            if price_input is None:
                continue
            for forward in forward_horizons:
                outcome = calculate_excursion(
                    price_input,
                    pd.Timestamp(row.signal_date),
                    direction,
                    int(forward["days"]),
                )
                if outcome is None:
                    continue
                records.append(
                    {
                        "signal_date": pd.Timestamp(row.signal_date),
                        "signal_horizon_id": str(row.horizon_id),
                        "signal_horizon_label": str(row.horizon_label),
                        "signal_horizon_days": int(row.horizon_days),
                        "regime_id": str(row.regime_id),
                        "ES": str(row.ES),
                        "NQ": str(row.NQ),
                        "DX": str(row.DX),
                        "ZB": str(row.ZB),
                        "target_asset": target,
                        "target_state": target_state,
                        "direction": direction,
                        "forward_id": str(forward["id"]),
                        "forward_label": str(forward["label"]),
                        "forward_days": int(forward["days"]),
                        **outcome,
                    }
                )
    return pd.DataFrame(records)


def summarize_regime_excursion_value(
    events: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    group_columns = [
        "signal_horizon_id",
        "signal_horizon_label",
        "signal_horizon_days",
        "regime_id",
        "ES",
        "NQ",
        "DX",
        "ZB",
        "target_asset",
        "target_state",
        "direction",
        "forward_id",
        "forward_label",
        "forward_days",
    ]
    baseline_columns = [
        "signal_horizon_id",
        "target_asset",
        "target_state",
        "direction",
        "forward_id",
    ]
    minimum = int(config.get("minimum_sample_size", 40))
    records: list[dict[str, Any]] = []
    periods = config.get("periods", [{"id": "full", "label": "Full history", "start": None}])
    for period in periods:
        start = pd.Timestamp(period["start"]) if period.get("start") else None
        sample = events if start is None else events.loc[events["signal_date"] >= start]
        baselines = {
            tuple(keys): group
            for keys, group in sample.groupby(baseline_columns, sort=False)
        }
        for keys, group in sample.groupby(group_columns, sort=False):
            key = dict(zip(group_columns, keys))
            baseline_key = tuple(key[column] for column in baseline_columns)
            baseline = baselines.get(baseline_key)
            if baseline is None:
                continue
            conditional_metrics = _metrics(group)
            baseline_metrics = _metrics(baseline)
            record = {
                **key,
                "period_id": str(period["id"]),
                "period_label": str(period["label"]),
                "sample_size": int(len(group)),
                "own_signal_baseline_size": int(len(baseline)),
                **conditional_metrics,
                **{f"baseline_{name}": value for name, value in baseline_metrics.items()},
            }
            record["mfe_uplift_risk"] = record["median_mfe_risk"] - record["baseline_median_mfe_risk"]
            record["mae_reduction_risk"] = record["baseline_median_mae_risk"] - record["median_mae_risk"]
            record["edge_uplift_risk"] = record["median_edge_risk"] - record["baseline_median_edge_risk"]
            record["incremental_regime_value"] = bool(
                len(group) >= minimum
                and record["mfe_uplift_risk"] > 0
                and record["mae_reduction_risk"] > 0
                and record["edge_uplift_risk"] > 0
            )
            record["incremental_regime_degradation"] = bool(
                len(group) >= minimum
                and record["mfe_uplift_risk"] < 0
                and record["mae_reduction_risk"] < 0
                and record["edge_uplift_risk"] < 0
            )
            records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    stability_keys = [
        column for column in group_columns if column not in {"signal_horizon_label", "signal_horizon_days", "forward_label", "forward_days"}
    ]
    expected_periods = len(periods)
    result["stable_across_periods"] = result.groupby(stability_keys)["incremental_regime_value"].transform(
        lambda values: len(values) == expected_periods and bool(values.all())
    )
    result["stable_degradation_across_periods"] = result.groupby(stability_keys)[
        "incremental_regime_degradation"
    ].transform(lambda values: len(values) == expected_periods and bool(values.all()))
    return result
