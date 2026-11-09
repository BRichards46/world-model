from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from world_model.data.csi import read_price_series
from world_model.data.norgate import read_catalog_price_series
from world_model.features.directional_momentum import classify_bias, normalized_futures_change


@dataclass(frozen=True)
class MomentumValidationResult:
    signals: pd.DataFrame
    events: pd.DataFrame
    baselines: pd.DataFrame
    summary: pd.DataFrame


class DirectionalMomentumValidator:
    def __init__(
        self,
        data_roots: dict[str, str],
        momentum_config: dict[str, Any],
        validation_config: dict[str, Any],
    ):
        self.data_roots = data_roots
        self.momentum_config = momentum_config
        self.validation_config = validation_config

    def run(self, momentum_history: pd.DataFrame) -> MomentumValidationResult:
        signals, evaluation_dates = build_signal_panel(
            momentum_history,
            self.momentum_config,
            self.validation_config,
        )
        price_inputs = self._load_price_inputs()
        events = build_excursion_events(
            signals,
            price_inputs,
            self.validation_config["forward_horizons"],
        )
        baselines = build_baseline_events(
            evaluation_dates,
            price_inputs,
            self.validation_config["forward_horizons"],
        )
        summary = summarize_excursion_value(events, baselines, self.validation_config)
        return MomentumValidationResult(signals, events, baselines, summary)

    def _load_price_inputs(self) -> dict[tuple[str, str], dict[str, Any]]:
        inputs: dict[tuple[str, str], dict[str, Any]] = {}
        for asset in self.momentum_config["futures"]:
            price = read_price_series(self.data_roots["csi_processed"], asset["file"])
            scale = price.abs().rolling(252, min_periods=63).median().shift(1)
            daily = normalized_futures_change(price)
            volatility = daily.rolling(63, min_periods=42).std().shift(1)
            inputs[(str(asset["id"]), "absolute")] = {
                "price": price,
                "scale": scale,
                "volatility": volatility,
                "mode": "futures",
            }

        equity = self.momentum_config["equities"]
        benchmark_id = str(equity["benchmark"]["id"])
        benchmark = read_catalog_price_series(
            self.data_roots["norgate_prices"],
            self.data_roots["norgate_metadata"],
            benchmark_id,
        )
        for asset in equity["assets"]:
            asset_id = str(asset["id"])
            price = read_catalog_price_series(
                self.data_roots["norgate_prices"],
                self.data_roots["norgate_metadata"],
                asset_id,
            )
            aligned = pd.concat([price.rename("asset"), benchmark.rename("benchmark")], axis=1, join="inner").dropna()
            asset_log = np.log(aligned["asset"] / aligned["asset"].shift(1))
            asset_vol = asset_log.rolling(63, min_periods=42).std().shift(1)
            ratio = aligned["asset"] / aligned["benchmark"]
            relative_log = np.log(ratio / ratio.shift(1))
            relative_vol = relative_log.rolling(63, min_periods=42).std().shift(1)
            inputs[(asset_id, "absolute")] = {
                "price": aligned["asset"],
                "volatility": asset_vol,
                "mode": "equity",
            }
            inputs[(asset_id, "relative")] = {
                "price": ratio,
                "volatility": relative_vol,
                "mode": "equity",
            }
        for price_input in inputs.values():
            prepare_price_input(price_input)
        return inputs


def prepare_price_input(price_input: dict[str, Any]) -> dict[str, Any]:
    """Cache aligned NumPy arrays used repeatedly by the excursion study."""
    price = price_input["price"].dropna().sort_index()
    index = pd.DatetimeIndex(price.index)
    price_input["price"] = price
    price_input["_index_values"] = index.asi8
    price_input["_price_values"] = price.to_numpy(dtype=float)
    price_input["_volatility_values"] = (
        price_input["volatility"].reindex(index, method="ffill").to_numpy(dtype=float)
    )
    if price_input["mode"] == "futures":
        price_input["_scale_values"] = price_input["scale"].reindex(index, method="ffill").to_numpy(dtype=float)
    return price_input


def build_signal_panel(
    history: pd.DataFrame,
    momentum_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frequency = str(validation_config.get("evaluation_frequency", "W-FRI"))
    frame["evaluation_period"] = frame["date"].dt.to_period(frequency)
    latest = frame.groupby(["asset_id", "evaluation_period"])["date"].transform("max")
    sampled = frame.loc[frame["date"].eq(latest)].drop(columns="evaluation_period")
    threshold = float(validation_config.get("component_signal_threshold", 0.25))
    classification = momentum_config["classification"]
    signals: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []

    for (asset_id, date), rows in sampled.groupby(["asset_id", "date"], sort=False):
        asset_class = str(rows.iloc[0]["asset_class"])
        lenses = ["absolute", "relative"] if asset_class == "sector_etf" else ["absolute"]
        for lens in lenses:
            evaluation_rows.append(
                {"asset_id": str(asset_id), "asset_class": asset_class, "lens": lens, "signal_date": date}
            )
            relative = lens == "relative"
            bias = classify_bias(rows, classification, relative=relative)
            direction = _bias_direction(str(bias["id"]))
            if direction:
                signals.append(
                    _signal_record(
                        asset_id,
                        asset_class,
                        lens,
                        date,
                        "composite",
                        "composite",
                        direction,
                        bias["score"],
                        "strong" if str(bias["id"]).startswith("strong_") else "standard",
                    )
                )

            raw_column = "relative_raw_momentum" if relative else "raw_momentum"
            adjusted_column = "relative_vol_adjusted_momentum" if relative else "vol_adjusted_momentum"
            for row in rows.itertuples(index=False):
                raw = getattr(row, raw_column)
                adjusted = getattr(row, adjusted_column)
                if pd.isna(raw) or pd.isna(adjusted) or abs(float(adjusted)) < threshold:
                    continue
                if np.sign(float(raw)) != np.sign(float(adjusted)):
                    continue
                signals.append(
                    _signal_record(
                        asset_id,
                        asset_class,
                        lens,
                        date,
                        "component",
                        str(row.horizon_label),
                        1 if adjusted > 0 else -1,
                        float(adjusted),
                        "strong" if abs(float(adjusted)) >= 1.0 else "standard",
                    )
                )

    signal_frame = pd.DataFrame(signals)
    evaluation_frame = pd.DataFrame(evaluation_rows).drop_duplicates()
    return signal_frame, evaluation_frame


def calculate_excursion(
    price_input: dict[str, Any],
    signal_date: pd.Timestamp,
    direction: int,
    forward_days: int,
) -> dict[str, float] | None:
    if "_price_values" not in price_input:
        prepare_price_input(price_input)
    index_values = price_input["_index_values"]
    price_values = price_input["_price_values"]
    signal_value = pd.Timestamp(signal_date).value
    location = int(np.searchsorted(index_values, signal_value, side="right") - 1)
    if location < 0 or location + forward_days >= len(price_values):
        return None
    start = float(price_values[location])
    future = price_values[location + 1 : location + forward_days + 1]
    if future.size != forward_days:
        return None

    if price_input["mode"] == "futures":
        scale = float(price_input["_scale_values"][location])
        if not np.isfinite(scale) or scale <= 0:
            return None
        path = (future - start) / scale
    else:
        if start <= 0 or np.any(future <= 0):
            return None
        path = future / start - 1.0

    volatility = float(price_input["_volatility_values"][location])
    if not np.isfinite(volatility) or volatility <= 0:
        return None
    signed_path = float(direction) * path
    mfe = max(0.0, float(np.max(signed_path)))
    mae = max(0.0, float(np.max(-signed_path)))
    terminal = float(signed_path[-1])
    risk_scale = volatility * math.sqrt(forward_days)
    return {
        "mfe": mfe,
        "mae": mae,
        "terminal": terminal,
        "edge": mfe - mae,
        "mfe_risk": mfe / risk_scale,
        "mae_risk": mae / risk_scale,
        "terminal_risk": terminal / risk_scale,
        "edge_risk": (mfe - mae) / risk_scale,
    }


def build_excursion_events(
    signals: pd.DataFrame,
    price_inputs: dict[tuple[str, str], dict[str, Any]],
    forward_horizons: list[dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for signal in signals.itertuples(index=False):
        price_input = price_inputs.get((str(signal.asset_id), str(signal.lens)))
        if price_input is None:
            continue
        for horizon in forward_horizons:
            outcome = calculate_excursion(
                price_input,
                pd.Timestamp(signal.signal_date),
                int(signal.direction),
                int(horizon["days"]),
            )
            if outcome is None:
                continue
            records.append(
                {
                    **signal._asdict(),
                    "forward_id": str(horizon["id"]),
                    "forward_label": str(horizon["label"]),
                    "forward_days": int(horizon["days"]),
                    **outcome,
                }
            )
    return pd.DataFrame(records)


def build_baseline_events(
    evaluation_dates: pd.DataFrame,
    price_inputs: dict[tuple[str, str], dict[str, Any]],
    forward_horizons: list[dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in evaluation_dates.itertuples(index=False):
        price_input = price_inputs.get((str(row.asset_id), str(row.lens)))
        if price_input is None:
            continue
        for direction in (-1, 1):
            for horizon in forward_horizons:
                outcome = calculate_excursion(
                    price_input,
                    pd.Timestamp(row.signal_date),
                    direction,
                    int(horizon["days"]),
                )
                if outcome is None:
                    continue
                records.append(
                    {
                        **row._asdict(),
                        "direction": direction,
                        "forward_id": str(horizon["id"]),
                        "forward_label": str(horizon["label"]),
                        "forward_days": int(horizon["days"]),
                        **outcome,
                    }
                )
    return pd.DataFrame(records)


def summarize_excursion_value(
    events: pd.DataFrame,
    baselines: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    group_columns = [
        "asset_id",
        "asset_class",
        "lens",
        "signal_type",
        "lookback",
        "direction",
        "forward_id",
        "forward_label",
        "forward_days",
    ]
    records: list[dict[str, Any]] = []
    minimum = int(config.get("minimum_sample_size", 40))
    for period in config.get("periods", [{"id": "full", "label": "Full history", "start": None}]):
        start = pd.Timestamp(period["start"]) if period.get("start") else None
        conditional = events if start is None else events.loc[pd.to_datetime(events["signal_date"]) >= start]
        baseline_period = baselines if start is None else baselines.loc[pd.to_datetime(baselines["signal_date"]) >= start]
        baseline_groups = {
            (str(asset_id), str(lens), int(direction), str(forward_id)): group
            for (asset_id, lens, direction, forward_id), group in baseline_period.groupby(
                ["asset_id", "lens", "direction", "forward_id"], sort=False
            )
        }
        for keys, group in conditional.groupby(group_columns, sort=False):
            key = dict(zip(group_columns, keys))
            baseline = baseline_groups.get(
                (
                    str(key["asset_id"]),
                    str(key["lens"]),
                    int(key["direction"]),
                    str(key["forward_id"]),
                )
            )
            if baseline is None or baseline.empty:
                continue
            conditional_metrics = _metrics(group)
            baseline_metrics = _metrics(baseline)
            record = {
                **key,
                "period_id": str(period["id"]),
                "period_label": str(period["label"]),
                "sample_size": int(len(group)),
                "baseline_size": int(len(baseline)),
                **conditional_metrics,
                **{f"baseline_{name}": value for name, value in baseline_metrics.items()},
            }
            record["mfe_uplift_risk"] = record["median_mfe_risk"] - record["baseline_median_mfe_risk"]
            record["mae_reduction_risk"] = record["baseline_median_mae_risk"] - record["median_mae_risk"]
            record["edge_uplift_risk"] = record["median_edge_risk"] - record["baseline_median_edge_risk"]
            record["robust_excursion_improvement"] = bool(
                len(group) >= minimum
                and record["mfe_uplift_risk"] > 0
                and record["mae_reduction_risk"] > 0
                and record["edge_uplift_risk"] > 0
            )
            records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    expected_periods = len(config.get("periods", [])) or 1
    stability_keys = [column for column in group_columns if column not in {"forward_label", "forward_days"}]
    result["stable_across_periods"] = result.groupby(stability_keys)[
        "robust_excursion_improvement"
    ].transform(lambda values: len(values) == expected_periods and bool(values.all()))
    return result


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "median_mfe": float(frame["mfe"].median()),
        "median_mae": float(frame["mae"].median()),
        "median_edge": float(frame["edge"].median()),
        "median_mfe_risk": float(frame["mfe_risk"].median()),
        "median_mae_risk": float(frame["mae_risk"].median()),
        "median_edge_risk": float(frame["edge_risk"].median()),
        "terminal_win_rate": float((frame["terminal"] > 0).mean()),
        "excursion_win_rate": float((frame["mfe"] > frame["mae"]).mean()),
    }


def _bias_direction(bias_id: str) -> int:
    if bias_id in {"positive", "strong_positive"}:
        return 1
    if bias_id in {"negative", "strong_negative"}:
        return -1
    return 0


def _signal_record(
    asset_id: Any,
    asset_class: str,
    lens: str,
    date: pd.Timestamp,
    signal_type: str,
    lookback: str,
    direction: int,
    score: Any,
    strength: str,
) -> dict[str, Any]:
    return {
        "asset_id": str(asset_id),
        "asset_class": asset_class,
        "lens": lens,
        "signal_date": pd.Timestamp(date).normalize(),
        "signal_type": signal_type,
        "lookback": lookback,
        "direction": int(direction),
        "direction_label": "positive" if direction > 0 else "negative",
        "signal_score": float(score),
        "signal_strength": strength,
    }
