from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table
from world_model.features.economic import build_realtime_indicator_from_vintage
from world_model.states.economic import EconomicStateEngine
from world_model.states.market import MarketStateEngine


@dataclass(frozen=True)
class TriggerSignal:
    value: pd.Series
    observation_date: pd.Series


@dataclass(frozen=True)
class ValidationTarget:
    name: str
    label: str
    horizon: str
    trading_days: int
    target: pd.Series
    state_baseline_score: pd.Series


@dataclass(frozen=True)
class EconomicTriggerValidationResult:
    trigger_panel: pd.DataFrame
    forecasts: pd.DataFrame
    summary: pd.DataFrame
    current_snapshot: pd.DataFrame


class EconomicTriggerValidator:
    def __init__(
        self,
        data_roots: dict[str, str],
        economic_config: dict[str, Any],
        trigger_config: dict[str, Any],
        market_config: dict[str, Any] | None = None,
    ):
        self.data_roots = data_roots
        self.economic_config = economic_config
        self.trigger_config = trigger_config
        self.market_config = market_config

    @property
    def fred_root(self) -> str:
        return self.data_roots["fred"]

    def run(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> EconomicTriggerValidationResult:
        economic_engine = EconomicStateEngine(self.data_roots, self.economic_config)
        economic_state = economic_engine.build_state_history()
        trigger_panel = self.build_trigger_panel()

        market_state = None
        if self.trigger_config.get("targets", {}).get("include_market_state", False) and self.market_config is not None:
            market_state = MarketStateEngine(self.data_roots, self.market_config).build_state_history()

        targets = build_validation_targets(
            economic_state=economic_state,
            trigger_config=self.trigger_config,
            market_state=market_state,
        )
        forecasts = self._run_forecasts(trigger_panel=trigger_panel, targets=targets, start=start, end=end)
        return EconomicTriggerValidationResult(
            trigger_panel=trigger_panel,
            forecasts=forecasts,
            summary=summarize_trigger_forecasts(forecasts),
            current_snapshot=current_trigger_snapshot(trigger_panel, self.trigger_config),
        )

    def build_trigger_panel(self) -> pd.DataFrame:
        signals: dict[str, TriggerSignal] = {}
        for trigger in self.trigger_config.get("triggers", []):
            signals[trigger["id"]] = self._load_trigger_signal(trigger)

        release_index = _combined_index([signal.value.index for signal in signals.values()])
        index = pd.date_range(
            release_index.min(),
            release_index.max(),
            freq="B",
            name="date",
        )
        panel = pd.DataFrame(index=index)
        trigger_ids: list[str] = []
        groups: dict[str, list[str]] = {}

        for trigger in self.trigger_config.get("triggers", []):
            trigger_id = trigger["id"]
            trigger_ids.append(trigger_id)
            groups.setdefault(trigger.get("group", "ungrouped"), []).append(trigger_id)
            signal = signals[trigger_id]
            threshold = float(trigger["threshold"])
            operator = str(trigger["healthy_operator"])
            value = signal.value.reindex(index).ffill()
            healthy = _healthy_mask(value, operator=operator, threshold=threshold)
            available = value.notna()
            failed = (~healthy).astype(float).where(available)

            panel[f"{trigger_id}_value"] = value
            panel[f"{trigger_id}_margin"] = _health_margin(value, operator=operator, threshold=threshold)
            panel[f"{trigger_id}_healthy"] = healthy.astype(float).where(available)
            panel[f"{trigger_id}_failed"] = failed
            panel[f"{trigger_id}_observation_date"] = signal.observation_date.reindex(index).ffill()

        healthy_columns = [f"{trigger_id}_healthy" for trigger_id in trigger_ids]
        failed_columns = [f"{trigger_id}_failed" for trigger_id in trigger_ids]
        panel["trigger_available_count"] = panel[healthy_columns].notna().sum(axis=1)
        panel["healthy_count"] = panel[healthy_columns].sum(axis=1, min_count=1)
        panel["failed_count"] = panel[failed_columns].sum(axis=1, min_count=1)
        panel["healthy_share"] = panel["healthy_count"] / panel["trigger_available_count"].replace(0, np.nan)
        panel["failed_share"] = panel["failed_count"] / panel["trigger_available_count"].replace(0, np.nan)
        health_trigger = float(self.trigger_config.get("econ_health_trigger", 10))
        panel["legacy_econ_score"] = panel["healthy_count"]
        panel["legacy_econ_healthy"] = (panel["healthy_count"] >= health_trigger).astype(float).where(panel["healthy_count"].notna())
        panel["legacy_econ_failed"] = (panel["healthy_count"] < health_trigger).astype(float).where(panel["healthy_count"].notna())

        for group, ids in groups.items():
            group_healthy = [f"{trigger_id}_healthy" for trigger_id in ids]
            group_failed = [f"{trigger_id}_failed" for trigger_id in ids]
            panel[f"{group}_healthy_count"] = panel[group_healthy].sum(axis=1, min_count=1)
            panel[f"{group}_failed_count"] = panel[group_failed].sum(axis=1, min_count=1)
        return panel

    def _run_forecasts(
        self,
        trigger_panel: pd.DataFrame,
        targets: list[ValidationTarget],
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        feature_specs = build_feature_specs(self.trigger_config)
        max_horizon = max(target.trading_days for target in targets)
        common_index = trigger_panel.index
        for target in targets:
            common_index = common_index.intersection(target.target.dropna().index)

        eval_dates = evaluation_dates(
            common_index=common_index,
            max_horizon=max_horizon,
            frequency=self.trigger_config.get("evaluation_frequency", "ME"),
            start=start,
            end=end,
        )
        min_train = int(self.trigger_config.get("min_train_observations", 60))
        min_bucket = int(self.trigger_config.get("min_bucket_observations", 15))

        for target in targets:
            target_frame = pd.concat(
                [
                    trigger_panel,
                    target.target.rename("target_value"),
                    target.state_baseline_score.rename("state_baseline_score"),
                ],
                axis=1,
            )
            eval_frame = target_frame.reindex(eval_dates)
            eval_contexts = self._target_eval_contexts(
                eval_frame=eval_frame,
                target=target,
                eval_dates=eval_dates,
                min_train=min_train,
                min_bucket=min_bucket,
            )
            for feature_name, feature_settings in feature_specs.items():
                if feature_name not in eval_frame.columns:
                    continue
                feature_values = eval_frame[feature_name]
                target_values = eval_frame["target_value"]
                for context in eval_contexts:
                    current_feature = feature_values.iloc[context["eval_pos"]]
                    if pd.isna(current_feature):
                        continue
                    train_feature = feature_values.iloc[: context["train_end_pos"] + 1]
                    train_target = target_values.iloc[: context["train_end_pos"] + 1]
                    train_sample = pd.concat([train_feature.rename("feature"), train_target.rename("target")], axis=1).dropna()
                    if len(train_sample) < min_train:
                        continue

                    forecast = conditional_probability(
                        train_feature=train_sample["feature"],
                        train_target=train_sample["target"],
                        current_feature=float(current_feature),
                        kind=feature_settings["kind"],
                        min_bucket_observations=min_bucket,
                    )

                    rows.append(
                        {
                            "forecast_date": context["forecast_date"],
                            "target": target.name,
                            "target_label": target.label,
                            "horizon": target.horizon,
                            "trading_days": target.trading_days,
                            "feature": feature_name,
                            "feature_label": feature_settings["label"],
                            "feature_kind": feature_settings["kind"],
                            "feature_value": float(current_feature),
                            "forecast_value": forecast,
                            "baseline_value": context["baseline"],
                            "state_bucket_baseline_value": context["state_baseline"],
                            "realized_value": context["realized"],
                            "train_observations": int(len(train_sample)),
                        }
                    )

        forecasts = pd.DataFrame(rows)
        if forecasts.empty:
            raise ValueError("Economic trigger validation produced no forecasts.")
        forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
        return forecasts.sort_values(["target", "horizon", "feature", "forecast_date"]).reset_index(drop=True)

    def _target_eval_contexts(
        self,
        eval_frame: pd.DataFrame,
        target: ValidationTarget,
        eval_dates: pd.DatetimeIndex,
        min_train: int,
        min_bucket: int,
    ) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        target_values = eval_frame["target_value"]
        state_values = eval_frame["state_baseline_score"]
        for eval_pos, forecast_date in enumerate(eval_dates):
            realized = target_values.iloc[eval_pos]
            current_state = state_values.iloc[eval_pos]
            if pd.isna(realized):
                continue
            train_cutoff = forecast_date - pd.tseries.offsets.BDay(target.trading_days)
            train_end_pos = int(eval_dates.searchsorted(train_cutoff, side="right")) - 1
            if train_end_pos < 0:
                continue
            train_target = target_values.iloc[: train_end_pos + 1]
            train_state = state_values.iloc[: train_end_pos + 1]
            train = pd.concat([train_target.rename("target"), train_state.rename("state")], axis=1).dropna(
                subset=["target"]
            )
            if len(train) < min_train:
                continue
            baseline = float(train["target"].mean())
            state_baseline = state_bucket_probability(
                train_state=train["state"],
                train_target=train["target"],
                current_state=float(current_state) if pd.notna(current_state) else np.nan,
                min_bucket_observations=min_bucket,
            )
            contexts.append(
                {
                    "forecast_date": forecast_date,
                    "eval_pos": eval_pos,
                    "train_end_pos": train_end_pos,
                    "baseline": baseline,
                    "state_baseline": state_baseline,
                    "realized": float(realized),
                }
            )
        return contexts

    def _load_trigger_signal(self, trigger: dict[str, Any]) -> TriggerSignal:
        transform = str(trigger["transform"])
        if transform == "ratio_yoy_pct":
            return ratio_yoy_signal(
                numerator=load_vintage_table(self.fred_root, trigger["numerator_series_id"]),
                denominator=load_vintage_table(self.fred_root, trigger["denominator_series_id"]),
            )
        if transform == "unemployment_vs_trend":
            return unemployment_vs_trend_signal(load_vintage_table(self.fred_root, trigger["series_id"]))

        result = build_realtime_indicator_from_vintage(
            load_vintage_table(self.fred_root, trigger["series_id"]),
            transform=transform,
            params=dict(trigger.get("params", {})),
        )
        return TriggerSignal(value=result.value, observation_date=result.latest_observation_date)


def build_validation_targets(
    economic_state: pd.DataFrame,
    trigger_config: dict[str, Any],
    market_state: pd.DataFrame | None = None,
) -> list[ValidationTarget]:
    horizons = trigger_config.get("horizons", {})
    settings = trigger_config.get("targets", {})
    targets: list[ValidationTarget] = []
    economic_composite = economic_state["composite_score"]
    economic_drop_threshold = float(settings.get("economic_deterioration_threshold", -0.25))
    economic_weak_threshold = float(settings.get("economic_weak_state_threshold", -0.25))

    for label, days in horizons.items():
        days = int(days)
        future_economic = economic_composite.shift(-days)
        economic_drop = (future_economic - economic_composite <= economic_drop_threshold).astype(float)
        targets.append(
            ValidationTarget(
                name=f"economic_state_deterioration_{label}",
                label=f"Economic state deterioration {label}",
                horizon=label,
                trading_days=days,
                target=economic_drop.where(future_economic.notna() & economic_composite.notna()),
                state_baseline_score=economic_composite,
            )
        )
        economic_weak = (future_economic <= economic_weak_threshold).astype(float)
        targets.append(
            ValidationTarget(
                name=f"economic_weak_state_{label}",
                label=f"Economic weak state {label}",
                horizon=label,
                trading_days=days,
                target=economic_weak.where(future_economic.notna()),
                state_baseline_score=economic_composite,
            )
        )

        if market_state is not None:
            market_composite = market_state["composite_score"]
            future_market = market_composite.shift(-days)
            market_drop_threshold = float(settings.get("market_deterioration_threshold", -0.25))
            market_drop = (future_market - market_composite <= market_drop_threshold).astype(float)
            targets.append(
                ValidationTarget(
                    name=f"market_state_deterioration_{label}",
                    label=f"Market state deterioration {label}",
                    horizon=label,
                    trading_days=days,
                    target=market_drop.where(future_market.notna() & market_composite.notna()),
                    state_baseline_score=market_composite,
                )
            )
    return targets


def build_feature_specs(trigger_config: dict[str, Any]) -> dict[str, dict[str, str]]:
    specs: dict[str, dict[str, str]] = {
        "failed_count": {"label": "Failed trigger count", "kind": "continuous"},
        "failed_share": {"label": "Failed trigger share", "kind": "continuous"},
        "healthy_count": {"label": "Healthy trigger count", "kind": "continuous"},
        "healthy_share": {"label": "Healthy trigger share", "kind": "continuous"},
        "legacy_econ_score": {"label": "Legacy healthy trigger count", "kind": "continuous"},
        "legacy_econ_failed": {"label": "Legacy economic score below trigger", "kind": "binary"},
    }
    if trigger_config.get("include_group_features", True):
        groups = sorted({trigger.get("group", "ungrouped") for trigger in trigger_config.get("triggers", [])})
        for group in groups:
            specs[f"{group}_failed_count"] = {"label": f"{group} failed count", "kind": "continuous"}
            specs[f"{group}_healthy_count"] = {"label": f"{group} healthy count", "kind": "continuous"}

    for trigger in trigger_config.get("triggers", []):
        trigger_id = trigger["id"]
        label = trigger.get("label", trigger_id)
        specs[f"{trigger_id}_failed"] = {"label": f"{label} failed", "kind": "binary"}
        if trigger_config.get("include_margin_features", False):
            specs[f"{trigger_id}_margin"] = {"label": f"{label} threshold margin", "kind": "continuous"}
    return specs


def summarize_trigger_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, horizon, feature), group in forecasts.groupby(["target", "horizon", "feature"], sort=False):
        clean = group.dropna(
            subset=["forecast_value", "baseline_value", "state_bucket_baseline_value", "realized_value", "feature_value"]
        )
        if clean.empty:
            continue
        feature_kind = str(clean["feature_kind"].iloc[0])
        model_error = clean["forecast_value"] - clean["realized_value"]
        baseline_error = clean["baseline_value"] - clean["realized_value"]
        state_error = clean["state_bucket_baseline_value"] - clean["realized_value"]
        active_rate = np.nan
        active_event_rate = np.nan
        inactive_event_rate = np.nan
        active_inactive_spread = np.nan
        odds_ratio = np.nan
        if feature_kind == "binary":
            active = clean[clean["feature_value"] >= 0.5]
            inactive = clean[clean["feature_value"] < 0.5]
            active_rate = float(len(active) / len(clean))
            if not active.empty:
                active_event_rate = float(active["realized_value"].mean())
            if not inactive.empty:
                inactive_event_rate = float(inactive["realized_value"].mean())
            if pd.notna(active_event_rate) and pd.notna(inactive_event_rate):
                active_inactive_spread = active_event_rate - inactive_event_rate
                odds_ratio = _odds_ratio(active_event_rate, inactive_event_rate)

        rows.append(
            {
                "target": target,
                "target_label": clean["target_label"].iloc[0],
                "horizon": horizon,
                "trading_days": int(clean["trading_days"].iloc[0]),
                "feature": feature,
                "feature_label": clean["feature_label"].iloc[0],
                "feature_kind": feature_kind,
                "observations": int(len(clean)),
                "realized_mean": float(clean["realized_value"].mean()),
                "forecast_mean": float(clean["forecast_value"].mean()),
                "baseline_mean": float(clean["baseline_value"].mean()),
                "state_bucket_baseline_mean": float(clean["state_bucket_baseline_value"].mean()),
                "forecast_realized_corr": _safe_corr(clean["forecast_value"], clean["realized_value"]),
                "feature_realized_corr": _safe_corr(clean["feature_value"], clean["realized_value"]),
                "q5_q1_realized_spread": _top_bottom_spread(clean, "feature_value"),
                "active_rate": active_rate,
                "active_event_rate": active_event_rate,
                "inactive_event_rate": inactive_event_rate,
                "active_inactive_spread": active_inactive_spread,
                "odds_ratio": odds_ratio,
                "model_brier": float(np.mean(model_error.pow(2))),
                "baseline_brier": float(np.mean(baseline_error.pow(2))),
                "state_bucket_brier": float(np.mean(state_error.pow(2))),
                "brier_improvement_vs_baseline": float(np.mean(baseline_error.pow(2)) - np.mean(model_error.pow(2))),
                "brier_improvement_vs_state_bucket": float(np.mean(state_error.pow(2)) - np.mean(model_error.pow(2))),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "trading_days", "feature"]).reset_index(drop=True)


def current_trigger_snapshot(trigger_panel: pd.DataFrame, trigger_config: dict[str, Any]) -> pd.DataFrame:
    row = trigger_panel.dropna(subset=["healthy_count"]).iloc[-1]
    records: list[dict[str, Any]] = []
    for trigger in trigger_config.get("triggers", []):
        trigger_id = trigger["id"]
        records.append(
            {
                "date": row.name,
                "trigger": trigger_id,
                "label": trigger.get("label", trigger_id),
                "group": trigger.get("group", "ungrouped"),
                "value": row.get(f"{trigger_id}_value"),
                "threshold": float(trigger["threshold"]),
                "healthy_operator": trigger["healthy_operator"],
                "margin": row.get(f"{trigger_id}_margin"),
                "healthy": row.get(f"{trigger_id}_healthy"),
                "failed": row.get(f"{trigger_id}_failed"),
                "observation_date": row.get(f"{trigger_id}_observation_date"),
            }
        )
    return pd.DataFrame(records)


def conditional_probability(
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


def state_bucket_probability(
    train_state: pd.Series,
    train_target: pd.Series,
    current_state: float,
    min_bucket_observations: int,
    buckets: int = 5,
) -> float:
    sample = pd.concat([train_state.rename("state"), train_target.rename("target")], axis=1).dropna()
    if sample.empty:
        return float("nan")
    baseline = float(sample["target"].mean())
    if pd.isna(current_state) or sample["state"].nunique() < 2:
        return baseline
    sample["bucket"] = pd.qcut(sample["state"], buckets, labels=False, duplicates="drop")
    bucket_edges = sample.groupby("bucket", observed=True)["state"].agg(["min", "max"])
    current_bucket = _bucket_for_value(bucket_edges, current_state)
    bucket_sample = sample[sample["bucket"].eq(current_bucket)]
    return float(bucket_sample["target"].mean()) if len(bucket_sample) >= min_bucket_observations else baseline


def ratio_yoy_signal(numerator: pd.DataFrame, denominator: pd.DataFrame) -> TriggerSignal:
    numerator_known: dict[pd.Timestamp, float] = {}
    denominator_known: dict[pd.Timestamp, float] = {}
    numerator_by_release = {pd.Timestamp(date): group for date, group in numerator.groupby("realtime_start", sort=True)}
    denominator_by_release = {pd.Timestamp(date): group for date, group in denominator.groupby("realtime_start", sort=True)}
    release_dates = sorted(set(numerator_by_release).union(denominator_by_release))
    values: list[float] = []
    observations: list[pd.Timestamp | pd.NaT] = []

    for release_date in release_dates:
        for _, row in numerator_by_release.get(release_date, pd.DataFrame()).iterrows():
            numerator_known[pd.Timestamp(row["observation_date"])] = float(row["value"]) if pd.notna(row["value"]) else np.nan
        for _, row in denominator_by_release.get(release_date, pd.DataFrame()).iterrows():
            denominator_known[pd.Timestamp(row["observation_date"])] = float(row["value"]) if pd.notna(row["value"]) else np.nan

        numerator_series = pd.Series(numerator_known, dtype=float)
        denominator_series = pd.Series(denominator_known, dtype=float)
        common = numerator_series.index.intersection(denominator_series.index)
        ratio = (numerator_series.reindex(common) / denominator_series.reindex(common)).sort_index().dropna()
        values.append(_yoy_latest(ratio))
        observations.append(ratio.index[-1] if not ratio.empty else pd.NaT)

    index = pd.DatetimeIndex(release_dates, name="realtime_start")
    return TriggerSignal(
        value=pd.Series(values, index=index, dtype=float),
        observation_date=pd.Series(observations, index=index, name="latest_observation_date"),
    )


def unemployment_vs_trend_signal(vintage: pd.DataFrame, trend_months: int = 12) -> TriggerSignal:
    known_values: dict[pd.Timestamp, float] = {}
    values: list[float] = []
    observations: list[pd.Timestamp | pd.NaT] = []
    release_dates: list[pd.Timestamp] = []
    for release_date, release_rows in vintage.groupby("realtime_start", sort=True):
        for _, row in release_rows.iterrows():
            known_values[pd.Timestamp(row["observation_date"])] = float(row["value"]) if pd.notna(row["value"]) else np.nan
        known = pd.Series(known_values, dtype=float).sort_index().dropna()
        release_dates.append(pd.Timestamp(release_date))
        observations.append(known.index[-1] if not known.empty else pd.NaT)
        if len(known) < trend_months:
            values.append(np.nan)
        else:
            values.append(float(known.tail(trend_months).mean() - known.iloc[-1]))
    index = pd.DatetimeIndex(release_dates, name="realtime_start")
    return TriggerSignal(
        value=pd.Series(values, index=index, dtype=float),
        observation_date=pd.Series(observations, index=index, name="latest_observation_date"),
    )


def evaluation_dates(
    common_index: pd.DatetimeIndex,
    max_horizon: int,
    frequency: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    usable_index = pd.DatetimeIndex(common_index).sort_values()
    if usable_index.empty:
        raise ValueError("No common index available for trigger validation.")
    start_ts = pd.Timestamp(start).normalize() if start is not None else usable_index.min()
    end_ts = pd.Timestamp(end).normalize() if end is not None else usable_index.max() - pd.tseries.offsets.BDay(max_horizon)
    candidate_dates = pd.date_range(start_ts, end_ts, freq=frequency)
    selected: list[pd.Timestamp] = []
    for candidate in candidate_dates:
        prior = usable_index[usable_index <= candidate]
        if not prior.empty:
            selected.append(pd.Timestamp(prior[-1]))
    return pd.DatetimeIndex(sorted(set(selected)))


def write_economic_trigger_outputs(result: EconomicTriggerValidationResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trigger_panel": root / "economic_trigger_panel.parquet",
        "forecasts": root / "economic_trigger_validation_forecasts.parquet",
        "summary": root / "economic_trigger_validation_summary.csv",
        "current_snapshot": root / "economic_trigger_current_snapshot.csv",
    }
    result.trigger_panel.to_parquet(paths["trigger_panel"])
    result.forecasts.to_parquet(paths["forecasts"])
    result.summary.to_csv(paths["summary"], index=False)
    result.current_snapshot.to_csv(paths["current_snapshot"], index=False)
    return paths


def _yoy_latest(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    latest_date = pd.Timestamp(series.index[-1])
    prior_date = latest_date - pd.DateOffset(months=12)
    prior = series.loc[series.index <= prior_date]
    if prior.empty:
        return float("nan")
    prior_value = float(prior.iloc[-1])
    if prior_value == 0.0:
        return float("nan")
    return float(series.iloc[-1]) / prior_value - 1.0


def _healthy_mask(value: pd.Series, operator: str, threshold: float) -> pd.Series:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"Unsupported trigger operator: {operator}")


def _health_margin(value: pd.Series, operator: str, threshold: float) -> pd.Series:
    if operator in {">", ">="}:
        return value - threshold
    if operator in {"<", "<="}:
        return threshold - value
    raise ValueError(f"Unsupported trigger operator: {operator}")


def _combined_index(indexes: list[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    if not indexes:
        raise ValueError("No trigger signal indexes were provided.")
    combined = indexes[0]
    for index in indexes[1:]:
        combined = combined.union(index)
    return pd.DatetimeIndex(combined).sort_values()


def _bucket_for_value(bucket_edges: pd.DataFrame, value: float) -> int:
    containing = bucket_edges[(bucket_edges["min"] <= value) & (value <= bucket_edges["max"])]
    if not containing.empty:
        return int(containing.index[0])
    distances = (bucket_edges.mean(axis=1) - value).abs()
    return int(distances.idxmin())


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
    realized_by_quintile = bucketed.groupby("_quintile")["realized_value"].mean()
    if len(realized_by_quintile) < 2:
        return float("nan")
    return float(realized_by_quintile.iloc[-1] - realized_by_quintile.iloc[0])


def _odds_ratio(active_event_rate: float, inactive_event_rate: float) -> float:
    eps = 1e-6
    active_odds = (active_event_rate + eps) / (1.0 - active_event_rate + eps)
    inactive_odds = (inactive_event_rate + eps) / (1.0 - inactive_event_rate + eps)
    return float(active_odds / inactive_odds)
