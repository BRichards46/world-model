from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import pandas as pd

from world_model.features.economic import historical_percentile_rank


MetricBuilder = Callable[[dict[str, pd.Series], list[str]], pd.DataFrame]
_MISSING = object()


def compute_native_indicator_history(
    definition: Mapping[str, Any],
    series_by_id: Mapping[str, pd.Series],
    history_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Calculate one dashboard row at its native observation frequency.

    Inputs must contain only values known at the caller's as-of date. The
    returned percentiles compare each observation with prior observations and
    therefore never include the current value in its own baseline.
    """
    metrics = _compute_native_metrics(definition, series_by_id)
    return _add_history_measures(metrics, definition, history_config)


def compute_native_indicator_snapshot(
    definition: Mapping[str, Any],
    series_by_id: Mapping[str, pd.Series],
    history_config: Mapping[str, Any],
) -> pd.Series:
    """Calculate only the latest native-frequency state in linear time."""
    metrics = _compute_native_metrics(definition, series_by_id)
    usable = metrics.dropna(subset=["level_value", "velocity_value"], how="all")
    if usable.empty:
        return pd.Series(dtype=object, name=None)
    position = metrics.index.get_loc(usable.index[-1])
    return _add_snapshot_measures(metrics, int(position), definition, history_config)


def compute_point_in_time_indicator_history(
    definition: Mapping[str, Any],
    vintage_by_id: Mapping[str, pd.DataFrame],
    history_config: Mapping[str, Any],
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rebuild release-time row states from standard FRED vintage tables.

    A snapshot is recalculated after each source release using only values whose
    ``realtime_start`` is on or before that release. Later revisions cannot
    alter an earlier output row.
    """
    series_ids = [str(series_id) for series_id in definition["series_ids"]]
    missing = [series_id for series_id in series_ids if series_id not in vintage_by_id]
    if missing:
        raise KeyError(f"Missing vintage source for {definition['id']}: {', '.join(missing)}")

    start_timestamp = pd.Timestamp(start_date).normalize() if start_date is not None else None
    events: dict[pd.Timestamp, list[tuple[str, pd.DataFrame]]] = {}
    known: dict[str, dict[pd.Timestamp, float]] = {series_id: {} for series_id in series_ids}
    validated_vintages: dict[str, pd.DataFrame] = {}
    for series_id in series_ids:
        vintage = _validated_vintage(vintage_by_id[series_id], series_id)
        validated_vintages[series_id] = vintage
        if start_timestamp is not None:
            earlier = vintage.loc[vintage["realtime_start"] < start_timestamp]
            if not earlier.empty:
                latest_known = earlier.drop_duplicates("observation_date", keep="last")
                known[series_id] = {
                    pd.Timestamp(row.observation_date): float(row.value) if pd.notna(row.value) else np.nan
                    for row in latest_known.itertuples(index=False)
                }
            vintage = vintage.loc[vintage["realtime_start"] >= start_timestamp]
        for release_date, rows in vintage.groupby("realtime_start", sort=True):
            events.setdefault(pd.Timestamp(release_date), []).append((series_id, rows))

    records: list[dict[str, Any]] = []
    latest_observation_release_date: pd.Timestamp | None = None
    prior_observation_date: pd.Timestamp | None = None
    prior_state: dict[str, Any] | None = None
    if start_timestamp is not None and all(known.values()):
        prior_series = {
            series_id: pd.Series(values, dtype=float).sort_index()
            for series_id, values in known.items()
        }
        initial = compute_native_indicator_snapshot(definition, prior_series, history_config)
        if not initial.empty:
            prior_state = initial.to_dict()
            prior_state["observation_date"] = pd.Timestamp(initial.name)
            prior_observation_date = pd.Timestamp(initial.name)
            anchor = _anchor_series_id(definition)
            anchor_rows = validated_vintages[anchor]
            introduced = anchor_rows.loc[
                anchor_rows["observation_date"] == prior_observation_date,
                "realtime_start",
            ]
            if not introduced.empty:
                latest_observation_release_date = pd.Timestamp(introduced.min())
    for release_date in sorted(events):
        introduced_observation = False
        revised_observation = False
        updated_series: set[str] = set()
        for series_id, rows in events[release_date]:
            for row in rows.itertuples(index=False):
                value = float(row.value) if pd.notna(row.value) else np.nan
                observation_date = pd.Timestamp(row.observation_date)
                prior_value = known[series_id].get(observation_date, _MISSING)
                if prior_value is _MISSING:
                    introduced_observation = True
                    updated_series.add(series_id)
                elif not _values_equal(prior_value, value):
                    revised_observation = True
                    updated_series.add(series_id)
                known[series_id][observation_date] = value

        if any(not values for values in known.values()):
            continue
        known_series = {
            series_id: pd.Series(values, dtype=float).sort_index()
            for series_id, values in known.items()
        }
        latest = compute_native_indicator_snapshot(definition, known_series, history_config)
        if latest.empty:
            continue
        record = latest.to_dict()
        record["realtime_start"] = release_date
        record["observation_date"] = pd.Timestamp(latest.name)
        state = {key: value for key, value in record.items() if key != "realtime_start"}
        if prior_state is not None and not _state_changed(prior_state, state):
            continue

        observation_advanced = (
            prior_observation_date is None
            or pd.Timestamp(latest.name) > prior_observation_date
        )
        if observation_advanced:
            latest_observation_release_date = release_date
        if latest_observation_release_date is None:
            latest_observation_release_date = release_date

        record["release_date"] = release_date
        record["observation_release_date"] = latest_observation_release_date
        record["release_lag_days"] = int((release_date - pd.Timestamp(latest.name)).days)
        record["revision_status"] = _revision_status(
            is_initial=prior_state is None,
            introduced_observation=introduced_observation,
            revised_observation=revised_observation,
        )
        record["is_revision"] = bool(revised_observation)
        record["source_series_updated"] = ",".join(sorted(updated_series))
        record["expected_next_release"] = _expected_next_release(
            latest_observation_release_date,
            str(definition["frequency"]),
        )
        record["expected_next_release_is_estimate"] = True
        records.append(record)
        prior_state = state
        prior_observation_date = pd.Timestamp(latest.name)

    if not records:
        return _empty_point_in_time_history()
    result = pd.DataFrame.from_records(records).set_index("realtime_start").sort_index()
    result.index = pd.DatetimeIndex(result.index, name="realtime_start")
    return result


def indicator_status_as_of(
    point_in_time_history: pd.DataFrame,
    definition: Mapping[str, Any],
    as_of: str | pd.Timestamp,
    freshness_config: Mapping[str, Any],
) -> pd.Series:
    """Return the latest release state with freshness evaluated at ``as_of``."""
    if point_in_time_history.empty:
        raise ValueError(f"No point-in-time history is available for {definition['id']}")
    as_of_date = pd.Timestamp(as_of).normalize()
    available = point_in_time_history.loc[point_in_time_history.index <= as_of_date]
    if available.empty:
        raise ValueError(f"No {definition['id']} release is available on or before {as_of_date.date()}")

    result = available.iloc[-1].copy()
    release_date = pd.Timestamp(result.get("release_date", available.index[-1])).normalize()
    observation_release = pd.Timestamp(
        result.get("observation_release_date", release_date)
    ).normalize()
    expected_next = result.get("expected_next_release")
    expected_next = pd.Timestamp(expected_next).normalize() if pd.notna(expected_next) else pd.NaT
    stale_after_days = int(
        definition.get(
            "stale_after_days",
            freshness_config.get("stale_after_days", {}).get(str(definition["frequency"]), 30),
        )
    )
    days_since_observation_release = int((as_of_date - observation_release).days)

    result["as_of_date"] = as_of_date
    result["days_since_release"] = int((as_of_date - release_date).days)
    result["days_since_observation_release"] = days_since_observation_release
    result["stale_after_days"] = stale_after_days
    result["freshness_status"] = (
        "stale"
        if days_since_observation_release > stale_after_days
        else "due"
        if pd.notna(expected_next) and as_of_date > expected_next
        else "fresh"
    )
    return result


def _compute_native_metrics(
    definition: Mapping[str, Any],
    series_by_id: Mapping[str, pd.Series],
) -> pd.DataFrame:
    series_ids = [str(series_id) for series_id in definition["series_ids"]]
    missing = [series_id for series_id in series_ids if series_id not in series_by_id]
    if missing:
        raise KeyError(f"Missing source series for {definition['id']}: {', '.join(missing)}")
    prepared = {
        series_id: _validated_series(series_by_id[series_id], series_id)
        for series_id in series_ids
    }
    recipe = str(definition["recipe"])
    try:
        metrics = _RECIPE_BUILDERS[recipe](prepared, series_ids)
    except KeyError as exc:
        raise ValueError(f"Unsupported macro dashboard recipe: {recipe}") from exc
    metrics.index = pd.DatetimeIndex(metrics.index, name="observation_date")
    return metrics.sort_index()


def _add_snapshot_measures(
    metrics: pd.DataFrame,
    position: int,
    definition: Mapping[str, Any],
    history_config: Mapping[str, Any],
) -> pd.Series:
    timestamp = pd.Timestamp(metrics.index[position])
    result = metrics.iloc[position].copy().astype(object)
    frequency = _percentile_frequency(str(definition["frequency"]))
    min_periods = int(history_config.get("minimum_observations", {}).get(frequency, 20))
    window_years = int(history_config.get("primary_window_years", 10))
    minimum_years_value = history_config.get("minimum_history_years", 5)
    min_history_years = int(minimum_years_value) if minimum_years_value is not None else None
    threshold = float(history_config.get("baseline_sensitive_threshold_pct_points", 25.0))

    flags: list[Any] = []
    statuses: list[str] = []
    warnings: list[str] = []
    for metric in ("level", "velocity"):
        values = metrics[f"{metric}_value"]
        trailing, count, years, status, reason = _percentile_at(
            values,
            position,
            window_years=window_years,
            min_periods=min_periods,
            min_history_years=min_history_years,
        )
        expanding, _, _, _, _ = _percentile_at(
            values,
            position,
            window_years=None,
            min_periods=min_periods,
            min_history_years=min_history_years,
        )
        gap = abs(trailing - expanding) if pd.notna(trailing) and pd.notna(expanding) else np.nan
        flag: Any = bool(gap >= threshold) if pd.notna(gap) else pd.NA
        flags.append(flag)
        statuses.append(status)
        if status != "available":
            warnings.append(f"{metric}:{reason}")

        result[f"{metric}_percentile_10y"] = trailing
        result[f"{metric}_percentile_expanding"] = expanding
        result[f"{metric}_percentile_gap"] = gap
        result[f"{metric}_baseline_sensitive"] = flag
        result[f"{metric}_history_status"] = status
        result[f"{metric}_history_reason"] = reason
        result[f"{metric}_history_observations"] = count
        result[f"{metric}_history_years"] = years

    result["baseline_sensitive"] = _combine_snapshot_flags(flags)
    result["history_status"] = (
        "available"
        if all(status == "available" for status in statuses)
        else "metric_unavailable"
        if "metric_unavailable" in statuses
        else "insufficient_history"
    )
    result["history_warning"] = "; ".join(warnings) if warnings else None
    persistence = _persistence_snapshot(
        metrics["velocity_value"],
        position,
        window_years=window_years,
        min_periods=min_periods,
        min_history_years=min_history_years,
        flat_percentile=float(history_config.get("flat_velocity_percentile", 20.0)),
    )
    for key, value in persistence.items():
        result[key] = value
    result.name = timestamp
    return result


def _percentile_at(
    values: pd.Series,
    position: int,
    window_years: int | None,
    min_periods: int,
    min_history_years: int | None,
) -> tuple[float, int, float, str, str | None]:
    timestamp = pd.Timestamp(values.index[position])
    current_value = values.iloc[position]
    history = values.iloc[:position]
    if window_years is not None:
        history = history.loc[history.index >= timestamp - pd.DateOffset(years=window_years)]
    history = history.dropna()
    count = len(history)
    years = (
        max(0.0, (timestamp - pd.Timestamp(history.index.min())).days / 365.2425)
        if not history.empty
        else 0.0
    )
    if pd.isna(current_value):
        return np.nan, count, years, "metric_unavailable", "metric_unavailable"
    if count < min_periods:
        return np.nan, count, years, "insufficient_history", "fewer_than_minimum_observations"
    if min_history_years is not None and years < float(min_history_years):
        return np.nan, count, years, "insufficient_history", "less_than_minimum_calendar_history"

    observations = history.to_numpy(dtype=float)
    less = float(np.count_nonzero(observations < float(current_value)))
    equal = float(np.count_nonzero(observations == float(current_value)))
    percentile = 100.0 * (less + 0.5 * equal) / len(observations)
    return percentile, count, years, "available", None


def _persistence_snapshot(
    velocity: pd.Series,
    position: int,
    window_years: int,
    min_periods: int,
    min_history_years: int | None,
    flat_percentile: float,
) -> dict[str, Any]:
    def direction_at(item_position: int) -> int:
        value = velocity.iloc[item_position]
        if pd.isna(value):
            return 0
        absolute_percentile, _, _, _, _ = _percentile_at(
            velocity.abs(),
            item_position,
            window_years=window_years,
            min_periods=min_periods,
            min_history_years=min_history_years,
        )
        if value == 0.0 or (pd.notna(absolute_percentile) and absolute_percentile <= flat_percentile):
            return 0
        return 1 if value > 0.0 else -1

    direction = direction_at(position)
    if pd.isna(velocity.iloc[position]):
        return {
            "persistence_direction": None,
            "persistence_count": pd.NA,
            "persistence_label": None,
        }
    if direction == 0:
        return {
            "persistence_direction": "flat",
            "persistence_count": 0,
            "persistence_label": "flat",
        }

    count = 1
    for prior_position in range(position - 1, -1, -1):
        if direction_at(prior_position) != direction:
            break
        count += 1
    return {
        "persistence_direction": "rising" if direction > 0 else "falling",
        "persistence_count": count,
        "persistence_label": "new turn" if count == 1 else "building" if count == 2 else "persistent",
    }


def _combine_snapshot_flags(flags: list[Any]) -> Any:
    if any(flag is True for flag in flags):
        return True
    if all(flag is not pd.NA for flag in flags):
        return False
    return pd.NA


def _add_history_measures(
    metrics: pd.DataFrame,
    definition: Mapping[str, Any],
    history_config: Mapping[str, Any],
) -> pd.DataFrame:
    result = metrics.copy()
    frequency = _percentile_frequency(str(definition["frequency"]))
    minimums = history_config.get("minimum_observations", {})
    min_periods = int(minimums.get(frequency, 20))
    window_years = int(history_config.get("primary_window_years", 10))
    minimum_years_value = history_config.get("minimum_history_years", 5)
    min_history_years = int(minimum_years_value) if minimum_years_value is not None else None

    for metric in ("level", "velocity"):
        values = result[f"{metric}_value"]
        trailing_percentile = historical_percentile_rank(
            values,
            window_years=window_years,
            min_periods=min_periods,
            min_history_years=min_history_years,
        )
        expanding_percentile = historical_percentile_rank(
            values,
            window_years=None,
            min_periods=min_periods,
            min_history_years=min_history_years,
        )
        result[f"{metric}_percentile_10y"] = trailing_percentile
        result[f"{metric}_percentile_expanding"] = expanding_percentile
        result = result.join(
            _history_availability(
                values,
                trailing_percentile,
                metric=metric,
                window_years=window_years,
                min_periods=min_periods,
                min_history_years=min_history_years,
            )
        )

        gap = (trailing_percentile - expanding_percentile).abs()
        comparable = trailing_percentile.notna() & expanding_percentile.notna()
        sensitive = pd.Series(pd.NA, index=result.index, dtype="boolean")
        sensitive.loc[comparable] = (
            gap.loc[comparable]
            >= float(history_config.get("baseline_sensitive_threshold_pct_points", 25.0))
        )
        result[f"{metric}_percentile_gap"] = gap
        result[f"{metric}_baseline_sensitive"] = sensitive

    persistence = compute_persistence_history(
        result["velocity_value"],
        window_years=window_years,
        min_periods=min_periods,
        min_history_years=min_history_years,
        flat_percentile=float(history_config.get("flat_velocity_percentile", 20.0)),
    )
    result = result.join(persistence)
    result["baseline_sensitive"] = _combine_baseline_flags(
        result["level_baseline_sensitive"],
        result["velocity_baseline_sensitive"],
    )
    result[["history_status", "history_warning"]] = _combined_history_status(result)
    return result


def _three_month_growth(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    values = _regularize(series[series_ids[0]], "M")
    average = values.rolling(3, min_periods=3).mean()
    level = _annualized_growth(average, average.shift(3), periods_per_year=4.0)
    return _metric_frame(level, level - level.shift(3))


def _three_month_average_change(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    values = _regularize(series[series_ids[0]], "M")
    level = values.rolling(3, min_periods=3).mean()
    return _metric_frame(level, level - level.shift(3))


def _quarterly_growth(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    values = _regularize(series[series_ids[0]], "Q")
    level = _annualized_growth(values, values.shift(1), periods_per_year=4.0)
    return _metric_frame(level, level - level.shift(1))


def _inflation_yoy_mom_change(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    """Use the familiar YoY rate for level and the latest MoM turn for velocity."""
    values = _regularize(series[series_ids[0]], "M")
    level = _annualized_growth(values, values.shift(12), periods_per_year=1.0)
    monthly_rate = _annualized_growth(values, values.shift(1), periods_per_year=1.0)
    velocity = monthly_rate - monthly_rate.shift(1)
    return _metric_frame(level, velocity)


def _latest_value_change(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    """Use a published monthly rate directly and compare it with the prior release."""
    values = _regularize(series[series_ids[0]], "M")
    return _metric_frame(values, values - values.shift(1))


def _real_policy_rate(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    if len(series_ids) != 2:
        raise ValueError("real_policy_rate requires DFF and PCEPILFE")
    daily_rate = series[series_ids[0]]
    price_index = _regularize(series[series_ids[1]], "M")

    rate_by_month = daily_rate.groupby(daily_rate.index.to_period("M")).mean()
    rate_by_month = rate_by_month.reindex(price_index.index.to_period("M"))
    rate_by_month.index = price_index.index
    three_month_rate = rate_by_month.rolling(3, min_periods=3).mean()
    inflation = _annualized_growth(price_index, price_index.shift(12), periods_per_year=1.0)
    level = three_month_rate - inflation
    velocity = 100.0 * (level - level.shift(3))
    return _metric_frame(level, velocity)


def _daily_average_change(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    # FRED daily tables can contain explicit holiday rows with null values.
    # The recipe is defined in observed business days, so holidays are not
    # allowed to consume one of the 20 or 63 observations.
    values = series[series_ids[0]].dropna()
    level = values.rolling(20, min_periods=20).mean()
    return _metric_frame(level, 100.0 * (level - level.shift(63)))


def _weekly_average_change(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    values = series[series_ids[0]]
    level = values.rolling(4, min_periods=4).mean()
    return _metric_frame(level, level - level.shift(13))


def _weekly_average_change_bps(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    result = _weekly_average_change(series, series_ids)
    result["velocity_value"] *= 100.0
    return result


def _permits_yoy_growth(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    values = _regularize(series[series_ids[0]], "M")
    average = values.rolling(3, min_periods=3).mean()
    level = _annualized_growth(average, average.shift(12), periods_per_year=1.0)
    return _metric_frame(level, level - level.shift(3))


def _housing_starts_intensity(series: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    if len(series_ids) != 2:
        raise ValueError("housing_starts_intensity requires HOUST and CLF16OV")
    starts = _regularize(series[series_ids[0]], "M")
    labor_force = _regularize(series[series_ids[1]], "M")
    ratio = 1000.0 * starts / labor_force.where(labor_force > 0.0)
    level = ratio.rolling(3, min_periods=3).mean()
    velocity = _annualized_growth(level, level.shift(12), periods_per_year=1.0)
    return _metric_frame(level, velocity)


def _annualized_growth(current: pd.Series, prior: pd.Series, periods_per_year: float) -> pd.Series:
    valid = current.where((current > 0.0) & (prior > 0.0))
    comparison = prior.where((current > 0.0) & (prior > 0.0))
    return 100.0 * (np.exp(periods_per_year * (np.log(valid) - np.log(comparison))) - 1.0)


def _metric_frame(level: pd.Series, velocity: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"level_value": level.astype(float), "velocity_value": velocity.astype(float)})


def _regularize(series: pd.Series, frequency: str) -> pd.Series:
    periods = series.index.to_period(frequency)
    if periods.has_duplicates:
        raise ValueError(f"Series contains multiple observations in the same {frequency} period")
    regular = pd.Series(series.to_numpy(dtype=float), index=periods, dtype=float)
    full_index = pd.period_range(regular.index.min(), regular.index.max(), freq=frequency)
    regular = regular.reindex(full_index)
    regular.index = regular.index.to_timestamp(how="start")
    return regular


def compute_persistence_history(
    velocity: pd.Series,
    window_years: int,
    min_periods: int,
    min_history_years: int | None,
    flat_percentile: float,
) -> pd.DataFrame:
    absolute_percentile = historical_percentile_rank(
        velocity.abs(),
        window_years=window_years,
        min_periods=min_periods,
        min_history_years=min_history_years,
    )
    directions: list[str | None] = []
    counts: list[int | pd.NA] = []
    labels: list[str | None] = []
    previous_sign = 0
    streak = 0

    for timestamp, value in velocity.items():
        if pd.isna(value):
            directions.append(None)
            counts.append(pd.NA)
            labels.append(None)
            continue
        is_flat = value == 0.0 or (
            pd.notna(absolute_percentile.loc[timestamp])
            and absolute_percentile.loc[timestamp] <= flat_percentile
        )
        if is_flat:
            directions.append("flat")
            counts.append(0)
            labels.append("flat")
            previous_sign = 0
            streak = 0
            continue

        sign = 1 if value > 0.0 else -1
        streak = streak + 1 if sign == previous_sign else 1
        previous_sign = sign
        directions.append("rising" if sign > 0 else "falling")
        counts.append(streak)
        labels.append("new turn" if streak == 1 else "building" if streak == 2 else "persistent")

    return pd.DataFrame(
        {
            "persistence_direction": directions,
            "persistence_count": pd.array(counts, dtype="Int64"),
            "persistence_label": labels,
        },
        index=velocity.index,
    )


def _history_availability(
    values: pd.Series,
    percentile: pd.Series,
    metric: str,
    window_years: int,
    min_periods: int,
    min_history_years: int | None,
) -> pd.DataFrame:
    statuses: list[str] = []
    reasons: list[str | None] = []
    observation_counts: list[int] = []
    history_years: list[float] = []

    for position, (timestamp, current_value) in enumerate(values.items()):
        history = values.iloc[:position]
        history = history.loc[history.index >= timestamp - pd.DateOffset(years=window_years)].dropna()
        count = len(history)
        years = (
            max(0.0, (timestamp - pd.Timestamp(history.index.min())).days / 365.2425)
            if not history.empty
            else 0.0
        )
        observation_counts.append(count)
        history_years.append(years)

        if pd.isna(current_value):
            statuses.append("metric_unavailable")
            reasons.append("metric_unavailable")
        elif pd.notna(percentile.iloc[position]):
            statuses.append("available")
            reasons.append(None)
        elif count < min_periods:
            statuses.append("insufficient_history")
            reasons.append("fewer_than_minimum_observations")
        elif min_history_years is not None and years < float(min_history_years):
            statuses.append("insufficient_history")
            reasons.append("less_than_minimum_calendar_history")
        else:
            statuses.append("insufficient_history")
            reasons.append("percentile_unavailable")

    return pd.DataFrame(
        {
            f"{metric}_history_status": statuses,
            f"{metric}_history_reason": reasons,
            f"{metric}_history_observations": pd.array(observation_counts, dtype="Int64"),
            f"{metric}_history_years": history_years,
        },
        index=values.index,
    )


def _combine_baseline_flags(level: pd.Series, velocity: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=level.index, dtype="boolean")
    result.loc[level.eq(True).fillna(False) | velocity.eq(True).fillna(False)] = True
    both_known = level.notna() & velocity.notna()
    result.loc[both_known & result.isna()] = False
    return result


def _combined_history_status(result: pd.DataFrame) -> pd.DataFrame:
    statuses: list[str] = []
    warnings: list[str | None] = []
    for row in result.itertuples():
        level_status = row.level_history_status
        velocity_status = row.velocity_history_status
        if level_status == "available" and velocity_status == "available":
            statuses.append("available")
            warnings.append(None)
            continue
        if "metric_unavailable" in {level_status, velocity_status}:
            statuses.append("metric_unavailable")
        else:
            statuses.append("insufficient_history")
        details = []
        if level_status != "available":
            details.append(f"level:{row.level_history_reason}")
        if velocity_status != "available":
            details.append(f"velocity:{row.velocity_history_reason}")
        warnings.append("; ".join(details))
    return pd.DataFrame(
        {"history_status": statuses, "history_warning": warnings},
        index=result.index,
    )


def _revision_status(
    is_initial: bool,
    introduced_observation: bool,
    revised_observation: bool,
) -> str:
    if is_initial:
        return "initial"
    if introduced_observation and revised_observation:
        return "new_observation_and_revision"
    if introduced_observation:
        return "new_observation"
    if revised_observation:
        return "revision"
    return "source_update"


def _expected_next_release(release_date: pd.Timestamp, frequency: str) -> pd.Timestamp:
    if frequency == "daily":
        return release_date + pd.offsets.BDay(1)
    if frequency == "weekly":
        return release_date + pd.DateOffset(weeks=1)
    if frequency in {"monthly", "mixed_daily_monthly"}:
        return release_date + pd.DateOffset(months=1)
    if frequency == "quarterly":
        return release_date + pd.DateOffset(months=3)
    raise ValueError(f"Unsupported dashboard frequency: {frequency}")


def _state_changed(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    keys = set(previous) | set(current)
    return any(not _values_equal(previous.get(key), current.get(key)) for key in keys)


def _values_equal(left: Any, right: Any) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _validated_series(value: pd.Series, series_id: str) -> pd.Series:
    if not isinstance(value.index, pd.DatetimeIndex):
        raise TypeError(f"{series_id} must use a DatetimeIndex")
    if value.index.has_duplicates:
        raise ValueError(f"{series_id} must have unique observation dates")
    result = value.astype(float).sort_index()
    result.index = result.index.normalize()
    return result


def _validated_vintage(vintage: pd.DataFrame, series_id: str) -> pd.DataFrame:
    required = {"observation_date", "value", "realtime_start"}
    missing = required - set(vintage.columns)
    if missing:
        raise ValueError(f"{series_id} vintage is missing columns: {', '.join(sorted(missing))}")
    result = vintage.copy()
    result["observation_date"] = pd.to_datetime(result["observation_date"], errors="coerce").dt.normalize()
    result["realtime_start"] = pd.to_datetime(result["realtime_start"], errors="coerce").dt.normalize()
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["observation_date", "realtime_start"])
    return result.sort_values(["realtime_start", "observation_date"])


def _percentile_frequency(frequency: str) -> str:
    if frequency == "mixed_daily_monthly":
        return "monthly"
    if frequency not in {"daily", "weekly", "monthly", "quarterly"}:
        raise ValueError(f"Unsupported dashboard frequency: {frequency}")
    return frequency


def _anchor_series_id(definition: Mapping[str, Any]) -> str:
    configured = definition.get("anchor_series_id")
    if configured:
        return str(configured)
    series_ids = [str(series_id) for series_id in definition["series_ids"]]
    return series_ids[-1] if definition.get("recipe") == "real_policy_rate" else series_ids[0]


def _empty_point_in_time_history() -> pd.DataFrame:
    columns = [
        "observation_date",
        "release_date",
        "observation_release_date",
        "release_lag_days",
        "revision_status",
        "is_revision",
        "source_series_updated",
        "expected_next_release",
        "expected_next_release_is_estimate",
        "level_value",
        "velocity_value",
        "level_percentile_10y",
        "level_percentile_expanding",
        "level_percentile_gap",
        "level_baseline_sensitive",
        "level_history_status",
        "level_history_reason",
        "level_history_observations",
        "level_history_years",
        "velocity_percentile_10y",
        "velocity_percentile_expanding",
        "velocity_percentile_gap",
        "velocity_baseline_sensitive",
        "velocity_history_status",
        "velocity_history_reason",
        "velocity_history_observations",
        "velocity_history_years",
        "baseline_sensitive",
        "history_status",
        "history_warning",
        "persistence_direction",
        "persistence_count",
        "persistence_label",
    ]
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="realtime_start"))


_RECIPE_BUILDERS: dict[str, MetricBuilder] = {
    "three_month_growth": _three_month_growth,
    "three_month_average_change": _three_month_average_change,
    "quarterly_growth": _quarterly_growth,
    "inflation_yoy_mom_change": _inflation_yoy_mom_change,
    "latest_value_change": _latest_value_change,
    "real_policy_rate": _real_policy_rate,
    "daily_average_change": _daily_average_change,
    "weekly_average_change": _weekly_average_change,
    "weekly_average_change_bps": _weekly_average_change_bps,
    "permits_yoy_growth": _permits_yoy_growth,
    "housing_starts_intensity": _housing_starts_intensity,
}
