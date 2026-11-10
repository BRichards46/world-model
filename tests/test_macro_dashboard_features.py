from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.config import default_config_path, load_yaml
from world_model.features.macro_dashboard import (
    compute_native_indicator_history,
    compute_native_indicator_snapshot,
    compute_persistence_history,
    compute_point_in_time_indicator_history,
    indicator_status_as_of,
)


def test_every_manifest_recipe_produces_native_metrics_and_history_measures() -> None:
    config = load_yaml(default_config_path("macro_dashboard.yml"))
    series = _synthetic_series()
    definitions = [*config["rows"]]
    definitions.extend(context for row in config["rows"] for context in row.get("context", []))

    for definition in definitions:
        result = compute_native_indicator_history(definition, series, config["history"])

        assert result.index.name == "observation_date"
        assert pd.notna(result["level_value"].iloc[-1]), definition["id"]
        assert pd.notna(result["velocity_value"].iloc[-1]), definition["id"]
        assert pd.notna(result["level_percentile_10y"].iloc[-1]), definition["id"]
        assert pd.notna(result["velocity_percentile_expanding"].iloc[-1]), definition["id"]
        assert result["history_status"].iloc[-1] == "available", definition["id"]
        assert pd.notna(result["baseline_sensitive"].iloc[-1]), definition["id"]
        assert result["persistence_label"].iloc[-1] in {
            "new turn",
            "building",
            "persistent",
            "flat",
        }


def test_linear_snapshot_matches_the_latest_full_history_row() -> None:
    config = load_yaml(default_config_path("macro_dashboard.yml"))
    definition = next(row for row in config["rows"] if row["id"] == "treasury_curve")
    series = _synthetic_series()

    history = compute_native_indicator_history(definition, series, config["history"])
    snapshot = compute_native_indicator_snapshot(definition, series, config["history"])
    expected = history.dropna(subset=["level_value", "velocity_value"], how="all").iloc[-1]

    for field in [
        "level_value",
        "velocity_value",
        "level_percentile_10y",
        "velocity_percentile_expanding",
        "level_history_status",
        "baseline_sensitive",
        "persistence_direction",
        "persistence_count",
        "persistence_label",
    ]:
        assert snapshot[field] == expected[field]


def test_three_month_growth_uses_non_overlapping_averages_and_does_not_bridge_missing_months() -> None:
    index = pd.date_range("2020-01-01", periods=15, freq="MS")
    values = pd.Series(np.exp(np.arange(15) * 0.01), index=index)
    definition = {
        "id": "growth",
        "series_ids": ["TEST"],
        "frequency": "monthly",
        "recipe": "three_month_growth",
    }
    history_config = _short_history_config()

    result = compute_native_indicator_history(definition, {"TEST": values}, history_config)
    latest = values.iloc[-3:].mean()
    prior = values.iloc[-6:-3].mean()
    expected_level = 100.0 * (np.exp(4.0 * (np.log(latest) - np.log(prior))) - 1.0)

    assert np.isclose(result["level_value"].iloc[-1], expected_level)

    with_gap = values.drop(index[-4])
    gap_result = compute_native_indicator_history(definition, {"TEST": with_gap}, history_config)
    assert pd.isna(gap_result["level_value"].iloc[-1])


def test_inflation_recipe_uses_yoy_level_and_latest_mom_turn() -> None:
    index = pd.date_range("2024-01-01", periods=15, freq="MS")
    values = pd.Series(100.0 * np.exp(np.arange(15) * 0.002), index=index)
    values.iloc[-1] *= 1.003
    definition = {
        "id": "inflation",
        "series_ids": ["TEST"],
        "frequency": "monthly",
        "recipe": "inflation_yoy_mom_change",
    }

    result = compute_native_indicator_history(definition, {"TEST": values}, _short_history_config())
    latest_yoy = 100.0 * (values.iloc[-1] / values.iloc[-13] - 1.0)
    latest_mom = 100.0 * (values.iloc[-1] / values.iloc[-2] - 1.0)
    prior_mom = 100.0 * (values.iloc[-2] / values.iloc[-3] - 1.0)

    assert np.isclose(result["level_value"].iloc[-1], latest_yoy)
    assert np.isclose(result["velocity_value"].iloc[-1], latest_mom - prior_mom)
    assert result["persistence_direction"].iloc[-1] == "rising"


def test_latest_value_change_uses_published_monthly_rate() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="MS")
    values = pd.Series([2.1, 2.3, 2.0, 2.4, 2.1, 2.8], index=index)
    definition = {
        "id": "median_inflation",
        "series_ids": ["TEST"],
        "frequency": "monthly",
        "recipe": "latest_value_change",
    }

    result = compute_native_indicator_history(definition, {"TEST": values}, _short_history_config())

    assert np.isclose(result["level_value"].iloc[-1], 2.8)
    assert np.isclose(result["velocity_value"].iloc[-1], 0.7)
    assert result["persistence_direction"].iloc[-1] == "rising"


def test_point_in_time_history_does_not_apply_a_revision_before_its_release() -> None:
    observations = pd.date_range("2020-01-01", periods=12, freq="MS")
    releases = observations + pd.DateOffset(months=1, days=14)
    vintage = pd.DataFrame(
        {
            "observation_date": observations,
            "value": np.linspace(100.0, 122.0, len(observations)),
            "realtime_start": releases,
        }
    )
    revision_date = pd.Timestamp("2021-02-20")
    vintage = pd.concat(
        [
            vintage,
            pd.DataFrame(
                {
                    "observation_date": [observations[-5]],
                    "value": [150.0],
                    "realtime_start": [revision_date],
                }
            ),
        ],
        ignore_index=True,
    )
    definition = {
        "id": "growth",
        "series_ids": ["TEST"],
        "frequency": "monthly",
        "recipe": "three_month_growth",
    }

    result = compute_point_in_time_indicator_history(
        definition,
        {"TEST": vintage},
        _short_history_config(),
    )

    before_revision = result.loc[result.index < revision_date].iloc[-1]
    after_revision = result.loc[revision_date]
    assert before_revision["observation_date"] == observations[-1]
    assert after_revision["observation_date"] == observations[-1]
    assert before_revision["level_value"] != after_revision["level_value"]
    assert after_revision["revision_status"] == "revision"
    assert bool(after_revision["is_revision"])
    assert after_revision["observation_release_date"] == releases[-1]
    assert after_revision["expected_next_release"] == releases[-1] + pd.DateOffset(months=1)

    freshness = {"stale_after_days": {"monthly": 50}}
    due = indicator_status_as_of(result, definition, "2021-02-21", freshness)
    stale = indicator_status_as_of(result, definition, "2021-04-01", freshness)
    assert due["days_since_release"] == 1
    assert due["days_since_observation_release"] == 37
    assert due["freshness_status"] == "due"
    assert stale["freshness_status"] == "stale"


def test_persistence_distinguishes_turn_building_persistent_and_flat() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="YS")
    velocity = pd.Series([1.0, 2.0, 3.0, 4.0, -2.5, -0.01], index=index)

    result = compute_persistence_history(
        velocity,
        window_years=10,
        min_periods=2,
        min_history_years=None,
        flat_percentile=20.0,
    )

    assert result["persistence_label"].tolist() == [
        "new turn",
        "building",
        "persistent",
        "persistent",
        "new turn",
        "flat",
    ]
    assert result["persistence_direction"].tolist()[-2:] == ["falling", "flat"]


def test_daily_recipe_ignores_explicit_holiday_rows() -> None:
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    values.iloc[-10] = np.nan
    definition = {
        "id": "curve",
        "series_ids": ["TEST"],
        "frequency": "daily",
        "recipe": "daily_average_change",
    }

    result = compute_native_indicator_history(definition, {"TEST": values}, _short_history_config())

    observed = values.dropna()
    expected_level = observed.iloc[-20:].mean()
    expected_velocity = 100.0 * (
        expected_level - observed.iloc[-83:-63].mean()
    )
    assert np.isclose(result["level_value"].iloc[-1], expected_level)
    assert np.isclose(result["velocity_value"].iloc[-1], expected_velocity)


def test_history_status_explains_a_calendar_history_shortfall() -> None:
    index = pd.date_range("2020-01-01", periods=48, freq="MS")
    values = pd.Series(100.0 + np.arange(len(index)), index=index)
    definition = {
        "id": "growth",
        "series_ids": ["TEST"],
        "frequency": "monthly",
        "recipe": "three_month_growth",
    }
    history_config = {
        **_short_history_config(),
        "minimum_history_years": 5,
    }

    result = compute_native_indicator_history(definition, {"TEST": values}, history_config)
    latest = result.iloc[-1]

    assert latest["level_history_status"] == "insufficient_history"
    assert latest["level_history_reason"] == "less_than_minimum_calendar_history"
    assert latest["history_status"] == "insufficient_history"
    assert pd.isna(latest["baseline_sensitive"])


def test_baseline_sensitivity_flags_a_structural_regime_difference() -> None:
    index = pd.date_range("1995-01-01", periods=360, freq="MS")
    values = pd.Series([100.0] * 240 + [0.1] * 117 + [50.0] * 3, index=index)
    definition = {
        "id": "rate",
        "series_ids": ["TEST"],
        "frequency": "monthly",
        "recipe": "three_month_average_change",
    }
    history_config = {
        **_short_history_config(),
        "minimum_history_years": 5,
        "minimum_observations": {"monthly": 60},
        "baseline_sensitive_threshold_pct_points": 25,
    }

    result = compute_native_indicator_history(definition, {"TEST": values}, history_config)
    latest = result.iloc[-1]

    assert latest["level_percentile_gap"] >= 25.0
    assert bool(latest["level_baseline_sensitive"])
    assert bool(latest["baseline_sensitive"])


def _short_history_config() -> dict[str, object]:
    return {
        "primary_window_years": 10,
        "minimum_history_years": None,
        "minimum_observations": {"daily": 2, "weekly": 2, "monthly": 2, "quarterly": 2},
        "flat_velocity_percentile": 20,
    }


def _synthetic_series() -> dict[str, pd.Series]:
    monthly = pd.date_range("2018-01-01", "2025-12-01", freq="MS")
    quarterly = pd.date_range("2018-01-01", "2025-10-01", freq="QS")
    weekly = pd.date_range("2018-01-05", "2025-12-26", freq="W-FRI")
    daily = pd.date_range("2018-01-01", "2025-12-31", freq="B")

    monthly_trend = pd.Series(100.0 * np.exp(np.arange(len(monthly)) * 0.002), index=monthly)
    quarterly_trend = pd.Series(100.0 * np.exp(np.arange(len(quarterly)) * 0.005), index=quarterly)
    return {
        "PAYEMS": monthly_trend,
        "UNRATE": pd.Series(4.0 + 0.4 * np.sin(np.arange(len(monthly)) / 8.0), index=monthly),
        "PCEC96": monthly_trend * 1.1,
        "INDPRO": monthly_trend * 0.9,
        "GDPC1": quarterly_trend,
        "PCEPILFE": pd.Series(100.0 * np.exp(np.arange(len(monthly)) * 0.0025), index=monthly),
        "CPILFESL": pd.Series(100.0 * np.exp(np.arange(len(monthly)) * 0.0023), index=monthly),
        "MEDCPIM158SFRBCLE": pd.Series(2.5 + 0.5 * np.sin(np.arange(len(monthly)) / 7.0), index=monthly),
        "EXPINF5YR": pd.Series(2.0 + 0.25 * np.sin(np.arange(len(monthly)) / 9.0), index=monthly),
        "DFF": pd.Series(2.0 + 0.5 * np.sin(np.arange(len(daily)) / 150.0), index=daily),
        "T10Y3M": pd.Series(0.5 + np.sin(np.arange(len(daily)) / 100.0), index=daily),
        "DGS10": pd.Series(4.0 + 0.5 * np.sin(np.arange(len(daily)) / 100.0), index=daily),
        "DGS3MO": pd.Series(3.5 - 0.5 * np.sin(np.arange(len(daily)) / 100.0), index=daily),
        "BAMLH0A0HYM2": pd.Series(4.0 + np.sin(np.arange(len(daily)) / 80.0), index=daily),
        "ANFCI": pd.Series(0.2 * np.sin(np.arange(len(weekly)) / 20.0), index=weekly),
        "PERMIT": pd.Series(1200.0 + np.arange(len(monthly)) * 2.0, index=monthly),
        "HOUST": pd.Series(1300.0 + np.arange(len(monthly)) * 1.5, index=monthly),
        "CLF16OV": pd.Series(160000.0 + np.arange(len(monthly)) * 20.0, index=monthly),
        "MORTGAGE30US": pd.Series(5.0 + 0.5 * np.sin(np.arange(len(weekly)) / 25.0), index=weekly),
    }


def test_inflation_yoy_bridges_an_unpublished_base_month_only() -> None:
    index = pd.date_range("2024-01-01", periods=15, freq="MS")
    values = pd.Series(100.0 * np.exp(np.arange(15) * 0.002), index=index)
    gap = index[2]  # never published; its own YoY and MoM stay missing
    published = values.drop(gap)
    definition = {"id": "inflation", "series_ids": ["TEST"], "frequency": "monthly", "recipe": "inflation_yoy_mom_change"}

    result = compute_native_indicator_history(definition, {"TEST": published}, _short_history_config())
    by_date = result.set_index("observation_date") if "observation_date" in result.columns else result
    expected = 100.0 * (values.iloc[14] / values.iloc[2] - 1.0)  # base log-interpolated between its neighbours
    assert np.isclose(by_date["level_value"].iloc[-1], expected)
