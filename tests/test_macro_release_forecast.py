from __future__ import annotations

import pandas as pd
import pytest

from world_model.states.macro_release_forecast import (
    _display_forecast,
    _feature_values,
    _metric,
    _target_events,
)


def _vintage(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": "TEST",
            "observation_date": pd.to_datetime([row[0] for row in rows]),
            "value": [row[1] for row in rows],
            "realtime_start": pd.to_datetime([row[2] for row in rows]),
            "realtime_end": pd.Timestamp("2099-12-31"),
        }
    )


def test_target_event_uses_prior_value_known_on_release_date() -> None:
    vintage = _vintage(
        [
            ("2025-12-01", 100.0, "2026-01-10"),
            ("2025-12-01", 101.0, "2026-02-10"),
            ("2026-01-01", 103.0, "2026-02-10"),
        ]
    )
    settings = {
        "target_transform": "monthly_diff",
        "target_scale": 1.0,
        "min_release_lag_days": 20,
        "max_release_lag_days": 70,
    }

    result = _target_events(vintage, settings, pd.Timestamp("2026-02-10"))

    assert result.iloc[-1]["target"] == 2.0
    assert result.iloc[-1]["release_date"] == pd.Timestamp("2026-02-10")


def test_features_cannot_see_release_on_or_after_target_day() -> None:
    table = _vintage(
        [
            ("2025-12-01", 100.0, "2026-01-10"),
            ("2026-01-01", 110.0, "2026-02-10"),
        ]
    )
    features = [{"id": "growth", "series_id": "TEST", "transform": "monthly_pct", "scale": 100.0}]

    before = _feature_values({"TEST": table}, features, pd.Timestamp("2026-02-09"))
    after = _feature_values({"TEST": table}, features, pd.Timestamp("2026-02-10"))

    assert pd.isna(before["growth"])
    assert after["growth"] == pytest.approx(10.0)


def test_monthly_growth_mean_and_job_display_conversion() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="MS")
    series = pd.Series([100.0, 101.0, 102.01, 103.0301], index=dates)

    result = _metric(
        series,
        "monthly_pct_mean",
        {"window": 3, "scale": 10000.0},
    )
    displayed = _display_forecast(
        result,
        pd.Series([159000.0], index=[dates[-1]]),
        {"display_as_level_change": True, "target_scale": 10000.0},
    )

    assert round(result, 8) == 100.0
    assert round(displayed, 8) == 1590.0


def test_site_payload_marks_only_validated_rows_and_keeps_display_units() -> None:
    from world_model.states.macro_release_forecast import MacroReleaseForecastResult, build_release_site_payload

    current = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp("2026-11-09")] * 2,
            "target_id": ["core_pce_mom", "payroll_change"],
            "target_label": ["Core PCE MoM", "Nonfarm payroll change"],
            "variant_id": ["pipeline", "core"],
            "forecast_observation_date": [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")],
            "display_forecast": [0.30, 178.0],
            "display_baseline_forecast": [0.26, 212.0],
            "display_prior_target": [0.32, 165.0],
            "unit": ["percent", "basis points of payroll employment"],
            "display_unit": ["percent", "thousand jobs"],
            "research_candidate": [True, False],
        }
    )
    summary = pd.DataFrame(
        {
            "target_id": ["core_pce_mom", "payroll_change"],
            "variant_id": ["pipeline", "core"],
            "model_mae": [0.076, 20.7],
            "baseline_mae": [0.119, 26.5],
            "direction_hit_rate": [0.74, 0.96],
            "observations": [77, 133],
        }
    )
    empty = pd.DataFrame()
    payload = build_release_site_payload(MacroReleaseForecastResult(empty, empty, summary, current, empty))
    pce, payroll = payload["forecasts"]
    assert payload["as_of_date"] == "2026-11-09"
    assert pce["validated"] is True and pce["model_mae"] == 0.076 and pce["reference_month"] == "2026-10"
    # MAE is in model units (bp of employment), so it is withheld from the thousand-jobs display.
    assert payroll["validated"] is False and payroll["model_mae"] is None


def test_ranges_use_recent_misses_in_display_units() -> None:
    from world_model.states.macro_release_forecast import MacroReleaseForecastResult, build_release_site_payload, forecast_ranges

    releases = pd.date_range("2020-01-15", periods=40, freq="MS")
    errors = [(-1.0 if index % 2 else 1.0) * (index % 5) for index in range(40)]
    forecasts = pd.DataFrame(
        {
            "target_id": "payroll_change",
            "variant_id": "core",
            "release_date": releases,
            "forecast": 5.0,
            "actual": [5.0 + error for error in errors],
        }
    )
    ranges = forecast_ranges(forecasts, "payroll_change", "core", window=30, minimum=24)
    recent = pd.Series(errors[-30:])
    assert ranges["observations"] == 30 and ranges["since"] == f"{releases[10]:%Y-%m}"
    assert ranges["error_80"] == [recent.quantile(0.1), recent.quantile(0.9)]
    assert forecast_ranges(forecasts, "payroll_change", "core", window=30, minimum=31) is None

    current = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp("2026-11-09")],
            "target_id": ["payroll_change"],
            "target_label": ["Nonfarm payroll change"],
            "variant_id": ["core"],
            "forecast_observation_date": [pd.Timestamp("2026-11-01")],
            "display_forecast": [160.0],
            "display_baseline_forecast": [150.0],
            "display_prior_target": [140.0],
            "display_scale": [16.0],  # thousand jobs per basis point of employment
            "unit": ["basis points of payroll employment"],
            "display_unit": ["thousand jobs"],
            "research_candidate": [False],
        }
    )
    empty = pd.DataFrame()
    row = build_release_site_payload(MacroReleaseForecastResult(empty, forecasts, empty, current, empty))["forecasts"][0]
    full = pd.Series(errors[-40:])
    assert row["range_80"] == [160.0 + 16.0 * full.quantile(0.1), 160.0 + 16.0 * full.quantile(0.9)]
    assert row["range_80"][0] < row["range_50"][0] <= row["range_50"][1] < row["range_80"][1]
