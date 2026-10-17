from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.economic_triggers import (
    EconomicTriggerValidator,
    conditional_probability,
    current_trigger_snapshot,
    ratio_yoy_signal,
    state_bucket_probability,
    summarize_trigger_forecasts,
    unemployment_vs_trend_signal,
)


def test_ratio_yoy_signal_uses_point_in_time_known_values() -> None:
    numerator = _vintage_frame("N", [("2020-01-01", 100.0, "2020-02-01"), ("2021-01-01", 120.0, "2021-02-01")])
    denominator = _vintage_frame("D", [("2020-01-01", 50.0, "2020-02-01"), ("2021-01-01", 50.0, "2021-02-01")])

    signal = ratio_yoy_signal(numerator, denominator)

    assert round(signal.value.loc["2021-02-01"], 4) == 0.20


def test_unemployment_vs_trend_matches_legacy_direction() -> None:
    rows = []
    for idx, observation_date in enumerate(pd.date_range("2020-01-01", periods=12, freq="MS")):
        rows.append((str(observation_date.date()), 5.0 + idx * 0.1, str((observation_date + pd.DateOffset(months=1)).date())))
    vintage = _vintage_frame("UNRATE", rows)

    signal = unemployment_vs_trend_signal(vintage, trend_months=12)

    assert round(signal.value.dropna().iloc[-1], 4) == round(pd.Series([5.0 + idx * 0.1 for idx in range(12)]).mean() - 6.1, 4)


def test_trigger_panel_counts_failed_and_healthy_rules(tmp_path: Path) -> None:
    fred_root = tmp_path / "fred"
    vintage_root = fred_root / "vintage"
    vintage_root.mkdir(parents=True)
    _write_monthly_vintage(vintage_root / "RSAFS.parquet", "RSAFS", start=100.0, step=1.0)
    _write_monthly_vintage(vintage_root / "CPIAUCSL.parquet", "CPIAUCSL", start=100.0, step=1.0)

    trigger_config = {
        "econ_health_trigger": 1,
        "triggers": [
            {
                "id": "retail_sales_yoy",
                "label": "Retail sales YoY",
                "group": "growth",
                "series_id": "RSAFS",
                "transform": "yoy_pct",
                "healthy_operator": ">",
                "threshold": 0.01,
            },
            {
                "id": "headline_cpi_yoy",
                "label": "Headline CPI YoY",
                "group": "inflation",
                "series_id": "CPIAUCSL",
                "transform": "yoy_pct",
                "healthy_operator": "<",
                "threshold": 0.025,
            },
        ],
    }
    validator = EconomicTriggerValidator(
        data_roots={"fred": str(fred_root)},
        economic_config={"components": {}},
        trigger_config=trigger_config,
    )

    panel = validator.build_trigger_panel()
    latest = panel.dropna(subset=["healthy_count"]).iloc[-1]

    assert latest["retail_sales_yoy_healthy"] == 1.0
    assert latest["headline_cpi_yoy_failed"] == 1.0
    assert latest["healthy_count"] == 1.0
    assert latest["failed_count"] == 1.0
    snapshot = current_trigger_snapshot(panel, trigger_config)
    assert set(snapshot["trigger"]) == {"retail_sales_yoy", "headline_cpi_yoy"}


def test_conditional_probability_and_state_bucket_baseline() -> None:
    feature = pd.Series([0, 0, 1, 1, 1], dtype=float)
    target = pd.Series([0, 0, 1, 1, 0], dtype=float)
    state = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)

    assert conditional_probability(feature, target, current_feature=1.0, kind="binary", min_bucket_observations=1) == 2 / 3
    assert state_bucket_probability(state, target, current_state=0.75, min_bucket_observations=1) in {0.0, 1.0}


def test_summarize_trigger_forecasts_reports_brier_improvement() -> None:
    forecasts = pd.DataFrame(
        {
            "forecast_date": pd.date_range("2020-01-31", periods=4, freq="ME"),
            "target": ["t"] * 4,
            "target_label": ["Target"] * 4,
            "horizon": ["one_month"] * 4,
            "trading_days": [21] * 4,
            "feature": ["f"] * 4,
            "feature_label": ["Feature"] * 4,
            "feature_kind": ["binary"] * 4,
            "feature_value": [0.0, 0.0, 1.0, 1.0],
            "forecast_value": [0.1, 0.1, 0.9, 0.9],
            "baseline_value": [0.5] * 4,
            "state_bucket_baseline_value": [0.5] * 4,
            "realized_value": [0.0, 0.0, 1.0, 1.0],
            "train_observations": [60] * 4,
        }
    )

    summary = summarize_trigger_forecasts(forecasts)

    assert summary.loc[0, "active_inactive_spread"] == 1.0
    assert summary.loc[0, "brier_improvement_vs_baseline"] > 0.0


def _vintage_frame(series_id: str, rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": series_id,
                "observation_date": pd.Timestamp(observation_date),
                "value": value,
                "realtime_start": pd.Timestamp(realtime_start),
                "realtime_end": "9999-12-31",
            }
            for observation_date, value, realtime_start in rows
        ]
    )


def _write_monthly_vintage(path: Path, series_id: str, start: float, step: float) -> None:
    observations = pd.date_range("2020-01-01", periods=18, freq="MS")
    rows = []
    for idx, observation_date in enumerate(observations):
        rows.append(
            {
                "series_id": series_id,
                "observation_date": observation_date,
                "value": start + idx * step,
                "realtime_start": observation_date + pd.DateOffset(months=1),
                "realtime_end": "9999-12-31",
            }
        )
    pd.DataFrame(rows).to_parquet(path)
