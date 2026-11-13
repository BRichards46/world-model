from __future__ import annotations

import pandas as pd

from world_model.features.macro_dashboard_details import (
    classify_curve_move,
    compute_curve_move_classification,
    compute_drilldown_metrics,
    point_in_time_series,
)


def test_point_in_time_series_uses_latest_known_vintage() -> None:
    vintage = pd.DataFrame(
        {
            "observation_date": ["2026-01-01", "2026-01-01", "2026-02-01"],
            "realtime_start": ["2026-02-01", "2026-03-01", "2026-03-01"],
            "value": [100.0, 101.0, 102.0],
        }
    )

    result = point_in_time_series(vintage)

    assert result.loc[pd.Timestamp("2026-01-01")] == 101.0
    assert result.loc[pd.Timestamp("2026-02-01")] == 102.0


def test_payroll_drilldown_includes_familiar_job_changes() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="MS")
    metrics = compute_drilldown_metrics(
        "payroll_growth",
        {"PAYEMS": pd.Series([1000.0, 1010.0, 1025.0, 1045.0, 1070.0], index=dates)},
    )

    assert [metric["label"] for metric in metrics] == [
        "Latest monthly payroll change",
        "Three-month average change",
    ]
    assert metrics[0]["value"] == 25.0
    assert metrics[1]["value"] == 20.0
    assert metrics[0]["observation_date"] == "2026-05-01"


def test_inflation_drilldown_leads_with_mom_and_yoy() -> None:
    dates = pd.date_range("2024-01-01", periods=15, freq="MS")
    values = {
        series_id: pd.Series(range(100, 115), index=dates, dtype=float)
        for series_id in ["PCEPILFE", "CPIAUCSL", "CPILFESL", "CPIENGSL", "CPIUFDSL"]
    }

    pce_metrics = compute_drilldown_metrics("core_pce_inflation", values)
    cpi_metrics = compute_drilldown_metrics("core_cpi_inflation", values)

    assert [metric["unit"] for metric in pce_metrics] == ["% MoM", "% YoY", "% 3m annualized"]
    assert [metric["source"] for metric in pce_metrics] == ["PCEPILFE"] * 3
    assert [metric["unit"] for metric in cpi_metrics] == ["% MoM", "% YoY", "% 3m annualized"]
    assert [metric["source"] for metric in cpi_metrics] == ["CPILFESL"] * 3
    assert all("percentile" not in metric for metric in [*pce_metrics, *cpi_metrics])


def test_inflation_drilldown_does_not_bridge_a_missing_calendar_month() -> None:
    dates = pd.date_range("2024-01-01", periods=15, freq="MS").delete(5)
    values = {"CPILFESL": pd.Series(range(100, 114), index=dates, dtype=float)}

    metrics = compute_drilldown_metrics("core_cpi_inflation", values)
    by_unit = {metric["unit"]: metric for metric in metrics}

    assert "% YoY" in by_unit
    assert by_unit["% YoY"]["observation_date"] == "2025-03-01"


def test_industrial_drilldown_adds_signals_and_diagnostic_components() -> None:
    dates = pd.date_range("2024-01-01", periods=15, freq="MS")
    values = {
        "INDPRO": pd.Series([110.0] * 12 + [109.0, 108.0, 107.0], index=dates),
        "AWHMAN": pd.Series([40.0] * 12 + [39.9, 39.8, 39.7], index=dates),
        "DGORDER": pd.Series([200.0] * 12 + [195.0, 190.0, 185.0], index=dates),
        "CPIAUCSL": pd.Series([100.0] * 15, index=dates),
        "IPMAN": pd.Series(range(100, 115), index=dates, dtype=float),
        "IPFINAL": pd.Series(range(100, 115), index=dates, dtype=float),
        "IPBUSEQ": pd.Series(range(100, 115), index=dates, dtype=float),
        "IPMAT": pd.Series(range(100, 115), index=dates, dtype=float),
    }

    metrics = compute_drilldown_metrics("industrial_production", values)
    by_label = {metric["label"]: metric for metric in metrics}

    assert by_label["Industrial production growth"]["signal_status"] == "warning"
    assert by_label["Workweek gap vs 12m average"]["signal_status"] == "warning"
    assert by_label["Real durable-goods orders"]["signal_status"] == "warning"
    assert by_label["Manufacturing production"]["signal_status"] == "diagnostic"


def test_curve_move_classifier_covers_canonical_and_twist_moves() -> None:
    cases = [
        ((30.0, 10.0), "bear_steepener"),
        ((-10.0, -30.0), "bull_steepener"),
        ((10.0, 30.0), "bear_flattener"),
        ((-30.0, -10.0), "bull_flattener"),
        ((20.0, -20.0), "twist_steepener"),
        ((-20.0, 20.0), "twist_flattener"),
        ((25.0, 23.0), "parallel_selloff"),
        ((-25.0, -23.0), "parallel_rally"),
        ((2.0, -1.0), "stable"),
    ]
    for changes, expected in cases:
        assert classify_curve_move(*changes)[0] == expected


def test_curve_move_classification_uses_same_20_by_63_window_as_dashboard() -> None:
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    values = {
        "DGS10": pd.Series(4.0 + pd.Series(range(100), index=dates) * 0.01, index=dates),
        "DGS3MO": pd.Series(4.0 + pd.Series(range(100), index=dates) * 0.004, index=dates),
    }
    result = compute_curve_move_classification(values)

    assert result["curve_move_id"] == "bear_steepener"
    assert result["curve_long_change_bps"] > result["curve_short_change_bps"] > 0.0


def test_credit_drilldown_shows_quality_ladder_and_dispersion() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    values = {
        "BAMLC0A0CM": pd.Series([0.8, 0.9, 1.0], index=dates),
        "BAMLC0A4CBBB": pd.Series([1.0, 1.1, 1.2], index=dates),
        "BAMLH0A0HYM2": pd.Series([3.0, 3.2, 3.5], index=dates),
        "BAMLH0A3HYC": pd.Series([9.0, 9.5, 10.0], index=dates),
        "BAA10Y": pd.Series([1.4, 1.5, 1.6], index=dates),
    }

    metrics = compute_drilldown_metrics("high_yield_spread", values)
    by_label = {metric["label"]: metric for metric in metrics}

    assert by_label["Investment-grade OAS"]["value"] == 1.0
    assert by_label["High-yield OAS"]["value"] == 3.5
    assert by_label["HY minus IG quality gap"]["value"] == 2.5
    assert by_label["CCC minus broad HY gap"]["value"] == 6.5
