from __future__ import annotations

import json

import pandas as pd

from world_model.cli.macro_dashboard import _write_site_data
from world_model.states.macro_dashboard import (
    MacroDashboardEngine,
    merge_macro_dashboard_history,
    read_macro_dashboard_table,
    write_macro_dashboard_table,
)


def test_engine_builds_normalized_history_and_current_snapshot(tmp_path) -> None:
    fred_root = tmp_path / "fred"
    vintage_root = fred_root / "vintage"
    vintage_root.mkdir(parents=True)
    observations = pd.date_range("2019-01-01", periods=24, freq="MS")
    releases = observations + pd.DateOffset(months=1, days=14)
    pd.DataFrame(
        {
            "series_id": "TEST",
            "observation_date": observations,
            "value": 100.0 + pd.Series(range(len(observations)), dtype=float),
            "realtime_start": releases,
            "realtime_end": "9999-12-31",
        }
    ).to_parquet(vintage_root / "TEST.parquet", index=False)

    config = {
        "build": {"bootstrap_history_years": 1, "incremental_overlap_days": 14},
        "history": {
            "primary_window_years": 10,
            "minimum_history_years": None,
            "minimum_observations": {"monthly": 2},
            "baseline_sensitive_threshold_pct_points": 25,
            "flat_velocity_percentile": 20,
        },
        "freshness": {"stale_after_days": {"monthly": 50}},
        "rows": [
            {
                "id": "test_growth",
                "order": 1,
                "label": "Test growth",
                "component": "activity",
                "series_ids": ["TEST"],
                "frequency": "monthly",
                "polarity": "higher_strength",
                "recipe": "three_month_growth",
            }
        ],
    }
    engine = MacroDashboardEngine({"fred": str(fred_root)}, config)

    result = engine.build(as_of="2021-01-20")

    assert result.as_of_date == pd.Timestamp("2021-01-20")
    assert set(result.history["indicator_id"]) == {"test_growth"}
    assert result.history["realtime_start"].min() >= pd.Timestamp("2020-01-20")
    assert result.snapshot.loc[0, "indicator_id"] == "test_growth"
    assert result.snapshot.loc[0, "observation_date"] == pd.Timestamp("2020-12-01")
    assert result.snapshot.loc[0, "days_since_release"] == 5
    assert result.snapshot.loc[0, "freshness_status"] == "fresh"
    assert result.snapshot.loc[0, "history_status"] == "available"
    assert json.loads(result.snapshot.loc[0, "detail_metrics"]) == []
    assert result.risk_overlay == {}
    assert result.asset_indications == {"assets": []}

    history_path = tmp_path / "history.parquet"
    snapshot_path = tmp_path / "snapshot.csv"
    write_macro_dashboard_table(result.history, history_path)
    write_macro_dashboard_table(result.snapshot, snapshot_path)
    assert len(read_macro_dashboard_table(history_path)) == len(result.history)
    assert len(read_macro_dashboard_table(snapshot_path)) == 1


def test_history_merge_replaces_overlapping_indicator_releases() -> None:
    existing = pd.DataFrame(
        {
            "indicator_id": ["a", "a", "a"],
            "realtime_start": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-02-15"]),
            "display_order": [1.0, 1.0, 1.0],
            "level_value": [1.0, 2.0, 2.5],
        }
    )
    new = pd.DataFrame(
        {
            "indicator_id": ["a", "a"],
            "realtime_start": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "display_order": [1.0, 1.0],
            "level_value": [20.0, 3.0],
        }
    )

    result = merge_macro_dashboard_history(existing, new, replace_from="2024-02-01")

    assert len(result) == 3
    assert result.loc[result["realtime_start"] == pd.Timestamp("2024-02-01"), "level_value"].item() == 20.0
    assert pd.Timestamp("2024-02-15") not in set(result["realtime_start"])


def test_write_site_data_emits_snapshot_and_compact_history(tmp_path) -> None:
    snapshot = pd.DataFrame(
        {
            "indicator_id": ["payroll_growth"],
            "as_of_date": [pd.Timestamp("2026-10-20")],
            "level_value": [2.5],
        }
    )
    history = pd.DataFrame(
        {
            "indicator_id": ["payroll_growth", "payroll_growth"],
            "realtime_start": pd.to_datetime(["2026-09-04", "2026-10-02"]),
            "level_percentile_10y": [45.0, 55.0],
            "velocity_percentile_10y": [40.0, 60.0],
            "level_value": [2.0, 2.5],
            "velocity_value": [0.1, 0.2],
            "internal_only": ["omit", "omit"],
        }
    )

    overlay = {"state_id": "normal", "state_label": "Normal"}
    assets = {"assets": [{"id": "equity"}]}
    _write_site_data(snapshot, history, overlay, assets, tmp_path)

    snapshot_json = pd.read_json(tmp_path / "snapshot.json")
    history_json = pd.read_json(tmp_path / "history.json")
    assert snapshot_json.loc[0, "indicator_id"] == "payroll_growth"
    assert len(history_json) == 2
    assert "internal_only" not in history_json
    assert list(history_json["level_percentile_10y"]) == [45, 55]
    assert json.loads((tmp_path / "risk_overlay.json").read_text()) == overlay
    assert json.loads((tmp_path / "asset_indications.json").read_text()) == assets
