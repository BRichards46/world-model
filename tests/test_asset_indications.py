from __future__ import annotations

import pandas as pd

from world_model.features.asset_indications import build_asset_indications


def test_asset_indications_convert_robust_signs_to_current_states(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            _row("usd_fx", "U.S. dollar futures", "forward_return", "level_value", 0.20),
            _row("usd_fx", "U.S. dollar futures", "future_volatility", "level_value", -0.25),
            _row("non_usd_fx", "Non-USD currency futures", "forward_return", "level_value", -0.20),
        ]
    ).to_csv(summary_path, index=False)
    snapshot = pd.DataFrame(
        {
            "indicator_id": ["payroll_growth"],
            "level_percentile_10y": [80.0],
            "velocity_percentile_10y": [55.0],
        }
    )

    result = build_asset_indications(snapshot, {}, summary_path)
    assets = {asset["id"]: asset for asset in result["assets"]}

    usd = assets["usd_fx"]["rows"][0]
    assert usd["outcomes"]["return"]["state"] == "supportive"
    assert usd["outcomes"]["volatility"]["state"] == "lower_vol"
    non_usd = assets["non_usd_fx"]["rows"][0]
    assert non_usd["outcomes"]["return"]["state"] == "adverse"


def test_asset_indications_merge_equity_gtt_inputs(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame([_row("equity", "Equity index futures", "future_volatility", "velocity_value", 0.20)]).to_csv(
        summary_path, index=False
    )
    snapshot = pd.DataFrame(
        {
            "indicator_id": ["unemployment_rate"],
            "level_percentile_10y": [50.0],
            "velocity_percentile_10y": [70.0],
        }
    )
    overlay = {
        "as_of_date": "2026-10-23",
        "state_id": "macro_watch",
        "state_label": "Macro watch",
        "message": "Macro conditions warrant attention.",
        "inputs": [
            {
                "id": "unemployment_trend",
                "value": 0.2,
                "unit": "pp vs 12m avg",
                "status": "warning",
                "role": "primary",
                "observation_date": "2026-09-01",
            },
            {
                "id": "real_durable_orders",
                "value": 3.0,
                "unit": "% YoY",
                "status": "healthy",
                "role": "confirmation",
                "observation_date": "2026-08-01",
            },
        ],
    }

    result = build_asset_indications(snapshot, overlay, summary_path)
    equity = next(asset for asset in result["assets"] if asset["id"] == "equity")
    rows = {row["indicator_id"]: row for row in equity["rows"]}

    assert equity["overlay"]["state_label"] == "Macro watch"
    assert rows["unemployment_rate"]["risk_gate"]["state"] == "active"
    assert rows["unemployment_rate"]["outcomes"]["volatility"]["state"] == "higher_vol"
    assert rows["real_durable_orders"]["risk_gate"]["state"] == "clear"


def test_asset_indications_add_active_additive_regime_read(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    regime_path = tmp_path / "regimes.csv"
    pd.DataFrame([_row("equity", "Equity index futures", "future_volatility", "velocity_value", 0.20)]).to_csv(
        summary_path, index=False
    )
    regime_row = {
                "combination_id": "indpro_pce_high",
                "combination_label": "Slowing production × elevated core PCE",
                "growth_indicator": "industrial_production",
                "inflation_indicator": "core_pce_inflation",
                "inflation_condition": "elevated",
                "target": "future_volatility",
                "horizon": "one_month",
                "additive_robust_candidate": True,
                "robust_candidate": False,
                "active_regime_mean_outcome": 0.14,
                "inactive_regime_mean_outcome": 0.10,
                "sample_end": "2026-05-29",
            }
    pd.DataFrame(
        [
            regime_row,
            {
                **regime_row,
                "target": "max_drawdown",
                "active_regime_mean_outcome": -0.04,
                "inactive_regime_mean_outcome": -0.02,
            },
        ]
    ).to_csv(regime_path, index=False)
    snapshot = pd.DataFrame(
        {
            "indicator_id": ["industrial_production", "core_pce_inflation"],
            "level_percentile_10y": [70.0, 66.0],
            "velocity_percentile_10y": [35.0, 84.0],
            "velocity_value": [-1.9, 0.12],
        }
    )

    result = build_asset_indications(
        snapshot,
        {},
        summary_path,
        regime_summary_path=regime_path,
    )
    equity = next(asset for asset in result["assets"] if asset["id"] == "equity")

    assert equity["combined_reads"][0]["label"] == "Slowing production × elevated core PCE"
    assert equity["combined_reads"][0]["outcomes"][0]["label"] == "Higher volatility"
    assert equity["combined_reads"][0]["evidence"]["interaction"] == "not incremental"


def _row(
    asset_group: str,
    asset_label: str,
    target: str,
    metric: str,
    correlation: float,
) -> dict[str, object]:
    return {
        "asset_group": asset_group,
        "asset_group_label": asset_label,
        "target": target,
        "horizon": "one_month",
        "indicator_id": "payroll_growth" if asset_group != "equity" else "unemployment_rate",
        "indicator_label": "Payroll growth" if asset_group != "equity" else "Unemployment rate",
        "component": "activity_labor",
        "metric": metric,
        "feature_realized_corr": correlation,
        "robust_candidate": True,
        "sample_end": "2026-05-29",
    }
