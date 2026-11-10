from __future__ import annotations

from world_model.states.macro_outlook import build_macro_outlook


def _quad() -> dict:
    return {
        "as_of_date": "2026-11-10",
        "assumptions": {"inflation_base_mom": 0.22},
        "cpi_comps": [{"month": "2026-11", "comp_mom": 0.13, "comp_interpolated": True}],
        "backtest": {"horizons": [{"horizon": 0, "quad_hit_rate": 0.59}, {"horizon": 1, "quad_hit_rate": 0.46}]},
        "quarters": [
            {"quarter": "2026Q3", "status": "realized", "quad": 4},
            {
                "quarter": "2026Q4", "status": "partial", "is_current_quarter": True, "quad": 2,
                "probability_quad": 2, "probability_top": 0.89, "inflation_hurdle_mom": -0.29,
                "growth_hurdle_saar": 0.21, "growth_assumption_saar": 3.48, "growth_assumption_source": "nowcast",
                "inflation_yoy": 3.9,
            },
            {
                "quarter": "2027Q1", "status": "projected", "quad": 4, "probability_quad": 4, "probability_top": 0.38,
                "inflation_hurdle_mom": 0.35, "growth_hurdle_saar": 2.49, "growth_assumption_saar": 2.46,
            },
        ],
    }


def _overlay() -> dict:
    return {
        "as_of_date": "2026-11-10",
        "state_label": "Normal",
        "inputs": [{"id": "market_trend", "value": 2.8}, {"id": "industrial_production", "value": 2.2}],
        "triggers": {
            "market_trend": {"month": "2026-11", "threshold_change_pct": -3.1},
            "unemployment_trend": {"month": "2026-11", "threshold_rate": 4.3545, "latest_rate": 4.1},
            "industrial_production": {"month": "2026-10", "threshold_change_pct": -2.6},
        },
    }


def _snapshot() -> list[dict]:
    return [
        {"indicator_id": "core_cpi_inflation", "expected_next_release": "2026-12-10T00:00:00", "expected_next_release_is_estimate": True},
        {"indicator_id": "payroll_growth", "expected_next_release": "2026-12-06T00:00:00", "expected_next_release_is_estimate": True,
         "detail_metrics": '[{"label": "Three-month average change", "value": 137.0}]'},
        {"indicator_id": "industrial_production", "observation_date": "2026-09-01", "expected_next_release": "2026-11-16T00:00:00"},
    ]


def _releases() -> dict:
    return {
        "forecasts": [
            {"target_id": "headline_cpi_mom", "variant_id": "pipeline", "reference_month": "2026-11", "forecast": 0.33,
             "range_50": [0.16, 0.48], "range_80": [0.02, 0.68], "unit": "percent", "validated": False},
            {"target_id": "payroll_change", "variant_id": "core", "reference_month": "2026-11", "forecast": 178.0,
             "range_50": [85.0, 202.0], "range_80": [-20.0, 329.0], "unit": "thousand jobs", "validated": False},
        ]
    }


def test_outlook_orders_releases_and_explains_what_would_change() -> None:
    outlook = build_macro_outlook(_quad(), _overlay(), _releases(), _snapshot())
    assert [r["id"] for r in outlook["releases"]] == ["industrial_production", "jobs", "cpi"]
    jobs = next(r for r in outlook["releases"] if r["id"] == "jobs")
    assert jobs["date"] == "2026-12-04"  # a Sunday estimate moves to the Friday
    assert "4.4% or higher" in jobs["meaning"][0]
    assert any("negative payroll" in line for line in jobs["meaning"])
    cpi = next(r for r in outlook["releases"] if r["id"] == "cpi")
    assert cpi["forecasts"][0]["range_80"] == [0.02, 0.68]
    assert "+0.2 pp" in cpi["meaning"][0] and "already rising" in cpi["meaning"][1]
    assert "fall 2.6%" in outlook["releases"][0]["meaning"][0]


def test_outlook_states_regime_conviction_and_risk_distance() -> None:
    outlook = build_macro_outlook(_quad(), _overlay(), _releases(), _snapshot())
    assert [a["conviction"] for a in outlook["ahead"]] == ["high", "low"]
    assert outlook["ahead"][1]["historical_hit_rate"] == 0.46
    assert "tracking Reflation" in outlook["headline"] and "leans Deflation" in outlook["headline"]
    assert "low conviction" in outlook["headline"] and "Quad" not in outlook["headline"]
    assert [a["name"] for a in outlook["ahead"]] == ["Reflation", "Deflation"]
    risk = next(item for item in outlook["now"] if item["label"] == "Equity risk")
    assert "3.1% below today's level" in risk["text"]
    jobs = next(item for item in outlook["now"] if item["label"] == "Jobs")
    assert "+137k" in jobs["text"]


def test_outlook_tolerates_missing_payloads() -> None:
    outlook = build_macro_outlook(None, None, None, [])
    assert outlook["releases"] == [] and outlook["ahead"] == [] and outlook["now"] == []
