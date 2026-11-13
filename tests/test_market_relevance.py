from __future__ import annotations

import pandas as pd

from world_model.features.market_relevance import load_market_relevance


def test_market_relevance_keeps_only_robust_screened_relationships(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "indicator_id": "unemployment_rate",
                "asset_group_label": "Equity index futures",
                "target": "future_volatility",
                "horizon": "one_month",
                "metric": "velocity_value",
                "robust_candidate": True,
                "sample_end": "2025-09-30",
            },
            {
                "indicator_id": "unemployment_rate",
                "asset_group_label": "Equity index futures",
                "target": "future_volatility",
                "horizon": "three_months",
                "metric": "velocity_percentile_10y",
                "robust_candidate": True,
                "sample_end": "2025-09-30",
            },
            {
                "indicator_id": "unemployment_rate",
                "asset_group_label": "Energy futures",
                "target": "forward_return",
                "horizon": "one_month",
                "metric": "level_value",
                "robust_candidate": False,
                "sample_end": "2025-09-30",
            },
        ]
    ).to_csv(path, index=False)

    result = load_market_relevance(path)
    relevance = result["unemployment_rate"]

    assert relevance["robust_test_variants"] == 2
    assert relevance["directional_test_variants"] == 0
    assert relevance["relationship_count"] == 1
    assert relevance["relationships"][0]["horizons"] == ["1m", "3m"]
    assert relevance["relationships"][0]["features"] == ["velocity percentile", "velocity"]
