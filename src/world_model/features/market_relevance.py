from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_market_relevance(
    summary_path: str | Path,
    max_relationships: int = 4,
) -> dict[str, dict[str, Any]]:
    """Summarize reviewed standalone macro/market findings for dashboard display."""
    path = Path(summary_path).expanduser()
    if not path.exists():
        return {}
    summary = pd.read_csv(path)
    required = {
        "indicator_id",
        "asset_group_label",
        "target",
        "horizon",
        "metric",
        "robust_candidate",
        "sample_end",
    }
    if not required <= set(summary.columns):
        return {}
    robust = summary.loc[_truthy(summary["robust_candidate"])].copy()
    if robust.empty:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for indicator_id, frame in robust.groupby("indicator_id", sort=False):
        relationships: list[dict[str, Any]] = []
        grouped = frame.groupby(["asset_group_label", "target"], sort=False)
        for (asset, target), group in grouped:
            relationships.append(
                {
                    "asset": str(asset),
                    "outcome": _outcome_label(str(target)),
                    "horizons": [_horizon_label(value) for value in _ordered_horizons(group["horizon"])],
                    "features": [_metric_label(value) for value in sorted(set(group["metric"].astype(str)))],
                    "test_variants": int(len(group)),
                }
            )
        relationships.sort(
            key=lambda item: (
                -item["test_variants"],
                item["outcome"] != "Future volatility",
                item["asset"],
            )
        )
        directional = int(frame["target"].eq("forward_return").sum())
        result[str(indicator_id)] = {
            "robust_test_variants": int(len(frame)),
            "relationship_count": int(len(relationships)),
            "directional_test_variants": directional,
            "risk_test_variants": int(len(frame) - directional),
            "sample_end": str(pd.to_datetime(frame["sample_end"], errors="coerce").max().date()),
            "relationships": relationships[:max_relationships],
            "has_more": len(relationships) > max_relationships,
            "note": "Standalone point-in-time screen. Association with future outcomes, not a live forecast or trading rule. Test variants are correlated and are not independent confirmations.",
        }
    return result


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _ordered_horizons(series: pd.Series) -> list[str]:
    order = ["one_month", "three_months", "six_months", "twelve_months"]
    values = set(series.dropna().astype(str))
    return [value for value in order if value in values] + sorted(values - set(order))


def _horizon_label(value: str) -> str:
    return {
        "one_month": "1m",
        "three_months": "3m",
        "six_months": "6m",
        "twelve_months": "12m",
    }.get(value, value.replace("_", " "))


def _metric_label(value: str) -> str:
    return {
        "level_value": "level",
        "velocity_value": "velocity",
        "level_percentile_10y": "level percentile",
        "velocity_percentile_10y": "velocity percentile",
        "level_velocity_interaction": "level × velocity",
    }.get(value, value.replace("_", " "))


def _outcome_label(value: str) -> str:
    return {
        "future_volatility": "Future volatility",
        "max_drawdown": "Maximum drawdown",
        "forward_return": "Forward return",
    }.get(value, value.replace("_", " ").title())
