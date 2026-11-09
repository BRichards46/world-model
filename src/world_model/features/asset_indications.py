from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ASSET_ORDER = [
    "equity",
    "rates_duration",
    "usd_fx",
    "non_usd_fx",
    "energy",
    "precious_metals",
    "industrial_metals",
    "grains_oilseeds",
    "softs",
    "livestock_dairy",
    "housing",
]

OUTCOME_KEYS = {
    "forward_return": "return",
    "future_volatility": "volatility",
    "max_drawdown": "drawdown",
}


def build_asset_indications(
    snapshot: pd.DataFrame,
    risk_overlay: dict[str, Any],
    summary_path: str | Path,
    deadband_percentile_points: float = 10.0,
    regime_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Invert robust macro findings into current asset-centric indications."""
    path = Path(summary_path).expanduser()
    if not path.exists():
        return {"assets": [], "note": "Standalone market-relevance output is unavailable."}
    summary = pd.read_csv(path)
    required = {
        "asset_group",
        "asset_group_label",
        "target",
        "horizon",
        "indicator_id",
        "indicator_label",
        "metric",
        "feature_realized_corr",
        "robust_candidate",
        "sample_end",
    }
    if not required <= set(summary.columns):
        return {"assets": [], "note": "Standalone market-relevance output is incomplete."}
    robust = summary.loc[_truthy(summary["robust_candidate"])].copy()
    current = snapshot.drop_duplicates("indicator_id", keep="last").set_index("indicator_id")
    assets: list[dict[str, Any]] = []

    for (asset_id, asset_label), asset_frame in robust.groupby(
        ["asset_group", "asset_group_label"], sort=False
    ):
        rows: list[dict[str, Any]] = []
        for indicator_id, indicator_frame in asset_frame.groupby("indicator_id", sort=False):
            snapshot_row = current.loc[indicator_id] if indicator_id in current.index else None
            outcomes: dict[str, dict[str, Any]] = {}
            for target, target_frame in indicator_frame.groupby("target", sort=False):
                key = OUTCOME_KEYS.get(str(target))
                if key is None:
                    continue
                outcomes[key] = _summarize_outcome(
                    target=str(target),
                    frame=target_frame,
                    snapshot_row=snapshot_row,
                    deadband=deadband_percentile_points / 50.0,
                )
            rows.append(
                {
                    "indicator_id": str(indicator_id),
                    "label": str(indicator_frame["indicator_label"].iloc[-1]),
                    "component": str(indicator_frame.get("component", pd.Series([""])).iloc[-1]),
                    "current": _current_summary(snapshot_row),
                    "risk_gate": None,
                    "outcomes": outcomes,
                    "evidence": {
                        "variants": int(len(indicator_frame)),
                        "horizons": _ordered_horizon_labels(indicator_frame["horizon"]),
                        "sample_end": _date_string(indicator_frame["sample_end"]),
                        "source": "Standalone screen",
                    },
                }
            )
        assets.append(
            {
                "id": str(asset_id),
                "label": str(asset_label),
                "rows": rows,
                "overlay": None,
            }
        )

    by_id = {asset["id"]: asset for asset in assets}
    equity = by_id.setdefault(
        "equity",
        {"id": "equity", "label": "U.S. equity index futures (ES + NQ)", "rows": [], "overlay": None},
    )
    _merge_equity_overlay(equity, risk_overlay)
    equity["combined_reads"] = _build_combined_reads(snapshot, regime_summary_path)
    for asset in by_id.values():
        asset["rows"] = sorted(
            asset["rows"],
            key=lambda row: (0 if row.get("risk_gate") else 1, row["label"]),
        )
        asset["summary"] = _asset_summary(asset["rows"])

    ordered = sorted(
        by_id.values(),
        key=lambda asset: (
            ASSET_ORDER.index(asset["id"]) if asset["id"] in ASSET_ORDER else len(ASSET_ORDER),
            asset["label"],
        ),
    )
    return {
        "as_of_date": risk_overlay.get("as_of_date"),
        "assets": ordered,
        "note": "Current indications inferred from robust point-in-time associations. Correlated variants are collapsed by macro series, asset, and outcome; mixed signs remain mixed. This is not a live forecast or aggregate trading score.",
    }


def _build_combined_reads(
    snapshot: pd.DataFrame,
    summary_path: str | Path | None,
) -> list[dict[str, Any]]:
    if not summary_path:
        return []
    path = Path(summary_path).expanduser()
    if not path.exists():
        return []
    summary = pd.read_csv(path)
    required = {
        "combination_id",
        "combination_label",
        "growth_indicator",
        "inflation_indicator",
        "inflation_condition",
        "target",
        "horizon",
        "additive_robust_candidate",
        "robust_candidate",
        "active_regime_mean_outcome",
        "inactive_regime_mean_outcome",
        "sample_end",
    }
    if not required <= set(summary.columns):
        return []
    current = snapshot.drop_duplicates("indicator_id", keep="last").set_index("indicator_id")
    robust = summary.loc[_truthy(summary["additive_robust_candidate"])].copy()
    reads: list[dict[str, Any]] = []
    for combination_id, frame in robust.groupby("combination_id", sort=False):
        identity = frame.iloc[0]
        growth_id = str(identity["growth_indicator"])
        inflation_id = str(identity["inflation_indicator"])
        if growth_id not in current.index or inflation_id not in current.index:
            continue
        growth = current.loc[growth_id]
        inflation = current.loc[inflation_id]
        growth_velocity = _number(growth.get("velocity_value"))
        condition = str(identity["inflation_condition"])
        # The research found that inflation levels, rather than a single latest
        # acceleration, carry the cleaner risk information. Keep the dashboard
        # focused on those higher-confidence level combinations.
        if condition != "elevated" or len(frame) < 2:
            continue
        inflation_value = (
            _number(inflation.get("velocity_value"))
            if condition == "rising"
            else _number(inflation.get("level_percentile_10y"))
        )
        active = (
            growth_velocity is not None
            and growth_velocity < 0.0
            and inflation_value is not None
            and ((condition == "rising" and inflation_value > 0.0) or (condition == "elevated" and inflation_value >= 60.0))
        )
        if not active:
            continue
        outcomes: list[dict[str, Any]] = []
        for target, target_frame in frame.groupby("target", sort=False):
            active_mean = _number(target_frame["active_regime_mean_outcome"].mean())
            inactive_mean = _number(target_frame["inactive_regime_mean_outcome"].mean())
            if active_mean is None or inactive_mean is None:
                continue
            adverse = (
                (target == "future_volatility" and active_mean > inactive_mean)
                or (target == "max_drawdown" and active_mean < inactive_mean)
                or (target == "forward_return" and active_mean < inactive_mean)
            )
            if not adverse:
                continue
            labels = {
                "future_volatility": "Higher volatility",
                "max_drawdown": "Deeper drawdowns",
                "forward_return": "Lower returns",
            }
            outcomes.append(
                {
                    "target": str(target),
                    "label": labels[str(target)],
                    "horizons": _ordered_horizon_labels(target_frame["horizon"]),
                    "variants": int(len(target_frame)),
                }
            )
        if not outcomes:
            continue
        interaction_robust = bool(_truthy(frame["robust_candidate"]).any())
        reads.append(
            {
                "id": str(combination_id),
                "label": str(identity["combination_label"]),
                "tone": "negative",
                "outcomes": outcomes,
                "evidence": {
                    "variants": int(len(frame)),
                    "sample_end": _date_string(frame["sample_end"]),
                    "model": "Additive pair",
                    "interaction": "validated" if interaction_robust else "not incremental",
                },
            }
        )
    return reads


def _summarize_outcome(
    target: str,
    frame: pd.DataFrame,
    snapshot_row: pd.Series | None,
    deadband: float,
) -> dict[str, Any]:
    effects: list[float] = []
    for record in frame.itertuples(index=False):
        position = _feature_position(snapshot_row, str(record.metric))
        correlation = _number(record.feature_realized_corr)
        if position is not None and correlation is not None:
            effects.append(float(np.sign(correlation) * position))
    nonneutral = [value for value in effects if abs(value) > deadband]
    mixed = any(value > deadband for value in nonneutral) and any(value < -deadband for value in nonneutral)
    effect = float(np.median(effects)) if effects else None
    if effect is None:
        state = "unavailable"
    elif mixed:
        state = "mixed"
    elif abs(effect) <= deadband:
        state = "neutral"
    elif target == "forward_return":
        state = "supportive" if effect > 0 else "adverse"
    elif target == "future_volatility":
        state = "higher_vol" if effect > 0 else "lower_vol"
    else:
        state = "reduced_drawdown" if effect > 0 else "elevated_drawdown"
    labels = {
        "supportive": "Supportive",
        "adverse": "Adverse",
        "higher_vol": "Higher vol",
        "lower_vol": "Lower vol",
        "reduced_drawdown": "Reduced risk",
        "elevated_drawdown": "Elevated risk",
        "neutral": "Neutral",
        "mixed": "Mixed",
        "unavailable": "Unavailable",
    }
    tones = {
        "supportive": "positive",
        "lower_vol": "positive",
        "reduced_drawdown": "positive",
        "adverse": "negative",
        "higher_vol": "negative",
        "elevated_drawdown": "negative",
        "mixed": "warning",
        "neutral": "neutral",
        "unavailable": "unavailable",
    }
    return {
        "state": state,
        "label": labels[state],
        "tone": tones[state],
        "variants": int(len(frame)),
        "horizons": _ordered_horizon_labels(frame["horizon"]),
    }


def _feature_position(snapshot_row: pd.Series | None, metric: str) -> float | None:
    if snapshot_row is None:
        return None
    level = _number(snapshot_row.get("level_percentile_10y"))
    velocity = _number(snapshot_row.get("velocity_percentile_10y"))
    if metric in {"level_value", "level_percentile_10y"}:
        return None if level is None else (level - 50.0) / 50.0
    if metric in {"velocity_value", "velocity_percentile_10y"}:
        return None if velocity is None else (velocity - 50.0) / 50.0
    if metric == "level_velocity_interaction":
        if level is None or velocity is None:
            return None
        return ((level - 50.0) / 50.0) * ((velocity - 50.0) / 50.0)
    return None


def _current_summary(snapshot_row: pd.Series | None) -> dict[str, Any]:
    if snapshot_row is None:
        return {"level_percentile": None, "velocity_percentile": None}
    return {
        "level_percentile": _number(snapshot_row.get("level_percentile_10y")),
        "velocity_percentile": _number(snapshot_row.get("velocity_percentile_10y")),
    }


def _merge_equity_overlay(equity: dict[str, Any], overlay: dict[str, Any]) -> None:
    equity["overlay"] = {
        "state_id": overlay.get("state_id"),
        "state_label": overlay.get("state_label"),
        "message": overlay.get("message"),
    }
    rows_by_id = {row["indicator_id"]: row for row in equity["rows"]}
    mappings = {
        "unemployment_trend": ("unemployment_rate", "Unemployment trend"),
        "industrial_production": ("industrial_production", "Industrial production"),
        "manufacturing_hours": ("manufacturing_hours", "Manufacturing hours"),
        "real_durable_orders": ("real_durable_orders", "Real durable orders"),
    }
    for item in overlay.get("inputs", []):
        if item.get("id") not in mappings:
            continue
        indicator_id, label = mappings[str(item["id"])]
        row = rows_by_id.get(indicator_id)
        if row is None:
            row = {
                "indicator_id": indicator_id,
                "label": label,
                "component": "activity_labor",
                "current": {
                    "display": _overlay_value(item),
                    "observation_date": item.get("observation_date"),
                },
                "risk_gate": None,
                "outcomes": {},
                "evidence": {
                    "variants": 0,
                    "horizons": [],
                    "sample_end": None,
                    "source": "GTT threshold",
                },
            }
            equity["rows"].append(row)
            rows_by_id[indicator_id] = row
        else:
            row["current"]["display"] = _overlay_value(item)
            row["current"]["observation_date"] = item.get("observation_date")
        warning = item.get("status") == "warning"
        role = str(item.get("role", "primary"))
        role_label = {
            "primary": "Core macro gate",
            "early warning": "Early warning",
            "confirmation": "Confirmation",
        }.get(role, role.title())
        row["risk_gate"] = {
            "state": "active" if warning else "clear",
            "label": "Active" if warning else "Clear",
            "tone": "negative" if warning else "positive",
            "role": role_label,
        }


def _asset_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "supportive": 0,
        "adverse": 0,
        "higher_risk": 0,
        "lower_risk": 0,
        "mixed": 0,
        "active_gates": 0,
    }
    for row in rows:
        gate = row.get("risk_gate")
        if gate and gate.get("state") == "active":
            counts["active_gates"] += 1
        for key, outcome in row.get("outcomes", {}).items():
            state = outcome.get("state")
            if state == "supportive":
                counts["supportive"] += 1
            elif state == "adverse":
                counts["adverse"] += 1
            elif state in {"higher_vol", "elevated_drawdown"}:
                counts["higher_risk"] += 1
            elif state in {"lower_vol", "reduced_drawdown"}:
                counts["lower_risk"] += 1
            elif state == "mixed":
                counts["mixed"] += 1
    return counts


def _overlay_value(item: dict[str, Any]) -> str:
    value = _number(item.get("value"))
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return f"{sign}{abs(value):.2f} {item.get('unit', '')}".strip()


def _ordered_horizon_labels(series: pd.Series) -> list[str]:
    order = ["one_month", "three_months", "six_months", "twelve_months"]
    labels = {"one_month": "1m", "three_months": "3m", "six_months": "6m", "twelve_months": "12m"}
    values = set(series.dropna().astype(str))
    return [labels.get(value, value) for value in order if value in values]


def _date_string(series: pd.Series) -> str | None:
    value = pd.to_datetime(series, errors="coerce").max()
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})
