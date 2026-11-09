from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from world_model.data.norgate import read_catalog_price_frame
from world_model.data.tanuki import latest_by_symbol, read_source_table, select_delta_expiry
from world_model.features.premium_selling import conditional_path_snapshot, touch_statistics


@dataclass(frozen=True)
class PremiumSellingResult:
    payload: dict[str, Any]
    as_of_date: pd.Timestamp


class PremiumSellingEngine:
    def __init__(self, data_roots: dict[str, str], config: dict[str, Any]):
        self.data_roots = data_roots
        self.config = config

    def build(self, as_of: str | pd.Timestamp | None = None) -> PremiumSellingResult:
        as_of_date = pd.Timestamp(as_of).normalize() if as_of else pd.Timestamp.today().normalize()
        assets = self.config["assets"]
        option_symbols = [str(asset["options_symbol"]) for asset in assets]
        options = self._load_options(option_symbols, as_of_date)
        rows: list[dict[str, Any]] = []
        for asset in assets:
            prices = read_catalog_price_frame(
                self.data_roots["norgate_prices"],
                self.data_roots["norgate_metadata"],
                str(asset["underlying_symbol"]),
            ).loc[:as_of_date]
            path = conditional_path_snapshot(
                prices,
                self.config["horizons"],
                forward_days=int(self.config["forward_days"]),
                analog_count=int(self.config["analog_count"]),
            )
            rows.append(self._asset_payload(asset, path, options))

        rows.sort(key=lambda item: (item["group_order"], item["order"]))
        payload = {
            "as_of_date": as_of_date.date().isoformat(),
            "forward_days": int(self.config["forward_days"]),
            "assets": rows,
            "uncovered_futures": self.config.get("uncovered_futures", []),
            "methodology": {
                "path": "Nearest historical weekly analogs using visible 1m, 3m, 6m, 10m, and 12m volatility-adjusted momentum; outcomes are measured over the next 21 sessions.",
                "touch": "Touch rates count whether the underlying crossed today's approximately 30-DTE 16-delta distance during comparable historical paths. They are not option assignment probabilities.",
                "volatility": "IV edge compares current IVX with median realized volatility during the analog paths; trailing IV-minus-HV is shown separately.",
                "evidence": "Grade B means the underlying path is historically tested and current option pricing is observed, but actual option P&L is not backtested for this asset.",
                "warning": "Research screen only. Strategy labels are judgment aids, not trade instructions; transaction costs, early exits, skew changes, and gap risk are not modeled.",
            },
        }
        return PremiumSellingResult(payload, as_of_date)

    def _load_options(self, symbols: list[str], as_of_date: pd.Timestamp) -> dict[str, Any]:
        root = self.data_roots["tanuki_options"]
        tables = {name: read_source_table(root, name, symbols) for name in ["ivx", "iv_rank", "realized_volatility", "call_put_skew", "expected_move_delta_ladder"]}
        for name in ["ivx", "iv_rank", "realized_volatility", "call_put_skew"]:
            frame = tables[name]
            if not frame.empty:
                dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.tz_localize(None)
                tables[name] = frame.loc[dates <= as_of_date].copy()
        ladder = tables["expected_move_delta_ladder"]
        if not ladder.empty:
            dates = pd.to_datetime(ladder["snapshot_at"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
            ladder = ladder.loc[dates <= as_of_date].copy()
        selected_ladder = select_delta_expiry(ladder, **self.config["expiry_selection"])
        return {
            "latest": {name: _rows_by_symbol(latest_by_symbol(frame)) for name, frame in tables.items() if name != "expected_move_delta_ladder"},
            "history": tables,
            "ladder": _rows_by_symbol(selected_ladder),
        }

    def _asset_payload(self, asset: dict[str, Any], path: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        symbol = str(asset["options_symbol"])
        ivx = options["latest"]["ivx"].get(symbol, {})
        iv_rank = options["latest"]["iv_rank"].get(symbol, {})
        realized = options["latest"]["realized_volatility"].get(symbol, {})
        skew = options["latest"]["call_put_skew"].get(symbol, {})
        ladder = options["ladder"].get(symbol, {})
        conditional = path["conditional"]
        unconditional = path["unconditional"]
        put_distance = _number(ladder.get("ladder__16__put__distance_pct"))
        call_distance = _number(ladder.get("ladder__16__call__distance_pct"))
        conditional_touch = touch_statistics(path["analogs"], put_distance, call_distance) if put_distance is not None and call_distance is not None else {}
        unconditional_touch = touch_statistics(path["samples"], put_distance, call_distance) if put_distance is not None and call_distance is not None else {}
        direction = _direction_state(conditional, self.config["classification"])
        current_iv = _number(ivx.get("ivx_avg"))
        forecast_rv = _number(conditional.get("median_forward_rv"))
        forward_edge = current_iv - forecast_rv if current_iv is not None and forecast_rv is not None else None
        vol_state = _volatility_state(forward_edge, self.config["classification"])
        strategy = _strategy_state(direction, vol_state, conditional_touch, unconditional_touch, self.config["classification"])
        skew_history = options["history"]["call_put_skew"]
        symbol_skew = skew_history[skew_history["symbol"].astype(str).eq(symbol)] if not skew_history.empty else pd.DataFrame()
        skew_value = _number(skew.get("call_put_skew"))
        skew_percentile = _percentile(symbol_skew.get("call_put_skew", pd.Series(dtype=float)), skew_value)
        option_surface_available = current_iv is not None and put_distance is not None and call_distance is not None
        evidence = (
            {"grade": "B", "label": "Path-tested", "note": "Actual option P&L pending"}
            if option_surface_available
            else {"grade": "C", "label": "Path only", "note": "Option surface unavailable"}
        )
        return {
            "id": str(asset["id"]),
            "label": str(asset["label"]),
            "group": str(asset["group"]),
            "group_label": str(asset["group_label"]),
            "group_order": int(asset["group_order"]),
            "order": int(asset["order"]),
            "underlying_symbol": str(asset["underlying_symbol"]),
            "options_symbol": symbol,
            "observation_date": _date(path["observation_date"]),
            "direction": direction,
            "strategy": strategy,
            "components": {key: round(value, 4) for key, value in path["components"].items()},
            "path": {
                "analog_count": int(path["analog_count"]),
                "sample_count": int(path["sample_count"]),
                "sample_start": _date(path["sample_start"]),
                "sample_end": _date(path["sample_end"]),
                "median_return_pct": _pct(conditional["median_return"]),
                "positive_rate_pct": _pct(conditional["positive_rate"]),
                "median_downside_mae_pct": _pct(conditional["median_downside_mae"]),
                "median_upside_mfe_pct": _pct(conditional["median_upside_mfe"]),
                "unconditional_return_pct": _pct(unconditional["median_return"]),
            },
            "options": {
                "snapshot_date": _date(ivx.get("snapshot_date")),
                "expiration_date": _date(ladder.get("expiration_date")),
                "dte": _integer(ladder.get("dte")),
                "ivx": current_iv,
                "iv_rank": _number(iv_rank.get("iv_rank")),
                "hv30": _number(realized.get("hv_30d")),
                "trailing_iv_hv_spread": _number(realized.get("iv_hv_spread_30_vol_pts")),
                "forward_rv_forecast": forecast_rv,
                "forward_iv_edge": _round(forward_edge),
                "vol_state": vol_state,
                "skew_value": skew_value,
                "skew_percentile": _round(skew_percentile),
                "put_16d_distance_pct": put_distance,
                "call_16d_distance_pct": call_distance,
                "put_25d_distance_pct": _number(ladder.get("ladder__25__put__distance_pct")),
                "call_25d_distance_pct": _number(ladder.get("ladder__25__call__distance_pct")),
            },
            "touch": {
                "put_conditional_pct": _pct(conditional_touch.get("put_touch_rate")),
                "put_unconditional_pct": _pct(unconditional_touch.get("put_touch_rate")),
                "call_conditional_pct": _pct(conditional_touch.get("call_touch_rate")),
                "call_unconditional_pct": _pct(unconditional_touch.get("call_touch_rate")),
                "put_terminal_breach_pct": _pct(conditional_touch.get("put_terminal_breach_rate")),
                "call_terminal_breach_pct": _pct(conditional_touch.get("call_terminal_breach_rate")),
            },
            "evidence": evidence,
        }


def _rows_by_symbol(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["symbol"]): row.to_dict() for _, row in frame.iterrows()} if not frame.empty else {}


def _direction_state(summary: dict[str, float], config: dict[str, Any]) -> dict[str, str]:
    median = summary["median_return"] * 100
    positive_rate = summary["positive_rate"] * 100
    threshold = float(config["direction_return_pct"])
    rate = float(config["direction_positive_rate_pct"])
    if median >= threshold and positive_rate >= rate:
        return {"id": "positive", "label": "Positive path", "tone": "positive"}
    if median <= -threshold and positive_rate <= 100 - rate:
        return {"id": "negative", "label": "Negative path", "tone": "negative"}
    return {"id": "mixed", "label": "Mixed path", "tone": "neutral"}


def _volatility_state(edge: float | None, config: dict[str, Any]) -> dict[str, str]:
    if edge is None:
        return {"id": "unavailable", "label": "IV edge unavailable", "tone": "neutral"}
    rich = float(config["rich_iv_edge_vol_pts"])
    cheap = float(config["cheap_iv_edge_vol_pts"])
    if edge >= rich:
        return {"id": "rich", "label": "IV rich to analog RV", "tone": "positive"}
    if edge <= cheap:
        return {"id": "cheap", "label": "IV cheap to analog RV", "tone": "negative"}
    return {"id": "fair", "label": "IV near analog RV", "tone": "neutral"}


def _strategy_state(direction: dict[str, str], volatility: dict[str, str], conditional: dict[str, float], unconditional: dict[str, float], config: dict[str, Any]) -> dict[str, str]:
    if volatility["id"] != "rich":
        return {"id": "wait", "label": "Wait / limited premium edge", "tone": "neutral", "reason": volatility["label"]}
    put = conditional.get("put_touch_rate")
    call = conditional.get("call_touch_rate")
    put_base = unconditional.get("put_touch_rate")
    call_base = unconditional.get("call_touch_rate")
    improvement = float(config["minimum_touch_improvement_pct"]) / 100
    put_helpful = put is not None and put_base is not None and put <= put_base - improvement
    call_helpful = call is not None and call_base is not None and call <= call_base - improvement
    if direction["id"] == "positive" and put_helpful:
        return {"id": "short_put", "label": "Short-put lean", "tone": "positive", "reason": "Positive path and lower conditional put-touch rate"}
    if direction["id"] == "negative" and call_helpful:
        return {"id": "short_call", "label": "Short-call lean", "tone": "negative", "reason": "Negative path and lower conditional call-touch rate"}
    ceiling = float(config["strangle_touch_ceiling_pct"]) / 100
    if direction["id"] == "mixed" and put is not None and call is not None and put <= ceiling and call <= ceiling:
        return {"id": "strangle", "label": "Short-strangle lean", "tone": "warning", "reason": "Mixed path with contained two-sided analog touches"}
    return {"id": "selective", "label": "Selective / defined-risk", "tone": "warning", "reason": "Premium is rich, but path or touch evidence is not clean"}


def _percentile(series: pd.Series, current: float | None) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if current is None or values.empty:
        return None
    return float((values <= current).mean() * 100)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(parsed) else parsed


def _round(value: Any) -> float | None:
    parsed = _number(value)
    return None if parsed is None else round(parsed, 2)


def _pct(value: Any) -> float | None:
    parsed = _number(value)
    return None if parsed is None else round(parsed * 100, 2)


def _integer(value: Any) -> int | None:
    parsed = _number(value)
    return None if parsed is None else int(parsed)


def _date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()
