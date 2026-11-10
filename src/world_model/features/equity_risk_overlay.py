from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from world_model.features.monthly_windows import fill_missing_months, pct_change_months, trailing_average


def compute_equity_risk_overlay(
    series_by_id: Mapping[str, pd.Series],
    market_history: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the current GTT-style equity risk state from explicit thresholds.

    ``market_history`` is the chained S&P 500 history from
    :func:`world_model.data.benchmarks.load_equity_benchmark_history` and must
    contain a ``normalized_index`` column.
    """
    as_of_date = pd.Timestamp(as_of).normalize()
    history = market_history.sort_index().loc[:as_of_date]
    if history.empty:
        raise ValueError(f"No S&P 500 history is available on or before {as_of_date.date()}")
    data_warnings: list[str] = []
    market_lag_days = int((as_of_date - pd.Timestamp(history.index[-1]).normalize()).days)
    max_market_lag_days = int(config.get("max_market_lag_days", 5))
    if market_lag_days > max_market_lag_days:
        data_warnings.append(
            f"S&P 500 history ends {history.index[-1]:%Y-%m-%d}, {market_lag_days} days before the "
            f"as-of date (limit {max_market_lag_days}); the market trend input is stale."
        )

    monthly = history["normalized_index"].groupby(history.index.to_period("M")).tail(1)
    window = int(config.get("moving_average_months", 10))
    moving_average = monthly.rolling(window, min_periods=window).mean()
    market_value = _last(monthly)
    market_average = _last(moving_average)
    market_gap = _percent_gap(market_value, market_average)
    price_warning = _less_than(market_value, market_average)

    # Macro windows are calendar-aligned so an unpublished month (for example,
    # October 2025) does not stretch a 12-month comparison to 13 months.
    unemployment = _series(series_by_id, "UNRATE", as_of_date)
    unemployment_value = _last(unemployment)
    unemployment_average_value = trailing_average(unemployment, 12)
    unemployment_gap = _difference(unemployment_value, unemployment_average_value)
    # Warn only when unemployment is more than the buffer above its 12-month average. A 0.1 pp buffer
    # passed every GTT robustness test and switched less often than the zero line
    # (docs/gtt_threshold_validation.md).
    unemployment_buffer = float(config.get("unemployment_buffer_pp", 0.1))
    unemployment_warning = _greater_than(unemployment_gap, unemployment_buffer)

    industrial_growth = pct_change_months(_series(series_by_id, "INDPRO", as_of_date), 12)
    industrial_warning = _nonpositive(industrial_growth)

    hours = _series(series_by_id, "AWHMAN", as_of_date)
    hours_value = _last(hours)
    hours_average_value = trailing_average(hours, 12)
    hours_gap = _difference(hours_value, hours_average_value)
    hours_warning = _less_than(hours_value, hours_average_value)

    cpi, _ = fill_missing_months(_series(series_by_id, "CPIAUCSL", as_of_date))
    real_orders = _real_series(_series(series_by_id, "DGORDER", as_of_date), cpi)
    orders_growth = pct_change_months(real_orders, 12)
    orders_warning = _nonpositive(orders_growth)

    core_macro_warning = _any_true(unemployment_warning, industrial_warning)
    supporting_warning = _any_true(hours_warning, orders_warning)
    state_id = _state_id(price_warning, core_macro_warning, orders_warning, supporting_warning)
    state_labels = {
        "normal": "Normal",
        "macro_watch": "Macro watch",
        "market_warning": "Market warning",
        "defensive": "Defensive",
        "high_conviction": "High-conviction defensive",
    }
    messages = {
        "normal": "The equity trend is positive and the tested macro danger signals are clear.",
        "macro_watch": "Macro conditions warrant attention, but the equity trend has not confirmed the warning.",
        "market_warning": "The equity trend is negative, but unemployment and industrial production have not confirmed macro deterioration.",
        "defensive": "The equity trend is negative and at least one core macro danger signal confirms it.",
        "high_conviction": "The equity trend and core macro signal are negative, with real durable orders providing additional confirmation.",
    }

    return {
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "state_id": state_id,
        "state_label": state_labels[state_id],
        "message": messages[state_id],
        "method_note": "Threshold signals, not historical percentiles. Signals use the latest point-in-time macro vintages.",
        "rule": {
            "unemployment_buffer_pp": unemployment_buffer,
            "moving_average_months": window,
        },
        "inputs": [
            _input("market_trend", "S&P 500 trend", market_gap, "% vs 10m avg", price_warning, "below average", history.index[-1]),
            _input("unemployment_trend", "Unemployment trend", unemployment_gap, "pp vs 12m avg", unemployment_warning, f"> {unemployment_buffer:+.1f} pp above average", _date(unemployment)),
            _input("industrial_production", "Industrial production", industrial_growth, "% YoY", industrial_warning, "contracting", _date(_series(series_by_id, "INDPRO", as_of_date))),
            _input("manufacturing_hours", "Manufacturing hours", hours_gap, "hours vs 12m avg", hours_warning, "below average", _date(hours), role="early warning"),
            _input("real_durable_orders", "Real durable orders", orders_growth, "% YoY", orders_warning, "contracting", _date(real_orders), role="confirmation"),
        ],
        "triggers": {
            "market_trend": market_trigger(monthly, window, as_of_date),
            "unemployment_trend": unemployment_trigger(unemployment, unemployment_buffer),
            "industrial_production": industrial_production_trigger(_series(series_by_id, "INDPRO", as_of_date)),
        },
        "source_note": "Norgate $SPX price returns are used before $SPXTR total-return history becomes available; total returns are used thereafter.",
        "data_warnings": data_warnings,
    }


def market_trigger(monthly: pd.Series, window: int, as_of: pd.Timestamp) -> dict[str, Any] | None:
    """Month-end level, relative to the latest close, at which the index would fall below its moving average.

    The month-end close is part of its own average, so with the other ``window - 1`` month-ends summing to S
    the close X breaks the trend when X < (S + X) / window, that is X < S / (window - 1).
    """
    values = monthly.dropna()
    if len(values) < window:
        return None
    latest_period = pd.Timestamp(values.index[-1]).to_period("M")
    in_progress = latest_period == as_of.to_period("M")
    others = values.iloc[-window:-1] if in_progress else values.iloc[-(window - 1):]
    threshold = float(others.mean())
    latest = float(values.iloc[-1])
    month = latest_period if in_progress else latest_period + 1
    return {
        "month": str(month),
        "threshold_change_pct": 100.0 * (threshold / latest - 1.0),
    }


def unemployment_trigger(unemployment: pd.Series, buffer: float) -> dict[str, Any] | None:
    """Unemployment rate the next month must exceed for the gap over its 12-month average to pass ``buffer``.

    With the other n published months of the window summing to S, the next reading X warns when
    X - (S + X) / (n + 1) > buffer, that is X > (buffer * (n + 1) + S) / n.
    """
    values = pd.to_numeric(unemployment, errors="coerce").dropna().sort_index()
    if values.empty:
        return None
    next_month = (pd.Timestamp(values.index[-1]) + pd.DateOffset(months=1)).normalize()
    others = values[values.index > next_month - pd.DateOffset(months=12)]
    if len(others) < 10:
        return None
    count = len(others)
    threshold = (buffer * (count + 1) + float(others.sum())) / count
    return {
        "month": next_month.strftime("%Y-%m"),
        "threshold_rate": threshold,
        "latest_rate": float(values.iloc[-1]),
    }


def industrial_production_trigger(industrial_production: pd.Series) -> dict[str, Any] | None:
    """Monthly change in industrial production that would take its YoY growth to zero next month."""
    values = pd.to_numeric(industrial_production, errors="coerce").dropna().sort_index()
    if values.empty:
        return None
    next_month = (pd.Timestamp(values.index[-1]) + pd.DateOffset(months=1)).normalize()
    base_key = next_month - pd.DateOffset(months=12)
    if base_key not in values.index:
        return None
    return {
        "month": next_month.strftime("%Y-%m"),
        "threshold_change_pct": 100.0 * (float(values.loc[base_key]) / float(values.iloc[-1]) - 1.0),
    }


def _state_id(
    price_warning: bool | None,
    core_macro_warning: bool | None,
    orders_warning: bool | None,
    supporting_warning: bool | None,
) -> str:
    if price_warning is True and core_macro_warning is True and orders_warning is True:
        return "high_conviction"
    if price_warning is True and core_macro_warning is True:
        return "defensive"
    if price_warning is True:
        return "market_warning"
    if core_macro_warning is True or supporting_warning is True:
        return "macro_watch"
    return "normal"


def _input(
    input_id: str,
    label: str,
    value: float | None,
    unit: str,
    warning: bool | None,
    warning_label: str,
    observation_date: pd.Timestamp | None,
    role: str = "primary",
) -> dict[str, Any]:
    status = "unavailable" if warning is None else "warning" if warning else "healthy"
    return {
        "id": input_id,
        "label": label,
        "value": value,
        "unit": unit,
        "status": status,
        "status_label": "unavailable" if warning is None else warning_label if warning else "clear",
        "role": role,
        "observation_date": observation_date.strftime("%Y-%m-%d") if observation_date is not None else None,
    }


def _series(values: Mapping[str, pd.Series], series_id: str, as_of: pd.Timestamp) -> pd.Series:
    series = values.get(series_id)
    if series is None:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="observation_date"))
    return pd.to_numeric(series, errors="coerce").dropna().sort_index().loc[:as_of]


def _real_series(nominal: pd.Series, prices: pd.Series) -> pd.Series:
    common = nominal.index.intersection(prices.index)
    if common.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="observation_date"))
    result = (nominal.reindex(common) / prices.reindex(common)).replace([float("inf"), float("-inf")], pd.NA)
    return result.dropna().astype(float)


def _last(series: pd.Series) -> float | None:
    usable = series.dropna()
    return float(usable.iloc[-1]) if not usable.empty else None


def _date(series: pd.Series) -> pd.Timestamp | None:
    usable = series.dropna()
    return pd.Timestamp(usable.index[-1]) if not usable.empty else None


def _difference(value: float | None, comparison: float | None) -> float | None:
    return None if value is None or comparison is None else value - comparison


def _percent_gap(value: float | None, comparison: float | None) -> float | None:
    if value is None or comparison in (None, 0.0):
        return None
    return 100.0 * (value / comparison - 1.0)


def _less_than(value: float | None, comparison: float | None) -> bool | None:
    return None if value is None or comparison is None else value < comparison


def _greater_than(value: float | None, comparison: float | None) -> bool | None:
    return None if value is None or comparison is None else value > comparison


def _nonpositive(value: float | None) -> bool | None:
    return None if value is None else value <= 0.0


def _any_true(*values: bool | None) -> bool | None:
    available = [value for value in values if value is not None]
    return any(available) if available else None
