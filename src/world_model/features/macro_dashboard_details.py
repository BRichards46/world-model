from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def compute_drilldown_metrics(
    indicator_id: str,
    series_by_id: Mapping[str, pd.Series],
) -> list[dict[str, Any]]:
    """Return familiar, unscored context using values known at the as-of date."""
    builders = {
        "payroll_growth": _payroll_details,
        "unemployment_rate": _unemployment_details,
        "real_consumer_spending": _consumer_details,
        "industrial_production": _industrial_details,
        "real_gdp_growth": _gdp_details,
        "core_pce_inflation": lambda values: _inflation_rate_details(values, "PCEPILFE", "Core PCE inflation"),
        "core_cpi_inflation": lambda values: _inflation_rate_details(values, "CPILFESL", "Core CPI"),
        "median_cpi": lambda values: [_latest_metric(values, "MEDCPIM158SFRBCLE", "Latest published monthly rate", "% annualized MoM", 2)],
        "five_year_inflation_expectations": lambda values: [_latest_metric(values, "EXPINF5YR", "Latest expectation", "%", 2)],
        "real_policy_rate": _policy_details,
        "treasury_curve": _treasury_curve_details,
        "high_yield_spread": _credit_spread_details,
        "adjusted_financial_conditions": lambda values: [_latest_metric(values, "ANFCI", "Latest index reading", "", 2)],
        "building_permits": _housing_details,
        "housing_starts_intensity": _housing_intensity_details,
        "mortgage_rate": lambda values: [_latest_metric(values, "MORTGAGE30US", "Latest mortgage rate", "%", 2)],
    }
    builder = builders.get(indicator_id)
    if builder is None:
        return []
    return [metric for metric in builder(series_by_id) if metric.get("value") is not None]


def compute_curve_move_classification(
    series_by_id: Mapping[str, pd.Series],
    smoothing_window: int = 20,
    comparison_lag: int = 63,
    deadband_bps: float = 5.0,
) -> dict[str, Any]:
    """Classify a 3m10y move from smoothed changes in its two yield legs."""
    legs = pd.concat(
        [
            _series(series_by_id, "DGS10").rename("long_yield"),
            _series(series_by_id, "DGS3MO").rename("short_yield"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    averages = legs.rolling(smoothing_window, min_periods=smoothing_window).mean()
    changes = 100.0 * (averages - averages.shift(comparison_lag))
    usable = changes.dropna()
    if usable.empty:
        return {}
    latest = usable.iloc[-1]
    long_change = float(latest["long_yield"])
    short_change = float(latest["short_yield"])
    move_id, move_label = classify_curve_move(
        long_change,
        short_change,
        deadband_bps=deadband_bps,
    )
    return {
        "curve_move_id": move_id,
        "curve_move_label": move_label,
        "curve_move_observation_date": pd.Timestamp(usable.index[-1]).strftime("%Y-%m-%d"),
        "curve_long_change_bps": long_change,
        "curve_short_change_bps": short_change,
        "curve_slope_change_bps": long_change - short_change,
    }


def classify_curve_move(
    long_change_bps: float,
    short_change_bps: float,
    deadband_bps: float = 5.0,
) -> tuple[str, str]:
    """Return canonical bull/bear/twist/parallel terminology for a curve move."""
    slope_change = long_change_bps - short_change_bps
    long_direction = _direction(long_change_bps, deadband_bps)
    short_direction = _direction(short_change_bps, deadband_bps)
    slope_direction = _direction(slope_change, deadband_bps)

    if slope_direction == 0:
        if long_direction > 0 and short_direction > 0:
            return "parallel_selloff", "Parallel selloff"
        if long_direction < 0 and short_direction < 0:
            return "parallel_rally", "Parallel rally"
        return "stable", "Stable curve"

    if slope_direction > 0:
        if long_direction > 0 and short_direction < 0:
            return "twist_steepener", "Twist steepener"
        if long_direction > 0:
            return "bear_steepener", "Bear steepener"
        return "bull_steepener", "Bull steepener"

    if long_direction < 0 and short_direction > 0:
        return "twist_flattener", "Twist flattener"
    if long_direction < 0:
        return "bull_flattener", "Bull flattener"
    return "bear_flattener", "Bear flattener"


def _direction(value: float, deadband: float) -> int:
    return 1 if value > deadband else -1 if value < -deadband else 0


def _treasury_curve_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    metrics = [
        _latest_metric(values, "T10Y3M", "Latest curve spread", "%", 2),
        _latest_metric(values, "DGS10", "Latest 10-year yield", "%", 2),
        _latest_metric(values, "DGS3MO", "Latest 3-month yield", "%", 2),
    ]
    classification = compute_curve_move_classification(values)
    observation_date = classification.get("curve_move_observation_date")
    for key, label in [
        ("curve_long_change_bps", "10-year yield change"),
        ("curve_short_change_bps", "3-month yield change"),
        ("curve_slope_change_bps", "3m10y slope change"),
    ]:
        if key not in classification:
            continue
        metrics.append(
            {
                "label": label,
                "value": classification[key],
                "unit": "bp · 20d average vs 63 business days earlier",
                "decimals": 1,
                "source": "DGS10 / DGS3MO",
                "observation_date": observation_date,
            }
        )
    return metrics


def point_in_time_series(vintage: pd.DataFrame) -> pd.Series:
    """Collapse an already as-of-filtered vintage table to its latest known values."""
    required = {"observation_date", "realtime_start", "value"}
    missing = required - set(vintage.columns)
    if missing:
        raise ValueError(f"Vintage table is missing columns: {', '.join(sorted(missing))}")
    frame = vintage.copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.normalize()
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce").dt.normalize()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["observation_date", "realtime_start"])
    frame = frame.sort_values(["realtime_start", "observation_date"])
    frame = frame.drop_duplicates("observation_date", keep="last")
    result = frame.set_index("observation_date")["value"].astype(float).sort_index()
    result.index = pd.DatetimeIndex(result.index, name="observation_date")
    return result


def _payroll_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    changes = _series(values, "PAYEMS").diff()
    return [
        _from_series(changes, "Latest monthly payroll change", "thousand jobs", 0, "PAYEMS"),
        _from_series(changes.rolling(3, min_periods=3).mean(), "Three-month average change", "thousand jobs", 0, "PAYEMS"),
    ]


def _unemployment_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    series = _series(values, "UNRATE")
    return [
        _from_series(series, "Latest unemployment rate", "%", 1, "UNRATE"),
        _from_series(series.rolling(3, min_periods=3).mean(), "Three-month average rate", "%", 2, "UNRATE"),
    ]


def _consumer_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    real_pce = _series(values, "PCEC96")
    return [
        _from_series(_pct_change(real_pce, 1), "Latest real spending change", "% MoM", 2, "PCEC96"),
        _from_series(_pct_change(real_pce, 12), "Real spending growth", "% YoY", 2, "PCEC96"),
        _from_series(_pct_change(_series(values, "RPI"), 12), "Real personal income growth", "% YoY", 2, "RPI"),
        _from_series(_pct_change(_series(values, "RSAFS"), 12), "Nominal retail sales growth", "% YoY", 2, "RSAFS"),
        _from_series(_pct_change(_series(values, "PCE"), 12), "Nominal PCE growth", "% YoY", 2, "PCE"),
        _latest_metric(values, "PCENOW", "PCE nowcast", "% QoQ annualized estimate", 2),
    ]


def _industrial_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    series = _series(values, "INDPRO")
    hours = _series(values, "AWHMAN")
    hours_average = hours.rolling(12, min_periods=12).mean()
    real_orders = _real_series(_series(values, "DGORDER"), _series(values, "CPIAUCSL"))
    metrics = [
        _from_series(_pct_change(series, 1), "Latest production change", "% MoM", 2, "INDPRO"),
        _signal_metric(
            _pct_change(series, 12),
            "Industrial production growth",
            "% YoY",
            2,
            "INDPRO",
            lambda value: value <= 0.0,
            "Contracting at or below 0% YoY",
        ),
        _from_series(hours, "Manufacturing workweek", "hours", 1, "AWHMAN"),
        _signal_metric(
            hours - hours_average,
            "Workweek gap vs 12m average",
            "hours",
            2,
            "AWHMAN",
            lambda value: value < 0.0,
            "Early warning when below average",
        ),
        _signal_metric(
            _pct_change(real_orders, 12),
            "Real durable-goods orders",
            "% YoY",
            2,
            "DGORDER / CPIAUCSL",
            lambda value: value <= 0.0,
            "Confirmation when contracting",
        ),
    ]
    for series_id, label in [
        ("IPMAN", "Manufacturing production"),
        ("IPFINAL", "Final-products production"),
        ("IPBUSEQ", "Business-equipment production"),
        ("IPMAT", "Materials production"),
    ]:
        metric = _from_series(_pct_change(_series(values, series_id), 12), label, "% YoY", 2, series_id)
        metric["signal_status"] = "diagnostic"
        metric["signal_note"] = "Diagnostic decomposition; not part of the risk state"
        metrics.append(metric)
    return metrics


def _gdp_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    real_gdp = _series(values, "GDPC1")
    nominal_gdp = _series(values, "GDP")
    return [
        _from_series(_pct_change(real_gdp, 4), "Real GDP growth", "% YoY", 2, "GDPC1"),
        _from_series(_annualized_change(nominal_gdp, 1, 4.0), "Nominal GDP growth", "% QoQ annualized", 2, "GDP"),
        _latest_metric(values, "GDPNOW", "GDPNow estimate", "% QoQ annualized", 2),
    ]


def _inflation_rate_details(
    values: Mapping[str, pd.Series],
    series_id: str,
    label: str,
) -> list[dict[str, Any]]:
    series = _monthly_series(values, series_id)
    definitions = [
        (_pct_change(series, 1), label, "% MoM", series_id),
        (_pct_change(series, 12), label, "% YoY", series_id),
        (_annualized_change(series, 3, 4.0), label, "% 3m annualized", series_id),
    ]
    return [_from_series(series, label, unit, 2, source) for series, label, unit, source in definitions]


def _monthly_series(values: Mapping[str, pd.Series], series_id: str) -> pd.Series:
    """Keep calendar gaps so MoM and YoY shifts cannot bridge missing months."""
    series = _series(values, series_id)
    if series.empty:
        return series
    periods = series.index.to_period("M")
    monthly = pd.Series(series.to_numpy(dtype=float), index=periods, dtype=float)
    full_index = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_index)
    monthly.index = monthly.index.to_timestamp(how="start")
    return monthly


def _policy_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    daily = _series(values, "DFF").dropna()
    monthly = daily.groupby(daily.index.to_period("M")).mean()
    monthly.index = monthly.index.to_timestamp(how="start")
    policy_average = monthly.rolling(3, min_periods=3).mean()
    inflation = _pct_change(_series(values, "PCEPILFE"), 12)
    inflation_date = inflation.dropna().index[-1] if not inflation.dropna().empty else None
    aligned_policy = policy_average.loc[:inflation_date] if inflation_date is not None else policy_average.iloc[0:0]
    return [
        _from_series(aligned_policy, "Effective fed funds rate", "% · 3m average", 2, "DFF"),
        _from_series(inflation, "Core PCE inflation", "% YoY", 2, "PCEPILFE"),
    ]


def _credit_spread_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    investment_grade = _series(values, "BAMLC0A0CM")
    high_yield = _series(values, "BAMLH0A0HYM2")
    ccc = _series(values, "BAMLH0A3HYC")
    metrics = [
        _latest_metric(values, "BAMLC0A0CM", "Investment-grade OAS", "%", 2),
        _latest_metric(values, "BAMLC0A4CBBB", "BBB OAS", "%", 2),
        _latest_metric(values, "BAMLH0A0HYM2", "High-yield OAS", "%", 2),
        _latest_metric(values, "BAMLH0A3HYC", "CCC and lower OAS", "%", 2),
        _latest_metric(values, "BAA10Y", "Long-history Baa–Treasury spread", "%", 2),
    ]
    hy_ig = pd.concat([high_yield.rename("hy"), investment_grade.rename("ig")], axis=1).dropna()
    ccc_hy = pd.concat([ccc.rename("ccc"), high_yield.rename("hy")], axis=1).dropna()
    metrics.extend(
        [
            _from_series(hy_ig["hy"] - hy_ig["ig"], "HY minus IG quality gap", "percentage points", 2, "BAMLH0A0HYM2 / BAMLC0A0CM"),
            _from_series(ccc_hy["ccc"] - ccc_hy["hy"], "CCC minus broad HY gap", "percentage points", 2, "BAMLH0A3HYC / BAMLH0A0HYM2"),
        ]
    )
    return metrics


def _housing_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    permits = _series(values, "PERMIT")
    return [
        _from_series(permits, "Latest building permits", "thousand units SAAR", 0, "PERMIT"),
    ]


def _housing_intensity_details(values: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    starts = _series(values, "HOUST")
    labor_force = _series(values, "CLF16OV")
    common_dates = starts.index.intersection(labor_force.index)
    if common_dates.empty:
        return []
    common_date = common_dates[-1]
    return [
        _from_series((starts.loc[[common_date]] / 1000.0), "Latest housing starts", "million units SAAR", 3, "HOUST"),
        _from_series((labor_force.loc[[common_date]] / 1000.0), "Civilian labor force", "million participants", 2, "CLF16OV"),
    ]


def _series(values: Mapping[str, pd.Series], series_id: str) -> pd.Series:
    if series_id not in values:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="observation_date"))
    result = values[series_id].astype(float).sort_index()
    return result.dropna()


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    prior = series.shift(periods)
    valid = series.where((series > 0.0) & (prior > 0.0))
    comparison = prior.where((series > 0.0) & (prior > 0.0))
    return 100.0 * (np.exp(np.log(valid) - np.log(comparison)) - 1.0)


def _annualized_change(series: pd.Series, periods: int, periods_per_year: float) -> pd.Series:
    prior = series.shift(periods)
    valid = series.where((series > 0.0) & (prior > 0.0))
    comparison = prior.where((series > 0.0) & (prior > 0.0))
    return 100.0 * (np.exp(periods_per_year * (np.log(valid) - np.log(comparison))) - 1.0)


def _real_series(nominal: pd.Series, prices: pd.Series) -> pd.Series:
    common = nominal.index.intersection(prices.index)
    if common.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="observation_date"))
    result = nominal.reindex(common) / prices.reindex(common)
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def _signal_metric(
    series: pd.Series,
    label: str,
    unit: str,
    decimals: int,
    source: str,
    is_warning: Any,
    note: str,
) -> dict[str, Any]:
    metric = _from_series(series, label, unit, decimals, source)
    value = metric.get("value")
    metric["signal_status"] = "unavailable" if value is None else "warning" if is_warning(value) else "healthy"
    metric["signal_note"] = note
    return metric


def _latest_metric(
    values: Mapping[str, pd.Series],
    series_id: str,
    label: str,
    unit: str,
    decimals: int,
) -> dict[str, Any]:
    return _from_series(_series(values, series_id), label, unit, decimals, series_id)


def _from_series(
    series: pd.Series,
    label: str,
    unit: str,
    decimals: int,
    source: str,
) -> dict[str, Any]:
    usable = series.dropna()
    if usable.empty:
        return {"label": label, "value": None, "unit": unit, "decimals": decimals, "source": source}
    timestamp = pd.Timestamp(usable.index[-1])
    return {
        "label": label,
        "value": float(usable.iloc[-1]),
        "unit": unit,
        "decimals": decimals,
        "source": source,
        "observation_date": timestamp.strftime("%Y-%m-%d"),
    }
