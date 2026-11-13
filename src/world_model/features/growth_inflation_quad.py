"""Growth/inflation quad classification with base-effect look-ahead.

Conventions
-----------
* Growth is real GDP year-over-year growth for each calendar quarter.
* Inflation is the quarterly average of monthly headline CPI year-over-year
  inflation.
* A quarter's quad is set by the direction of change of both rates versus the
  prior quarter:

  ========  ==================  =====================
  Quad      Growth YoY          Inflation YoY
  ========  ==================  =====================
  1         accelerating        decelerating
  2         accelerating        accelerating
  3         decelerating        accelerating
  4         decelerating        decelerating
  ========  ==================  =====================

Base effects
------------
For GDP, year-over-year growth accelerates in quarter ``t`` exactly when the
quarter's own annualized growth exceeds the annualized growth of quarter
``t - 4`` (the comparison quarter dropping out of the window). That comparison
rate is the quarter's *growth hurdle*. For CPI, the hurdle is the constant
monthly inflation rate over the quarter's unreported months that would leave
the quarterly-average YoY rate unchanged from the prior quarter.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from world_model.features.monthly_windows import fill_missing_months


QUAD_DESCRIPTIONS = {
    1: "Growth accelerating, inflation decelerating",
    2: "Growth accelerating, inflation accelerating",
    3: "Growth decelerating, inflation accelerating",
    4: "Growth decelerating, inflation decelerating",
}

SCENARIO_NAMES = ("low", "base", "high")


def classify_quad(growth_change: float | None, inflation_change: float | None) -> int | None:
    """Return the quad for a pair of quarter-over-quarter changes in YoY rates."""
    if growth_change is None or inflation_change is None:
        return None
    if not np.isfinite(growth_change) or not np.isfinite(inflation_change):
        return None
    growth_up = growth_change > 0.0
    inflation_up = inflation_change > 0.0
    if growth_up and inflation_up:
        return 2
    if growth_up:
        return 1
    if inflation_up:
        return 3
    return 4


def quarter_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("Q").start_time.normalize()


def quarter_label(value: pd.Timestamp) -> str:
    period = pd.Timestamp(value).to_period("Q")
    return f"{period.year}Q{period.quarter}"


def quarterly_levels(series: pd.Series) -> pd.Series:
    """Normalize a quarterly level series to quarter-start dates."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0.0]
    if values.empty:
        return values.astype(float)
    index = pd.DatetimeIndex(values.index).to_period("Q").start_time.normalize()
    result = pd.Series(values.to_numpy(dtype=float), index=index).sort_index()
    return result[~result.index.duplicated(keep="last")]


def monthly_levels(series: pd.Series) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Normalize a monthly level series and log-interpolate interior gaps.

    Returns the filled series and the list of interpolated months, which the
    caller should surface (for example, the unpublished October 2025 CPI).
    """
    return fill_missing_months(series)


def extend_quarterly(levels: pd.Series, through: pd.Timestamp, saar_by_quarter: Mapping[pd.Timestamp, float]) -> pd.Series:
    """Extend quarterly levels through ``through`` using annualized growth per quarter."""
    result = levels.copy()
    target = quarter_start(through)
    current = result.index.max()
    while current < target:
        next_quarter = (current + pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        rate = float(saar_by_quarter[next_quarter])
        result.loc[next_quarter] = result.loc[current] * (1.0 + rate / 100.0) ** 0.25
        current = next_quarter
    return result.sort_index()


def extend_monthly(levels: pd.Series, through: pd.Timestamp, monthly_pct: float, overrides: Mapping[pd.Timestamp, float] | None = None) -> pd.Series:
    """Extend monthly levels through ``through`` at a constant month-over-month rate."""
    result = levels.copy()
    target = pd.Timestamp(through).to_period("M").start_time.normalize()
    current = result.index.max()
    overrides = overrides or {}
    while current < target:
        next_month = (current + pd.offsets.MonthBegin(1)).normalize()
        rate = float(overrides.get(next_month, monthly_pct))
        result.loc[next_month] = result.loc[current] * (1.0 + rate / 100.0)
        current = next_month
    return result.sort_index()


def gdp_yoy(levels: pd.Series) -> pd.Series:
    return 100.0 * (levels / levels.shift(4) - 1.0)


def gdp_saar(levels: pd.Series) -> pd.Series:
    return 100.0 * ((levels / levels.shift(1)) ** 4 - 1.0)


def cpi_monthly_yoy(levels: pd.Series) -> pd.Series:
    return 100.0 * (levels / levels.shift(12) - 1.0)


def cpi_quarterly_yoy(levels: pd.Series) -> pd.Series:
    """Average of monthly YoY rates in each quarter; quarters need all three months."""
    yoy = cpi_monthly_yoy(levels).dropna()
    if yoy.empty:
        return yoy
    quarters = yoy.index.to_period("Q").start_time.normalize()
    grouped = yoy.groupby(quarters)
    averages = grouped.mean()[grouped.count() == 3]
    averages.index = pd.DatetimeIndex(averages.index)
    return averages


@dataclass(frozen=True)
class QuadInputs:
    """Point-in-time inputs known on ``as_of``."""

    as_of: pd.Timestamp
    gdp: pd.Series
    cpi: pd.Series
    core_cpi: pd.Series
    nowcast_saar: pd.Series
    interpolated_cpi_months: tuple[pd.Timestamp, ...] = ()


@dataclass(frozen=True)
class QuadSettings:
    forward_quarters: int = 3
    growth_trend_quarters: int = 4
    growth_offset_saar: float = 1.0
    inflation_trend_months: int = 6
    inflation_offset_mom: float = 0.1
    history_quarters: int = 12
    use_nowcast: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "QuadSettings":
        values = dict(values or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: values[key] for key in fields if key in values})


def growth_assumptions(inputs: QuadInputs, settings: QuadSettings, through: pd.Timestamp) -> dict[str, object]:
    """Base annualized growth for each unreported quarter through ``through``."""
    known = inputs.gdp
    saar = gdp_saar(known).dropna()
    trend = float(saar.tail(settings.growth_trend_quarters).mean()) if not saar.empty else 2.0
    last_known = known.index.max()
    quarters = pd.date_range(
        (last_known + pd.offsets.QuarterBegin(startingMonth=1)).normalize(), quarter_start(through), freq="QS"
    )
    base: dict[pd.Timestamp, float] = {}
    sources: dict[pd.Timestamp, str] = {}
    for quarter in quarters:
        nowcast = inputs.nowcast_saar.get(quarter) if settings.use_nowcast else None
        if nowcast is not None and np.isfinite(nowcast):
            base[quarter] = float(nowcast)
            sources[quarter] = "nowcast"
        else:
            base[quarter] = trend
            sources[quarter] = "trend"
    return {"trend_saar": trend, "base": base, "sources": sources, "unreported_quarters": list(quarters)}


def inflation_assumptions(inputs: QuadInputs, settings: QuadSettings) -> dict[str, float]:
    """Base monthly headline CPI assumption: the recent core CPI run-rate (energy flat)."""
    core = inputs.core_cpi
    mom = (100.0 * (core / core.shift(1) - 1.0)).dropna()
    base = float(mom.tail(settings.inflation_trend_months).mean()) if not mom.empty else 0.25
    return {"base_mom": base}


def _project(
    inputs: QuadInputs,
    through: pd.Timestamp,
    growth_base: Mapping[pd.Timestamp, float],
    growth_offset: float,
    inflation_mom: float,
    inflation_overrides: Mapping[pd.Timestamp, float] | None = None,
    inflation_offset: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    gdp_path = extend_quarterly(
        inputs.gdp, through, {quarter: rate + growth_offset for quarter, rate in growth_base.items()}
    )
    month_end = quarter_start(through) + pd.offsets.MonthBegin(2)
    overrides = {month: rate + inflation_offset for month, rate in (inflation_overrides or {}).items()}
    cpi_path = extend_monthly(inputs.cpi, month_end, inflation_mom + inflation_offset, overrides)
    return gdp_yoy(gdp_path), cpi_quarterly_yoy(cpi_path)


def inflation_hurdle(
    inputs: QuadInputs,
    quarter: pd.Timestamp,
    base_mom: float,
    path_overrides: Mapping[pd.Timestamp, float] | None = None,
) -> float | None:
    """Constant monthly CPI change over the quarter's unreported months that holds its average YoY flat.

    Earlier unreported months follow ``path_overrides`` where given, otherwise ``base_mom``.
    """
    quarter = quarter_start(quarter)
    months = pd.date_range(quarter, periods=3, freq="MS")
    known_last = inputs.cpi.index.max()
    unknown = [month for month in months if month > known_last]
    if not unknown:
        return None
    previous = (quarter - pd.offsets.QuarterBegin(startingMonth=1)).normalize()

    def change(rate: float) -> float:
        overrides = {**(path_overrides or {}), **{month: rate for month in unknown}}
        path = extend_monthly(inputs.cpi, months[-1], base_mom, overrides)
        averages = cpi_quarterly_yoy(path)
        return float(averages.get(quarter, np.nan) - averages.get(previous, np.nan))

    low, high = -3.0, 5.0
    f_low, f_high = change(low), change(high)
    if not (np.isfinite(f_low) and np.isfinite(f_high)) or f_low > 0 or f_high < 0:
        return None
    for _ in range(60):
        middle = 0.5 * (low + high)
        if change(middle) > 0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def build_quad_outlook(
    inputs: QuadInputs,
    settings: QuadSettings | None = None,
    include_hurdles: bool = True,
    growth_override: Mapping[pd.Timestamp, float] | None = None,
    inflation_override: Mapping[pd.Timestamp, float] | None = None,
    override_source: str | Mapping[pd.Timestamp, str] = "model",
) -> dict[str, object]:
    """Realized quads plus scenario projections for every unreported quarter through the horizon.

    ``growth_override`` replaces the base annualized growth for specific unreported
    quarters; ``inflation_override`` replaces the base monthly CPI change for
    specific unreported months. Scenario offsets are applied on top of either.
    """
    settings = settings or QuadSettings()
    current_quarter = quarter_start(inputs.as_of)
    horizon_end = (current_quarter + pd.DateOffset(months=3 * settings.forward_quarters)).normalize()
    growth = growth_assumptions(inputs, settings, horizon_end)
    for quarter, rate in (growth_override or {}).items():
        if quarter in growth["base"] and rate is not None and np.isfinite(rate):
            growth["base"][quarter] = float(rate)
            growth["sources"][quarter] = (
                override_source.get(quarter, "model") if isinstance(override_source, Mapping) else override_source
            )
    inflation = inflation_assumptions(inputs, settings)
    inflation_months = {
        month: float(rate)
        for month, rate in (inflation_override or {}).items()
        if month > inputs.cpi.index.max() and rate is not None and np.isfinite(rate)
    }
    growth_offsets = {"low": -settings.growth_offset_saar, "base": 0.0, "high": settings.growth_offset_saar}
    inflation_rates = {
        "low": inflation["base_mom"] - settings.inflation_offset_mom,
        "base": inflation["base_mom"],
        "high": inflation["base_mom"] + settings.inflation_offset_mom,
    }

    paths: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
    for growth_name, inflation_name in product(SCENARIO_NAMES, SCENARIO_NAMES):
        paths[(growth_name, inflation_name)] = _project(
            inputs,
            horizon_end,
            growth["base"],
            growth_offsets[growth_name],
            inflation["base_mom"],
            inflation_months,
            inflation_rates[inflation_name] - inflation["base_mom"],
        )

    realized_growth = gdp_yoy(inputs.gdp).dropna()
    realized_inflation = cpi_quarterly_yoy(inputs.cpi)
    last_gdp_quarter = inputs.gdp.index.max()
    last_cpi_month = inputs.cpi.index.max()
    gdp_known_saar = gdp_saar(inputs.gdp)

    quarters: list[dict[str, object]] = []
    first_quarter = min(
        realized_growth.index.max() if not realized_growth.empty else current_quarter,
        realized_inflation.index.max() if not realized_inflation.empty else current_quarter,
    )
    history_start = (first_quarter - pd.DateOffset(months=3 * (settings.history_quarters - 1))).normalize()
    for quarter in pd.date_range(history_start, horizon_end, freq="QS"):
        previous = (quarter - pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        growth_reported = quarter <= last_gdp_quarter
        cpi_months_reported = int(sum(month <= last_cpi_month for month in pd.date_range(quarter, periods=3, freq="MS")))
        realized = growth_reported and cpi_months_reported == 3
        base_growth, base_inflation = paths[("base", "base")]
        record: dict[str, object] = {
            "quarter": quarter_label(quarter),
            "quarter_start": quarter.strftime("%Y-%m-%d"),
            "status": "realized" if realized else ("partial" if growth_reported or cpi_months_reported else "projected"),
            "is_current_quarter": bool(quarter == current_quarter),
            "growth_reported": bool(growth_reported),
            "inflation_months_reported": cpi_months_reported,
            "growth_yoy": _value(base_growth, quarter),
            "inflation_yoy": _value(base_inflation, quarter),
            "growth_change": _difference(base_growth, quarter, previous),
            "inflation_change": _difference(base_inflation, quarter, previous),
        }
        record["quad"] = classify_quad(record["growth_change"], record["inflation_change"])
        if not realized:
            counts = {quad: 0 for quad in (1, 2, 3, 4)}
            growth_range = []
            inflation_range = []
            for (growth_name, inflation_name), (growth_path, inflation_path) in paths.items():
                quad = classify_quad(
                    _difference(growth_path, quarter, previous), _difference(inflation_path, quarter, previous)
                )
                if quad is not None:
                    counts[quad] += 1
                growth_range.append(_value(growth_path, quarter))
                inflation_range.append(_value(inflation_path, quarter))
            total = sum(counts.values())
            record["scenario_counts"] = {str(quad): count for quad, count in counts.items()}
            record["scenario_total"] = total
            record["agreement"] = (counts.get(record["quad"], 0) / total) if total and record["quad"] else None
            record["growth_yoy_range"] = _range(growth_range)
            record["inflation_yoy_range"] = _range(inflation_range)
            comp_quarter = (quarter - pd.DateOffset(years=1)).normalize()
            record["growth_hurdle_saar"] = _value(gdp_known_saar, comp_quarter) if not growth_reported else None
            record["growth_assumption_saar"] = growth["base"].get(quarter) if not growth_reported else None
            record["growth_assumption_source"] = growth["sources"].get(quarter) if not growth_reported else None
            unknown_months = [
                month for month in pd.date_range(quarter, periods=3, freq="MS") if month > last_cpi_month
            ]
            record["inflation_assumption_mom"] = (
                float(np.mean([inflation_months.get(month, inflation["base_mom"]) for month in unknown_months]))
                if unknown_months
                else None
            )
            hurdle = (
                inflation_hurdle(inputs, quarter, inflation_rates["base"], inflation_months) if include_hurdles else None
            )
            record["inflation_hurdle_mom"] = hurdle
            record["inflation_hurdle_annualized"] = _annualize(hurdle)
        quarters.append(record)

    comps = _cpi_comps(inputs.cpi, last_cpi_month, months=12, interpolated=inputs.interpolated_cpi_months)
    return {
        "as_of_date": inputs.as_of.strftime("%Y-%m-%d"),
        "current_quarter": quarter_label(current_quarter),
        "last_gdp_quarter": quarter_label(last_gdp_quarter),
        "last_cpi_month": last_cpi_month.strftime("%Y-%m"),
        "assumptions": {
            "growth_trend_saar": growth["trend_saar"],
            "growth_trend_quarters": settings.growth_trend_quarters,
            "growth_offset_saar": settings.growth_offset_saar,
            "inflation_base_mom": inflation["base_mom"],
            "inflation_base_annualized": _annualize(inflation["base_mom"]),
            "inflation_trend_months": settings.inflation_trend_months,
            "inflation_offset_mom": settings.inflation_offset_mom,
            "nowcast_quarters": {
                quarter_label(quarter): rate
                for quarter, rate in growth["base"].items()
                if growth["sources"].get(quarter) == "nowcast"
            },
        },
        "quarters": quarters,
        "cpi_comps": comps,
        "interpolated_cpi_months": [month.strftime("%Y-%m") for month in inputs.interpolated_cpi_months],
    }


def _cpi_comps(
    cpi: pd.Series,
    last_month: pd.Timestamp,
    months: int,
    interpolated: Iterable[pd.Timestamp] = (),
) -> list[dict[str, object]]:
    """Month-over-month CPI changes that roll out of the YoY window over the coming months."""
    mom = 100.0 * (cpi / cpi.shift(1) - 1.0)
    interpolated_set = set(interpolated)
    rows = []
    for step in range(1, months + 1):
        target = (last_month + pd.DateOffset(months=step)).normalize()
        comp = (target - pd.DateOffset(years=1)).normalize()
        value = _value(mom, comp)
        prior = (comp - pd.DateOffset(months=1)).normalize()
        rows.append(
            {
                "month": target.strftime("%Y-%m"),
                "comp_month": comp.strftime("%Y-%m"),
                "comp_mom": value,
                "comp_annualized": _annualize(value),
                "comp_interpolated": bool(comp in interpolated_set or prior in interpolated_set),
            }
        )
    return rows


def _value(series: pd.Series, key: pd.Timestamp) -> float | None:
    if key not in series.index:
        return None
    value = series.loc[key]
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _difference(series: pd.Series, key: pd.Timestamp, previous: pd.Timestamp) -> float | None:
    current = _value(series, key)
    prior = _value(series, previous)
    return None if current is None or prior is None else current - prior


def _range(values: Iterable[float | None]) -> list[float] | None:
    usable = [value for value in values if value is not None]
    return [min(usable), max(usable)] if usable else None


def _annualize(monthly_pct: float | None) -> float | None:
    if monthly_pct is None:
        return None
    return 100.0 * ((1.0 + monthly_pct / 100.0) ** 12 - 1.0)
