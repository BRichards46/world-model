"""Forecasting models for unreported quarters in the growth/inflation quad.

* Growth, current quarter: GDPNow when available, otherwise a bridge regression
  from monthly real PCE, industrial production and payrolls (each extended at
  its recent pace through the quarter).
* Growth, later quarters: ridge regression of k-quarter-ahead annualized growth
  on the current nowcast, trailing trend, initial claims YoY, the 10y-3m curve
  and building permits YoY.
* Inflation: core CPI run-rate plus gasoline pass-through for months with
  gasoline information (month-to-date averages and next-month carry-over at a
  flat price); later months use the core run-rate.

Data timing: GDP, CPI and the monthly activity series use true FRED vintages.
Gasoline, the 10y-3m curve and initial claims use latest values with a
publication lag; they are market prices or weekly series with negligible
revisions, and their vintage history starts only in 2009-2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table
from world_model.features.growth_inflation_quad import QuadInputs, QuadSettings, gdp_saar, quarter_start
from world_model.features.macro_dashboard_details import point_in_time_series
from world_model.features.monthly_windows import fill_missing_months


BRIDGE_SERIES = ("PCEC96", "INDPRO", "PAYEMS")
LEADING_MONTHLY = ("PERMIT",)
LAGGED_LATEST = {"GASREGW": 1, "T10Y3M": 1, "ICSA": 5}

GROWTH_VARIANTS = ("baseline", "bridge", "bridge_nowcast", "leading")
INFLATION_VARIANTS = ("baseline", "energy")


def _latest_store(fred_root: str, series_id: str) -> pd.Series:
    path = f"{fred_root}/latest/{series_id}.parquet"
    frame = pd.read_parquet(path)
    series = frame.set_index(pd.to_datetime(frame["observation_date"]))["latest_value"].astype(float)
    return series.dropna().sort_index()


@dataclass
class ForecastContext:
    """Extra forecasting inputs: vintage monthly activity data and lagged market/weekly series."""

    monthly_tables: dict[str, pd.DataFrame]
    lagged: dict[str, pd.Series]

    @classmethod
    def load(cls, fred_root: str) -> "ForecastContext":
        monthly = {series: load_vintage_table(fred_root, series) for series in (*BRIDGE_SERIES, *LEADING_MONTHLY)}
        lagged = {series: _latest_store(fred_root, series) for series in LAGGED_LATEST}
        return cls(monthly_tables=monthly, lagged=lagged)

    def monthly_as_of(self, series_id: str, as_of: pd.Timestamp) -> pd.Series:
        table = self.monthly_tables[series_id]
        values, _ = fill_missing_months(point_in_time_series(table[table["realtime_start"] <= as_of]))
        return values

    def lagged_as_of(self, series_id: str, as_of: pd.Timestamp) -> pd.Series:
        series = self.lagged[series_id]
        return series[series.index + pd.Timedelta(days=LAGGED_LATEST[series_id]) <= as_of]


# --------------------------------------------------------------------------- growth


def _extend_monthly_at_trend(series: pd.Series, through: pd.Timestamp, months: int = 3) -> pd.Series:
    """Extend a monthly level series through ``through`` at its trailing average log change."""
    values = series.dropna()
    if values.empty:
        return values
    step = float(np.log(values).diff().tail(months).mean()) if len(values) > months else 0.0
    result = values.copy()
    current = result.index.max()
    target = pd.Timestamp(through).to_period("M").start_time
    while current < target:
        nxt = (current + pd.offsets.MonthBegin(1)).normalize()
        result.loc[nxt] = result.loc[current] * np.exp(step)
        current = nxt
    return result


def _quarterly_growth(monthly: pd.Series) -> pd.Series:
    """Annualized log growth of quarterly-average levels for complete quarters."""
    quarters = monthly.index.to_period("Q").start_time
    grouped = monthly.groupby(quarters)
    average = grouped.mean()[grouped.count() == 3]
    average.index = pd.DatetimeIndex(average.index)
    full = average.reindex(pd.date_range(average.index.min(), average.index.max(), freq="QS"))
    return 400.0 * np.log(full / full.shift(1))


def bridge_nowcast(context: ForecastContext, inputs: QuadInputs, quarters: list[pd.Timestamp]) -> dict[pd.Timestamp, float]:
    """Nowcast annualized GDP growth for unreported quarters from monthly activity data."""
    if not quarters:
        return {}
    last_quarter_end = quarters[-1] + pd.offsets.MonthBegin(2)
    features: dict[str, pd.Series] = {}
    for series_id in BRIDGE_SERIES:
        monthly = context.monthly_as_of(series_id, inputs.as_of)
        if monthly.empty:
            return {}
        features[series_id] = _quarterly_growth(_extend_monthly_at_trend(monthly, last_quarter_end))
    frame = pd.DataFrame(features)
    target = gdp_saar(inputs.gdp).rename("gdp")
    train = pd.concat([frame, target], axis=1).dropna()
    train = train[train.index >= pd.Timestamp("1985-01-01")]
    if len(train) < 40:
        return {}
    coefficients = _ols(train[list(BRIDGE_SERIES)].to_numpy(), train["gdp"].to_numpy())
    result = {}
    for quarter in quarters:
        if quarter in frame.index and frame.loc[quarter].notna().all():
            result[quarter] = float(coefficients[0] + frame.loc[quarter].to_numpy() @ coefficients[1:])
    return result


def _ols(x: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def leading_features(context: ForecastContext, as_of: pd.Timestamp) -> pd.DataFrame:
    """Quarterly leading-indicator features measured over each quarter, as known on ``as_of``."""
    claims = context.lagged_as_of("ICSA", as_of)
    curve = context.lagged_as_of("T10Y3M", as_of)
    permits = context.monthly_as_of("PERMIT", as_of)
    claims_q = claims.groupby(claims.index.to_period("Q").start_time).mean()
    curve_q = curve.groupby(curve.index.to_period("Q").start_time).mean()
    permits_q = permits.groupby(permits.index.to_period("Q").start_time).mean()
    frame = pd.DataFrame(
        {
            "claims_yoy": 100.0 * np.log(claims_q / claims_q.shift(4)),
            "curve": curve_q,
            "permits_yoy": 100.0 * np.log(permits_q / permits_q.shift(4)),
        }
    )
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def leading_forecast(
    context: ForecastContext,
    inputs: QuadInputs,
    origin_quarter: pd.Timestamp,
    origin_saar: float,
    targets: list[pd.Timestamp],
) -> dict[pd.Timestamp, float]:
    """Ridge forecasts of annualized growth k quarters after the origin quarter."""
    saar = gdp_saar(inputs.gdp)
    trend = saar.rolling(4).mean()
    features = leading_features(context, inputs.as_of)
    # Features for the origin quarter use what is known so far in that quarter.
    base = pd.concat([saar.rename("saar"), trend.rename("trend"), features], axis=1)
    result = {}
    for target in targets:
        steps = int(round((target - origin_quarter).days / 91.3))
        if steps < 1:
            continue
        frame = base.copy()
        frame["target"] = saar.shift(-steps)
        train = frame.dropna()
        train = train[train.index >= pd.Timestamp("1985-01-01")]
        if len(train) < 40:
            continue
        columns = ["saar", "trend", "claims_yoy", "curve", "permits_yoy"]
        mean = train[columns].mean()
        scale = train[columns].std().replace(0.0, 1.0)
        coefficients = _ols(((train[columns] - mean) / scale).to_numpy(), train["target"].to_numpy(), ridge=5.0)
        origin = {
            "saar": origin_saar,
            "trend": float(saar.dropna().tail(4).mean()),
            **{name: _latest_value(features[name], origin_quarter) for name in ("claims_yoy", "curve", "permits_yoy")},
        }
        if any(value is None or not np.isfinite(value) for value in origin.values()):
            continue
        x = ((pd.Series(origin)[columns] - mean) / scale).to_numpy(dtype=float)
        result[target] = float(coefficients[0] + x @ coefficients[1:])
    return result


def _latest_value(series: pd.Series, quarter: pd.Timestamp) -> float | None:
    usable = series.loc[:quarter].dropna()
    return float(usable.iloc[-1]) if not usable.empty else None


# --------------------------------------------------------------------------- inflation


def energy_inflation_path(context: ForecastContext, inputs: QuadInputs, base_mom: float) -> dict[pd.Timestamp, float]:
    """Headline CPI monthly changes for unreported months using gasoline pass-through."""
    gas = context.lagged_as_of("GASREGW", inputs.as_of)
    if gas.empty:
        return {}
    gas_monthly = gas.groupby(gas.index.to_period("M").start_time).mean()
    gas_monthly.index = pd.DatetimeIndex(gas_monthly.index)
    gas_change = 100.0 * np.log(gas_monthly / gas_monthly.shift(1))
    headline = 100.0 * (inputs.cpi / inputs.cpi.shift(1) - 1.0)
    core = 100.0 * (inputs.core_cpi / inputs.core_cpi.shift(1) - 1.0)
    last_cpi = inputs.cpi.index.max()
    # Only complete gasoline months are used for training.
    complete_gas = gas_change[gas_change.index < quarter_start(inputs.as_of).to_period("M").start_time]
    train = pd.concat([(headline - core).rename("gap"), complete_gas.rename("gas")], axis=1).dropna()
    train = train[train.index <= last_cpi]
    if len(train) < 36:
        return {}
    coefficients = _ols(train[["gas"]].to_numpy(), train["gap"].to_numpy())
    current_month = inputs.as_of.to_period("M").start_time
    last_price = float(gas.iloc[-1])
    path: dict[pd.Timestamp, float] = {}
    month = (last_cpi + pd.offsets.MonthBegin(1)).normalize()
    horizon = (quarter_start(inputs.as_of) + pd.DateOffset(months=12)).normalize()
    while month <= horizon:
        if month in gas_monthly.index and month <= current_month:
            change = float(gas_change.get(month, np.nan))
        elif month == current_month + pd.offsets.MonthBegin(1) and current_month in gas_monthly.index:
            change = 100.0 * np.log(last_price / gas_monthly.loc[current_month])
        else:
            change = None  # no gasoline information: fall back to the core run-rate
        if change is None or not np.isfinite(change):
            path[month] = base_mom
        else:
            path[month] = base_mom + float(coefficients[0]) + float(coefficients[1]) * change
        month = (month + pd.offsets.MonthBegin(1)).normalize()
    return path


@dataclass(frozen=True)
class ModelOverrides:
    growth: dict
    inflation: dict
    growth_sources: dict


def model_overrides(
    context: ForecastContext,
    inputs: QuadInputs,
    settings: QuadSettings,
    growth_model: str = "leading",
    inflation_model: str = "energy",
) -> ModelOverrides:
    """Growth and inflation assumptions for every unreported period through the horizon."""
    current = quarter_start(inputs.as_of)
    horizon_end = (current + pd.DateOffset(months=3 * settings.forward_quarters)).normalize()
    unreported = list(
        pd.date_range((inputs.gdp.index.max() + pd.offsets.QuarterBegin(startingMonth=1)).normalize(), horizon_end, freq="QS")
    )
    growth: dict = {}
    sources: dict = {}
    if growth_model in ("bridge", "bridge_nowcast", "leading"):
        nowcast_quarters = [quarter for quarter in unreported if quarter <= current]
        bridge = bridge_nowcast(context, inputs, nowcast_quarters)
        for quarter in nowcast_quarters:
            if growth_model != "bridge" and settings.use_nowcast and quarter in inputs.nowcast_saar.index:
                growth[quarter] = float(inputs.nowcast_saar[quarter])
                sources[quarter] = "nowcast"
            elif quarter in bridge:
                growth[quarter] = bridge[quarter]
                sources[quarter] = "bridge"
        if growth_model == "leading" and current in growth:
            later = [quarter for quarter in unreported if quarter > current]
            for quarter, value in leading_forecast(context, inputs, current, growth[current], later).items():
                growth[quarter] = value
                sources[quarter] = "leading"
    inflation: dict = {}
    if inflation_model == "energy":
        base_mom = float(
            (100.0 * (inputs.core_cpi / inputs.core_cpi.shift(1) - 1.0)).dropna().tail(settings.inflation_trend_months).mean()
        )
        inflation = energy_inflation_path(context, inputs, base_mom)
    return ModelOverrides(growth=growth, inflation=inflation, growth_sources=sources)


def calibrated_probabilities(
    records: pd.DataFrame,
    minimum_history: int = 24,
    knowledge_lag_days: int = 120,
) -> pd.DataFrame:
    """Quad probabilities from the empirical distribution of past forecast errors at the same horizon.

    Only errors for quarters whose truth would have been known (quarter end plus
    ``knowledge_lag_days``) before the forecast date are used.
    """
    frame = records.sort_values("forecast_date").copy()
    frame["known_after"] = frame["quarter_start"] + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=knowledge_lag_days)
    frame["growth_error"] = frame["realized_growth_change"] - frame["forecast_growth_change"]
    frame["inflation_error"] = frame["realized_inflation_change"] - frame["forecast_inflation_change"]
    out = []
    for horizon, group in frame.groupby("horizon"):
        history = group.dropna(subset=["growth_error", "inflation_error"])
        for row in group.itertuples():
            past = history[history["known_after"] <= row.forecast_date]
            if len(past) < minimum_history or row.forecast_growth_change is None or pd.isna(row.forecast_growth_change):
                continue
            growth_up = float(np.mean(row.forecast_growth_change + past["growth_error"].to_numpy() > 0))
            inflation_up = float(np.mean(row.forecast_inflation_change + past["inflation_error"].to_numpy() > 0))
            probabilities = {
                1: growth_up * (1 - inflation_up),
                2: growth_up * inflation_up,
                3: (1 - growth_up) * inflation_up,
                4: (1 - growth_up) * (1 - inflation_up),
            }
            out.append(
                {
                    "forecast_date": row.forecast_date,
                    "quarter": row.quarter,
                    "horizon": horizon,
                    "p_growth_up": growth_up,
                    "p_inflation_up": inflation_up,
                    **{f"p_q{quad}": value for quad, value in probabilities.items()},
                    "realized_quad": row.realized_quad,
                    **{f"scenario_q{quad}": getattr(row, f"scenario_q{quad}") for quad in (1, 2, 3, 4)},
                }
            )
    return pd.DataFrame.from_records(out)


def score_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    frame = probabilities.dropna(subset=["realized_quad"]).copy()
    rows = []
    for horizon, group in frame.groupby("horizon"):
        truth = np.stack([(group["realized_quad"] == quad).astype(float).to_numpy() for quad in (1, 2, 3, 4)], axis=1)
        calibrated = group[[f"p_q{quad}" for quad in (1, 2, 3, 4)]].to_numpy()
        scenario = group[[f"scenario_q{quad}" for quad in (1, 2, 3, 4)]].to_numpy()
        climatology = np.full_like(calibrated, 0.25)
        confident = group[calibrated.max(axis=1) >= 0.5]
        rows.append(
            {
                "horizon": horizon,
                "observations": len(group),
                "brier_calibrated": float(np.mean(np.sum((calibrated - truth) ** 2, axis=1))),
                "brier_scenario_grid": float(np.mean(np.sum((scenario - truth) ** 2, axis=1))),
                "brier_uniform": float(np.mean(np.sum((climatology - truth) ** 2, axis=1))),
                "top_quad_hit": float(np.mean(calibrated.argmax(axis=1) + 1 == group["realized_quad"].to_numpy())),
                "confident_share": float(len(confident) / len(group)),
                "confident_hit": float(
                    np.mean(confident[[f"p_q{quad}" for quad in (1, 2, 3, 4)]].to_numpy().argmax(axis=1) + 1 == confident["realized_quad"].to_numpy())
                )
                if len(confident)
                else None,
                "mean_top_probability": float(calibrated.max(axis=1).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def quad_probabilities_from_errors(
    forecast_growth_change: float | None,
    forecast_inflation_change: float | None,
    growth_errors: np.ndarray,
    inflation_errors: np.ndarray,
) -> dict | None:
    """Quad probabilities for one forecast, given past (realized - forecast) errors at the same horizon."""
    if forecast_growth_change is None or forecast_inflation_change is None or len(growth_errors) == 0:
        return None
    growth_up = float(np.mean(forecast_growth_change + growth_errors > 0))
    inflation_up = float(np.mean(forecast_inflation_change + inflation_errors > 0))
    return {
        "p_growth_up": growth_up,
        "p_inflation_up": inflation_up,
        "quads": {
            "1": growth_up * (1 - inflation_up),
            "2": growth_up * inflation_up,
            "3": (1 - growth_up) * inflation_up,
            "4": (1 - growth_up) * (1 - inflation_up),
        },
    }
