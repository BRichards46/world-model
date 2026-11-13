"""Growth × inflation quad with payrolls as the growth leg.

The GDP quad (``growth_inflation_quad``) measures growth with real GDP, which is quarterly, published a month
after the quarter ends and heavily revised. This variant measures growth with nonfarm payrolls:

- **Growth:** year-over-year growth of quarterly-average payrolls (``PAYEMS``). A quarter's growth is
  accelerating when that rate rises versus the prior quarter.
- **Inflation:** the same leg as the GDP quad (quarterly average of headline CPI year-over-year rates), taken
  from that model's own forecast so the two quads differ only in growth.
- **Unreported months:** payrolls are extended at their average monthly change over the last three months.

In the 1998–2026 walk-forward the payroll quad's current-quarter growth direction was right 82% of the time
(GDP: 68%), and its projected Quad 3 was followed by the S&P 500, small caps, discretionary and materials
lagging and utilities leading, in both halves of the sample (docs/growth_inflation_quad.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table
from world_model.features.growth_inflation_quad import (
    QUAD_DESCRIPTIONS,
    classify_quad,
    cpi_quarterly_yoy,
    monthly_levels,
    quarter_label,
    quarter_start,
)
from world_model.features.macro_dashboard_details import point_in_time_series
from world_model.features.monthly_windows import fill_missing_months

HORIZON_LABELS = {0: "current quarter", 1: "next quarter", 2: "two quarters ahead"}


@dataclass(frozen=True)
class LaborQuadSettings:
    pace_months: int = 3
    forward_quarters: int = 2
    history_quarters: int = 8
    minimum_history: int = 24
    knowledge_lag_days: int = 120

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LaborQuadSettings":
        values = {**dict(config.get("settings") or {}), **dict(config.get("probabilities") or {})}
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


# --------------------------------------------------------------------------- growth leg


def extend_at_pace(monthly: pd.Series, through: pd.Timestamp, months: int = 3) -> pd.Series:
    """Extend a monthly level series through ``through`` at its average log change over the last ``months``."""
    values = monthly.dropna().sort_index()
    if values.empty:
        return values
    step = float(np.log(values).diff().tail(months).mean()) if len(values) > months else 0.0
    result = values.copy()
    current = result.index.max()
    target = pd.Timestamp(through).to_period("M").start_time
    while current < target:
        following = (current + pd.offsets.MonthBegin(1)).normalize()
        result.loc[following] = result.loc[current] * np.exp(step)
        current = following
    return result


def quarterly_average(monthly: pd.Series) -> pd.Series:
    """Average level of each quarter with all three months present, on quarter-start dates."""
    if monthly.empty:
        return monthly.astype(float)
    grouped = monthly.groupby(monthly.index.to_period("Q").start_time)
    average = grouped.mean()[grouped.count() == 3]
    average.index = pd.DatetimeIndex(average.index)
    if average.empty:
        return average
    return average.reindex(pd.date_range(average.index.min(), average.index.max(), freq="QS"))


def growth_path(known: pd.Series, through: pd.Timestamp, pace_months: int = 3) -> pd.DataFrame:
    """Payroll YoY growth and its quarterly change, with unreported months extended at the recent pace.

    ``months_reported`` counts the quarter's months that were actually published; ``source`` is
    ``reported`` (all three), ``partial`` or ``projected``.
    """
    extended = extend_at_pace(known, through, pace_months)
    average = quarterly_average(extended)
    yoy = 100.0 * (average / average.shift(4) - 1.0)
    frame = pd.DataFrame({"growth_yoy": yoy, "growth_change": yoy.diff()})
    reported = known.dropna().groupby(known.dropna().index.to_period("Q").start_time).size()
    reported.index = pd.DatetimeIndex(reported.index)
    frame["months_reported"] = reported.reindex(frame.index).fillna(0).astype(int)
    frame["source"] = np.select([frame["months_reported"] >= 3, frame["months_reported"] > 0], ["reported", "partial"], "projected")
    return frame


def inflation_changes(cpi_levels: pd.Series) -> pd.Series:
    """Quarterly change in the quarterly average of headline CPI YoY rates (complete quarters only)."""
    return cpi_quarterly_yoy(cpi_levels).diff()


def growth_probability(forecast_change: float, past_errors: np.ndarray) -> float | None:
    """Probability growth is accelerating, from past (realized - forecast) errors at the same horizon."""
    if past_errors.size == 0 or not np.isfinite(forecast_change):
        return None
    return float(np.mean(forecast_change + past_errors > 0))


def quad_probabilities(p_growth_up: float | None, p_inflation_up: float | None) -> dict[str, float] | None:
    if p_growth_up is None or p_inflation_up is None:
        return None
    g, i = p_growth_up, p_inflation_up
    return {"1": g * (1 - i), "2": g * i, "3": (1 - g) * i, "4": (1 - g) * (1 - i)}


# --------------------------------------------------------------------------- engine


class LaborQuadEngine:
    """Point-in-time payroll quad, its walk-forward, and the live outlook."""

    def __init__(self, data_roots: Mapping[str, Any], config: Mapping[str, Any]):
        self.config = dict(config)
        self.settings = LaborQuadSettings.from_config(self.config)
        series = self.config.get("series", {})
        self.series_ids = {"payrolls": str(series.get("growth", "PAYEMS")), "cpi": str(series.get("inflation", "CPIAUCSL"))}
        self.tables = {}
        for key, series_id in self.series_ids.items():
            table = load_vintage_table(data_roots["fred"], series_id).copy()
            table["realtime_start"] = pd.to_datetime(table["realtime_start"].astype(str).str[:10])
            self.tables[key] = table.sort_values("realtime_start")

    def payrolls_as_of(self, as_of: pd.Timestamp) -> pd.Series:
        table = self.tables["payrolls"]
        values, _ = fill_missing_months(point_in_time_series(table[table["realtime_start"] <= as_of]))
        return values

    def cpi_as_of(self, as_of: pd.Timestamp) -> pd.Series:
        table = self.tables["cpi"]
        values, _ = monthly_levels(point_in_time_series(table[table["realtime_start"] <= as_of]))
        return values

    def walk_forward(
        self,
        as_of: pd.Timestamp,
        inflation_forecasts: pd.DataFrame,
        start: str | None = None,
    ) -> pd.DataFrame:
        """Month-end payroll-quad projections from data known at the time, scored against the latest data.

        ``inflation_forecasts`` is the GDP quad's walk-forward (``forecast_date``, ``quarter_start``,
        ``forecast_inflation_change``): the inflation leg both quads share.
        """
        settings = self.settings
        start = start or str(self.config.get("backtest", {}).get("start", "1998-01-31"))
        latest = growth_path(self.payrolls_as_of(as_of), as_of, settings.pace_months)
        truth_growth = latest.loc[latest["source"] == "reported", "growth_change"]
        truth_inflation = inflation_changes(self.cpi_as_of(as_of))
        forecasts = inflation_forecasts.assign(
            forecast_date=pd.to_datetime(inflation_forecasts["forecast_date"]),
            quarter_start=pd.to_datetime(inflation_forecasts["quarter_start"]),
        ).set_index(["forecast_date", "quarter_start"])["forecast_inflation_change"]
        records = []
        for date in pd.date_range(start, as_of, freq="ME"):
            current = quarter_start(date)
            last = (current + pd.DateOffset(months=3 * settings.forward_quarters)).normalize()
            path = growth_path(self.payrolls_as_of(date), last + pd.offsets.MonthBegin(2), settings.pace_months)
            cpi_known = inflation_changes(self.cpi_as_of(date)).dropna()
            reported = path[path["source"] == "reported"].index.intersection(cpi_known.index)
            persistence = None
            if len(reported):
                quarter = reported.max()
                persistence = classify_quad(float(path.loc[quarter, "growth_change"]), float(cpi_known[quarter]))
            for quarter in pd.date_range(current, last, freq="QS"):
                if quarter not in path.index:
                    continue
                growth = float(path.loc[quarter, "growth_change"])
                inflation = float(cpi_known[quarter]) if quarter in cpi_known.index else forecasts.get((date, quarter), np.nan)
                realized_growth = truth_growth.get(quarter, np.nan)
                realized_inflation = truth_inflation.get(quarter, np.nan)
                records.append(
                    {
                        "forecast_date": date,
                        "quarter": quarter_label(quarter),
                        "quarter_start": quarter,
                        "horizon": int(round((quarter - current).days / 91.3)),
                        "growth_source": path.loc[quarter, "source"],
                        "forecast_growth_change": growth,
                        "forecast_inflation_change": inflation,
                        "realized_growth_change": realized_growth,
                        "realized_inflation_change": realized_inflation,
                        "projected_quad": _quad(growth, inflation),
                        "persistence_quad": persistence,
                        "realized_quad": _quad(realized_growth, realized_inflation),
                    }
                )
        return pd.DataFrame.from_records(records)

    def outlook(self, as_of: pd.Timestamp, gdp_quad: Mapping[str, Any], walk_forward: pd.DataFrame) -> dict[str, Any]:
        """Payroll-quad quarters for the dashboard: recent realized quarters and the ones ahead.

        Inflation comes from the GDP quad's payload for the same quarters (realized or projected).
        """
        settings = self.settings
        current = quarter_start(as_of)
        last = (current + pd.DateOffset(months=3 * settings.forward_quarters)).normalize()
        known = self.payrolls_as_of(as_of)
        path = growth_path(known, last + pd.offsets.MonthBegin(2), settings.pace_months)
        inflation = {pd.Timestamp(row["quarter_start"]): row for row in gdp_quad.get("quarters", [])}
        errors = _growth_errors(walk_forward, as_of, settings)
        rows = []
        first = current - pd.DateOffset(months=3 * settings.history_quarters)
        for quarter in path.index[(path.index >= first) & (path.index <= last)]:
            source = str(path.loc[quarter, "source"])
            growth = float(path.loc[quarter, "growth_change"])
            gdp_row = inflation.get(quarter, {})
            inflation_change = gdp_row.get("inflation_change")
            realized = source == "reported" and gdp_row.get("inflation_months_reported") == 3
            row = {
                "quarter": quarter_label(quarter),
                "quarter_start": f"{quarter:%Y-%m-%d}",
                "status": "realized" if realized else ("partial" if source == "partial" or quarter <= current else "projected"),
                "is_current_quarter": bool(quarter == current),
                "payroll_months_reported": int(path.loc[quarter, "months_reported"]),
                "growth_yoy": _float(path.loc[quarter, "growth_yoy"]),
                "growth_change": _float(growth),
                "inflation_yoy": gdp_row.get("inflation_yoy"),
                "inflation_change": inflation_change,
                "quad": _quad(growth, inflation_change),
            }
            if not realized:
                horizon = int(round((quarter - current).days / 91.3))
                p_growth = float(growth > 0) if source == "reported" else growth_probability(growth, errors.get(horizon, np.array([])))
                p_inflation = 1.0 * (inflation_change > 0) if gdp_row.get("inflation_months_reported") == 3 and inflation_change is not None else gdp_row.get("p_inflation_up")
                probabilities = quad_probabilities(p_growth, p_inflation)
                row.update({"horizon": horizon, "p_growth_up": p_growth, "p_inflation_up": p_inflation, "probabilities": probabilities})
            rows.append(row)
        return {
            "as_of_date": f"{as_of:%Y-%m-%d}",
            "current_quarter": quarter_label(current),
            "last_payroll_month": f"{known.index.max():%Y-%m}" if not known.empty else None,
            "quarters": rows,
            "quad_descriptions": {str(quad): text for quad, text in QUAD_DESCRIPTIONS.items()},
            "series": self.series_ids,
        }


def _growth_errors(walk_forward: pd.DataFrame, as_of: pd.Timestamp, settings: LaborQuadSettings) -> dict[int, np.ndarray]:
    """Past (realized - forecast) growth errors by horizon, for quarters whose outcome was known by ``as_of``."""
    if walk_forward.empty:
        return {}
    history = walk_forward.dropna(subset=["realized_growth_change", "forecast_growth_change"])
    known_after = pd.to_datetime(history["quarter_start"]) + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=settings.knowledge_lag_days)
    history = history[(known_after <= as_of) & (history["growth_source"] != "reported")]
    out = {}
    for horizon, group in history.groupby("horizon"):
        if len(group) >= settings.minimum_history:
            out[int(horizon)] = (group["realized_growth_change"] - group["forecast_growth_change"]).to_numpy(float)
    return out


def summarize_walk_forward(labor: pd.DataFrame, gdp: pd.DataFrame | None = None) -> dict[str, Any]:
    """Hit rates by horizon for the payroll quad, with the GDP quad on the same month-ends for comparison."""
    scored = labor.dropna(subset=["realized_quad", "projected_quad"])
    rows = []
    for horizon, group in scored.groupby("horizon"):
        row = {
            "horizon": int(horizon),
            "label": HORIZON_LABELS.get(int(horizon), f"{horizon} quarters ahead"),
            "observations": int(len(group)),
            "quad_hit_rate": float((group["projected_quad"] == group["realized_quad"]).mean()),
            "growth_direction_hit_rate": _direction_rate(group, "growth"),
            "inflation_direction_hit_rate": _direction_rate(group, "inflation"),
            "persistence_hit_rate": float((group["persistence_quad"] == group["realized_quad"]).mean()),
        }
        if gdp is not None and not gdp.empty:
            other = gdp[(gdp["horizon"] == horizon)].dropna(subset=["realized_quad", "projected_quad"])
            other = other[pd.to_datetime(other["forecast_date"]).isin(pd.to_datetime(group["forecast_date"]))]
            if len(other):
                row["gdp_quad_hit_rate"] = float((other["projected_quad"] == other["realized_quad"]).mean())
                row["gdp_growth_direction_hit_rate"] = _direction_rate(other, "growth")
        rows.append(row)
    dates = pd.to_datetime(scored["forecast_date"]) if len(scored) else pd.Series(dtype="datetime64[ns]")
    return {
        "start": f"{dates.min():%Y-%m}" if len(dates) else None,
        "end": f"{dates.max():%Y-%m}" if len(dates) else None,
        "horizons": rows,
        "note": "Month-end projections made with payroll and CPI vintages known at the time, scored against the latest revised data. "
        "GDP columns score the GDP quad on the same month-ends.",
    }


def _direction_rate(frame: pd.DataFrame, leg: str) -> float | None:
    usable = frame.dropna(subset=[f"forecast_{leg}_change", f"realized_{leg}_change"])
    if usable.empty:
        return None
    return float(((usable[f"forecast_{leg}_change"] > 0) == (usable[f"realized_{leg}_change"] > 0)).mean())


# --------------------------------------------------------------------------- Fed path


def fed_path_by_quad(
    signal: pd.Series,
    fed_funds: pd.Series,
    months: int,
    minimum_rotation_gap: int = 12,
) -> dict[str, Any]:
    """Average change in the fed funds rate over the ``months`` after each signal date, by quad.

    ``p_value`` is a circular-rotation test of the spread between quads (it keeps both series' autocorrelation).
    """
    labels = signal.dropna().astype(int)
    rates = fed_funds.dropna().sort_index()
    if labels.empty or rates.empty:
        return {"months": months, "by_quad": {}, "p_value": None, "observations": 0}
    ends = labels.index + pd.DateOffset(months=months)
    usable = ends <= rates.index.max()
    labels = labels[usable]
    change = pd.Series(rates.asof(labels.index + pd.DateOffset(months=months)).to_numpy() - rates.asof(labels.index).to_numpy(), index=labels.index)
    frame = pd.concat([labels.rename("quad"), change.rename("change")], axis=1).dropna()
    quads, values = frame["quad"].to_numpy(int), frame["change"].to_numpy(float)

    def dispersion(assigned: np.ndarray) -> float:
        mean = values.mean()
        return float(sum((assigned == q).sum() * (values[assigned == q].mean() - mean) ** 2 for q in (1, 2, 3, 4) if (assigned == q).any()))

    observed = dispersion(quads)
    rotated = [dispersion(np.roll(quads, shift)) for shift in range(minimum_rotation_gap, len(quads) - minimum_rotation_gap + 1)]
    p_value = float((np.sum(np.asarray(rotated) >= observed) + 1) / (len(rotated) + 1)) if rotated else None
    by_quad = {
        str(q): {"months": int((quads == q).sum()), "mean_change": float(values[quads == q].mean())}
        for q in (1, 2, 3, 4)
        if (quads == q).any()
    }
    return {"months": months, "by_quad": by_quad, "p_value": p_value, "observations": int(len(values)),
            "share_explained": float(observed / (values.var() * len(values))) if values.var() > 0 else None}


def known_quads_from(walk_forward: pd.DataFrame, knowledge_lag_days: int = 120) -> Callable[[pd.Timestamp], dict[pd.Timestamp, int]]:
    """Quarters' realized payroll quads, counted as known ``knowledge_lag_days`` after each quarter ends.

    Used by the playbook test's learned playbook. Realized quads come from the latest revised data.
    """
    realized = walk_forward.dropna(subset=["realized_quad"]).groupby("quarter_start")["realized_quad"].first()
    realized.index = pd.DatetimeIndex(realized.index)
    lag = pd.Timedelta(days=knowledge_lag_days)

    def known(as_of: pd.Timestamp) -> dict[pd.Timestamp, int]:
        return {quarter: int(quad) for quarter, quad in realized.items() if quarter + pd.offsets.QuarterEnd(0) + lag <= as_of}

    return known


def _quad(growth: Any, inflation: Any) -> int | None:
    if growth is None or inflation is None:
        return None
    if not (np.isfinite(growth) and np.isfinite(inflation)):
        return None
    return classify_quad(float(growth), float(inflation))


def _float(value: Any) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None
