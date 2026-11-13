from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.data.fred import load_vintage_table
from world_model.features.growth_inflation_quad import (
    QUAD_DESCRIPTIONS,
    QuadInputs,
    QuadSettings,
    build_quad_outlook,
    classify_quad,
    cpi_quarterly_yoy,
    gdp_yoy,
    monthly_levels,
    quarter_label,
    quarter_start,
    quarterly_levels,
)
from world_model.features.macro_dashboard_details import point_in_time_series
from world_model.features.quad_forecast_models import (
    ForecastContext,
    calibrated_probabilities,
    model_overrides,
    quad_probabilities_from_errors,
    score_probabilities,
)


@dataclass(frozen=True)
class QuadResult:
    as_of_date: pd.Timestamp
    payload: dict[str, Any]
    backtest: pd.DataFrame


class GrowthInflationQuadEngine:
    """Point-in-time growth/inflation quads with base-effect look-ahead and a walk-forward check."""

    def __init__(self, data_roots: Mapping[str, Any], config: Mapping[str, Any]):
        self.fred_root = data_roots["fred"]
        self.config = dict(config)
        self.settings = QuadSettings.from_mapping(self.config.get("settings"))
        series = self.config.get("series", {})
        self.series_ids = {
            "gdp": str(series.get("growth", "GDPC1")),
            "cpi": str(series.get("inflation", "CPIAUCSL")),
            "core_cpi": str(series.get("inflation_trend", "CPILFESL")),
            "nowcast": str(series.get("growth_nowcast", "GDPNOW")),
        }
        self.tables = {key: load_vintage_table(self.fred_root, series_id) for key, series_id in self.series_ids.items()}
        models = self.config.get("models", {})
        self.growth_model = str(models.get("growth", "leading"))
        self.inflation_model = str(models.get("inflation", "energy"))
        self.probability_settings = dict(models.get("probabilities", {}))
        uses_models = self.growth_model != "trend" or self.inflation_model != "core"
        self.context = ForecastContext.load(self.fred_root) if uses_models else None

    def outlook_for(self, inputs: QuadInputs, include_hurdles: bool = True) -> dict[str, Any]:
        """Quad outlook with the configured growth and inflation models applied."""
        if self.context is None:
            return build_quad_outlook(inputs, self.settings, include_hurdles=include_hurdles)
        overrides = model_overrides(self.context, inputs, self.settings, self.growth_model, self.inflation_model)
        return build_quad_outlook(
            inputs,
            self.settings,
            include_hurdles=include_hurdles,
            growth_override=overrides.growth,
            inflation_override=overrides.inflation,
            override_source=overrides.growth_sources,
        )

    def inputs_as_of(self, as_of: str | pd.Timestamp) -> QuadInputs:
        as_of_date = pd.Timestamp(as_of).normalize()
        known = {key: point_in_time_series(table[table["realtime_start"] <= as_of_date]) for key, table in self.tables.items()}
        cpi, interpolated = monthly_levels(known["cpi"])
        core_cpi, _ = monthly_levels(known["core_cpi"])
        nowcast = known["nowcast"].dropna()
        nowcast.index = pd.DatetimeIndex(nowcast.index).to_period("Q").start_time.normalize()
        return QuadInputs(
            as_of=as_of_date,
            gdp=quarterly_levels(known["gdp"]),
            cpi=cpi,
            core_cpi=core_cpi,
            nowcast_saar=nowcast[~nowcast.index.duplicated(keep="last")],
            interpolated_cpi_months=tuple(interpolated),
        )

    def build(self, as_of: str | pd.Timestamp | None = None, run_backtest: bool = True) -> QuadResult:
        as_of_date = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
        outlook = self.outlook_for(self.inputs_as_of(as_of_date))
        backtest = self.walk_forward(as_of_date) if run_backtest else pd.DataFrame()
        if not backtest.empty:
            attach_probabilities(outlook, backtest, as_of_date, self.probability_settings)
        payload = {
            **outlook,
            "quad_descriptions": {str(quad): text for quad, text in QUAD_DESCRIPTIONS.items()},
            "series": self.series_ids,
            "backtest": summarize_backtest(backtest),
            "method": {
                "growth": "Real GDP year-over-year growth by quarter. The current quarter uses GDPNow when available, otherwise a bridge from monthly real consumer spending, industrial production and payrolls. Later quarters use a ridge model on the nowcast, trend growth, jobless claims, the 10y–3m curve and building permits.",
                "inflation": "Quarterly average of monthly headline CPI year-over-year rates. Unreported months use the six-month core CPI run-rate plus gasoline pass-through where gasoline prices are known (month to date, and next month's carry-over at a flat price).",
                "scenarios": "Probabilities come from the empirical distribution of this model's own past forecast errors at the same horizon, using only quarters whose outcome was known by the as-of date. The 3×3 scenario grid (growth ±1.0 pp, inflation ±0.1 pp/month) is kept for reference but proved overconfident.",
                "hurdles": "A quarter's growth YoY accelerates only if its annualized growth beats the hurdle (the same quarter last year). Its inflation YoY accelerates only if monthly CPI beats the inflation hurdle.",
                "warning": "Descriptive rate-of-change regime, not a return forecast. Projections are mechanical base-effect arithmetic under stated assumptions; the walk-forward check shows how often they matched realized quads.",
            },
        }
        return QuadResult(as_of_date=as_of_date, payload=payload, backtest=backtest)

    def walk_forward(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Month-end projections made with vintages known at the time, scored against latest data."""
        backtest = self.config.get("backtest", {})
        start = pd.Timestamp(backtest.get("start", "1998-01-31"))
        final = self.inputs_as_of(as_of)
        realized = _realized_quads(final)
        records: list[dict[str, Any]] = []
        for date in pd.date_range(start, as_of, freq="ME"):
            inputs = self.inputs_as_of(date)
            if inputs.gdp.empty or inputs.cpi.empty or inputs.core_cpi.empty:
                continue
            outlook = self.outlook_for(inputs, include_hurdles=False)
            prior_realized = [row for row in outlook["quarters"] if row["status"] == "realized"]
            persistence = prior_realized[-1]["quad"] if prior_realized else None
            current = quarter_start(date)
            for row in outlook["quarters"]:
                if row["status"] == "realized":
                    continue
                quarter = pd.Timestamp(row["quarter_start"])
                horizon = int(round((quarter - current).days / 91.3))
                # Keep unrealized targets (truth unknown) so the signal history runs to the as-of date.
                truth = realized.get(quarter) or {
                    "quad": None,
                    "growth_up": None,
                    "inflation_up": None,
                    "growth_change": None,
                    "inflation_change": None,
                }
                counts = row.get("scenario_counts") or {}
                records.append(
                    {
                        "forecast_date": date,
                        "quarter": quarter_label(quarter),
                        "quarter_start": quarter,
                        "horizon": horizon,
                        "forecast_growth_change": row["growth_change"],
                        "forecast_inflation_change": row["inflation_change"],
                        "realized_growth_change": truth["growth_change"],
                        "realized_inflation_change": truth["inflation_change"],
                        **{f"scenario_q{quad}": counts.get(str(quad), 0) / 9.0 for quad in (1, 2, 3, 4)},
                        "projected_quad": row["quad"],
                        "agreement": row.get("agreement"),
                        "persistence_quad": persistence,
                        "realized_quad": truth["quad"],
                        "projected_growth_up": _up(row["growth_change"]),
                        "projected_inflation_up": _up(row["inflation_change"]),
                        "realized_growth_up": truth["growth_up"],
                        "realized_inflation_up": truth["inflation_up"],
                    }
                )
        return pd.DataFrame.from_records(records)


def _realized_quads(inputs: QuadInputs) -> dict[pd.Timestamp, dict[str, Any]]:
    growth = gdp_yoy(inputs.gdp).dropna()
    inflation = cpi_quarterly_yoy(inputs.cpi)
    frame = pd.concat([growth.rename("growth"), inflation.rename("inflation")], axis=1).dropna()
    changes = frame.diff().dropna()
    return {
        quarter: {
            "quad": classify_quad(row["growth"], row["inflation"]),
            "growth_up": bool(row["growth"] > 0),
            "inflation_up": bool(row["inflation"] > 0),
            "growth_change": float(row["growth"]),
            "inflation_change": float(row["inflation"]),
        }
        for quarter, row in changes.iterrows()
    }


def attach_probabilities(
    outlook: dict[str, Any],
    backtest: pd.DataFrame,
    as_of: pd.Timestamp,
    settings: Mapping[str, Any] | None = None,
) -> None:
    """Add calibrated quad probabilities to each unrealized quarter of ``outlook`` in place."""
    settings = settings or {}
    minimum = int(settings.get("minimum_history", 24))
    lag = pd.Timedelta(days=int(settings.get("knowledge_lag_days", 120)))
    history = backtest.dropna(subset=["realized_growth_change", "realized_inflation_change", "forecast_growth_change"]).copy()
    history["known_after"] = pd.to_datetime(history["quarter_start"]) + pd.offsets.QuarterEnd(0) + lag
    history = history[history["known_after"] <= as_of]
    current = quarter_start(as_of)
    for row in outlook["quarters"]:
        if row["status"] == "realized":
            continue
        horizon = int(round((pd.Timestamp(row["quarter_start"]) - current).days / 91.3))
        past = history[history["horizon"] == horizon]
        if len(past) < minimum:
            continue
        result = quad_probabilities_from_errors(
            row["growth_change"],
            row["inflation_change"],
            (past["realized_growth_change"] - past["forecast_growth_change"]).to_numpy(dtype=float),
            (past["realized_inflation_change"] - past["forecast_inflation_change"]).to_numpy(dtype=float),
        )
        if result is None:
            continue
        row["probabilities"] = result["quads"]
        row["p_growth_up"] = result["p_growth_up"]
        row["p_inflation_up"] = result["p_inflation_up"]
        top = max(result["quads"], key=result["quads"].get)
        row["probability_quad"] = int(top)
        row["probability_top"] = result["quads"][top]
        row["probability_history"] = int(len(past))


def _up(value: float | None) -> bool | None:
    return None if value is None else bool(value > 0)


def summarize_backtest(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"horizons": [], "observations": 0}
    rows = []
    for horizon, group in frame.groupby("horizon"):
        group = group.dropna(subset=["projected_quad", "realized_quad"])
        # Month-ends where the prior quarter's GDP was still unreported are too rare to score.
        if len(group) < 30:
            continue
        persistence = group.dropna(subset=["persistence_quad"])
        high = group[group["agreement"].fillna(0) >= 7 / 9]
        rows.append(
            {
                "horizon": int(horizon),
                "label": "current quarter" if horizon == 0 else ("prior quarter (unreported)" if horizon < 0 else f"{horizon} quarter{'s' if horizon > 1 else ''} ahead"),
                "observations": int(len(group)),
                "quad_hit_rate": _rate(group["projected_quad"] == group["realized_quad"]),
                "persistence_hit_rate": _rate(persistence["persistence_quad"] == persistence["realized_quad"]),
                "unconditional_hit_rate": _modal_rate(group["realized_quad"]),
                "growth_direction_hit_rate": _rate(group["projected_growth_up"] == group["realized_growth_up"]),
                "inflation_direction_hit_rate": _rate(group["projected_inflation_up"] == group["realized_inflation_up"]),
                "high_agreement_observations": int(len(high)),
                "high_agreement_hit_rate": _rate(high["projected_quad"] == high["realized_quad"]) if len(high) else None,
            }
        )
    try:
        probability_scores = score_probabilities(calibrated_probabilities(frame)).set_index("horizon")
    except (KeyError, ValueError):
        probability_scores = pd.DataFrame()
    for row in rows:
        if row["horizon"] in probability_scores.index:
            scores = probability_scores.loc[row["horizon"]]
            row["brier_calibrated"] = float(scores["brier_calibrated"])
            row["brier_scenario_grid"] = float(scores["brier_scenario_grid"])
            row["brier_uniform"] = float(scores["brier_uniform"])
            row["confident_share"] = float(scores["confident_share"])
            row["confident_hit_rate"] = None if pd.isna(scores["confident_hit"]) else float(scores["confident_hit"])
    return {
        "observations": int(frame["realized_quad"].notna().sum()),
        "start": pd.Timestamp(frame["forecast_date"].min()).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(frame.loc[frame["realized_quad"].notna(), "forecast_date"].max()).strftime("%Y-%m-%d"),
        "horizons": rows,
        "note": "Projections use only vintages known at each month-end; realized quads use the latest revised data. Persistence assumes the last realized quad repeats.",
    }


def _rate(values: pd.Series) -> float | None:
    return float(np.mean(values.astype(float))) if len(values) else None


def _modal_rate(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.value_counts(normalize=True).iloc[0])
