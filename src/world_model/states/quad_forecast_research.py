"""Walk-forward comparison of quad forecasting variants.

Each month-end, every variant builds growth and inflation assumptions from data
known on that date (see ``world_model.features.quad_forecast_models``), runs the
standard quad projection, and is scored against the realized quads from the
latest revised data.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.features.growth_inflation_quad import (
    QuadInputs,
    build_quad_outlook,
    classify_quad,
    cpi_quarterly_yoy,
    gdp_saar,
    gdp_yoy,
    quarter_label,
    quarter_start,
)
from world_model.features.quad_forecast_models import (  # noqa: F401  (re-exported for research use)
    ForecastContext,
    calibrated_probabilities,
    model_overrides,
    score_probabilities,
)
from world_model.states.growth_inflation_quad import GrowthInflationQuadEngine


GROWTH_VARIANTS = {"baseline": "trend", "bridge": "bridge", "bridge_nowcast": "bridge_nowcast", "leading": "leading"}
INFLATION_VARIANTS = {"baseline": "core", "energy": "energy"}


def realized_changes(inputs: QuadInputs) -> pd.DataFrame:
    growth = gdp_yoy(inputs.gdp).dropna()
    inflation = cpi_quarterly_yoy(inputs.cpi)
    frame = pd.concat([growth.rename("growth"), inflation.rename("inflation")], axis=1).dropna().diff().dropna()
    frame["quad"] = [classify_quad(g, i) for g, i in zip(frame["growth"], frame["inflation"])]
    frame["saar"] = gdp_saar(inputs.gdp).reindex(frame.index)
    return frame


def run_forecast_research(
    data_roots: Mapping[str, Any],
    quad_config: Mapping[str, Any],
    as_of: pd.Timestamp,
    start: str = "1998-01-31",
    combos: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    engine = GrowthInflationQuadEngine(data_roots, quad_config)
    context = ForecastContext.load(data_roots["fred"])
    settings = engine.settings
    combos = combos or [
        ("baseline", "baseline"),
        ("bridge", "baseline"),
        ("bridge_nowcast", "baseline"),
        ("leading", "baseline"),
        ("baseline", "energy"),
        ("leading", "energy"),
    ]
    truth = realized_changes(engine.inputs_as_of(as_of))
    rows: list[dict[str, Any]] = []
    for date in pd.date_range(start, as_of, freq="ME"):
        inputs = engine.inputs_as_of(date)
        if inputs.gdp.empty or inputs.cpi.empty:
            continue
        current = quarter_start(date)
        cache: dict[tuple[str, str], Any] = {}
        for growth_variant, inflation_variant in combos:
            key = (GROWTH_VARIANTS[growth_variant], INFLATION_VARIANTS[inflation_variant])
            if key not in cache:
                cache[key] = model_overrides(context, inputs, settings, *key)
            overrides = cache[key]
            outlook = build_quad_outlook(
                inputs,
                settings,
                include_hurdles=False,
                growth_override=overrides.growth,
                inflation_override=overrides.inflation,
            )
            for row in outlook["quarters"]:
                if row["status"] == "realized":
                    continue
                quarter = pd.Timestamp(row["quarter_start"])
                horizon = int(round((quarter - current).days / 91.3))
                actual = truth.loc[quarter] if quarter in truth.index else None
                counts = row.get("scenario_counts") or {}
                rows.append(
                    {
                        "forecast_date": date,
                        "growth_variant": growth_variant,
                        "inflation_variant": inflation_variant,
                        "quarter": quarter_label(quarter),
                        "quarter_start": quarter,
                        "horizon": horizon,
                        "growth_reported": row["growth_reported"],
                        "forecast_growth_change": row["growth_change"],
                        "forecast_inflation_change": row["inflation_change"],
                        "forecast_quad": row["quad"],
                        **{f"scenario_q{quad}": counts.get(str(quad), 0) / 9.0 for quad in (1, 2, 3, 4)},
                        "realized_growth_change": None if actual is None else float(actual["growth"]),
                        "realized_inflation_change": None if actual is None else float(actual["inflation"]),
                        "realized_quad": None if actual is None else actual["quad"],
                    }
                )
    return pd.DataFrame.from_records(rows)


# --------------------------------------------------------------------------- scoring


def score_variants(records: pd.DataFrame, horizons: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    scored = records.dropna(subset=["realized_quad", "forecast_quad"]).copy()
    scored = scored[scored["horizon"].isin(horizons)]
    scored["quad_hit"] = scored["forecast_quad"] == scored["realized_quad"]
    scored["growth_hit"] = (scored["forecast_growth_change"] > 0) == (scored["realized_growth_change"] > 0)
    scored["inflation_hit"] = (scored["forecast_inflation_change"] > 0) == (scored["realized_inflation_change"] > 0)
    scored["growth_error"] = scored["forecast_growth_change"] - scored["realized_growth_change"]
    scored["inflation_error"] = scored["forecast_inflation_change"] - scored["realized_inflation_change"]
    scored["era"] = np.where(scored["forecast_date"] < pd.Timestamp("2012-01-01"), "1998-2011", "2012-2026")
    summary = (
        scored.groupby(["growth_variant", "inflation_variant", "horizon"])
        .agg(
            observations=("quad_hit", "size"),
            quad_hit=("quad_hit", "mean"),
            growth_hit=("growth_hit", "mean"),
            inflation_hit=("inflation_hit", "mean"),
            growth_rmse=("growth_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            inflation_rmse=("inflation_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
    )
    by_era = (
        scored.groupby(["growth_variant", "inflation_variant", "horizon", "era"])["quad_hit"].mean().unstack("era").reset_index()
    )
    return summary.merge(by_era, on=["growth_variant", "inflation_variant", "horizon"], how="left")
