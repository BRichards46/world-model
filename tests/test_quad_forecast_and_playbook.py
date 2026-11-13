from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.features.growth_inflation_quad import QuadInputs, QuadSettings, build_quad_outlook
from world_model.features.quad_forecast_models import (
    ForecastContext,
    energy_inflation_path,
    quad_probabilities_from_errors,
)
from world_model.states.quad_playbook_test import learned_scores, portfolio_return, textbook_scores


def _inputs(as_of: str) -> QuadInputs:
    gdp_dates = pd.date_range("2015-01-01", "2022-10-01", freq="QS")
    cpi_dates = pd.date_range("2015-01-01", "2022-12-01", freq="MS")
    cpi = pd.Series(100.0 * 1.002 ** np.arange(len(cpi_dates)), index=cpi_dates)
    return QuadInputs(
        as_of=pd.Timestamp(as_of),
        gdp=pd.Series(100.0 * 1.005 ** np.arange(len(gdp_dates)), index=gdp_dates),
        cpi=cpi,
        core_cpi=cpi,
        nowcast_saar=pd.Series(dtype=float),
    )


def test_probabilities_follow_error_distribution() -> None:
    result = quad_probabilities_from_errors(0.5, -0.5, np.array([-1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    assert result["p_growth_up"] == 0.75
    assert result["p_inflation_up"] == 0.25
    assert np.isclose(sum(result["quads"].values()), 1.0)
    assert max(result["quads"], key=result["quads"].get) == "1"


def test_growth_and_inflation_overrides_change_the_projection() -> None:
    inputs = _inputs("2023-02-15")
    settings = QuadSettings(forward_quarters=1)
    base = {row["quarter"]: row for row in build_quad_outlook(inputs, settings)["quarters"]}
    hot = {
        row["quarter"]: row
        for row in build_quad_outlook(
            inputs,
            settings,
            growth_override={pd.Timestamp("2023-01-01"): 8.0},
            inflation_override={pd.Timestamp("2023-01-01"): 1.0},
            override_source={pd.Timestamp("2023-01-01"): "bridge"},
        )["quarters"]
    }
    assert hot["2023Q1"]["growth_change"] > base["2023Q1"]["growth_change"]
    assert hot["2023Q1"]["inflation_change"] > base["2023Q1"]["inflation_change"]
    assert hot["2023Q1"]["growth_assumption_source"] == "bridge"


def test_energy_path_carries_gasoline_moves_then_reverts_to_core() -> None:
    inputs = _inputs("2023-01-20")
    weeks = pd.date_range("2015-01-05", "2023-01-16", freq="W-MON")
    rng = np.random.default_rng(0)
    gas = pd.Series(3.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(weeks)))), index=weeks)
    gas.loc["2023-01-01":] *= 1.10  # a jump in the current month
    headline = inputs.cpi.copy()
    gas_monthly = gas.groupby(gas.index.to_period("M").start_time).mean()
    change = np.log(gas_monthly / gas_monthly.shift(1)).reindex(headline.index).fillna(0.0)
    headline = 100.0 * np.exp(np.cumsum(0.002 + 0.03 * change.to_numpy()))
    inputs = QuadInputs(
        as_of=inputs.as_of,
        gdp=inputs.gdp,
        cpi=pd.Series(headline, index=inputs.cpi.index),
        core_cpi=inputs.core_cpi,
        nowcast_saar=inputs.nowcast_saar,
    )
    context = ForecastContext(monthly_tables={}, lagged={"GASREGW": gas, "T10Y3M": pd.Series(dtype=float), "ICSA": pd.Series(dtype=float)})
    path = energy_inflation_path(context, inputs, base_mom=0.2)
    assert path[pd.Timestamp("2023-01-01")] > 0.2  # gasoline up this month lifts headline
    assert path[pd.Timestamp("2023-06-01")] == 0.2  # no gasoline information: core run-rate


def test_playbook_portfolio_is_dollar_neutral_and_signed() -> None:
    scores = pd.Series({"a": 1.0, "b": 1.0, "c": -1.0, "d": 0.0})
    volatility = pd.Series({"a": 10.0, "b": 20.0, "c": 10.0, "d": 10.0})
    forward = pd.Series({"a": 2.0, "b": 2.0, "c": -2.0, "d": 50.0})
    # Long legs earn +2 on half the book, the short leg earns +2 on the other half; d is ignored.
    assert np.isclose(portfolio_return(scores, volatility, forward), 2.0)
    assert np.isnan(portfolio_return(pd.Series({"a": 1.0}), volatility, forward))


def test_learned_playbook_only_uses_quarters_known_before_the_date() -> None:
    quarters = pd.date_range("2010-01-01", periods=12, freq="QS")
    in_quarter = pd.DataFrame({"x": 1.0, "y": 0.0, "z": -1.0}, index=quarters)
    known = {quarter: 4 for quarter in quarters}
    early = learned_scores(known, in_quarter, pd.Timestamp("2010-12-31"), minimum_quarters=6, leaders=1)
    late = learned_scores(known, in_quarter, pd.Timestamp("2013-01-31"), minimum_quarters=6, leaders=1)
    assert (early[4] == 0).all()
    assert late[4]["x"] == 1.0 and late[4]["z"] == -1.0


def test_textbook_scores_map_config_lists() -> None:
    scores = textbook_scores({"textbook": {4: {"long": ["bonds"], "short": ["stocks"]}}}, ["bonds", "stocks", "gold"])
    assert scores[4].to_dict() == {"bonds": 1.0, "stocks": -1.0, "gold": 0.0}
    assert (scores[1] == 0).all()
