import numpy as np
import pandas as pd
import pytest

from world_model.states.labor_quad import (
    extend_at_pace,
    fed_path_by_quad,
    growth_path,
    known_quads_from,
    quad_probabilities,
    summarize_walk_forward,
)
from world_model.states.quad_playbook_test import learned_scores


def _payrolls(end="2026-10-01", start="2020-01-01", growth=0.001):
    months = pd.date_range(start, end, freq="MS")
    return pd.Series(150_000 * np.exp(growth * np.arange(len(months))), index=months)


def test_extension_continues_the_recent_pace():
    known = _payrolls(end="2026-10-01")
    extended = extend_at_pace(known, pd.Timestamp("2026-12-01"))
    assert extended.index.max() == pd.Timestamp("2026-12-01")
    assert np.log(extended["2026-12-01"] / extended["2026-11-01"]) == pytest.approx(0.001)


def test_growth_path_marks_how_much_of_each_quarter_is_reported():
    known = _payrolls(end="2026-10-01")
    path = growth_path(known, pd.Timestamp("2027-06-01"))
    assert path.loc["2026-07-01", "source"] == "reported"
    assert path.loc["2026-10-01", "source"] == "partial"
    assert path.loc["2026-10-01", "months_reported"] == 1
    assert path.loc["2027-01-01", "source"] == "projected"
    # Constant growth: YoY flat, so the quad's growth leg neither accelerates nor slows.
    assert path.loc["2026-07-01", "growth_change"] == pytest.approx(0.0, abs=1e-9)


def test_growth_path_sees_acceleration():
    known = _payrolls(end="2026-09-01", growth=0.001)
    known.loc["2026-01-01":] *= np.exp(0.002 * np.arange(1, 10))  # payroll growth picks up in 2026
    path = growth_path(known, pd.Timestamp("2026-12-01"))
    assert path.loc["2026-04-01", "growth_change"] > 0


def test_quad_probabilities_combine_the_two_legs():
    odds = quad_probabilities(0.8, 0.25)
    assert sum(odds.values()) == pytest.approx(1.0)
    assert max(odds, key=odds.get) == "1"
    assert quad_probabilities(None, 0.5) is None


def test_fed_path_averages_rate_changes_after_each_quad():
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    quads = pd.Series(np.where(np.arange(120) % 24 < 12, 2, 4), index=dates)
    days = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    # Rates rise through Quad 2 stretches and fall through Quad 4 stretches.
    slope = np.where(((days.year - 2000) * 12 + days.month - 1) % 24 < 12, 0.01, -0.01)
    funds = pd.Series(np.cumsum(slope) / 30.0 + 3.0, index=days)
    result = fed_path_by_quad(quads, funds, months=3)
    assert result["by_quad"]["2"]["mean_change"] > result["by_quad"]["4"]["mean_change"]
    assert result["observations"] > 100


def test_summary_compares_with_the_gdp_quad_on_the_same_dates():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    labor = pd.DataFrame({
        "forecast_date": dates, "horizon": 0, "quarter_start": pd.Timestamp("2020-01-01"),
        "projected_quad": [1, 1, 2, 2], "realized_quad": [1, 1, 2, 3], "persistence_quad": [1, 2, 2, 2],
        "forecast_growth_change": [1, 1, 1, -1], "realized_growth_change": [1, 1, 1, -1],
        "forecast_inflation_change": [-1, -1, 1, 1], "realized_inflation_change": [-1, -1, 1, 1],
    })
    gdp = labor.assign(projected_quad=[4, 4, 4, 3])
    row = summarize_walk_forward(labor, gdp)["horizons"][0]
    assert row["quad_hit_rate"] == pytest.approx(0.75)
    assert row["gdp_quad_hit_rate"] == pytest.approx(0.25)


def test_known_quads_wait_for_the_quarter_and_the_lag():
    walk = pd.DataFrame({"quarter_start": pd.to_datetime(["2020-01-01", "2020-04-01"]), "realized_quad": [3, 4]})
    known = known_quads_from(walk, knowledge_lag_days=30)
    assert known(pd.Timestamp("2020-05-15")) == {pd.Timestamp("2020-01-01"): 3}
    assert len(known(pd.Timestamp("2020-08-15"))) == 2


def test_learned_playbook_handles_no_known_quarters():
    in_quarter = pd.DataFrame({"a": [0.1], "b": [-0.1]}, index=[pd.Timestamp("2020-01-01")])
    scores = learned_scores({}, in_quarter, pd.Timestamp("2020-06-30"), minimum_quarters=1, leaders=1)
    assert all((scores[q] == 0).all() for q in (1, 2, 3, 4))
