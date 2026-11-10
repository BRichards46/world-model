from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.features.trigger_odds import (
    first_print_changes,
    frequency_at_least,
    frequency_at_most,
    market_break_odds,
)


def _vintage(rows: list[tuple[str, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime([row[0] for row in rows]),
            "value": [row[1] for row in rows],
            "realtime_start": pd.to_datetime([row[2] for row in rows]),
            "realtime_end": pd.to_datetime([row[3] for row in rows]),
        }
    )


def test_first_print_changes_compare_within_one_release() -> None:
    # March's first release also rebases history (x2). Its change must be measured against February as
    # published in that same release, not February's first print.
    end = "2099-12-31"
    rows = [
        ("2020-01-01", 100.0, "2020-02-05", "2020-04-04"),
        ("2020-02-01", 101.0, "2020-03-05", "2020-04-04"),
        ("2020-01-01", 200.0, "2020-04-05", end),
        ("2020-02-01", 202.0, "2020-04-05", end),
        ("2020-03-01", 199.98, "2020-04-05", end),
    ]
    changes = first_print_changes(_vintage(rows), start="2020-02-01", percent=True, average_months=2).set_index("observation_date")
    assert np.isclose(changes.loc["2020-02-01", "change"], 1.0)
    assert np.isclose(changes.loc["2020-03-01", "change"], -1.0)
    assert np.isclose(changes.loc["2020-03-01", "prior_gap"], 1.0)  # 202 vs the 201 average of Jan-Feb


def test_frequencies_count_hits_and_months() -> None:
    values = pd.Series([0.0, 0.1, 0.3, -0.2, np.nan])
    assert frequency_at_least(values, 0.25) == {"probability": 0.25, "hits": 1, "months": 4}
    assert frequency_at_most(values, -0.1)["hits"] == 1
    assert frequency_at_least(pd.Series(dtype=float), 0.1) is None


def test_market_break_odds_rise_with_a_closer_threshold() -> None:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("1995-01-02", "2026-11-10")
    levels = pd.Series(np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(days)))), index=days)
    near = market_break_odds(levels, -1.0, pd.Timestamp("2026-11-10"))
    far = market_break_odds(levels, -6.0, pd.Timestamp("2026-11-10"))
    assert near["trading_days_left"] == 14  # Nov 11-30, 2026 (holidays are not excluded)
    assert 0.0 < far["probability"] < near["probability"] < 0.5
    assert 12.0 < near["annualized_volatility"] < 20.0
    month_done = market_break_odds(levels.loc[:"2026-10-30"], -1.0, pd.Timestamp("2026-10-31"))
    assert month_done is None
