from __future__ import annotations

import pandas as pd

from world_model.states.market import summarize_forward_distribution


def test_summarize_forward_distribution() -> None:
    analogs = pd.DataFrame(
        {
            "probability_weight": [0.2, 0.3, 0.5],
            "one_month_return": [0.02, -0.01, 0.04],
            "one_month_max_drawdown": [-0.02, -0.08, -0.01],
        },
        index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
    )

    result = summarize_forward_distribution(analogs, {"one_month": 21})

    assert result.loc[0, "analog_count"] == 3
    assert result.loc[0, "probability_positive"] == 0.7
    assert round(result.loc[0, "weighted_mean_return"], 4) == 0.021
