from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.macro_regime_forecast import (
    _fit_predict_logistic,
    _roc_auc,
    summarize_macro_regime_forecasts,
)


def test_logistic_probability_and_auc() -> None:
    x = pd.DataFrame({"signal": np.linspace(-2.0, 2.0, 40)})
    y = x["signal"].gt(0.0).astype(float)
    low = _fit_predict_logistic(x, y, pd.Series({"signal": -1.0}), 1.0)
    high = _fit_predict_logistic(x, y, pd.Series({"signal": 1.0}), 1.0)
    assert 0.0 < low < high < 1.0
    assert _roc_auc(y, x["signal"]) == 1.0


def test_summary_requires_both_halves_to_improve() -> None:
    dates = pd.date_range("2020-01-31", periods=80, freq="ME")
    realized = pd.Series(([0.0, 1.0] * 40), dtype=float)
    probability = realized.mul(0.8).add(0.1)
    forecasts = pd.DataFrame(
        {
            "forecast_date": dates,
            "target_id": "risk",
            "target_label": "Risk",
            "horizon": "one_month",
            "trading_days": 21,
            "probability": probability,
            "baseline_probability": 0.5,
            "realized": realized,
        }
    )
    summary = summarize_macro_regime_forecasts(
        forecasts, {"minimum_oos_observations": 60}
    )
    assert bool(summary.iloc[0]["research_candidate"])
