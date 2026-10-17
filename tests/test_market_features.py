from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.features.market import (
    cot_crowding_score,
    momentum_score,
    trend_score,
    volatility_score,
)


def test_market_scores_are_bounded() -> None:
    index = pd.bdate_range("2020-01-01", periods=500)
    close = pd.Series(np.linspace(100, 150, len(index)), index=index)

    for score in [trend_score(close), momentum_score(close), volatility_score(close)]:
        clean = score.dropna()
        assert not clean.empty
        assert clean.between(-1, 1).all()


def test_cot_crowding_high_long_scores_negative() -> None:
    index = pd.bdate_range("2020-01-01", periods=400)
    cot = pd.Series(np.linspace(-0.2, 0.4, len(index)), index=index)

    score = cot_crowding_score(cot)

    assert score.dropna().iloc[-1] < 0
