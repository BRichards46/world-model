from __future__ import annotations

import pandas as pd

from world_model.states.forecast import (
    conditional_distribution,
    fit_component_rank_model,
    rank_ic,
    score_components,
)


def test_rank_ic_detects_positive_ranking() -> None:
    index = pd.bdate_range("2020-01-01", periods=100)
    feature = pd.Series(range(100), index=index, dtype=float)
    target = feature * 0.01

    assert rank_ic(feature, target) > 0.99


def test_fit_component_rank_model_selects_signed_components() -> None:
    index = pd.bdate_range("2020-01-01", periods=150)
    train = pd.DataFrame(
        {
            "trend_score": range(150),
            "momentum_score": list(reversed(range(150))),
            "breadth_score": 0.0,
            "volatility_score": 0.0,
            "rates_score": 0.0,
            "dollar_score": 0.0,
            "commodities_score": 0.0,
            "positioning_score": 0.0,
            "forward_return": range(150),
        },
        index=index,
    )

    model = fit_component_rank_model(train, min_abs_rank_ic=0.1, max_components=2)

    assert model.loc["trend_score", "weight"] > 0
    assert model.loc["momentum_score", "weight"] < 0


def test_score_components_uses_model_weights() -> None:
    features = pd.DataFrame(
        {
            "trend_score": [1.0],
            "momentum_score": [-1.0],
        }
    )
    model = pd.DataFrame(
        {
            "weight": [0.5, -0.5],
            "rank_ic": [0.5, -0.5],
            "quintile_spread": [0.1, -0.1],
        },
        index=["trend_score", "momentum_score"],
    )

    score = score_components(features, model)

    assert score.iloc[0] == 1.0


def test_conditional_distribution_uses_nearby_score_bucket() -> None:
    scores = pd.Series([-2, -1, 0, 1, 2] * 40, dtype=float)
    returns = pd.Series([-0.2, -0.1, 0.0, 0.1, 0.2] * 40, dtype=float)
    drawdowns = pd.Series([-0.25, -0.15, -0.05, -0.02, 0.0] * 40, dtype=float)

    distribution = conditional_distribution(scores, returns, drawdowns, current_score=2.0)

    assert distribution["mean_return"] > 0
    assert distribution["probability_positive"] == 1.0
