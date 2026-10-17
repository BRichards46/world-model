from __future__ import annotations

import pandas as pd

from world_model.data.norgate import select_common_stock_universe
from world_model.features.market import breadth_score
from world_model.states.market import select_diverse_analogs


def test_select_common_stock_universe_filters_non_common_assets() -> None:
    metadata = pd.DataFrame(
        [
            {
                "assetid": 1,
                "symbol": "AAA",
                "base_type": "Stock Market",
                "subtype1": "Equity",
                "subtype2": "Operating/Holding Company",
                "domicile": "United States Of America",
                "currency": "USD",
                "exchange_name": "NYSE",
            },
            {
                "assetid": 2,
                "symbol": "ETF",
                "base_type": "Stock Market",
                "subtype1": "Exchange Traded Product",
                "subtype2": "Exchange Traded Fund (ETF)",
                "domicile": "United States Of America",
                "currency": "USD",
                "exchange_name": "NYSE Arca",
            },
            {
                "assetid": 3,
                "symbol": "OTC",
                "base_type": "Stock Market",
                "subtype1": "Equity",
                "subtype2": "Operating/Holding Company",
                "domicile": "United States Of America",
                "currency": "USD",
                "exchange_name": "OTC",
            },
        ]
    )

    selected = select_common_stock_universe(metadata)

    assert selected["symbol"].tolist() == ["AAA"]


def test_breadth_score_is_bounded() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    breadth = pd.DataFrame(
        {
            "pct_above_50d": [0.2, 0.4, 0.5, 0.6, 0.8],
            "pct_above_200d": [0.2, 0.4, 0.5, 0.6, 0.8],
            "net_new_52w_high_low": [-0.2, -0.05, 0.0, 0.05, 0.2],
            "pct_advancing_10d": [0.3, 0.45, 0.5, 0.55, 0.7],
        },
        index=index,
    )

    score = breadth_score(breadth)

    assert score.between(-1, 1).all()
    assert score.iloc[-1] > score.iloc[0]


def test_select_diverse_analogs_enforces_gap() -> None:
    index = pd.bdate_range("2020-01-01", periods=10)
    distances = pd.Series(range(10), index=index, dtype=float)

    selected = select_diverse_analogs(distances, top_n=3, min_trading_days=3)

    assert len(selected) == 3
    selected_positions = [index.get_loc(date) for date in selected.index]
    assert selected_positions == [0, 3, 6]
