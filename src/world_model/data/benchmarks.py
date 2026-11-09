from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from world_model.data.norgate import read_norgate_index_series


DEFAULT_PRICE_SYMBOL = "$SPX"
DEFAULT_TOTAL_RETURN_SYMBOL = "$SPXTR"


def chain_price_total_return(price: pd.Series, total_return: pd.Series | None = None) -> pd.DataFrame:
    """Chain price-index log returns with total-return log returns wherever available."""
    price = pd.to_numeric(price, errors="coerce").dropna().sort_index()
    positive_price = price.where(price > 0.0)
    combined = np.log(positive_price / positive_price.shift(1)).rename("log_return")
    source = pd.Series("price_return", index=combined.index, dtype="object", name="return_source")

    if total_return is not None and not total_return.empty:
        total_return = pd.to_numeric(total_return, errors="coerce").dropna().sort_index()
        positive_total_return = total_return.where(total_return > 0.0)
        total_return_change = np.log(positive_total_return / positive_total_return.shift(1)).dropna()
        available = combined.index.intersection(total_return_change.index)
        combined.loc[available] = total_return_change.loc[available]
        source.loc[available] = "total_return"

    combined = combined.replace([np.inf, -np.inf], np.nan)
    normalized = 100.0 * np.exp(combined.fillna(0.0).cumsum())
    return pd.concat([combined, source, normalized.rename("normalized_index")], axis=1)


def load_equity_benchmark_history(
    data_roots: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Load the S&P 500 benchmark from Norgate: ``$SPX`` price history chained to ``$SPXTR``."""
    settings = config or {}
    metadata_root = data_roots["norgate_metadata"]
    price_symbol = str(settings.get("price_symbol", DEFAULT_PRICE_SYMBOL))
    total_return_symbol = settings.get("total_return_symbol", DEFAULT_TOTAL_RETURN_SYMBOL)
    price = read_norgate_index_series(metadata_root, price_symbol)
    total_return = (
        read_norgate_index_series(metadata_root, str(total_return_symbol)) if total_return_symbol else None
    )
    return chain_price_total_return(price, total_return)
