"""GTT on assets beyond the S&P 500: Russell 2000, Treasuries and the dollar.

Macro logic is fixed per asset class in config before any results are seen.
Danger flags come from the point-in-time values built by the threshold study,
and the scoring reuses the GTT risk test and the monthly timing backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from world_model.states.gtt_threshold_validation import Threshold, combine_or


def synthetic_treasury_log_returns(yields_pct: pd.Series, maturity_years: float = 10.0) -> pd.Series:
    """Daily log total return of a par bond repriced at each day's yield (semiannual coupons).

    Each day buys a new par bond at the prior yield, earns that yield for the elapsed calendar days, and
    is repriced at today's yield with the remaining maturity treated as unchanged (the usual
    constant-maturity approximation).
    """
    y = pd.to_numeric(yields_pct, errors="coerce").dropna().sort_index() / 100.0
    prior = y.shift(1)
    periods = 2.0 * maturity_years
    coupon = prior / 2.0
    rate = y / 2.0
    discount = (1.0 + rate) ** (-periods)
    price = np.where(rate.abs() > 1e-12, coupon / rate * (1.0 - discount) + discount, 1.0 + coupon * periods)
    days = pd.Series(y.index, index=y.index).diff().dt.days
    carry = prior * days / 365.0
    simple = pd.Series(price, index=y.index) - 1.0 + carry
    return np.log1p(simple).iloc[1:].rename("log_return")


def cash_yield(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Annualized % cash yield: ``primary`` where available, otherwise ``fallback``."""
    primary = pd.to_numeric(primary, errors="coerce").dropna().sort_index()
    fallback = pd.to_numeric(fallback, errors="coerce").dropna().sort_index()
    combined = fallback.copy()
    combined.loc[primary.index] = primary
    return combined.sort_index()


def build_logic_filters(values: pd.DataFrame) -> pd.DataFrame:
    """Danger flags for every building block named in the cross-asset config."""
    indpro_weak = Threshold("indpro_yoy", "<=", 0.0).flag(values)
    unemployment_weak = Threshold("unemployment_gap", ">", 0.1).flag(values)
    real_retail_weak = Threshold("real_retail_yoy", "<=", 0.0).flag(values)
    growth_weak = combine_or([indpro_weak, unemployment_weak])
    growth_healthy = (1.0 - growth_weak).where(growth_weak.notna())
    cpi_above_2_5 = Threshold("cpi_yoy", ">=", 2.5).flag(values)
    cpi_above_4 = Threshold("cpi_yoy", ">=", 4.0).flag(values)
    cpi_below_2_5 = (1.0 - cpi_above_2_5).where(cpi_above_2_5.notna())
    both = pd.concat([growth_healthy, cpi_above_2_5], axis=1)
    healthy_and_hot = both.eq(1.0).all(axis=1).astype(float).where(both.notna().all(axis=1))
    return pd.DataFrame(
        {
            "growth_weak": growth_weak,
            "growth_healthy": growth_healthy,
            "indpro_weak": indpro_weak,
            "unemployment_weak": unemployment_weak,
            "indpro_or_real_retail_weak": combine_or([indpro_weak, real_retail_weak]),
            "cpi_above_2_5": cpi_above_2_5,
            "cpi_above_4": cpi_above_4,
            "cpi_below_2_5": cpi_below_2_5,
            "growth_healthy_and_cpi_above_2_5": healthy_and_hot,
        }
    )
