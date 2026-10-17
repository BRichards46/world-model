# DX Growth-Trend Timing Handoff

This is the tested GTT rule for `DX` / U.S. Dollar Index futures.

## Selected Rule

Best tested setup:

```text
asset: DX
macro filter: inflation_high
trend rule: monthly_12m
model type: long/short GTT
transaction cost: 5 bps one-way
signal timing: signal known at close, applied to next return
```

Grid result, `1996-02-01` to `2026-06-17`:

| CAGR | Vol | Sharpe-like | Max DD | Long Share | Short Share | Trades |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.60% | 8.10% | 0.44 | -20.23% | 74.92% | 25.08% | 32 |

## Logic

DX performed best when the dollar was allowed to stay long during high-inflation
macro states, otherwise it deferred to a slow price trend filter.

```text
trend_up = DX month-end close > 12-month SMA
macro_ok = inflation pressure is high

if macro_ok or trend_up:
    exposure = +1
else:
    exposure = -1
```

In words: be long DX when inflation pressure is high or DX is in an uptrend.
Flip short only when inflation pressure is not high and DX is below its
12-month trend.

## Macro Filter

`inflation_high` uses point-in-time FRED data:

- `CPIAUCSL`: headline CPI
- `CPILFESL`: core CPI

For each series:

1. Compute YoY percent change using data known as of each FRED vintage release.
2. Compute a rolling z-score with a 120-observation window and 36-observation
   minimum.
3. Shift the rolling mean/std by one observation so the current value is not in
   its own baseline.
4. Squash the z-score with `tanh(z / 2)`.
5. Average headline and core scores into `inflation_pressure`.
6. `inflation_high = inflation_pressure >= 0`.

## Portable Code Sketch

```python
import numpy as np
import pandas as pd


def rolling_z_score(value: pd.Series, window: int = 120, min_periods: int = 36) -> pd.Series:
    value = value.astype(float)
    baseline = value.rolling(window=window, min_periods=min_periods)
    mean = baseline.mean().shift(1)
    std = baseline.std(ddof=0).shift(1)
    return (value - mean) / std.replace(0.0, np.nan)


def squashed_score(value: pd.Series) -> pd.Series:
    return np.tanh(rolling_z_score(value) / 2.0)


def inflation_high_signal(headline_cpi_yoy: pd.Series, core_cpi_yoy: pd.Series) -> pd.Series:
    inflation_pressure = pd.concat(
        [squashed_score(headline_cpi_yoy), squashed_score(core_cpi_yoy)],
        axis=1,
    ).mean(axis=1)
    return inflation_pressure.ge(0.0).astype(float).where(inflation_pressure.notna())


def monthly_12m_trend_signal(close: pd.Series) -> pd.Series:
    price = close.dropna().sort_index()
    month_close = price.resample("ME").last()
    sma = month_close.rolling(12, min_periods=12).mean()
    trend = month_close.gt(sma).astype(float).where(sma.notna())
    return trend.reindex(price.index).ffill()


def dx_gtt_exposure(dx_close: pd.Series, inflation_high: pd.Series) -> pd.Series:
    trend_up = monthly_12m_trend_signal(dx_close)
    macro_ok = inflation_high.reindex(trend_up.index).ffill()

    exposure = pd.Series(np.nan, index=trend_up.index, dtype=float)
    valid = trend_up.notna() & macro_ok.notna()
    exposure.loc[valid] = np.where(
        macro_ok.loc[valid].eq(1.0) | trend_up.loc[valid].eq(1.0),
        1.0,
        -1.0,
    )
    return exposure.rename("DX_GTT_exposure")


def strategy_returns(dx_close: pd.Series, exposure: pd.Series, one_way_cost: float = 0.0005) -> pd.Series:
    returns = dx_close.pct_change()
    exposure = exposure.reindex(returns.index).ffill()
    effective_exposure = exposure.shift(1)
    turnover = exposure.diff().abs().fillna(0.0)
    return (effective_exposure * returns - turnover * one_way_cost).where(effective_exposure.notna())
```

Use point-in-time FRED vintages for any historical test. Final revised CPI data
is acceptable for a quick implementation check, but not for a clean backtest.
