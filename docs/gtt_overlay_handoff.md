# Growth-Trend Timing Overlay Handoff

This note summarizes the Growth-Trend Timing work and provides portable code for
another project that already runs a monthly 10-month SMA strategy on ETFs.

The basic idea is:

```text
Existing rule:
  Long if price is above its 10-month SMA, otherwise defensive.

GTT overlay:
  Long if the macro filter is healthy.
  If the macro filter is not healthy, defer to the 10-month SMA rule.

Equivalent:
  exposure = 1.0 when macro_ok is true
  exposure = trend_signal when macro_ok is false
```

The overlay is not a return forecast. It is a whipsaw-reduction rule. The macro
filter tells the strategy when a trend break should be respected versus ignored.

## Tested Setup

Data:

- ETF prices: Norgate `adjustment=totalreturn` close.
- Macro data: local FRED point-in-time vintage files.
- Defensive asset: cash.
- Transaction cost: `5 bps` one-way on exposure changes.
- Signal timing: signal observed at close, applied to the next return.
- Trend rule: monthly close versus 10-month SMA.

Important implementation detail:

The current test used a prior 10-month average:

```text
month_end_close > rolling_10_month_average.shift(1)
```

If the production strategy uses the current month in the SMA, retest with that
same convention before changing live logic.

## 10-Month SMA ETF Findings

The constrained ETF test used only the monthly 10-month SMA as the trend rule.

| ETF | Best GTT Macro Filter | Buy & Hold CAGR | 10M Trend CAGR | GTT CAGR | Buy & Hold DD | 10M Trend DD | GTT DD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SPY` | Unemployment downtrend | 9.44% | 7.94% | 11.26% | -55.19% | -36.42% | -33.72% |
| `QQQ` | Retail/IP positive | 12.76% | 10.89% | 15.39% | -56.11% | -28.56% | -33.93% |
| `EFA` | Retail/IP positive | 7.88% | 6.81% | 9.14% | -61.04% | -36.89% | -34.19% |
| `GLD` | Activity negative | 10.01% | 9.76% | 11.34% | -45.56% | -27.69% | -29.41% |
| `DBC` | Activity negative | 1.54% | 3.27% | 4.24% | -76.36% | -50.67% | -62.49% |
| `TLT` | Retail/IP positive | 3.46% | 1.40% | 5.04% | -48.35% | -27.37% | -36.43% |
| `MBB` | Growth weak or disinflation | 2.50% | 1.82% | 2.68% | -17.64% | -9.95% | -12.82% |
| `MUB` | Growth weak or disinflation | 3.17% | 1.93% | 3.13% | -13.68% | -13.68% | -13.68% |
| `VNQ` | Activity positive | 6.31% | 1.25% | 4.38% | -73.07% | -63.09% | -63.09% |

Recommended interpretation:

- Best candidates: `SPY`, `QQQ`, `EFA`, `GLD`.
- Interesting but tradeoff-heavy: `DBC`, `TLT`.
- Marginal: `MBB`.
- Do not prioritize: `MUB`, `VNQ`.

For a production overlay, start with:

```python
ETF_GTT_FILTER = {
    "SPY": "unemployment_downtrend",
    "QQQ": "retail_ip_positive",
    "EFA": "retail_ip_positive",
    "GLD": "activity_negative",
    "DBC": "activity_negative",
    "TLT": "retail_ip_positive",
    "MBB": "growth_weak_or_disinflation",
    # MUB and VNQ did not justify an overlay in the first pass.
    "MUB": None,
    "VNQ": None,
}
```

## Existing Project Integration

If the other project already computes a 10-month SMA trend signal, only one new
step is needed:

```python
import numpy as np


def overlay_gtt(existing_trend_signal, macro_ok):
    """
    existing_trend_signal: Series of 1.0 or 0.0 from the existing 10M SMA rule.
    macro_ok: Series of booleans or 1.0/0.0 indexed by date.

    Returns 1.0 when macro is healthy, otherwise the existing trend signal.
    """
    trend = existing_trend_signal.astype(float)
    macro = macro_ok.reindex(trend.index).ffill().astype(float)
    exposure = trend.copy()
    valid = trend.notna() & macro.notna()
    exposure.loc[valid] = np.where(macro.loc[valid].eq(1.0), 1.0, trend.loc[valid])
    return exposure
```

The macro signal must be lag-clean. Do not use final revised FRED values as if
they were known historically. Use point-in-time vintages where possible.

## FRED Point-In-Time Data

The local FRED store used in this project lives at:

```text
/Users/brichards/Documents/data/fred/vintage/{series_id}.parquet
```

Expected columns:

```text
series_id
observation_date
value
realtime_start
realtime_end
```

The workflow:

1. Read each vintage file.
2. Walk forward by `realtime_start`.
3. Maintain the latest known observation values as of that release date.
4. Compute the desired transform from the known data only.
5. Reindex the released signal to the ETF trading calendar and forward-fill.

The minimal series needed for the recommended filters:

| Filter | Series |
| --- | --- |
| `retail_ip_positive` | `RRSFS`, `INDPRO` |
| `unemployment_downtrend` | `UNRATE` |
| `activity_negative` | `RRSFS`, `INDPRO`, plus optional `PAYEMS`, `RPI`, `CFNAI` |
| `growth_weak_or_disinflation` | activity proxy plus `CPIAUCSL`, `CPILFESL` |

## Portable Code

The code below can be copied into the other project as `gtt_overlay.py`. It is
deliberately written without imports from this repository.

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ETF_GTT_FILTER = {
    "SPY": "unemployment_downtrend",
    "QQQ": "retail_ip_positive",
    "EFA": "retail_ip_positive",
    "GLD": "activity_negative",
    "DBC": "activity_negative",
    "TLT": "retail_ip_positive",
    "MBB": "growth_weak_or_disinflation",
    "MUB": None,
    "VNQ": None,
}


def load_fred_vintage(fred_root: str | Path, series_id: str) -> pd.DataFrame:
    path = Path(fred_root).expanduser() / "vintage" / f"{series_id}.parquet"
    df = pd.read_parquet(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values(["realtime_start", "observation_date"])


def realtime_indicator(
    fred_root: str | Path,
    series_id: str,
    transform: str,
    period_months: int = 12,
    max_prior_gap_days: int | None = None,
) -> pd.Series:
    vintage = load_fred_vintage(fred_root, series_id)
    known_values: dict[pd.Timestamp, float] = {}
    rows: list[tuple[pd.Timestamp, float]] = []

    for release_date, release_rows in vintage.groupby("realtime_start", sort=True):
        for row in release_rows.itertuples(index=False):
            known_values[pd.Timestamp(row.observation_date)] = float(row.value) if pd.notna(row.value) else np.nan

        known = pd.Series(known_values, dtype=float).sort_index().dropna()
        rows.append((pd.Timestamp(release_date), _compute_metric(known, transform, period_months, max_prior_gap_days)))

    if not rows:
        return pd.Series(dtype=float)
    index, values = zip(*rows)
    return pd.Series(values, index=pd.DatetimeIndex(index), name=series_id).sort_index()


def _compute_metric(
    known: pd.Series,
    transform: str,
    period_months: int,
    max_prior_gap_days: int | None,
) -> float:
    if known.empty:
        return np.nan
    if transform == "level":
        return float(known.iloc[-1])
    if transform == "yoy_pct":
        latest_date = pd.Timestamp(known.index[-1])
        prior = _prior_value(known, latest_date, period_months, max_prior_gap_days)
        if pd.isna(prior) or prior == 0.0:
            return np.nan
        return float(known.iloc[-1]) / float(prior) - 1.0
    raise ValueError(f"Unsupported transform: {transform}")


def _prior_value(
    known: pd.Series,
    latest_date: pd.Timestamp,
    period_months: int,
    max_prior_gap_days: int | None,
) -> float:
    prior_date = latest_date - pd.DateOffset(months=period_months)
    if max_prior_gap_days is None:
        max_prior_gap_days = 45 if period_months >= 12 else max(45, int(period_months * 31 * 0.75))
    candidates = known.loc[known.index <= prior_date]
    if candidates.empty:
        return np.nan
    candidate_date = pd.Timestamp(candidates.index[-1])
    if (prior_date - candidate_date).days > max_prior_gap_days:
        return np.nan
    return float(candidates.iloc[-1])


def rolling_z_score(value: pd.Series, window: int = 120, min_periods: int = 36) -> pd.Series:
    baseline = value.astype(float).rolling(window=window, min_periods=min_periods)
    mean = baseline.mean().shift(1)
    std = baseline.std(ddof=0).shift(1)
    return (value - mean) / std.replace(0.0, np.nan)


def squashed_score(value: pd.Series, invert: bool = False) -> pd.Series:
    score = np.tanh(rolling_z_score(value) / 2.0)
    score = pd.Series(score, index=value.index)
    return -score if invert else score


def build_macro_signals(fred_root: str | Path, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    index = pd.DatetimeIndex(trading_index).sort_values()
    raw = pd.DataFrame(index=index)

    raw["real_retail_sales_yoy"] = realtime_indicator(fred_root, "RRSFS", "yoy_pct").reindex(index).ffill()
    raw["industrial_production_yoy"] = realtime_indicator(fred_root, "INDPRO", "yoy_pct").reindex(index).ffill()
    raw["unemployment_rate"] = realtime_indicator(fred_root, "UNRATE", "level").reindex(index).ffill()

    optional_inputs = {}
    for name, series_id, transform in [
        ("payrolls_yoy", "PAYEMS", "yoy_pct"),
        ("real_income_yoy", "RPI", "yoy_pct"),
        ("cfnai_level", "CFNAI", "level"),
        ("headline_cpi_yoy", "CPIAUCSL", "yoy_pct"),
        ("core_cpi_yoy", "CPILFESL", "yoy_pct"),
    ]:
        try:
            optional_inputs[name] = realtime_indicator(fred_root, series_id, transform).reindex(index).ffill()
        except FileNotFoundError:
            optional_inputs[name] = pd.Series(np.nan, index=index)

    raw = raw.assign(**optional_inputs)

    signals = pd.DataFrame(index=index)
    signals["retail_ip_positive"] = (
        raw["real_retail_sales_yoy"].gt(0.0) & raw["industrial_production_yoy"].gt(0.0)
    ).astype(float).where(raw[["real_retail_sales_yoy", "industrial_production_yoy"]].notna().all(axis=1))

    unrate_trend = raw["unemployment_rate"].rolling(252, min_periods=126).mean()
    signals["unemployment_downtrend"] = raw["unemployment_rate"].le(unrate_trend).astype(float).where(
        raw["unemployment_rate"].notna() & unrate_trend.notna()
    )

    activity_parts = pd.concat(
        [
            squashed_score(raw["real_retail_sales_yoy"]),
            squashed_score(raw["industrial_production_yoy"]),
            squashed_score(raw["payrolls_yoy"]),
            squashed_score(raw["real_income_yoy"]),
            squashed_score(raw["cfnai_level"]),
        ],
        axis=1,
    )
    activity_score = activity_parts.mean(axis=1)
    signals["activity_positive"] = activity_score.ge(0.0).astype(float).where(activity_score.notna())
    signals["activity_negative"] = activity_score.le(0.0).astype(float).where(activity_score.notna())

    inflation_parts = pd.concat(
        [
            squashed_score(raw["headline_cpi_yoy"]),
            squashed_score(raw["core_cpi_yoy"]),
        ],
        axis=1,
    )
    inflation_pressure = inflation_parts.mean(axis=1)
    inflation_low = inflation_pressure.le(0.0).astype(float).where(inflation_pressure.notna())
    signals["growth_weak_or_disinflation"] = (
        signals["activity_negative"].eq(1.0) | inflation_low.eq(1.0)
    ).astype(float).where(signals["activity_negative"].notna() | inflation_low.notna())

    return signals


def monthly_10m_sma_signal(
    total_return_close: pd.Series,
    include_current_month_in_sma: bool = False,
) -> pd.Series:
    price = total_return_close.dropna().sort_index()
    month_close = price.resample("ME").last()
    sma = month_close.rolling(10, min_periods=10).mean()
    if not include_current_month_in_sma:
        sma = sma.shift(1)
    signal = month_close.gt(sma).astype(float).where(sma.notna())
    return signal.reindex(price.index).ffill().rename("trend_10m")


def gtt_overlay_exposure(
    total_return_close: pd.Series,
    macro_signals: pd.DataFrame,
    macro_filter: str | None,
    existing_trend_signal: pd.Series | None = None,
) -> pd.Series:
    if existing_trend_signal is None:
        trend = monthly_10m_sma_signal(total_return_close)
    else:
        trend = existing_trend_signal.astype(float).reindex(total_return_close.index).ffill()

    if macro_filter is None:
        return trend.rename("exposure")

    macro_ok = macro_signals[macro_filter].reindex(total_return_close.index).ffill().astype(float)
    exposure = trend.copy()
    valid = trend.notna() & macro_ok.notna()
    exposure.loc[valid] = np.where(macro_ok.loc[valid].eq(1.0), 1.0, trend.loc[valid])
    return exposure.rename("exposure")


def strategy_returns(
    total_return_close: pd.Series,
    exposure: pd.Series,
    one_way_cost: float = 0.0005,
) -> pd.Series:
    returns = total_return_close.pct_change()
    exposure = exposure.astype(float).reindex(returns.index).ffill()
    next_day_exposure = exposure.shift(1)
    turnover = exposure.diff().abs().fillna(0.0)
    return (next_day_exposure * returns - turnover * one_way_cost).where(next_day_exposure.notna())


def apply_gtt_to_etf(
    symbol: str,
    total_return_close: pd.Series,
    fred_root: str | Path,
    existing_trend_signal: pd.Series | None = None,
) -> pd.DataFrame:
    macro_filter = ETF_GTT_FILTER.get(symbol)
    macro = build_macro_signals(fred_root, total_return_close.index)
    exposure = gtt_overlay_exposure(
        total_return_close=total_return_close,
        macro_signals=macro,
        macro_filter=macro_filter,
        existing_trend_signal=existing_trend_signal,
    )
    out = pd.DataFrame(index=total_return_close.index)
    out["close"] = total_return_close
    out["macro_filter"] = macro_filter
    out["trend_10m"] = monthly_10m_sma_signal(total_return_close) if existing_trend_signal is None else existing_trend_signal
    out["gtt_exposure"] = exposure
    out["gtt_return"] = strategy_returns(total_return_close, exposure)
    out["gtt_equity"] = (1.0 + out["gtt_return"].fillna(0.0)).cumprod()
    return out
```

## Using The Code

Example integration when the target project already has total-return close
series and an existing 10M SMA signal:

```python
from gtt_overlay import apply_gtt_to_etf

FRED_ROOT = "/Users/brichards/Documents/data/fred"

# close is the ETF total-return adjusted close.
# existing_signal is your current 10M SMA signal, indexed by date.
overlay = apply_gtt_to_etf(
    symbol="QQQ",
    total_return_close=qqq_total_return_close,
    fred_root=FRED_ROOT,
    existing_trend_signal=qqq_10m_sma_signal,
)

qqq_gtt_exposure = overlay["gtt_exposure"]
```

If the existing project trades only at month-end, sample the GTT exposure on the
same rebalance dates used by the current strategy and apply it to the next
holding period. Do not compare a daily-applied GTT signal to a monthly-applied
baseline without making the execution assumptions identical.

## Exact Reproduction From This Repository

The implementation used for these tests lives here:

```text
src/world_model/states/growth_trend.py
config/growth_trend_timing.yml
outputs/growth_trend_etfs/etf_10m_sma_comparison.csv
outputs/growth_trend_etfs/etf_10m_sma_top_filters.csv
```

The exact ETF 10M SMA comparison file is:

```text
outputs/growth_trend_etfs/etf_10m_sma_comparison.csv
```

To rerun the broader ETF grid from this project, use the existing
`GrowthTrendTimingEngine` patterns and Norgate total-return ETF files under:

```text
/Users/brichards/Documents/data/equity/norgate/prices/adjustment=totalreturn
```

## Cautions

- The ETF tests selected the best macro filter in-sample. Treat the mapping as
  a candidate, not proof.
- The strongest and most intuitive results are `SPY`, `QQQ`, and `EFA`.
- `GLD` worked with an activity-negative filter, which is plausible but should
  be retested separately.
- `DBC` and `TLT` improved return but gave up some of the drawdown benefit of
  trend-only.
- `MUB` and `VNQ` did not justify an overlay in the first pass.
- The activity and inflation proxies in the portable code are simplified. For
  exact reproduction, use the `world_model` economic state engine.
