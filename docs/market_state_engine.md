# Market State Engine

The market state engine is the first modeling layer in this repository. It
does not own the raw data. It reads the existing local CSI futures and COT
datasets, transforms them into daily component scores, and stores a structured
market state vector.

## State Vector

Each date receives component scores from -1 to +1:

- `trend_score`: trailing trend across equity index futures
- `momentum_score`: risk-adjusted trailing momentum across equity index futures
- `breadth_score`: Norgate common-stock participation and new-high/new-low data
- `volatility_score`: realized ES volatility plus VX trend
- `rates_score`: Treasury futures trend, positive when Treasury prices are rising
- `dollar_score`: inverted dollar futures trend, positive when the dollar is easing
- `commodities_score`: crude and copper momentum
- `positioning_score`: COT large-spec positioning, positive when crowding is less hostile
- `composite_score`: weighted average of the component scores
- `regime_label`: coarse label derived from the composite and key qualifiers

The label is intentionally secondary. The component vector is the durable model
output because it can be compared across time and used for forward analogs.

## Forward Analogs

For a requested `as_of` date, the engine:

1. Finds the latest available market state on or before that date.
2. Compares that component vector with prior daily component vectors.
3. Keeps the closest historical analog dates with enough component coverage.
4. Computes subsequent ES returns over configured horizons.
5. Reports weighted mean, quantiles, probability of positive returns, and
   probability of drawdowns greater than 5% or 10%.

This is an empirical conditional distribution, not a forecast point estimate.
The important question is: what tended to happen after similar measured states?

## Walk-Forward Validation

The walk-forward validator tests whether the analog distribution had historical
value out of sample.

For each historical forecast date, it:

1. Uses that date's market state vector.
2. Restricts candidate analog dates to dates whose full forward outcomes would
   already have been known.
3. Builds the forward distribution from only those prior analogs.
4. Compares the forecast distribution with the realized future ES return.
5. Compares the result with unconditional-history and random-analog baselines.

Run the default month-end validation:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_walk_forward
```

Use weekly or daily validation when needed:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_walk_forward --frequency W-FRI
PYTHONPATH=src python3 -m world_model.cli.market_walk_forward --frequency B
```

Generated outputs are written to `outputs/`:

- `market_walk_forward_report.md`
- `market_walk_forward_forecasts.parquet`
- `market_walk_forward_summary.csv`
- `market_walk_forward_quintiles.csv`
- `market_walk_forward_calibration.csv`
- `market_walk_forward_drawdown_calibration.csv`

The most important first-pass check is the forecast quintile table. If high
forecast-return quintiles do not produce better realized returns than low
forecast-return quintiles, the state vector is not adding much forward signal.

## Forecast Ranker

The forecast ranker is a separate predictive layer. It should not be confused
with the descriptive state vector.

For each forecast date and horizon, it:

1. Builds the forward-return training sample that would have been known at that
   date.
2. Measures each component's prior rank correlation with that horizon's forward
   return.
3. Keeps the strongest components above a minimum absolute rank-IC threshold.
4. Allows signs to flip when a component has historically been contrarian.
5. Scores the current state with those signed weights.
6. Evaluates realized forward returns by forecast-score quintile.

Run it with:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_forecast
```

Rolling training windows can be tested with:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_forecast --train-window-years 5 --min-train-observations 500
PYTHONPATH=src python3 -m world_model.cli.market_forecast --train-window-years 10
```

The value bar is the same: high forecast-score quintiles should produce higher
realized returns than low forecast-score quintiles, and this should persist
across horizons and subperiods.

## Alternative Target Validation

The alternative target validator tests whether market state has more value for
risk or relative-return targets than for raw ES returns.

Run it with:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_targets
```

Current targets:

- ES 10% drawdown risk
- ES realized volatility expansion
- market-state deterioration
- ES relative to 10-year Treasury futures
- risk-on basket relative to defensive basket

For binary targets, the key tests are Brier score versus baseline and monotonic
event rates by forecast quintile. For continuous relative-return targets, the
key tests are Q5-Q1 spread, correlation, and RMSE versus baseline.

## Breadth Cache

Build the Norgate breadth cache with:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_breadth
```

The current universe definition is:

- `base_type == "Stock Market"`
- `subtype1 == "Equity"`
- `subtype2 == "Operating/Holding Company"`
- `domicile == "United States Of America"`
- `currency == "USD"`
- exchange in `NYSE`, `Nasdaq`, or `NYSE American`

This includes active and delisted names. Daily breadth counts use the
`major_exchange_listed` column from each price file, so a security only counts
on dates when it was major-exchange listed.

Current breadth outputs:

- percent above 50/100/200-day moving averages
- percent making 52-week highs/lows
- net 52-week high-low share
- percent advancing/declining
- 10-day average advancing share

## Current Limitations

- Breadth is broad common-stock breadth, not index-constituent breadth.
- Credit spread and cash Treasury yield inputs are not yet included.
- Component weights and thresholds are deliberately simple starting values.
- The analog model uses distance over current component scores, not a trained
  predictive model.
- The walk-forward validator currently evaluates ES as the forward target.
- The current analog forecast still does not beat baseline in full-sample
  walk-forward validation.

Those are the next areas to improve after validating the first state history.
