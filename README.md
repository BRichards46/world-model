# world-model

Research layer for building a holistic state model of global capital markets.

This repository is intended to sit on top of the existing local data lake:

- CSI continuous futures and COT data
- Norgate equity data
- FRED/ALFRED point-in-time economic data

The first implemented module is the market state engine. It produces daily
structured state vectors, regime labels, and empirical forward-return
distributions based on closest historical analogs.

The second implemented module is the economic state engine. It consumes the
point-in-time FRED/ALFRED vintage store and produces a structured macro state
vector covering growth, labor, inflation, liquidity, credit, and housing.

## Market State MVP

Run the current market state snapshot:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_state
```

By default this writes a markdown report to:

```text
outputs/market_state_report.md
```

Run for a specific date and write artifacts:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_state \
  --as-of 2026-06-18 \
  --history-out outputs/market_state_history.parquet \
  --analogs-out outputs/market_state_analogs.parquet \
  --distribution-out outputs/market_state_distribution.csv \
  --report-out outputs/market_state_report.md
```

The engine currently uses CSI futures and COT files configured in:

```text
config/data_roots.yml
config/market_state.yml
```

Create the machine-local data-root configuration from the committed example:

```bash
cp config/data_roots.example.yml config/data_roots.yml
```

Then edit `config/data_roots.yml` to point to the data directories on your
machine. The local file is ignored by Git.

The state vector includes:

- trend
- momentum
- breadth
- volatility
- rates
- dollar
- commodities
- positioning
- composite score
- regime label

The analog layer compares the current component vector to prior daily vectors,
then summarizes subsequent ES returns over 1m, 3m, 6m, and 12m horizons.

## Norgate Breadth

Build the breadth cache from the local Norgate price mirror:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_breadth
```

This writes:

```text
outputs/norgate_market_breadth.parquet
```

The current breadth universe is U.S.-domiciled USD common operating companies
on NYSE, Nasdaq, and NYSE American. It includes active and delisted names and
uses each asset's `major_exchange_listed` time series when counting daily
eligibility.

## Walk-Forward Validation

Run the default month-end validation:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_walk_forward
```

This writes:

```text
outputs/market_walk_forward_report.md
outputs/market_walk_forward_forecasts.parquet
outputs/market_walk_forward_summary.csv
outputs/market_walk_forward_quintiles.csv
outputs/market_walk_forward_calibration.csv
outputs/market_walk_forward_drawdown_calibration.csv
```

Use `--frequency W-FRI` for weekly validation or `--frequency B` for daily
validation.

Current note: the first breadth-plus-analog model should be treated as
experimental. The walk-forward report is the arbiter of whether the forward
distribution is adding value versus baseline.

## Forecast Ranker

Run the horizon-specific component ranker:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_forecast
```

This is separate from the analog model. For each forecast date and horizon, it
learns signed component weights from prior component/return relationships only,
then tests whether the resulting score ranks future ES returns.

Use a rolling training window to test whether stale relationships are hurting
the expanding-window model:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_forecast --train-window-years 5 --min-train-observations 500
PYTHONPATH=src python3 -m world_model.cli.market_forecast --train-window-years 10
```

Outputs:

```text
outputs/market_forecast_ranker_report.md
outputs/market_forecast_ranker_forecasts.parquet
outputs/market_forecast_ranker_summary.csv
outputs/market_forecast_ranker_quintiles.csv
outputs/market_component_diagnostics.csv
outputs/market_forecast_ranker_weights.parquet
```

## Alternative Target Validation

Test whether market state is more useful for targets other than raw ES returns:

```bash
PYTHONPATH=src python3 -m world_model.cli.market_targets
```

Targets currently include:

- ES 10% drawdown risk
- ES realized volatility expansion
- market-state deterioration
- ES relative to 10-year Treasury futures
- risk-on basket relative to defensive basket

Outputs:

```text
outputs/market_target_validation_report.md
outputs/market_target_validation_summary.csv
outputs/market_target_validation_quintiles.csv
outputs/market_target_component_diagnostics.csv
outputs/market_target_validation_weights.parquet
```

## Economic State

Run the current point-in-time economic snapshot:

```bash
PYTHONPATH=src python3 -m world_model.cli.economic_state
```

By default this writes:

```text
outputs/economic_state_report.md
```

Write the full daily state history:

```bash
PYTHONPATH=src python3 -m world_model.cli.economic_state \
  --history-out outputs/economic_state_history.parquet \
  --report-out outputs/economic_state_report.md
```

The current economic state vector includes:

- growth
- labor
- inflation pressure
- liquidity
- credit
- housing

The engine reads `fred/vintage/{series_id}.parquet`, not latest-only FRED files,
so historical economic states are based on values available at the time. See:

```text
docs/economic_state_engine.md
config/economic_state.yml
```

The repo does not currently download FRED data directly. `.env` is ignored, and
`.env.example` documents the expected `FRED_API_KEY` name for future downloader
work or for the existing data-management updater.

## Economic Trigger Validation

Validate the legacy threshold trigger framework against future state outcomes:

```bash
PYTHONPATH=src python3 -m world_model.cli.economic_triggers
```

This writes:

```text
outputs/economic_trigger_validation_report.md
outputs/economic_trigger_validation_summary.csv
outputs/economic_trigger_validation_forecasts.parquet
outputs/economic_trigger_current_snapshot.csv
outputs/economic_trigger_panel.parquet
```

The validator tests individual failed triggers, group counts, and aggregate
legacy trigger counts against future economic-state deterioration, future weak
economic state, and market-state deterioration. It compares trigger forecasts
against both an unconditional baseline and a current-state-bucket baseline.

Configuration and documentation:

```text
config/economic_triggers.yml
docs/economic_trigger_validation.md
```

## Combined State Validation

Test whether economic state conditions market-state usefulness:

```bash
PYTHONPATH=src python3 -m world_model.cli.combined_state
```

This compares unconditional, market-only, economic-only, and joint
market/economic bucket forecasts for future returns, drawdowns, state
deterioration, and risk-on versus defensive returns.

Outputs:

```text
outputs/combined_state_validation_report.md
outputs/combined_state_validation_summary.csv
outputs/combined_state_validation_forecasts.parquet
outputs/combined_state_panel.parquet
```

Configuration and documentation:

```text
config/combined_state_validation.yml
docs/combined_state_validation.md
```

## Simple Market Regime

Build a four-block market regime using only equity, rates, commodities, and FX:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_market \
  --history-out outputs/simple_market_regime_history.parquet
```

Run validation against the detailed market composite and common target outcomes:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_market --validate
```

Outputs:

```text
outputs/simple_market_regime_report.md
outputs/simple_market_regime_history.parquet
outputs/simple_market_validation_report.md
outputs/simple_market_validation_summary.csv
```

Configuration and documentation:

```text
config/simple_market_regime.yml
docs/simple_market_regime.md
```

## Simple Economic Regime

Build a three-axis economic regime using only activity, inflation pressure, and
liquidity / credit:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_economic \
  --history-out outputs/simple_economic_regime_history.parquet
```

Run validation against the detailed economic composite and common market-facing
targets:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_economic --validate
```

Outputs:

```text
outputs/simple_economic_regime_report.md
outputs/simple_economic_regime_history.parquet
outputs/simple_economic_validation_report.md
outputs/simple_economic_validation_summary.csv
```

Configuration and documentation:

```text
config/simple_economic_regime.yml
docs/simple_economic_regime.md
```

## Growth-Trend Timing

Test Philosophical Economics style growth-trend timing rules:

```bash
PYTHONPATH=src python3 -m world_model.cli.growth_trend
```

This compares buy and hold, trend-only timing, economic-only timing, and
growth-trend timing variants where weak growth turns on the trend filter.

Outputs:

```text
outputs/growth_trend_timing_report.md
outputs/growth_trend_timing_summary.csv
outputs/growth_trend_timing_annual_returns.csv
outputs/growth_trend_timing_panel.parquet
outputs/growth_trend_timing_trades.csv
outputs/growth_trend_timing_current_snapshot.csv
```

Configuration and documentation:

```text
config/growth_trend_timing.yml
docs/growth_trend_timing.md
```

## Development

Run tests:

```bash
PYTHONPATH=src python3 -m pytest
```
