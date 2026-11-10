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

The dashboard opens with an outlook (`outlook.json`, built by
`world_model.cli.macro_outlook` from the other payloads):

- **Where we are:** the quad, the equity risk overlay and how far the S&P is
  from breaking trend, growth, jobs and inflation.
- **Where the quad model points:** this quarter and the next two, with
  probability and the model's historical hit rate at each horizon.
- **Coming up:** each scheduled release with our forecast, 50% and 80% ranges
  from the model's last five years of out-of-sample misses, and what the print
  would change. That includes the exact level that would trip an overlay
  signal (for example the unemployment rate that crosses the 0.1 pp buffer)
  and the monthly CPI pace each quarter needs for inflation to accelerate.

Below it, a "What to do now" panel shows only validated signals: the equity
risk overlay, 6–12 month trend on the main futures, and option pricing
(implied vs expected realized vol). Everything else on the page is labeled
context.

The Trend tab shows the directional-momentum tape for the selected CSI futures
and the 11 Select Sector SPDR ETFs: 1-, 3-, 6-, 10-, and 12-month
volatility-adjusted components, with sectors also relative to SPY. The
composite weights 6–12 months most, following
[`docs/trend_range_study.md`](docs/trend_range_study.md). See
[`docs/directional_momentum.md`](docs/directional_momentum.md).

The Options tab shows observed option pricing for option-covered index/metal
proxies and sector ETFs. Its analog-based strategy leans are labeled untested
until an option P&L backtest exists. See
[`docs/premium_selling.md`](docs/premium_selling.md).

Release-aware payroll and inflation forecasts are refreshed by the same daily
workflow and written to `release_forecasts.json`. Only validated forecasts are
presented as signals; the rest appear as labeled previews. See
[`docs/macro_release_forecast.md`](docs/macro_release_forecast.md).

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

## Macro Dashboard Dataset

For the normal interactive workflow, run:

```bash
./run_macro.sh
```

The launcher prefers this project's virtual environment, falling back to the
first working shell Python only when the venv lacks the declared dependencies. It
validates the local FRED vintage store, incrementally rebuilds the dashboard
data, and starts the local monitor at `http://127.0.0.1:8765/` if it is not
already running. It does not fetch or update FRED. Your separate data-management
process owns that work and writes the point-in-time files to the shared local
FRED store; this launcher only consumes those files. Pass `--no-serve` when a
scheduled job should rebuild the dashboard files without starting a web server.
Other arguments, such as `--as-of YYYY-MM-DD` or `--no-cache`, are forwarded to
the dashboard CLI.

After the separate data-management process has updated the shared FRED store,
build the normalized dashboard history and current snapshot with:

```bash
PYTHONPATH=src python3 -m world_model.cli.macro_dashboard
```

The first run bootstraps five years of point-in-time release states. Later runs
reuse that history and recalculate a 14-day overlap so revisions are picked up
without rebuilding the full vintage archive. It writes:

```text
outputs/macro_dashboard_history.parquet
outputs/macro_dashboard_snapshot.parquet
outputs/macro_dashboard_snapshot.json
sites/macro-dashboard/dist/data/snapshot.json
sites/macro-dashboard/dist/data/history.json
```

The browser-ready files feed the dense monitor in `sites/macro-dashboard`.
Preview the interface locally after any dashboard build with:

```bash
python3 -m http.server 8765 --directory sites/macro-dashboard/dist
```

Then open `http://127.0.0.1:8765/`. The interface includes category filters,
search, an attention-only view, level and velocity percentiles, recent-history
sparklines, freshness, plain-language momentum streaks, and an auditable detail
drawer for every row. Each drawer also includes
selectable 1-year, 3-year, 5-year, and full release histories for the reported
reading and both percentile measures, with revision releases marked explicitly.
The checked-in site is buildless; publishing a new hosted version remains an
explicit deployment step after the daily data refresh.

The detail drawers also expose familiar, unscored context: payroll job changes,
headline and component inflation, income and retail-sales confirmation, nominal
and real growth comparisons, GDP/PCE nowcasts, policy-rate components, latest
market spreads, permits, and housing starts relative to the labor force. These
values use the same point-in-time vintage store but do not create additional
primary dashboard scores.

Housing starts intensity is displayed as a secondary Housing row rather than an
isolated context statistic. It ranks the three-month average of `HOUST /
CLF16OV` (starts per 1,000 labor-force participants) and the ratio's year-over-year
change against their respective histories.

Use `--as-of YYYY-MM-DD` for a historical run, `--start-date YYYY-MM-DD` to
control the recalculation boundary, or `--no-cache` to replace the saved
history from the full available vintage archive. This command is ready to be invoked by a daily scheduler;
the repository does not install a scheduler or publish the hosted site
automatically.

## Macro / Market Relevance Validation

Test the dashboard's point-in-time levels, velocities, percentiles, and
interactions against future daily-market outcomes before defining a
market-facing macro regime:

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python \
  -m world_model.cli.macro_market_validation \
  --macro-history outputs/macro_market_validation/macro_release_history_full.parquet
```

The walk-forward study covers the configured CSI futures groups and tests
forward normalized returns, realized volatility, and drawdowns over 1, 3, 6,
and 12-month horizons. It compares each macro feature both with an unconditional
baseline and incrementally with the existing market-state composite. Daily
observations determine the complete forward outcome paths; forecast origins are
sampled monthly to limit pseudo-replication from overlapping windows. See
`docs/macro_market_validation.md` for methodology and interpretation.

To test the configured futures contracts individually, keep the output separate
from the class-level file consumed by the dashboard:

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python \
  -m world_model.cli.macro_market_validation \
  --individual-assets \
  --output-dir outputs/macro_market_asset_validation \
  --report-out outputs/macro_market_asset_validation/macro_market_asset_validation_report.md
```

## GTT Drawdown and Volatility Validation

Test whether Growth–Trend Timing double-confirmation states identify worse
future volatility and drawdowns than a negative price trend alone:

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python \
  -m world_model.cli.gtt_risk_validation
```

The study reconstructs point-in-time unemployment, industrial-production, and
real-retail-sales warnings from the FRED vintage store, combines them with
10-month and 200-day price trends, and evaluates the resulting states against
daily market paths. Its primary equity history uses Norgate `$SPX` before total-return
data begin and `$SPXTR` thereafter, chaining returns rather than raw index
levels. See `docs/gtt_risk_validation.md`.

The follow-on production decomposition is repeatable with:

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python \
  -m world_model.cli.gtt_production_validation
```

It compares point-in-time production components, manufacturing hours, real
orders, activity indexes, and capacity utilization against the `INDPRO`
benchmark without altering the core GTT specification. See
`docs/gtt_production_validation.md`.

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
