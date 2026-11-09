# Macro / Market Relevance Validation

This study tests whether the dashboard's point-in-time macro states contain
stable information about subsequent market outcomes. It comes before any
market-facing macro composite or regime label.

## Inputs and timing

The routine operational macro feature panel is built from
`outputs/macro_dashboard_history.parquet`. The full research run uses
`outputs/macro_market_validation/macro_release_history_full.parquet`, a
point-in-time release-state backfill spanning 1994 through the present (with
later starts for series whose vintage stores are shorter). A feature changes
only on its `realtime_start`, then remains known until the next release.
Observation dates are never used as availability dates.

Market outcomes use the daily CSI processed futures universe configured in
`config/economic_asset_drivers.yml`. Each asset group is evaluated on its own
valid trading calendar so exchange holidays and missing contracts do not
silently eliminate other groups. Because CSI continuous contracts are
additively back-adjusted, the default return proxy is normalized point change,
not percentage or log return.

The existing daily market-state composite is the nested baseline. Evaluation
stops at its last valid observation rather than carrying a stale market state
forward.

## Features

Every dashboard indicator, including explicitly marked context rows, is tested
through:

- transformed level;
- transformed velocity;
- 10-year level percentile;
- 10-year velocity percentile;
- centered level-by-velocity percentile interaction.

These are information tests. Polarity changes interpretation but does not add
information, so inverted duplicates are not added to the statistical grid.

## Outcomes

The default grid covers 1, 3, 6, and 12-month horizons for:

- forward normalized return;
- future realized volatility;
- maximum drawdown within the forward window.

The daily data are not collapsed when outcomes are calculated: every trading
day inside each forward window contributes to cumulative return, realized
volatility, and path-dependent drawdown. Forecast origins are sampled at each
month-end to avoid treating a mass of overlapping daily forecasts as
independent observations.

At each month-end, forecasts are trained only on earlier evaluation dates whose
own forward windows had already completed. The study compares a macro-only
expanding rank regression with an unconditional baseline and a nested
market-plus-macro rank regression with the market-state-only regression.

## Robustness screens

The output includes:

- feature/target rank correlation;
- top-minus-bottom quintile outcome spread;
- quintile monotonicity;
- early-half and late-half rank correlations;
- walk-forward MSE improvement versus unconditional and market-state baselines;
- an overlap-adjusted approximate correlation p-value;
- a Benjamini-Hochberg q-value over the full search grid.

A row is called a robust candidate only when it has at least 60 out-of-sample
observations, a stable correlation sign across both sample halves, an
approximate q-value at or below 10%, and positive incremental MSE improvement
beyond the market-state baseline. Even then it is a research candidate, not a
trading rule.

## Run

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python \
  -m world_model.cli.macro_market_validation \
  --macro-history outputs/macro_market_validation/macro_release_history_full.parquet
```

Outputs are written under `outputs/macro_market_validation/`:

```text
macro_market_validation_report.md
macro_market_summary.csv
macro_market_forecasts.parquet
macro_market_feature_panel.parquet
macro_market_feature_metadata.csv
macro_market_asset_group_returns.parquet
macro_market_target_panel.parquet
```

Use `--individual-assets` with a separate output directory to replace each
equal-weight asset class with its configured constituent contracts. Repeat
`--indicator-id` to limit a targeted rerun to changed macro definitions. The
dashboard continues to consume only the class-level summary.

The equity class is intentionally limited to ES and NQ. YM and NK are excluded
so the class represents the U.S. broad-market and growth-index exposures used
in this dashboard rather than an equal-weight global/index-style basket.

The research archive has now been backfilled from the underlying FRED vintage
store. Overall feature coverage begins in 1994; individual indicators enter
when their own defensible vintage history begins. The operational dashboard
history remains a smaller rolling artifact and is not overwritten by this
research backfill.

The statistical output should be interpreted in relationship families rather
than raw row counts. Level values and level percentiles, adjacent horizons, and
related risk targets often restate the same economic effect. In the first full
run, the evidence is concentrated in future volatility and drawdown, especially
for commodity groups. Directional-return evidence is sparse. That supports a
next-stage forecasting study for selected risk relationships; it does not yet
support naming market regimes.
