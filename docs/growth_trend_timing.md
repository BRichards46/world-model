# Growth-Trend Timing

The growth-trend timing engine tests the Philosophical Economics style idea
that trend following works better when it is conditioned by macro growth.

The rule is intentionally simple:

```text
If growth is healthy: stay long.
If growth is weak: defer to the trend rule.
```

This differs from the earlier state-vector tests. It is not trying to forecast
returns from a static state score. It is testing whether macro weakness helps
distinguish a meaningful trend break from an ordinary whipsaw.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.growth_trend
```

Outputs:

```text
outputs/growth_trend_timing_report.md
outputs/growth_trend_timing_panel.parquet
outputs/growth_trend_timing_summary.csv
outputs/growth_trend_timing_annual_returns.csv
outputs/growth_trend_timing_trades.csv
outputs/growth_trend_timing_current_snapshot.csv
```

Configuration:

```text
config/growth_trend_timing.yml
```

## Included Strategies

The default configuration tests:

- buy and hold
- monthly 10-month moving average trend following
- daily 200-day moving average trend following
- economic-only retail sales / industrial production growth
- GTT with retail sales / industrial production and monthly trend
- GTT with retail sales / industrial production and daily trend
- GTT with unemployment trend and monthly trend
- GTT with the simplified activity axis and monthly trend

## Current Assumptions

The first pass uses:

- `ES` as the default target asset
- cash as the defensive asset
- point-in-time FRED economic inputs through the economic state engine
- a configurable one-way transaction cost on exposure changes
- next-day application of signals observed at the close

Those assumptions are deliberately conservative and easy to inspect. Cash yield
and alternate defensive assets can be added after the basic signal test is
understood.
