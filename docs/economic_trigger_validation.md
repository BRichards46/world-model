# Economic Trigger Validation

This module tests the older threshold trigger framework against future state
outcomes.

The legacy framework treats each condition as healthy when it is above or below
a fixed threshold, then counts healthy conditions. For example, industrial
production YoY is healthy above `1%`, while CPI YoY is healthy below `2.5%`.

## What It Builds

The validator creates a point-in-time trigger panel from FRED vintages:

```text
outputs/economic_trigger_panel.parquet
```

For each trigger it stores:

- raw transformed value
- threshold margin, where positive means healthy
- healthy flag
- failed flag
- latest economic observation date known at the time

It also stores aggregate counts:

- `healthy_count`
- `failed_count`
- `healthy_share`
- `failed_share`
- `legacy_econ_score`
- `legacy_econ_failed`

The default legacy health threshold is `10`.

## Targets

The first validation report tests trigger features against:

- economic state deterioration
- future weak economic state
- market state deterioration, when market state config is available

Each target is tested at 1m, 3m, 6m, and 12m horizons.

## Baselines

The report compares trigger forecasts against two baselines:

- unconditional historical event rate
- current-state-bucket event rate

The second baseline is important. It asks whether a trigger adds information
beyond the current economic or market state level. This avoids over-crediting a
trigger for simple bounded-score mean reversion.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.economic_triggers
```

Outputs:

```text
outputs/economic_trigger_validation_report.md
outputs/economic_trigger_validation_summary.csv
outputs/economic_trigger_validation_forecasts.parquet
outputs/economic_trigger_current_snapshot.csv
outputs/economic_trigger_panel.parquet
```

To skip market-state targets and test economic targets only:

```bash
PYTHONPATH=src python3 -m world_model.cli.economic_triggers --no-market-targets
```

## Interpretation

Treat the report as evidence ranking, not as a final model. A trigger is more
interesting when it:

- has a meaningful active-vs-inactive event-rate spread
- improves Brier score versus the current-state-bucket baseline
- works across adjacent horizons
- makes economic sense

Triggers that only beat the unconditional baseline are not strong evidence.
