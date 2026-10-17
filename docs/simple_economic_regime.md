# Simple Economic Regime

The simple economic regime engine is a deliberately coarse alternative to the
detailed FRED economic state engine.

It reduces economic state to three axes:

- `activity_score`
- `inflation_pressure_score`
- `liquidity_credit_score`

The simple composite gives positive weight to activity and liquidity / credit,
and negative weight to inflation pressure.

## Why This Exists

The detailed economic model includes growth, labor, inflation, liquidity,
credit, housing, and the individual point-in-time FRED indicators underneath
each component. That is useful for diagnosis, but it can make the macro state
too granular for historical bucket testing.

The simple regime layer asks whether a lower-dimensional macro map is more
stable:

```text
Activity / Inflation Pressure / Liquidity-Credit
```

## Current Regime Report

Run:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_economic \
  --history-out outputs/simple_economic_regime_history.parquet
```

Outputs:

```text
outputs/simple_economic_regime_report.md
outputs/simple_economic_regime_history.parquet
```

## Validation

Run:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_economic --validate
```

Outputs:

```text
outputs/simple_economic_validation_report.md
outputs/simple_economic_validation_summary.csv
outputs/simple_economic_validation_forecasts.parquet
outputs/simple_economic_validation_panel.parquet
```

The validation compares:

- unconditional baseline
- detailed economic composite bucket
- simple economic composite bucket
- activity axis bucket
- inflation pressure axis bucket
- liquidity / credit axis bucket

Targets include:

- simple economic deterioration
- detailed economic deterioration
- market state deterioration
- ES forward return
- ES 10% drawdown risk
- risk-on vs defensive returns

Use `--no-market-targets` to test only the economic deterioration targets.

## Interpretation

This model should be judged first as a stable descriptive map. Predictive value
is a separate requirement. The validation report tests whether the simpler axes
or composite improve out-of-sample bucket forecasts versus the detailed
economic composite.
