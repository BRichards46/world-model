# Combined State Validation

This module tests whether economic state conditions the usefulness of market
state.

The core question is:

```text
Does a joint market + economic state bucket forecast future outcomes better
than a market-state bucket alone?
```

## Models Compared

The validator compares:

- unconditional baseline
- market composite bucket
- economic composite bucket
- market x economic composite bucket
- market x economic activity bucket
- market x inflation pressure bucket
- market x liquidity/credit bucket

Joint buckets are deliberately simple. They are not fitted regressions or
machine-learning models. They ask whether similar historical combinations of
market and economic state had different future outcomes.

## Targets

The default target set includes:

- ES forward return
- ES 10% drawdown risk
- market-state deterioration
- economic-state deterioration
- risk-on basket relative to defensive basket

Each target is tested over 1m, 3m, 6m, and 12m horizons.

## Interpretation

The important column is:

```text
improvement_vs_market_bucket
```

Positive values mean the joint market/economic bucket had lower error than the
market-only bucket. Negative values mean economic conditioning made the forecast
worse.

For binary targets, error is Brier score. For continuous targets, error is RMSE.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.combined_state
```

Outputs:

```text
outputs/combined_state_validation_report.md
outputs/combined_state_validation_summary.csv
outputs/combined_state_validation_forecasts.parquet
outputs/combined_state_panel.parquet
```
