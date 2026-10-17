# Simple Market Regime

The simple market regime engine is a deliberately coarse alternative to the
detailed market state engine.

It reduces market state to four blocks:

- `equity_score`
- `rates_score`
- `commodities_score`
- `fx_score`

Each block is built from trend and momentum only. The FX block is inverted, so
positive FX means weaker dollar / easier dollar-liquidity pressure.

## Why This Exists

The detailed market state model has many dimensions: trend, momentum, breadth,
volatility, rates, dollar, commodities, and positioning. That is useful for
description, but it can make historical states too unique for analog or bucket
testing.

The simple regime layer asks whether a lower-dimensional regime is more stable:

```text
Equity / Rates / Commodities / FX
```

## Current Regime Report

Run:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_market \
  --history-out outputs/simple_market_regime_history.parquet
```

Outputs:

```text
outputs/simple_market_regime_report.md
outputs/simple_market_regime_history.parquet
```

## Validation

Run:

```bash
PYTHONPATH=src python3 -m world_model.cli.simple_market --validate
```

Outputs:

```text
outputs/simple_market_validation_report.md
outputs/simple_market_validation_summary.csv
outputs/simple_market_validation_forecasts.parquet
outputs/simple_market_validation_panel.parquet
```

The validation compares:

- unconditional baseline
- detailed market composite bucket
- simple composite bucket
- equity block bucket
- rates block bucket
- commodities block bucket
- FX / dollar block bucket

Targets include:

- ES forward return
- ES 10% drawdown risk
- risk-on vs defensive returns
- simple regime deterioration

## Interpretation

The simple model should not be judged only by whether it predicts ES returns.
The first question is whether it creates a more stable, interpretable regime
map. The second question is whether any individual block adds predictive value
against specific targets.

Initial validation suggests the simple composite is materially better at
predicting future deterioration in the simple regime itself, especially over
6m to 12m. It does not currently improve ES drawdown or return prediction
versus the detailed market composite.
