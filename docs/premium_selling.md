# Premium-Selling Suitability Monitor

The dashboard's Options tab is a research and judgment aid for one-month
premium selling. It does not claim to reproduce option P&L where historical
option chains are unavailable.

## What it measures

For every covered asset, the builder:

1. Reads total-return OHLC history from the canonical Norgate catalog.
2. Builds the visible 1-, 3-, 6-, 10-, and 12-month volatility-adjusted
   momentum components.
3. Selects the 100 nearest completed weekly historical states.
4. Measures the next 21 sessions' terminal return, upside MFE, downside MAE,
   and realized volatility from the following session's open.
5. Reads current IVX, IV rank, trailing realized volatility, skew, and the
   expiry nearest 30 DTE from the Tanuki normalized option tables.
6. Applies today's 16-delta put and call distances to the historical paths to
   estimate conditional and unconditional underlying touch rates.

A strategy lean requires a sufficiently rich current-IV edge and a compatible
path/touch result. The thresholds are explicit in
`config/premium_selling.yml`.

## Evidence grades

- `B — Path-tested`: underlying outcomes and current option pricing are
  available; actual option P&L is not yet backtested.
- `C — Path only`: an underlying path estimate exists, but the current option
  surface is unavailable.

Touch rates are not assignment probabilities. The model does not include
transaction costs, early exits, skew changes, gap execution, or option Greeks
through the holding period.

## Coverage

The initial option-covered futures proxies are ES/SPY, NQ/QQQ, RTY/IWM,
GC/GLD, and SI/SLV. The 11 Select Sector SPDR ETFs are included, although an
asset is explicitly marked path-only when its Tanuki option surface is absent.
Other configured futures remain on the Trend tab until an appropriate
option surface is available.

## Run

The normal dashboard workflow runs the builder automatically:

```bash
~/PyBry/world-model/run_macro.sh
```

Run only this builder with:

```bash
PYTHONPATH=src python -m world_model.cli.premium_selling
```

It writes:

- `outputs/premium_selling_snapshot.json`
- `sites/macro-dashboard/dist/data/premium_selling.json`
