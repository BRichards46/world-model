# Directional Momentum Monitor

The dashboard's Trend tab is a descriptive cross-asset tape. It is separate
from the macro state models and does not treat momentum as a forecast by itself.

## Universe

The first production universe contains 14 CSI continuous futures and the 11
Select Sector SPDR ETFs. SPY is the relative-performance benchmark for the
sector ETFs but is not displayed as a sector row.

Configuration lives in `config/directional_momentum.yml`. The configured
horizons are 21, 63, 126, 210, and 252 trading days, displayed as 1, 3, 6, 10,
and 12 months.

## Data sources

- Futures are read from the configured CSI processed-data root.
- ETFs and SPY are resolved through the canonical Norgate
  `metadata/price_file_catalog.parquet` and read from the `TOTALRETURN` price
  files. Stable Norgate asset IDs, rather than constructed filenames, identify
  the authoritative files.

## Calculations

CSI continuous futures are additively back-adjusted and can cross zero. Their
raw component is therefore a normalized point move, not a percentage return:

```text
daily move = price change / prior trailing-252-day median absolute price
horizon raw momentum = sum of daily moves over the horizon
```

Sector ETF raw momentum is the ordinary total return from Norgate's adjusted
close. Relative raw momentum is the change in the sector/SPY total-return
ratio.

For both asset classes, volatility-adjusted momentum is:

```text
cumulative horizon move / (daily realized volatility × sqrt(horizon days))
```

Sector relative momentum uses the volatility of daily sector-minus-SPY log
returns, i.e. realized tracking error.

Positive always means the named asset's price is rising. Treasury and dollar
signals are not inverted into a generic risk-on convention.

## Bias labels

The headline bias is a weighted average of the five volatility-adjusted
components. It is only directional when the raw and adjusted composite signs
agree and enough raw horizon components point the same way. Otherwise the
label is `Mixed`. The dashboard always shows the underlying horizon scores,
raw values, alignment count, and confidence description.

The weights are 5%, 15%, 25%, 25%, and 30% from shortest to longest horizon
(changed in November 2026 from 10/20/25/25/20). The trend study in
`docs/trend_range_study.md` found 12-month trend the strongest and 3-month the
weakest across our futures and sector indices. Thresholds and weights remain configuration, not hidden UI logic.

## Outputs and operation

Run directly with:

```bash
PYTHONPATH=src python -m world_model.cli.directional_momentum
```

The normal `run_macro.sh` workflow runs the builder automatically. It writes:

- `outputs/directional_momentum_history.parquet`
- `outputs/directional_momentum_snapshot.parquet`
- `outputs/directional_momentum_snapshot.json`
- `sites/macro-dashboard/dist/data/directional_momentum.json`

The dashboard snapshot reports source freshness. Historical outputs preserve
the individual components for later forward-return validation and alternative
weighting tests.

## Excursion validation

The research validation is intentionally separate from the daily dashboard
build. Run it with:

```bash
PYTHONPATH=src python -m world_model.cli.directional_momentum_validation
```

It samples signals weekly and evaluates the subsequent 1-week, 1-, 3-, and
6-month paths. For each signal orientation it measures maximum favorable
excursion (MFE), maximum adverse excursion (MAE), and MFE minus MAE in ex-ante
volatility units. These are compared with the same asset, lens, direction, and
forward horizon on all eligible weekly dates. Terminal directional accuracy is
reported but is not the pass criterion.

A relationship passes the strict screen only when the conditional median has
more MFE, less MAE, and a better MFE-minus-MAE balance, with at least 40 signal
observations. `stable_across_periods` additionally requires a pass in both the
full-history and 2015-onward samples. Because multi-week paths overlap and the
variants are correlated, this is a descriptive stability screen rather than a
formal significance claim.

Outputs are written to `outputs/directional_momentum_validation/`.

## Matching-horizon cross-asset regimes

The ES/NQ/DX/ZB regime study is run separately with:

```bash
PYTHONPATH=src python -m world_model.cli.directional_momentum_regime_validation
```

At each configured momentum horizon, every asset is classified as positive,
negative, or neutral using the same component threshold and raw/adjusted sign
agreement rule as the main validation. The observed four-asset configuration
is then tested for each target asset and forward window.

The comparison baseline is not unconditional: it is the target asset's own
momentum state at that same lookback. A supportive regime must add MFE, reduce
MAE, and improve excursion balance beyond that own-signal baseline. An adverse
modifier must worsen all three measures. Stable flags require the condition in
both the full and 2015-onward samples.

Outputs are written to `outputs/directional_momentum_regime_validation/`.
