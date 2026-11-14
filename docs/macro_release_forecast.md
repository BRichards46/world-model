# Payroll and inflation release forecasting

The release forecast module evaluates whether information available before an
economic release improves on a simple autoregressive forecast.

It currently targets:

- first-release nonfarm payroll growth, translated back into thousands of jobs;
- headline CPI month over month;
- core CPI month over month;
- core PCE month over month; and
- industrial production month over month.

All feature values are reconstructed from FRED/ALFRED vintages dated before the
target release. The target itself uses the value and revised prior-month value
known on its first release date. This prevents final revised history from
leaking into either the inputs or the reported payroll change.

Run:

```bash
PYTHONPATH=src python3 -m world_model.cli.macro_release_forecast
```

Outputs are written under `outputs/macro_release_forecast/`. A specification is
only marked as a research candidate when it beats the autoregressive benchmark
on mean absolute error over the full, early, and late out-of-sample windows and
meets the directional threshold. The shorter ADP variant is reported
separately and requires at least 36 out-of-sample releases before promotion.

Each forecast in `release_forecasts.json` carries 50% and 80% ranges. They are
the 25th–75th and 10th–90th percentiles of the model's last 60 out-of-sample
misses (actual minus forecast), added to the current forecast and converted to
display units. A recent window keeps a persistent bias inside the range: for
example, payrolls have recently come in below the model, so their range sits
mostly below the point forecast. The dashboard's outlook shows these ranges for
every upcoming release; only validated forecasts are labeled as such.

The daily `run_macro.sh` workflow refreshes these artifacts. They remain
research outputs; a forecast is not promoted to the dashboard merely because a
single current estimate is available.

## Industrial production (added 2026-11-13)

The jobs report comes out about ten days before industrial production, and the
Fed builds much of its early manufacturing estimate from production-worker
hours. The `hours` variant forecasts the first print's monthly change from:
- the prior month's change and its 3-month average;
- the target month's change in manufacturing average weekly hours (`AWHMAN`);
- the target month's change in total private aggregate hours (`AWHI`);
- payroll growth (`PAYEMS`).

It requires every feature, so it trains only from 1997, when aggregate-hours
vintages begin. Walk-forward, 295 releases from 2002 to 2026:

| | Average miss (pp) | Excluding 2020 |
|---|---:|---:|
| Hours model | 0.48 | 0.42 |
| Prior-month autoregression | 0.56 | 0.47 |

It beats the autoregression in both halves of the sample and gets the sign of
the change right 67% of the time, so it is validated. Manufacturing output
alone is more predictable: a separate test cut its miss by 24% (0.47 to 0.36).
The total is held back by utilities (weather) and mining, which hours don't
cover. Factory orders and oil prices added nothing.

Data that should help further, not yet in the FRED store:
- `MANEMP` (manufacturing employment);
- `CES3000000034` (aggregate weekly hours of production workers in
  manufacturing, the Fed's own input);
- `IPUTIL` and `IPMINE` (to model utilities and mining separately);
- heating and cooling degree days, from NOAA rather than FRED.

The dashboard's "Coming up" section shows this forecast and its ranges.

Credit monitoring is kept separate from economic-release forecasting. The
high-yield drill-down shows investment-grade, BBB, broad high-yield, and CCC
option-adjusted spreads plus quality-dispersion gaps. ICE/FRED currently makes
only about three years of history available, so `BAA10Y` is retained as the
long-history bridge for regime research.
