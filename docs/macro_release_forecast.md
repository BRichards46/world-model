# Payroll and inflation release forecasting

The release forecast module evaluates whether information available before an
economic release improves on a simple autoregressive forecast.

It currently targets:

- first-release nonfarm payroll growth, translated back into thousands of jobs;
- headline CPI month over month;
- core CPI month over month; and
- core PCE month over month.

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

The daily `run_macro.sh` workflow refreshes these artifacts. They remain
research outputs; a forecast is not promoted to the dashboard merely because a
single current estimate is available.

Credit monitoring is kept separate from economic-release forecasting. The
high-yield drill-down shows investment-grade, BBB, broad high-yield, and CCC
option-adjusted spreads plus quality-dispersion gaps. ICE/FRED currently makes
only about three years of history available, so `BAA10Y` is retained as the
long-history bridge for regime research.
