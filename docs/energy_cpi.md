# Bottom-up energy and headline CPI nowcast

`world_model.states.energy_cpi` builds headline CPI from blocks, each forecast
from its own driver, using only data published by the day before each CPI
release.

| Block | Driver |
|---|---|
| Gasoline | The month's average weekly retail price (`GASREGW`) against last month's, calibrated to unadjusted gasoline CPI over the trailing five years. It is then seasonally adjusted with last year's seasonal step between the same two months, taken from the adjusted and unadjusted indexes as published at the time. |
| Electricity | Average of its last three monthly changes |
| Utility gas | Henry Hub spot this month and the two before (trailing regression) |
| Fuel oil | Retail diesel this month and last (trailing regression) |
| Energy | The four blocks, with weights fitted on the trailing three years |
| Food | Average of its last three monthly changes |
| Core | Our walk-forward core CPI forecast (`macro_release_forecast`) |
| Headline | Core, food and energy, with weights fitted on the trailing three years |

Retail gasoline, diesel and Henry Hub prices are effectively never revised.
Their values up to the cutoff stand in for vintages. Every CPI series is read
as published on the cutoff date. For a month still in progress, the latest
weekly pump price is held flat to month end.

```bash
PYTHONPATH=src python3 -m world_model.cli.energy_cpi
```

Outputs, in `outputs/energy_cpi/`:
- `energy_cpi_backtest.parquet`: one row per release, with every block's forecast and first print.
- `energy_cpi_summary.json`: accuracy, the comparison with the market and the Cleveland Fed, and the live nowcast.

## Results (run 2026-11-12, releases June 2013 to October 2026)

| Block | Average miss (pp) | Typical monthly move (sd) |
|---|---:|---:|
| Gasoline | 0.41 | 5.67 |
| Electricity | 0.64 | 0.81 |
| Utility gas | 1.69 | 2.46 |
| Fuel oil | 1.96 | 4.50 |
| Energy | 0.44 | 2.99 |
| Food | 0.16 | 0.29 |
| Core (ours) | 0.105 | 0.17 |
| **Headline** | **0.100** | 0.30 |

With core and food known exactly, headline from the energy block misses by
0.044 on average. The energy side is close to solved, and what remains is mostly
core and food.

Headline CPI against other forecasts (same releases):

| Sample | Bottom-up | Previous model | Market (Kalshi) | Cleveland Fed | Naive |
|---|---:|---:|---:|---:|---:|
| 2019-12 to 2026-10 (81) | **0.127** | 0.209 | n/a | 0.133 | 0.220 |
| 2021-10 to 2026-10 (59) | 0.108 | 0.211 | **0.089** | 0.123 | n/a |

The bottom-up build:
- **Halves our previous headline miss.**
- **Beats the Cleveland Fed nowcast.**
- **Still trails the market.** It adds no information beyond the market's
  expectation (slope -0.10, t -0.5).

On the headline-minus-core gap, the part energy and food drive, it misses by
0.050 against the market's 0.040. The rest of the shortfall is our core
forecast.

## On the dashboard

The bottom-up nowcast is the dashboard's headline CPI forecast. `energy_cpi`
runs after `macro_release_forecast` and replaces the headline row in
`release_forecasts.json` (variant `bottom_up`). Its 50% and 80% ranges come from
its own last 60 walk-forward misses.

It is marked validated: it beats the release module's autoregressive benchmark
over the full 92 releases (0.118 vs 0.210) and in both halves.
`release_benchmarks` then scores headline CPI on this nowcast against the market
and the Cleveland Fed.

## Next refinements

- **Gasoline futures for the rest of the month.** Use RBOB (CSI `RB`) instead of
  holding the latest pump price flat. This matters early in the month, before
  most of the weekly prices are in.
- **Electricity and food.** These are now simple trailing averages: electricity
  has a correlation of 0.15 with the outcome, food 0.65. EIA prices and PPI food
  would help.
- **Core.** The larger remaining gap: shelter from new-lease rents, used cars
  from Manheim, car insurance.
