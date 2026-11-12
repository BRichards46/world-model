# Release forecasts against the market and the Cleveland Fed

`world_model.cli.release_benchmarks` scores our walk-forward release forecasts
(`macro_release_forecast`) against two outside expectations. All three are
compared with the first print.

- **Kalshi.** Each release has a ladder of "above X" contracts. Their prices
  just before trading closes (8:25 ET on release day) give the market's
  probability for each range. The median of that distribution is the market
  estimate.
  - Kalshi settles on the rounded print, so strikes are shifted by half a
    rounding step.
  - Prices that rise with the strike are repaired.
  - Where one month was listed under two ticker prefixes, the busier event is
    kept.
  - The data is public, no key needed. It is cached in
    `~/Documents/data/kalshi/release_ladders/`, and later runs fetch only new
    releases.
- **Cleveland Fed inflation nowcast.** The last daily nowcast shown before the
  release, read from the Cleveland Fed's chart file. It covers 2013 onward and
  is saved to `~/Documents/data/cleveland_nowcast/`. Its stored actuals match
  our first prints exactly. Whether the stored daily values are as originally
  shown has not been confirmed independently.

March to December 2020 is excluded from every average. Payrolls use the core
variant, converted to thousands with the prior month's employment level as
published on release day.

```bash
PYTHONPATH=src python3 -m world_model.cli.release_benchmarks            # fetch new data, then score
PYTHONPATH=src python3 -m world_model.cli.release_benchmarks --offline  # cached data only
```

Outputs, in `outputs/release_benchmarks/`:
- `release_comparison.csv` / `.parquet`: one row per release.
- `release_benchmark_summary.json`: accuracy, the edge tests, and the market
  and Cleveland Fed expectations for upcoming releases.

## Edge test

Smallest error is not the only question. When our forecast differs from the
market, does the print land on our side? Each release's surprise against the
market (actual minus Kalshi) is regressed on our difference from it (ours minus
Kalshi).
- A slope near 1 means our differences carry information.
- A slope near 0 means they are noise.

The same-side rate counts releases where we differed by more than 0.05 pp for
inflation, or 25k for payrolls.

## Results (run 2026-11-11)

Mean absolute error on the releases every source covers:

| Release | Releases | Ours | Kalshi | Cleveland Fed | Naive |
|---|---:|---:|---:|---:|---:|
| Headline CPI MoM (2021-10 to 2026-10) | 59 | 0.211 | **0.089** | 0.123 | 0.216 |
| Core CPI MoM (2022-06 to 2026-10) | 50 | 0.098 | **0.078** | 0.093 | 0.097 |
| Core PCE MoM (2022-11 to 2026-09) | 24 | 0.071 | **0.039** | 0.069 | 0.085 |
| Payrolls, thousands (2023-03 to 2026-10) | 41 | 80.8 | **64.9** | n/a | 88.2 |

Edge over the market (slope of surprise on our gap from it):

| Release | Ours over Kalshi | Cleveland Fed over Kalshi |
|---|---|---|
| Headline CPI | -0.10 (t -1.5), 49% same side | -0.06 (t -0.4) |
| Core CPI | -0.33 (t -1.4), 52% | +0.20 (t +1.2), 64% |
| Core PCE | +0.19 (t +1.4), 54% | +0.20 (t +1.6), 43% |
| Payrolls | +0.36 (t +1.4), 66% of 35 | n/a |

What this says:
- **The market is the best forecaster of every release here.** It is also not
  a last-minute effect: prices a day earlier score the same.
- **None of our forecasts adds reliable information to the market.** The
  payroll result (66% same side) is suggestive but not significant. It is
  partly explained by our forecasts running high on average, at +64k.
- **Against the Cleveland Fed alone, our core forecasts do carry information.**
  Slopes are +0.71 (t 3.3) for core CPI and +0.66 (t 3.6) for core PCE over the
  longer common sample. The market already prices it.
- **Headline CPI is our weakest model.** It is no better than naive, while the
  market's error is less than half of ours. The gap is energy prices, which our
  model does not use.

The practical consequence: use the market's expectation as the baseline for
"what's expected", and treat a surprise against it, not against our forecast,
as the news.

## On the dashboard

Headline and core CPI are scored on their bottom-up nowcasts (docs/energy_cpi.md,
docs/core_cpi.md), which are what the dashboard shows.

The run writes `sites/macro-dashboard/dist/data/release_benchmarks.json`, and
`macro_outlook` attaches it to each upcoming release card. Each card shows:
- the market's expectation and 80% range (blue);
- the Cleveland Fed nowcast;
- our forecast and its ranges;
- each source's average miss.

Market prices for upcoming releases are live quotes taken at run time. A range
end marked "or more" or "or less" means the market puts more than 10% beyond the
outermost listed strike. Where the market and our forecast differ by 0.1 pp or
more on CPI, or by 50k or more on payrolls, the card says so and notes which
source has been more accurate.
