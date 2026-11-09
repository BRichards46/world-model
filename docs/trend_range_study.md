# TREND, TRADE range and the quad filter

A test of the Hedgeye-style process: intermediate-term TREND for direction, a
short-term TRADE range for entries, and the growth/inflation quad as a filter on
which trends to take. Hedgeye's Risk Range is proprietary. Here it is
approximated by distance from the 15-day average in units of 15-day volatility.

## Design

- **Assets:** S&P 500, Russell 2000 and ten S&P 500 sector total-return indices
  (Norgate), TLT and GLD, and five CSI futures groups (Treasury duration,
  dollar, precious metals, energy, industrial metals). The configuration comes
  from `config/quad_asset_performance.yml`.
- **TREND:** the average of vol-adjusted 3-, 6- and 12-month momentum. Single
  3-month and 12-month lookbacks are run as sensitivity checks.
- **TRADE band:** (price − 15-day average) / (63-day daily vol × √15).
  - *Favorable* entries are dips in uptrends and rallies in downtrends.
  - *Unfavorable* entries chase the move.
  - Thresholds are 0 (halves) and 0.5 (extremes, about 36% of observations).
- **Portfolios:** inverse-vol weighted with unit gross exposure.
  - Signals are weekly for TREND and TRADE, with 1- and 3-week holds.
  - The quad filter uses month-end quad signals with 1- and 3-month holds.
  - Every position starts one session after the signal.
- **Quad filter:** trend positions in the 13 playbook assets, split into ones the
  textbook quad playbook agrees with and ones it disagrees with.
  - The null hypothesis is that the quad adds nothing beyond trend.
  - It is tested by rotating the quad sequence against dates.
- **Statistics:**
  - Newey-West t-stats.
  - Benjamini-Hochberg within study and horizon.
  - *Significant* also requires the same sign in both halves of the sample.

## Findings (1990–November 2026; quad filter from 1998)

| Question | Result |
| --- | --- |
| Does TREND work? | **Yes.** Composite +3.3 to +3.9%/yr (Sharpe 0.4–0.5, q ≤ 0.05). The 12-month lookback is strongest (+4.5 to +4.9%/yr, Sharpe 0.55–0.65). The 3-month lookback, closest to Hedgeye's TREND, is weakest (+1.7 to +2.1%/yr, not significant at the 1-week hold). TREND is weaker since 2010 (composite +0.5 to +2.0%/yr). |
| Does range timing improve trend entries? | **No.** Favorable minus unfavorable entries is −2.3 to +4.4%/yr, with every t below 1.4 and no consistent sign across halves. The result is the same for equities and for other assets; outside equities, chasing did slightly better. |
| Does fading the band alone work? | **No.** +0.1%/yr at 1 week and −1.0%/yr at 3 weeks; not significant. |
| Does the quad add to TREND? | **Not with any tradeable quad.** Agree minus disagree ranges from −3.9 to +1.5%/yr (rotation p 0.38–0.89). With the projected next-quarter quad, the positions it agrees with did *worse*. |
| Would knowing the quad help? | **Yes.** With hindsight quads, agreeing positions beat disagreeing ones by +6.2 to +6.6%/yr (3-month hold: q ≈ 0.05, stable in both halves). |

**Reading:** the trend-following leg of the process has real evidence behind
it. The longer lookbacks are better than the 3-month one. The range-entry leg
and the quad-filter leg add nothing detectable on this data. The quad would
help if it were known in time, but the market moves before the data
confirms it. Trend already captures what is knowable.

## Limits

- The universe is equity-heavy: 12 of 19 assets are equity indices, so the
  trend portfolio is partly a market-timing bet.
- No transaction costs. Range timing would add turnover, but it shows no gross
  edge to protect.
- Hedgeye's range also uses volume and implied volatility. This band uses price
  and realized volatility only.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.trend_range_study
```

It needs `outputs/quad_asset_performance/quad_signals.parquet`, which the quad
asset performance run writes. Outputs go to `outputs/trend_range_study/`:

- `trend_range_report.md`, including the current trend, band and quad stance
  for each asset
- `trend_range_summary.csv`
- `trend_range_snapshot.csv`
- the return series
