# Bottom-up core CPI nowcast

`world_model.states.core_cpi` builds core CPI from blocks that together make up
all of it, using only data published by the day before each CPI release.

| Group | Blocks | Share of core (Dec 2025) |
|---|---|---:|
| Shelter | Owners' equivalent rent, rent, lodging away from home | 44% |
| Core goods | New vehicles, used cars and trucks, apparel, other core goods | 24% |
| Services ex shelter | Medical care services, transportation services, other services | 32% |

"Other core goods" and "other services" are what is left of core goods and core
services after their named parts, so the blocks add up to core. With the actual
block changes, the sum reproduces core to within 0.006 pp.

**Forecast.** Each block's adjusted monthly change is an equal mix of two
averages of its own recent changes:
- **Adaptive:** the average over the window (1, 3, 6 or 12 months) that would
  have forecast that block best over the previous three years. It is chosen
  again at each release.
- **Anchor:** the 12-month average.

**Weights.** Each block's share of core comes from the BLS December relative
importance. Year *y*'s weights count once the January *y*+1 CPI report that
publishes them is out.

**Data.** Every CPI series is a FRED vintage read as published on the cutoff
date. Component vintages start March 2011. Relative importance comes from the
BLS store (`data_roots.yml` key `bls`).

```bash
PYTHONPATH=src python3 -m world_model.cli.core_cpi
```

Outputs, in `outputs/core_cpi/`:
- `core_cpi_backtest.parquet`: one row per release, with the forecast, first
  print, each group's contribution and each block's chosen window.
- `core_cpi_summary.json`: accuracy, the comparison with the market and the
  Cleveland Fed, validation, and the live nowcast.

## Results (run 2026-11-12, releases June 2011 to October 2026)

Core CPI against other forecasts (same releases, average miss in pp):

| Sample | Bottom-up | Previous model | Market (Kalshi) | Cleveland Fed | Naive (AR) |
|---|---:|---:|---:|---:|---:|
| 2019-12 to 2026-10 (81) | **0.124** | 0.137 | n/a | 0.134 | 0.136 |
| 2022-06 to 2026-10 (50) | 0.089 | 0.098 | **0.078** | 0.093 | n/a |

It is validated: it beats the autoregressive benchmark over the full 81 releases
and in both halves (by 0.019 early, 0.007 late). It beats our previous core
model and the Cleveland Fed nowcast. It still trails the market.

Using it as the core block of the headline build cuts the headline miss from
0.100 to 0.091 (161 releases). Over the market sample, headline now misses by
0.102, against the market's 0.089.

**What decides the miss.** With each block's actual change known in turn, the
core miss (0.094 for the adaptive part alone) falls most for:

| Block known exactly | Core miss |
|---|---:|
| Transportation services | 0.077 |
| Used cars and trucks | 0.083 |
| Other core goods | 0.089 |
| Apparel | 0.090 |
| Lodging away from home | 0.090 |
| Owners' equivalent rent | 0.092 |

Shelter is already well forecast from its own history. Most of the remaining
miss is in volatile blocks with no driver in our data.

**What didn't help:**
- **Calendar-month corrections:** core runs hot in January, but correcting for
  it raised the miss (0.096 to 0.108).
- **The adaptive window alone, or a longer selection period:** each did worse
  than the mix. The adaptive part catches turns (2019–22); the anchor wins in
  calm periods (2023 on).
- **Jet fuel for airfares:** it trims the airfare miss from 2.83 to 2.77, which
  is negligible at airfares' 1% weight.
- **Splitting transportation services** into insurance, maintenance and
  airfares: each part is noisier than the whole.
- **Car insurance momentum** (tested 2026-11-12): see below.

### Car insurance and airfares (market sample, June 2022 – October 2026)

**Car insurance.** We rebuilt adjusted car insurance CPI point-in-time: the
unadjusted index (`CUUR0000SETE`, with its release dates) divided by each
year's projected seasonal factor from the BLS factor files, dated from the
factors' publication. For 2026 it matches BLS's adjusted series exactly. Earlier
years use the projected factors, not the later revisions. The series starts in
2012.

Its monthly changes carry on a little (autocorrelation 0.3), and recent
averages beat the 12-month average:

| Insurance forecast | Average miss (pp) |
|---|---:|
| Last 3 months | 0.56 |
| Last 6 months | 0.57 |
| Fitted autoregression | 0.60 |
| Last 12 months | 0.63 |
| Last month | 0.67 |

Insurance is only about 3.4% of core, so this barely moves core:

| Core CPI miss (50 releases) | |
|---|---:|
| This model | 0.089 |
| With insurance from 6-month momentum | 0.091 |
| With insurance known exactly | 0.086 |
| Kalshi | 0.078 |

Even a perfect insurance forecast closes only about a quarter of the gap to the
market, so insurance isn't modelled separately.

**Airfares.** About half the transportation services miss comes from airfares
(about 1.1% of core). Knowing them exactly would bring core to 0.078, level with
the market. Nothing we have predicts them:
- **Jet fuel:** trims the airfare miss only from 2.83 to 2.77.
- **Airline producer prices** (`PCU481111481111`): correlation with the airfare
  CPI is 0.28 in the same month and 0.15 the month before. The PPI also came out
  before CPI for only 10 of 65 releases since mid-2022.

The market's edge on core is spread across noisy blocks (airfares, lodging,
apparel, other goods), not concentrated in one that a single dataset would fix.
On the market sample, knowing used cars exactly doesn't help at all (0.087 to
0.088). The used-car gain in the full sample comes from 2020–22.

## On the dashboard

`core_cpi` runs after `macro_release_forecast` and before `energy_cpi`. It
replaces the core row in `release_forecasts.json` (variant `bottom_up`). Its
50% and 80% ranges come from its own last 60 walk-forward misses.

`energy_cpi` then uses this nowcast as the headline's core block, and
`release_benchmarks` scores core CPI on it against the market and the Cleveland
Fed.

## Next refinements

These need data we don't have yet:
- **Airfares:** the largest single lever since 2022, but airfare data such as
  TSA passenger counts or fare indexes is untested, and the airfare CPI itself is
  very noisy.
- **Used cars:** the Manheim used-vehicle value index leads the CPI by about two
  months. It's widely watched, and since 2022 even a perfect used-car forecast
  wouldn't have improved core.
- **Car insurance:** state rate filings. A perfect insurance forecast is worth
  only 0.003 on core since 2022.
- **Shelter turns:** the BLS new-tenant rent index, or Zillow/Apartment List
  asking rents.
- **Lodging:** hotel rates (STR).
