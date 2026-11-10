# GTT beyond the S&P 500

Tests the growth-trend timing idea on the Russell 2000, Treasuries and the
dollar. Macro inputs are point-in-time. Each asset class's macro logic is fixed
in `config/gtt_cross_asset_validation.yml` before any results are seen. This
replaces the June 2026 futures grid, which picked the best of roughly 100
filter-and-trend combinations per contract and could not be verified.

## Method

- **Rule:** hold the asset unless its month-end 10-month trend is down and
  the macro "danger" filter is on; then hold cash. Switches cost 5 bp.
- **Cash:** T-bills, using the 1-year Treasury yield before 1981. Futures are
  excess returns with cash at 0.
- **Risk test:** the same 16 GTT risk tests and robustness gates as the S&P
  study (`docs/gtt_risk_validation.md`).
- **Macro logic:**
  - **Equity:** a growth warning confirms a falling trend. The main filter is
    the overlay rule: industrial production YoY ≤ 0 OR unemployment more than
    0.1 pp above its 12-month average.
  - **Bonds:** strong growth and inflation hurt bonds. Danger = growth healthy
    (the overlay rule is clear), CPI ≥ 2.5%, CPI ≥ 4%, or both healthy growth
    and CPI ≥ 2.5%.
  - **Dollar:** no settled theory, so both directions of growth and of
    inflation (CPI 2.5%) are tested.
- **Data:**
  - Russell 2000: Norgate `$RUTTR`, from 1980.
  - 10-year Treasury: a total-return series built from the daily 10-year
    yield (a constant-maturity par bond), from 1972 so CPI filters apply. It
    matches known years: 2022 −16%, 2008 +20%, 1994 −7%.
  - CSI ZN and DX futures: from 1996 and 1995, about three recessions each.

## Findings (November 2026)

| Asset | Best filter (risk tests passed) | GTT vs trend alone vs buy-and-hold | Verdict |
| --- | --- | --- | --- |
| Russell 2000, 1980+ | Industrial production ≤ 0: 6/16; overlay rule 4/16 | Overlay rule 11.4% CAGR, −36% max DD, Sharpe 0.47; trend 7.8%, −32%, 0.30; buy-and-hold 10.5%, −53%, 0.40 | Works, weaker evidence |
| S&P 500, same window | Overlay rule 14/16 | 13.1%, −30%, 0.68; trend 10.0%, −24%, 0.54; buy-and-hold 11.3%, −51%, 0.51 | Reference |
| 10-year Treasury, 1972+ | CPI ≥ 4%: 7/16; healthy growth: 0/16 | Every filter 6.2–6.4% CAGR; trend alone 6.1%, −15% DD; buy-and-hold 6.2%, −26% | Trend alone is enough |
| ZN futures, 1996+ | 0–1/16 | 2.6–3.4% vs buy-and-hold 3.5% | No macro edge |
| DX futures, 1995+ | 0/16 for all four | Best (CPI < 2.5% as danger) Sharpe 0.27 vs trend 0.21 | No macro edge |

- **Russell 2000: the GTT effect has the right sign in all 16 tests, but it is
  noisier than the S&P's.**
  - Incremental lift over trend is positive everywhere.
  - Precision is 18–33% against a 10% base rate.
  - Every test spans nine stress eras.
  - Most failures are bootstrap lower bounds just below zero.
  - Trend following alone does badly on small caps: 2 switches a year and a
    lower return than buy-and-hold. The macro filter fixes that by ignoring
    most of those trend breaks.
  - The S&P remains the better place to apply the rule.
  - The stricter AND version (industrial production AND unemployment) is
    worse on both: Russell 2/16, Sharpe 0.42; S&P 9/16, Sharpe 0.62.
- **Treasuries: macro filters add nothing reliable.**
  - Healthy growth does not separate bond downtrends that follow through
    (0/16, lift slightly negative).
  - High inflation helps a little only in the 1970s–80s sample (CPI ≥ 4%:
    7/16) and fails on ZN since 1996.
  - Trend following alone keeps the bond return while cutting the drawdown
    from −26% to −15%.
- **Dollar: nothing passes.**
  - The June DX handoff's "inflation high is dollar-bullish" choice has the
    right sign: CPI below 2.5% as danger gives the best Sharpe, 0.27 vs 0.21.
    But it passes 0/16 risk tests and was one of four hypotheses.
  - It also rests on about three recessions.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.gtt_cross_asset_validation
```

- **Inputs:** reuses the point-in-time values saved by the threshold study
  (`outputs/gtt_threshold_validation/point_in_time_values.parquet`) and
  rebuilds them if missing.
- **Outputs:** `outputs/gtt_cross_asset_validation/`.
- **Runtime:** about 15 seconds.
