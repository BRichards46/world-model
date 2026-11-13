# GTT macro filters: thresholds, OR combinations and counts

Tests whether the legacy trigger levels (for example industrial production
above 1% YoY, unemployment no more than 0.1 pp above its 12-month average,
CPI below 2.5%) work better as GTT macro filters than the zero lines. It also
tests OR combinations and trigger counts of the viable growth filters.

## Method

- **Values:** each series is replayed release by release from the FRED
  vintage store, so a value is known only from its `realtime_start`. Filters
  are thresholds on those values.
- **GTT risk test:** from `gtt_risk_validation`, applied to the S&P 500.
  - It looks at months when the S&P is below trend (10-month or 200-day).
  - It asks whether months with macro danger saw more future volatility and
    drawdown than months where the filter stayed healthy.
  - There are 16 tests: two trends × four horizons × two outcomes.
  - A test passes only if it clears every robustness gate: a bootstrap lower
    bound above 0, stable halves, at least three stress eras, and positive
    results after leaving out any single stress era.
- **Timing backtest:** hold the S&P 500 total return unless the month-end
  10-month trend and the filter both warn; then hold 3-month T-bills (0%
  before September 1981). Switches cost 5 bp.
- **Windows:** 1962+, 1972+ (adds CPI, so every long-history filter shares one
  sample) and 2001+ (every input, but only about three stress eras, so few
  tests can pass there).
- **CFNAI family and real personal income:** excluded. Their point-in-time
  history starts in 2011–2012.

## Findings (November 2026)

Matched sample, September 1972 onward. Baselines: buy-and-hold 9.7% CAGR,
−51% max drawdown, Sharpe 0.47; trend only 8.8%, −24%, 0.53.

| Filter | Risk tests passed | CAGR | Max DD | Sharpe | Time in cash |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unemployment > 12m avg + 0.1 pp (your level) | 14/16 | 11.2% | −30% | 0.63 | 11% |
| Unemployment > 12m avg (zero line) | 14/16 | 10.5% | −30% | 0.58 | 13% |
| Industrial production YoY ≤ 0 | 13/16 | 10.7% | −33% | 0.58 | 7% |
| Industrial production YoY ≤ 1% (your level) | 14/16 | 10.0% | −30% | 0.55 | 10% |
| Industrial production OR unemployment | 16/16 | 10.6% | −30% | 0.60 | 13% |
| **Industrial production ≤ 0 OR unemployment + 0.1 pp** | **16/16** | **11.4%** | −30% | **0.64** | 12% |
| Industrial production ≤ 1% OR unemployment + 0.1 pp | 16/16 | 10.3% | −30% | 0.58 | 13% |
| Manufacturing hours < 12m avg | 16/16 | 9.9% | −24% | 0.57 | 16% |
| Industrial production OR hours | 16/16 | 9.8% | −24% | 0.56 | 17% |
| 2+ of 5 long-history zero lines | 16/16 | 10.2% | −28% | 0.60 | 17% |
| 2+ of your 5 long-history triggers | 14/16 | 11.0% | −30% | 0.64 | 15% |
| Payrolls YoY ≤ 1% (your level) | 10/16 | 10.0% | −46% | 0.54 | 9% |
| Housing starts / labor force ≤ −10% (your level) | 4/16 | 10.9% | −43% | 0.60 | 10% |
| CPI ≥ 2.5% (your level) | 0/16 | 10.4% | −28% | 0.60 | 20% |

- **Unemployment trend is the strongest single filter.**
  - The 0.1 pp buffer sits on a wide plateau: every level from −0.1 to
    +0.3 pp passes 15–16 of 16 tests from 1962.
  - It has the best Sharpe of the single filters.
  - It switches less often than the zero line (0.55 vs 0.81 per year).
- **Best combination: industrial production ≤ 0 OR unemployment more than
  0.1 pp above its 12-month average.** The dashboard's equity risk overlay
  has used this rule since November 2026, set by `unemployment_buffer_pp` in
  `config/macro_dashboard.yml`. It was previously the zero line.
  - It passes 16/16 tests from 1962 and from 1972.
  - Since 2001 it passes 11/16, close to the best combinations (12/16).
  - It has the best Sharpe in every window: 0.62 from 1962, 0.64 from 1972
    and 0.89 from 2001.
  - It switches less than the zero-line version (0.63 vs 0.88 times a year).
  - Keeping industrial production in the OR matters: unemployment with the
    buffer passes only 2/16 tests on its own since 2001.
  - Raising industrial production to 1% inside the OR makes it worse.
- **Industrial production works at any level from 0% to 3%.**
  - The 1% level passes slightly more tests from 1962 (14 vs 12) but slightly
    fewer from 2001 (10 vs 12).
  - It also earns less, so neither level is clearly better.
- **Payrolls and housing add little.**
  - Payrolls pass 8–10/16 and miss drawdowns (−46%).
  - Housing results jump around between neighboring levels, with no plateau.
- **CPI and core CPI fail the risk test at every level from 2% to 5%.**
  - Inflation does not separate trend breaks that follow through from
    whipsaws.
  - CPI's decent backtest comes from being above 2.5% for most of 1972–2001,
    when it simply behaved like trend following.
- **Retail sales:**
  - Nominal retail sales fail at every level from 0% to 5%.
  - Real retail sales alone pass 1/16.
  - Industrial production OR real retail sales (the original GTT growth
    signal) is the best combination since 2001: 12/16 tests, Sharpe 0.85,
    −21% drawdown. That rests on about three stress eras.
- **Hours and durable orders work alone but do not improve industrial
  production as OR terms.**
  - Hours raise time in cash to about 17% and switching to about 1.5 times a
    year, which makes GTT behave like plain trend following.
  - Durable orders pass 12–13/16 since 1999, but industrial production OR
    durable orders passes only 7–9/16.
- **AND rules (danger only when both warn) trade coverage for precision, and
  lose.**
  - Precision rises a little: industrial production AND unemployment is
    37% vs 33% for the OR rule.
  - Time in cash halves (6% vs 12%), but the rule misses episodes.
  - From 1972 it passes 10–11/16 tests (vs 16/16), with a deeper drawdown
    (−33% vs −30%) and a lower Sharpe (0.57 vs 0.64).
  - Industrial production AND unemployment behaves almost exactly like
    industrial production alone: unemployment rarely warns without it.
  - ANDs with real retail sales have the highest precision (51–56%) but are
    defensive only 4–7% of the time. They pass 0/16 and draw down 36%.
  - Industrial production AND durable orders is the one AND that holds up
    since 2001 (13/16, Sharpe 0.81). It still trails the OR rule (Sharpe
    0.89) and rests on about three stress eras.
  - Why: GTT already requires the price trend to confirm, so it is an AND
    with trend by construction. Its value comes from the macro side catching
    enough episodes, and adding more confirmation mostly removes good
    warnings.
- **Counts:**
  - "2 or more" of the long-history growth filters is robust (16/16).
  - "1 or more" is just an OR and spends too much time in cash.
  - Your 5-trigger count at 2 or more has the best Sharpe (0.64). Its strength
    comes from the growth members: its CPI member fails alone.
- **Differences among the good filters are small.** CAGR spreads about half a
  point and Sharpe 0.55–0.64, across roughly eight recessions. That is within
  noise. Robustness across levels and eras is the better guide.

## Current state (November 6, 2026)

| Filter | State | Why |
| --- | --- | --- |
| Every zero-line growth filter and OR combination | Clear | — |
| Your payrolls trigger | Failing | 0.56% YoY, below 1% |
| Your CPI and core CPI triggers | Failing | 3.7% and 2.6% |
| Your 5-trigger count (2 or more) | Warning | Payrolls and CPI only |

The count's warning comes from the two weakest members (payrolls passes 10/16
on its own; CPI fails every test). The S&P 500 is above its trend, so no GTT
variant is defensive.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.gtt_threshold_validation
```

- **Settings:** `config/gtt_threshold_validation.yml`.
- **Outputs:** `outputs/gtt_threshold_validation/`, including the full report
  with threshold sweeps.
- **Runtime:** about two minutes.
- **Rebuild only the report:** `--from-outputs`.
