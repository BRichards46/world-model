# Equity risk context

The equity risk overlay decides the stance: stay invested, or go defensive
when the S&P is below its 10-month average and a macro danger signal agrees.
That stance is the only validated timing rule. This page adds how much risk
the current state has historically carried, so "invested" doesn't read as
"safe".

## Method

- **States:** every month-end from September 1972 is classified, using macro
  values as known on that date (FRED vintages, release by release):
  - **Invested, legacy triggers clear:** S&P above trend, overlay macro clear,
    and fewer than 2 of the 5 legacy triggers warning.
  - **Invested, legacy trigger count on:** the same, but 2 or more warn.
  - **Invested, macro warning without a trend break.**
  - **Trend break without a macro warning.**
  - **Defensive:** trend break and macro warning.
- **Legacy triggers:** industrial production YoY ≤ 1%, payrolls YoY ≤ 1%,
  unemployment more than 0.1 pp above its 12-month average, housing starts
  per worker YoY ≤ −10%, CPI YoY ≥ 2.5%.
- **Outcomes:** the worst drawdown and the realized volatility over the next
  63 trading days. "Worst decile" means the worst 10% of all months: a
  drawdown of about −13.5% or worse, or volatility above about 23%.
- **Tiers:** a state's worst-decile drawdown rate against the 10% base rate.
  Under 0.7× is below average, 0.7–1.4× is average, 1.4–2.5× is elevated, and
  above 2.5× is high.
- **"Above normal for this stance"** is flagged when an invested state's rate
  is more than 1.5× that of invested months with the triggers clear, or when
  the production-and-inflation read below is on.
- **Episodes:** bad months no more than three months apart count as one
  episode, so a single crash is not counted three times.

## Production and inflation

`docs/macro_regime_validation.md` found that slowing industrial production
(3-month growth falling) together with elevated core inflation (core PCE or
core CPI at or above the 60th percentile of its 10-year range) was followed
by higher volatility and deeper drawdowns on ES/NQ. The two signals add
information separately; having both at once adds nothing more. When the
production read and an elevated measure are both on, the dashboard shows the
study's effect sizes: for core PCE since 2005, 3-month volatility of 12.7%
against 11.4%.

## Findings (November 2026)

| State | Months | Worst-decile drawdown followed | Worst-decile volatility followed | Tier |
| --- | ---: | ---: | ---: | --- |
| Invested, legacy triggers clear | 261 | 3% | 5% | Below average |
| Invested, legacy count on | 76 | 12% | 9% | Average |
| Invested, macro warning | 135 | 7% | 3% | Below average |
| Trend break, no macro warning | 100 | 11% | 17% | Average |
| Defensive | 75 | 36% | 32% | High |

- **Today:** invested with the legacy count on (payrolls and CPI), and
  slowing production with elevated core PCE. The dashboard shows "stay
  invested, but risk is above normal for this stance".
- **The legacy-count tier rests on five episodes:** late 1979, 1987, mid-1990,
  2011 and January 2025.
- **This is descriptive.** The legacy count passed 14 of 16 GTT risk tests
  only as a filter after a trend break. When the S&P is above trend, it has
  not been tested as a timing signal.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.equity_risk_context
```

- **Settings:** `equity_risk_context` in `config/macro_dashboard.yml`.
- **Output:** `sites/macro-dashboard/dist/data/risk_context.json`. The outlook
  and the overlay panel read it.
- **Inputs:** `snapshot.json` and the production-and-inflation summary from
  `world_model.cli.macro_regime_validation`. `run_macro.sh` runs both first.
- **Runtime:** about 10 seconds.
