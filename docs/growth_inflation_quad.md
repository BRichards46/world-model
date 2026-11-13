# Growth × Inflation Quads

A rate-of-change regime model with a base-effect look-ahead. It answers a
different question from the level/velocity monitor: not "is the economy strong
or weak?" but "are year-over-year growth and inflation accelerating or
decelerating, and what will base effects do to them over the next few
quarters?"

## Definitions

- **Growth:** real GDP (`GDPC1`) year-over-year growth for each quarter.
- **Inflation:** the quarterly average of monthly headline CPI (`CPIAUCSL`)
  year-over-year rates.
- **Quad:** set by the direction of change of both rates versus the prior
  quarter.

| Quad | Growth YoY | Inflation YoY |
| --- | --- | --- |
| 1 | accelerating | decelerating |
| 2 | accelerating | accelerating |
| 3 | decelerating | accelerating |
| 4 | decelerating | decelerating |

## Base effects and hurdles

YoY growth in quarter `t` accelerates exactly when the quarter's annualized
growth beats the annualized growth of the same quarter a year earlier. That
earlier rate is the **growth hurdle**. For inflation, the **inflation hurdle** is
the constant monthly CPI change over the quarter's unreported months that would
leave the quarterly-average YoY rate unchanged. The dashboard shows each
upcoming quarter's hurdle against the base-case assumption.

## Projection models

Configured under `models:` in `config/growth_inflation_quad.yml`. The code is in
`world_model.features.quad_forecast_models`.

- **Current-quarter growth:** GDPNow (`GDPNOW`) when available. Otherwise a
  bridge regression from monthly real PCE, industrial production and payrolls,
  each extended at its recent pace through the quarter and refit monthly.
- **Later-quarter growth:** a ridge regression of k-quarter-ahead annualized
  growth on the nowcast, trailing trend, initial claims YoY, the 10y–3m curve
  and building permits YoY.
- **Inflation:** the six-month core CPI run-rate plus gasoline pass-through
  (`GASREGW`) for months with gasoline information: month to date, and next
  month's carry-over at a flat price. Later months use the core run-rate.
- **Probabilities:** each quad's probability comes from the empirical
  distribution of the model's own past forecast errors at the same horizon.
  Only quarters whose outcome was known by the as-of date are used. Growth and
  inflation are treated as independent.
- **Scenario grid:** growth ±1.0 pp and inflation ±0.1 pp per month around the
  base case. It is kept for reference but proved overconfident: one to three
  quarters ahead it scores worse than uniform 25% odds.
- Unpublished months (October 2025 CPI) are log-interpolated and flagged.
- Gasoline, the curve and claims use latest values with a publication lag.
  They are market prices or weekly series with negligible revisions, and their
  vintage history only starts in 2009–2014.

### Forecast research

`world_model.cli.quad_forecast_research` compares the variants walk-forward. It
writes to `outputs/quad_forecast_research/`. Results as of November 2026
(1998–2026):

| Horizon | Baseline quad hit | Production model | Gain (95% CI, clustered by quarter) |
| --- | ---: | ---: | --- |
| Current quarter | 51% | 59% | +7.8 pp (+1.7, +13.9) |
| 1 quarter ahead | 43% | 46% | +3.5 pp (−1.8, +8.8) |
| 2 quarters ahead | 41% | 45% | +3.8 pp (−2.9, +10.9) |

The bridge nowcast mainly improves growth direction before GDPNow existed.
Gasoline pass-through lifts current-quarter inflation direction from 81% to 89%.
Leading indicators help growth direction only two or more quarters ahead.

All inputs come from the point-in-time FRED vintage store. Settings live in
`config/growth_inflation_quad.yml`.

## Walk-forward check

Every month-end since 1998, the engine rebuilds the projection using only the
vintages known on that date and scores it against the latest revised quads. It
compares against two baselines: the last realized quad repeating, and always
picking the most common quad. Results are written to
`outputs/growth_inflation_quad_backtest.parquet` and summarized on the
dashboard.

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.growth_inflation_quad
```

`run_macro.sh` runs it automatically. Outputs:

- `outputs/growth_inflation_quad_snapshot.json`
- `outputs/growth_inflation_quad_backtest.parquet`
- `sites/macro-dashboard/dist/data/growth_inflation_quad.json`

## Asset performance by quad

`world_model.cli.quad_asset_performance` (config
`config/quad_asset_performance.yml`) tests whether quads separate forward asset
returns. It uses four monthly signals from the walk-forward: last realized
quad, projected current quarter, projected next quarter, and a hindsight
(non-tradeable) current-quarter quad as an upper bound.

- Assets: Norgate total-return indices (S&P 500, Russell 2000, ten S&P 500
  sectors, each also measured against the S&P 500), TLT from 2002, and CSI
  futures groups (Treasuries, dollar, precious metals, energy, industrial
  metals; normalized price change).
- Forward 1- and 3-month returns start one trading day after the signal date.
- Significance: a circular rotation of the quad sequence against returns,
  which preserves autocorrelation and the overlap of multi-month returns.
  Benjamini-Hochberg adjustment is applied within each signal and horizon.
- Evidence tiers: *robust* = q ≤ 0.10. *Suggestive* = q ≤ 0.25. Both also
  require the same sign in both halves of the sample, on medians, and when
  2008–09 and 2020 are excluded.

Findings as of November 2026 (1998–2026 sample):

- Only the hindsight quad clears the joint test (2 of 29 assets, 3-month
  horizon). Its clearest effect is Quad 3: the S&P 500 trails its average by
  about 11% annualized, while utilities, staples and health care beat the
  S&P 500 by 7–11%.
- No tradeable signal clears the joint test. A few quad × asset results are
  suggestive.
- A projected Quad 4 for the next quarter was followed by small caps,
  materials and industrial metals outperforming, and long Treasuries
  underperforming. That is the opposite of the common Quad 4 playbook
  (long bonds and defensives), consistent with markets pricing the slowdown
  before it shows up in the data.

With the upgraded forecasts, projected-next-quarter Quad 4 → materials beating
the S&P 500 passes the joint test (q ≈ 0.09). Related cyclical tilts after
projected Quad 4 (small caps, consumer discretionary) are suggestive.

### Follow vs fade the playbook

`world_model.states.quad_playbook_test` builds a risk-balanced long/short
portfolio of each quad's leaders minus laggards. *Follow* holds it; *fade* (a
"recovery" view of a stale quad) holds the opposite. There are two playbooks:
a textbook one fixed in config, and one learned only from quarters whose quad
was known at the time.

- With hindsight quads, following the textbook playbook earns about
  +3.3%/yr (3-month hold, t ≈ 2.9, q ≈ 0.05), consistent in both halves of the
  sample. The playbook is right when the quad is known.
- With every tradeable signal, neither following nor fading has a
  statistically detectable edge (q ≥ 0.45). Fading the stale realized quad
  earned +0.5 to +1.7%/yr, all of it before about 2012.
- Following only when the current-quarter probability is at least 70% did not
  help.

`docs/trend_range_study.md` tests the quad as a filter on intermediate-term
trend. With tradeable quads it adds nothing; with hindsight quads it adds
about +6%/yr.

Outputs are in `outputs/quad_asset_performance/`, including
`quad_asset_performance_report.md`. The Quads tab summarizes the current
signals.

## Payroll quad

`world_model.states.labor_quad` (config `config/labor_quad.yml`) builds the
same quad with nonfarm payrolls (`PAYEMS`) as the growth leg:
- **Growth:** year-over-year growth of each quarter's average payroll level.
  Unreported months grow at the average change of the last three months.
- **Inflation:** the GDP quad's own inflation leg and forecast, so the two quads
  differ only in growth.

It runs after `growth_inflation_quad` and `quad_asset_performance`:

```bash
PYTHONPATH=src python3 -m world_model.cli.labor_quad
```

### Why payrolls (tested 2026-11-13)

GDP makes a poor growth leg:
- **Revisions cap accuracy.** Even a perfect forecast of the first GDP print
  gets the growth direction right only 86% of the time against revised data. A
  further 7 points are lost because the year-ago quarter (the hurdle) is revised
  too.
- **It's mostly base effects.** When the forecast is within 1 pp (annualized)
  of the hurdle, about a third of cases, the direction is a coin flip. The
  leading model adds nothing one quarter ahead (61%, the same as trend growth).
- **It barely persists.** Whether GDP growth is speeding up this quarter says
  little about next quarter (55% the same). For payrolls it's 76%.
- **It says nothing about the Fed.** Over 1990–2026, the fed funds rate rose
  0.42 pp more over the next two quarters after payroll growth sped up than
  after it slowed (t 3.3). After GDP growth sped up, it rose only 0.07 pp more
  (t 0.5).

Walk-forward, 1998–2026, point-in-time vintages, each quad scored against its
own revised data on the same month-ends:

| Horizon | Payroll quad | GDP quad | Payroll growth direction | GDP growth direction |
|---|---:|---:|---:|---:|
| Current quarter | **72%** | 59% | **82%** | 68% |
| Next quarter | **53%** | 46% | **70%** | 61% |
| Two quarters ahead | 46% | 45% | 66% | 63% |

Last quarter's payroll quad is known within about a week of the quarter ending.
Unemployment and a payroll/unemployment/industrial production composite gave
similar current-quarter accuracy. Payrolls were best one quarter ahead.

**Assets.** The existing asset test, run on the payroll quad's signals, finds
what the GDP quad found only with hindsight. After a *projected* current-quarter
Quad 3 (3-month returns, annualized vs all months):

| Asset | Spread | Early half | Late half | Evidence |
|---|---:|---:|---:|---|
| Russell 2000 | −15.1% | −16.9% | −14.3% | robust |
| Consumer discretionary | −13.6% | −21.0% | −9.0% | robust |
| Materials | −12.8% | −19.6% | −8.2% | robust |
| S&P 500 | −10.2% | −14.3% | −8.4% | robust |
| Utilities vs S&P 500 | +9.9% | +9.9% | +10.6% | robust |
| Financials | −16.2% | −26.5% | −10.1% | suggestive |

The unemployment version finds the same Quad 3 pattern. Projected Quad 4 has
utilities lagging the S&P 500 by 11% (robust). The full four-quad playbook is
still not significant: the edge is in Quad 3. Three growth legs were tried, so
allow for some selection.

**The Fed path.** After a projected payroll Quad 2, the fed funds rate rose
about 0.4 pp over the next 12 months; after Quad 4 it fell about 0.5 pp. The
quad explains about 5% of the variation (p ≈ 0.2–0.3 on the rotation test). No
quad, GDP or payroll, predicted the 2-year yield's next 3-month change: the
market already prices the Fed's leaning. Since 2008 the Fed has followed core
PCE against its 2% target more than either growth measure.

**Inflation path.** Using the bottom-up core CPI nowcast (docs/core_cpi.md) as
the quad's inflation run-rate did not help: the inflation-direction hit rate
changed by −1.6 to +1.1 points over 0–2 quarters ahead, none significant.

Outputs:
- `outputs/labor_quad_snapshot.json` and
  `sites/macro-dashboard/dist/data/labor_quad.json`: quarters, probabilities,
  walk-forward accuracy, Fed path by quad and asset evidence.
- `outputs/labor_quad_backtest.parquet`: the walk-forward.
- `outputs/quad_asset_performance_labor/`: the asset and playbook tests.

The payroll quad leads the dashboard's regime calls: the outlook headline, its
"Where the regime model points" quarters and the quad strip all use it. The GDP
quad's call appears beside them only where it disagrees. If the payroll quad is
missing, the GDP quad leads as before. The Quads tab shows the payroll quad
above the GDP quad's full detail.

## Limits

This is descriptive base-effect arithmetic under stated assumptions, not a
return forecast. The asset results are historical conditional averages from
about 28 years of monthly data, not a validated trading rule.
