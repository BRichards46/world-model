# Macro combination validation

## Question

Do economically meaningful combinations add information for ES/NQ beyond the
standalone macro relationships? The first screen tests industrial-production
deceleration with three measures of underlying inflation: core PCE, core CPI,
and median CPI.

## Definitions

- **Slowing production:** the point-in-time industrial-production velocity is below zero.
- **Elevated inflation:** the point-in-time inflation level is at or above its 60th percentile over the trailing ten years.
- **Rising inflation:** the point-in-time inflation velocity is above zero.
- **Additive pair:** a walk-forward model containing the market baseline, production state, and inflation state.
- **Interaction:** the simultaneous two-condition regime added on top of the market baseline and both components.

All features use the vintage available on the forecast date. Outcomes are
one-, three-, six-, and twelve-month ES/NQ forward returns, volatility, and
maximum drawdown. Results require at least 60 out-of-sample observations,
stable subperiod signs, positive out-of-sample MSE improvement, and a 10% FDR
threshold.

## Current result

The October 23, 2026 run produced 72 test variants:

- **13 additive pairs survived.** They are concentrated in future volatility
  and drawdown, with core PCE providing the broadest horizon coverage.
- **0 interaction terms survived.** Simultaneous slowing and inflation pressure
  did not add stable information after the two component signals were already
  included.
- **Inflation level was more economically useful than the latest acceleration.**
  Slowing production plus elevated core CPI/PCE showed the adverse descriptive
  pattern. Slowing production plus merely rising inflation did not consistently
  do so.
- **No robust return-direction regime survived.** The useful information is
  risk-oriented: volatility and drawdown, not a directional Equity forecast.

## Dashboard interpretation

`run_macro.sh` refreshes this study daily again (since November 10, 2026). The
dashboard's equity risk context uses its surviving elevated-inflation pairs.
When slowing production and an elevated core inflation measure are both on,
the outlook says so and quotes the study's volatility and drawdown
differences. See [`equity_risk_context.md`](equity_risk_context.md). The older
Assets tab display described below was removed in November 2026. When it is shown, an
**Aligned macro read** appears when a validated additive pair is currently
active and its active historical observations had worse risk outcomes. This is a confidence/hierarchy layer, not a new independent signal.
The evidence label explicitly says `interaction not incremental` so the display
does not overstate a stagflation multiplier that the tests did not validate.

## Run

The focused screen is refreshed automatically by the normal one-line runner:

```bash
~/PyBry/world-model/run_macro.sh
```

For a research-only rerun without rebuilding or serving the dashboard:

```bash
cd ~/PyBry/world-model
PYTHONPATH=src python -m world_model.cli.macro_regime_validation
```

Research output is written to `outputs/macro_regime_validation/`.
