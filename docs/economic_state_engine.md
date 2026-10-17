# Economic State Engine

The economic state engine builds a point-in-time macro state vector from the
local FRED/ALFRED store.

It is descriptive first. The output is not a return forecast. It is a structured
snapshot of the economic backdrop that can later be combined with the market
state engine.

## Data Source

The engine reads the point-in-time FRED vintage files:

```text
/Users/brichards/Documents/data/fred/vintage/{series_id}.parquet
```

Each value is processed on its `realtime_start` date, which means historical
states are based on data available at the time. The engine does not use the
latest revised history to reconstruct past states.

The project currently consumes the local FRED store. It does not download new
FRED series directly. If a new series is needed, add it through the existing
data-management updater, then include it in:

```text
config/economic_state.yml
```

## State Vector

The default state vector includes six component scores:

- `growth_score`
- `labor_score`
- `inflation_score`
- `liquidity_score`
- `credit_score`
- `housing_score`

It also computes three higher-level axes:

- `activity_score`: growth, labor, and housing
- `inflation_pressure_score`: inflation component
- `liquidity_credit_score`: liquidity and credit impulse

Scores are rolling z-scores squashed with `tanh`, so they usually live between
`-1` and `+1`. Positive inflation means higher inflation pressure, not a
favorable condition. The composite score subtracts inflation pressure.

## Regime Labels

The regime label is based mainly on activity and inflation pressure:

- `goldilocks_expansion`: activity strong, inflation pressure low
- `reflationary_expansion`: activity strong, inflation pressure high
- `stagflation_pressure`: activity weak, inflation pressure high
- `disinflationary_slowdown`: activity weak, inflation pressure low or falling
- `mixed_cycle`: no clean quadrant

The engine adds liquidity, housing, and labor qualifiers when those dimensions
are extreme enough, for example:

```text
mixed_cycle_liquidity_tight
```

## Run

```bash
PYTHONPATH=src python3 -m world_model.cli.economic_state \
  --history-out outputs/economic_state_history.parquet \
  --report-out outputs/economic_state_report.md
```

Default report:

```text
outputs/economic_state_report.md
```

## Candidate Missing Series

The report lists useful FRED series that are not currently present in the
point-in-time store. The most important next additions are likely:

- policy/rates: `FEDFUNDS`, `DFF`, `GS10`, `T10Y2Y`, `T10Y3M`
- credit spreads: `BAMLH0A0HYM2`, `BAMLC0A0CM`, `BAMLC0A4CBBB`
- financial conditions: `NFCI`, `ANFCI`, `STLFSI4`
- housing affordability: `MORTGAGE30US`
- bank credit and debt service: `BUSLOANS`, `TDSP`
- household savings: `PSAVERT`

Those are not required for the first pass, but they would make the economic
state engine more complete.
