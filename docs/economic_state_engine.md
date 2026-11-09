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

Scores are rolling z-scores over a trailing 10-calendar-year comparison window,
with at least five years of usable history required. They are squashed with
`tanh`, so they usually live between `-1` and `+1`. Positive inflation means
higher inflation pressure, not a favorable condition. The composite score
subtracts inflation pressure.

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

## Candidate Expansion Series

The report lists useful FRED series that are not currently present in the
point-in-time store. The dashboard source audit added or confirmed `PCEC96`,
`DFF`, `T10Y3M`, `BAMLH0A0HYM2`, and `MORTGAGE30US`. Useful later additions
include:

- policy/rates: `FEDFUNDS`, `GS10`, `T10Y2Y`
- credit spreads: `BAMLC0A0CM`, `BAMLC0A4CBBB`
- financial conditions: `NFCI`, `ANFCI`, `STLFSI4`
- bank credit and debt service: `BUSLOANS`, `TDSP`
- household savings: `PSAVERT`

Those are not required for the first pass, but they would make the economic
state engine more complete.

## Macro Dashboard Plan

The next iteration should repurpose the economic state engine into a transparent
macro monitor. The first dashboard should emphasize individual indicators and
their changes rather than introduce another composite model.

The main questions are:

- Where is each indicator relative to a relevant historical baseline?
- Is it accelerating or decelerating?
- Is that movement persistent or just one noisy release?
- How fresh is the reading?
- Do recent-history and long-history comparisons tell different stories?

The dashboard remains descriptive first. Composite scores, regime labels, and
market implications should be layered on only after the individual readings are
clear and trustworthy.

### Core Indicator Set

The first visible set should contain 13 familiar primary rows. ISM is deferred
until an authorized source is available. The mortgage rate is supporting
context for the housing row rather than an independently scored row.
Specialized composites such as WEI and CFNAI remain useful as secondary
confirmations, but should not lead the dashboard.

#### Activity and labor

Display these in this order:

1. Payroll growth
2. Unemployment rate
3. Real consumer spending or real retail sales
4. Industrial production
5. Real GDP growth

Payrolls should emphasize recent momentum rather than only year-over-year
growth. Real consumer spending is the preferred broad demand measure; real
retail sales can be used as a fallback. ISM Manufacturing and Services are
sidelined for the first version because ISM does not provide a suitable free API
and the project has no licensed ISM source.

#### Inflation

- Core PCE and Core CPI, led by the familiar month-over-month and year-over-year rates
- One robust underlying-inflation measure, such as trimmed-mean or median
  inflation
- One market- or survey-based inflation-expectations measure

Avoid presenting several closely related CPI variants as independent evidence.
Headline inflation can remain visible as shock context without receiving equal
weight in an underlying-inflation composite.

#### Policy and financial conditions

- Real policy rate
- Treasury yield-curve slope
- High-yield option-adjusted spread

Adjusted financial conditions remain a desired secondary row. The initial
datafeed series is `ANFCI`. It must be delivered through the same point-in-time
FRED pipeline as the other indicators; the dashboard must not add a separate
download path for it.

#### Housing

- Building permits
- 30-year mortgage rate as affordability context

Housing starts, initial claims, temporary-help employment, WEI, CFNAI, and
additional credit measures can appear in drill-downs or an `early signals`
section.

### Indicator Recipe Specification

Each primary row has an explicit recipe. The dashboard may show additional raw
values for familiarity, but the `level metric` is the value ranked against
history and the `velocity metric` is the value ranked to describe acceleration
or deceleration.

All growth calculations use point-in-time observations known on the release
date. Percentage growth should use log differences when every observation is
positive. Annualized changes are expressed as percentages, while changes in
rates, diffusion indexes, and spreads are expressed in percentage points or
basis points as appropriate.

| Order | Dashboard row | Primary source | Displayed value | Level metric | Velocity metric | Higher means |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Payroll growth | `PAYEMS` | Latest monthly change and 3-month average change, thousands | Annualized log growth of the latest 3-month average level versus the preceding 3-month average level | Current 3m/3m annualized growth minus the same measure calculated three observations earlier | Stronger labor demand |
| 2 | Unemployment rate | `UNRATE` | Latest rate and 3-month average rate | Latest 3-month average unemployment rate | Latest 3-month average minus the preceding 3-month average, percentage points | More labor-market slack |
| 3 | Real consumer spending | `PCEC96`; temporary fallback `RRSFS` | Latest 3m/3m annualized real growth | Annualized log growth of the latest 3-month average level versus the preceding 3-month average level | Current 3m/3m annualized growth minus the same measure calculated three observations earlier | Stronger real demand |
| 4 | Industrial production | `INDPRO` | Latest 3m/3m annualized growth | Annualized log growth of the latest 3-month average level versus the preceding 3-month average level | Current 3m/3m annualized growth minus the same measure calculated three observations earlier | Stronger industrial activity |
| 5 | Real GDP growth | `GDPC1` | Latest QoQ annualized growth; YoY growth as context | Latest quarter-over-quarter annualized log growth | Current QoQ annualized growth minus the previous quarter's QoQ annualized growth | Stronger aggregate activity |
| 6 | Core PCE inflation | `PCEPILFE` | Latest MoM, YoY, and 3-month annualized inflation | Latest YoY inflation | Latest MoM rate minus the prior month's MoM rate, percentage points | Hotter underlying inflation |
| 7 | Core CPI inflation | `CPILFESL` | Latest MoM, YoY, and 3-month annualized inflation | Latest YoY inflation | Latest MoM rate minus the prior month's MoM rate, percentage points | Hotter underlying inflation |
| 8 | Median CPI | `MEDCPIM158SFRBCLE` | Latest published monthly annualized rate | Latest published monthly annualized rate | Latest rate minus the prior month's published rate, percentage points | Hotter broad inflation |
| 9 | Five-year inflation expectations | `EXPINF5YR` | Latest expected inflation rate and 3-month average | Latest 3-month average expected inflation rate | Latest 3-month average minus the preceding 3-month average, percentage points | Higher expected inflation |
| 10 | Real policy rate | `DFF` minus `PCEPILFE` YoY inflation | Latest nominal policy rate, inflation rate, and ex-post real rate | Latest 3-month average effective federal funds rate minus latest available 12-month core PCE inflation | Current real policy rate minus its value three monthly observations earlier, basis points | Tighter realized policy stance |
| 11 | Treasury curve | `T10Y3M`; `DGS10` and `DGS3MO` as classification context | Latest and 20-business-day average 10-year minus 3-month spread; latest yield legs and canonical bull/bear/twist classification | Latest 20-business-day average spread | Latest 20-day average minus the 20-day average ending 63 business days earlier, basis points; the same comparison for each yield leg determines the curve-move type | A steeper curve; not inherently better or worse |
| 12 | High-yield spread | `BAMLH0A0HYM2` | Latest and 20-business-day average option-adjusted spread | Latest 20-business-day average spread | Latest 20-day average minus the 20-day average ending 63 business days earlier, basis points | Greater credit stress |
| 13 | Adjusted financial conditions | `ANFCI` | Latest and 4-week average index | Latest 4-week average index | Latest 4-week average minus the 4-week average ending 13 weeks earlier | Tighter-than-economic-baseline conditions |
| 14 | Building permits | `PERMIT` | Latest annualized level, 3-month average, and 3-month-average YoY growth | Year-over-year log growth of the latest 3-month average level | Current 3-month-average YoY growth minus the same measure three observations earlier | Stronger forward housing activity |

Mortgage affordability context uses `MORTGAGE30US`. Display the latest 4-week
average mortgage rate and its change from the 4-week average ending 13 weeks
earlier in basis points. It does not receive an independent core-row score in
the first version.

Housing starts intensity uses `HOUST / CLF16OV` as a secondary Housing row.
Its displayed level is the three-month average annualized starts per 1,000
labor-force participants; velocity is the year-over-year percentage change in
that ratio. It receives its own historical percentiles and momentum streak but
is excluded from core-row coverage counts and future composites.

#### Source status

The machine-readable first-version source manifest is:

```text
config/macro_dashboard.yml
```

The complete dependency set is declared once in `required_fred_series_ids` in
the manifest. Dashboard rows refer to that universe; they do not define
individual download routes.

```text
PAYEMS
UNRATE
PCEC96
INDPRO
GDPC1
PCEPILFE
MEDCPIM158SFRBCLE
EXPINF5YR
DFF
T10Y3M
DGS10
DGS3MO
BAMLH0A0HYM2
ANFCI
PERMIT
MORTGAGE30US
HOUST
CLF16OV
```

Only three series were missing from the existing upstream datafeed catalog:

```text
PCEC96
ANFCI
MORTGAGE30US
```

They are declared separately in `datafeed_additions` and are now wired through
the standard point-in-time pipeline. All three have been bootstrapped into the
local store. The ANFCI vintage file contains availability history from 2011 and
observations back to 1971, which is sufficient for the dashboard's trailing
10-year comparison window.

There is one explicit evidence limitation. The local `BAMLH0A0HYM2`
point-in-time file begins in 2023, which is too short for a 10-year percentile.
The datafeed must either obtain sufficient standard point-in-time coverage or
the row must report `insufficient history`. The dashboard must not use a
separate latest-history fallback.

The first version deliberately defers ISM Manufacturing and Services because no
authorized API or licensed local source is configured.

#### Recipe conventions

For a positive monthly level series `x`, 3m/3m annualized growth is:

```text
100 * (exp(4 * (log(mean(x[t-2:t])) - log(mean(x[t-5:t-3])))) - 1)
```

Six-month annualized inflation from a price index `p` is:

```text
100 * (exp(2 * (log(p[t]) - log(p[t-6]))) - 1)
```

Quarter-over-quarter annualized real GDP growth is:

```text
100 * (exp(4 * (log(GDPC1[t]) - log(GDPC1[t-1]))) - 1)
```

The formulas above define elapsed-period annualization, not a forecast. Missing
months or quarters must not be bridged silently. A recipe is unavailable when
the required current or comparison observations exceed the series-specific
date-gap tolerance.

The dashboard stores the raw percentile without sign inversion. Economic
interpretation is carried separately through a `polarity` field:

- `higher_strength`: payrolls, consumption, production, GDP, and permits;
- `higher_slack`: unemployment;
- `higher_heat`: core PCE, median CPI, and inflation expectations;
- `higher_tightness`: real policy rate and ANFCI;
- `higher_stress`: high-yield spreads;
- `two_sided`: the Treasury curve.

This avoids turning every high percentile green or every low percentile red.
Color and plain-language labels should reflect the category-specific meaning,
while the numeric percentile continues to mean only `high or low versus
history`.

The machine-readable `recipe` on every manifest row is implemented in
`src/world_model/features/macro_dashboard.py`. The calculation layer returns a
native-frequency history with these Step 4 fields:

```text
level_value
velocity_value
level_percentile_10y
level_percentile_expanding
velocity_percentile_10y
velocity_percentile_expanding
persistence_direction
persistence_count
persistence_label
```

It can calculate from one as-of snapshot of the source series or rebuild
successive states from the standard vintage tables. In the latter mode each
release is calculated before any later vintage is applied, so revisions do not
leak backward into earlier states. Daily recipes ignore explicit FRED holiday
rows with null values and count actual observed business days.

#### Percentile calculation

For both level and velocity:

1. Calculate the metric only on native-frequency release observations.
2. Compare the current metric with prior metrics whose release dates fall in the
   trailing 10 calendar years, excluding the current observation.
3. Require at least 60 monthly-equivalent observations and at least 5 calendar
   years of usable history. Frequency-specific counts are 260 weekly, 60
   monthly, or 20 quarterly observations.
4. Calculate the empirical percentile using mid-rank treatment for ties.
5. Calculate a second expanding-history percentile using all prior usable
   point-in-time observations.
6. Mark the row `baseline sensitive` when the two percentiles differ by at
   least 25 percentile points.

Percentiles are not winsorized. Extreme observations should remain extreme in
rank space. Raw values and changes may be clipped only for chart-axis rendering,
never for stored calculations.

#### Persistence and status labels

Persistence is not a third score. It is a compact description of whether the
latest velocity direction has survived multiple releases:

- `new turn`: direction changed on the latest release;
- `building`: same non-neutral direction for 2 consecutive releases;
- `persistent`: same non-neutral direction for at least 3 consecutive releases;
- `flat`: absolute raw velocity is at or below the 20th percentile of its own
  trailing historical absolute changes.

The initial plain-language row status is constructed from level, velocity, and
polarity. Examples include `strong but slowing`, `weak and deteriorating`,
`inflation elevated but cooling`, and `credit stress low but widening`. These
labels describe the inputs and must not imply a market forecast.

The internal persistence labels remain available for calculations and alert
filters, but they are not shown directly in the monitor. The interface instead
combines an indicator-specific momentum description with the direction streak:
for example, payrolls display `Decelerating · 2 releases`, inflation can display
`Cooling`, and credit spreads can display `Widening`. A one-release streak is
shown as `turned this release`. This prevents `building` from being mistaken for
an improving economic signal.

The monitor also avoids the implementation label `transformed level`. Each
detail view names the economic reading directly—such as `Current growth pace`,
`Current inflation pace`, or `Current credit spread`—and places the concise
calculation basis beneath it. The full transformation remains available in the
recipe section.

The detail drawer carries a separate `Additional context` section for familiar
raw and supporting readings. It includes payroll changes, real income and retail
sales, headline/core/food/energy inflation, nominal GDP and PCE, GDPNow and PCE
nowcasts, the policy-rate inputs, current spreads, permit levels, and housing
starts relative to the labor force. These values are calculated from vintages
available at the dashboard as-of date and are deliberately excluded from the
row's percentile and momentum scores.

### Level Measurement

The primary displayed level should be a percentile against the trailing 10
calendar years of point-in-time history. An expanding or full-history percentile
should be retained as secondary structural context.

For each series, display:

```text
current value | actual change | 10-year percentile | full-history percentile
```

The difference between the 10-year and full-history percentiles is itself useful.
Flag large disagreements as `baseline sensitive` rather than silently choosing
one history.

Historical windows must be calendar-based. The previous `window: 120` scoring
configuration counted observations, which represented about 120 weeks for
weekly data, 10 years for monthly data, and 30 years for quarterly data. Step 3
replaced it with a trailing 10-calendar-year window across frequencies, with a
five-year minimum-history requirement.

Before ranking, each series must use an economically meaningful transformation:

- nominal-dollar series should become real, per-capita, ratio, or growth
  measures where appropriate;
- secularly trending quantities should not be ranked in raw levels;
- bounded diffusion indices can often be ranked in levels;
- rates and spreads can be ranked in levels but should retain basis-point units;
- inverted indicators must preserve an explicit label explaining their economic
  direction.

### Velocity Measurement

Velocity must be computed from each series at its native observation or release
frequency before the result is forward-filled into the daily state panel.
Calculating changes from the daily forward-filled history would create long runs
of zero followed by artificial release-day jumps.

Do not force a single velocity formula across every series. Initial recipes
should be defined explicitly:

- payrolls: 3-month average job creation and/or 3-month annualized growth;
- unemployment: change in the 3-month average, with distance from the 12-month
  low available as context;
- consumption and industrial production: 3-month annualized growth and its
  change;
- inflation: 3-month and 6-month annualized rates and their change;
- weekly claims: 4-week average compared with the prior 13 weeks;
- rates and spreads: 3-month basis-point change;
- GDP: quarter-over-quarter annualized growth, with acceleration treated as a
  low-frequency and revision-sensitive signal.

The displayed velocity score should be the percentile of the resulting change
measure against a trailing calendar-based history. The raw change must remain
visible beside the percentile so the score can be sanity-checked.

### Freshness and Revisions

Freshness is a first-class dashboard dimension. Every indicator row should
carry:

- latest observation date;
- release date or point-in-time availability date;
- days since release;
- expected next release when available;
- revision status when the source supports it.

Step 5 implements this metadata on the release-time indicator history. It
distinguishes the date of the latest value-changing release from the release
that introduced the latest observation. Consequently, a revision is visible
but does not make an old observation appear fresh. The row carries:

```text
release_date
observation_release_date
release_lag_days
revision_status
is_revision
source_series_updated
expected_next_release
expected_next_release_is_estimate
```

Expected release dates are cadence estimates rather than publisher-calendar
promises: next business day for daily rows, seven days for weekly rows, one
month for monthly and mixed-frequency rows, and three months for quarterly
rows. A daily snapshot adds `days_since_release`,
`days_since_observation_release`, and a `fresh`, `due`, or `stale` status.
Configurable stale thresholds are four days for daily data, 14 for weekly, 50
for monthly, and 130 for quarterly.

History quality is explicit rather than inferred from a blank percentile. Each
level and velocity measure carries its usable observation count, calendar span,
status, and reason. Current reason codes include
`fewer_than_minimum_observations`, `less_than_minimum_calendar_history`, and
`metric_unavailable`. The absolute difference between the trailing and
expanding percentiles is stored for both level and velocity; a gap of at least
25 percentile points sets `baseline_sensitive`.

The build should fail visibly or show a warning when the data store or generated
dashboard is stale. Slow quarterly data should not appear as current merely
because it has been forward-filled to today's date.

### Presentation

The primary view should be a dense monitor rather than a single scatterplot.
Each row should show:

- recognizable series name;
- current transformed value;
- actual recent change;
- 10-year level percentile;
- velocity percentile;
- short trend sparkline;
- freshness;
- baseline-sensitivity or revision flag when relevant.

A secondary level-versus-velocity map can summarize the state, preferably as
separate facets for activity, inflation, and financial conditions. A single map
mixing all series can be misleading because `high and accelerating` has a
different economic meaning for payrolls, inflation, claims, and credit spreads.

Category summaries should use plain language such as:

```text
activity mixed | labor cooling | inflation elevated but decelerating
```

Named regime labels should remain secondary until the underlying policy,
rates, spreads, and financial-conditions coverage is complete and validated.

### Implementation Sequence

1. **Complete:** define and document one level and velocity recipe for every
   core series. See `Indicator Recipe Specification` above.
2. **Complete:** declare and populate the FRED universe for real consumption,
   policy, curve, high-yield spreads, adjusted financial conditions, permits,
   and mortgage rates through the existing point-in-time datafeed. ISM remains
   deferred.
3. **Complete:** replace observation-count comparison windows with trailing
   calendar-based windows. The economic engine now uses a 10-year window with a
   5-year minimum history, and reusable percentile calculations exclude the
   current observation from its own baseline.
4. **Complete:** compute native-frequency transformed values, changes,
   trailing and expanding percentiles, and persistence measures without
   look-ahead. Each row now declares a machine-readable recipe, and the
   release-time calculator applies revisions only from their availability date.
5. **Complete:** add freshness, release, revision, history-quality, and
   baseline-sensitivity fields. Revisions do not reset observation freshness,
   and expected release dates are explicitly identified as cadence estimates.
6. **Complete:** produce a normalized indicator-level release-history dataset
   and current snapshot through `world_model.cli.macro_dashboard`. The first
   run bootstraps five years; daily runs merge a 14-day recalculation overlap to
   capture revisions without replaying the full archive. Parquet history and
   snapshot files plus a JSON snapshot are written under `outputs/` by default.
7. **Complete:** build the dense monitor as the first interface. The buildless
   site under `sites/macro-dashboard` reads browser-ready snapshot and compact
   release-history JSON written by the daily CLI. It provides category filters,
   text search, an attention-only mode, percentile bars, recent-history
   sparklines, plain-language momentum streaks and freshness states, and a detail drawer that
   exposes source IDs, timing, history coverage, comparison baselines, and the
   transformation recipe. Category color identifies subject matter and is not
   a favorable/unfavorable signal.
8. **Complete:** add historical drill-downs and category maps.
   - **8A — Complete:** the drawer provides selectable 1-year, 3-year, 5-year,
     and full-history views of the reported reading, level percentile, and
     velocity percentile, with revisions marked explicitly.
   - **8B — Complete:** four separate level-versus-velocity map facets preserve
     the different economic meanings of activity, inflation,
     financial-conditions, and housing data. Each point shows its current
     trailing-percentile coordinates. One selected series per facet shows a
     common 12-month path sampled from the latest point-in-time state at each
     month-end, so daily, weekly, monthly, and quarterly indicators cover the
     same calendar horizon and revision events do not create extra path steps.
     Start rings, aging monthly dots, endpoint arrows, and before/after
     percentile labels make direction explicit. Map coordinates are
     polarity-aligned within each facet while the monitor and drawer retain raw
     percentiles: unemployment is inverted to labor strength, mortgage rates
     are inverted to housing strength, and the Treasury curve is inverted to
     policy/financial restrictiveness. Context indicators use hollow markers
     and rows without sufficient comparison history are named but not plotted.
9. **Complete:** test each point-in-time
   level, velocity, percentile, and level-by-velocity interaction against
   subsequent asset-group returns, volatility, and drawdowns. The validator
   uses each futures group's own daily trading calendar, trains only on
   completed prior outcome windows, compares against the existing market-state
   baseline, and applies stability and multiple-testing screens.
10. **Complete as a research screen:** backfill the longest defensible
    point-in-time release history for each indicator and rerun the
    market-relevance tests. The full archive begins in 1994 overall, while
    shorter-vintage series enter later. Surviving evidence is concentrated in
    volatility and drawdown relationships rather than directional returns, so
    the output remains a candidate set rather than a regime definition.
    - **10B — Complete as an initial GTT screen:** reconstruct point-in-time
      unemployment-trend, industrial-production, and real-retail-sales danger
      signals; intersect them with 10-month and 200-day price downtrends; and
      test whether double confirmation predicts more subsequent volatility and
      drawdown than price trend alone. Daily paths determine outcomes, while
      month-end signal dates limit overlap. Candidate relationships must also
      improve worst-decile precision above its unconditional base rate and
      retain positive incremental lift under a 12-month-or-longer time-block
      bootstrap. Contiguous signals are grouped into episodes and then into
      12-month-clustered stress eras; a candidate must remain useful after any
      one entire era is removed.
11. For signals that survive Step 10, test whether their next level, velocity,
    or state transition can itself be forecast out of sample.
12. Only then construct market-facing macro composites or regime labels from
    the validated and, where useful, forecastable inputs.

### First-Version Acceptance Criteria

The first version is ready when:

- every visible number can be traced to a source series and transformation;
- every historical comparison is point-in-time and free of look-ahead;
- comparison windows represent the same calendar duration across frequencies;
- velocity is computed at the correct native frequency;
- stale observations are visually distinguishable from fresh releases;
- the raw value and raw change are visible beside normalized scores;
- the dashboard remains useful without relying on a composite or regime label;
- missing or stale data produces an explicit warning rather than a plausible
  but incomplete state.
