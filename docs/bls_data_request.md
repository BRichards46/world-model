# BLS data request for the world-model CPI build

What the world-model needs from the Bureau of Labor Statistics, how to store
it, and why. The consumer is the bottom-up CPI nowcast
(`world_model.states.energy_cpi`, and the core CPI build that comes next).

Source: BLS Public Data API v2 (`https://api.bls.gov/publicAPI/v2/timeseries/data/`).
Store the key as `BLS_API_KEY` in the data-management `.env`. Limits with a
key: 500 queries a day, 50 series and 20 years per query.

Two items (seasonal factors and relative importance) are files on bls.gov,
not API series. They are listed separately below.

---

## 1. Storage: does it need the vintage structure?

**Yes. Use the same four-layer layout and column names as the FRED store**, so
the world-model's existing loaders (`load_vintage_table`, `known(...)`) read it
unchanged.

```
/Users/brichards/Documents/data/bls/
  vintage/<series_id>.parquet      series_id, observation_date, value, realtime_start, realtime_end, + extra columns below
  latest/<series_id>.parquet       series_id, observation_date, latest_value, first_release_date, ...
  initial/<series_id>.parquet      series_id, observation_date, initial_value, realtime_start, realtime_end
  metadata/<series_id>.json        series title, seasonal flag, item code, last download, API footnotes
  seasonal_factors/                one file per published year, kept as published (section 4)
  relative_importance/             one file per year (section 5)
  release_calendar/                BLS release schedule (section 6)
```

Dates are stored as dates, with `realtime_end = 9999-12-31` for current
values, the same as FRED.

### Why vintages matter, and how to get them without ALFRED

The BLS API returns only current values, never as-published history. The two
kinds of series behave differently:

| Kind | Revised? | How to fill `realtime_start` |
|---|---|---|
| **Not seasonally adjusted** (`CUUR…`) | **Never**, apart from rare corrections | The CPI release date for that month (section 6). One row per observation, `realtime_end = 9999-12-31`. This history is exactly as published. |
| **Seasonally adjusted** (`CUSR…`) | Every February, for the previous five years | **Backfill:** today's values with `realtime_start` = the download date and `is_backfill = true`. They are honest about being revised and never pretend to be first prints. **From now on:** after every release, compare with the stored latest values and append changed or new observations as new vintage rows (close the old row's `realtime_end`). That builds a true vintage history from today. |

Extra columns on the vintage layer:
- `is_backfill` (bool): true for rows reconstructed from a later download.
- `source` (`"bls_api"`).
- `footnotes` (text): the API's footnote codes, e.g. preliminary.

The world-model works mainly from the unadjusted series plus the published
seasonal factors (section 4). With those, as-published adjusted values can be
rebuilt for any past month.

---

## 2. CPI series (API)

All are US city average, monthly. Pull **both** forms of each item:
`CUSR0000<item>` (seasonally adjusted) and `CUUR0000<item>` (not seasonally
adjusted). Full history, from 1947 or the series start.

Some codes may come back invalid in one form (a few items aren't seasonally
adjusted). Log it and continue; don't fail the run. Items marked † should be
confirmed against the API on the first run.

### Aggregates
| Item | Name |
|---|---|
| `SA0` | All items |
| `SA0L1E` | All items less food and energy (core) |
| `SA0L2` | All items less shelter |
| `SA0L12E` † | All items less food, shelter and energy |
| `SAC` | Commodities |
| `SACL1E` | Commodities less food and energy commodities (core goods) |
| `SAS` | Services |
| `SASLE` | Services less energy services (core services) |
| `SASL2RS` † | Services less rent of shelter |

### Food and energy
| Item | Name |
|---|---|
| `SAF1` | Food |
| `SAF11` | Food at home |
| `SEFV` | Food away from home |
| `SA0E` | Energy |
| `SACE` | Energy commodities |
| `SETB01` | Gasoline (all types) |
| `SEHE` | Fuel oil and other fuels |
| `SEHF` | Energy services |
| `SEHF01` | Electricity |
| `SEHF02` | Utility (piped) gas service |

### Shelter
| Item | Name |
|---|---|
| `SAH1` | Shelter |
| `SEHA` | Rent of primary residence |
| `SEHC` | Owners' equivalent rent of residences |
| `SEHC01` | Owners' equivalent rent of primary residence |
| `SEHB` | Lodging away from home |
| `SEHG` † | Water, sewer and trash collection services |

### Core goods
| Item | Name |
|---|---|
| `SETA01` | New vehicles |
| `SETA02` | Used cars and trucks |
| `SAA` | Apparel |
| `SAH3` | Household furnishings and operations |
| `SAM1` | Medical care commodities |
| `SAR` † | Recreation |
| `SEGA` † | Tobacco and smoking products |

### Core services excluding shelter
| Item | Name |
|---|---|
| `SETE` | **Motor vehicle insurance**: the main reason for this request; not on FRED |
| `SETD` | Motor vehicle maintenance and repair |
| `SETG01` | Airline fares |
| `SAS4` | Transportation services |
| `SAM2` | Medical care services |
| `SEMC01` † | Physicians' services |
| `SEMD01` † | Hospital services |
| `SEME` † | Health insurance |
| `SAE` † | Education and communication |
| `SAG` † | Other goods and services |

About 90 series in all, two per item. A full-history pull needs about 8–10
API calls: two groups of 50 series, each split into 20-year windows from 1947.
A routine update needs two.

### Average prices (API, `APU0000…`, not seasonally adjusted)
| Series | Name |
|---|---|
| `APU000074714` | Gasoline, unleaded regular, per gallon (already on FRED; fine to duplicate) |
| `APU000074715` † | Gasoline, unleaded midgrade |
| `APU000074716` † | Gasoline, unleaded premium |
| `APU000072610` | Electricity per kWh |
| `APU000072620` | Utility (piped) gas per therm |

---

## 3. PPI series (API), phase 2: the CPI-to-PCE bridge

Core PCE takes several components from PPI rather than CPI. These let core
PCE be built once CPI and PPI are out. All are monthly; store them with the
same vintage layout (PPI is revised four months after first release, so the
snapshot approach in section 1 applies). All need confirming on the first run.

| Series † | Name |
|---|---|
| `PCU481111481111` | Scheduled passenger air transportation |
| `PCU621111621111` | Offices of physicians |
| `PCU622---622---` | Hospitals |
| `PCU6216--6216--` | Home health care services |
| `PCU623110623110` | Nursing care facilities |
| `PCU5239205239201` | Portfolio management (already on FRED) |
| `WPUFD4131` | Final demand services less trade, transportation, warehousing |
| `WPUFD41312` | Final demand: portfolio management (if present) |

---

## 4. CPI seasonal factors (files, not API)

BLS publishes the CPI seasonal factors each February for the year ahead, at
`https://www.bls.gov/cpi/tables/seasonal-adjustment/`. The page links current
and prior years' factor files and the list of seasonally adjusted series.

- **Download every year available, and keep each one as published.** Don't
  overwrite: the factors for a past year changed when that year was revised,
  and we need the version known at the time.
- Store one parquet per file, named for the year it was published:
  - item code;
  - month;
  - factor;
  - published date;
  - `is_projected`, for factors published in advance vs revised later.
- **Why:** the published factor gives the exact adjusted change for any month
  from the unadjusted one. Today the model estimates it from last year's
  factor. This is the biggest single accuracy gain for gasoline, and it
  rebuilds as-published adjusted history for items not on FRED.

---

## 5. Relative importance (files, not API)

Each item's share of CPI as of December each year, at
`https://www.bls.gov/cpi/tables/relative-importance/`. Weights are updated
annually; the tables go back to the 1980s.

- One parquet with year (the December reference), item code, item name and
  relative importance for all urban consumers (CPI-U). If the files give
  CPI-W, include it as a second column.
- **Why:** the model currently combines blocks with weights fitted on the
  trailing three years. That adds about 0.02 pp of error per release. BLS's
  December weights, rolled forward month by month with each item's relative
  price change, make the combination exact.

---

## 6. Release calendar

The BLS release schedule, available as an iCalendar feed
(`https://www.bls.gov/schedule/news_release/bls.ics`) and as yearly schedule
pages.

- One parquet with release name (CPI, PPI, Employment Situation, JOLTS, Real
  Earnings), reference period, release date and time (ET), and download date.
- Keep past releases, since the feed only shows a window.
- **Why:**
  1. The dashboard currently estimates release dates as last release plus a
     month, shown with "≈". This makes them exact.
  2. The release dates set `realtime_start` for the unadjusted CPI series in
     section 1.

---

## 7. Update cadence

| What | When |
|---|---|
| CPI and average-price series | After each CPI release (about 8:35 ET on release day), or a daily run that exits early if nothing changed. Seasonally adjusted series must be snapshotted on the February release, when five years of history are revised. |
| PPI series | After each PPI release; same snapshot logic. |
| Seasonal factors | Each February, after BLS posts them. |
| Relative importance | Each January/February, when the new December weights appear. |
| Release calendar | Weekly. |

Each run should write a short summary to metadata: series fetched, rows added,
vintages opened or closed, invalid IDs. The world-model can then check
freshness, as it does for FRED.

---

## 8. Implementation notes

### API calls
- POST JSON to `https://api.bls.gov/publicAPI/v2/timeseries/data/` with:
  - `seriesid` (a list of up to 50);
  - `startyear` and `endyear` (at most 20 years apart);
  - `registrationkey`;
  - `catalog: true` on the first pull, which returns series titles for the metadata layer.
- Each response has a `status` of `REQUEST_SUCCEEDED`, `REQUEST_NOT_PROCESSED` or
  `REQUEST_FAILED`, plus a `message` list. Invalid IDs come back as messages
  ("Series does not exist…") while the rest of the request succeeds. Log those
  and continue. A daily-limit message should stop the run cleanly.
- Each data point has `year`, `period` (`M01`–`M12`, or `M13` for the annual
  average) and `value` (a string, sometimes `-` for unavailable), plus
  `footnotes` and sometimes `latest`.
  - Keep `M01`–`M12` only; drop `M13`.
  - Convert `-` to null.
  - `observation_date` = the first of the month.
- Leave a short pause between calls. 500 queries a day is plenty.

### Downloading files from bls.gov
- bls.gov rejects scripted requests that look like bots, usually with 403
  Forbidden. Send a `User-Agent` header that names the client and a contact
  email; BLS asks for this. This applies to the seasonal factor files, the
  relative importance tables and the calendar feed. The API itself doesn't
  need it.
- Formats vary by year and haven't been checked:
  - **Seasonal factors and relative importance:** recent years are text or
    Excel; older years may be PDF only. Take what is machine-readable, record
    in metadata which years are missing or PDF-only, and don't scrape PDFs in
    the first version.
  - **Relative importance in particular:** expect machine-readable tables back
    to roughly the mid-2010s, with older years possibly PDF only. Recent years
    matter most.

### Release dates for the unadjusted backfill
- The iCalendar feed only covers recent and upcoming releases. For older CPI
  releases, use the first-release dates already in the FRED store: the
  earliest `realtime_start` per `observation_date` in
  `/Users/brichards/Documents/data/fred/vintage/CPIAUCSL.parquet` (back to 1972).
  PPI release dates can come the same way from `PPIACO.parquet`.
- If neither source has a date, use the download date and set `is_backfill = true`.

### Recording new values after a release
- If a run follows a known release (calendar), use that **release date** as
  `realtime_start` for new and changed rows, not the download date.
- Only the backfill and unexplained changes get the download date.
- When a value changes, set the old row's `realtime_end` to the day before the
  new `realtime_start`.

### Checks the first run should pass
- **Unadjusted matches FRED:** `CUUR0000SA0` equals FRED `CPIAUCNS`
  (`latest/`) for every month, and `CUUR0000SA0L1E` equals `CPILFENS`.
- **Adjusted matches FRED:** `CUSR0000SA0` equals FRED `CPIAUCSL` latest
  values, and `CUSR0000SETB01` equals FRED `CUSR0000SETB01`.
- **No duplicates:** no duplicate (`series_id`, `observation_date`,
  `realtime_start`) rows; at most one open row (`realtime_end = 9999-12-31`)
  per observation.
- **Every † code resolved** as valid or invalid in metadata.

### On the world-model side (not for data-management)
- Add `bls: /Users/brichards/Documents/data/bls` to `config/data_roots.yml`
  (done 2026-11-12).
- `load_vintage_table(root, series_id)` reads it unchanged because the layout
  matches FRED.

---

## 9. First-run review (2026-11-12)

The first full run passed almost everything. 96 of 97 series arrived. The
missing one, `CUSR0000SEME`, doesn't exist at BLS, which section 2 allows.
All † codes resolved as valid. Values match FRED exactly (`CPIAUCNS`,
`CPILFENS`, `CPIAUCSL`, `CPILFESL`, `CPIUFDSL`, `CPIENGSL`, gasoline,
electricity, utility gas, fuel oil). Release dates since Dec 2007 match
FRED's first prints for all 226 months, and the October 2025 shutdown gap is
stored as missing with its footnote. The 2026 seasonal factors reproduce
unadjusted ÷ adjusted exactly. Four fixes remain, in priority order.

### Fix 1: release dates for unadjusted rows before Dec 2007
`realtime_start` is null for every unadjusted row before Dec 2007: 732 months
for `CUUR0000SA0`, and 47 of the 55 unadjusted series. The section 8
fallback wasn't applied.
- **CPI (`CUUR…`) and average prices (`APU…`):** set `realtime_start` to
  the earliest `realtime_start` for that `observation_date` in
  `/Users/brichards/Documents/data/fred/vintage/CPIAUCSL.parquet`. That covers
  425 months, 1972-07 to 2007-11. Average prices come out in the CPI report,
  so they use the same dates.
- **Earlier months:** FRED has nothing before 1972-07, so leave those null
  and count them in metadata. Don't invent dates.
- **Check:** for every unadjusted CPI or average-price series, nothing is
  null from 1972-07 on except months whose value is itself missing.

### Fix 2: a publication date for seasonal factors
`published_date` is null in every seasonal factor file. Without it the
factors can't be used point-in-time. BLS publishes each year's factors with
the January CPI report (released in February).
- **2008 onward:** set `published_date` to that report's `release_date`
  from the release calendar (CPI, `reference_period` = January of the
  factor year).
- **Earlier years:** use the first `realtime_start` for January of that
  year in FRED `CPIAUCSL`.

### Fix 3: state JOLTS in the release calendar
State JOLTS releases are labelled `JOLTS`, which duplicates 52 reference
months. Name them `State JOLTS`, using the summary that starts "State Job
Openings and Labor Turnover". After the fix, each `release_name` should have
at most one row per `reference_period`.

### Fix 4: duplicate rows in the 1947–1986 weights
`historical-relative-importance-1947-1986.xlsx` gives each item twice, once
in capitals and once in normal case, with identical values. That makes 77
duplicate (`year`, `item_code`) rows. Keep one per (`year`, `item_code`).
Where the values ever differ, keep both and log them. 35% of rows have no
`item_code` (11% from 1987 on). That's fine for items we don't use, but
`SAH1`, `SEHA`, `SEHC`, `SETB01` and the other section 2 items must all have
codes.

### Aside: the FRED store marks several rows current
Not BLS, but found during the checks. In
`/Users/brichards/Documents/data/fred/vintage/CPIAUCSL.parquet`, older rows
are never closed when a new snapshot arrives, so most months have several
rows with `realtime_end = 9999-12-31` (5,654 of 5,656 open rows). The
world-model reads the newest row and isn't affected. The data-mgmt project
may still want to close superseded rows (set `realtime_end` to the day before
the next `realtime_start`) so the FRED store passes the same at-most-one-open-row
check as the BLS store.

---

## Priority if this is done in stages

1. **Motor vehicle insurance** (`SETE`), adjusted and unadjusted, plus the
   unadjusted versions of the section 2 items. These unlock the core build.
2. **Seasonal factor files** and **relative importance**.
3. **Release calendar**.
4. **The remaining adjusted series**, with the snapshot logic.
5. **PPI** for the PCE bridge.
