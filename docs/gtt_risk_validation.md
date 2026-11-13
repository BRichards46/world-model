# GTT Drawdown and Volatility Research

This layer asks whether the Growth–Trend Timing (GTT) double-confirmation idea
helps identify periods of elevated future volatility and drawdown. It is a risk
study, not a trading recommendation or a direct reproduction of the original
equity backtest.

## Concept

The source articles make two distinct claims:

1. A negative price trend is a warning that a large drawdown has become more
   likely, but price trend alone suffers costly whipsaws.
2. A broad growth warning can distinguish negative price breaks that are more
   likely to coincide with the business cycle. When growth is healthy, GTT
   ignores the negative price signal. When growth is unhealthy, it defers to
   price trend.

The resulting defensive state is an intersection, not a macro forecast:

```text
double confirmation = macro danger AND negative price trend
```

The unemployment version defines macro danger as the current unemployment rate
being above its trailing 12-month average. The original preferred growth
version defines danger when either real retail-sales year-over-year growth or
industrial-production year-over-year growth is non-positive.

## Point-in-time construction

All macro states are reconstructed from the local FRED vintage store. A release
changes the state only on its `realtime_start`; later revisions do not leak
backward. Real retail sales are calculated as nominal retail sales (`RSAFS`)
divided by headline CPI (`CPIAUCSL`) for the latest common observation month.

The study includes four macro filters:

- unemployment relative to its trailing 12-month average;
- industrial-production year-over-year growth;
- the dual real-retail-sales / industrial-production growth warning;
- an any-warning sensitivity aggregate.

Price trend is tested with both a 10-month moving average and a 200-trading-day
moving average. The primary equity benchmark chains daily S&P 500 price-index
returns from Norgate `$SPX` with total returns from `$SPXTR` whenever the latter can
be calculated (January 5, 1988 onward). The two raw index levels are never
spliced; their log returns form a continuous normalized path. Consequently,
the long pre-1988 segment excludes dividends, while the later segment includes
reinvested dividends. S&P volatility and drawdown outcomes are calculated from
the same chained returns, with drawdowns compounded into percentage wealth
losses.

The CSI contracts remain supplementary cross-asset tests. Because they are
additively back-adjusted, their trend calculation continues to use cumulative
normalized point changes rather than compounding.

Daily data determine every observation inside the subsequent 1-, 3-, 6-, and
12-month volatility and maximum-drawdown windows. Signal origins are sampled at
month-end to limit pseudo-replication from overlapping daily windows.

## Direct incremental test

The key comparison is not simply double confirmation versus all other dates.
It is:

```text
macro danger + negative trend
versus
healthy macro + negative trend
```

That tests the central GTT proposition: whether the macro filter separates a
negative price trend with follow-through risk from one more likely to whipsaw.
The report also shows worst-decile capture and precision. Candidate status
requires adequate observations in each state, positive risk lift, positive
incremental lift over price trend alone, each sample half retaining at least
10% of the full-sample lift, worst-decile precision above its unconditional
base rate, and a positive lower bound from a 90% moving-block bootstrap of the
incremental lift over price trend alone. Bootstrap blocks are at least 12 months
long and expand to the target horizon when necessary.

Contiguous month-end double-confirmation observations are also grouped into
episodes. A candidate needs at least three distinct active episodes and must
retain positive incremental lift after removing each episode in turn. This
leave-one-episode-out test prevents a single crisis from carrying the result.

Because one economic downturn can contain several signal interruptions, nearby
episodes are additionally clustered into stress eras whenever successive
active observations are no more than 12 months apart. A candidate must span at
least three such eras and retain positive incremental lift after removing each
entire era. This is the more demanding test of dependence on the dot-com, GFC,
or pandemic periods.

The block bootstrap and leave-one-episode-out screen reduce false confidence
from overlapping windows and single-crisis dependence. The small absolute
number of historical recessions remains a hard sample limitation.

## Run

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python \
  -m world_model.cli.gtt_risk_validation
```

Outputs are written to `outputs/gtt_risk_validation/`. The file
`gtt_sp500_hybrid_history.parquet` records each daily log return, whether it
came from the price or total-return index, and the resulting continuous
normalized index, making the 1988 source handoff auditable.

## Sources

- https://www.philosophicaleconomics.com/2016/02/uetrend/
- https://www.philosophicaleconomics.com/2016/01/gtt/
- https://www.philosophicaleconomics.com/2016/01/movingaverage/

## Alternative thresholds and combinations

`docs/gtt_threshold_validation.md` tests the legacy trigger levels against the
zero lines, plus OR combinations and trigger counts. Summary: unemployment
more than 0.1 pp above its 12-month average is the strongest single filter.
Industrial production works at any level from 0% to 3%. CPI fails at every
level. Hours and durable orders do not improve industrial production as OR
terms. The dashboard overlay now uses industrial production ≤ 0 OR
unemployment more than 0.1 pp above its 12-month average. This study keeps
testing the zero lines.
