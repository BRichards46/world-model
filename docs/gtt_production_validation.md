# GTT Production-Signal Validation

This exploratory layer tests whether predefined industrial-production
decompositions and related leading indicators improve the classification of
S&P 500 risk when price trend is already negative.

The benchmark remains point-in-time `INDPRO` year-over-year growth at or below
zero. Alternatives are constructed only from values available on each FRED
vintage date:

- `IPMAN`, `IPFINAL`, `IPBUSEQ`, and `IPMAT`: year-over-year and three-month
  non-positive growth;
- `AWHMAN`: below its trailing 12-month average and lower than three months ago;
- `NEWORDER`, `AMTMNO`, and `DGORDER`: CPI-deflated year-over-year and
  three-month non-positive growth;
- `CFNAI` and `CFNAIMA3`: below zero;
- `CUMFNS`: below its trailing 12-month average and lower than three months ago.

The study also tests OR and AND combinations of `INDPRO` with the two strongest
alternatives, manufacturing hours and real durable-goods orders. OR combinations
measure added warning coverage; AND combinations measure confirmation and
precision. These combinations are exploratory and do not constitute an
independent holdout test.

Most direct production-component vintage histories begin in 2007 or later.
Large estimated effects from those series cannot clear the same multi-era
robustness standard as `INDPRO` or `AWHMAN` and must not be interpreted using
their longer, subsequently revised observation histories.

Run:

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python \
  -m world_model.cli.gtt_production_validation
```

Outputs are written to `outputs/gtt_production_validation/`.
