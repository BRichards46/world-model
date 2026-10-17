# Economic Asset Driver Validation

This study tests whether the legacy economic trigger framework has useful
asset-group information when evaluated point-in-time and walk-forward.

The model does not assume the thresholds are correct. It treats each threshold
as a hypothesis, then asks whether that known-at-the-time condition helped
forecast subsequent asset-group returns.

## Inputs

Trigger definitions live in:

```text
config/economic_triggers.yml
```

Asset groups and validation settings live in:

```text
config/economic_asset_drivers.yml
```

Each asset group can also define `driver_groups`. These are economic priors
used for reporting, not for hiding the full result grid. For example, lumber is
mapped to housing/income, employment, and CFNAI; energy is mapped to growth,
CFNAI, and inflation.

The trigger config encodes the legacy rules:

- retail sales, industrial production, payrolls, real personal income, and
  housing starts per labor force use YoY rates of change
- CFNAI components use their released level
- unemployment uses the 12-month average unemployment rate minus the latest
  unemployment rate
- CPI and core CPI are healthy when YoY inflation is below `2.5%`

## Point-In-Time Handling

The trigger panel is built from FRED vintage files under the configured FRED
root. For each release date, the validator uses only observations available in
that vintage as of that release date.

For every trigger, the panel stores:

- transformed value
- threshold margin, where positive means healthy
- healthy flag
- failed flag
- latest economic observation date known at the time

The panel is forward-filled to business days after release, which makes it
safe to join to daily futures data without using future economic releases.

## Asset Returns

CSI processed futures are additively back-adjusted. Log returns on those series
can be invalid because adjusted prices may become zero or negative.

The default asset return proxy is therefore:

```text
daily normalized point change = price.diff() / trailing median(abs(price))
```

The trailing median is shifted by one day, so the scale is known before the
return day. This is still a return proxy, not a trade-ready futures P&L series.
It is suitable for ranking macro driver evidence, not for reporting strategy
returns.

## Walk-Forward Test

At each month-end evaluation date:

1. The current economic trigger values are observed.
2. The future asset-group return is measured over 1, 3, 6, and 12 months.
3. The conditional forecast is trained only on prior evaluation dates whose
   own future return windows had already completed.
4. The model compares the conditional forecast against the unconditional
   historical baseline available at that date.

Binary trigger features use active-versus-inactive historical means. Continuous
features use historical quintile buckets with a minimum bucket sample.

## Outputs

Run:

```bash
PYTHONPATH=src /Users/brichards/anaconda3/bin/python -m world_model.cli.economic_asset_drivers
```

Outputs:

```text
outputs/economic_asset_drivers/economic_asset_driver_report.md
outputs/economic_asset_drivers/economic_asset_driver_summary.csv
outputs/economic_asset_drivers/economic_asset_driver_forecasts.parquet
outputs/economic_asset_drivers/economic_asset_driver_group_returns.parquet
outputs/economic_asset_drivers/economic_asset_driver_trigger_panel.parquet
```

## Interpretation

The strongest result is not the highest in-sample spread. The useful rows are
the ones that:

- have enough out-of-sample observations
- improve MSE versus the rolling unconditional baseline
- show sensible forecast/realized correlation
- make economic sense for that asset group
- survive adjacent horizons or related series

The markdown report ranks robust rows with at least 60 out-of-sample
observations and a configured economic driver prior first. Thin rows and
non-prior rows remain in the CSV, but should be treated as hypotheses for later
testing.
