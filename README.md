# whereami

Where does the S&P 500 stand in its own history? One page, updated every
weekday, with readings computed the way a person would have seen them on that
day. No hindsight refits. A gauge, not a signal.

Page: https://wwmgianclaww.github.io/whereami

## What the reading is

1. Take the S&P 500 level and adjust it for inflation (CPI, all urban
   consumers, not seasonally adjusted, so it is never revised).
2. Fit a straight line through the logarithm of that level against time by
   least squares. A straight line in log space is a constant yearly growth
   rate, which is the right shape for an index that grows by percent.
3. For each date, fit the line and its typical gap (sigma) using only data up
   to that date. The reading is that date's gap divided by that sigma, in log
   units, so a doubling and a halving are the same size step.
4. The percentile is the share of earlier months in the same fit with a
   smaller gap. It makes no bell-curve promise.
5. Two lines: price only, and total return with dividends reinvested.

Monthly points from 1950 to about ten years ago, daily points since. Readings
start in 1970 so every fit has at least twenty years behind it.

## Why "seen on the day"

Most trend charts refit the line on all data to today, so every old reading
shifts each time new data arrives. What such a chart shows for 1974 or 2000
is not what anyone saw at the time. Here the reading for any date uses only
what existed on that date.

Published readings are never recomputed. `docs/data.json` is append-only:
each run keeps every row already there and adds rows for new dates. As a
guard, the run also recomputes the old rows and refuses to write anything if
one would drift by more than 0.05 sigma or 2 percentile points, which would
mean a source had been revised or truncated.

Two small exceptions, stated so they are not hidden: the first publication
backfilled 1970 to 2026 using each month's own CPI, which in real life was
published a few weeks after that month (effect under 0.01 sigma), and the CPI
for October 2025 was never published and is interpolated between September
and November.

## What it is not

Tested this way, the reading has shown no useful link to what the market did
over the next 1, 3, 5 or 10 years. High readings have stayed high for years.
"Standard deviations" is a ruler for the typical size of the gap, not a
probability; the gap is a slow cycle, not a bell curve.

## Sources

All four are required. If any is unreachable, fails a sanity check, or is
missing the latest trading day, the run writes nothing and the page keeps its
previous data and date.

- Shiller monthly price and dividend data, via https://datahub.io/core/s-and-p-500
  (dividends in that copy stop in mid-2023; after that the last known
  dividend yield is carried, which only matters for the 1950 to 1987 chain)
- FRED SP500 (daily closes, last ten years): https://fred.stlouisfed.org/series/SP500
- FRED CPIAUCNS: https://fred.stlouisfed.org/series/CPIAUCNS
- S&P 500 total-return index from 1988, via Yahoo Finance

Only derived readings are published in `docs/data.json`. No index levels are
stored in this repository.

## Run it yourself

Python 3, standard library only.

```bash
python3 -m unittest discover -s tests -v
python3 compute.py update
python3 -m http.server 8765 --directory docs
```

`compute.py update` fetches the sources into `.cache/` (ignored by git),
validates them, computes fresh readings, merges them append-only into
`docs/data.json`, and leaves the file untouched when nothing new arrived.

## Hosting

GitHub Pages serves the `docs/` folder of the `main` branch. The daily job is
`.github/workflows/update.yml`: it runs the tests, then the update, and
commits `docs/data.json` only when it changed. `docs/data.json` must be
committed once by hand before the first scheduled run.
