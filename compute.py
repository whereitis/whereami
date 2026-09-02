#!/usr/bin/env python3
"""whereami: market gauges as a person would have seen them on the day.

Stdlib only. Commands:
    python3 compute.py fetch    download sources into .cache/ (gitignored)
    python3 compute.py build    compute from .cache/ and update docs/data.json
    python3 compute.py update   fetch, then build (the daily job)

Method (see README.md): fit an exponential trend by least squares on the log
of the inflation-adjusted index, using only data available on the reading
date (expanding window). Reading = residual / sigma of that same fit, in log
units. Percentile = share of that fit's residuals smaller than the reading's.

docs/data.json is append-only. Rows already published are kept exactly as
they were; a run only adds rows for new dates. If a recomputed old row would
differ from the published one beyond a small tolerance, the run fails and
writes nothing. No raw index levels are written, only derived readings.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".cache")
OUT = os.path.join(ROOT, "docs", "data.json")
START = dt.date(1950, 1, 1)          # history starts here
READINGS_FROM = dt.date(1970, 1, 1)  # 20-year minimum window before a reading
TR_ANCHOR = "1988-01"                # first month of the official total-return index
MIN_DAYS_FULL_MONTH = 15
FRED_MIN_ROWS = 2400                 # FRED serves ten years, about 2515 rows
TOLERANCE_Z = 0.05                   # a published row may not drift more than this
TOLERANCE_PCT = 2.0
UA = "whereami/1.0"

SOURCES = {
    "shiller": "https://datahub.io/core/s-and-p-500/r/data.csv",
    "fred_sp500": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
    "fred_cpi": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS",
    "yahoo_tr": ("https://query1.finance.yahoo.com/v8/finance/chart/%5ESP500TR"
                 "?period1=567993600&period2={now}&interval=1d"),
}


class ValidationError(Exception):
    """A source or result failed a check. Nothing is written when this is raised."""


# ---------------------------------------------------------------- fetching

def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def cmd_fetch() -> None:
    os.makedirs(CACHE, exist_ok=True)
    now = int(time.time())
    for name, url in SOURCES.items():
        try:
            data = http_get(url.format(now=now))
        except Exception as e:  # noqa: BLE001 - every source is required; report and stop
            raise ValidationError(f"fetch {name} failed: {e}") from e
        with open(os.path.join(CACHE, name), "wb") as f:
            f.write(data)
        print(f"fetched {name}: {len(data)} bytes")


def read_cache(name: str) -> str:
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        raise ValidationError(f"missing cached source {name}; run fetch first")
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------- dates

def ym(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def ym_next(m: str) -> str:
    y, mo = int(m[:4]), int(m[5:7])
    return f"{y + 1:04d}-01" if mo == 12 else f"{y:04d}-{mo + 1:02d}"


def ym_mid(m: str) -> dt.date:
    return dt.date(int(m[:4]), int(m[5:7]), 15)


def years_since_start(d: dt.date) -> float:
    return (d - START).days / 365.25


# ---------------------------------------------------------------- parsing

def parse_shiller(text: str) -> dict[str, tuple[float, float]]:
    """month -> (monthly average price, annual dividend rate). Zero means not yet known."""
    out: dict[str, tuple[float, float]] = {}
    try:
        for r in csv.DictReader(io.StringIO(text)):
            m = r["Date"][:7]
            if m < ym(START):
                continue
            out[m] = (float(r["SP500"] or 0), float(r["Dividend"] or 0))
    except (KeyError, ValueError, TypeError) as e:
        raise ValidationError(f"shiller: cannot parse: {e}") from e
    return out


def parse_fred(text: str, name: str = "fred") -> list[tuple[dt.date, float]]:
    out: list[tuple[dt.date, float]] = []
    try:
        for r in csv.DictReader(io.StringIO(text)):
            keys = list(r.keys())
            v = (r[keys[1]] or "").strip()
            if v in ("", "."):
                continue
            out.append((dt.date.fromisoformat(r[keys[0]]), float(v)))
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise ValidationError(f"{name}: cannot parse: {e}") from e
    return out


def parse_yahoo(text: str) -> list[tuple[dt.date, float]]:
    try:
        res = json.loads(text)["chart"]["result"][0]
        out: list[tuple[dt.date, float]] = []
        for t, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]):
            if c is None:
                continue
            out.append((dt.datetime.fromtimestamp(t, dt.timezone.utc).date(), float(c)))
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise ValidationError(f"yahoo_tr: cannot parse response: {e}") from e
    return out


# ---------------------------------------------------------------- validation

def validate_daily(rows: list[tuple[dt.date, float]], name: str, today: dt.date,
                   min_rows: int = 1500, max_gap_days: int = 7, max_age_days: int = 7) -> None:
    if len(rows) < min_rows:
        raise ValidationError(f"{name}: only {len(rows)} rows")
    for i, (d, v) in enumerate(rows):
        if v <= 0:
            raise ValidationError(f"{name}: non-positive value on {d}")
        if i:
            gap = (d - rows[i - 1][0]).days
            if gap <= 0:
                raise ValidationError(f"{name}: dates not increasing at {d}")
            if gap > max_gap_days:
                raise ValidationError(f"{name}: {gap}-day gap before {d}")
    if (today - rows[-1][0]).days > max_age_days:
        raise ValidationError(f"{name}: stale, last date {rows[-1][0]}")


def validate_months(present: dict, name: str, first: str, last: str) -> None:
    m = first
    while m <= last:
        if m not in present:
            raise ValidationError(f"{name}: missing month {m}")
        m = ym_next(m)


def fill_missing_months(series: dict[str, float], max_gap: int = 3) -> tuple[dict[str, float], list[str]]:
    """Fill months missing inside a monthly series by log-linear interpolation.

    Needed because the CPI for October 2025 was never published (US government
    shutdown). More than max_gap consecutive missing months is an error.
    """
    months = sorted(series)
    out = dict(series)
    filled: list[str] = []
    m, last = months[0], months[-1]
    while m < last:
        nxt = ym_next(m)
        if nxt in out:
            m = nxt
            continue
        gap: list[str] = []
        probe = nxt
        while probe not in out:
            gap.append(probe)
            probe = ym_next(probe)
            if len(gap) > max_gap:
                raise ValidationError(f"{len(gap)} consecutive months missing from {gap[0]}")
        lo, hi = math.log(out[m]), math.log(out[probe])
        for i, g in enumerate(gap, 1):
            out[g] = math.exp(lo + (hi - lo) * i / (len(gap) + 1))
        filled.extend(gap)
        m = probe
    return out, filled


def validate_cpi_steps(cpi: dict[str, float], max_log_step: float = 0.05) -> None:
    months = sorted(cpi)
    for a, b in zip(months, months[1:]):
        if cpi[a] <= 0 or cpi[b] <= 0:
            raise ValidationError(f"cpi: non-positive value in {a} or {b}")
        if abs(math.log(cpi[b] / cpi[a])) > max_log_step:
            raise ValidationError(f"cpi: implausible jump between {a} and {b}")


def monthly_averages(rows: list[tuple[dt.date, float]]) -> tuple[dict[str, float], dict[str, int]]:
    acc: dict[str, list[float]] = {}
    for d, v in rows:
        acc.setdefault(ym(d), []).append(v)
    return {m: sum(v) / len(v) for m, v in acc.items()}, {m: len(v) for m, v in acc.items()}


# ---------------------------------------------------------------- the fit

def fit_from_sums(n: int, sx: float, sy: float, sxx: float, sxy: float, syy: float):
    """Least squares y = a + b x from running sums. Returns (a, b, sigma)."""
    mx, my = sx / n, sy / n
    sxx_c = sxx - n * mx * mx
    sxy_c = sxy - n * mx * my
    b = sxy_c / sxx_c
    a = my - b * mx
    sse = syy - a * sy - b * sxy
    return a, b, math.sqrt(max(sse, 0.0) / n)


def prefix_sums(xs: list[float], ys: list[float]) -> list[tuple]:
    """prefix[k] = sums over the first k points (prefix[0] is all zeros)."""
    out = [(0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    n = sx = sy = sxx = sxy = syy = 0.0
    for x, y in zip(xs, ys):
        n += 1
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
        syy += y * y
        out.append((int(n), sx, sy, sxx, sxy, syy))
    return out


def series_readings(grid_months: list[str], grid_log: list[float], daily_from: str,
                    daily: list[tuple[dt.date, float]]) -> tuple[list[tuple], dict]:
    """Seen-on-the-day readings for one series.

    grid_months / grid_log: complete months from 1950-01, log of the real level.
    daily: (date, log real level) for trading days; only dates in months at or
    after daily_from are read daily. Months before daily_from get one monthly
    reading each (dated the 15th).
    A monthly reading for month m uses grid months up to and including m.
    A daily reading on date d uses grid months strictly before d's month, plus d.
    Returns rows (date, z, residual, percentile) and the meta of the last fit.
    """
    xs = [years_since_start(ym_mid(m)) for m in grid_months]
    pre = prefix_sums(xs, grid_log)
    index = {m: k for k, m in enumerate(grid_months)}
    rows: list[tuple] = []
    for k, m in enumerate(grid_months):
        if ym_mid(m) < READINGS_FROM or m >= daily_from:
            continue
        a, b, s = fit_from_sums(*pre[k + 1])
        r = grid_log[k] - (a + b * xs[k])
        below = sum(1 for j in range(k + 1) if grid_log[j] - (a + b * xs[j]) < r)
        rows.append((ym_mid(m), r / s, r, 100.0 * below / (k + 1)))
    meta = {}
    for d, y in daily:
        if d < READINGS_FROM or ym(d) < daily_from:
            continue
        k = index.get(ym(d), len(grid_months))
        n, sx, sy, sxx, sxy, syy = pre[k]
        x = years_since_start(d)
        a, b, s = fit_from_sums(n + 1, sx + x, sy + y, sxx + x * x, sxy + x * y, syy + y * y)
        r = y - (a + b * x)
        below = sum(1 for j in range(k) if grid_log[j] - (a + b * xs[j]) < r)
        rows.append((d, r / s, r, 100.0 * below / (k + 1)))
        meta = {"slope_real_pct_per_year": (math.exp(b) - 1) * 100, "sigma": s, "points_in_fit": k + 1}
    return rows, meta


# ---------------------------------------------------------------- build

def build(shiller: dict, fred: list, cpi: dict, yahoo_tr: list, today: dt.date,
          daily_from: str | None = None) -> dict:
    """Compute fresh readings from the sources. daily_from pins the first month
    read daily (taken from the published file, so the sliding FRED window
    never turns a published daily month back into a monthly one)."""
    notes: list[str] = []
    validate_daily(fred, "fred_sp500", today, min_rows=FRED_MIN_ROWS)
    last_day = fred[-1][0]
    current_month = ym(last_day)
    fred_avg, fred_cnt = monthly_averages(fred)
    fred_months = sorted(m for m in fred_avg if m < current_month and fred_cnt[m] >= MIN_DAYS_FULL_MONTH)
    if not fred_months:
        raise ValidationError("fred_sp500: no complete month")
    fred_first = fred_months[0]
    validate_months(fred_avg, "fred_sp500 months", fred_first, fred_months[-1])
    for m in fred_months:
        if fred_cnt[m] < MIN_DAYS_FULL_MONTH:
            raise ValidationError(f"fred_sp500: month {m} has only {fred_cnt[m]} rows")
    last_complete = fred_months[-1]
    if daily_from is None:
        daily_from = fred_first
    if daily_from > fred_first:
        raise ValidationError(f"published daily_from {daily_from} is later than the data allows ({fred_first})")

    shiller_ok = {m: v for m, v in shiller.items() if v[0] > 0}
    validate_months(shiller_ok, "shiller", ym(START), fred_first)
    cpi = {m: v for m, v in cpi.items() if m >= ym(START)}
    cpi, filled = fill_missing_months(cpi)
    if filled:
        notes.append("cpi months not published, interpolated: " + ", ".join(filled))
    cpi_months = sorted(cpi)
    validate_months(cpi, "cpi", ym(START), cpi_months[-1])
    validate_cpi_steps(cpi)
    if (today - ym_mid(cpi_months[-1])).days > 100:
        raise ValidationError(f"cpi: stale, last month {cpi_months[-1]}")
    cpi_last = cpi_months[-1]

    def cpi_at(m: str) -> float:
        return cpi[m] if m in cpi else cpi[cpi_last]

    deflator_ref = cpi[cpi_last]

    # nominal monthly price grid: Shiller averages, then FRED averages
    grid_months: list[str] = []
    m = ym(START)
    while m <= last_complete:
        grid_months.append(m)
        m = ym_next(m)
    price_nom = {m: (shiller_ok[m][0] if m < fred_first else fred_avg[m]) for m in grid_months}

    overlap = [m for m in fred_months if m in shiller_ok]
    if overlap:
        max_diff = max(abs(math.log(shiller_ok[m][0] / fred_avg[m])) for m in overlap)
        notes.append(f"shiller vs fred monthly average, max log diff over {len(overlap)} months: {max_diff:.4f}")
        if max_diff > 0.10:
            raise ValidationError(f"splice: shiller and fred disagree by {max_diff:.3f} in log units")

    # dividends: known from Shiller, then carried forward as a yield
    div_months = [m for m, v in shiller.items() if v[1] > 0 and v[0] > 0]
    if not div_months:
        raise ValidationError("shiller: no dividend data")
    last_div_month = max(div_months)
    yield_last = shiller[last_div_month][1] / shiller[last_div_month][0]

    def annual_yield(m: str) -> float:
        v = shiller.get(m)
        if v and v[1] > 0 and v[0] > 0 and m <= last_div_month:
            return v[1] / v[0]
        return yield_last

    shiller_tr: dict[str, float] = {}
    prev = None
    for m in grid_months:
        if prev is None:
            shiller_tr[m] = price_nom[m]
        else:
            shiller_tr[m] = shiller_tr[prev] * (price_nom[m] + annual_yield(m) * price_nom[m] / 12) / price_nom[prev]
        prev = m

    # total return: Shiller chain to 1987-12, official index from 1988-01 (required)
    validate_daily(yahoo_tr, "yahoo_tr", today, min_rows=9000)
    y_avg, y_cnt = monthly_averages(yahoo_tr)
    need = [m for m in grid_months if m >= TR_ANCHOR]
    for m in need:
        if m not in y_avg or y_cnt[m] < MIN_DAYS_FULL_MONTH:
            raise ValidationError(f"yahoo_tr: incomplete month {m}")
    scale = shiller_tr[TR_ANCHOR] / y_avg[TR_ANCHOR]
    tr_grid = dict(shiller_tr)
    for m in need:
        tr_grid[m] = y_avg[m] * scale
    check = [m for m in need if m <= last_div_month]
    if check:
        drift = max(abs(math.log(tr_grid[m] / shiller_tr[m])) for m in check)
        notes.append(f"official TR vs shiller TR, max log diff over {len(check)} months: {drift:.4f}")
    tr_daily = {d: c * scale for d, c in yahoo_tr}

    # daily points inside the FRED window, both series, real, log
    daily_price: list[tuple[dt.date, float]] = []
    daily_tr: list[tuple[dt.date, float]] = []
    for d, c in fred:
        if ym(d) < fred_first:
            continue
        if d not in tr_daily:
            raise ValidationError(f"yahoo_tr: no total-return value for {d}")
        defl = deflator_ref / cpi_at(ym(d))
        daily_price.append((d, math.log(c * defl)))
        daily_tr.append((d, math.log(tr_daily[d] * defl)))

    grid_price_log = [math.log(price_nom[m] * deflator_ref / cpi_at(m)) for m in grid_months]
    grid_tr_log = [math.log(tr_grid[m] * deflator_ref / cpi_at(m)) for m in grid_months]

    rows_p, meta_p = series_readings(grid_months, grid_price_log, daily_from, daily_price)
    rows_t, meta_t = series_readings(grid_months, grid_tr_log, daily_from, daily_tr)
    if [r[0] for r in rows_p] != [r[0] for r in rows_t]:
        raise ValidationError("price and total-return readings are on different dates")
    if not rows_p:
        raise ValidationError("no readings produced")

    points = [[rp[0].isoformat(), round(rp[1], 2), round(rt[1], 2), round(rp[3], 1), round(rt[3], 1)]
              for rp, rt in zip(rows_p, rows_t)]
    lp, lt = rows_p[-1], rows_t[-1]
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_date": lp[0].isoformat(),
        "today": {
            "price": {"z": round(lp[1], 2), "pct_vs_trend": round((math.exp(lp[2]) - 1) * 100, 1), "percentile": round(lp[3], 1)},
            "total_return": {"z": round(lt[1], 2), "pct_vs_trend": round((math.exp(lt[2]) - 1) * 100, 1), "percentile": round(lt[3], 1)},
        },
        "meta": {
            "price": {k: round(v, 4) for k, v in meta_p.items()},
            "total_return": {k: round(v, 4) for k, v in meta_t.items()},
            "grid_first_month": grid_months[0],
            "grid_last_complete_month": last_complete,
            "daily_from": daily_from,
            "fred_window_first_month": fred_first,
            "cpi_series": "CPIAUCNS",
            "cpi_last_month": cpi_last,
            "shiller_last_dividend_month": last_div_month,
            "total_return_source": "official-total-return-index",
            "readings_from": READINGS_FROM.isoformat(),
            "notes": notes,
        },
        "columns": ["date", "z_price", "z_total_return", "percentile_price", "percentile_total_return"],
        "points": points,
    }


def merge(previous: dict | None, fresh: dict) -> dict:
    """Append-only merge. Published rows are kept verbatim; rows for new dates
    are added. A recomputed old row that drifts beyond tolerance is an error."""
    if not previous:
        out = dict(fresh)
        out["meta"] = dict(fresh["meta"])
        out["meta"]["first_published_utc"] = fresh["generated_utc"]
        return out
    prev_rows = {r[0]: r for r in previous["points"]}
    prev_last = previous["last_date"]
    drift = []
    for row in fresh["points"]:
        old = prev_rows.get(row[0])
        if old is None:
            continue
        if (abs(row[1] - old[1]) > TOLERANCE_Z or abs(row[2] - old[2]) > TOLERANCE_Z
                or abs(row[3] - old[3]) > TOLERANCE_PCT or abs(row[4] - old[4]) > TOLERANCE_PCT):
            drift.append((row[0], old, row))
    if drift:
        d0, old, new = drift[0]
        raise ValidationError(f"{len(drift)} published rows would change beyond tolerance; first {d0}: {old} -> {new}")
    new_rows = [r for r in fresh["points"] if r[0] > prev_last]
    out = dict(fresh)
    out["points"] = list(previous["points"]) + new_rows
    out["meta"] = dict(fresh["meta"])
    out["meta"]["first_published_utc"] = previous["meta"].get("first_published_utc", previous.get("generated_utc"))
    if new_rows:
        out["meta"]["rows_added"] = len(new_rows)
    else:
        out["last_date"] = prev_last
        out["today"] = previous["today"]
    return out


# ---------------------------------------------------------------- cli

def load_previous() -> dict | None:
    if not os.path.exists(OUT):
        return None
    with open(OUT) as f:
        return json.load(f)


def cmd_build(today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    previous = load_previous()
    shiller = parse_shiller(read_cache("shiller"))
    fred = parse_fred(read_cache("fred_sp500"), "fred_sp500")
    cpi = {ym(d): v for d, v in parse_fred(read_cache("fred_cpi"), "fred_cpi")}
    yahoo = parse_yahoo(read_cache("yahoo_tr"))
    pinned = previous["meta"].get("daily_from") if previous else None
    fresh = build(shiller, fred, cpi, yahoo, today, daily_from=pinned)
    data = merge(previous, fresh)
    if previous and data["points"] == previous["points"] and data["today"] == previous["today"]:
        print(f"no change: {len(data['points'])} points, last date {data['last_date']}; file left as is")
        return previous
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    t = data["today"]
    print(f"wrote {OUT}: {len(data['points'])} points, last date {data['last_date']}"
          + (f", {data['meta'].get('rows_added', 0)} rows added" if previous else ", first publication"))
    print(f"  price only    z={t['price']['z']:+.2f}  {t['price']['pct_vs_trend']:+.1f}% vs trend  percentile {t['price']['percentile']}")
    print(f"  total return  z={t['total_return']['z']:+.2f}  {t['total_return']['pct_vs_trend']:+.1f}% vs trend  percentile {t['total_return']['percentile']}")
    print(f"  meta: {json.dumps(data['meta'])}")
    return data


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "update"
    try:
        if cmd == "fetch":
            cmd_fetch()
        elif cmd == "build":
            cmd_build()
        elif cmd == "update":
            cmd_fetch()
            cmd_build()
        else:
            print(__doc__)
            return 1
    except ValidationError as e:
        print(f"validation failed, nothing written: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
