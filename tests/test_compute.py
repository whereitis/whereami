"""Unit tests for compute.py. Run: python3 -m unittest discover -s tests -v"""
import datetime as dt
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compute  # noqa: E402


def direct_fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, math.sqrt(sum(r * r for r in res) / n)


def months(first, count):
    out = [first]
    for _ in range(count - 1):
        out.append(compute.ym_next(out[-1]))
    return out


class FitTests(unittest.TestCase):
    def test_running_sums_match_direct_fit(self):
        rnd = random.Random(1)
        xs = [i / 12 for i in range(500)]
        ys = [0.03 * x + rnd.gauss(0, 0.3) for x in xs]
        pre = compute.prefix_sums(xs, ys)
        for k in (10, 123, 500):
            a, b, s = compute.fit_from_sums(*pre[k])
            a2, b2, s2 = direct_fit(xs[:k], ys[:k])
            self.assertAlmostEqual(a, a2, places=9)
            self.assertAlmostEqual(b, b2, places=9)
            self.assertAlmostEqual(s, s2, places=9)

    def test_slope_recovered_on_exponential_growth(self):
        xs = [i / 12 for i in range(921)]
        ys = [math.log(100 * math.exp(0.04 * x)) for x in xs]
        _, b, s = compute.fit_from_sums(*compute.prefix_sums(xs, ys)[-1])
        self.assertAlmostEqual(b, 0.04, places=9)
        self.assertLess(s, 1e-6)  # running sums lose a little precision; real sigma is about 0.3


class ReadingTests(unittest.TestCase):
    def synthetic(self, seed=7, n_months=800, daily_days=400):
        rnd = random.Random(seed)
        grid = months("1950-01", n_months)
        log = [math.log(20 * math.exp(0.035 * compute.years_since_start(compute.ym_mid(m))))
               + 0.3 * math.sin(i / 40) + rnd.gauss(0, 0.02) for i, m in enumerate(grid)]
        daily_from = grid[-24]
        d = compute.ym_mid(daily_from).replace(day=1)
        daily = []
        for _ in range(daily_days):
            if d.weekday() < 5:
                daily.append((d, math.log(20 * math.exp(0.035 * compute.years_since_start(d))) + rnd.gauss(0, 0.05)))
            d += dt.timedelta(days=1)
        return grid, log, daily_from, daily

    def test_past_readings_do_not_change_when_future_arrives(self):
        grid, log, daily_from, daily = self.synthetic()
        rows_all, _ = compute.series_readings(grid, log, daily_from, daily)
        cut = 300
        rows_cut, _ = compute.series_readings(grid[:cut], log[:cut], daily_from, [])
        self.assertEqual(len(rows_cut), sum(1 for r in rows_all if r[0] <= compute.ym_mid(grid[cut - 1])))
        for a, b in zip(rows_cut, rows_all):
            self.assertEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1], places=12)
            self.assertAlmostEqual(a[3], b[3], places=12)

    def test_daily_reading_uses_only_earlier_months_plus_itself(self):
        grid, log, daily_from, daily = self.synthetic()
        rows, meta = compute.series_readings(grid, log, daily_from, daily)
        d, z, r, pct = rows[-1]
        k = grid.index(compute.ym(d)) if compute.ym(d) in grid else len(grid)
        xs = [compute.years_since_start(compute.ym_mid(m)) for m in grid[:k]] + [compute.years_since_start(d)]
        ys = log[:k] + [daily[-1][1]]
        a, b, s = direct_fit(xs, ys)
        self.assertAlmostEqual(z, (ys[-1] - (a + b * xs[-1])) / s, places=9)
        self.assertEqual(meta["points_in_fit"], k + 1)

    def test_percentile_matches_definition(self):
        grid, log, daily_from, daily = self.synthetic()
        rows, _ = compute.series_readings(grid, log, daily_from, daily)
        for _, z, _, pct in rows:
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 100.0)
        d, z, r, pct = rows[-1]
        k = grid.index(compute.ym(d)) if compute.ym(d) in grid else len(grid)
        xs = [compute.years_since_start(compute.ym_mid(m)) for m in grid[:k]] + [compute.years_since_start(d)]
        ys = log[:k] + [daily[-1][1]]
        a, b, s = direct_fit(xs, ys)
        res = [y - (a + b * x) for x, y in zip(xs, ys)]
        self.assertAlmostEqual(pct, 100.0 * sum(1 for v in res[:-1] if v < res[-1]) / len(res), places=9)
        d, z, r, pct = rows[100]
        k = grid.index(compute.ym(d))
        xs = [compute.years_since_start(compute.ym_mid(m)) for m in grid[:k + 1]]
        a, b, s = direct_fit(xs, log[:k + 1])
        res = [y - (a + b * x) for x, y in zip(xs, log[:k + 1])]
        self.assertAlmostEqual(pct, 100.0 * sum(1 for v in res if v < res[-1]) / len(res), places=9)

    def test_readings_start_in_1970(self):
        grid, log, daily_from, daily = self.synthetic()
        rows, _ = compute.series_readings(grid, log, daily_from, daily)
        self.assertGreaterEqual(rows[0][0], dt.date(1970, 1, 1))
        self.assertLess(rows[0][0], dt.date(1970, 3, 1))

    def test_pinned_daily_from_skips_monthly_rows_it_cannot_read_daily(self):
        grid, log, daily_from, daily = self.synthetic()
        earlier = grid[-30]  # pinned earlier than the daily data reaches
        rows, _ = compute.series_readings(grid, log, earlier, daily)
        dates = [r[0] for r in rows]
        self.assertNotIn(compute.ym_mid(grid[-28]), dates)
        self.assertIn(compute.ym_mid(grid[-31]), dates)


class ValidationTests(unittest.TestCase):
    def daily(self, n=2600, start=dt.date(2016, 1, 4)):
        rows, d = [], start
        while len(rows) < n:
            if d.weekday() < 5:
                rows.append((d, 2000.0 + len(rows)))
            d += dt.timedelta(days=1)
        return rows

    def test_accepts_clean_series(self):
        rows = self.daily()
        compute.validate_daily(rows, "x", rows[-1][0])

    def test_rejects_gap(self):
        rows = self.daily()
        rows = rows[:1000] + [(d + dt.timedelta(days=30), v) for d, v in rows[1000:]]
        with self.assertRaises(compute.ValidationError):
            compute.validate_daily(rows, "x", rows[-1][0])

    def test_rejects_nonpositive(self):
        rows = self.daily()
        rows[500] = (rows[500][0], 0.0)
        with self.assertRaises(compute.ValidationError):
            compute.validate_daily(rows, "x", rows[-1][0])

    def test_rejects_stale(self):
        rows = self.daily()
        with self.assertRaises(compute.ValidationError):
            compute.validate_daily(rows, "x", rows[-1][0] + dt.timedelta(days=30))

    def test_rejects_too_short(self):
        rows = self.daily(n=100)
        with self.assertRaises(compute.ValidationError):
            compute.validate_daily(rows, "x", rows[-1][0])

    def test_missing_month_detected(self):
        present = {m: 1 for m in months("1950-01", 100)}
        del present["1955-06"]
        with self.assertRaises(compute.ValidationError):
            compute.validate_months(present, "x", "1950-01", "1958-04")

    def test_fill_missing_month_interpolates_in_log(self):
        series = {m: 100.0 * (1.01 ** i) for i, m in enumerate(months("2025-01", 12))}
        gone = series.pop("2025-10")
        filled, names = compute.fill_missing_months(series)
        self.assertEqual(names, ["2025-10"])
        self.assertAlmostEqual(filled["2025-10"], gone, places=9)

    def test_fill_rejects_long_gap(self):
        series = {m: 100.0 for m in months("2025-01", 12)}
        for m in ("2025-05", "2025-06", "2025-07", "2025-08"):
            del series[m]
        with self.assertRaises(compute.ValidationError):
            compute.fill_missing_months(series)

    def test_cpi_jump_rejected(self):
        series = {m: 100.0 for m in months("2025-01", 12)}
        series["2025-06"] = 130.0
        with self.assertRaises(compute.ValidationError):
            compute.validate_cpi_steps(series)

    def test_yahoo_error_body_is_validation_error(self):
        with self.assertRaises(compute.ValidationError):
            compute.parse_yahoo('{"chart":{"result":null,"error":{"code":"Not Found"}}}')
        with self.assertRaises(compute.ValidationError):
            compute.parse_yahoo("<html>rate limited</html>")


def synthetic_sources(today=dt.date(2026, 9, 2), fred_start=dt.date(2016, 8, 1), drop_yahoo_last=False):
    rnd = random.Random(3)
    grid = months("1950-01", 921)  # to 2026-09
    shiller = {}
    for i, m in enumerate(grid):
        p = 20 * math.exp(0.07 * i / 12) * (1 + 0.2 * math.sin(i / 30))
        div = 0.0 if m > "2023-06" else p * 0.03
        shiller[m] = (p if m <= "2026-08" else 0.0, div)
    fred, d = [], fred_start
    while d <= today:
        if d.weekday() < 5:
            i = (d.year - 1950) * 12 + d.month - 1
            fred.append((d, 20 * math.exp(0.07 * i / 12) * (1 + 0.2 * math.sin(i / 30)) * (1 + rnd.gauss(0, 0.005))))
        d += dt.timedelta(days=1)
    cpi = {m: 24 * math.exp(0.035 * i / 12) for i, m in enumerate(grid) if m <= "2026-07"}
    yahoo, d = [], dt.date(1988, 1, 4)
    while d <= today:
        if d.weekday() < 5:
            i = (d.year - 1950) * 12 + d.month - 1
            yahoo.append((d, 1000 * math.exp(0.10 * i / 12) * (1 + 0.2 * math.sin(i / 30))))
        d += dt.timedelta(days=1)
    if drop_yahoo_last:
        yahoo = yahoo[:-1]
    return shiller, fred, cpi, yahoo, today


class BuildTests(unittest.TestCase):
    """End-to-end on synthetic sources shaped like the real ones."""

    def test_build_shape(self):
        data = compute.build(*synthetic_sources())
        self.assertEqual(data["meta"]["total_return_source"], "official-total-return-index")
        self.assertEqual(data["meta"]["daily_from"], "2016-08")  # August 2016 has 23 weekdays, so it counts as complete
        self.assertEqual(data["meta"]["grid_last_complete_month"], "2026-08")
        self.assertEqual(data["last_date"], "2026-09-02")
        self.assertEqual(data["points"][0][0], "1970-01-15")
        for row in data["points"]:
            self.assertEqual(len(row), 5)
        self.assertGreater(len(data["points"]), 2500)

    def test_missing_yahoo_day_fails_closed(self):
        with self.assertRaises(compute.ValidationError):
            compute.build(*synthetic_sources(drop_yahoo_last=True))

    def test_build_rejects_broken_splice(self):
        shiller, fred, cpi, yahoo, today = synthetic_sources()
        shiller = {m: (p * (3.0 if "2017-01" <= m <= "2017-12" else 1.0), d) for m, (p, d) in shiller.items()}
        with self.assertRaises(compute.ValidationError):
            compute.build(shiller, fred, cpi, yahoo, today)

    def test_output_has_no_raw_levels(self):
        data = compute.build(*synthetic_sources())
        text = str(data)
        for key in ("close", "price_nom", "level"):
            self.assertNotIn(key, text)

    def test_pinned_daily_from_survives_sliding_window(self):
        first = compute.build(*synthetic_sources())
        # a later run where FRED's window has slid forward two months
        later = compute.build(*synthetic_sources(fred_start=dt.date(2016, 10, 1)), daily_from=first["meta"]["daily_from"])
        self.assertEqual(later["meta"]["daily_from"], "2016-08")
        self.assertEqual(later["meta"]["fred_window_first_month"], "2016-10")
        dates = {r[0] for r in later["points"]}
        self.assertNotIn("2016-08-15", dates)  # no monthly row sneaks in for a month published daily
        self.assertNotIn("2016-09-15", dates)


class MergeTests(unittest.TestCase):
    def test_first_publication_passes_through(self):
        fresh = compute.build(*synthetic_sources())
        out = compute.merge(None, fresh)
        self.assertEqual(out["points"], fresh["points"])
        self.assertIn("first_published_utc", out["meta"])

    def test_published_rows_are_kept_and_new_dates_appended(self):
        earlier = compute.merge(None, compute.build(*synthetic_sources(today=dt.date(2026, 8, 20))))
        fresh = compute.build(*synthetic_sources(), daily_from=earlier["meta"]["daily_from"])
        out = compute.merge(earlier, fresh)
        n_old = len(earlier["points"])
        self.assertEqual(out["points"][:n_old], earlier["points"])
        self.assertGreater(len(out["points"]), n_old)
        self.assertTrue(all(r[0] > earlier["last_date"] for r in out["points"][n_old:]))
        self.assertEqual(out["last_date"], "2026-09-02")
        self.assertEqual(out["meta"]["rows_added"], len(out["points"]) - n_old)

    def test_no_new_dates_keeps_today_and_last_date(self):
        published = compute.merge(None, compute.build(*synthetic_sources()))
        fresh = compute.build(*synthetic_sources(), daily_from=published["meta"]["daily_from"])
        out = compute.merge(published, fresh)
        self.assertEqual(out["points"], published["points"])
        self.assertEqual(out["today"], published["today"])
        self.assertEqual(out["last_date"], published["last_date"])

    def test_drift_beyond_tolerance_fails(self):
        published = compute.merge(None, compute.build(*synthetic_sources()))
        fresh = compute.build(*synthetic_sources())
        fresh["points"][500] = list(fresh["points"][500])
        fresh["points"][500][1] += 0.2
        with self.assertRaises(compute.ValidationError):
            compute.merge(published, fresh)

    def test_drift_within_tolerance_is_ignored(self):
        published = compute.merge(None, compute.build(*synthetic_sources()))
        fresh = compute.build(*synthetic_sources())
        fresh["points"][500] = list(fresh["points"][500])
        fresh["points"][500][1] += 0.01
        out = compute.merge(published, fresh)
        self.assertEqual(out["points"][500], published["points"][500])


if __name__ == "__main__":
    unittest.main()
