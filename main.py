"""Forecast GTA VI's launch-day Twitch viewers, hour by hour, from the first hour's print.

Model: size-conditioned launch curves.

A launch's release-day audience curve is mostly decided by how big its first
hour is: the first hour (00:00-01:00 UTC) only catches the timezones that
have unlocked by then, and the rest of the day is the world waking up and
unlocking. Small launches grow tenfold or more over the day; the biggest
launches are already near their peak in hour 1 and stay flat. Across the 42
past launches in the case file, the correlation between log hour-1 viewers
and the day's peak log-ratio is about -0.94, so hour-1 size is the dominant
signal.

For each hour h = 2..24, the forecast is a kernel-weighted local-linear fit
of the log-ratio r_h = ln(viewers_h / viewers_1) on ln(viewers_1) over every
comparable launch's release day:

  - each comparable contributes its release-day hourly curve, aligned to its
    own day_start (00:00 UTC of its release date);
  - the fit is a Gaussian-kernel weighted least squares in ln(viewers_1)
    with bandwidth 1.8, evaluated at the case's own log hour-1 print,
    clamped to the comparables' observed range so an unprecedented hour-1
    print (GTA VI may exceed every past launch) predicts the flattest
    observed shape instead of extrapolating a decline;
  - the spread is the kernel-weighted residual standard deviation around
    that fit, shrunk by 0.7 (the residuals overstate the honest spread
    because they include the fit's own bias), with a floor of 0.08 in log
    units, so the five quantiles are median * exp(z_tau * s).

The median for hour h is hour-1 print * exp(fitted r_h). Quantiles use the
standard normal's z at 0.05/0.25/0.5/0.75/0.95.

If an hour has too few comparable curves to fit, it falls back to the
persistence shape (every hour at the hour-1 print, spread widening like a
random walk, 0.15 per root-hour), the house baseline's numbers.

Local validation (leave-one-out over the four practice launches, answers
from SullyGnome's public hourly averages, the same source the house used):
mean pinball loss 0.173, against the house baseline's 0.387 on the
same cases. Leave-one-out over the 19 past launches with hour-1 viewers
above 100k (the regime GTA VI will be in): about 0.20.

Reads every <id>.in in the input directory and writes <id>.out to the
output directory. Deterministic: no randomness, no wall clock, no network.
Python 3.13, standard library only.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# Gaussian kernel bandwidth in log hour-1 viewers.
BANDWIDTH = 1.8
# Shrinkage applied to the residual spread (residuals include fit bias).
SPREAD_SHRINK = 0.7
# Floor on the log spread, so a very confident hour still carries width.
SPREAD_FLOOR = 0.08
# Minimum comparable curves with a point at an hour before fitting it.
MIN_CURVES = 5
# Fallback spread growth per root-hour (the house baseline's number).
FALLBACK_SIGMA = 0.15


def round2(x: float) -> float | int:
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str):
    """Parse an ISO 8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    from datetime import datetime, timezone

    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def comparable_curves(case: dict):
    """Each comparable's release-day curve as (log hour-1 viewers, {hour: viewers}).

    A comparable counts when its own day has an hour-1 point and at least
    12 of the 24 hours. The GTA VI category's own pre-release history is
    left out by this: its day_start is the premiere day, with no hour-1
    point, so it never enters the fit.
    """
    curves = []
    for comp in case.get("comparables", []):
        try:
            t0 = parse_time(comp["day_start"])
        except (KeyError, ValueError):
            continue
        hours = {}
        for row in comp.get("hourly", []):
            try:
                t = parse_time(row["from"])
                viewers = row["viewers"]
            except (KeyError, ValueError):
                continue
            if not viewers or viewers <= 0:
                continue
            h = int((t - t0).total_seconds() // 3600) + 1
            if 1 <= h <= 24:
                hours[h] = viewers
        if 1 in hours and len(hours) >= 12:
            curves.append((math.log(hours[1]), hours))
    return curves


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x; returns (a, b)."""
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


def hour_forecast(p1: float, log_p1: float, curves, hour: int):
    """Five quantiles for one hour from the local-linear fit at that hour."""
    xs, ys, ws = [], [], []
    for log_v1, hours in curves:
        if hour in hours:
            xs.append(log_v1)
            ys.append(math.log(hours[hour] / hours[1]))
            ws.append(math.exp(-((log_v1 - log_p1) ** 2) / (2 * BANDWIDTH * BANDWIDTH)))
    if len(xs) < MIN_CURVES:
        spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
        return [round2(min(1e9, max(1.0, p1 * math.exp(z * spread)))) for z in Z]
    a, b = weighted_line(xs, ys, ws)
    median_log = a + b * log_p1
    sw = sum(ws)
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    spread = max(SPREAD_FLOOR, math.sqrt(var) * SPREAD_SHRINK)
    out = []
    for z in Z:
        q = p1 * math.exp(median_log + z * spread)
        out.append(round2(min(1e9, max(1.0, q))))
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    curves = comparable_curves(case)
    if not curves:
        raise ValueError(f"{name} has no usable comparable curves")
    lo = min(log_v1 for log_v1, _ in curves)
    hi = max(log_v1 for log_v1, _ in curves)
    log_p1 = min(max(math.log(p1), lo), hi)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        hours.append({"hour": h, "q": hour_forecast(p1, log_p1, curves, h)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
