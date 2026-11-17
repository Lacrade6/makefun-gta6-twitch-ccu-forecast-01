# GTA VI launch-day Twitch CCU forecaster

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), as five quantiles an hour, from
the first hour's print. Entry for make.fun bounty
`gta6-twitch-ccu-forecast-01`.

## Model: size-conditioned launch curves

A launch's release-day audience curve is mostly decided by how big its
first hour is: hour 1 (00:00-01:00 UTC) only catches the timezones that
have unlocked by then, and the rest of the day is the world waking up and
unlocking. Small launches grow tenfold or more over the day; the biggest
launches are already near their peak in hour 1 and stay flat. Across the
past launches in the case file, the correlation between log hour-1 viewers
and the day's peak log-ratio is about -0.94.

For each hour h = 2..24, the forecast is a kernel-weighted local-linear fit
of the log-ratio ln(viewers_h / viewers_1) on ln(viewers_1) over every
comparable launch's release day (Gaussian kernel, bandwidth 1.8, evaluated
at the case's log hour-1 print, clamped to the comparables' observed range
so an unprecedented hour-1 print predicts the flattest observed shape
instead of extrapolating a decline). The spread is the kernel-weighted
residual standard deviation around that fit, shrunk by 0.7, with a floor of
0.08 in log units.

## Local validation

Practice answers are SullyGnome's public hourly averages (the same source
the house used), fetched for the four practice launches only:

- mean pinball loss on the four practice cases: **0.173**
  (house baseline: 0.387 on the same cases)
- leave-one-out over the 19 past launches with hour-1 viewers above 100k
  (the regime GTA VI will be in): **0.20**

## Run

`python3 main.py` reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). Python 3.13, standard library only, deterministic, no network.
