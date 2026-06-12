"""
Phase 1 — Event catalog.

Build the master event log: for each of 10 mechanical event types, scan every
M1 bar across all 5 cached pairs, record event occurrences with their
timestamp + features + forward returns at N = 5, 15, 30, 60 bars.

Events fire on the LAST bar of the pattern so forward returns from that bar
are unbiased (no look-ahead).

Output: tools/event_log.parquet (long format: one row per event occurrence)

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/event_catalog.py
"""
from __future__ import annotations

import sys
import time
import calendar
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

CACHE_DIR  = PROJECT_ROOT / "edge_prepared_cache"
OUT_PATH   = PROJECT_ROOT / "tools" / "event_log.parquet"

# Forward return horizons (M1 bars)
HORIZONS = [5, 15, 30, 60]

# Round-number increment per pair (in price units, not pips).
# Majors: 0.0100 = 100 pips. JPY-quote pairs: 1.0000 = 100 pips.
ROUND_INC = {
    "EUR_USD": 0.01,
    "GBP_USD": 0.01,
    "AUD_USD": 0.01,
    "USD_JPY": 1.0,
    "EUR_JPY": 1.0,
}


# =============================================================================
# Event detection functions
# Each takes (df, pair) and returns a boolean Series aligned to df.index.
# =============================================================================

def evt_range_sweep_close_inside_15m(df: pd.DataFrame, pair: str) -> pd.Series:
    """Within the last 15 M1 bars, did high exceed asian_high (or low go below
    asian_low) AND the current bar closes back inside the range by
    ≥0.3·range_size? Fires on bars at minute=0/15/30/45 only."""
    ah = df["asian_high"]
    al = df["asian_low"]
    rng = ah - al

    high_swept = (df["high"].rolling(15, min_periods=1).max() > ah)
    low_swept  = (df["low"].rolling(15,  min_periods=1).min() < al)

    closed_inside_below_high = df["close"] < ah - 0.3 * rng
    closed_inside_above_low  = df["close"] > al + 0.3 * rng

    swept_high = high_swept & closed_inside_below_high
    swept_low  = low_swept  & closed_inside_above_low

    m15_aligned = (df["minute"] % 15 == 0)
    valid = rng > 0
    return (swept_high | swept_low) & m15_aligned & valid


def evt_round_number_touch(df: pd.DataFrame, pair: str) -> pd.Series:
    """Bar's low ≤ round_level ≤ high (round_level = nearest 100-pip / 1000-pip
    increment) AND next 5 bars all stay within ±0.2·ATR of the touched level.
    Fires on bar 5 (the last dwell bar)."""
    inc = ROUND_INC.get(pair, 0.01)
    # Nearest round level for each bar's mid
    mid = (df["high"] + df["low"]) / 2
    rl = (mid / inc).round() * inc
    touched = (df["low"] <= rl) & (rl <= df["high"])

    # The touch we care about happened 5 bars ago. Shift to align "touch at i-5"
    # with current bar i.
    touched_shift5 = touched.shift(5).fillna(False).astype(bool)
    rl_shift5      = rl.shift(5)
    atr_shift5     = df["atr"].shift(5)

    # Dwell: bars i-4..i all within ±0.2·ATR of the touched level.
    dwell_ok = pd.Series(True, index=df.index)
    for k in range(0, 5):
        # bar i-4+k = shift(4-k); k=0 → shift(4); k=4 → shift(0) = current
        shifted_close = df["close"].shift(4 - k) if k < 4 else df["close"]
        dwell_ok = dwell_ok & ((shifted_close - rl_shift5).abs() < 0.2 * atr_shift5)

    return touched_shift5 & dwell_ok.fillna(False)


def _compute_session_range(df: pd.DataFrame, hour_lo: int, hour_hi: int) -> tuple[pd.Series, pd.Series]:
    """Within each date, compute the cummax/cummin of high/low across bars
    where hour_lo ≤ hour < hour_hi. Returns (cum_high, cum_low) aligned to df.index,
    NaN outside the window."""
    in_session = df["hour"].between(hour_lo, hour_hi - 1)
    hi_in   = df["high"].where(in_session)
    lo_in   = df["low"].where(in_session)
    cum_h   = hi_in.groupby(df["date"]).cummax()
    cum_l   = lo_in.groupby(df["date"]).cummin()
    return cum_h, cum_l


def evt_first_m15_sweep_of_prior_session(df: pd.DataFrame, pair: str) -> pd.Series:
    """At the first M15-aligned bar of a session (07:00, 13:00, 21:00 UTC),
    did the prior 5 M1 bars sweep the prior session's range?

    For London open (07:00): asian range = asian_high/asian_low (in data).
    For NY open (13:00): london range = cummax/cummin over hours [7, 13).
    For Asian open (21:00): NY range = cummax/cummin over hours [13, 21).
    """
    hour = df["hour"]
    minute = df["minute"]

    # Forward-fill the previous session's range up to its session-open bar.
    london_high, london_low = _compute_session_range(df, 7, 13)
    ny_high,     ny_low     = _compute_session_range(df, 13, 21)

    # Forward-fill so the session-open bar can see the final range
    london_high = london_high.ffill()
    london_low  = london_low.ffill()
    ny_high     = ny_high.ffill()
    ny_low      = ny_low.ffill()

    # London open (07:00:00): prior 5 bars (02:55-02:59 wait, that's wrong —
    # bars at hour=6, minute=55..59). Compare against asian range (asian_high/low).
    prior_high_5 = df["high"].shift(1).rolling(5, min_periods=1).max()
    prior_low_5  = df["low"].shift(1).rolling(5, min_periods=1).min()

    london_open  = (hour == 7)  & (minute == 0) & (
        (prior_high_5 > df["asian_high"]) | (prior_low_5 < df["asian_low"])
    )
    ny_open      = (hour == 13) & (minute == 0) & (
        (prior_high_5 > london_high) | (prior_low_5 < london_low)
    )
    asian_open   = (hour == 21) & (minute == 0) & (
        (prior_high_5 > ny_high) | (prior_low_5 < ny_low)
    )
    return london_open | ny_open | asian_open


def evt_gap_open(df: pd.DataFrame, pair: str) -> pd.Series:
    """First bar after a >2-hour timestamp gap (weekend/holiday) where
    |open − prior_close| ≥ 0.3·ATR."""
    ts_diff = df["timestamp"].diff().dt.total_seconds()
    gap = ts_diff > 7200  # > 2 hours
    move = (df["open"] - df["close"].shift(1)).abs()
    return gap & (move >= 0.3 * df["atr"])


def evt_nfp_wednesday_close(df: pd.DataFrame, pair: str) -> pd.Series:
    """Wednesday's 21:00 UTC bar in NFP week (first Friday of the month)."""
    ts = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    dt = ts.dt
    is_wed_21 = (dt.weekday == 2) & (df["hour"] == 21) & (df["minute"] == 0)

    # The Friday in this week (date + (4 - weekday) days)
    weekday = dt.weekday
    days_to_friday = (4 - weekday).astype(int)
    friday_date = (ts + pd.to_timedelta(days_to_friday, unit="D")).dt.date
    friday_day = pd.Series([d.day if d is not None else 0 for d in friday_date], index=df.index)
    is_first_friday = friday_day <= 7  # first Friday is always day 1-7
    return is_wed_21 & is_first_friday


def evt_month_end_london_close(df: pd.DataFrame, pair: str) -> pd.Series:
    """Bar at 14:00-16:00 UTC on the last business day of each month."""
    ts = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    in_window = df["hour"].between(14, 15) & (df["minute"] == 0)
    # Last business day of month: date == last business date of its (year, month)
    def _is_last_bday(d):
        if d is None:
            return False
        year, month = d.year, d.month
        last = calendar.monthrange(year, month)[1]
        # Step backward until weekday < 5 (Mon-Fri)
        for day in range(last, last - 4, -1):
            if pd.Timestamp(year=year, month=month, day=day).weekday() < 5:
                return d.day == day
        return False
    last_bday = df["date"].apply(_is_last_bday)
    return in_window & last_bday


def evt_spread_spike_then_calm(df: pd.DataFrame, pair: str) -> pd.Series:
    """Bar with spread_mean > 2·spread_median followed by 5 bars all with
    spread_mean < spread_median. Fires on bar 5 (last calm bar)."""
    sm  = df["spread_mean"]
    smd = df["spread_median"]
    spike_at_i_minus5 = (sm.shift(5) > 2 * smd.shift(5)).fillna(False)
    calm = pd.Series(True, index=df.index)
    # bars i-4 .. i all below median
    for k in range(0, 5):
        # k=0 → bar i-4 (shift 4); k=4 → bar i (no shift)
        offset = 4 - k
        calm = calm & (sm.shift(offset) < smd.shift(offset))
    return spike_at_i_minus5 & calm.fillna(False)


def evt_tick_imb_streak_3(df: pd.DataFrame, pair: str) -> pd.Series:
    """3 consecutive bars with tick_imbalance > 2·rolling-std(tick_imbalance, 60).
    Fires on the 3rd bar."""
    ti = df["tick_imbalance"]
    rstd = ti.rolling(60, min_periods=20).std()
    big = (ti.abs() > 2 * rstd).fillna(False)
    streak3 = big.rolling(3, min_periods=3).sum() == 3
    return streak3.fillna(False)


def evt_atr_regime_shift(df: pd.DataFrame, pair: str) -> pd.Series:
    """atr_ratio crosses from <0.8 to >1.5 (or vice versa). Fires on crossing bar."""
    ar  = df["atr_ratio"]
    prev = ar.shift(1)
    up   = (prev < 0.8) & (ar > 1.5)
    down = (prev > 1.5) & (ar < 0.8)
    return (up | down).fillna(False)


def evt_adx_trend_entry(df: pd.DataFrame, pair: str) -> pd.Series:
    """ADX crosses from <20 to >25. Fires on crossing bar."""
    adx = df["adx"]
    return ((adx.shift(1) < 20) & (adx > 25)).fillna(False)


EVENT_DETECTORS = {
    "range_sweep_close_inside_15m":      evt_range_sweep_close_inside_15m,
    "round_number_touch":                evt_round_number_touch,
    "first_m15_sweep_of_prior_session":  evt_first_m15_sweep_of_prior_session,
    "gap_open":                          evt_gap_open,
    "nfp_wednesday_close":               evt_nfp_wednesday_close,
    "month_end_london_close":            evt_month_end_london_close,
    "spread_spike_then_calm":            evt_spread_spike_then_calm,
    "tick_imb_streak_3":                 evt_tick_imb_streak_3,
    "atr_regime_shift":                  evt_atr_regime_shift,
    "adx_trend_entry":                   evt_adx_trend_entry,
}


# =============================================================================
# Catalog scanner
# =============================================================================

def scan_pair(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Return long-format event log for one pair."""
    if "ts" not in df.columns and "timestamp" in df.columns:
        df = df.copy()
        df["ts"] = df["timestamp"]

    # Precompute forward returns at every horizon (raw price diff)
    fwd_rets = {N: df["close"].shift(-N) - df["close"] for N in HORIZONS}

    rows = []
    for event_type, detect in EVENT_DETECTORS.items():
        try:
            t0 = time.time()
            mask = detect(df, pair).fillna(False).astype(bool)
            # Drop occurrences too close to end of data (forward returns NaN)
            max_h = max(HORIZONS)
            mask.iloc[-max_h:] = False
            occ_idx = mask[mask].index
            elapsed = time.time() - t0
            if len(occ_idx) == 0:
                print(f"  [{pair}] {event_type}: 0 occurrences ({elapsed:.1f}s)")
                continue

            sub = pd.DataFrame({
                "event_type":    event_type,
                "pair":          pair,
                "timestamp":     df.loc[occ_idx, "timestamp"].values,
                "hour":          df.loc[occ_idx, "hour"].astype(int).values,
                "dow":           pd.to_datetime(df.loc[occ_idx, "timestamp"]).dt.tz_localize(None).dt.weekday.values,
                "regime":        df.loc[occ_idx, "regime"].astype(str).values if "regime" in df.columns else "UNDEFINED",
                "atr_at_event":  df.loc[occ_idx, "atr"].values,
            })
            for N in HORIZONS:
                sub[f"forward_ret_{N}"] = fwd_rets[N].loc[occ_idx].values
            # Drop rows where any forward return is NaN
            sub = sub.dropna(subset=[f"forward_ret_{N}" for N in HORIZONS])
            rows.append(sub)
            print(f"  [{pair}] {event_type}: {len(sub):>6} occurrences ({elapsed:.1f}s)")
        except Exception as e:
            import traceback
            print(f"  [{pair}] {event_type}: ERROR — {type(e).__name__}: {e}")
            traceback.print_exc()
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    t0 = time.time()
    parquets = sorted(CACHE_DIR.glob("*_m1.parquet"))
    print(f"[catalog] Found {len(parquets)} cached pairs")

    all_events = []
    for fp in parquets:
        pair = fp.stem.replace("_m1", "")
        print(f"\n[catalog] === {pair} ===")
        ts0 = time.time()
        df = pd.read_parquet(fp)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        print(f"  loaded {len(df):,} bars in {time.time()-ts0:.1f}s")
        event_df = scan_pair(df, pair)
        if not event_df.empty:
            all_events.append(event_df)

    if not all_events:
        print("\n[catalog] NO events detected across any pair. Detection logic broken.")
        return 1

    out = pd.concat(all_events, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    elapsed = time.time() - t0
    print(f"\n[catalog] DONE in {elapsed/60:.1f} min")
    print(f"  Total events: {len(out):,}")
    print(f"  Output: {OUT_PATH}")

    print("\n[catalog] Event-type frequency table:")
    summary = (out.groupby(["event_type", "pair"]).size()
                  .unstack(fill_value=0))
    print(summary.to_string())
    print(f"\n  Total by event_type:")
    print(out["event_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
