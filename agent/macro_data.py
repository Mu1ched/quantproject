"""
Phase 9 — free-data expansion.

Five sources, all free, all best-effort cached to disk. Each loader:
  - Caches to agent/macro_cache/<source>_<key>.parquet (or .json)
  - Returns an empty DataFrame on any failure (network, parse, missing dep)
  - Is idempotent within MACRO_CACHE_TTL_HOURS

Sources:
  1. ForexFactory weekly XML  → high-impact event times per pair
  2. CFTC Commitments of Traders (COT) → cot_net_position_pct, cot_extreme_flag
  3. Yahoo Finance daily macro → DXY, US10Y, SPX, gold, VIX
  4. OANDA retail positioning → retail_long_pct (contrarian fade signal)
  5. Multi-pair Dukascopy is already supported by edge_engine.fetch_dukascopy
     so no new code is needed here — just expose the list of pairs.

Wired into edge_engine.prepare_df via merge_macro_features().
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / 'macro_cache'
CACHE_DIR.mkdir(exist_ok=True)
MACRO_CACHE_TTL_HOURS = 12

_SESSION = requests.Session()
_SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (Quantproject macro fetch)'})

# ── currency → which legs of pair we annotate with macro context ─────────────
PAIR_BASE_QUOTE = {
    'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'),
    'AUDUSD': ('AUD', 'USD'), 'NZDUSD': ('NZD', 'USD'),
    'USDJPY': ('USD', 'JPY'), 'USDCAD': ('USD', 'CAD'), 'USDCHF': ('USD', 'CHF'),
    'EURJPY': ('EUR', 'JPY'), 'GBPJPY': ('GBP', 'JPY'),
    'EURGBP': ('EUR', 'GBP'),
    'XAUUSD': ('XAU', 'USD'),
}


def _cache_fresh(path: Path, ttl_h: float = MACRO_CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h < ttl_h


# =============================================================================
# 1. ForexFactory weekly XML calendar
# =============================================================================

FF_XML_URL = 'https://nfs.faireconomy.media/ff_calendar_thisweek.xml'


def fetch_forexfactory_week() -> pd.DataFrame:
    """Returns DataFrame[date, time_utc, currency, impact, title].
    impact ∈ {'high','medium','low'}.
    """
    cache = CACHE_DIR / 'ff_thisweek.parquet'
    if _cache_fresh(cache, ttl_h=6):
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    try:
        r = _SESSION.get(FF_XML_URL, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        rows = []
        for ev in root.findall('.//event'):
            title    = (ev.findtext('title')    or '').strip()
            country  = (ev.findtext('country')  or '').strip().upper()
            date     = (ev.findtext('date')     or '').strip()
            t_str    = (ev.findtext('time')     or '').strip()
            impact   = (ev.findtext('impact')   or '').strip().lower()
            if not date or not t_str or impact not in ('high', 'medium', 'low'):
                continue
            try:
                dt = datetime.strptime(f"{date} {t_str}", '%m-%d-%Y %I:%M%p')
            except Exception:
                continue
            rows.append({
                'date':     dt.date().isoformat(),
                'time_utc': dt.strftime('%H:%M'),
                'hour':     dt.hour,
                'minute':   dt.minute,
                'currency': country,
                'impact':   impact,
                'title':    title,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(cache, index=False)
        return df
    except Exception:
        return pd.DataFrame()


def high_impact_times_for_pair(pair: str, date_iso: str) -> list:
    """Return [(h, m, impact), ...] for the calendar day filtered to legs of `pair`."""
    df = fetch_forexfactory_week()
    if df.empty:
        return []
    base, quote = PAIR_BASE_QUOTE.get(pair, ('', ''))
    legs = {base, quote, 'ALL'}
    sub = df[(df['date'] == date_iso) & (df['currency'].isin(legs))]
    return [(int(r.hour), int(r.minute), r.impact) for r in sub.itertuples()]


# =============================================================================
# 2. CFTC Commitments of Traders (free, weekly)
# =============================================================================

# CFTC publishes the Legacy COT reports as text + Excel. The simplest free,
# stable-schema endpoint is the CME's "Specs & Hedgers" CSV mirror. We
# fall back gracefully when offline.
COT_FUTURES_URL = (
    'https://www.cftc.gov/dea/newcot/FinComWk.txt'
)


def fetch_cot_weekly() -> pd.DataFrame:
    """Returns DataFrame[market, date, large_spec_long, large_spec_short, net].
    Empty on failure. Cached weekly.
    """
    cache = CACHE_DIR / 'cot_weekly.parquet'
    if _cache_fresh(cache, ttl_h=24 * 6):  # weekly release
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    try:
        r = _SESSION.get(COT_FUTURES_URL, timeout=15)
        r.raise_for_status()
        # FinComWk is comma-separated. We only need market name + dealer/asset
        # manager/leveraged-funds long/short. Parse defensively.
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        keep_cols = [c for c in df.columns if c.startswith('Market_and_Exchange') or
                     c.startswith('Report_Date') or 'Lev_Money_Positions' in c]
        if not keep_cols:
            return pd.DataFrame()
        out = df[keep_cols].copy()
        out.columns = [c.lower() for c in out.columns]
        out.to_parquet(cache, index=False)
        return out
    except Exception:
        return pd.DataFrame()


def cot_net_position_pct(pair: str) -> tuple[float, int]:
    """Return (net_pct_in_52w_range, extreme_flag) for `pair`. extreme_flag is
    +1 if net long > 90th pct, -1 if < 10th, else 0. (0.5, 0) on missing data.
    """
    cot = fetch_cot_weekly()
    if cot.empty:
        return 0.5, 0
    # Best-effort market-name match; CFTC names are ugly. Skip if no obvious
    # match — feature is opt-in so missing values shouldn't kill the run.
    base, quote = PAIR_BASE_QUOTE.get(pair, ('', ''))
    name_hint = base if base in ('EUR', 'GBP', 'AUD', 'NZD', 'JPY', 'CAD', 'CHF') else quote
    try:
        col_market = next(c for c in cot.columns if 'market_and_exchange' in c)
        col_long   = next(c for c in cot.columns if 'lev_money' in c and 'long_all' in c)
        col_short  = next(c for c in cot.columns if 'lev_money' in c and 'short_all' in c)
    except StopIteration:
        return 0.5, 0
    sub = cot[cot[col_market].str.contains(name_hint, case=False, na=False)]
    if sub.empty:
        return 0.5, 0
    net = sub[col_long].astype(float) - sub[col_short].astype(float)
    if len(net) < 4:
        return 0.5, 0
    pct = (net.rank(pct=True).iloc[-1])
    if pct >= 0.90:
        return float(pct), 1
    if pct <= 0.10:
        return float(pct), -1
    return float(pct), 0


# =============================================================================
# 3. Yahoo Finance daily macro context
# =============================================================================

MACRO_TICKERS = {
    'DXY':    'DX-Y.NYB',
    'US10Y':  '^TNX',
    'SPX':    '^GSPC',
    'GOLD':   'GC=F',
    'VIX':    '^VIX',
}


def fetch_macro_daily(start: str = '2020-01-01') -> pd.DataFrame:
    """Returns wide DataFrame indexed by date with one column per macro series.
    Uses yfinance if available; empty DF if offline or yfinance missing.
    """
    cache = CACHE_DIR / 'macro_daily.parquet'
    if _cache_fresh(cache, ttl_h=12):
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()
    try:
        out = pd.DataFrame()
        for label, sym in MACRO_TICKERS.items():
            hist = yf.Ticker(sym).history(start=start, auto_adjust=False)
            if hist.empty:
                continue
            s = hist['Close'].rename(label)
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            out = out.join(s, how='outer') if not out.empty else s.to_frame()
        if not out.empty:
            out = out.ffill()
            # Daily change for each series — what an entry function actually wants
            for c in list(out.columns):
                out[f'{c}_chg'] = out[c].pct_change()
            out.index.name = 'date'
            out.to_parquet(cache)
        return out
    except Exception:
        return pd.DataFrame()


# =============================================================================
# 4. OANDA retail positioning (public, no auth on labs endpoint)
# =============================================================================

OANDA_POS_URL = 'https://www1.oanda.com/lfr/forex_lab/forex_position'


def fetch_oanda_positioning() -> pd.DataFrame:
    """Returns DataFrame[pair, retail_long_pct]. Empty on failure."""
    cache = CACHE_DIR / 'oanda_pos.parquet'
    if _cache_fresh(cache, ttl_h=2):
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    try:
        r = _SESSION.get(OANDA_POS_URL, timeout=15)
        r.raise_for_status()
        # Endpoint returns HTML with embedded JSON; tolerate schema drift.
        import re
        m = re.search(r'\{.*"positions".*\}', r.text, flags=re.DOTALL)
        if not m:
            return pd.DataFrame()
        data = json.loads(m.group(0))
        rows = []
        for entry in data.get('positions', []):
            pair = (entry.get('instrument') or '').replace('_', '').upper()
            lp   = entry.get('long_position_ratio')
            if pair and lp is not None:
                rows.append({'pair': pair, 'retail_long_pct': float(lp)})
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(cache, index=False)
        return df
    except Exception:
        return pd.DataFrame()


def retail_long_pct(pair: str) -> float:
    df = fetch_oanda_positioning()
    if df.empty:
        return 0.5
    sub = df[df['pair'] == pair]
    if sub.empty:
        return 0.5
    return float(sub['retail_long_pct'].iloc[0])


# =============================================================================
# 5. Bulk merge into the per-pair preparation DataFrame
# =============================================================================

def merge_macro_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Add Phase 9 columns to df. Idempotent — leaves df unchanged on failure.

    Adds (when sources are reachable):
      - dxy_chg, us10y_chg, spx_chg, gold_chg, vix_chg, vix_level
      - cot_net_position_pct, cot_extreme_flag
      - retail_long_pct
    """
    if df is None or df.empty:
        return df

    # 3) Macro daily — joined on date
    macro = fetch_macro_daily()
    if not macro.empty and 'date' in df.columns:
        macro_small = macro[[c for c in macro.columns if c.endswith('_chg')] +
                            (['VIX'] if 'VIX' in macro.columns else [])].copy()
        macro_small.rename(columns={
            'DXY_chg': 'dxy_chg', 'US10Y_chg': 'us10y_chg',
            'SPX_chg': 'spx_chg', 'GOLD_chg': 'gold_chg',
            'VIX_chg': 'vix_chg', 'VIX': 'vix_level',
        }, inplace=True)
        macro_small.index = pd.to_datetime(macro_small.index).date
        df = df.merge(macro_small, left_on='date', right_index=True, how='left')

    # 2) COT — single value broadcast across the pair's frame (weekly cadence)
    try:
        net_pct, extreme = cot_net_position_pct(pair)
        df['cot_net_position_pct'] = net_pct
        df['cot_extreme_flag']     = extreme
    except Exception:
        pass

    # 4) OANDA retail — single value broadcast
    try:
        df['retail_long_pct'] = retail_long_pct(pair)
    except Exception:
        pass

    return df


# =============================================================================
# Convenience: callers that only want the calendar block
# =============================================================================

def expand_calendar_into_session_cfg(session_cfg: dict, pair: str,
                                     date_iso: str) -> dict:
    """Return a copy of session_cfg with `high_impact_times` augmented from
    ForexFactory for the given pair/date. Existing entries are preserved.
    Used by edge_engine.prepare_df when called with a known date.
    """
    if not isinstance(session_cfg, dict):
        return session_cfg
    cfg = dict(session_cfg)
    existing = list(cfg.get('high_impact_times') or [])
    extra = high_impact_times_for_pair(pair, date_iso)
    if extra:
        cfg['high_impact_times'] = existing + extra
    return cfg
