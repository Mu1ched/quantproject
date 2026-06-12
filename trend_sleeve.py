# -*- coding: utf-8 -*-
"""
Trend Sleeve — standalone MA100 long/flat companion to the ORB bot (Edge 3)
===========================================================================
A *separate* process from MT5Live.2.py, with its OWN magic number, so it can
NEVER interfere with the ORB/Dow engine. It holds positions for weeks (the ORB
bot flattens nightly — this one does not), which is why it lives apart.

Strategy (bull-beta diversifier, validated as a sleeve, ~0.3 trades/wk):
  Once per day, after the US cash open, for each of US500 / NAS100 / XAUUSD:
    prev daily close > MA100(daily)  -> hold ONE long position
    else                             -> flat
  Rebalance only on a cross. Sized small: a -2.5% adverse day ~= 0.2% of equity
  per instrument (TREND_NOTIONAL_FRAC = 0.08 of equity notional each).

Risk: hard kill if equity draws down to the firm's 6%-from-initial floor, or
2% on the day (closes all sleeve positions, waits for the next day).

Run:   python trend_sleeve.py
Stop:  Ctrl+C  (positions are NOT auto-closed on Ctrl+C — they are meant to be
       held; close manually in MT5 if you want flat)

NOTE: this attaches to the SAME running MT5 terminal as MT5Live.2.py. Two
processes on one terminal normally coexist fine; if you see contention, run the
sleeve against a second terminal/login or fold it into the main process.
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta

import pytz
import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MT5_LOGIN    = int(os.environ.get("MT5_LOGIN", 0))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER", "")
if not MT5_LOGIN or not MT5_PASSWORD or not MT5_SERVER:
    raise ValueError("Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in your .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  trend  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("trend_sleeve.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
TREND_MAGIC          = 770802          # distinct from ORB (770801) / DOW (770803)
MA_PERIOD            = 100
TREND_PAIRS          = ["US500", "NAS100", "XAUUSD"]
TREND_NOTIONAL_FRAC  = 0.08            # notional per instrument = 8% of equity
INITIAL_DEPOSIT      = 10000.0         # trailing-drawdown lock level (funded size)
MAX_DD_FRAC          = 0.06            # 6% TRAILING from HWM (Blue Guardian Instant)
DD_SAFETY_MARGIN     = 0.005           # flatten 0.5% BEFORE the firm's line
DAILY_DD_FRAC        = 0.02            # 2% on the day -> flatten + wait
# Rebalance at 15:35 London (10:35 ET) — clear of the 10:00 ET high-impact
# releases (ISM etc.); Blue Guardian voids profit on trades within ±2 min of
# high-impact news, and this sleeve has no news-calendar feed.
CHECK_HOUR_LONDON    = 15
CHECK_MIN_LONDON     = 35
POLL_SECS            = 120
LONDON_TZ            = pytz.timezone("Europe/London")
NY_TZ                = pytz.timezone("America/New_York")

# Shared persistent prop state (same file as MT5Live.2.py — read-merge-write,
# races self-heal since HWM is re-maxed and the first day-baseline writer wins).
PROP_STATE_FILE      = "prop_state.json"


def _load_prop_state() -> dict:
    try:
        with open(PROP_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_prop_state(updates: dict):
    try:
        state = _load_prop_state()
        if "hwm" in updates and "hwm" in state:
            updates = dict(updates)
            updates["hwm"] = max(float(updates["hwm"]), float(state["hwm"]))
        state.update(updates)
        tmp = PROP_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, PROP_STATE_FILE)
    except Exception as e:
        log.warning(f"prop_state save failed: {e}")


def prop_day_today() -> str:
    """Blue Guardian's day rolls at 17:00 New York."""
    now_ny = datetime.now(pytz.utc).astimezone(NY_TZ)
    return (now_ny + timedelta(hours=7)).date().isoformat()


def trailing_floor(hwm: float) -> float:
    """BG Instant: floor trails 6% (of initial) behind the balance/equity HWM,
    locks at breakeven once the account is +6% over initial."""
    return min(hwm - MAX_DD_FRAC * INITIAL_DEPOSIT, INITIAL_DEPOSIT)

# Broker symbol resolution (indices use different base names per broker)
_SUFFIXES = ["", ".r", ".raw", "m", ".m", ".pro", "-pro", "_pro", ".s", ".sml"]
_ALIASES = {
    "US500":  ["US500", "SPX500", "USA500", "SP500", "US500.cash"],
    "NAS100": ["NAS100", "USTEC", "US100", "NDX100", "NAS100.cash"],
    "XAUUSD": ["XAUUSD"],
}
_SYMBOL_MAP = {}


def connect():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    log.info("MT5 connected")


def resolve_symbols():
    for base in TREND_PAIRS:
        found = None
        for alias in _ALIASES.get(base, [base]):
            for suf in _SUFFIXES:
                cand = alias + suf
                info = mt5.symbol_info(cand)
                if info is not None:
                    if not info.visible:
                        mt5.symbol_select(cand, True)
                    found = cand
                    break
            if found:
                break
        if not found:
            raise RuntimeError(f"Could not resolve broker symbol for {base} "
                               f"(tried {_ALIASES.get(base, [base])}). Add the right name to _ALIASES.")
        _SYMBOL_MAP[base] = found
        log.info(f"  resolved {base} -> {found}")


def _sym(base):
    return _SYMBOL_MAP.get(base, base)


def london_now():
    return datetime.now(pytz.utc).astimezone(LONDON_TZ)


def daily_signal(base):
    """Return ('long' | 'flat', prev_close, ma) from completed daily bars."""
    rates = mt5.copy_rates_from_pos(_sym(base), mt5.TIMEFRAME_D1, 1, MA_PERIOD + 5)
    if rates is None or len(rates) < MA_PERIOD + 1:
        return None, None, None
    closes = [float(r["close"]) for r in rates]
    prev_close = closes[-1]                          # last completed daily close
    ma = sum(closes[-MA_PERIOD:]) / MA_PERIOD
    return ("long" if prev_close > ma else "flat"), prev_close, ma


def current_lots(base):
    """Signed lots of our sleeve position on this instrument (0.0 if flat)."""
    positions = mt5.positions_get(symbol=_sym(base)) or []
    lots = 0.0
    for p in positions:
        if getattr(p, "magic", 0) != TREND_MAGIC:
            continue
        lots += p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume
    return lots


def _value_per_point(base):
    info = mt5.symbol_info(_sym(base))
    tv = float(getattr(info, "trade_tick_value", 0) or 0) if info else 0.0
    ts = float(getattr(info, "trade_tick_size",  0) or 0) if info else 0.0
    return (tv / ts) if (tv > 0 and ts > 0) else 0.0


def _round_lots(base, lots):
    info = mt5.symbol_info(_sym(base))
    step = float(getattr(info, "volume_step", 0.01) or 0.01) if info else 0.01
    vmin = float(getattr(info, "volume_min",  0.01) or 0.01) if info else 0.01
    n = max(1, int(lots / step))
    return max(vmin, round(n * step, 2))


def target_lots(base, equity):
    """Lots so notional ~= TREND_NOTIONAL_FRAC * equity.

    Min-lot guard: if the ideal size is far below the broker's minimum lot,
    taking the position at volume_min would be a multiple of the intended risk
    (e.g. gold on a $10k account: ideal ~0.002 lots vs 0.01 min = ~6x oversize,
    where one -2.5% day costs ~1.2% of the account). Skip the instrument as
    unsizeable — it re-enables itself automatically as equity grows."""
    info = mt5.symbol_info(_sym(base))
    tick = mt5.symbol_info_tick(_sym(base))
    vpp = _value_per_point(base)
    if info is None or tick is None or vpp <= 0:
        return 0.0
    price = (tick.bid + tick.ask) / 2
    if price <= 0:
        return 0.0
    notional_per_lot = price * vpp                   # $ value of 1.0 lot
    if notional_per_lot <= 0:
        return 0.0
    ideal = (TREND_NOTIONAL_FRAC * equity) / notional_per_lot
    vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
    if ideal < vmin / 2:
        log.info(f"[{base}] unsizeable at this equity (ideal {ideal:.4f} lots < "
                 f"min {vmin}/2) — skipped until the account grows")
        return 0.0
    return _round_lots(base, ideal)


def _send(base, order_type, lots, comment):
    sym = _sym(base)
    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        log.error(f"[{base}] no tick — skip {comment}")
        return False
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    info = mt5.symbol_info(sym)
    supported = info.filling_mode if info else 7
    filling = (mt5.ORDER_FILLING_IOC if supported & 2 else
               mt5.ORDER_FILLING_RETURN if supported & 4 else mt5.ORDER_FILLING_FOK)
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       sym,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "deviation":    30,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
        "magic":        TREND_MAGIC,
        "comment":      comment,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"[{base}] {comment} failed: retcode={getattr(res,'retcode',None)} "
                  f"comment={getattr(res,'comment','')}")
        return False
    log.info(f"[{base}] {comment} {lots} lots @ {price}")
    return True


def go_long(base, equity):
    lots = target_lots(base, equity)
    if lots <= 0:
        log.warning(f"[{base}] target lots <= 0 — cannot size, skipping")
        return
    _send(base, mt5.ORDER_TYPE_BUY, lots, "trend_long")


def go_flat(base):
    positions = [p for p in (mt5.positions_get(symbol=_sym(base)) or [])
                 if getattr(p, "magic", 0) == TREND_MAGIC]
    for p in positions:
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(_sym(base))
        price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
        info = mt5.symbol_info(_sym(base))
        supported = info.filling_mode if info else 7
        filling = (mt5.ORDER_FILLING_FOK if supported & 1 else
                   mt5.ORDER_FILLING_IOC if supported & 2 else mt5.ORDER_FILLING_RETURN)
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": _sym(base), "volume": p.volume,
               "type": close_type, "position": p.ticket, "price": price, "deviation": 30,
               "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling,
               "magic": TREND_MAGIC, "comment": "trend_flat"}
        res = mt5.order_send(req)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        log.info(f"[{base}] flatten ticket {p.ticket}: {'ok' if ok else 'FAILED'}")


def account_equity():
    info = mt5.account_info()
    return (float(info.balance), float(info.equity)) if info else (None, None)


def rebalance():
    bal, eq = account_equity()
    if eq is None:
        log.warning("account_info() failed — skip rebalance")
        return
    for base in TREND_PAIRS:
        sig, prev_close, ma = daily_signal(base)
        if sig is None:
            log.warning(f"[{base}] insufficient daily history — skip")
            continue
        held = current_lots(base)
        log.info(f"[{base}] signal={sig} prev_close={prev_close:.2f} ma{MA_PERIOD}={ma:.2f} held={held}")
        if sig == "long" and held <= 0:
            go_long(base, eq)
        elif sig == "flat" and held > 0:
            go_flat(base)


def risk_kill_check():
    """Return True if a kill was triggered (positions flattened).

    Blue Guardian's 6% max drawdown TRAILS the balance/equity high-water mark
    (locking at breakeven once +6%) — so the floor rises as the account grows.
    HWM is persisted in prop_state.json (shared with MT5Live.2.py) so a restart
    can never lower the floor. We flatten DD_SAFETY_MARGIN before the line."""
    bal, eq = account_equity()
    if eq is None:
        return False
    state = _load_prop_state()
    hwm = max(float(state.get("hwm", 0.0)), INITIAL_DEPOSIT, bal or 0.0, eq)
    if hwm > float(state.get("hwm", 0.0)):
        _save_prop_state({"hwm": hwm})
    floor = trailing_floor(hwm) + DD_SAFETY_MARGIN * INITIAL_DEPOSIT
    if eq <= floor:
        log.warning(f"TRAILING MAX-DD kill: equity {eq:.2f} <= floor {floor:.2f} "
                    f"(HWM {hwm:.2f}) — flattening sleeve")
        for base in TREND_PAIRS:
            go_flat(base)
        return True
    return False


def get_day_baseline(bal, eq):
    """Daily-loss baseline = max(balance, equity) at the 5 PM NY day change.
    Adopt today's persisted baseline if the other process (or a previous run)
    already snapshotted it — first writer wins; restarts can't reset it."""
    today_id = prop_day_today()
    state = _load_prop_state()
    if state.get("prop_day") == today_id and state.get("day_baseline"):
        return float(state["day_baseline"])
    baseline = max(bal or 0.0, eq or 0.0)
    _save_prop_state({"prop_day": today_id, "day_baseline": baseline})
    return baseline


def main():
    # FTMO standard (non-Swing) accounts PROHIBIT weekend holding — this sleeve
    # holds positions for weeks and is therefore INCOMPATIBLE with FTMO 1-Step.
    # Only run it on firms that allow weekend holds (e.g. Blue Guardian).
    log.warning("=" * 60)
    log.warning("  TREND SLEEVE holds over weekends — DO NOT run on FTMO")
    log.warning("  standard/1-Step accounts (weekend-hold rule = breach).")
    log.warning("=" * 60)
    connect()
    resolve_symbols()
    bal, eq = account_equity()
    log.info(f"Trend sleeve started. Balance={bal} Equity={eq} | pairs={TREND_PAIRS} "
             f"magic={TREND_MAGIC}")
    last_check_day = None
    daily_blocked  = False
    prop_day       = None
    day_baseline   = None
    try:
        while True:
            now = london_now()
            today = now.date()

            # Prop day rolls at 5 PM New York (Blue Guardian's day change) —
            # baseline = max(balance, equity), persisted/shared via prop_state.
            if prop_day != prop_day_today():
                prop_day = prop_day_today()
                bal, eq = account_equity()
                day_baseline = get_day_baseline(bal, eq)
                daily_blocked = False
                log.info(f"-- Prop day {prop_day}  daily baseline {day_baseline}")

            # hard trailing max-DD kill (checked every poll)
            if risk_kill_check():
                time.sleep(POLL_SECS)
                continue

            # daily 2% stop — flatten and wait for the next prop day
            _, eq = account_equity()
            if (not daily_blocked and day_baseline and eq is not None
                    and (day_baseline - eq) >= day_baseline * DAILY_DD_FRAC):
                daily_blocked = True
                log.warning(f"DAILY 2% stop: flattening sleeve for the day (eq {eq:.2f})")
                for base in TREND_PAIRS:
                    go_flat(base)

            # weekday once-a-day decision — 15:35 London, clear of 10:00 ET news
            after_check_time = (now.hour > CHECK_HOUR_LONDON or
                                (now.hour == CHECK_HOUR_LONDON and
                                 now.minute >= CHECK_MIN_LONDON))
            if (not daily_blocked and now.weekday() < 5
                    and after_check_time and last_check_day != today):
                last_check_day = today
                log.info("Daily rebalance check")
                rebalance()

            time.sleep(POLL_SECS)
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C). Sleeve positions LEFT OPEN (held by design).")
    finally:
        mt5.shutdown()
        log.info("MT5 connection closed.")


if __name__ == "__main__":
    main()
