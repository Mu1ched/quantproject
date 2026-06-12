# -*- coding: utf-8 -*-
"""
London + NY ORB — Live Trading Engine (MT5 Edition)
Automates two sessions: London ORB (GBPUSD, GBPJPY) and NY ORB (EURUSD).

Architecture:
  Stream thread  — polls symbol_info_tick() every 100 ms; applies BE and profit
                   lock the instant price crosses the level (no polling delay)
  Fill thread    — polls orders/positions every 1 s for instant fill detection
  Main thread    — handles time-based events: daily reset, 08:30 order placement,
                   order fill detection (30 s poll fallback), 13:00 session exit,
                   balance check

Requirements:
  MetaTrader 5 terminal must be running and logged in (Windows only).
  .env file (same folder) must contain:
    MT5_LOGIN=your_account_number
    MT5_PASSWORD=your_password
    MT5_SERVER=your_broker_server   e.g. ICMarkets-Live01

Run:    python live_trader.py
Stop:   Ctrl+C  (open positions are NOT auto-closed — close manually in MT5)
"""


import os
import re
import time
import logging
import csv
import threading
import requests
import pytz
import MetaTrader5 as mt5
import pandas as pd

from collections import deque
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# =============================================================================
# SETUP
# =============================================================================
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MT5_LOGIN         = int(os.environ.get("MT5_LOGIN", 0))
MT5_PASSWORD      = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER        = os.environ.get("MT5_SERVER", "")
# Optional: full path to a SPECIFIC terminal64.exe. Required when running two
# bot instances / two accounts on one machine — each Python process must attach
# to its own MT5 terminal installation (the MT5 API is one-terminal-per-process).
MT5_TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH", "")
if not MT5_LOGIN or not MT5_PASSWORD or not MT5_SERVER:
    raise ValueError("Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in your .env file")

from logging.handlers import RotatingFileHandler

# review#20 — install secret-redaction filter BEFORE basicConfig so the
# stream/file handlers it creates inherit the filter from the start. Was
# previously called after basicConfig, leaving an early-init window where
# MT5 connection failures could log the password verbatim.
try:
    from agent.log_filter import install_global_redaction
    install_global_redaction()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=open(1, 'w', encoding='utf-8', closefd=False)),
        # 50 MB per file, keep 10 backups (~500 MB cap). Stream thread writes
        # ~10 messages/sec under fault conditions; without rotation the log
        # file balloons to multi-GB and slows down Windows file I/O.
        RotatingFileHandler("live_trader.log", maxBytes=50_000_000,
                            backupCount=10, encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)

def connect_mt5():
    """Initialise and log in to the MT5 terminal. Raises on failure.
    If MT5_TERMINAL_PATH is set, attaches to THAT terminal installation —
    essential when two bot instances run two accounts on one machine."""
    if MT5_TERMINAL_PATH:
        ok = mt5.initialize(MT5_TERMINAL_PATH, login=MT5_LOGIN,
                            password=MT5_PASSWORD, server=MT5_SERVER)
    else:
        ok = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not ok:
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")


# ── Broker symbol resolution ─────────────────────────────────────────────────
# Some brokers append suffixes like ".r", "m", or ".pro" to symbol names.
# We probe at startup and remap internal base names to the broker's actuals.
_SYMBOL_MAP: dict         = {}
_REVERSE_SYMBOL_MAP: dict = {}   # broker_sym -> base; populated by resolve_broker_symbols
_SUFFIX_CANDIDATES = ["", ".r", ".raw", "m", ".m", ".pro", "-pro", "_pro", ".s", ".sml"]

# Index brokers use different *base* names (not just suffixes). For index
# instruments we probe these alias bases × the suffix candidates above. If your
# broker uses a name not listed here, add it (or set the exact name as the first
# alias). The clear RuntimeError below tells you which one failed to resolve.
_INDEX_ALIASES = {
    "US500":  ["US500", "SPX500", "USA500", "SP500", "US500.cash", "US.500", "SPX500m"],
    "NAS100": ["NAS100", "USTEC", "US100", "NDX100", "USATEC", "NAS100.cash", "USTECm"],
    "US30":   ["US30", "DJ30", "WS30", "US30.cash", "USA30", "DOW30", "US30m"],
}


def _sym(base: str) -> str:
    """Translate an internal base symbol (e.g. 'GBPUSD') to the broker's actual symbol."""
    return _SYMBOL_MAP.get(base, base)


def _base(broker_sym: str) -> str:
    """Translate a broker symbol (e.g. 'GBPUSD.r') back to the internal base name."""
    return _REVERSE_SYMBOL_MAP.get(broker_sym, broker_sym)


def resolve_broker_symbols():
    """
    Probe MT5 for the actual symbol name for each base pair.
    Populates _SYMBOL_MAP and ensures each symbol is selected in market watch.
    Raises with a clear message if any base symbol cannot be resolved.
    """
    missing = []
    for base in PAIRS:
        resolved = None
        # For indices, probe broker-specific alias bases; for FX/gold just the base.
        alias_bases = _INDEX_ALIASES.get(base, [base])
        for alias in alias_bases:
            for suffix in _SUFFIX_CANDIDATES:
                candidate = alias + suffix
                info = mt5.symbol_info(candidate)
                if info is not None:
                    resolved = candidate
                    if not info.visible:
                        mt5.symbol_select(candidate, True)
                    break
            if resolved is not None:
                break
        if resolved is None:
            missing.append(base)
            continue
        _SYMBOL_MAP[base]            = resolved
        _REVERSE_SYMBOL_MAP[resolved] = base
        if resolved != base:
            log.info(f"  Symbol resolved: {base} -> {resolved}")
        else:
            log.info(f"  Symbol resolved: {base}")
    if missing:
        raise RuntimeError(
            f"Could not resolve broker symbols for: {missing}. "
            f"Check the symbols list in your MT5 terminal — the broker may use "
            f"a suffix that isn't in _SUFFIX_CANDIDATES."
        )

def api_retry(fn, *args, retries=3, base_delay=1.0, **kwargs):
    """
    Retry a callable up to `retries` times with linear back-off.
    Used for critical calls (SL/TP modification, trade close) where a transient
    error should not silently drop the operation.
    """
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            delay = base_delay * (attempt + 1)
            log.warning(f"Call failed ({e}) — retrying in {delay:.0f}s "
                        f"(attempt {attempt + 1}/{retries})")
            time.sleep(delay)

LONDON_TZ = pytz.timezone("Europe/London")

# =============================================================================
# TELEGRAM ALERTS — credentials must be in .env (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
# TELEGRAM_CHANNEL_ID). If missing, alerts are silently disabled (logged once).
# =============================================================================
TELEGRAM_TOKEN      = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID", "")

DISCLAIMER = "\n\n⚠️ Educational purposes only. Not financial advice. Trading forex carries significant risk of loss."

_telegram_disabled_warned = False

def send_telegram(msg: str, channel: bool = False):
    """
    Send a Telegram message with retries.
    channel=False  → private alert to your personal chat
    channel=True   → public post to your signal channel
    Silently no-ops if the relevant env var is missing (warning is logged once).
    """
    global _telegram_disabled_warned
    chat = TELEGRAM_CHANNEL_ID if channel else TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not chat:
        if not _telegram_disabled_warned:
            log.warning("Telegram disabled — set TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, "
                        "TELEGRAM_CHANNEL_ID in .env to enable alerts")
            _telegram_disabled_warned = True
        return

    text = msg + (DISCLAIMER if channel else "")
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat, "text": text, "parse_mode": "HTML"}

    # Manual retry with back-off — a transient blip shouldn't lose an alert.
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(url, data=payload, timeout=5)
            if resp.ok:
                return
            last_err = f"status={resp.status_code} body={resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.0 + attempt)
    log.warning(f"Telegram send failed after 3 retries: {last_err}")

def signal(msg: str):
    """Post an educational signal to the public channel."""
    send_telegram(msg, channel=True)

# =============================================================================
# AUTO-RECONNECT WATCHDOG
# =============================================================================

_mt5_was_connected = True   # assume connected at startup
# Set when a reconnect (shutdown+initialize) is in progress. Worker threads
# (stream, transaction_stream) check this and sleep rather than call MT5 RPCs
# during the window — using the API mid-shutdown can crash the terminal client.
_mt5_reconnecting  = threading.Event()

def check_mt5_connection():
    """
    Check MT5 terminal connection. Attempt reconnect if lost.
    Returns True if connected, False if still down after retry.
    """
    global _mt5_was_connected
    info = mt5.terminal_info()
    connected = info is not None and info.connected

    if not connected:
        if _mt5_was_connected:
            # Just lost connection — alert immediately
            _mt5_was_connected = False
            log.error("MT5 connection lost — attempting reconnect")
            send_telegram("⚠️ MT5 connection lost — attempting to reconnect")
            signal("⚠️ <b>System Notice</b>\nTemporary connection issue detected. Reconnecting now.")

        # Attempt reconnect — block worker threads from issuing MT5 RPCs
        # while we tear down and re-init the terminal client.
        _mt5_reconnecting.set()
        try:
            time.sleep(0.3)  # let worker threads observe the flag
            mt5.shutdown()
            time.sleep(2)
            connect_mt5()
            info = mt5.terminal_info()
            if info and info.connected:
                _mt5_was_connected = True
                log.info("MT5 reconnected successfully")
                send_telegram("✅ MT5 reconnected successfully")
                signal("✅ <b>System Notice</b>\nConnection restored. All systems operational.")
                return True
        except Exception as e:
            log.error(f"MT5 reconnect failed: {e}")
        finally:
            _mt5_reconnecting.clear()
        return False

    if not _mt5_was_connected:
        # Was down, now back
        _mt5_was_connected = True
        log.info("MT5 connection restored")
        send_telegram("✅ MT5 reconnected")

    return True

# =============================================================================
# LIVE VS BACKTEST TRACKER
# =============================================================================

# Backtest baseline — validated ORB (gold + US500/NAS100/US30, full-history backtest 2024-26).
# Win rate ~44-56% across instruments (breakout: low WR, TP 2.5R > SL 1R); avg R-multiple net
# of realistic slippage ~0.4R. Real Sharpe ~1-2 (sim fills flatter it ~2x). See edge validation.
BACKTEST_WIN_RATE   = 0.46   # blended validated ORB win rate
BACKTEST_AVG_RR     = 2.5    # take-profit R-multiple (TP = 2.5 x range)
BACKTEST_AVG_TRADE  = 8.0    # avg $ P&L per trade, indicative (scales with risk/account)

def post_live_vs_backtest():
    """
    Post a weekly comparison of live results vs backtest expectations.
    Called every Friday at 21:00 alongside the weekly stats.
    """
    try:
        import datetime as dt_mod
        today      = london_now().date()
        week_start = today - dt_mod.timedelta(days=today.weekday())

        # Read all live trades from log
        all_trades  = []
        week_trades = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                for row in csv.DictReader(f):
                    if not row.get("pnl"):
                        continue
                    try:
                        pnl      = float(row["pnl"])
                        row_date = dt_mod.date.fromisoformat(row["date"])
                        all_trades.append(pnl)
                        if row_date >= week_start:
                            week_trades.append(pnl)
                    except Exception:
                        continue

        if not all_trades:
            return

        # Overall live stats
        total_n   = len(all_trades)
        total_w   = sum(1 for t in all_trades if t > 0)
        live_wr   = total_w / total_n
        live_avg  = sum(all_trades) / total_n

        # Week stats
        wn  = len(week_trades)
        ww  = sum(1 for t in week_trades if t > 0)
        wwr = (ww / wn * 100) if wn > 0 else 0

        # Vs backtest
        wr_diff  = (live_wr - BACKTEST_WIN_RATE) * 100
        avg_diff = live_avg - BACKTEST_AVG_TRADE
        wr_icon  = "✅" if wr_diff >= -5 else "⚠️"
        avg_icon = "✅" if avg_diff >= -5 else "⚠️"

        signal(
            f"📈 <b>Live vs Backtest Tracker</b>\n\n"
            f"<b>This Week ({wn} trades)</b>\n"
            f"W: {ww}  L: {wn-ww}  WR: {wwr:.0f}%\n\n"
            f"<b>All-Time Live ({total_n} trades)</b>\n"
            f"Win Rate:   {live_wr:.0%}  {wr_icon}  (backtest: {BACKTEST_WIN_RATE:.0%}  diff: {wr_diff:+.1f}%)\n"
            f"Avg Trade:  {live_avg:+.2f}  {avg_icon}  (backtest: {BACKTEST_AVG_TRADE:+.2f}  diff: {avg_diff:+.2f})\n\n"
            f"The closer live results track backtest expectations,\n"
            f"the more confident we can be in the strategy edge."
        )
    except Exception as e:
        log.warning(f"Live vs backtest post failed: {e}")

# =============================================================================
# EDUCATIONAL CONTENT — rotating tips and prop firm advice
# =============================================================================

RISK_TIPS = [
    ("Risk Management", "On prop firm challenges, consistency beats big wins. A 0.4% risk per trade means you need 12 consecutive losses to hit a 5% daily limit — highly unlikely with proper filters in place.\n\nProtect your account first. Profits follow."),
    ("Position Sizing", "Never risk more than you're comfortable losing on a single trade. Our system risks 0.4% per trade — this means on a $5,000 account, maximum loss per trade is $20.\n\nSmall, consistent losses are survivable. One oversized loss can end a challenge."),
    ("Stop Losses", "Every trade has a defined stop loss placed below the range low (long) or above the range high (short).\n\nA stop loss is not a sign of weakness — it is the single most important tool for surviving long enough to be profitable."),
    ("Daily Loss Limits", "Prop firms typically impose a 4–5% daily loss limit. We track this in real time and halt trading automatically when it's approached.\n\nIf you ever feel the urge to 'revenge trade' after losses — step away. The market will be there tomorrow."),
    ("Drawdown Management", "Maximum drawdown limits (typically 8–10% on prop challenges) are the hardest rules to recover from if broken.\n\nThis system monitors drawdown every 30 seconds and closes all positions if the limit is approached. Automation removes emotion from the equation."),
    ("Win Rate vs Risk:Reward", "A 40% win rate with a 2:1 reward-to-risk ratio is profitable. You don't need to win most trades — you need your winners to be bigger than your losers.\n\nThis is why we target 2× the range on London pairs and 2.5× on NY — the maths works in your favour over time."),
    ("Correlation Risk", "Trading GBPUSD and GBPJPY simultaneously increases risk because both pairs are driven by GBP sentiment.\n\nThis system automatically halves position size on the second pair when both are active — protecting against correlated losses."),
    ("News Events", "High-impact news causes spreads to widen and price to spike unpredictably — conditions where stop orders can fill at terrible prices.\n\nThis system buffers 15 minutes around major GBP and USD news events, cancelling pending orders automatically."),
    ("Patience", "Not every session produces a valid setup. The body filter, range filter, MA filter, and news filter all exist to reduce low-quality trades.\n\nA day with no trades is not a wasted day — it is a day where capital was preserved."),
    ("Breakeven Management", "Once price reaches 1× the range in profit, the stop loss is moved to entry automatically.\n\nThis converts a risk trade into a free trade — the worst outcome becomes breakeven, not a loss. This is critical for prop firm capital preservation."),
]

PROP_TIPS = [
    ("Passing Your Evaluation", "The most common reason traders fail prop challenges is not a single bad trade — it's slow accumulation of losses from overtrading outside their rules.\n\nStick strictly to session windows. London: 08:30–11:00. New York: 15:00–21:00. Nothing outside these times."),
    ("The Evaluation Mindset", "Treat your prop challenge like a job interview, not a casino. The firm is testing whether you can follow rules under pressure.\n\nConsistent, small gains with controlled drawdown will pass every evaluation. Trying to pass in one week usually ends in failure."),
    ("Profit Targets", "Most prop firms require 8–10% profit to pass Phase 1. At 0.4% risk and 2:1 R:R, you need roughly 15–20 winning trades net to reach this target.\n\nRush it and you blow the account. Trade it properly and the target takes care of itself."),
    ("Two-Phase Challenges", "Many prop firms run two evaluation phases before funding. Phase 2 typically has the same drawdown rules but a lower profit target.\n\nDo not change your strategy between phases. If it worked in Phase 1, trust it in Phase 2."),
    ("Funded Account Rules", "Once funded, the rules become even stricter — and breaking them means losing the account permanently.\n\nThe same discipline that passed the evaluation must continue on the funded account. Consistency is the product."),
    ("Scaling Plans", "Most prop firms offer scaling — if you hit consistent monthly returns, they increase your allocation.\n\nThis is where systematic trading pays off. A repeatable strategy that returns 4–5% monthly consistently will scale faster than an inconsistent trader hitting 15% one month and -8% the next."),
]

STRATEGY_TIPS = [
    ("What is ORB?", "Opening Range Breakout (ORB) is one of the oldest and most documented institutional strategies.\n\nThe concept: price often establishes a range in the first 30 minutes of a session as institutions position themselves. When that range breaks with conviction, it frequently continues in the breakout direction."),
    ("The MA200 Filter", "We only take long trades when price is above the 200-period moving average, and short trades below it.\n\nThe MA200 represents the long-term trend bias. Trading in the direction of the trend dramatically improves the probability of the breakout following through."),
    ("The Body Filter", "A valid ORB entry requires the breakout candle body to be at least 30% of the range size.\n\nThis filters out false breakouts — wicks that poke through the range but immediately reverse. A strong body indicates genuine institutional commitment to the move."),
    ("Range Filters", "We reject ranges smaller than 3 pips (too tight — spread-sensitive) and larger than 30 pips (too volatile — stop too wide).\n\nThe ideal range sits between 8–20 pips: large enough to be meaningful, tight enough for a sensible stop loss."),
    ("Silver Bullet Setup", "The ICT Silver Bullet is a liquidity-based entry model:\n1. Price sweeps a prior high or low (taking out stop losses)\n2. A Fair Value Gap (price imbalance) forms in the reversal direction\n3. Entry at the 50% level of the FVG\n\nThis works because institutions deliberately sweep liquidity before reversing — we position in the same direction."),
    ("Session Characteristics", "London (08:00–13:00): Highest volume session. GBP pairs most active. ORB at 08:30 captures the institutional open.\n\nNew York (14:30–21:00): Second highest volume. USD pairs react to US economic data. ORB at 15:00 captures the Wall Street open overlap."),
]

_tip_index = {"risk": 0, "prop": 0, "strategy": 0}

def post_daily_tip():
    """Rotate through risk, prop, and strategy tips — one per day."""
    day = london_now().weekday()
    if day % 3 == 0:
        tips = RISK_TIPS
        key  = "risk"
        cat  = "Risk Management"
    elif day % 3 == 1:
        tips = PROP_TIPS
        key  = "prop"
        cat  = "Prop Firm Education"
    else:
        tips = STRATEGY_TIPS
        key  = "strategy"
        cat  = "Strategy Education"

    idx   = _tip_index[key] % len(tips)
    title, body = tips[idx]
    _tip_index[key] += 1

    signal(f"📚 <b>{cat} — {title}</b>\n\n{body}")

def post_monday_prop_tip():
    """Post a prop-firm specific tip every Monday morning."""
    idx   = _tip_index["prop"] % len(PROP_TIPS)
    title, body = PROP_TIPS[idx]
    _tip_index["prop"] += 1
    signal(f"💡 <b>Prop Firm Tip — Week Start</b>\n\n<b>{title}</b>\n\n{body}")

def post_pre_session_briefing(pair, news_times):
    """Post a pre-session briefing for a pair before its range window opens."""
    cfg = PAIR_SESSION[pair]
    try:
        candles = get_candles(pair, count=500, granularity="M1")
        if not candles:
            return
        df        = pd.DataFrame(candles)
        ma        = df["close"].rolling(MA_PERIOD).mean().iloc[-1]
        last_close = df["close"].iloc[-1]
        trend     = "BULLISH 📈" if last_close > ma else "BEARISH 📉"

        # Yesterday high/low
        df["date"] = df["time"].dt.date
        today      = london_now().date()
        import datetime as dt_mod
        yesterday  = today - dt_mod.timedelta(days=1)
        yest_bars  = df[df["date"] == yesterday]
        yh = f"{yest_bars['high'].max():.{PAIR_DECIMALS[pair]}f}" if not yest_bars.empty else "N/A"
        yl = f"{yest_bars['low'].min():.{PAIR_DECIMALS[pair]}f}"  if not yest_bars.empty else "N/A"

        pair_news = news_times.get(cfg["news_currency"], [])
        news_str  = ", ".join(f"{h:02d}:{m:02d}" for h, m in pair_news) if pair_news else "None today ✅"
        session   = "London" if PAIR_SESSION[pair]["range_hour"] == 8 else "New York"
        rh        = cfg["range_hour"]
        rs        = cfg["range_min_start"]
        re        = cfg["range_min_end"]
        ah        = cfg["after_range_hour"]
        am        = cfg["after_range_min"]
        eh        = cfg["entry_window_end_hour"]
        xh        = cfg["exit_hour"]

        signal(
            f"📋 <b>{session} Session Briefing — {pair}</b>\n\n"
            f"Range Window: {rh:02d}:{rs:02d}–{rh:02d}:{re:02d}\n"
            f"Entry Window: {ah:02d}:{am:02d}–{eh:02d}:00\n"
            f"Session Close: {xh:02d}:00\n\n"
            f"Yesterday High: <b>{yh}</b>\n"
            f"Yesterday Low:  <b>{yl}</b>\n"
            f"Current Price:  <b>{last_close:.{PAIR_DECIMALS[pair]}f}</b>\n\n"
            f"MA200 Trend: <b>{trend}</b>\n"
            f"News ({cfg['news_currency']}): {news_str}\n\n"
            f"Watching for ORB breakout after {ah:02d}:{am:02d}."
        )
    except Exception as e:
        log.warning(f"Pre-session briefing failed for {pair}: {e}")

def post_trade_reasoning(pair, direction, rh, rl, rs_size, ma, last_close, news_clear):
    """Post the reasoning behind a setup before the order fills."""
    pip        = PAIR_PIP_SIZE[pair]
    dec        = PAIR_DECIMALS[pair]
    r_pips     = rs_size / pip
    trend      = "above MA200" if last_close > ma else "below MA200"
    breakout_level = rh if direction == 'long' else rl
    bias_str   = 'LONG' if direction == 'long' else 'SHORT'
    bo_dir_str = 'above' if direction == 'long' else 'below'
    news_str   = 'clear' if news_clear else 'near news - caution'
    signal(
        f"Setup Identified - {pair}\n\n"
        f"Range: {rl:.{dec}f} - {rh:.{dec}f} ({r_pips:.1f} pips)\n"
        f"MA200 filter: price {trend}\n"
        f"Direction bias: {bias_str}\n"
        f"News: {news_str}\n\n"
        f"Watching for breakout {bo_dir_str} {breakout_level:.{dec}f}"
    )

def post_trade_breakdown(pair, direction, ep, xp, xpnl, be_set, pl_set, rs_size):
    """Post an educational breakdown after a trade closes."""
    pip    = PAIR_PIP_SIZE[pair]
    dec    = PAIR_DECIMALS[pair]
    r_pips = rs_size / pip if rs_size else 0
    result = "WIN ✅" if (xpnl or 0) > 0 else "LOSS ❌"
    pips   = round((xp - ep) / pip, 1) if direction == "long" else round((ep - xp) / pip, 1) if xp and ep else 0

    if (xpnl or 0) > 0:
        if pl_set:
            outcome = "Ideal sequence: BE secured → profit lock triggered → full TP reached.\nThis is the best case ORB outcome."
        elif be_set:
            outcome = "BE was secured before TP. Trade ran to target from a risk-free position.\nSolid execution."
        else:
            outcome = "Trade hit TP before BE could be set — fast move in our favour.\nClean breakout with strong follow-through."
    else:
        if be_set:
            outcome = "Stop was at breakeven — zero loss on this trade.\nBE management protected the account perfectly."
        else:
            outcome = "Price reversed before reaching BE. Stop loss triggered as planned.\nThe filter system did its job — this is a normal part of any strategy."

    signal(
        f"📖 <b>Trade Breakdown — {pair} {result}</b>\n\n"
        f"Direction: {'LONG' if direction == 'long' else 'SHORT'}\n"
        f"Entry: <b>{ep:.{dec}f}</b>  Exit: <b>{xp:.{dec}f}</b>\n"
        f"Result: <b>{pips:+.1f} pips</b>\n"
        f"Range was: {r_pips:.1f} pips\n"
        f"Breakeven set: {'✅' if be_set else '❌'}  Profit lock: {'✅' if pl_set else '❌'}\n\n"
        f"{outcome}"
    )

def post_weekly_stats():
    """Post weekly performance stats every Friday at 21:00."""
    try:
        today = london_now().date()
        import datetime as dt_mod
        week_start = today - dt_mod.timedelta(days=today.weekday())

        trades = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                for row in csv.DictReader(f):
                    try:
                        row_date = dt_mod.date.fromisoformat(row["date"])
                        if row_date >= week_start:
                            trades.append(row)
                    except Exception:
                        continue

        n    = len(trades)
        wins = sum(1 for t in trades if t.get("pnl") and float(t["pnl"]) > 0)
        loss = n - wins
        wr   = (wins / n * 100) if n > 0 else 0
        pnl  = sum(float(t["pnl"]) for t in trades if t.get("pnl"))

        by_pair = {}
        for t in trades:
            p = t.get("pair", "?")
            by_pair.setdefault(p, []).append(float(t["pnl"]) if t.get("pnl") else 0)
        pair_lines = "\n".join(
            f"  {p}: {sum(1 for x in pnls if x > 0)}W "
            f"{sum(1 for x in pnls if x <= 0)}L  ({sum(pnls):+.2f})"
            for p, pnls in by_pair.items()
        )

        signal(
            f"📊 <b>Weekly Review — Week {today.isocalendar()[1]}</b>\n\n"
            f"Trades: {n}  ✅ {wins}W / ❌ {loss}L  ({wr:.0f}% WR)\n"
            f"Week P&amp;L: <b>{pnl:+.2f}</b>\n\n"
            f"By pair:\n{pair_lines if pair_lines else '  No trades this week'}"
        )
    except Exception as e:
        log.warning(f"Weekly stats post failed: {e}")

# =============================================================================
# PARAMETERS — identical to backtest
# =============================================================================
# Risk per trade 0.65% — user-chosen for the FTMO 1-Step (MC: ~99% pass,
# median ~5-7 weeks; the trailing 10% DD starts taxing pass-rate above ~0.8%,
# so RISK_MAX is capped there).
RISK_PER_TRADE        = 0.0065
BREAKOUT_BODY_MIN_PCT = 0.30
NEWS_BUFFER_MINS      = 15

# Dynamic risk sizing — scales with recent performance
RISK_BASE      = 0.0065  # normal risk per trade
RISK_MAX       = 0.008   # cap after consecutive wins (MC: pass-rate drops past this)
RISK_MIN       = 0.0035  # floor after consecutive losses
RISK_WIN_STEP  = 0.001   # increase per consecutive win
RISK_LOSS_STEP = 0.001   # decrease per consecutive loss

# TP and profit lock are per-session — London window (4.5 h) rarely reaches 2.5×
# London (GBPUSD, GBPJPY): TP=2.0×, lock 1 R at 1.5×
# NY     (EURUSD):          TP=2.5×, lock 1.5 R at 2.0×  (6 h window supports higher target)

PAIR_PIP_SIZE = {
    "GBPUSD": 0.0001, "GBPJPY": 0.010, "EURUSD": 0.0001,
    "USDJPY": 0.010,  "EURJPY": 0.010, "XAUUSD": 1.00,
    # Indices: "pip" = 1 index point.
    "US500": 1.00, "NAS100": 1.00, "US30": 1.00,
}
PAIR_DECIMALS = {
    "GBPUSD": 5, "GBPJPY": 3, "EURUSD": 5,
    "USDJPY": 3, "EURJPY": 3, "XAUUSD": 2,
    # Indices: most brokers quote 1 dp (confirm on the broker; harmless if 0).
    "US500": 1, "NAS100": 1, "US30": 1,
}

MIN_RANGE_BARS = 20   # shared — both sessions require at least 20 completed range bars

# Per-pair session configuration — all time values are London time (Europe/London tz)
PAIR_SESSION = {
    "GBPUSD": {
        "range_hour":            8,    # range candles: 08:00–08:29
        "range_min_start":       0,
        "range_min_end":         29,
        "after_range_hour":      8,    # after_range = h>8 or (h==8 and m>=30)
        "after_range_min":       30,
        "entry_window_end_hour": 11,   # no new orders at or after 11:00
        "exit_hour":             13,   # hard session exit at 13:00
        "min_range_pips":        3,
        "max_range_pips":        30,
        "news_currency":         "GBP",
        "tp_multiplier":         2.0,  # London 4.5 h window — 2.0× backtested
        "partial_tp_r":          1.0,  # close 50% at 0.5× range; move SL to entry
        "profit_lock_trigger":   1.5,  # lock profit once price reaches 1.5× range
        "profit_lock_sl_pct":    1.0,  # lock in 1 R at 1.5× — sequence: BE@0.5×, lock@1.5×, TP@2×
    },
    "GBPJPY": {
        "range_hour":            8,
        "range_min_start":       0,
        "range_min_end":         29,
        "after_range_hour":      8,
        "after_range_min":       30,
        "entry_window_end_hour": 11,
        "exit_hour":             13,
        "min_range_pips":        3,
        "max_range_pips":        30,
        "news_currency":         "GBP",
        "tp_multiplier":         2.0,  # same as GBPUSD — identical parameters, no additional fitting
        "partial_tp_r":          1.0,
        "profit_lock_trigger":   1.5,
        "profit_lock_sl_pct":    1.0,
    },
    "EURUSD": {
        "range_hour":            14,   # range candles: 14:30–14:59 (09:30–09:59 EST)
        "range_min_start":       30,
        "range_min_end":         59,
        "after_range_hour":      15,   # after_range = h >= 15
        "after_range_min":       0,
        "entry_window_end_hour": 21,   # no new orders at or after 21:00
        "exit_hour":             21,   # hard session exit at 21:00 (16:00 EST close)
        "min_range_pips":        3,
        "max_range_pips":        30,
        "news_currency":         "USD",
        "tp_multiplier":         2.5,  # NY 6 h window — 2.5× backtested at 98% MC pass rate
        "partial_tp_r":          1.0,  # close 50% at 0.5× range; move SL to entry
        "profit_lock_trigger":   2.0,  # lock profit once price reaches 2.0× range
        "profit_lock_sl_pct":    1.5,  # lock in 1.5 R at 2.0× — sequence: BE@0.5×, lock@2×, TP@2.5×
    },
    "USDJPY": {                        # NY session — same timing as EURUSD
        "range_hour":            14,
        "range_min_start":       30,
        "range_min_end":         59,
        "after_range_hour":      15,
        "after_range_min":       0,
        "entry_window_end_hour": 21,
        "exit_hour":             21,
        "min_range_pips":        3,
        "max_range_pips":        30,
        "news_currency":         "USD",
        "tp_multiplier":         2.5,
        "partial_tp_r":          1.0,
        "profit_lock_trigger":   2.0,
        "profit_lock_sl_pct":    1.5,
    },
    "EURJPY": {                        # London session — same timing as GBP pairs
        "range_hour":            8,
        "range_min_start":       0,
        "range_min_end":         29,
        "after_range_hour":      8,
        "after_range_min":       30,
        "entry_window_end_hour": 11,
        "exit_hour":             13,
        "min_range_pips":        3,
        "max_range_pips":        30,
        "news_currency":         "GBP",
        "tp_multiplier":         2.0,
        "partial_tp_r":          1.0,
        "profit_lock_trigger":   1.5,
        "profit_lock_sl_pct":    1.0,
    },
    "XAUUSD": {                        # NY session — wider range filter for gold
        "range_hour":            14,
        "range_min_start":       30,
        "range_min_end":         59,
        "after_range_hour":      15,
        "after_range_min":       0,
        "entry_window_end_hour": 21,
        "exit_hour":             21,
        "min_range_pips":        5,    # $5 minimum range (pip = $1 for gold)
        "max_range_pips":        50,   # $50 maximum range
        "news_currency":         "USD",
        "tp_multiplier":         2.5,
        "partial_tp_r":          1.0,
        "profit_lock_trigger":   2.0,
        "profit_lock_sl_pct":    1.5,
    },
    "US500": {                         # NY session — validated ORB (Sharpe +3.8 sim / ~1-2 real)
        "range_hour":            14,   # range 14:30-14:59 London = 09:30-09:59 ET (cash open)
        "range_min_start":       30,
        "range_min_end":         59,
        "after_range_hour":      15,
        "after_range_min":       0,
        "entry_window_end_hour": 21,
        "exit_hour":             21,   # 16:00 ET cash close
        "min_range_pips":        5,    # points; median 1R ~21pt, 0.1-1.0% of price band
        "max_range_pips":        80,
        "news_currency":         "USD",
        "tp_multiplier":         2.5,
        "partial_tp_r":          1.0,
        "profit_lock_trigger":   2.0,
        "profit_lock_sl_pct":    1.5,
    },
    "NAS100": {                        # NY session — validated ORB (Sharpe +3.5 sim / ~1-2 real)
        "range_hour":            14,
        "range_min_start":       30,
        "range_min_end":         59,
        "after_range_hour":      15,
        "after_range_min":       0,
        "entry_window_end_hour": 21,
        "exit_hour":             21,
        "min_range_pips":        30,   # points; median 1R ~116pt
        "max_range_pips":        350,
        "news_currency":         "USD",
        "tp_multiplier":         2.5,
        "partial_tp_r":          1.0,
        "profit_lock_trigger":   2.0,
        "profit_lock_sl_pct":    1.5,
    },
    "US30": {                          # NY session — validated ORB (Sharpe +3.6 sim / ~1-2 real)
        "range_hour":            14,
        "range_min_start":       30,
        "range_min_end":         59,
        "after_range_hour":      15,
        "after_range_min":       0,
        "entry_window_end_hour": 21,
        "exit_hour":             21,
        "min_range_pips":        50,   # points; median 1R ~191pt
        "max_range_pips":        550,
        "news_currency":         "USD",
        "tp_multiplier":         2.5,
        "partial_tp_r":          1.0,
        "profit_lock_trigger":   2.0,
        "profit_lock_sl_pct":    1.5,
    },
}

# Fallbacks used only if the live calendar fetch fails (London times)
HIGH_IMPACT_FALLBACK = {
    "GBP": [(7, 0), (9, 0), (9, 30), (10, 0)],
    "USD": [(13, 30), (15, 0), (15, 30)],
}

# ── Firm profile — verified rule sets (June 2026). Flip FIRM to switch. ──────
# FTMO_1STEP (ftmo.com/en/trading-objectives, June 2026):
#   * 3% daily loss, baseline = BALANCE at 00:00 CE(S)T (Prague midnight)
#   * 10% maximum loss, TRAILING the highest midnight balance; NEVER locks
#     (withdrawals reset it); challenge target +10%, unlimited time
#   * funded "Best Day" rule: best day <= 50% of Positive Days' Profit
#   * funded: 90% split from first payout; 2-min news window; no weekend holds
#     on funded (trend_sleeve.py must NOT run on FTMO)
# BLUE_GUARDIAN ($10k Instant):
#   * 3% daily, baseline = max(balance, equity) at 5 PM New York
#   * 6% TRAILING max DD (locks at breakeven at +6%); 20% consistency;
#     payout needs 5 days >= 0.5% closed, bi-weekly
FIRM = "FTMO_1STEP"

FIRM_PROFILES = {
    "FTMO_1STEP": dict(
        daily_firm=0.03, dd_dist=0.10, dd_lock_at_initial=False,
        day_tz="Europe/Prague", day_roll_shift_h=0, day_baseline_mode="balance",
        guard_cap=0.03,   # loose: FTMO Best-Day is per positive-days, not total
        qual_days_needed=0, consistency_mode="best_le_50pct_positive",
    ),
    "BLUE_GUARDIAN": dict(
        daily_firm=0.03, dd_dist=0.06, dd_lock_at_initial=True,
        day_tz="America/New_York", day_roll_shift_h=7, day_baseline_mode="max",
        guard_cap=0.015,
        qual_days_needed=5, consistency_mode="best_lt_20pct_total",
    ),
}
PROFILE = FIRM_PROFILES[FIRM]

PROP_DAILY_LOSS_LIMIT   = 0.02   # 2% self-imposed daily cap (firms allow 3%)

# Initial deposit — drawdown reference level. Set to the funded size.
INITIAL_DEPOSIT: float = 10000.0

# Module-level starting balance — populated once `main()` connects to MT5 and
# read by `_pre_trade_risk_gate` from any worker thread without having to
# thread the value through every entry-function call.
_STARTING_BALANCE: float = 0.0
PROP_MAX_DRAWDOWN_LIMIT = PROFILE["dd_dist"]   # trailing distance (of initial)
PROP_DD_SAFETY_MARGIN   = 0.005  # flatten 0.5% BEFORE the firm's line

# Consistency day-cap guard: once today's CLOSED profit reaches this fraction of
# INITIAL_DEPOSIT, stop opening new trades for the day (open positions keep
# managing). Tight (1.5%) under BG's 20%-of-total rule; loose (3%) under FTMO's
# Best-Day<=50%-of-positive-days, where speed matters more than smoothing.
CONSISTENCY_DAY_CAP     = PROFILE["guard_cap"]
CONSISTENCY_GUARD_ON    = True

# ── Persistent prop state — survives restarts; shared with trend_sleeve.py ────
# {"hwm": float, "prop_day": "YYYY-MM-DD", "day_baseline": float,
#  "last_payout_date": "YYYY-MM-DD"}  (edit last_payout_date after each payout)
PROP_STATE_FILE = "prop_state.json"
_PROP_DAY_TZ = pytz.timezone(PROFILE["day_tz"])


def _load_prop_state() -> dict:
    try:
        import json as _json
        with open(PROP_STATE_FILE, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}


def _save_prop_state(updates: dict):
    """Read-merge-write with atomic replace — tolerates the trend-sleeve process
    writing the same file (HWM is re-maxed on merge, so a lost race self-heals)."""
    try:
        import json as _json
        state = _load_prop_state()
        if "hwm" in updates and "hwm" in state:
            updates = dict(updates)
            updates["hwm"] = max(float(updates["hwm"]), float(state["hwm"]))
        state.update(updates)
        tmp = PROP_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(state, f, indent=2)
        os.replace(tmp, PROP_STATE_FILE)
    except Exception as e:
        log.warning(f"prop_state save failed: {e}")


def prop_day_today() -> str:
    """The firm's trading-day id. FTMO rolls at 00:00 CE(S)T (shift 0, Prague);
    Blue Guardian at 17:00 New York (shift +7h makes the date flip at 5 PM)."""
    now_tz = datetime.now(pytz.utc).astimezone(_PROP_DAY_TZ)
    return (now_tz + timedelta(hours=PROFILE["day_roll_shift_h"])).date().isoformat()


def trailing_floor(hwm: float) -> float:
    """Firm max-loss floor, trailing the HWM by the profile's distance.
    FTMO 1-Step: never locks (floor keeps rising with the HWM forever; a
    withdrawal resets it — handled manually via prop_state hwm). Blue Guardian:
    locks at breakeven once HWM >= initial + distance.
    Note: we track the HWM of intraday max(balance, equity), which is >= FTMO's
    midnight-balance HWM — deliberately conservative."""
    floor = hwm - PROP_MAX_DRAWDOWN_LIMIT * INITIAL_DEPOSIT
    if PROFILE["dd_lock_at_initial"]:
        floor = min(floor, INITIAL_DEPOSIT)
    return floor

# Magic number — stamped on every order/position so we can distinguish ours from
# manual trades or other EAs running on the same account.
ORB_MAGIC   = 770801
# Companion-sleeve magics on the SAME account: trend_sleeve.py (separate process)
# and the Dow overlay (defined with its config below). Listed here so risk
# accounting can recognise every position the whole book owns.
TREND_MAGIC = 770802   # trend_sleeve.py — MA100 long/flat, held across days

# Roster policy: every instrument must pass the validation gauntlet (p<0.01,
# both directions positive, train/test/holdout all positive, cost-stressed).
# Validated 2026-06: EURUSD, USDJPY, GBPUSD, EURJPY (fx_orb_validation.py),
# XAUUSD, US500, NAS100, US30 (index_gold_orb.py / index_orb_indices.py).
# GBPJPY REMOVED — no M1 data available, never validated; re-add only after
# fetching its history and passing the same gauntlet.
PAIRS            = ["GBPUSD", "EURUSD", "USDJPY", "EURJPY", "XAUUSD",
                    "US500", "NAS100", "US30"]
STREAM_PAIRS     = ["GBPUSD", "EURUSD", "USDJPY", "EURJPY", "XAUUSD",
                    "US500", "NAS100", "US30"]
# Instruments that share the same dominant exposure — when *any* of these is
# active, the others size at half-risk. EURUSD/USDJPY removed: they are typically
# anti-correlated (both driven by USD), so halving on those would amplify USD
# exposure when running opposing trades, not dampen it.
# US indices form a ~0.95-correlated cluster — half-size when co-active so the
# (otherwise concentrated long-equity) book doesn't stack one big beta bet.
CORRELATION_GROUPS = [
    {"GBPUSD", "GBPJPY", "EURJPY"},   # GBP/cross-yen cluster
    {"US500", "NAS100", "US30"},      # US equity-index cluster
]
MA_PERIOD  = 200

# Dow-dispersion overlay (Edge 2): beta-free weekly pattern — long Monday /
# short Thursday, intraday on the US-equity cash session, exit at the close.
# Validated on US500 + NAS100 (ANOVA p<=0.001); equity-specific (doesn't
# generalise to gold/FX). Protective stop caps a bad day; edge is hold-to-close.
DOW_PAIRS          = ["US500", "NAS100"]
DOW_STOP_PCT       = 0.007   # 0.7% protective stop (the strategy itself has no SL)
DOW_LONG_WEEKDAY   = 0       # Monday
DOW_SHORT_WEEKDAY  = 3       # Thursday
DOW_MAGIC          = 770803  # distinct tag so Dow positions are identifiable

# Normal spreads — used for spread guard (reject entry if spread > 2× normal).
# Index spreads are broker-specific; these are typical Blue Guardian-ish points —
# the rolling-median spread guard adapts after warm-up, so these are warm-up only.
PAIR_SPREAD_NORMAL = {
    "GBPUSD": 0.00015, "GBPJPY": 0.025,  "EURUSD": 0.00012,
    "USDJPY": 0.010,   "EURJPY": 0.020,  "XAUUSD": 0.30,
    "US500": 0.50,     "NAS100": 2.00,   "US30": 3.00,
}

# =============================================================================
# REGIME DETECTION PARAMETERS
# =============================================================================
REGIME_ADX_TREND    = 25     # ADX > 25 → TRENDING (breakouts extend)
REGIME_ADX_RANGE    = 20     # ADX < 20 → RANGING  (fade false breakouts; SB at full risk)
REGIME_ATR_VOLATILE = 1.5    # ATR_ratio > 1.5 → VOLATILE (skip all strategies)
REGIME_ADX_PERIOD   = 14
REGIME_ATR_REF_BARS = 30     # bars for rolling ATR mean

# ORB risk multiplier per regime — 0.0 = skip ORB entirely
# Regime audit 2026-06-10 (regime_audit.py, 2122 validated trades): the old
# RANGING=0.0 skip forfeited 47% of trades averaging +0.34R (positive on all 8
# instruments) — the "false breakouts dominate in ranging" premise is false for
# this strategy; ADX-14 on M1 at the open mostly measures the quiet overnight
# session. TRANSITIONING was the BEST bucket (+0.42R) — 0.75x unjustified.
# VOLATILE kept at 0.0 as cheap tail-risk insurance (n=2, both losers).
REGIME_ORB_MULT = {
    'TRENDING':      1.0,    # +0.36R avg (657 trades)
    'RANGING':       1.0,    # +0.34R avg (995 trades) — was 0.0, validated profitable
    'VOLATILE':      0.0,    # n=2, both losses — keep skipping vol spikes
    'TRANSITIONING': 1.0,    # +0.42R avg (468 trades) — was 0.75
    'UNDEFINED':     0.5,    # regime data missing — stay conservative
}

# Consecutive loss pause — stop trading after this many full-stop losses in a day
MAX_CONSECUTIVE_LOSSES = 3   # 3 losses in a row (~12.5% chance at 50% WR) — 2 was too aggressive
POLL_SECS  = 30   # main thread: fill detection + balance check interval

# =============================================================================
# ECONOMIC CALENDAR — fetches real GBP high-impact events for today
# =============================================================================

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

def fetch_todays_news_times():
    """
    Fetch today's high-impact GBP and USD events from ForexFactory's public calendar.
    Returns a dict {"GBP": [...], "USD": [...]} of (hour, minute) tuples in London time.
    Falls back to HIGH_IMPACT_FALLBACK on any error.
    """
    try:
        resp = requests.get(FF_CALENDAR_URL, timeout=10)
        resp.raise_for_status()
        events = resp.json()

        today_str = london_now().strftime("%m-%d-%Y")   # FF format: "04-09-2026"
        by_currency = {"GBP": [], "USD": []}
        for e in events:
            currency = e.get("country")
            if currency not in by_currency:
                continue
            if e.get("impact") != "High":
                continue
            if e.get("date", "").startswith(today_str):
                raw = e.get("date", "")
                try:
                    dt_str, offset_str = re.match(r"(.+)([+-]\d{4})$", raw).groups()
                    dt_naive = datetime.strptime(dt_str, "%m-%d-%YT%H:%M:%S")
                    sign     = 1 if offset_str[0] == "+" else -1
                    oh, om   = int(offset_str[1:3]), int(offset_str[3:5])
                    offset   = sign * (oh * 60 + om)
                    dt_utc   = dt_naive.replace(tzinfo=timezone.utc) - timedelta(minutes=offset)
                    dt_lon   = dt_utc.astimezone(LONDON_TZ)
                    by_currency[currency].append((dt_lon.hour, dt_lon.minute))
                except Exception:
                    continue

        for ccy, times in by_currency.items():
            if times:
                log.info(f"News filter [{ccy}]: {len(times)} high-impact event(s): {times}")
            else:
                log.info(f"News filter [{ccy}]: no high-impact events today")
        return by_currency

    except Exception as e:
        log.warning(f"News calendar fetch failed ({e}) — using fallback times")
        return dict(HIGH_IMPACT_FALLBACK)

# =============================================================================
# SHARED STATE — accessed by both threads; always hold _lock when reading/writing
# =============================================================================
_price_cache = {}   # symbol -> {"bid": float, "ask": float, "mid": float}

# Today's high-impact news times, published by the main thread for the stream
# thread's news-window guard. Updated by whole-dict ASSIGNMENT (atomic in
# CPython), so readers don't need the lock.
_news_times_today: dict = dict(HIGH_IMPACT_FALLBACK)
_lock        = threading.Lock()

# =============================================================================
# ORDER-SEND LATENCY MONITOR
# Median latency over recent order_send calls. Elevated medians signal a
# degraded VPS-broker network path long before any single send fails.
# =============================================================================
_order_latencies          = deque(maxlen=100)
_ORDER_LATENCY_WARN_MS    = 200      # median over the rolling window
_ORDER_LATENCY_WARN_COOLDOWN = 600   # at most one warn every 10 min
_last_latency_warn_ts     = 0.0


def _record_order_latency(ms: float):
    """Record an order_send round-trip; warn if the rolling median degrades."""
    global _last_latency_warn_ts
    _order_latencies.append(float(ms))
    if len(_order_latencies) < 20:
        return
    ordered = sorted(_order_latencies)
    median  = ordered[len(ordered) // 2]
    if median > _ORDER_LATENCY_WARN_MS and (time.time() - _last_latency_warn_ts) > _ORDER_LATENCY_WARN_COOLDOWN:
        log.warning(
            f"Order-send median latency elevated: {median:.0f}ms over last "
            f"{len(_order_latencies)} calls — VPS-broker network may be degraded"
        )
        _last_latency_warn_ts = time.time()


# =============================================================================
# ROLLING SPREAD GUARD
# Static per-pair "normal spread" thresholds drift over time and across brokers.
# Sample live spreads in the stream thread, then gate entries on the rolling
# median × multiplier. The deque is bounded so the window is roughly the last
# few minutes of stream activity.
# =============================================================================
_SPREAD_WINDOW             = 600    # ~60s at 100ms stream cadence
_SPREAD_MIN_SAMPLES        = 50     # require this many before trusting the median
_SPREAD_GATE_MULTIPLIER    = 2.5    # gate fires above median × this
_spread_history: dict      = {}     # pair (base) -> deque[float]


def _record_spread_sample(pair: str, spread: float):
    """Append a spread sample for `pair`. Caller MUST already hold _lock."""
    dq = _spread_history.get(pair)
    if dq is None:
        dq = deque(maxlen=_SPREAD_WINDOW)
        _spread_history[pair] = dq
    if spread >= 0:
        dq.append(float(spread))


def _rolling_spread_median(pair: str):
    """Return the rolling median spread for `pair`, or None if not enough samples."""
    with _lock:
        dq = _spread_history.get(pair)
        if dq is None or len(dq) < _SPREAD_MIN_SAMPLES:
            return None
        snapshot = list(dq)
    snapshot.sort()
    return snapshot[len(snapshot) // 2]


# =============================================================================
# MT5 API HELPERS
# =============================================================================

def units_to_lots(units):
    """Convert currency units to MT5 lots (1 lot = 100,000 units). Min 0.01."""
    return max(0.01, round(units / 100_000, 2))

def get_balance():
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
    return float(info.balance)

def get_account_state():
    """Return (balance, equity) — equity includes unrealized P&L."""
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
    return float(info.balance), float(info.equity)

def get_candles(pair, count=250, granularity="M1"):
    """
    Fetch the last `count` completed M1 candles.
    Returns a list of dicts: {time, open, high, low, close}.
    start_pos=1 skips the currently-forming bar.
    """
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1,
    }
    tf    = tf_map.get(granularity, mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_from_pos(_sym(pair), tf, 1, count)
    if rates is None or len(rates) == 0:
        return []
    result = []
    for r in rates:
        ts = pd.Timestamp(int(r["time"]), unit="s", tz="UTC").tz_convert(LONDON_TZ)
        result.append({
            "time":  ts,
            "open":  float(r["open"]),
            "high":  float(r["high"]),
            "low":   float(r["low"]),
            "close": float(r["close"]),
        })
    return result

def get_open_trade(pair, trade_id=None, direction=None):
    """
    Return the open position dict for this pair, or None.

    trade_id  — when provided, only return the position with this specific ticket.
                Used by IN_TRADE closed-detection: guarantees we check our own
                position even if another strategy also holds a position on the pair.
    direction — when provided (and trade_id is None), only return a position whose
                side matches ("long" / "short").  Used by fill-detection fallbacks
                to avoid adopting a position placed by the other strategy.
    """
    positions = mt5.positions_get(symbol=_sym(pair))
    if not positions:
        return None
    # Only consider OUR positions — manual trades or other EAs use a different magic.
    positions = [p for p in positions if getattr(p, "magic", 0) == ORB_MAGIC]
    if not positions:
        return None

    if trade_id is not None:
        pos = next((p for p in positions if str(p.ticket) == str(trade_id)), None)
    elif direction is not None:
        expected_type = mt5.POSITION_TYPE_BUY if direction == "long" else mt5.POSITION_TYPE_SELL
        pos = next((p for p in positions if p.type == expected_type), None)
    else:
        pos = positions[0]

    if pos is None:
        return None

    signed_units = pos.volume * 100_000 if pos.type == mt5.POSITION_TYPE_BUY else -pos.volume * 100_000
    return {
        "id":             str(pos.ticket),
        "instrument":     pos.symbol,
        "price":          pos.price_open,
        "currentUnits":   signed_units,
        "stopLossOrder":  {"price": str(pos.sl)} if pos.sl else {},
        "takeProfitOrder":{"price": str(pos.tp)} if pos.tp else {},
        "_raw":           pos,
    }

def get_pending_order(pair, order_type=None):
    """
    Return a pending order dict for this pair, or None.
    If order_type is "STOP" or "LIMIT", only match that type.
    """
    _stop_types  = {mt5.ORDER_TYPE_BUY_STOP,  mt5.ORDER_TYPE_SELL_STOP}
    _limit_types = {mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT}
    orders = mt5.orders_get(symbol=_sym(pair))
    if not orders:
        return None
    for o in orders:
        if getattr(o, "magic", 0) != ORB_MAGIC:
            continue
        if order_type == "STOP"  and o.type not in _stop_types:
            continue
        if order_type == "LIMIT" and o.type not in _limit_types:
            continue
        ot = "STOP" if o.type in _stop_types else "LIMIT"
        return {"id": str(o.ticket), "type": ot, "instrument": pair, "_raw": o}
    return None

_MT5_RETCODES = {
    10004: "Requote",            10006: "Request rejected",
    10007: "Request cancelled",  10010: "Only part of request done",
    10013: "Invalid request",    10014: "Invalid volume",
    10015: "Invalid price",      10016: "Invalid stops",
    10017: "Trade disabled",     10018: "Market closed",
    10019: "Not enough money",   10020: "Prices changed",
    10021: "No quotes",          10022: "Invalid expiration",
    10024: "Too many requests",  10025: "No changes",
    10027: "AutoTrading disabled",10028: "Broker busy",
    10030: "Frozen",             10031: "Invalid fill",
    10033: "Too many orders",    10036: "Expiration denied",
    10038: "Trade disabled for symbol",
}

def place_stop_order(pair, direction, units, trigger, sl, tp, expire_utc):
    """
    Place an MT5 stop order. Returns order ID string or None on failure.

    Idempotent: tags every order with a deterministic comment
    (ORB_<pair>_<YYYYMMDD>_<direction>) so duplicate placement attempts after
    a network blip can be detected by querying open orders for the tag.
    Also enforces a pre-trade tick-age check and logs order_send latency.
    """
    dec        = PAIR_DECIMALS[pair]
    broker_sym = _sym(pair)
    lots       = units_to_lots(units)
    order_type = mt5.ORDER_TYPE_BUY_STOP if direction == "long" else mt5.ORDER_TYPE_SELL_STOP

    # Pre-trade tick-age check — refuse to place if quote is stale (>2s old).
    tick = mt5.symbol_info_tick(broker_sym)
    if tick is None:
        log.error(f"[{pair}] No tick available — refusing to place order")
        return None
    tick_age = time.time() - tick.time
    if tick_age > 2.0:
        log.warning(f"[{pair}] Stale tick ({tick_age:.1f}s old) — refusing to place order")
        return None

    # Enforce broker's minimum stop distance (retcode 10016 if violated)
    sl_orig = sl
    sym = mt5.symbol_info(broker_sym)
    if sym and sym.trade_stops_level > 0:
        min_dist = sym.trade_stops_level * sym.point
        if direction == "long":
            sl = min(sl, round(trigger - min_dist, dec))
            tp = max(tp, round(trigger + min_dist, dec))
        else:
            sl = max(sl, round(trigger + min_dist, dec))
            tp = min(tp, round(trigger - min_dist, dec))

    # Re-size: if broker pushed SL further from entry, the position is now
    # overleveraged for the requested risk. Scale units down proportionally
    # so dollar-risk stays constant.
    if direction == "long":
        orig_dist = trigger - sl_orig
        new_dist  = trigger - sl
    else:
        orig_dist = sl_orig - trigger
        new_dist  = sl - trigger
    if new_dist > orig_dist > 0:
        scale = orig_dist / new_dist
        units = max(1, int(units * scale))
        lots  = units_to_lots(units)
        log.warning(f"[{pair}] Broker SL clamp widened stop {orig_dist:.{dec}f}->{new_dist:.{dec}f} "
                    f"— rescaling units by {scale:.3f} to preserve risk (units={units}, lots={lots})")

    # Round lots to broker volume_step (XAUUSD often requires 0.01 step
    # but some brokers require 0.1; rejection retcode 10014 if violated).
    step, vmin = _broker_volume_step(pair)
    lots = max(vmin, _round_to_step(lots, step, vmin))

    # Determine which filling modes the broker actually supports (bitmask)
    supported = sym.filling_mode if sym else 7
    candidates = []
    if supported & 1:  candidates.append(mt5.ORDER_FILLING_FOK)
    if supported & 2:  candidates.append(mt5.ORDER_FILLING_IOC)
    if supported & 4:  candidates.append(mt5.ORDER_FILLING_RETURN)
    if not candidates: candidates = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC,
                                      mt5.ORDER_FILLING_FOK]

    # Idempotency: deterministic tag lets us detect and adopt a duplicate place
    # attempt rather than firing a second order on the same setup.
    tag = f"ORB_{pair}_{london_now().strftime('%Y%m%d')}_{direction}"
    existing = mt5.orders_get(symbol=broker_sym) or []
    for o in existing:
        if (o.comment or "") == tag:
            log.warning(f"[{pair}] Idempotent guard: order with tag {tag} already exists "
                        f"(id={o.ticket}) — adopting instead of placing again")
            return str(o.ticket)

    # If the caller's expiration is already in the past or too close to now,
    # MT5 rejects it with retcode 10022. Fall back to GTC (we cancel the order
    # ourselves once the entry window closes).
    now_ts = int(datetime.now(timezone.utc).timestamp())
    use_specified = expire_utc and (expire_utc - now_ts) >= 60

    for filling in candidates:
        request = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       broker_sym,
            "volume":       lots,
            "type":         order_type,
            "price":        round(trigger, dec),
            "sl":           round(sl, dec),
            "tp":           round(tp, dec),
            "type_time":    mt5.ORDER_TIME_SPECIFIED if use_specified else mt5.ORDER_TIME_GTC,
            "type_filling": filling,
            "magic":        ORB_MAGIC,
            "comment":      tag,
        }
        if use_specified:
            request["expiration"] = expire_utc
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        _record_order_latency(latency_ms)
        if result is None:
            log.warning(f"[{pair}] order_send None — last_error={mt5.last_error()} "
                        f"filling={filling} trigger={trigger:.{dec}f} "
                        f"sl={sl:.{dec}f} tp={tp:.{dec}f} lots={lots} "
                        f"latency={latency_ms:.0f}ms")
            continue
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            oid = str(result.order)
            log.info(f"[{pair}] Stop order placed — {direction.upper()} "
                     f"trigger={trigger:.{dec}f}  SL={sl:.{dec}f}  TP={tp:.{dec}f}  "
                     f"lots={lots}  id={oid}  filling={filling}  latency={latency_ms:.0f}ms")
            return oid
        meaning = _MT5_RETCODES.get(result.retcode, "unknown")
        log.warning(f"[{pair}] retcode={result.retcode} ({meaning}) "
                    f"comment='{result.comment}' filling={filling} "
                    f"trigger={trigger:.{dec}f} sl={sl:.{dec}f} tp={tp:.{dec}f} "
                    f"lots={lots} latency={latency_ms:.0f}ms")

    log.error(f"[{pair}] Failed to place order after all filling modes — "
              f"last_error={mt5.last_error()}")
    return None

def modify_trade_sl(trade_id, new_sl, pair):
    """
    Move the stop loss on an open position.
    MT5 SLTP action requires both SL and TP — reads current TP from the position.
    Retries up to 3 times on transient errors.
    """
    dec = PAIR_DECIMALS[pair]
    broker_sym = _sym(pair)
    def _do():
        positions = mt5.positions_get(symbol=broker_sym)
        pos = next((p for p in positions if p.ticket == int(trade_id)), None) if positions else None
        current_tp = round(pos.tp, dec) if pos and pos.tp else 0.0
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   broker_sym,
            "position": int(trade_id),
            "sl":       round(new_sl, dec),
            "tp":       current_tp,
            "magic":    ORB_MAGIC,
        }
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        _record_order_latency((time.perf_counter() - t0) * 1000)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"retcode={result.retcode if result else 'None'} "
                               f"comment={result.comment if result else ''}")
    try:
        api_retry(_do)
        log.info(f"[{pair}] SL moved to {new_sl:.{dec}f}  (trade {trade_id})")
    except Exception as e:
        log.error(f"[{pair}] Failed to modify SL after retries: {e}")

def modify_trade_tp(trade_id, new_tp, pair):
    """
    Update the take profit on an open position.
    MT5 SLTP action requires both SL and TP — reads current SL from the position.
    Retries up to 3 times on transient errors.
    """
    dec = PAIR_DECIMALS[pair]
    broker_sym = _sym(pair)
    def _do():
        positions = mt5.positions_get(symbol=broker_sym)
        pos = next((p for p in positions if p.ticket == int(trade_id)), None) if positions else None
        current_sl = round(pos.sl, dec) if pos and pos.sl else 0.0
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   broker_sym,
            "position": int(trade_id),
            "sl":       current_sl,
            "tp":       round(new_tp, dec),
            "magic":    ORB_MAGIC,
        }
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        _record_order_latency((time.perf_counter() - t0) * 1000)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"retcode={result.retcode if result else 'None'} "
                               f"comment={result.comment if result else ''}")
    try:
        api_retry(_do)
        log.info(f"[{pair}] TP updated to {new_tp:.{dec}f}  (trade {trade_id})")
    except Exception as e:
        log.error(f"[{pair}] Failed to modify TP after retries: {e}")

def close_trade_market(trade_id, pair, reason=""):
    """
    Close an open position at market.
    Retries up to 3 times. Returns (exit_price, pnl) or (None, None).
    """
    result_holder = {}
    broker_sym = _sym(pair)
    def _do():
        positions = mt5.positions_get(symbol=broker_sym)
        pos = next((p for p in positions if p.ticket == int(trade_id)), None) if positions else None
        if pos is None:
            raise RuntimeError("Position not found — already closed?")
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick        = mt5.symbol_info_tick(broker_sym)
        close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       broker_sym,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     int(trade_id),
            "price":        close_price,
            "deviation":    20,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": _close_filling_mode(pair),
            "magic":        ORB_MAGIC,
            "comment":      reason,
        }
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        _record_order_latency((time.perf_counter() - t0) * 1000)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"retcode={result.retcode if result else 'None'} "
                               f"comment={result.comment if result else ''}")
        # Detect partial fill — broker honoured volume only partially.
        if hasattr(result, "volume") and result.volume and result.volume + 1e-9 < pos.volume:
            log.warning(f"[{pair}] Partial close: requested {pos.volume} got {result.volume} "
                        f"— retrying for residual")
            raise RuntimeError(f"partial_fill residual={pos.volume - result.volume:.2f}")
        result_holder["deal"] = result.deal
    try:
        api_retry(_do)
        time.sleep(0.5)  # brief wait for deal to appear in history
        exit_price = pnl = None
        deals = mt5.history_deals_get(position=int(trade_id))
        if deals:
            closing = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), None)
            if closing:
                exit_price = closing.price
                pnl = closing.profit + closing.swap + closing.commission
        log.info(f"[{pair}] Trade {trade_id} closed — {reason}  "
                 f"exit={exit_price}  pnl={pnl}")
        return exit_price, pnl
    except Exception as e:
        log.error(f"[{pair}] Failed to close trade {trade_id} after retries: {e}")
        return None, None

def close_partial(trade_id, pair, lots, reason="partial_tp"):
    """
    Close `lots` of an open position at market (partial close).
    Returns (exit_price, pnl) or (None, None) on failure.
    Minimum lot size is 0.01 — caller must ensure remaining lots >= 0.01.
    """
    broker_sym = _sym(pair)
    def _do():
        positions = mt5.positions_get(symbol=broker_sym)
        pos = next((p for p in positions if p.ticket == int(trade_id)), None) if positions else None
        if pos is None:
            raise RuntimeError("Position not found")
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick        = mt5.symbol_info_tick(broker_sym)
        close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        # Round requested volume to broker's step; many brokers (esp. XAUUSD)
        # use step=0.1, not 0.01 — the unrounded request would retcode 10014.
        step, vmin = _broker_volume_step(pair)
        req_vol    = max(vmin, _round_to_step(lots, step, vmin))
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       broker_sym,
            "volume":       req_vol,
            "type":         close_type,
            "position":     int(trade_id),
            "price":        close_price,
            "deviation":    20,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": _close_filling_mode(pair),
            "magic":        ORB_MAGIC,
            "comment":      reason,
        }
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        _record_order_latency((time.perf_counter() - t0) * 1000)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"retcode={result.retcode if result else 'None'} "
                               f"comment={result.comment if result else ''}")
        if hasattr(result, "volume") and result.volume and result.volume + 1e-9 < req_vol:
            log.warning(f"[{pair}] Partial close (partial fill): requested {req_vol} got {result.volume}")
    try:
        api_retry(_do)
        time.sleep(0.5)
        dec   = PAIR_DECIMALS[pair]
        deals = mt5.history_deals_get(position=int(trade_id))
        if deals:
            partials = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
            if partials:
                last = partials[-1]
                pnl  = last.profit + last.swap + last.commission
                log.info(f"[{pair}] Partial close {lots} lots — price={last.price:.{dec}f}  pnl={pnl:.2f}")
                return last.price, pnl
        return None, None
    except Exception as e:
        log.error(f"[{pair}] Failed to partial close trade {trade_id}: {e}")
        return None, None

def get_closed_trade_details(trade_id):
    """Fetch exit price and realized P&L for a position MT5 has already closed."""
    try:
        deals = mt5.history_deals_get(position=int(trade_id))
        if not deals:
            time.sleep(1.0)
            deals = mt5.history_deals_get(position=int(trade_id))
        if deals:
            closing = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), None)
            if closing:
                pnl = closing.profit + closing.swap + closing.commission
                return closing.price, pnl
        return None, None
    except Exception as e:
        log.error(f"Failed to get closed trade details for {trade_id}: {e}")
        return None, None

def _market_filling_mode(pair):
    """Return the best supported filling mode for a market order on this symbol."""
    sym = mt5.symbol_info(_sym(pair))
    supported = sym.filling_mode if sym else 7
    if supported & 2: return mt5.ORDER_FILLING_IOC
    if supported & 4: return mt5.ORDER_FILLING_RETURN
    if supported & 1: return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_IOC


def _close_filling_mode(pair):
    """For closes/exits, prefer FOK so a partial fill can't leave a residual position."""
    sym = mt5.symbol_info(_sym(pair))
    supported = sym.filling_mode if sym else 7
    if supported & 1: return mt5.ORDER_FILLING_FOK   # full-or-nothing
    if supported & 2: return mt5.ORDER_FILLING_IOC
    if supported & 4: return mt5.ORDER_FILLING_RETURN
    return mt5.ORDER_FILLING_FOK


def _round_to_step(value: float, step: float, minimum: float = 0.01) -> float:
    """Round `value` down to the nearest `step` for broker volume compliance."""
    if step <= 0:
        return max(minimum, round(value, 2))
    n = int(value / step)
    out = round(n * step, 2)
    return max(minimum, out)


def _broker_volume_step(pair: str):
    """Return (volume_step, volume_min) for a pair; falls back to (0.01, 0.01)."""
    sym = mt5.symbol_info(_sym(pair))
    step = float(getattr(sym, "volume_step", 0.01)) if sym else 0.01
    vmin = float(getattr(sym, "volume_min",  0.01)) if sym else 0.01
    return step or 0.01, vmin or 0.01

def cancel_pending(order_id, pair):
    """Cancel a pending order. Retries up to 3 times on transient errors."""
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":  int(order_id),
        "magic":  ORB_MAGIC,
    }
    def _do():
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        _record_order_latency((time.perf_counter() - t0) * 1000)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"retcode={result.retcode if result else 'None'} "
                               f"comment={result.comment if result else ''}")
    try:
        api_retry(_do)
        log.info(f"[{pair}] Order {order_id} cancelled")
    except Exception as e:
        log.error(f"[{pair}] Failed to cancel order {order_id} after retries: {e}")

# =============================================================================
# STRATEGY HELPERS
# =============================================================================

def london_now():
    return datetime.now(pytz.utc).astimezone(LONDON_TZ)

def spread_gate(pair):
    """Return True if current spread is elevated — skip entry.

    Compares live bid/ask spread against the known normal spread for the pair.
    Threshold: 2.5× normal.  Prevents entering during news spikes or low liquidity.
    """
    with _lock:
        cached = _price_cache.get(pair)
    if not cached:
        return True
    current_spread = cached["ask"] - cached["bid"]
    # Prefer the rolling median once we have enough samples — adapts to broker
    # and time-of-day. Fall back to the static PAIR_SPREAD_NORMAL during warm-up.
    rolling = _rolling_spread_median(pair)
    if rolling is not None and rolling > 0:
        threshold = rolling * _SPREAD_GATE_MULTIPLIER
        elevated  = current_spread > threshold
        if elevated:
            log.debug(f"[{pair}] spread_gate (rolling): {current_spread:.6f} > "
                      f"{threshold:.6f} ({current_spread / rolling:.1f}× median)")
        return elevated
    normal_spread = PAIR_SPREAD_NORMAL[pair]
    if normal_spread <= 0:
        return False
    elevated = current_spread > normal_spread * _SPREAD_GATE_MULTIPLIER
    if elevated:
        log.debug(f"[{pair}] spread_gate (static): {current_spread:.6f} > "
                  f"{normal_spread * _SPREAD_GATE_MULTIPLIER:.6f} "
                  f"({current_spread / normal_spread:.1f}× normal)")
    return elevated

def is_near_news(now_lon, news_times):
    mins = now_lon.hour * 60 + now_lon.minute
    for h, m in news_times:
        if abs(mins - (h * 60 + m)) <= NEWS_BUFFER_MINS:
            return True
    return False

def fetch_and_cache_range(pair, st):
    """
    Expensive 500-candle fetch — called ONCE per day when range becomes available.
    Caches range + MA in PairState so every subsequent poll is cheap.
    Returns True on success, False on strategy rejection (→ set DONE), None on API error (→ retry).
    """
    try:
        candles = get_candles(pair, count=500, granularity="M1")
    except Exception as e:
        log.warning(f"[{pair}] API error fetching candles: {e}")
        return None

    if not candles:
        return None

    df = pd.DataFrame(candles)
    df["date"]   = df["time"].dt.date
    df["hour"]   = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute

    ma = df["close"].rolling(MA_PERIOD).mean().iloc[-1]
    if pd.isna(ma):
        log.info(f"[{pair}] MA not ready (insufficient history)")
        return False

    cfg      = PAIR_SESSION[pair]
    today    = london_now().date()
    rng_bars = df[
        (df["date"]   == today) &
        (df["hour"]   == cfg["range_hour"]) &
        (df["minute"] >= cfg["range_min_start"]) &
        (df["minute"] <= cfg["range_min_end"])
    ]

    if len(rng_bars) < MIN_RANGE_BARS:
        log.info(f"[{pair}] Only {len(rng_bars)} range bars — need {MIN_RANGE_BARS}, retrying")
        return None   # not enough bars yet — retry next poll, not a permanent rejection

    pip = PAIR_PIP_SIZE[pair]
    rh  = rng_bars["high"].max()
    rl  = rng_bars["low"].min()
    rs  = rh - rl

    min_pip = cfg["min_range_pips"]
    max_pip = cfg["max_range_pips"]
    if rs < min_pip * pip:
        log.info(f"[{pair}] Range too small ({rs/pip:.1f} pips < {min_pip})")
        return False
    if rs > max_pip * pip:
        log.info(f"[{pair}] Range too large ({rs/pip:.1f} pips > {max_pip})")
        return False

    with _lock:
        st.cached_range_high = rh
        st.cached_range_low  = rl
        st.cached_range_size = rs
        st.cached_ma         = ma
        st.range_cached      = True

    # Regime detection — uses same 500-candle dataset; called once per day
    regime   = compute_regime_live(candles)
    orb_mult = REGIME_ORB_MULT.get(regime, 0.5)
    with _lock:
        st.regime          = regime
        st.orb_risk_mult   = orb_mult
        st.regime_computed = True
    log.info(f"[{pair}] Range cached: high={rh}  low={rl}  size={rs/pip:.1f}pips  MA={ma:.5f}")
    log.info(f"[{pair}] Regime: {regime}  ORB mult: {orb_mult:.2f}")
    return True

def get_latest_bar(pair):
    """
    Cheap 2-candle fetch — called every poll to check body filter + latest close.
    Returns (open, close) or None on error.
    """
    try:
        candles = get_candles(pair, count=2, granularity="M1")
        if not candles:
            return None
        last = candles[-1]
        return last["open"], last["close"]
    except Exception as e:
        log.warning(f"[{pair}] API error fetching latest bar: {e}")
        return None

def compute_units(balance, entry, sl, risk_pct=RISK_PER_TRADE, pair=None):
    """
    Risk risk_pct of balance.  Returns integer units (minimum 1) — units_to_lots()
    later converts to MT5 lots via lots = units / 100_000 (forex convention).

    For USD-account trading XXX/USD (GBPUSD, EURUSD): loss per unit = dist USD,
        so units = balance * risk_pct / dist.
    For USD-account trading XXX/JPY (GBPJPY etc.): dist is in JPY per unit;
        convert to USD by multiplying by USDJPY.
    For XAUUSD: gold contract size is broker-specific (typically 100 oz/lot);
        loss per lot = dist * contract_size USD.  Units = lots * 100_000.
    For US500/NAS100/US30 (index CFDs): NOT forex 100k-unit lots. Size from the
        symbol's tick economics — value_per_point = tick_value / tick_size — so
        lots = balance*risk / (dist * value_per_point); units = lots * 100_000 so
        the shared units_to_lots(/100_000) recovers the correct lots. Refuse to
        size (return 0) if the broker doesn't report tick value/size.
    """
    dist = abs(entry - sl)
    if dist == 0:
        return 0
    if pair in ("US500", "NAS100", "US30"):
        info = mt5.symbol_info(_sym(pair))
        tv = float(getattr(info, "trade_tick_value", 0) or 0) if info else 0.0
        ts = float(getattr(info, "trade_tick_size",  0) or 0) if info else 0.0
        if tv <= 0 or ts <= 0:
            log.error(f"[{pair}] tick value/size unavailable — refusing to size index blind")
            return 0
        value_per_point = tv / ts                      # $ per 1.0 price point per 1.0 lot
        lots = (balance * risk_pct) / (dist * value_per_point)
        return max(1, int(lots * 100_000))             # units_to_lots(/100k) -> lots
    if pair in ("GBPJPY", "EURJPY", "USDJPY"):
        with _lock:
            usdjpy_cache = _price_cache.get("USDJPY")
        usdjpy = usdjpy_cache["mid"] if usdjpy_cache else None
        if usdjpy is None or usdjpy <= 0:
            # Cache cold — pull a live tick directly. Better to do one extra RPC
            # than size with a stale 150.0 (current rate is ~158, ~5% sizing error).
            tick = mt5.symbol_info_tick(_sym("USDJPY"))
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                usdjpy = (tick.bid + tick.ask) / 2
            else:
                log.error(f"[{pair}] USDJPY price unavailable — refusing to size JPY pair")
                return 0
        return max(1, int((balance * risk_pct * usdjpy) / dist))
    # review#17 follow-up — USDXXX pairs: 1 unit moves $1 notional but the
    # loss is in XXX (CAD/CHF), so units = balance × risk × USDXXX_price / dist.
    # The pair's own entry price IS USDXXX, so no extra rate fetch needed.
    # Was previously falling through to the bare `balance × risk / dist`
    # branch, undersizing USD_CAD by ~25% and USD_CHF by ~10% vs backtest.
    if pair in ("USDCAD", "USDCHF"):
        return max(1, int((balance * risk_pct * entry) / dist))
    if pair == "XAUUSD":
        info = mt5.symbol_info(_sym(pair))
        contract = float(info.trade_contract_size) if info and info.trade_contract_size else 100.0
        # lots = (balance * risk_pct) / (dist * contract);  units = lots * 100_000
        return max(1, int((balance * risk_pct * 100_000) / (dist * contract)))
    return max(1, int((balance * risk_pct) / dist))

def _build_open_book() -> list:
    """
    Snapshot currently-open positions for the risk manager's currency-exposure
    cap. risk_usd per position is approximated from (units, entry, sl) so we
    don't need to re-load the original sizing intent; for JPY/XAU pairs the
    JPY/oz conversion is folded in via compute_units's inverse.
    """
    book = []
    try:
        positions = mt5.positions_get() or []
    except Exception as e:
        log.debug(f"_build_open_book mt5 read failed: {e}")
        return book
    for pos in positions:
        # Count the WHOLE book — ORB, Dow overlay, and trend sleeve — so the
        # risk gate's aggregate exposure cap sees correlated stacking across
        # sleeves (e.g. Monday: Dow long + ORB long + trend long on US500).
        if getattr(pos, "magic", 0) not in (ORB_MAGIC, DOW_MAGIC, TREND_MAGIC):
            continue
        pair = _base(pos.symbol)
        try:
            dist = abs(pos.price_open - pos.sl) if pos.sl else 0.0
            risk_usd = abs(pos.volume * 100_000 * dist)
            if pair in ("GBPJPY", "EURJPY", "USDJPY"):
                with _lock:
                    cache = _price_cache.get("USDJPY")
                usdjpy = cache["mid"] if cache else 150.0
                risk_usd = risk_usd / max(usdjpy, 1.0)
            elif pair == "XAUUSD":
                info = mt5.symbol_info(_sym(pair))
                contract = float(info.trade_contract_size) if info and info.trade_contract_size else 100.0
                risk_usd = abs(pos.volume * contract * dist)
            elif pair in ("US500", "NAS100", "US30"):
                info = mt5.symbol_info(_sym(pair))
                tv = float(getattr(info, "trade_tick_value", 0) or 0) if info else 0.0
                ts = float(getattr(info, "trade_tick_size",  0) or 0) if info else 0.0
                vpp = (tv / ts) if (tv > 0 and ts > 0) else 0.0
                risk_usd = abs(pos.volume * vpp * dist)
            if risk_usd == 0.0 and getattr(pos, "magic", 0) == TREND_MAGIC:
                # Trend-sleeve positions carry no hard SL (dist=0) — proxy their
                # at-risk amount as a -2.5% adverse day on position notional,
                # which is how the sleeve is sized. Without this the exposure
                # cap would see the position but count it as zero risk.
                info = mt5.symbol_info(pos.symbol)
                tv = float(getattr(info, "trade_tick_value", 0) or 0) if info else 0.0
                ts = float(getattr(info, "trade_tick_size",  0) or 0) if info else 0.0
                vpp = (tv / ts) if (tv > 0 and ts > 0) else 0.0
                risk_usd = abs(pos.volume * vpp * pos.price_open * 0.025)
        except Exception:
            risk_usd = 0.0
        book.append({'pair': pair, 'risk_usd': risk_usd})
    return book


def _pre_trade_risk_gate(strategy_name: str, pair: str, proposed_risk_pct: float,
                         balance: float, equity: float,
                         starting_balance: float) -> dict:
    """
    Phase 6 — single risk gate consulted before sizing/placing each order.
    Returns the risk_manager.risk_check() dict; caller checks 'allowed' and
    uses 'risk_pct' for compute_units. Defaults to allowing the trade with
    the proposed risk on any internal error (so a risk-module bug never
    silently kills the trader).
    """
    try:
        from agent import risk_manager as _rm
        realized_dd = max(0.0, (starting_balance - equity) / max(starting_balance, 1.0))
        hrp = _rm.hrp_weight_for(pair)
        return _rm.risk_check(
            strategy_name        = strategy_name,
            pair                 = pair,
            proposed_risk        = proposed_risk_pct,
            open_book            = _build_open_book(),
            equity               = equity,
            realized_dd_pct      = realized_dd,
            daily_loss_limit_pct = PROP_DAILY_LOSS_LIMIT,
            hrp_weight           = hrp,
        )
    except Exception as e:
        log.debug(f"_pre_trade_risk_gate failed ({e}) — defaulting allow")
        return {'allowed': True, 'risk_pct': proposed_risk_pct, 'reason': 'gate-error',
                'multipliers': {}, 'exposure_pct': {}}


def dynamic_risk(consecutive_wins: int, consecutive_losses: int) -> float:
    """
    Scale risk based on recent streak.
      Win streak  → increase risk up to RISK_MAX
      Loss streak → decrease risk down to RISK_MIN
      No streak   → RISK_BASE
    """
    if consecutive_wins > 0:
        return min(RISK_MAX, RISK_BASE + consecutive_wins * RISK_WIN_STEP)
    if consecutive_losses > 0:
        return max(RISK_MIN, RISK_BASE - consecutive_losses * RISK_LOSS_STEP)
    return RISK_BASE


# =============================================================================
# PER-PAIR KELLY SIZING — overlays dynamic_risk() with a pair-specific
# performance multiplier based on rolling live results. Allocates more to
# pairs that are working, less to pairs that are cold.
# =============================================================================

KELLY_MIN_TRADES   = 30     # need at least this many closed trades per pair
KELLY_LOOKBACK     = 60     # use the most recent N closed trades per pair
KELLY_FRACTION     = 0.25   # fractional Kelly — full Kelly is too aggressive
KELLY_MULT_FLOOR   = 0.4    # never scale below 40% of base
KELLY_MULT_CEILING = 1.5    # never scale above 150% of base

# Cache: { pair -> (last_computed_ts, multiplier) }. Recompute at most once
# every 5 min so we don't re-read the CSV on every poll.
_kelly_cache: dict      = {}
_KELLY_CACHE_TTL_SECS   = 300


def _read_pair_trades_for_kelly(pair: str, n: int) -> list:
    """Return up to `n` most recent {pnl, sl, entry, exit} dicts for `pair` from the trade log."""
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    try:
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("pair") != pair:
                    continue
                pnl_s = r.get("pnl", "")
                if not pnl_s:
                    continue
                try:
                    pnl   = float(pnl_s)
                    entry = float(r.get("entry") or 0)
                    sl    = float(r.get("sl") or 0)
                    exitp = float(r.get("exit_price") or 0)
                except (ValueError, TypeError):
                    continue
                rows.append({"pnl": pnl, "entry": entry, "sl": sl, "exit": exitp})
    except Exception as e:
        log.debug(f"[{pair}] kelly trade-log read failed: {e}")
        return []
    return rows[-n:]


def kelly_pair_multiplier(pair: str) -> float:
    """
    Per-pair Bayesian-Kelly multiplier on top of base risk.

    Replaces the prior point-estimate Kelly with a posterior-integrated
    fraction. The posterior over (win_rate, mean_win, mean_loss) is
    constructed from observed trades and a weak prior; the Kelly
    fraction is integrated over that posterior so small-sample
    overconfidence is automatically dampened.

    Mathematical model:
      win_rate ~ Beta(α, β)              with α = 1 + n_wins, β = 1 + n_losses
      avg_RR   ~ NormalInverseGamma      with conjugate update on
                                          (mean, variance) of (win_pnl, |loss_pnl|)
      f*       = E_posterior[ WR - (1 - WR) / avg_RR ]
                 ≈ mean of M Monte-Carlo Kelly fractions

    Effects vs. point-estimate Kelly:
      * 30-trade samples no longer push the multiplier to its ceiling
      * Pairs with high WR but few trades get a sensibly-discounted size
      * Sign-flip robustness: kelly < 0 on small N is treated with
        uncertainty, not panic — multiplier glides toward 0.5× rather
        than slamming to KELLY_MULT_FLOOR

    Returns a scalar in [KELLY_MULT_FLOOR, KELLY_MULT_CEILING]. Returns
    1.0 until the pair has KELLY_MIN_TRADES closed trades.

    Cached for KELLY_CACHE_TTL_SECS to avoid re-reading the CSV on every poll.
    """
    now_ts = time.time()
    cached = _kelly_cache.get(pair)
    if cached and (now_ts - cached[0]) < _KELLY_CACHE_TTL_SECS:
        return cached[1]

    trades = _read_pair_trades_for_kelly(pair, KELLY_LOOKBACK)
    if len(trades) < KELLY_MIN_TRADES:
        _kelly_cache[pair] = (now_ts, 1.0)
        return 1.0

    wins   = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]
    n_w, n_l = len(wins), len(losses)

    if n_w == 0 or n_l == 0:
        _kelly_cache[pair] = (now_ts, 1.0)
        return 1.0

    # ── Posterior over win_rate: Beta(1 + n_w, 1 + n_l) ──
    # Posterior over win/loss magnitudes: bootstrap (small-sample-safe NIG proxy).
    try:
        import numpy as _np
        rng = _np.random.default_rng(seed=abs(hash(pair)) & 0xFFFF_FFFF)
        M   = 4000

        # Beta posterior samples
        wr_samples = rng.beta(1.0 + n_w, 1.0 + n_l, size=M)

        # Bootstrap for win and loss magnitude — approximates NIG posterior
        # without requiring scipy.stats.invgamma. With ≥KELLY_MIN_TRADES
        # observations this approaches the true posterior closely.
        wins_arr = _np.asarray(wins, dtype=float)
        loss_arr = _np.asarray(losses, dtype=float)
        win_means  = rng.choice(wins_arr, size=(M, n_w)).mean(axis=1)
        loss_means = rng.choice(loss_arr, size=(M, n_l)).mean(axis=1)
        loss_means = _np.where(loss_means > 1e-9, loss_means, 1e-9)
        rr_samples = win_means / loss_means

        kelly_samples = wr_samples - (1.0 - wr_samples) / rr_samples
        # Conservative posterior summary: 25th percentile of f*.
        # Using the lower quartile rather than the mean enforces
        # automatic shrinkage when the posterior is wide.
        f_post = float(_np.percentile(kelly_samples, 25))
        f_mean = float(_np.mean(kelly_samples))
    except Exception as e:
        log.debug(f"[{pair}] Bayesian Kelly fallback to point estimate: {e}")
        wr      = n_w / (n_w + n_l)
        avg_rr  = (sum(wins) / n_w) / max(sum(losses) / n_l, 1e-9)
        f_post  = wr - (1.0 - wr) / avg_rr
        f_mean  = f_post

    if f_post <= 0:
        # Posterior 25th percentile says edge is uncertain or negative.
        # Halve risk rather than slamming the floor so we keep learning.
        mult = max(KELLY_MULT_FLOOR, 0.5)
    else:
        mult = (KELLY_FRACTION * f_post) / max(RISK_BASE, 1e-9)
        mult = max(KELLY_MULT_FLOOR, min(KELLY_MULT_CEILING, mult))

    _kelly_cache[pair] = (now_ts, mult)
    log.info(
        f"[{pair}] BayesKelly: f*_p25={f_post:+.3f} f*_mean={f_mean:+.3f} "
        f"-> mult={mult:.2f} (n_w={n_w} n_l={n_l})"
    )
    return mult


# =============================================================================
# VOLATILITY-TARGETED RISK MULTIPLIER
# Scales risk inversely to current realized vol vs its 30-day baseline. When
# vol spikes (news / liquidation cascade), automatically size down. When vol
# is subdued, can take incrementally larger positions on a fixed-edge trade.
# =============================================================================

VOLTARGET_LOOKBACK_BARS  = 30          # bars used for current vol estimate
VOLTARGET_BASELINE_BARS  = 30 * 24 * 60 # roughly 30 days of M1 bars
VOLTARGET_FLOOR          = 0.5         # never go below 50% of base
VOLTARGET_CEILING        = 1.3         # never exceed 130% of base


def vol_target_multiplier(pair: str) -> float:
    """
    Return a multiplier on risk_pct based on current vol vs 30-day baseline.

    Uses Yang-Zhang-like estimator on the most recent M1 candles. The principle:
    a strategy designed and backtested on "normal" vol will be over-leveraged
    by ~2x in a vol regime that's 2x baseline. Scale down to keep dollar risk
    constant in vol terms, not price terms.

    Returns 1.0 (no-op) if there isn't enough history yet, or if MT5 fetch fails.
    """
    try:
        candles = get_candles(pair, count=VOLTARGET_BASELINE_BARS, granularity="M1")
    except Exception as e:
        log.debug(f"[{pair}] vol-target candle fetch failed: {e}")
        return 1.0
    if not candles or len(candles) < VOLTARGET_LOOKBACK_BARS * 2:
        return 1.0

    import math
    closes  = [c["close"] for c in candles if c.get("close")]
    if len(closes) < VOLTARGET_LOOKBACK_BARS * 2:
        return 1.0

    def _close_to_close_sd(values: list) -> float:
        if len(values) < 5:
            return 0.0
        rets = []
        prev = values[0]
        for v in values[1:]:
            if prev > 0 and v > 0:
                rets.append(math.log(v / prev))
            prev = v
        if len(rets) < 5:
            return 0.0
        mean = sum(rets) / len(rets)
        var  = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
        return math.sqrt(var)

    cur_vol      = _close_to_close_sd(closes[-VOLTARGET_LOOKBACK_BARS:])
    baseline_vol = _close_to_close_sd(closes)
    if baseline_vol <= 0 or cur_vol <= 0:
        return 1.0

    # Multiplier inversely proportional to current/baseline vol ratio.
    ratio = cur_vol / baseline_vol
    mult  = 1.0 / max(ratio, 1e-9)
    mult  = max(VOLTARGET_FLOOR, min(VOLTARGET_CEILING, mult))
    log.info(f"[{pair}] Vol-target: cur_sd={cur_vol:.6f} base_sd={baseline_vol:.6f} "
             f"ratio={ratio:.2f} -> mult={mult:.2f}")
    return mult


# =============================================================================
# BAYESIAN ONLINE CHANGE-POINT DETECTION (BOCPD, Adams & MacKay 2007)
# Maintains a posterior over the "run length" — number of bars since the last
# regime break — given recent trade returns. When P(change-point) exceeds a
# threshold, we set a leading-indicator flag that halves risk *before* the
# lagging WR-based kill switch fires. Catches drawdowns 5-15 trades earlier.
#
# Model:
#   * Likelihood: trade returns ~ N(μ, σ²) within a run. Conjugate
#     Normal-Inverse-Gamma prior so the predictive distribution is a t.
#   * Hazard: constant 1/λ — i.e. expected run length = λ trades.
#   * Update: r_t+1 = r_t + 1 with prob (1-H), or 0 with prob H, weighted
#     by predictive probability of the new observation.
# =============================================================================

BOCPD_HAZARD_LAMBDA   = 80      # expected run length in trades (~2 weeks of activity)
BOCPD_MAX_RUN         = 200     # truncate posterior at this run length
BOCPD_ALERT_THRESHOLD = 0.50    # P(run_length=0) above this triggers warning
BOCPD_RISK_HALF_THR   = 0.30    # above this we halve risk preemptively
BOCPD_PRIOR_MU        = 0.0
BOCPD_PRIOR_KAPPA     = 1.0     # prior strength (small = weak prior)
BOCPD_PRIOR_ALPHA     = 1.5
BOCPD_PRIOR_BETA      = 50.0    # in (pnl-units squared) — calibrate to typical PnL var


class _BOCPDState:
    """Sufficient statistics for the run-length posterior. All vectors are
    indexed by hypothesised run length r = 0, 1, …, BOCPD_MAX_RUN."""

    def __init__(self):
        import numpy as _np
        self.np = _np
        self.run_log_post = _np.array([0.0])  # log P(r=0) = 0 initially
        self.mu     = _np.array([BOCPD_PRIOR_MU])
        self.kappa  = _np.array([BOCPD_PRIOR_KAPPA])
        self.alpha  = _np.array([BOCPD_PRIOR_ALPHA])
        self.beta   = _np.array([BOCPD_PRIOR_BETA])

    def _log_t_pdf(self, x: float):
        """Vectorised log-pdf of the predictive Student-t at all run lengths."""
        import math as _math
        np = self.np
        df    = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        z     = (x - self.mu) / np.maximum(scale, 1e-9)
        # log of standard t: lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi)
        # - log(scale) - (df+1)/2 * log(1 + z²/df)
        lg_num = np.array([_math.lgamma((d + 1.0) / 2.0) for d in df])
        lg_den = np.array([_math.lgamma(d / 2.0)         for d in df])
        return (
            lg_num - lg_den
            - 0.5 * np.log(df * _math.pi)
            - np.log(np.maximum(scale, 1e-9))
            - 0.5 * (df + 1.0) * np.log1p((z * z) / df)
        )

    def update(self, x: float):
        """Push one observation; return P(run_length = 0) (i.e. P(change-point))."""
        import numpy as np
        H = 1.0 / BOCPD_HAZARD_LAMBDA

        log_pred = self._log_t_pdf(float(x))
        # Numerically stable log-space update of the run-length posterior.
        # P(growth)     ∝ P(r_t) · π(x|r_t) · (1-H)
        # P(changepoint)∝ Σ_r P(r_t) · π(x|r_t) · H
        log_growth = self.run_log_post + log_pred + np.log(1.0 - H)
        log_cp     = np.logaddexp.reduce(self.run_log_post + log_pred) + np.log(H)

        new_log = np.empty(len(self.run_log_post) + 1)
        new_log[0]  = log_cp
        new_log[1:] = log_growth

        # Truncate to BOCPD_MAX_RUN and normalise.
        if len(new_log) > BOCPD_MAX_RUN:
            new_log = new_log[:BOCPD_MAX_RUN]
        norm = np.logaddexp.reduce(new_log)
        new_log -= norm

        # Conjugate updates of NIG sufficient statistics for each r > 0.
        new_mu     = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        new_kappa  = self.kappa + 1.0
        new_alpha  = self.alpha + 0.5
        new_beta   = self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))

        # r=0 (changepoint) re-seeds with the prior.
        self.mu     = np.concatenate(([BOCPD_PRIOR_MU],    new_mu))
        self.kappa  = np.concatenate(([BOCPD_PRIOR_KAPPA], new_kappa))
        self.alpha  = np.concatenate(([BOCPD_PRIOR_ALPHA], new_alpha))
        self.beta   = np.concatenate(([BOCPD_PRIOR_BETA],  new_beta))

        if len(self.mu) > BOCPD_MAX_RUN:
            self.mu     = self.mu[:BOCPD_MAX_RUN]
            self.kappa  = self.kappa[:BOCPD_MAX_RUN]
            self.alpha  = self.alpha[:BOCPD_MAX_RUN]
            self.beta   = self.beta[:BOCPD_MAX_RUN]

        self.run_log_post = new_log
        # Posterior probability of a change-point right now.
        return float(np.exp(self.run_log_post[0]))


_bocpd_state    = _BOCPDState()
_bocpd_p_cp     = 0.0
_bocpd_alerted  = False


def update_bocpd(trade_pnl: float) -> float:
    """
    Push one trade's PnL into the BOCPD posterior. Returns the posterior
    probability of a regime break right now. Caller decides what to do
    with it (telegram alert, halve risk, hard kill).
    """
    global _bocpd_state, _bocpd_p_cp, _bocpd_alerted
    try:
        p = _bocpd_state.update(float(trade_pnl))
    except Exception as e:
        log.debug(f"BOCPD update failed: {e}")
        return _bocpd_p_cp
    _bocpd_p_cp = p
    if p >= BOCPD_ALERT_THRESHOLD and not _bocpd_alerted:
        _bocpd_alerted = True
        log.warning(f"BOCPD: P(change-point)={p:.2f} — leading regime-break signal")
        try:
            send_telegram(
                f"BOCPD ALERT: P(change-point)={p:.2f}\n"
                f"Regime break detected ahead of WR kill-switch. Risk halved."
            )
        except Exception:
            pass
    elif p < 0.10 and _bocpd_alerted:
        _bocpd_alerted = False
        log.info("BOCPD: alert cleared (P(change-point) back below 0.10)")
    return p


def bocpd_risk_factor() -> float:
    """Return a multiplicative risk factor [0.5 … 1.0] based on current
    change-point posterior. Used as a pre-emptive guard, layered before
    the existing rolling-WR kill switch."""
    if _bocpd_p_cp >= BOCPD_RISK_HALF_THR:
        return 0.5
    return 1.0


def compute_regime_live(candles):
    """
    Classify market regime from a list of OHLC candle dicts.
    Uses Wilder's EMA ADX(14) and ATR_ratio (current ATR / 30-bar rolling mean).
    Returns one of: 'TRENDING' | 'RANGING' | 'TRANSITIONING' | 'VOLATILE' | 'UNDEFINED'.
    Called once per day from fetch_and_cache_range() after range is confirmed valid.
    """
    min_bars = REGIME_ADX_PERIOD + REGIME_ATR_REF_BARS + 10
    if not candles or len(candles) < min_bars:
        return 'UNDEFINED'

    df        = pd.DataFrame(candles)
    hi        = df['high']
    lo        = df['low']
    cl        = df['close']

    tr  = pd.concat([
        (hi - lo),
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up  = hi.diff()
    dn  = lo.diff().mul(-1)
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)

    alpha = 1.0 / REGIME_ADX_PERIOD
    atr   = tr.ewm(alpha=alpha, min_periods=REGIME_ADX_PERIOD, adjust=False).mean()
    safe  = atr.replace(0, float('nan'))
    pdi   = 100 * pdm.ewm(alpha=alpha, min_periods=REGIME_ADX_PERIOD, adjust=False).mean() / safe
    ndi   = 100 * ndm.ewm(alpha=alpha, min_periods=REGIME_ADX_PERIOD, adjust=False).mean() / safe
    dsum  = (pdi + ndi).replace(0, float('nan'))
    dx    = ((pdi - ndi).abs() / dsum * 100).fillna(0)
    adx   = dx.ewm(alpha=alpha, min_periods=REGIME_ADX_PERIOD, adjust=False).mean()

    atr_mean  = atr.rolling(REGIME_ATR_REF_BARS).mean()
    atr_ratio = (atr / atr_mean).where(atr_mean > 0)

    adx_val       = adx.iloc[-1]
    atr_ratio_val = atr_ratio.iloc[-1]

    if pd.isna(adx_val) or pd.isna(atr_ratio_val):
        return 'UNDEFINED'
    if atr_ratio_val > REGIME_ATR_VOLATILE:
        return 'VOLATILE'
    if adx_val > REGIME_ADX_TREND:
        return 'TRENDING'
    if adx_val < REGIME_ADX_RANGE:
        return 'RANGING'
    return 'TRANSITIONING'


# =============================================================================
# PER-PAIR STATE
# =============================================================================

class PairState:
    """
    Tracks the lifecycle of one pair for a single trading day.

    Phases:
      IDLE         — waiting for 08:30 or range not valid
      ORDER_PLACED — pending stop order is live on MT5
      IN_TRADE     — stop order filled; stream thread manages BE/profit lock
      DONE         — nothing more to do today
    """
    def __init__(self, pair):
        self.pair = pair
        self.reset()

    def reset(self):
        self.phase           = "IDLE"
        self.order_id        = None
        self.trade_id        = None
        self.direction       = None
        self.entry_price     = None
        self.stop_loss       = None
        self.take_profit     = None
        self.range_size      = None
        self.breakeven_set   = False
        self.profit_lock_set = False
        self.partial_tp_set  = False   # True once 50% closed at partial_tp_r
        self.partial_lots    = None    # lots of the partial close
        # Per-session TP/profit lock — loaded from PAIR_SESSION at order placement
        cfg = PAIR_SESSION[self.pair]
        self.tp_multiplier       = cfg["tp_multiplier"]
        self.partial_tp_r        = cfg.get("partial_tp_r", 0.5)
        self.profit_lock_trigger = cfg["profit_lock_trigger"]
        self.profit_lock_sl_pct  = cfg["profit_lock_sl_pct"]
        # Cached range — computed once at 08:30, reused every poll
        self.range_cached    = False
        self.cached_range_high = None
        self.cached_range_low  = None
        self.cached_range_size = None
        self.cached_ma         = None
        self.exec_logged       = False
        # Regime — computed alongside range; gates ORB risk
        self.regime          = 'UNDEFINED'
        self.orb_risk_mult   = 1.0
        self.regime_computed = False
        log.info(f"[{self.pair}] State reset")


# =============================================================================
# DOW DISPERSION STATE
# =============================================================================

class DowState:
    """
    Tracks the day's Dow-dispersion position for one index.
      IDLE     — before the cash open / not a Mon-Thu day
      IN_TRADE — market position open; exits at the session close
      DONE     — nothing more to do today
    """
    def __init__(self, pair):
        self.pair = pair
        self.reset()

    def reset(self):
        self.phase       = "IDLE"
        self.trade_id    = None
        self.direction   = None
        self.entry_price = None
        self.stop_loss   = None
        self.take_profit = None
        log.info(f"[DOW_{self.pair}] State reset")


# =============================================================================
# TRADE LOG (ORB)
# =============================================================================

LOG_FILE    = "live_trade_log.csv"
LOG_HEADERS = ["date", "strategy", "pair", "direction", "entry", "exit_price", "pnl",
               "sl", "tp", "exit_reason", "range_size_pips",
               "breakeven_set", "profit_lock_set"]

def append_trade_log(pair, direction, entry, exit_price, pnl,
                     sl, tp, exit_reason, range_size, be, profit_locked,
                     strategy_name='ORB'):
    pip = PAIR_PIP_SIZE[pair]
    dec = PAIR_DECIMALS[pair]
    row = {
        "date":            london_now().strftime("%Y-%m-%d"),
        "strategy":        strategy_name,
        "pair":            pair,
        "direction":       direction,
        "entry":           f"{entry:.{dec}f}"      if entry      else "",
        "exit_price":      f"{exit_price:.{dec}f}" if exit_price else "",
        "pnl":             f"{pnl:.2f}"            if pnl is not None else "",
        "sl":              f"{sl:.{dec}f}"         if sl         else "",
        "tp":              f"{tp:.{dec}f}"         if tp         else "",
        "exit_reason":     exit_reason,
        "range_size_pips": f"{range_size/pip:.1f}" if range_size else "",
        "breakeven_set":   be,
        "profit_lock_set": profit_locked,
    }
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if write_header:
            w.writeheader()
        w.writerow(row)

# =============================================================================
# EXECUTION QUALITY LOG — tracks slippage and spread on every fill
# =============================================================================

EXEC_LOG_FILE    = "execution_quality.csv"
EXEC_LOG_HEADERS = ["date", "time", "strategy", "pair", "direction", "expected_fill",
                    "actual_fill", "slippage_pips", "spread_pips",
                    "latency_ms_median"]

def append_exec_log(pair, direction, expected_fill, actual_fill, strategy_name='ORB'):
    pip = PAIR_PIP_SIZE[pair]
    dec = PAIR_DECIMALS[pair]
    # SIGNED slippage: positive = adverse (worse than trigger), negative = favourable.
    # For BUY_STOP (long), adverse means filled higher than trigger.
    # For SELL_STOP (short), adverse means filled lower than trigger.
    if direction == "long":
        slippage = (actual_fill - expected_fill) / pip
    else:
        slippage = (expected_fill - actual_fill) / pip
    with _lock:
        cached = _price_cache.get(pair)
    spread = ((cached["ask"] - cached["bid"]) / pip) if cached else 0.0
    # Snapshot rolling order-send latency at fill time so post-trade analysis
    # can correlate slippage with broker round-trip degradation.
    if _order_latencies:
        ordered = sorted(_order_latencies)
        latency_med = ordered[len(ordered) // 2]
    else:
        latency_med = 0.0
    now = london_now()
    row = {
        "date":              now.strftime("%Y-%m-%d"),
        "time":              now.strftime("%H:%M:%S"),
        "strategy":          strategy_name,
        "pair":              pair,
        "direction":         direction,
        "expected_fill":     f"{expected_fill:.{dec}f}",
        "actual_fill":       f"{actual_fill:.{dec}f}",
        "slippage_pips":     f"{slippage:+.1f}",   # signed; +ve = adverse
        "spread_pips":       f"{spread:.1f}",
        "latency_ms_median": f"{latency_med:.0f}",
    }
    write_header = not os.path.exists(EXEC_LOG_FILE)
    with open(EXEC_LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXEC_LOG_HEADERS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    log.info(f"[{pair}] Execution quality: expected={expected_fill:.{dec}f}  "
             f"actual={actual_fill:.{dec}f}  slip={slippage:+.1f}pip "
             f"({'adverse' if slippage > 0 else 'favourable'})  spread={spread:.1f}pip")

# =============================================================================
# PAYOUT READINESS — Blue Guardian Instant: 20% consistency + 5×0.5% days
# =============================================================================

_daily_pnl_cache = {"ts": 0.0, "by_date": {}}


def _closed_pnl_by_date() -> dict:
    """Closed P&L summed per date from live_trade_log.csv (60s TTL cache)."""
    now_ts = time.time()
    if now_ts - _daily_pnl_cache["ts"] < 60:
        return _daily_pnl_cache["by_date"]
    by_date = {}
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                for row in csv.DictReader(f):
                    if not row.get("pnl"):
                        continue
                    try:
                        by_date[row["date"]] = by_date.get(row["date"], 0.0) + float(row["pnl"])
                    except (ValueError, TypeError):
                        continue
    except Exception as e:
        log.debug(f"closed-pnl scan failed: {e}")
    _daily_pnl_cache["ts"] = now_ts
    _daily_pnl_cache["by_date"] = by_date
    return by_date


def _todays_closed_pnl() -> float:
    return _closed_pnl_by_date().get(london_now().strftime("%Y-%m-%d"), 0.0)


def payout_readiness() -> dict:
    """
    Firm payout requirements since the last payout (prop_state.json:
    "last_payout_date" — edit it after each payout).
      FTMO_1STEP:    total > 0; best day <= 50% of POSITIVE days' profit
                     (Best Day rule); >= 14 days since last payout.
      BLUE_GUARDIAN: total > 0; best day < 20% of period TOTAL; >= 5
                     qualifying days (>= 0.5% closed); >= 14 days.
    """
    since = _load_prop_state().get("last_payout_date", "")
    by_date = {d: p for d, p in _closed_pnl_by_date().items() if d > since}
    total = sum(by_date.values())
    best_day = max(by_date.values()) if by_date else 0.0
    positive_sum = sum(p for p in by_date.values() if p > 0)
    qual_days = sum(1 for p in by_date.values()
                    if p >= 0.005 * INITIAL_DEPOSIT)
    try:
        import datetime as _dt
        days_since = ((london_now().date() - _dt.date.fromisoformat(since)).days
                      if since else 999)
    except Exception:
        days_since = 999
    if PROFILE["consistency_mode"] == "best_le_50pct_positive":
        best_share = (best_day / positive_sum) if positive_sum > 0 else 0.0
        consistent = best_share <= 0.50
    else:
        best_share = (best_day / total) if total > 0 else 0.0
        consistent = best_share < 0.20
    ready = (total > 0 and consistent and
             qual_days >= PROFILE["qual_days_needed"] and days_since >= 14)
    return {"total": total, "best_day": best_day, "best_share": best_share,
            "qual_days": qual_days, "days_since": days_since, "ready": ready}


# =============================================================================
# DAILY SUMMARY — logged once at 21:00 after all sessions close
# =============================================================================

def log_daily_summary(starting_balance, recent_trades):
    """
    Reads today's trades from live_trade_log.csv and logs a comprehensive
    end-of-day summary. Called once at 21:00 London time.
    """
    today_str = london_now().strftime("%Y-%m-%d")

    today_trades = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date") == today_str:
                    today_trades.append(row)

    try:
        balance, equity = get_account_state()
    except Exception:
        balance = equity = None

    day_pnl = sum(float(t["pnl"]) for t in today_trades if t.get("pnl"))
    wins    = sum(1 for t in today_trades if t.get("pnl") and float(t["pnl"]) > 0)
    losses  = sum(1 for t in today_trades if t.get("pnl") and float(t["pnl"]) <= 0)
    n       = len(today_trades)
    wr      = (wins / n * 100) if n > 0 else 0

    dd     = starting_balance - equity if equity else 0
    dd_pct = (dd / starting_balance * 100) if starting_balance else 0

    roll_n   = len(recent_trades)
    roll_wr  = (sum(1 for t in recent_trades if t > 0) / roll_n * 100) if roll_n > 0 else 0
    roll_avg = (sum(recent_trades) / roll_n) if roll_n > 0 else 0

    log.info("=" * 60)
    log.info("  DAILY SUMMARY — %s", today_str)
    log.info("=" * 60)
    log.info("  Trades today:     %d  (W: %d  L: %d  WR: %.0f%%)", n, wins, losses, wr)

    by_pair = {}
    for t in today_trades:
        p   = t.get("pair", "?")
        pnl = float(t["pnl"]) if t.get("pnl") else 0
        by_pair.setdefault(p, []).append(pnl)
    for p, pnls in by_pair.items():
        pw = sum(1 for x in pnls if x > 0)
        log.info("    %-8s  trades: %d  W: %d  L: %d  PnL: %.2f",
                 p, len(pnls), pw, len(pnls) - pw, sum(pnls))

    # Per-strategy decomposition — survivor strategies appear as separate
    # blocks alongside the ORB block above. Older rows that pre-date the
    # strategy column are bucketed under 'ORB' so totals stay comparable.
    by_strat = {}
    for t in today_trades:
        sname = t.get("strategy") or "ORB"
        pnl   = float(t["pnl"]) if t.get("pnl") else 0
        by_strat.setdefault(sname, []).append(pnl)
    if len(by_strat) > 1 or (by_strat and next(iter(by_strat)) != "ORB"):
        log.info("  By strategy:")
        for sname, pnls in by_strat.items():
            sw = sum(1 for x in pnls if x > 0)
            log.info("    %-30s  trades: %d  W: %d  L: %d  PnL: %.2f",
                     sname, len(pnls), sw, len(pnls) - sw, sum(pnls))

    log.info("  Day PnL:          %.2f", day_pnl)
    if balance is not None:
        log.info("  Balance:          %.2f", balance)
        log.info("  Equity:           %.2f", equity)
    log.info("  Drawdown:         %.2f (%.2f%% of starting %.2f)",
             dd, dd_pct, starting_balance)
    log.info("  Rolling %d-trade (max 40): WR=%.0f%%  avg_pnl=%.2f", roll_n, roll_wr, roll_avg)

    # Payout readiness (firm-profile rules)
    _rule = ("<=50% of positive days" if PROFILE["consistency_mode"] ==
             "best_le_50pct_positive" else "<20% of total")
    try:
        pr = payout_readiness()
        qual_str = (f" | qualifying days {pr['qual_days']}/{PROFILE['qual_days_needed']}"
                    if PROFILE["qual_days_needed"] else "")
        log.info("  Payout readiness [%s]: total %+.2f | best day %+.2f (%.0f%%, "
                 "need %s)%s | %d days since payout%s",
                 FIRM, pr["total"], pr["best_day"], pr["best_share"] * 100,
                 _rule, qual_str, pr["days_since"],
                 "  >>> REQUEST PAYOUT NOW <<<" if pr["ready"] else "")
    except Exception as e:
        pr = None
        log.debug(f"payout readiness failed: {e}")
    log.info("=" * 60)
    bal_str = f"{balance:.2f}" if balance is not None else "N/A"
    eq_str  = f"{equity:.2f}"  if equity  is not None else "N/A"
    pr_str = ""
    if pr:
        pr_str = (f"\nPayout [{FIRM}]: total {pr['total']:+.2f} | best day "
                  f"{pr['best_share']:.0%} (need {_rule})"
                  + (f" | qual days {pr['qual_days']}/{PROFILE['qual_days_needed']}"
                     if PROFILE["qual_days_needed"] else "")
                  + ("\n*** REQUEST PAYOUT NOW (then set last_payout_date in "
                     "prop_state.json) ***" if pr["ready"] else ""))
    send_telegram(
        f"Daily Summary {today_str}\n"
        f"Trades: {n}  W: {wins}  L: {losses}  WR: {wr:.0f}%\n"
        f"Day PnL: {day_pnl:.2f}\n"
        f"Balance: {bal_str}  Equity: {eq_str}\n"
        f"Drawdown: {dd:.2f} ({dd_pct:.2f}%)\n"
        f"Rolling WR: {roll_wr:.0f}%  avg_pnl: {roll_avg:.2f}"
        + pr_str
    )
    signal(
        f"📊 <b>Daily Performance — {today_str}</b>\n"
        f"Trades: {n}  ✅ {wins}W / ❌ {losses}L  ({wr:.0f}% WR)\n"
        f"Day P&amp;L: <b>{day_pnl:+.2f}</b>\n"
        f"Rolling {roll_n} trades: WR {roll_wr:.0f}%  avg {roll_avg:+.2f}"
    )


# =============================================================================
# STOP MANAGEMENT — called on every price tick from the stream thread
# =============================================================================

def on_tick(st, mid):
    """
    Applies breakeven, partial TP, and profit lock on each price tick.
    Called from the stream thread — must hold _lock before calling.
    Returns (action, value) where value is new_sl for SL moves,
    or (trade_id, pair, lots) tuple for partial close, else (None, None).
    """
    if st.phase != "IN_TRADE":
        return None, None

    entry        = st.entry_price
    rng_s        = st.range_size
    # Partial trigger is multiples of R (entry-to-SL), not range. After a
    # broker SL clamp these can differ — using R keeps the "close half at NR"
    # semantics intact regardless of how wide the actual stop ended up.
    r_dist       = abs(entry - st.stop_loss) if st.stop_loss else rng_s
    partial_dist = st.partial_tp_r * r_dist

    if st.direction == "long":
        # Partial TP at 1R — close 50%; move SL to entry (breakeven on remainder)
        if not st.partial_tp_set and mid >= entry + partial_dist:
            st.partial_tp_set = True
            st.stop_loss      = entry
            st.breakeven_set  = True   # set now so subsequent ticks don't re-fire
            return "partial_tp", (st.trade_id, st.pair, st.partial_lots)
        if st.breakeven_set and not st.profit_lock_set:
            if mid >= entry + st.profit_lock_trigger * rng_s:
                new_sl = entry + st.profit_lock_sl_pct * rng_s
                st.stop_loss       = new_sl
                st.profit_lock_set = True
                return "profit_lock", new_sl

    else:  # short
        # Partial TP at 1R — close 50%; move SL to entry (breakeven on remainder)
        if not st.partial_tp_set and mid <= entry - partial_dist:
            st.partial_tp_set = True
            st.stop_loss      = entry
            st.breakeven_set  = True
            return "partial_tp", (st.trade_id, st.pair, st.partial_lots)
        if st.breakeven_set and not st.profit_lock_set:
            if mid <= entry - st.profit_lock_trigger * rng_s:
                new_sl = entry - st.profit_lock_sl_pct * rng_s
                st.stop_loss       = new_sl
                st.profit_lock_set = True
                return "profit_lock", new_sl

    return None, None


def make_stream_handler(states):
    """
    Returns a function that processes each price tick.
    Runs in the stream thread — reads state under lock, makes API calls outside lock.
    Handles ORB (PairState) breakeven / partial-TP / profit-lock management.
    """
    def handle_tick(pair, mid):
        # News-window guard: defer EA-initiated stop management (partial closes
        # count as "trading" under Blue Guardian's ±2-min high-impact news rule,
        # which removes profit on violation; our buffer is deliberately wider).
        # Broker-side SL/TP fills are passive and unaffected; emergency exits
        # (daily-limit / max-DD / session exit) run from the main thread and
        # are NOT gated — breaching drawdown outranks a news-window foul.
        ccy = PAIR_SESSION.get(pair, {}).get("news_currency")
        if ccy:
            times = _news_times_today.get(ccy, [])
            if times and is_near_news(london_now(), times):
                return

        # ── ORB stop management ──
        with _lock:
            st = states.get(pair)
            if st is not None and st.phase == "IN_TRADE":
                action, value = on_tick(st, mid)
                trade_id      = st.trade_id
            else:
                action, value, trade_id = None, None, None

        if action == "partial_tp" and value is not None:
            tid, pr, lots = value
            if lots and lots >= 0.01:
                xp, xpnl = close_partial(tid, pr, lots, "partial_tp")
                if xpnl is not None:
                    dec = PAIR_DECIMALS[pr]
                    send_telegram(f"[{pr}] Partial TP closed {lots} lots @ {xp:.{dec}f}  pnl={xpnl:.2f}")
                    signal(
                        f"🎯 <b>Partial TP Hit — {pr}</b>\n"
                        f"Closed 50% @ <b>{xp:.{dec}f}</b>\n"
                        f"Stop Loss moved to breakeven\n"
                        f"Remainder running to full target"
                    )
            # BE was already set in on_tick — just push the SL update to MT5
            if trade_id:
                with _lock:
                    st = states.get(pair)
                    ep = st.entry_price if st else None
                if ep is not None:
                    modify_trade_sl(trade_id, ep, pair)
        elif action == "breakeven" and trade_id is not None:
            modify_trade_sl(trade_id, value, pair)
            signal(f"🔒 <b>Breakeven — {pair}</b>\nStop Loss moved to entry. Risk-free trade.")
        elif action == "profit_lock" and trade_id is not None:
            modify_trade_sl(trade_id, value, pair)
            dec = PAIR_DECIMALS[pair]
            signal(f"🔐 <b>Profit Locked — {pair}</b>\nStop Loss secured in profit @ <b>{value:.{dec}f}</b>")

    return handle_tick

def stream_with_handler(states, stop_event):
    """
    Stream thread entry point — polls MT5 ticks every 100 ms.
    Updates the shared price cache and applies BE/profit-lock on every tick.
    MT5 has no push streaming; 100 ms polling gives sub-pip latency for stop management.
    """
    handler = make_stream_handler(states)
    consec_none = {sym: 0 for sym in STREAM_PAIRS}
    paged       = {sym: False for sym in STREAM_PAIRS}

    while not stop_event.is_set():
        # Pause MT5 RPCs while a reconnect is in progress.
        if _mt5_reconnecting.is_set():
            time.sleep(0.5)
            continue
        try:
            for symbol in STREAM_PAIRS:
                tick = mt5.symbol_info_tick(_sym(symbol))
                if tick is None:
                    consec_none[symbol] += 1
                    # 50 consecutive Nones at 100ms = ~5 seconds of dead feed.
                    # Page operator and let connection watchdog kick in.
                    if consec_none[symbol] >= 50 and not paged[symbol]:
                        paged[symbol] = True
                        log.error(f"[{symbol}] STREAM DEAD — {consec_none[symbol]} consecutive None ticks")
                        send_telegram(
                            f"🚨 [{symbol}] Tick stream dead "
                            f"({consec_none[symbol]} consecutive None) — check MT5 connection"
                        )
                    continue
                if consec_none[symbol] > 0:
                    if paged[symbol]:
                        log.info(f"[{symbol}] Stream recovered after {consec_none[symbol]} None ticks")
                        send_telegram(f"✅ [{symbol}] Tick stream recovered")
                    consec_none[symbol] = 0
                    paged[symbol]       = False
                bid = tick.bid
                ask = tick.ask
                mid = (bid + ask) / 2
                with _lock:
                    _price_cache[symbol] = {"bid": bid, "ask": ask, "mid": mid}
                    _record_spread_sample(symbol, ask - bid)
                handler(symbol, mid)
            time.sleep(0.1)
        except Exception as e:
            if not stop_event.is_set():
                log.error(f"Stream error: {e} — retrying in 5s")
                time.sleep(5)

def transaction_stream(states, stop_event):
    """
    Fill-detection thread — polls MT5 every 1 s for order fills.
    When a pending order disappears and a matching position appears, transitions
    the state to IN_TRADE and corrects SL/TP to actual fill price.
    Mirrors the instant-fill behaviour of the OANDA transaction stream.
    """
    _stop_types = {mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP}

    while not stop_event.is_set():
        # Pause MT5 RPCs while a reconnect is in progress.
        if _mt5_reconnecting.is_set():
            time.sleep(0.5)
            continue
        try:
            time.sleep(1.0)

            # ── ORB fill detection ────────────────────────────────────────
            # Snapshot the dict under lock to avoid "dictionary changed size
            # during iteration" if the main thread mutates `states` mid-loop.
            with _lock:
                state_snapshot = list(states.items())
            for symbol, st in state_snapshot:
                with _lock:
                    if st.phase != "ORDER_PLACED":
                        continue
                    order_id  = st.order_id
                    direction = st.direction
                    rs        = st.range_size
                    tp_mult   = st.tp_multiplier
                    orig_sl   = st.stop_loss
                    trigger   = (st.cached_range_high if direction == "long"
                                 else st.cached_range_low)

                broker_sym = _sym(symbol)
                orders = mt5.orders_get(symbol=broker_sym)
                if orders and any(str(o.ticket) == str(order_id) for o in orders):
                    continue  # still pending

                positions = mt5.positions_get(symbol=broker_sym)
                # Only consider OUR ORB positions — the Dow overlay (DOW_MAGIC) and
                # trend sleeve (TREND_MAGIC) hold positions on the same index symbols;
                # matching by direction alone would adopt their (older) ticket and
                # rewrite its SL/TP to ORB levels. Mirrors get_open_trade's filter.
                if positions:
                    positions = [p for p in positions
                                 if getattr(p, "magic", 0) == ORB_MAGIC]
                if not positions:
                    continue  # order cancelled/expired (or only other-sleeve positions)

                pos = next(
                    (p for p in positions
                     if ("long" if p.type == mt5.POSITION_TYPE_BUY else "short") == direction),
                    None
                )
                if pos is None:
                    continue  # no position matching expected direction

                fill_price = pos.price_open
                trade_id   = str(pos.ticket)

                if rs is None:
                    continue

                # Fill-price sanity gate: warn loudly if the fill is wildly worse than
                # the trigger. Stop orders normally fill within a few pips of the
                # trigger; >25% of the range size implies a stale tick, news spike,
                # or a broker quoting issue we want a human to see.
                if trigger:
                    slip = abs(fill_price - trigger)
                    if slip > 0.25 * rs:
                        dec_warn = PAIR_DECIMALS[symbol]
                        log.warning(
                            f"[{symbol}] FILL-PRICE ANOMALY — trigger={trigger:.{dec_warn}f} "
                            f"fill={fill_price:.{dec_warn}f} slip={slip:.{dec_warn}f} "
                            f"(>{0.25*100:.0f}% of range {rs:.{dec_warn}f}) — "
                            f"investigate news/liquidity"
                        )
                        send_telegram(
                            f"⚠️ [{symbol}] Fill anomaly: slipped {slip:.{dec_warn}f} "
                            f"({slip/rs*100:.0f}% of range)"
                        )

                original_risk = abs(trigger - orig_sl) if orig_sl and trigger else rs
                correct_tp = (fill_price + tp_mult * rs if direction == "long"
                              else fill_price - tp_mult * rs)
                correct_sl = (fill_price - original_risk if direction == "long"
                              else fill_price + original_risk)

                modify_trade_tp(trade_id, correct_tp, symbol)
                modify_trade_sl(trade_id, correct_sl, symbol)

                should_log = False
                with _lock:
                    if not st.exec_logged:
                        st.exec_logged = True
                        should_log = True
                if should_log and trigger:
                    append_exec_log(symbol, direction, trigger, fill_price)

                # Live spread sanity at fill — flag if current spread is
                # >3x rolling median (news spike, thin liquidity, fat-finger
                # quote). The order is already filled; this is for alerting
                # and post-trade analysis, not refusal.
                fill_tick = mt5.symbol_info_tick(broker_sym)
                if fill_tick and fill_tick.ask > 0 and fill_tick.bid > 0:
                    cur_spread = fill_tick.ask - fill_tick.bid
                    median_spread = _rolling_spread_median(symbol)
                    if median_spread and cur_spread > 3 * median_spread:
                        log.warning(
                            f"[{symbol}] FILL-TIME SPREAD SPIKE — current "
                            f"{cur_spread:.6f} > 3x median {median_spread:.6f}"
                        )
                        send_telegram(
                            f"⚠️ [{symbol}] Spread spike at fill: "
                            f"{cur_spread:.5f} (median {median_spread:.5f})"
                        )

                total_lots   = units_to_lots(abs(pos.volume * 100_000))
                partial_lots = max(0.01, round(total_lots * 0.5, 2)) if total_lots >= 0.02 else None

                with _lock:
                    st.phase        = "IN_TRADE"
                    st.trade_id     = trade_id
                    st.entry_price  = fill_price
                    st.stop_loss    = correct_sl
                    st.take_profit  = correct_tp
                    st.partial_lots = partial_lots

                dec = PAIR_DECIMALS[symbol]
                log.info(f"[{symbol}] ORB fill via fill-thread — "
                         f"{direction.upper()} at {fill_price:.{dec}f}  trade={trade_id}")
                arrow = "🟢" if direction == "long" else "🔴"
                send_telegram(
                    f"[ORB {symbol}] {direction.upper()} filled @ {fill_price:.{dec}f}\n"
                    f"SL={correct_sl:.{dec}f}  TP={correct_tp:.{dec}f}  lots={total_lots}"
                )
                signal(
                    f"{arrow} <b>ORB Setup — {symbol}</b>\n"
                    f"Direction: <b>{'LONG' if direction == 'long' else 'SHORT'}</b>\n"
                    f"Entry: <b>{fill_price:.{dec}f}</b>\n"
                    f"Stop Loss: <b>{correct_sl:.{dec}f}</b>\n"
                    f"Take Profit: <b>{correct_tp:.{dec}f}</b>\n"
                    f"Session: {'London' if PAIR_SESSION[symbol]['range_hour'] == 8 else 'New York'}"
                )


        except Exception as e:
            if not stop_event.is_set():
                log.error(f"Fill-detection error: {e} — retrying in 5s")
                time.sleep(5)

# =============================================================================
# MAIN THREAD — time-based events + order fill detection
# =============================================================================

def run_pair(pair, st, states, now, balance, day_blocked, news_times,
             consecutive_wins=0, consecutive_losses=0):
    """
    Called every POLL_SECS from the main thread.
    Handles: session exit, daily limit, order placement, fill detection,
             and proactive cancellation of pending orders near news events.
    Stop management (BE/profit lock) is handled by the stream thread via on_tick().
    """
    h, m   = now.hour, now.minute
    cfg    = PAIR_SESSION[pair]

    with _lock:
        phase = st.phase

    if phase == "DONE":
        return None

    # ── SESSION EXIT ──────────────────────────────────────────────────────────
    if h >= cfg["exit_hour"]:
        trade_id = order_id = None
        with _lock:
            cur = st.phase
            if cur == "IN_TRADE":
                trade_id  = st.trade_id
                direction = st.direction
                sl        = st.stop_loss
                tp        = st.take_profit
                rs        = st.range_size
                be        = st.breakeven_set
                pl        = st.profit_lock_set
                ep        = st.entry_price
            elif cur == "ORDER_PLACED":
                order_id = st.order_id
            st.phase = "DONE"

        if cur == "IN_TRADE" and trade_id:
            xp, xpnl = close_trade_market(trade_id, pair, "session_exit")
            append_trade_log(pair, direction, ep, xp, xpnl, sl, tp, "session_exit", rs, be, pl)
            return xpnl
        elif cur == "ORDER_PLACED" and order_id:
            cancel_pending(order_id, pair)
        return None

    # ── DAILY LOSS LIMIT ──────────────────────────────────────────────────────
    if day_blocked:
        trade_id = order_id = None
        with _lock:
            cur = st.phase
            if cur == "IN_TRADE":
                trade_id  = st.trade_id
                direction = st.direction
                sl        = st.stop_loss
                tp        = st.take_profit
                rs        = st.range_size
                be        = st.breakeven_set
                pl        = st.profit_lock_set
                ep        = st.entry_price
            elif cur == "ORDER_PLACED":
                order_id = st.order_id
            st.phase = "DONE"

        if cur == "IN_TRADE" and trade_id:
            xp, xpnl = close_trade_market(trade_id, pair, "daily_limit")
            append_trade_log(pair, direction, ep, xp, xpnl, sl, tp, "daily_limit", rs, be, pl)
            return xpnl
        elif cur == "ORDER_PLACED" and order_id:
            cancel_pending(order_id, pair)
        return None

    # ── IDLE — try to place order ─────────────────────────────────────────────
    with _lock:
        cur_phase    = st.phase
        range_cached = st.range_cached

    if cur_phase == "IDLE":
        # Phase 1 — TCA-driven kill switch. If the agent has marked this
        # strategy KILL based on live-vs-backtest decay, skip outright.
        # 'orb' is the legacy hardcoded family name; once the registry-driven
        # path is wired in, this will key off the active strategy's name.
        try:
            from agent.db import get_live_kill
            _kill = get_live_kill('orb_ny') or get_live_kill('orb')
        except Exception:
            _kill = None
        if _kill and _kill.get('verdict') == 'KILL':
            log.info(f"[{pair}] TCA kill-switch active (decay={_kill.get('decay')}) — skipping ORB today")
            with _lock:
                st.phase = "DONE"
            return
        _live_size_mult = 0.5 if (_kill and _kill.get('verdict') == 'REDUCE') else 1.0

        after_range   = (h > cfg["after_range_hour"] or
                         (h == cfg["after_range_hour"] and m >= cfg["after_range_min"]))
        within_window = h < cfg["entry_window_end_hour"]

        if not after_range or not within_window:
            return
        pair_news = news_times.get(cfg["news_currency"], [])
        if is_near_news(now, pair_news):
            log.info(f"[{pair}] Near news — skipping this poll")
            return

        if not range_cached:
            result = fetch_and_cache_range(pair, st)
            if result is None:
                return           # API error — retry next poll
            if result is False:
                with _lock:
                    st.phase = "DONE"
                return

        with _lock:
            rh = st.cached_range_high
            rl = st.cached_range_low
            rs = st.cached_range_size
            ma = st.cached_ma

        pip = PAIR_PIP_SIZE[pair]

        # Regime gate — skip ORB if regime multiplier is 0
        with _lock:
            # If regime hasn't been computed yet, halve risk (defensive default).
            # We don't know the regime, so don't take full size.
            regime_mult = st.orb_risk_mult if st.regime_computed else 0.5
            regime_str  = st.regime
        if regime_mult == 0.0:
            log.info(f"[{pair}] Regime {regime_str} — ORB skipped, no entry today")
            with _lock:
                st.phase = "DONE"
            return

        bar = get_latest_bar(pair)
        if bar is None:
            return
        last_open, last_close = bar
        last_body = abs(last_close - last_open)
        if last_body < BREAKOUT_BODY_MIN_PCT * rs:
            log.info(f"[{pair}] Body filter failed — skipping this poll")
            return

        tp_mult = cfg["tp_multiplier"]
        if last_close > ma:
            direction = "long"
            trigger   = rh
            sl_price  = rl - pip * 0.2
            entry_est = rh + pip
            tp_price  = entry_est + tp_mult * rs
        elif last_close < ma:
            direction = "short"
            trigger   = rl
            sl_price  = rh + pip * 0.2
            entry_est = rl - pip
            tp_price  = entry_est - tp_mult * rs
        else:
            log.info(f"[{pair}] Price exactly on MA — done today")
            with _lock:
                st.phase = "DONE"
            return

        with _lock:
            cached_price = _price_cache.get(pair)
        if cached_price:
            # BUY_STOP invalid if ask >= trigger; SELL_STOP invalid if bid <= trigger
            if direction == "long" and cached_price["ask"] >= trigger:
                log.info(f"[{pair}] Ask already at/past trigger — done today")
                with _lock:
                    st.phase = "DONE"
                return
            if direction == "short" and cached_price["bid"] <= trigger:
                log.info(f"[{pair}] Bid already at/past trigger — done today")
                with _lock:
                    st.phase = "DONE"
                return
            if spread_gate(pair):
                log.info(f"[{pair}] spread_gate: elevated spread — skipping entry")
                return

        # Correlation guard — symmetric: any other group member already active OR
        # about to be placed in this same poll halves risk on this pair.
        group = next((g for g in CORRELATION_GROUPS if pair in g), None)
        peers_active = False
        active_peer = None
        if group:
            with _lock:
                for peer in group:
                    if peer == pair:
                        continue
                    if peer in states and states[peer].phase in ("IN_TRADE", "ORDER_PLACED"):
                        peers_active = True
                        active_peer = peer
                        break
        base_risk  = dynamic_risk(consecutive_wins, consecutive_losses) * regime_mult
        # Per-pair Kelly overlay — boosts hot pairs, throttles cold ones.
        # Returns 1.0 (no-op) until the pair has KELLY_MIN_TRADES history.
        kelly_mult = kelly_pair_multiplier(pair)
        # Vol-target overlay — scales risk inversely with current vs baseline
        # realized vol so dollar-risk stays constant across vol regimes.
        vt_mult    = vol_target_multiplier(pair)
        # BOCPD leading-indicator: 1.0 normally, 0.5 when posterior P(change-point)
        # exceeds BOCPD_RISK_HALF_THR. Layered in front of the lagging WR kill switch.
        bocpd_mult = bocpd_risk_factor()
        base_risk  = base_risk * kelly_mult * vt_mult * bocpd_mult
        risk_pct   = base_risk * 0.5 if peers_active else base_risk
        if peers_active:
            log.info(f"[{pair}] Correlation guard: {active_peer} active in same group "
                     f"— halving risk to {risk_pct:.3f}")

        # Min SL distance vs live spread — if SL is closer to trigger than
        # 1.5x current spread, the spread alone will trip the stop. Push SL out.
        spread = None
        if cached_price:
            spread = cached_price["ask"] - cached_price["bid"]
        else:
            tk = mt5.symbol_info_tick(_sym(pair))
            if tk and tk.ask > 0 and tk.bid > 0:
                spread = tk.ask - tk.bid
        if spread and spread > 0:
            min_sl_dist = 1.5 * spread
            if direction == "long":
                cur_dist = trigger - sl_price
                if cur_dist < min_sl_dist:
                    sl_price = round(trigger - min_sl_dist, PAIR_DECIMALS[pair])
                    log.info(f"[{pair}] SL widened to 1.5x spread "
                             f"({cur_dist:.5f}->{min_sl_dist:.5f})")
            else:
                cur_dist = sl_price - trigger
                if cur_dist < min_sl_dist:
                    sl_price = round(trigger + min_sl_dist, PAIR_DECIMALS[pair])
                    log.info(f"[{pair}] SL widened to 1.5x spread "
                             f"({cur_dist:.5f}->{min_sl_dist:.5f})")

        # Phase 6 — central risk gate. Combines:
        #   - Phase 1 TCA verdict (was _live_size_mult)
        #   - HRP weight across active live strategies
        #   - Drawdown-adjusted sizing (halve once 50% of daily loss eaten)
        #   - Aggregate currency-exposure cap
        #   - Per-strategy consecutive-loss circuit breaker
        try:
            equity_now = get_account_state()[1]
        except Exception:
            equity_now = balance
        gate = _pre_trade_risk_gate(
            strategy_name      = f'orb_{pair.lower()}',
            pair               = pair,
            proposed_risk_pct  = risk_pct * _live_size_mult,
            balance            = balance,
            equity             = equity_now,
            starting_balance   = _STARTING_BALANCE if _STARTING_BALANCE > 0 else balance,
        )
        if not gate['allowed']:
            log.info(f"[{pair}] RISK GATE blocked entry: {gate['reason']}")
            with _lock:
                st.phase = "DONE"
            return
        units = compute_units(balance, entry_est, sl_price, gate['risk_pct'], pair=pair)
        if units == 0:
            with _lock:
                st.phase = "DONE"
            return

        existing = get_pending_order(pair, order_type="STOP")
        if existing:
            log.warning(f"[{pair}] Pending order already exists on MT5 (id={existing['id']}) "
                        f"— adopting it, not placing a second")
            with _lock:
                st.phase       = "ORDER_PLACED"
                st.order_id    = existing["id"]
                st.direction   = direction
                st.stop_loss   = sl_price
                st.take_profit = tp_price
                st.range_size  = rs
            return

        window_end = now.replace(hour=cfg["entry_window_end_hour"],
                                 minute=0, second=0, microsecond=0)
        # Don't place a stop order that would outlive its session entry window.
        # If window_end is too close (<2 min) for MT5 to reliably accept the
        # specified-expiration order, just skip — we shouldn't extend the
        # expiration past the configured window.
        seconds_to_window_end = (window_end - now).total_seconds()
        if seconds_to_window_end < 120:
            log.info(f"[{pair}] Entry window closing in {seconds_to_window_end:.0f}s "
                     f"— skipping order placement")
            with _lock:
                st.phase = "DONE"
            return
        expire_lon = window_end
        # MT5 expiration must be a Unix timestamp integer
        expire_utc = int(expire_lon.astimezone(pytz.utc).timestamp())

        oid = place_stop_order(pair, direction, units, trigger,
                               sl_price, tp_price, expire_utc)
        if oid:
            with _lock:
                st.phase       = "ORDER_PLACED"
                st.order_id    = oid
                st.direction   = direction
                st.stop_loss   = sl_price
                st.take_profit = tp_price
                st.range_size  = rs
            news_clear = not is_near_news(now, news_times.get(cfg["news_currency"], []))
            post_trade_reasoning(pair, direction, rh, rl, rs, ma, last_close, news_clear)
        else:
            with _lock:
                st.phase = "DONE"
            log.info(f"[{pair}] Order placement failed — done today")
        return

    # ── ORDER_PLACED — poll for fill ─────────────────────────────────────────
    if cur_phase == "ORDER_PLACED":
        pair_news = news_times.get(cfg["news_currency"], [])
        if is_near_news(now, pair_news):
            order_id = None
            with _lock:
                if st.phase == "ORDER_PLACED":
                    order_id = st.order_id
                    st.phase = "DONE"
            if order_id:
                cancel_pending(order_id, pair)
                log.info(f"[{pair}] Pending order cancelled — entering news window")
            return

        # Fill-detection thread handles fills every 1 s — this is a 30 s safety-net fallback
        with _lock:
            already_filled = st.phase == "IN_TRADE"
        if already_filled:
            return

        order = get_pending_order(pair, order_type="STOP")
        if order is not None:
            return  # still pending

        with _lock:
            if st.phase != "ORDER_PLACED":
                return  # fill-thread beat us to it
            rs            = st.range_size
            direction     = st.direction
            trigger       = st.cached_range_high if direction == "long" else st.cached_range_low
            original_sl   = st.stop_loss
            tp_multiplier = st.tp_multiplier
            entry_est     = st.cached_range_high if direction == "long" else st.cached_range_low

        trade = get_open_trade(pair, direction=direction)
        if trade:
            fill_price = float(trade["price"])
            dec        = PAIR_DECIMALS[pair]
            log.info(f"[{pair}] Fill detected via poll fallback at {fill_price:.{dec}f}")

            should_log_exec = False
            if trigger:
                with _lock:
                    if not st.exec_logged:
                        st.exec_logged = True
                        should_log_exec = True
            if should_log_exec:
                append_exec_log(pair, direction, trigger, fill_price)
            original_risk = abs(entry_est - original_sl) if original_sl else rs

            correct_tp = (fill_price + tp_multiplier * rs
                          if direction == "long"
                          else fill_price - tp_multiplier * rs)
            correct_sl = (fill_price - original_risk
                          if direction == "long"
                          else fill_price + original_risk)

            modify_trade_tp(trade["id"], correct_tp, pair)
            modify_trade_sl(trade["id"], correct_sl, pair)

            raw_pos = trade.get("_raw")
            total_lots   = raw_pos.volume if raw_pos else None
            partial_lots = (max(0.01, round(total_lots * 0.5, 2))
                            if total_lots and total_lots >= 0.02 else None)

            with _lock:
                st.phase        = "IN_TRADE"
                st.trade_id     = trade["id"]
                st.entry_price  = fill_price
                st.stop_loss    = correct_sl
                st.take_profit  = correct_tp
                st.partial_lots = partial_lots
        else:
            log.info(f"[{pair}] Order expired/cancelled — done today")
            with _lock:
                st.phase = "DONE"
        return

    # ── IN_TRADE — stop management handled by stream thread ──────────────────
    if cur_phase == "IN_TRADE":
        with _lock:
            trade_id  = st.trade_id
            direction = st.direction
            ep        = st.entry_price
            sl        = st.stop_loss
            tp        = st.take_profit
            rs        = st.range_size
            be        = st.breakeven_set
            pl        = st.profit_lock_set

        trade = get_open_trade(pair, trade_id=trade_id)
        trade_closed = (trade is None)
        if trade_closed:
            other = get_open_trade(pair)
            if other is not None:
                log.info(f"[{pair}] Our trade {trade_id} closed; {other['id']} still open (other strategy)")
            else:
                log.info(f"[{pair}] Trade closed by MT5 (SL or TP hit)")
            xp, xpnl = get_closed_trade_details(trade_id)
            append_trade_log(pair, direction, ep, xp, xpnl, sl, tp, "sl_or_tp", rs, be, pl)
            with _lock:
                st.phase = "DONE"
            if xp is not None and ep is not None:
                dec    = PAIR_DECIMALS[pair]
                pip    = PAIR_PIP_SIZE[pair]
                pips   = round((xp - ep) / pip, 1) if direction == "long" else round((ep - xp) / pip, 1)
                result = "✅ WIN" if (xpnl or 0) > 0 else "❌ LOSS"
                send_telegram(f"[ORB {pair}] {result}  pnl={xpnl:.2f}  exit={xp:.{dec}f}")
                signal(
                    f"{result} <b>ORB Closed — {pair}</b>\n"
                    f"Direction: {'LONG' if direction == 'long' else 'SHORT'}\n"
                    f"Entry: <b>{ep:.{dec}f}</b>  Exit: <b>{xp:.{dec}f}</b>\n"
                    f"Result: <b>{pips:+.1f} pips</b>"
                )
                post_trade_breakdown(pair, direction, ep, xp, xpnl, be, pl, rs)
            return xpnl

    return None


# =============================================================================
# MARKET ORDER (used by the Dow-dispersion overlay)
# =============================================================================

def place_market_order(pair, direction, units, sl, tp, comment="MKT", magic=ORB_MAGIC):
    """
    Place an MT5 market order with SL and TP.
    Returns trade ID string (position ticket) or None on failure.
    Used by the Dow overlay (and any market-entry strategy). The Dow overlay
    passes magic=DOW_MAGIC so its positions are never confused with concurrent
    ORB positions on the same index.
    """
    dec        = PAIR_DECIMALS[pair]
    broker_sym = _sym(pair)
    lots       = units_to_lots(units)
    order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
    tick       = mt5.symbol_info_tick(broker_sym)
    if tick is None:
        log.error(f"[{comment}_{pair}] symbol_info_tick returned None")
        return None
    # Reject stale ticks — same guard as place_stop_order.
    tick_age = time.time() - tick.time
    if tick_age > 2.0:
        log.warning(f"[{comment}_{pair}] Stale tick ({tick_age:.1f}s old) — skipping market order")
        return None
    price = tick.ask if direction == "long" else tick.bid
    step, vmin = _broker_volume_step(pair)
    req_vol = max(vmin, _round_to_step(lots, step, vmin))
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       broker_sym,
        "volume":       req_vol,
        "type":         order_type,
        "price":        round(price, dec),
        "sl":           round(sl, dec),
        "tp":           round(tp, dec),
        "deviation":    20,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": _market_filling_mode(pair),
        "magic":        magic,
        "comment":      comment,
    }
    try:
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        _record_order_latency(latency_ms)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"retcode={result.retcode if result else 'None'} "
                               f"comment={result.comment if result else ''}")
        # Brief wait for position to appear in MT5
        time.sleep(0.4)
        positions = mt5.positions_get(symbol=broker_sym)
        if positions:
            pos = positions[-1]  # most recent
            log.info(f"[{comment}_{pair}] Market {direction.upper()} filled @ {pos.price_open:.{dec}f}  "
                     f"SL={sl:.{dec}f}  TP={tp:.{dec}f}  lots={lots}  id={pos.ticket}  "
                     f"latency={latency_ms:.0f}ms")
            return str(pos.ticket)
        log.warning(f"[{comment}_{pair}] Order sent but no position found")
        return None
    except Exception as e:
        log.error(f"[{comment}_{pair}] Failed to place market order: {e}")
        return None


# =============================================================================
# DOW DISPERSION — runner (Edge 2): long Monday / short Thursday, intraday
# =============================================================================

def _dow_position_by_ticket(pair, trade_id):
    """Return the DOW position dict for this ticket, or None if it's gone.
    Magic-agnostic lookup by ticket so it never collides with ORB positions."""
    try:
        positions = mt5.positions_get(symbol=_sym(pair)) or []
    except Exception:
        return None
    return next((p for p in positions if str(p.ticket) == str(trade_id)), None)


def run_dow(pair, dow_st, now, balance, day_blocked, news_times):
    """
    Dow-dispersion overlay — called every POLL_SECS from the main thread.
    Long on Monday, short on Thursday, entered just after the US cash open and
    exited at the cash close. Beta-free weekly pattern; protective stop only.
    Uses DOW_MAGIC so its positions never collide with concurrent ORB trades.
    """
    if pair not in DOW_PAIRS:
        return None
    cfg = PAIR_SESSION[pair]
    h, m = now.hour, now.minute
    dec  = PAIR_DECIMALS[pair]

    with _lock:
        phase = dow_st.phase
    if phase == "DONE":
        return None

    # ── SESSION EXIT / daily block — flatten the held position ───────────────
    if h >= cfg["exit_hour"] or day_blocked:
        trade_id = None
        with _lock:
            cur = dow_st.phase
            if cur == "IN_TRADE":
                trade_id  = dow_st.trade_id
                direction = dow_st.direction
                ep        = dow_st.entry_price
                sl        = dow_st.stop_loss
                tp        = dow_st.take_profit
            dow_st.phase = "DONE"
        if cur == "IN_TRADE" and trade_id:
            reason = "daily_limit" if day_blocked else "session_exit"
            xp, xpnl = close_trade_market(trade_id, pair, f"dow_{reason}")
            append_trade_log(pair, direction, ep, xp, xpnl, sl, tp, reason,
                             None, False, False, strategy_name='DOW')
            return xpnl
        return None

    # ── IDLE — enter only on Mon (long) / Thu (short), just after the open ───
    if phase == "IDLE":
        wd = now.weekday()
        if wd == DOW_LONG_WEEKDAY:
            direction = "long"
        elif wd == DOW_SHORT_WEEKDAY:
            direction = "short"
        else:
            with _lock:
                dow_st.phase = "DONE"
            return None

        open_min = cfg["range_hour"] * 60 + cfg["range_min_start"]  # cash open
        now_min  = h * 60 + m
        if now_min < open_min:
            return None                       # before the open — wait
        if now_min > open_min + 15:           # missed the entry window
            with _lock:
                dow_st.phase = "DONE"
            return None
        if is_near_news(now, news_times.get(cfg["news_currency"], [])):
            return None
        if spread_gate(pair):
            return None

        with _lock:
            cached = _price_cache.get(pair)
        if not cached:
            return None
        entry = cached["ask"] if direction == "long" else cached["bid"]
        if direction == "long":
            sl_price = round(entry * (1 - DOW_STOP_PCT), dec)
            tp_price = round(entry * (1 + 0.03), dec)   # wide — runs to the close
        else:
            sl_price = round(entry * (1 + DOW_STOP_PCT), dec)
            tp_price = round(entry * (1 - 0.03), dec)

        # Central risk gate — same as ORB entries: HRP weight, drawdown-adjusted
        # sizing, aggregate exposure cap (which now sees Dow/trend positions too),
        # and the per-strategy circuit breaker. Base risk unchanged; the gate can
        # only reduce it.
        try:
            equity_now = get_account_state()[1]
        except Exception:
            equity_now = balance
        gate = _pre_trade_risk_gate(
            strategy_name      = f'dow_{pair.lower()}',
            pair               = pair,
            proposed_risk_pct  = RISK_BASE,
            balance            = balance,
            equity             = equity_now,
            starting_balance   = _STARTING_BALANCE if _STARTING_BALANCE > 0 else balance,
        )
        if not gate['allowed']:
            log.info(f"[DOW_{pair}] RISK GATE blocked entry: {gate['reason']}")
            with _lock:
                dow_st.phase = "DONE"
            return None
        units = compute_units(balance, entry, sl_price, gate['risk_pct'], pair=pair)
        if units == 0:
            with _lock:
                dow_st.phase = "DONE"
            return None

        trade_id = place_market_order(pair, direction, units, sl_price, tp_price,
                                      comment="DOW", magic=DOW_MAGIC)
        if trade_id:
            time.sleep(0.3)
            pos = _dow_position_by_ticket(pair, trade_id)
            actual_entry = float(pos.price_open) if pos else entry
            with _lock:
                dow_st.phase       = "IN_TRADE"
                dow_st.trade_id    = trade_id
                dow_st.direction   = direction
                dow_st.entry_price = actual_entry
                dow_st.stop_loss   = sl_price
                dow_st.take_profit = tp_price
            day_label = "Mon-long" if direction == "long" else "Thu-short"
            send_telegram(f"[DOW {pair}] {direction.upper()} @ {actual_entry:.{dec}f}  "
                          f"SL={sl_price:.{dec}f}  ({day_label} dispersion)")
        else:
            with _lock:
                dow_st.phase = "DONE"
        return None

    # ── IN_TRADE — detect protective-stop hit before the close ──────────────
    if phase == "IN_TRADE":
        with _lock:
            trade_id  = dow_st.trade_id
            direction = dow_st.direction
            ep        = dow_st.entry_price
            sl        = dow_st.stop_loss
            tp        = dow_st.take_profit
        if _dow_position_by_ticket(pair, trade_id) is None:
            xp, xpnl = get_closed_trade_details(trade_id)
            append_trade_log(pair, direction, ep, xp, xpnl, sl, tp, "sl_or_tp",
                             None, False, False, strategy_name='DOW')
            with _lock:
                dow_st.phase = "DONE"
            if xp is not None and ep is not None:
                result = "✅ WIN" if (xpnl or 0) > 0 else "❌ LOSS"
                send_telegram(f"[DOW {pair}] {result}  pnl={xpnl:.2f}  exit={xp:.{dec}f}")
            return xpnl
    return None


def main():
    log.info("=" * 60)
    log.info("  London + NY ORB Live Trader — MT5")
    log.info("=" * 60)

    # Connect to MT5 and resolve broker symbol suffixes BEFORE any thread spawns
    # or any state-dependent code runs. Both must succeed or we abort startup.
    connect_mt5()
    log.info("  MT5 connected")
    resolve_broker_symbols()
    log.info("  Broker symbols resolved")

    states = {pair: PairState(pair) for pair in PAIRS}

    # Dow-dispersion overlay — one DowState per eligible index (US500/NAS100)
    dow_states = {pair: DowState(pair) for pair in DOW_PAIRS}


    current_day          = None
    news_times           = dict(HIGH_IMPACT_FALLBACK)

    global _STARTING_BALANCE, _news_times_today
    while True:
        try:
            starting_balance = get_balance()
            break
        except Exception as e:
            log.warning(f"Startup balance fetch failed ({e}) — retrying in 10s")
            time.sleep(10)
    initial_balance   = starting_balance
    _STARTING_BALANCE = starting_balance   # exposed for _pre_trade_risk_gate
    log.info(f"  Starting balance: {starting_balance:.2f}")
    send_telegram(f"Trader started. Balance: {starting_balance:.2f}")

    # ── Prop-state init (survives restarts; shared with trend_sleeve.py) ──────
    # HWM drives the TRAILING 6% floor; the persisted day baseline means a
    # mid-day restart cannot weaken today's daily-loss floor.
    _pstate      = _load_prop_state()
    prop_hwm     = max(float(_pstate.get("hwm", 0.0)), INITIAL_DEPOSIT, starting_balance)
    prop_day     = _pstate.get("prop_day", "")
    day_baseline = float(_pstate.get("day_baseline", starting_balance))
    if prop_day != prop_day_today():
        prop_day     = prop_day_today()
        day_baseline = starting_balance
        _save_prop_state({"prop_day": prop_day, "day_baseline": day_baseline,
                          "hwm": prop_hwm})
    log.info(f"  Prop state: HWM={prop_hwm:.2f}  floor={trailing_floor(prop_hwm):.2f}  "
             f"day_baseline={day_baseline:.2f} ({prop_day})")

    consecutive_losses = 0
    consecutive_wins   = 0
    loss_pause_active  = False
    consistency_guard_fired = False

    from collections import deque
    recent_trades    = deque(maxlen=40)
    strategy_disabled = False

    stop_event    = threading.Event()
    stream_thread = threading.Thread(
        target=stream_with_handler,
        args=(states, stop_event),
        daemon=True,
        name="price-stream"
    )
    stream_thread.start()
    log.info("  Price stream (100 ms poll) started")

    txn_thread = threading.Thread(
        target=transaction_stream,
        args=(states, stop_event),
        daemon=True,
        name="fill-detect"
    )
    txn_thread.start()
    log.info("  Fill-detection thread started")

    # Startup reconciliation — reconnect to any trades/orders left from a previous run.
    # Adopt only positions tagged with our magic number; foreign EAs / manual trades
    # on the same account are left alone (we don't manage SL/TP/partials for them).
    try:
        open_positions = mt5.positions_get() or []
        for pos in open_positions:
            # Stray DOW positions are intraday and can't have their state
            # reconstructed across a restart — flatten them for a clean slate.
            if getattr(pos, "magic", 0) == DOW_MAGIC:
                log.warning(f"  Flattening stray DOW position {pos.symbol} id={pos.ticket} "
                            f"(intraday — cannot reconcile across restart)")
                try:
                    close_trade_market(str(pos.ticket), _base(pos.symbol), "dow_startup_flatten")
                except Exception as e:
                    log.error(f"  DOW startup flatten failed for {pos.ticket}: {e}")
                continue
            if getattr(pos, "magic", 0) != ORB_MAGIC:
                log.info(f"  Skipping foreign position {pos.symbol} id={pos.ticket} "
                         f"magic={getattr(pos, 'magic', 0)} (not ours)")
                continue
            # MT5 returns positions keyed by the broker's symbol (e.g. 'GBPUSD.r').
            # Map back to the internal base name so `states` lookups succeed.
            instrument = _base(pos.symbol)
            direction  = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
            ep         = pos.price_open
            trade_id   = str(pos.ticket)
            sl_price   = pos.sl if pos.sl else None
            tp_price   = pos.tp if pos.tp else None

            if instrument in states:
                cfg    = PAIR_SESSION[instrument]
                tp_mult = cfg["tp_multiplier"]
                # Derive range_size from TP distance — reliable even after SL has moved to BE
                if tp_price and tp_price != 0:
                    rs = abs(tp_price - ep) / tp_mult
                elif sl_price and sl_price != ep:
                    rs = abs(ep - sl_price)
                else:
                    rs = None

                # Determine whether BE/partial were already applied
                be_set = bool(sl_price and abs(sl_price - ep) < (rs * 0.05 if rs else 1e-8))
                partial_already = be_set  # if SL is at entry the partial already fired

                total_lots = pos.volume
                # If the partial already fired pre-restart, current volume is
                # the unrealised remainder — don't compute a second partial off
                # it. Leaving partial_lots None means on_tick will skip the
                # partial branch entirely.
                if partial_already:
                    partial_lots = None
                else:
                    partial_lots = (max(0.01, round(total_lots * 0.5, 2))
                                    if total_lots and total_lots >= 0.02 else None)

                with _lock:
                    st = states[instrument]
                    st.phase          = "IN_TRADE"
                    st.trade_id       = trade_id
                    st.direction      = direction
                    st.entry_price    = ep
                    st.stop_loss      = sl_price
                    st.range_size     = rs
                    st.take_profit    = tp_price
                    st.partial_lots   = partial_lots
                    st.breakeven_set  = be_set
                    st.partial_tp_set = partial_already
                log.warning(f"  RECONCILED ORB trade {instrument}: {direction.upper()} "
                            f"entry={ep}  trade_id={trade_id}  rs={rs}  "
                            f"be_set={be_set}  stop_management resumed")

            else:
                log.warning(f"  Unrecognised open position: {instrument} id={trade_id} — close manually")

        pending_orders = mt5.orders_get() or []
        for o in pending_orders:
            if getattr(o, "magic", 0) != ORB_MAGIC:
                continue  # foreign order — leave alone
            instrument = _base(o.symbol)
            is_stop = o.type in {mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP}

            if instrument in states and is_stop:
                with _lock:
                    st = states[instrument]
                    if st.phase == "IDLE":
                        st.phase    = "ORDER_PLACED"
                        st.order_id = str(o.ticket)
                log.warning(f"  RECONCILED ORB order {instrument}: id={o.ticket} phase=ORDER_PLACED")

            elif instrument not in states:
                log.warning(f"  Unrecognised pending order: {instrument} id={o.ticket} — cancel manually")

    except Exception as e:
        log.warning(f"  Reconciliation failed ({e}) — check MT5 manually for open positions")

    last_heartbeat         = 0
    summary_logged_today   = False
    daily_limit_alerted    = False
    briefings_posted       = set()   # tracks which pairs had briefing posted today
    tip_posted_today       = False
    monday_tip_posted      = False
    weekly_stats_posted    = False
    try:
        while True:
            now   = london_now()
            today = now.date()
            h     = now.hour

            # ── Heartbeat — log every 15 minutes ─────────────────────────
            ts = time.time()
            if ts - last_heartbeat >= 900:
                phases    = {p: states[p].phase for p in PAIRS}
                log.info(f"  Heartbeat: {now.strftime('%H:%M')}  ORB={phases}")
                last_heartbeat = ts

            # ── Daily reset ───────────────────────────────────────────────
            if today != current_day:
                current_day         = today
                summary_logged_today = False
                daily_limit_alerted  = False
                try:
                    initial_balance = get_balance()
                except Exception as e:
                    log.warning(f"Daily balance fetch failed ({e}) — using previous balance")
                news_times = fetch_todays_news_times()
                _news_times_today = dict(news_times)   # publish for the stream guard
                with _lock:
                    for st in states.values():
                        st.reset()
                    for ds in dow_states.values():
                        ds.reset()
                consecutive_losses  = 0
                consecutive_wins    = 0
                loss_pause_active   = False
                strategy_disabled   = False
                consistency_guard_fired = False
                briefings_posted    = set()
                tip_posted_today    = False
                weekly_stats_posted = False
                if now.weekday() == 0:
                    monday_tip_posted = False
                log.info(f"  -- New day: {today}  Balance: {initial_balance:.2f} --")

            # ── Weekend skip ──────────────────────────────────────────────
            if now.weekday() >= 5:
                time.sleep(3600)
                continue

            # ── Monday prop tip — posted once at 07:30 ───────────────────
            if now.weekday() == 0 and h == 7 and now.minute >= 30 and not monday_tip_posted:
                monday_tip_posted = True
                post_monday_prop_tip()

            # ── Daily educational tip — posted once at 07:45 ─────────────
            if h == 7 and now.minute >= 45 and not tip_posted_today:
                tip_posted_today = True
                post_daily_tip()

            # ── Pre-session briefings ─────────────────────────────────────
            # London pairs — briefing at 07:50
            for lp in ("GBPUSD", "GBPJPY", "EURJPY"):
                if h == 7 and now.minute >= 50 and lp not in briefings_posted:
                    briefings_posted.add(lp)
                    post_pre_session_briefing(lp, news_times)
            # NY pairs — briefing at 14:00 (FX/gold + the validated indices)
            for np_ in ("EURUSD", "USDJPY", "XAUUSD", "US500", "NAS100", "US30"):
                if h == 14 and now.minute >= 0 and np_ not in briefings_posted:
                    briefings_posted.add(np_)
                    post_pre_session_briefing(np_, news_times)

            # ── Daily summary — after all sessions close (21:00+) ─────────
            if h >= 21 and not summary_logged_today:
                summary_logged_today = True
                log_daily_summary(initial_balance, recent_trades)

            # ── Weekly stats — every Friday at 21:00 ─────────────────────
            if now.weekday() == 4 and h >= 21 and not weekly_stats_posted:
                weekly_stats_posted = True
                post_weekly_stats()
                post_live_vs_backtest()

            # ── Outside session hours ─────────────────────────────────────
            if h < 7 or h >= 22:
                time.sleep(120)
                continue

            # ── MT5 connection watchdog ───────────────────────────────────
            if not check_mt5_connection():
                log.warning("MT5 still disconnected — skipping this poll")
                time.sleep(POLL_SECS)
                continue

            # ── Balance + prop firm checks ────────────────────────────────
            try:
                balance, equity = get_account_state()
            except Exception as e:
                log.warning(f"Balance fetch failed ({e}) — skipping this poll")
                time.sleep(POLL_SECS)
                continue

            # High-water mark for the TRAILING 6% floor (BG trails max(balance,
            # equity); the floor RISES as the account grows, locking at breakeven
            # once +6%). Persist increases so restarts can't lower the floor.
            new_hwm = max(prop_hwm, balance, equity)
            if new_hwm > prop_hwm:
                prop_hwm = new_hwm
                _save_prop_state({"hwm": prop_hwm})

            # Firm day rollover — re-snapshot the daily baseline and persist,
            # so a mid-day restart reuses the original baseline. FTMO baselines
            # on BALANCE at midnight CE(S)T; Blue Guardian on max(balance,
            # equity) at 5 PM New York.
            if prop_day != prop_day_today():
                prop_day     = prop_day_today()
                day_baseline = (balance if PROFILE["day_baseline_mode"] == "balance"
                                else max(balance, equity))
                _save_prop_state({"prop_day": prop_day, "day_baseline": day_baseline})
                log.info(f"  Prop day rolled ({prop_day}): daily baseline {day_baseline:.2f}")

            # Trailing max-DD kill — flatten BEFORE the firm's line (safety margin).
            if equity <= trailing_floor(prop_hwm) + PROP_DD_SAFETY_MARGIN * INITIAL_DEPOSIT:
                log.warning("MAX DRAWDOWN LIMIT HIT -- closing all and stopping")
                for pair in PAIRS:
                    tid = oid = None
                    trade_info = None
                    with _lock:
                        st = states[pair]
                        if st.phase == "IN_TRADE":
                            tid = st.trade_id
                            trade_info = (st.direction, st.entry_price, st.stop_loss,
                                          st.take_profit, st.range_size,
                                          st.breakeven_set, st.profit_lock_set)
                        elif st.phase == "ORDER_PLACED":
                            oid = st.order_id
                        st.phase = "DONE"
                    if tid and trade_info:
                        xp, xpnl = close_trade_market(tid, pair, "max_drawdown")
                        _dir, _ep, _sl, _tp, _rs, _be, _pl2 = trade_info
                        append_trade_log(pair, _dir, _ep, xp, xpnl,
                                         _sl, _tp, "max_drawdown", _rs, _be, _pl2)
                    if oid:
                        cancel_pending(oid, pair)
                for pair, dow_st in dow_states.items():
                    with _lock:
                        if dow_st.phase == "IN_TRADE":
                            dtid  = dow_st.trade_id
                            d_dir = dow_st.direction
                            d_ep  = dow_st.entry_price
                            d_sl  = dow_st.stop_loss
                            d_tp  = dow_st.take_profit
                            dow_st.phase = "DONE"
                        else:
                            dtid = None
                    if dtid:
                        xp, xpnl = close_trade_market(dtid, pair, "max_drawdown")
                        append_trade_log(pair, d_dir, d_ep, xp, xpnl, d_sl, d_tp,
                                         "max_drawdown", None, False, False,
                                         strategy_name='DOW')
                log.warning("  All closed. Restart tomorrow.")
                send_telegram(f"MAX DRAWDOWN HIT. All positions closed. Balance: {balance:.2f}. Restart tomorrow.")
                break

            # Daily loss vs the firm's baseline: max(balance, equity) at 5 PM NY.
            day_blocked = (day_baseline - equity) >= day_baseline * PROP_DAILY_LOSS_LIMIT
            if day_blocked:
                log.warning(f"Daily loss limit -- no new entries (balance: {balance:.2f})")
                if not daily_limit_alerted:
                    daily_limit_alerted = True
                    send_telegram(f"Daily loss limit hit. No new entries today. Balance: {balance:.2f}")

            if loss_pause_active:
                day_blocked = True

            if strategy_disabled:
                day_blocked = True

            # ── Consistency day-cap guard (20% rule) ──────────────────────
            # Once today's CLOSED profit hits the cap, stop OPENING trades for
            # the day (IDLE→DONE, pendings cancelled; open positions keep
            # managing normally — this never force-closes anything).
            if CONSISTENCY_GUARD_ON and not consistency_guard_fired:
                todays_pnl = _todays_closed_pnl()
                if todays_pnl >= CONSISTENCY_DAY_CAP * INITIAL_DEPOSIT:
                    consistency_guard_fired = True
                    log.warning(f"CONSISTENCY GUARD: today closed {todays_pnl:+.2f} "
                                f">= cap — no new entries today (20% rule protection)")
                    send_telegram(f"Consistency guard: +{todays_pnl:.2f} closed today — "
                                  f"stopping new entries to protect the 20% rule")
                    pending_to_cancel = []
                    with _lock:
                        for p in PAIRS:
                            if states[p].phase == "IDLE":
                                states[p].phase = "DONE"
                            elif states[p].phase == "ORDER_PLACED":
                                pending_to_cancel.append((states[p].order_id, p))
                                states[p].phase = "DONE"
                        for ds in dow_states.values():
                            if ds.phase == "IDLE":
                                ds.phase = "DONE"
                    for oid, p in pending_to_cancel:
                        if oid:
                            cancel_pending(oid, p)

            # ── Per-pair ORB logic ────────────────────────────────────────
            for pair in PAIRS:
                try:
                    trade_pnl = run_pair(pair, states[pair], states, now, balance,
                                         day_blocked, news_times,
                                         consecutive_wins=consecutive_wins,
                                         consecutive_losses=consecutive_losses)
                except Exception as e:
                    log.error(f"[{pair}] Error in run_pair: {e}", exc_info=True)
                    trade_pnl = None

                if trade_pnl is not None:
                    recent_trades.append(trade_pnl)

                    # BOCPD: leading-indicator regime-break detector.
                    # Runs BEFORE the WR-based kill switch below — fires earlier
                    # on average (~5-15 trades) and triggers a soft 0.5× risk
                    # halving via bocpd_risk_factor() rather than a hard disable.
                    try:
                        update_bocpd(trade_pnl)
                    except Exception as e:
                        log.debug(f"BOCPD wiring error: {e}")

                    # Phase 6.4 — feed per-strategy circuit breaker. The breaker
                    # halts a strategy for LOSS_HALT_HOURS after K consecutive
                    # losses, independently of the global daily-loss pause.
                    try:
                        from agent import risk_manager as _rm
                        _rm.record_trade_outcome(f'orb_{pair.lower()}', trade_pnl)
                    except Exception as e:
                        log.debug(f"risk_manager outcome record failed: {e}")

                    if trade_pnl < 0:
                        consecutive_losses += 1
                        consecutive_wins    = 0
                        risk_pct = dynamic_risk(consecutive_wins, consecutive_losses)
                        send_telegram(
                            f"[ORB {pair}] LOSS  pnl={trade_pnl:.2f}\n"
                            f"Streak: {consecutive_losses} losses  next risk={risk_pct:.1%}"
                        )
                        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES and not loss_pause_active:
                            loss_pause_active = True
                            log.warning(f"CONSECUTIVE LOSS PAUSE: {consecutive_losses} losses in a row "
                                        f"-- no new entries today")
                            send_telegram(f"LOSS PAUSE: {consecutive_losses} losses in a row — no new entries today")
                            with _lock:
                                for p in PAIRS:
                                    if states[p].phase == "IDLE":
                                        states[p].phase = "DONE"
                    elif trade_pnl > 0:
                        consecutive_wins  += 1
                        consecutive_losses = 0
                        risk_pct = dynamic_risk(consecutive_wins, consecutive_losses)
                        send_telegram(
                            f"[ORB {pair}] WIN  pnl=+{trade_pnl:.2f}\n"
                            f"Streak: {consecutive_wins} wins  next risk={risk_pct:.1%}"
                        )
                    else:
                        # Breakeven close (e.g., SL moved to entry) — no streak update.
                        send_telegram(f"[ORB {pair}] BREAKEVEN  pnl=0.00 (streak unchanged)")

                    if len(recent_trades) >= 40:
                        wins    = sum(1 for t in recent_trades if t > 0)
                        roll_wr = wins / len(recent_trades)
                        avg_pnl = sum(recent_trades) / len(recent_trades)
                        if (roll_wr < 0.35 or avg_pnl < -50) and not strategy_disabled:
                            strategy_disabled = True
                            log.warning(f"KILL SWITCH: rolling WR={roll_wr:.0%}  "
                                        f"avg_pnl={avg_pnl:.2f} -- strategy disabled")
                        elif strategy_disabled and roll_wr >= 0.40 and avg_pnl >= 0:
                            strategy_disabled = False
                            log.info(f"Kill switch lifted: rolling WR={roll_wr:.0%}  "
                                     f"avg_pnl={avg_pnl:.2f}")

            # ── Dow dispersion overlay (US500/NAS100, Mon long / Thu short) ─
            for pair, dow_st in dow_states.items():
                try:
                    dow_pnl = run_dow(pair, dow_st, now, balance, day_blocked, news_times)
                except Exception as e:
                    log.error(f"[DOW_{pair}] Error in run_dow: {e}", exc_info=True)
                    dow_pnl = None
                if dow_pnl is not None:
                    recent_trades.append(dow_pnl)
                    # Feed the per-strategy circuit breaker (same as the ORB block).
                    try:
                        from agent import risk_manager as _rm
                        _rm.record_trade_outcome(f'dow_{pair.lower()}', dow_pnl)
                    except Exception as e:
                        log.debug(f"risk_manager outcome record failed: {e}")

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C).")
    finally:
        # review#3 — flatten ORB-magic positions BEFORE disconnecting from MT5.
        # Previously this `finally` only called shutdown(), leaving open
        # positions live with no software watching them through the night.
        try:
            positions = mt5.positions_get() or []
            ours_open = [p for p in positions
                         if getattr(p, 'magic', 0) in (ORB_MAGIC, DOW_MAGIC)]
            if ours_open:
                log.warning("Emergency flatten: %d ORB/DOW position(s) open at shutdown",
                            len(ours_open))
                for pos in ours_open:
                    try:
                        close_trade_market(str(pos.ticket), _base(pos.symbol),
                                           "emergency_shutdown")
                    except Exception as e:
                        log.error("emergency_shutdown close failed for ticket %s: %s",
                                  pos.ticket, e)
        except Exception as e:
            log.error("emergency_shutdown flatten loop failed: %s", e)
        stop_event.set()
        stream_thread.join(timeout=5)
        txn_thread.join(timeout=5)
        mt5.shutdown()
        log.info("MT5 connection closed. Stream threads stopped.")


if __name__ == "__main__":
    main()
