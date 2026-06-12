"""
Inventory + download trigger for cached historical FX data. Powers the GUI
Data tab — shows what's loaded, lets the user download a specific pair, and
publishes a JSON status file the GUI polls for live progress bars.

Scans the on-disk caches directly (no DB, no network); cheap enough to call
on every Streamlit rerun.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from edge_engine import (
    ALL_PAIRS, CACHE_DIR, DUKA_INST, PREPARED_CACHE_DIR,
    fetch_all_data, prepare_df, ticks_to_m1,
)

log = logging.getLogger(__name__)

RUNTIME_DIR = Path(__file__).parent.parent / "runtime"
STATUS_PATH = RUNTIME_DIR / "download_status.json"

# DUKA_INST is canonical -> ticker (e.g. 'GBP_USD' -> 'GBPUSD'); invert for lookup.
_TICKER_TO_PAIR = {v: k for k, v in DUKA_INST.items()}

_RAW_FNAME_RE = re.compile(r"^(?P<ticker>[A-Z]{6})_(?P<date>\d{4}-\d{2}-\d{2})\.parquet$")


def _scan_raw_cache() -> dict[str, list[date]]:
    """Return {pair: sorted list of dates with at least one cached tick file}."""
    by_pair: dict[str, list[date]] = {}
    if not CACHE_DIR.exists():
        return by_pair
    for f in CACHE_DIR.glob("*.parquet"):
        m = _RAW_FNAME_RE.match(f.name)
        if not m:
            continue
        pair = _TICKER_TO_PAIR.get(m.group("ticker"))
        if pair is None:
            continue
        try:
            d = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
        except ValueError:
            continue
        by_pair.setdefault(pair, []).append(d)
    for v in by_pair.values():
        v.sort()
    return by_pair


def _prepared_path(pair: str) -> Path:
    ticker = DUKA_INST.get(pair, pair.replace("_", ""))
    return PREPARED_CACHE_DIR / f"{ticker}_m1.parquet"


def coverage() -> dict[str, dict]:
    """Return one entry per known pair: status, first/last date, day count.

    Status values:
      'loaded'       — raw cache present AND prepared cache present
      'raw_only'     — raw days cached, but no prepared parquet yet
      'not_downloaded' — no raw cache files at all
    """
    raw = _scan_raw_cache()
    out: dict[str, dict] = {}
    for pair in ALL_PAIRS:
        days = raw.get(pair, [])
        prep = _prepared_path(pair).exists()
        if not days:
            out[pair] = {
                "status":       "not_downloaded",
                "first":        None,
                "last":         None,
                "n_days":       0,
                "age_days":     None,
                "prepared":     prep,
            }
            continue
        last = days[-1]
        age = (datetime.now(timezone.utc).date() - last).days
        out[pair] = {
            "status":   "loaded" if prep else "raw_only",
            "first":    days[0],
            "last":     last,
            "n_days":   len(days),
            "age_days": age,
            "prepared": prep,
        }
    return out


def missing_pairs(target_pairs: list[str]) -> list[str]:
    """Return the subset of target_pairs that have no raw cache at all."""
    cov = coverage()
    return [p for p in target_pairs
            if cov.get(p, {}).get("status") == "not_downloaded"]


def staleness(pair: str) -> int | None:
    """Days since the most recent cached bar for this pair, or None if not cached."""
    info = coverage().get(pair, {})
    return info.get("age_days")


# ── Download trigger + progress publishing ──────────────────────────────────

_STATUS_LOCK = threading.Lock()


def _atomic_write(path: Path, payload: dict) -> None:
    RUNTIME_DIR.mkdir(exist_ok=True)
    import os, tempfile
    fd, tmp = tempfile.mkstemp(prefix=".dl_status_", suffix=".json", dir=str(RUNTIME_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as e:
        try: os.unlink(tmp)
        except OSError: pass
        log.debug("[data_inventory] status write failed: %s", e)


def _write_status(payload: dict) -> None:
    """Legacy single-pair writer.

    Also fans the update out to the per-pair section of the multi-pair
    status file so callers using download_pairs see live progress without
    needing to opt in.
    """
    _atomic_write(STATUS_PATH, payload)
    pair = payload.get("pair")
    if pair:
        _update_multi_status(pair, payload)


def read_status() -> dict | None:
    """Read the last-published download status, or None if no run has started."""
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Multi-pair status (one entry per pair in a single JSON file) ─────────────

MULTI_STATUS_PATH = RUNTIME_DIR / "download_status_multi.json"


def _read_multi() -> dict:
    if not MULTI_STATUS_PATH.exists():
        return {"pairs": {}, "active_pair": None, "ts": None}
    try:
        d = json.loads(MULTI_STATUS_PATH.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return {"pairs": {}, "active_pair": None, "ts": None}
        d.setdefault("pairs", {})
        return d
    except Exception:
        return {"pairs": {}, "active_pair": None, "ts": None}


def _update_multi_status(pair: str, entry: dict, *, active: bool = True) -> None:
    """Merge `entry` into the per-pair multi-status JSON atomically."""
    with _STATUS_LOCK:
        cur = _read_multi()
        cur["pairs"][pair] = {**cur["pairs"].get(pair, {}), **entry,
                              "ts": datetime.now(timezone.utc).isoformat()}
        if active:
            cur["active_pair"] = pair
        cur["ts"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(MULTI_STATUS_PATH, cur)


def read_multi_status() -> dict:
    """Read the multi-pair download status. Always returns a dict with
    keys 'pairs' (dict[pair] -> entry), 'active_pair', 'ts'."""
    return _read_multi()


def reset_multi_status(pairs: list[str]) -> None:
    """Initialise the multi-pair status for a fresh download run."""
    with _STATUS_LOCK:
        _atomic_write(MULTI_STATUS_PATH, {
            "pairs": {p: {"phase": "queued", "days_done": 0, "days_total": 0}
                      for p in pairs},
            "active_pair": None,
            "ts": datetime.now(timezone.utc).isoformat(),
        })


def download_pair(pair: str) -> dict:
    """Synchronous: download tick data for one pair, prepare m1 features, save
    to prepared cache. Publishes per-day progress to STATUS_PATH so a GUI
    polling that file can render a live bar.

    Returns a summary dict: {pair, ok, n_bars, error?}.
    """
    if pair not in ALL_PAIRS:
        return {"pair": pair, "ok": False, "error": f"unknown pair {pair}"}

    def _cb(p):
        _write_status({**p, "ts": datetime.now(timezone.utc).isoformat()})

    _write_status({
        "pair": pair, "phase": "starting", "days_done": 0, "days_total": 0,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    try:
        raw = fetch_all_data(pair, progress_callback=_cb)
    except Exception as e:
        _write_status({"pair": pair, "phase": "error", "error": str(e),
                       "ts": datetime.now(timezone.utc).isoformat()})
        return {"pair": pair, "ok": False, "error": str(e)}

    if raw.empty:
        _write_status({"pair": pair, "phase": "error",
                       "error": "no tick data returned",
                       "ts": datetime.now(timezone.utc).isoformat()})
        return {"pair": pair, "ok": False, "error": "no tick data"}

    _write_status({"pair": pair, "phase": "preparing", "days_done": 0,
                   "days_total": 0, "ts": datetime.now(timezone.utc).isoformat()})
    try:
        m1 = ticks_to_m1(raw)
        df = prepare_df(m1, pair)
        if df.empty:
            return {"pair": pair, "ok": False, "error": "prepare_df returned empty"}
        save_df = df.copy()
        save_df["date"] = save_df["date"].astype(str)
        cache_path = PREPARED_CACHE_DIR / f"{pair}_m1.parquet"
        save_df.to_parquet(cache_path, index=False)
    except Exception as e:
        _write_status({"pair": pair, "phase": "error", "error": str(e),
                       "ts": datetime.now(timezone.utc).isoformat()})
        return {"pair": pair, "ok": False, "error": str(e)}

    _write_status({"pair": pair, "phase": "done", "n_bars": len(df),
                   "ts": datetime.now(timezone.utc).isoformat()})
    return {"pair": pair, "ok": True, "n_bars": len(df)}


def download_pair_async(pair: str) -> threading.Thread:
    """Kick off download_pair in a daemon thread. The GUI polls read_status()
    instead of blocking on the call."""
    t = threading.Thread(target=download_pair, args=(pair,), daemon=True,
                         name=f"download_{pair}")
    t.start()
    return t


def download_pairs(pairs: list[str], stop_check=None) -> list[dict]:
    """Sequentially download multiple pairs, updating per-pair multi-status.

    Args:
      pairs:      List of pair names (e.g. ['EUR_USD', 'GBP_USD']).
      stop_check: Optional zero-arg callable returning True to abort between
                  pairs. Used by the mission-control GUI to honour stop
                  requests cleanly without killing mid-pair work.

    Already-cached pairs short-circuit to 'done' without network calls.
    Returns a list of summary dicts in the same order as `pairs`.
    """
    pairs = [p for p in pairs if p in ALL_PAIRS]
    reset_multi_status(pairs)
    cov = coverage()
    summaries: list[dict] = []

    for pair in pairs:
        if stop_check is not None and stop_check():
            _update_multi_status(pair, {"phase": "stopped"}, active=False)
            summaries.append({"pair": pair, "ok": False, "error": "stopped"})
            continue

        info = cov.get(pair, {})
        if info.get("status") == "loaded":
            _update_multi_status(pair, {"phase": "done", "n_days": info.get("n_days", 0)},
                                 active=False)
            summaries.append({"pair": pair, "ok": True, "n_bars": None,
                              "cached": True})
            continue

        _update_multi_status(pair, {"phase": "starting", "days_done": 0,
                                    "days_total": 0})
        summary = download_pair(pair)
        summaries.append(summary)
        if summary.get("ok"):
            _update_multi_status(pair, {"phase": "done",
                                        "n_bars": summary.get("n_bars", 0)},
                                 active=False)
        else:
            _update_multi_status(pair, {"phase": "error",
                                        "error": summary.get("error", "unknown")},
                                 active=False)

    return summaries


def download_pairs_async(pairs: list[str], stop_check=None) -> threading.Thread:
    """Daemon-thread wrapper around download_pairs."""
    t = threading.Thread(target=download_pairs, args=(pairs, stop_check),
                         daemon=True, name=f"download_pairs_{len(pairs)}")
    t.start()
    return t
