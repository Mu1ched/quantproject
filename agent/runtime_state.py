"""
Runtime state + control for the autonomous agent.

Two JSON files in ./runtime/:

  agent_status.json   — written by agent loop, read by GUI.
                        Always-current phase, counters, and current strategy.

  agent_control.json  — written by GUI, read by agent loop.
                        Holds the latest command: pause / resume / stop.

Both are atomically written (tmp + os.replace) so a partial read is impossible.
The GUI rerenders on a timer and reads the status; the loop polls the control
file at safe checkpoints (between hypotheses) and acts.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RUNTIME_DIR  = Path(__file__).parent.parent / "runtime"
STATUS_PATH  = RUNTIME_DIR / "agent_status.json"
CONTROL_PATH = RUNTIME_DIR / "agent_control.json"

_LOCK = threading.Lock()


# ── Atomic write helper ──────────────────────────────────────────────────────

def _atomic_write(path: Path, payload: dict) -> None:
    RUNTIME_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rt_", suffix=".json", dir=str(RUNTIME_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as e:
        try: os.unlink(tmp)
        except OSError: pass
        log.debug("[runtime_state] atomic_write failed: %s", e)


def _read_or(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else dict(default)
    except Exception:
        return dict(default)


# ── Status (agent → GUI) ─────────────────────────────────────────────────────

_DEFAULT_STATUS: dict = {
    "phase":             "idle",     # idle | downloading | preparing | generating | backtesting | paused | stopped
    "round":             0,
    "tested_today":      0,
    "survivors_today":   0,
    "spend_today_usd":   0.0,
    "current_strategy":  None,
    "current_pair":      None,
    "current_session":   None,
    "activity":          None,       # free-form fine-grained "now doing X" string
    "last_error":        None,
    "started_at":        None,
    "ts":                None,
}


def read_status() -> dict:
    return _read_or(STATUS_PATH, _DEFAULT_STATUS)


def update_status(**fields: Any) -> dict:
    """Merge fields into the current status and persist atomically.

    Unknown keys are accepted — callers can stash arbitrary diagnostic
    fields without us forcing a schema migration.
    """
    with _LOCK:
        cur = read_status()
        cur.update({k: v for k, v in fields.items() if v is not None
                    or k in ("current_strategy", "current_pair",
                             "current_session", "activity", "last_error")})
        cur["ts"] = datetime.now(timezone.utc).isoformat()
        if cur.get("started_at") is None and cur["phase"] != "idle":
            cur["started_at"] = cur["ts"]
        _atomic_write(STATUS_PATH, cur)
        return cur


def reset_status() -> None:
    with _LOCK:
        _atomic_write(STATUS_PATH, {**_DEFAULT_STATUS,
                                    "ts": datetime.now(timezone.utc).isoformat()})


# ── Control (GUI → agent) ────────────────────────────────────────────────────

_DEFAULT_CONTROL: dict = {"command": "run", "ts": None}

# Valid commands. Anything else is treated as "run".
_VALID_COMMANDS = {"run", "pause", "resume", "stop"}


def read_control() -> dict:
    return _read_or(CONTROL_PATH, _DEFAULT_CONTROL)


def set_command(command: str) -> dict:
    """GUI-side helper. `resume` is normalised to `run`."""
    if command == "resume":
        command = "run"
    if command not in _VALID_COMMANDS:
        raise ValueError(f"invalid command {command!r}")
    payload = {"command": command, "ts": datetime.now(timezone.utc).isoformat()}
    with _LOCK:
        _atomic_write(CONTROL_PATH, payload)
    return payload


def current_command() -> str:
    return read_control().get("command", "run")


# ── Loop-side checkpoint ─────────────────────────────────────────────────────

class StopRequested(Exception):
    """Raised inside the agent loop when the GUI signals stop."""


def checkpoint(poll_secs: float = 2.0) -> None:
    """Call at every safe point inside the agent loop.

    Behaviour by command:
      run     — return immediately.
      pause   — block, polling every `poll_secs`, until command flips to
                run or stop. While paused, status.phase is updated to
                'paused' so the GUI shows the correct state.
      stop    — raise StopRequested. Caller wraps the loop in try/except
                so SQLite + Claude state stays consistent.
    """
    cmd = current_command()
    if cmd == "stop":
        update_status(phase="stopped")
        raise StopRequested()
    if cmd != "pause":
        return

    prev_phase = read_status().get("phase", "idle")
    update_status(phase="paused")
    while True:
        time.sleep(poll_secs)
        cmd = current_command()
        if cmd == "stop":
            update_status(phase="stopped")
            raise StopRequested()
        if cmd != "pause":
            update_status(phase=prev_phase)
            return


def stop_requested() -> bool:
    """Non-blocking variant — returns True if a stop is pending. Useful for
    long inner loops (e.g. download_pairs) that want to exit cleanly."""
    return current_command() == "stop"
