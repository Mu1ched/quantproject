"""
Phase 8 — secret redaction filter for Python logging.

Anthropic API keys, Telegram bot tokens, MT5 logins, and arbitrary
"key=value" debug dumps from .env are easy to leak into agent.log
or stdout. This filter scrubs known shapes before they ever hit a
handler.

Wired by agent/main.py and MT5Live.2.py at startup. Failure-mode is
to leave the message unchanged — never raise.

Patterns we redact:
  * sk-ant-...                            Anthropic API keys
  * <6-12 digits>:<30+ alphanum chars>     Telegram bot tokens
  * password=...                           secret key=val pairs in .env dumps
  * Bearer <token>                         HTTP auth headers
"""

from __future__ import annotations

import logging
import re

_SECRET_PATTERNS = [
    re.compile(r'sk-ant-[A-Za-z0-9_\-]{20,}'),
    re.compile(r'\b\d{6,12}:[A-Za-z0-9_\-]{25,}\b'),
    re.compile(r'(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|bearer)\s*[=:]\s*[\'"]?([^\s\'"]{6,})'),
    re.compile(r'(?i)bearer\s+[A-Za-z0-9_\-\.]{16,}'),
]

_REDACT = '<redacted>'


def _scrub(text: str) -> str:
    out = text
    for pat in _SECRET_PATTERNS:
        try:
            if pat.groups >= 2:
                # Preserve the key name; mask only the value
                out = pat.sub(lambda m: f"{m.group(1)}={_REDACT}", out)
            else:
                out = pat.sub(_REDACT, out)
        except Exception:
            continue
    return out


class SecretRedactionFilter(logging.Filter):
    """Scrubs known secret shapes from log records in-place. Never raises."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Scrub the formatted message — handles %-style args correctly.
            msg = record.getMessage()
            scrubbed = _scrub(msg)
            if scrubbed != msg:
                record.msg = scrubbed
                record.args = ()
        except Exception:
            pass
        return True


def install_global_redaction() -> None:
    """Attach a redaction filter to the root logger so every handler picks it up.

    Idempotent — safe to call multiple times.
    """
    root = logging.getLogger()
    for f in root.filters:
        if isinstance(f, SecretRedactionFilter):
            return
    root.addFilter(SecretRedactionFilter())
    # Also add to existing handlers so messages emitted *before* basicConfig
    # picks up our filter still get scrubbed.
    for h in root.handlers:
        already = any(isinstance(f, SecretRedactionFilter) for f in h.filters)
        if not already:
            h.addFilter(SecretRedactionFilter())
