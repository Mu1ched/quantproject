"""Tests for the AST security validator in agent/code_writer.py (review#1)."""
import ast

import pytest

from agent.code_writer import _SecurityValidator, _validate_security


def _violations(code: str) -> list[str]:
    v = _SecurityValidator()
    v.visit(ast.parse(code))
    return v.violations


# ── Allowed code paths ────────────────────────────────────────────────────────


def test_clean_strategy_passes():
    code = """
def entry_demo(bst, slot, row, ts, pair, slip, hspd, sess_cfg, regime, regime_mult, **kw):
    if regime != 'TRENDING':
        return False
    return True
"""
    assert _violations(code) == []


def test_whitelisted_imports_pass():
    code = """
import numpy as np
import pandas
from math import isnan
from datetime import time
from edge_engine import spread_gate
"""
    assert _violations(code) == []


# ── Forbidden imports ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    "import os",
    "import subprocess",
    "import socket",
    "import sys",
    "import importlib",
    "import pickle",
    "import requests",
    "import shutil",
    "import ctypes",
])
def test_forbidden_top_level_import_rejected(bad):
    vs = _violations(bad)
    assert any("forbidden import" in v for v in vs), vs


@pytest.mark.parametrize("bad", [
    "from os import system",
    "from subprocess import Popen",
    "from socket import socket",
    "from importlib import import_module",
])
def test_forbidden_from_import_rejected(bad):
    vs = _violations(bad)
    assert any("forbidden 'from" in v for v in vs), vs


def test_relative_import_rejected():
    vs = _violations("from . import sibling")
    assert any("relative import" in v for v in vs), vs


# ── Forbidden builtin calls ───────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    "eval('1+1')",
    "exec('x=1')",
    "compile('1', '<s>', 'eval')",
    "__import__('os')",
    "open('/etc/passwd')",
    "globals()",
    "locals()",
])
def test_forbidden_builtin_call_rejected(bad):
    vs = _violations(bad)
    assert any("forbidden call to builtin" in v for v in vs), vs


# ── Dunder escape hatches ─────────────────────────────────────────────────────


def test_dunder_access_rejected():
    # Classic sandbox-escape pattern: ().__class__.__bases__[0].__subclasses__()
    code = "obj.__class__.__bases__[0].__subclasses__()"
    vs = _violations(code)
    assert any("forbidden dunder access" in v and "__bases__" in v for v in vs), vs


def test_innocuous_dunders_allowed():
    code = "x = obj.__class__.__name__"
    assert _violations(code) == []


# ── End-to-end via _validate_security ────────────────────────────────────────


def test_validate_security_raises_on_violation():
    tree = ast.parse("import os")
    with pytest.raises(ValueError, match="security validator"):
        _validate_security(tree)


def test_validate_security_passes_clean_code():
    tree = ast.parse("def entry_x(): return 1")
    _validate_security(tree)  # must not raise
