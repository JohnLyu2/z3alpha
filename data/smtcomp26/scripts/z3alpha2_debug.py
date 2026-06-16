#!/usr/bin/env python3
"""
z3-alpha2 compatibility checker and debug runner for SMT-COMP 2026.

Verifies the submission environment: Python version, vendored packages,
selector loading, and feature extraction.

On success: all check output is written to /tmp/z3alpha2_debug.log and the
process execs into z3alpha2.py — stdout is clean (only sat/unsat) so the
competition test infrastructure passes.

On failure: accumulated check output plus the error are printed to stderr
and the script exits non-zero — the competition test fails with the reason
visible in CI logs.

Usage:
    ./z3alpha2_debug.py <benchmark.smt2>   # check + run solver
    ./z3alpha2_debug.py                    # check only (exits 0 on success)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
LOG_FILE       = Path("/tmp/z3alpha2_debug.log")

_log: list[str] = []


def info(msg: str) -> None:
    _log.append(f"[z3alpha2-debug] {msg}")


def fail(msg: str) -> None:
    for line in _log:
        print(line, file=sys.stderr)
    print(f"[z3alpha2-debug] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ── 1. Python version ─────────────────────────────────────────────────────────
info(f"Python {sys.version}")
if sys.version_info[:2] != (3, 12):
    fail(f"expected Python 3.12, got {sys.version_info[:2]}")
info("Python version OK")

# ── 2. Paths ──────────────────────────────────────────────────────────────────
vendor = SUBMISSION_DIR / "vendor"
lib    = SUBMISSION_DIR / "lib"
for p in (vendor, lib):
    if not p.is_dir():
        fail(f"directory not found: {p}")
sys.path.insert(0, str(vendor))
sys.path.insert(0, str(lib))
info(f"vendor={vendor}")
info(f"lib={lib}")

# ── 3. Vendored packages ──────────────────────────────────────────────────────
for pkg in ("numpy", "sklearn", "joblib", "scipy", "xgboost"):
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        info(f"  {pkg} {version}  ({Path(mod.__file__).parent})")
    except ImportError as e:
        fail(f"import {pkg} failed: {e}")

# ── 4. Inference library ──────────────────────────────────────────────────────
try:
    from smt_select import PairwiseSelector, bench_feature_vector, FEATURE_NAMES
    info(f"smt_select loaded  ({len(FEATURE_NAMES)} features)")
except Exception as e:
    fail(f"import smt_select failed: {e}")

# ── 5. Selectors ──────────────────────────────────────────────────────────────
selectors_dir = SUBMISSION_DIR / "selectors"
if not selectors_dir.is_dir():
    fail(f"selectors/ not found: {selectors_dir}")

for group_dir in sorted(selectors_dir.iterdir()):
    if not group_dir.is_dir():
        continue
    meta_path = group_dir / "meta.json"
    if not meta_path.is_file():
        continue
    try:
        meta  = json.loads(meta_path.read_text())
        logic = meta.get("logic", group_dir.name)
    except Exception as e:
        fail(f"{group_dir.name}/meta.json unreadable: {e}")

    if "fixed_strategy" in meta:
        info(f"  {logic}: fixed_strategy")
        continue

    joblib_path = group_dir / "selector.joblib"
    if not joblib_path.is_file():
        fail(f"  {logic}: selector.joblib missing")
    try:
        sel = PairwiseSelector.load(joblib_path)
        info(f"  {logic}: loaded  strategies={sel.strategies}")
    except Exception as e:
        fail(f"  {logic}: selector load failed: {e}")

# ── 6. Z3 binary ─────────────────────────────────────────────────────────────
z3 = SUBMISSION_DIR / "bin" / "z3"
if not z3.is_file():
    fail(f"Z3 binary not found: {z3}")
try:
    out = subprocess.check_output([str(z3), "--version"], text=True).strip()
    info(f"Z3: {out}")
except Exception as e:
    fail(f"Z3 binary failed: {e}")

# ── 7. All checks passed — write log to file, exec solver ────────────────────
info("all compatibility checks passed")
try:
    LOG_FILE.write_text("\n".join(_log) + "\n")
except Exception:
    pass  # non-fatal: log file write failure should not block the solver

if len(sys.argv) > 1:
    os.execv(sys.executable, [sys.executable, str(SUBMISSION_DIR / "z3alpha2.py")] + sys.argv[1:])
