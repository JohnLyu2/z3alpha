"""Shared z3alpha2 submission smoke cases (data/smoke/benchmarks/cases.json)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BENCH_DIR = _PROJECT_ROOT / "data" / "smoke" / "benchmarks"
_CASES_FILE = _BENCH_DIR / "cases.json"
_SMTLIB_FALLBACK = _BENCH_DIR / "smtlib"

REQUIRED_SELECTOR_LOGICS = (
    "LIA", "LRA", "NIA", "NRA",
    "QF_ANIA", "QF_BV", "QF_DT", "QF_IDL", "QF_LIA", "QF_NIA", "QF_NRA",
    "QF_UFDT", "QF_UFDTNIA", "QF_UFNIA", "QF_UFNRA",
)


def smtlib_root() -> Path | None:
    config_path = _PROJECT_ROOT / "env_config.json"
    if config_path.is_file():
        root = json.loads(config_path.read_text()).get("smtlib_root")
        if root:
            return Path(root)
    env = os.environ.get("SMTLIB_ROOT")
    return Path(env) if env else None


def load_cases() -> list[dict[str, Any]]:
    return json.loads(_CASES_FILE.read_text())


def resolve_benchmark(smtlib_rel: str) -> Path | None:
    """Resolve an SMT-COMP benchmark path from smtlib_root or local fallback mirror."""
    root = smtlib_root()
    if root is not None:
        candidate = root / smtlib_rel
        if candidate.is_file():
            return candidate
    fallback = _SMTLIB_FALLBACK / smtlib_rel
    if fallback.is_file():
        return fallback
    return None


def benchmark_path(case: dict[str, Any]) -> Path | None:
    return resolve_benchmark(case["smtlib_rel"])


def run_case(
    z3alpha2: Path,
    submission_dir: Path,
    case: dict[str, Any],
    *,
    python: str | None = None,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run z3alpha2 --debug on a case; return (rc, stdout, stderr).

    Cases with require_verdict=false may use a per-case timeout (case["timeout"])
    and tolerate a kill: on TimeoutExpired rc is returned as -1 and whatever
    stderr was captured before the kill is returned.
    """
    bench = benchmark_path(case)
    if bench is None:
        raise FileNotFoundError(case["smtlib_rel"])
    effective_timeout = case.get("timeout", timeout)
    try:
        proc = subprocess.run(
            [python or sys.executable, str(z3alpha2), "--debug", str(bench)],
            cwd=submission_dir,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return -1, "", stderr


def check_case(case: dict[str, Any], stderr: str, stdout: str, rc: int) -> list[str]:
    """Return list of failure messages (empty if ok).

    Cases with require_verdict=false skip the sat/unsat verdict check and
    tolerate rc=-1 (process killed at timeout boundary).
    """
    errors: list[str] = []
    require_verdict = case.get("require_verdict", True)

    if require_verdict and rc != 0:
        errors.append(f"exit code {rc}")

    verdict = stdout.strip().split("\n")[0] if stdout.strip() else ""
    if require_verdict:
        if verdict not in ("sat", "unsat"):
            errors.append(f"verdict {verdict!r} not sat/unsat")
        expected = case.get("verdict")
        if expected and verdict != expected:
            errors.append(f"verdict {verdict!r} != {expected!r}")

    for needle in case.get("must_contain", []):
        if needle not in stderr:
            errors.append(f"stderr missing {needle!r}")

    for needle in case.get("must_not_contain", []):
        if needle in stderr:
            errors.append(f"stderr must not contain {needle!r}")

    one_of = case.get("one_of", [])
    if one_of and not any(all(n in stderr for n in group) for group in one_of):
        errors.append(f"stderr matched none of one_of groups: {one_of!r}")

    return errors
