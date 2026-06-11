#!/usr/bin/env python3
"""Run z3alpha2 --debug on SMT-LIB cases from cases.json."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _project_root(bench_dir: Path) -> Path:
    return bench_dir.resolve().parent.parent.parent


def _smtlib_root(bench_dir: Path) -> Path | None:
    import os

    env = os.environ.get("SMTLIB_ROOT")
    if env:
        return Path(env)
    config = _project_root(bench_dir) / "env_config.json"
    if config.is_file():
        root = json.loads(config.read_text()).get("smtlib_root")
        if root:
            return Path(root)
    return None


def _resolve(smtlib_rel: str, bench_dir: Path) -> Path | None:
    bench_dir = bench_dir.resolve()
    root = _smtlib_root(bench_dir)
    if root is not None:
        candidate = root / smtlib_rel
        if candidate.is_file():
            return candidate.resolve()
    fallback = bench_dir / "smtlib" / smtlib_rel
    if fallback.is_file():
        return fallback.resolve()
    return None


def _check(case: dict, stderr: str, stdout: str, rc: int) -> list[str]:
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
    if one_of and not any(all(n in stderr for n in g) for g in one_of):
        errors.append(f"stderr matched none of one_of groups: {one_of!r}")

    return errors


def _run_case(z3alpha2: Path, submission: Path, case: dict, bench: Path, path: Path) -> tuple[int, str, str]:
    effective_timeout = case.get("timeout", 120)
    try:
        proc = subprocess.run(
            [sys.executable, str(z3alpha2), "--debug", str(path)],
            cwd=submission,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        stderr_bytes = exc.stderr or b""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if isinstance(stderr_bytes, bytes) else (stderr_bytes or "")
        return -1, "", stderr


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <submission_dir> <bench_dir>", file=sys.stderr)
        return 2

    submission = Path(sys.argv[1]).resolve()
    bench = Path(sys.argv[2]).resolve()
    z3alpha2 = submission / "z3alpha2.py"
    cases = json.loads((bench / "cases.json").read_text())

    failed = 0
    skipped = 0
    for case in cases:
        rel = case["smtlib_rel"]
        path = _resolve(rel, bench)
        if path is None:
            print(f"SKIP {case['name']}: missing {rel}")
            skipped += 1
            continue

        rc, stdout, stderr = _run_case(z3alpha2, submission, case, bench, path)
        errors = _check(case, stderr, stdout, rc)
        if errors:
            failed += 1
            print(f"FAIL {case['name']} ({case['logic']})")
            for err in errors:
                print(f"  - {err}")
            print(stderr)
        else:
            print(f"OK {case['name']} ({case['logic']})")

    if failed:
        return 1
    if skipped == len(cases):
        print("no benchmarks available (set SMTLIB_ROOT)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
