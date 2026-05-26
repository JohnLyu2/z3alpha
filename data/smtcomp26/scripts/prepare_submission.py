#!/usr/bin/env python3
"""
Prepare the z3alpha2 submission directory for packaging.

Copies inference code and trained selectors into data/smtcomp26/submission/,
vendors Python dependencies (numpy, scikit-learn, joblib, scipy) for the
competition machine (Python 3.12, x86-64 Linux), then verifies all expected
files are present.

Usage:
    python prepare_submission.py

Selector pinning is configured in SELECTOR_PINS below.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parents[3]
SUBMISSION_DIR  = REPO_ROOT / "data/smtcomp26/submission"
SELECTION_RUNS  = REPO_ROOT / "data/smtcomp26/selection_runs"

# ── Configure which selector run to pin for each logic group ──────────────────
# Edit these when you want to promote a new run to the submission.
SELECTOR_PINS: dict[str, str] = {
    "QF_Datatypes": "20260525_153708_k5",
}

# ── Inference library files to bundle ─────────────────────────────────────────
LIB_SOURCES = [
    REPO_ROOT / "z3alpha/ml_selector.py",
    REPO_ROOT / "z3alpha/smtlib_features.py",
]

# ── Python packages to vendor for competition machine (Python 3.12, x86-64) ──
VENDOR_PACKAGES = ["numpy", "scikit-learn", "joblib", "scipy"]
VENDOR_PYTHON_VERSION = "3.12"
VENDOR_PLATFORM = "manylinux_2_17_x86_64"
VENDOR_ABI = "cp312"


def copy_lib():
    lib_dir = SUBMISSION_DIR / "lib"
    lib_dir.mkdir(exist_ok=True)
    for src in LIB_SOURCES:
        dst = lib_dir / src.name
        shutil.copy2(src, dst)
        print(f"  copied {src.relative_to(REPO_ROOT)} → submission/lib/{src.name}")


def vendor_packages():
    vendor_dir = SUBMISSION_DIR / "vendor"
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.mkdir()

    pip = REPO_ROOT / ".venv" / "bin" / "pip"
    if not pip.is_file():
        pip = "pip"

    wheel_dir = SUBMISSION_DIR / ".vendor_wheels"
    wheel_dir.mkdir(exist_ok=True)

    print(f"  downloading wheels (Python {VENDOR_PYTHON_VERSION}, {VENDOR_PLATFORM}) ...")
    subprocess.run(
        [
            str(pip), "download",
            "--dest", str(wheel_dir),
            "--python-version", VENDOR_PYTHON_VERSION,
            "--platform", VENDOR_PLATFORM,
            "--abi", VENDOR_ABI,
            "--only-binary", ":all:",
            "--implementation", "cp",
            "--quiet",
        ] + VENDOR_PACKAGES,
        check=True,
    )

    import zipfile
    for whl in wheel_dir.glob("*.whl"):
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(vendor_dir)
        print(f"  installed {whl.name}")

    shutil.rmtree(wheel_dir)


def copy_selectors():
    for group, run_name in SELECTOR_PINS.items():
        src_dir = SELECTION_RUNS / group / run_name
        dst_dir = SUBMISSION_DIR / "selectors" / group
        if not src_dir.is_dir():
            sys.exit(f"Pinned run not found: {src_dir}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("selector.joblib", "meta.json"):
            shutil.copy2(src_dir / fname, dst_dir / fname)
        print(f"  copied {group}/{run_name} → submission/selectors/{group}/")


def verify():
    errors = []

    vendor_dir = SUBMISSION_DIR / "vendor"
    vendor_checks = [vendor_dir / pkg for pkg in ("numpy", "sklearn", "joblib", "scipy")]

    expected = (
        [SUBMISSION_DIR / "z3alpha2.py", SUBMISSION_DIR / "bin" / "z3"]
        + [SUBMISSION_DIR / "lib" / src.name for src in LIB_SOURCES]
        + vendor_checks
        + [
            SUBMISSION_DIR / "selectors" / group / fname
            for group in SELECTOR_PINS
            for fname in ("selector.joblib", "meta.json")
        ]
    )

    for path in expected:
        if path.exists():
            print(f"  ok  {path.relative_to(SUBMISSION_DIR)}")
        else:
            print(f"  MISSING  {path.relative_to(SUBMISSION_DIR)}")
            errors.append(path)

    return errors


def main():
    print("── Copying inference library ────────────────────────────────────")
    copy_lib()

    print("\n── Vendoring Python dependencies ────────────────────────────────")
    vendor_packages()

    print("\n── Copying pinned selectors ─────────────────────────────────────")
    copy_selectors()

    print("\n── Verifying submission directory ───────────────────────────────")
    errors = verify()

    if errors:
        sys.exit(f"\nSubmission incomplete: {len(errors)} file(s) missing.")
    else:
        print(f"\nSubmission ready at {SUBMISSION_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
