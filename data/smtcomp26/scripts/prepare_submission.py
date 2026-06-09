#!/usr/bin/env python3
"""
Prepare the z3alpha2 submission directory for packaging.

Copies the entry point, inference code, and trained selectors into
data/smtcomp26/submission/, vendors Python dependencies (numpy, scikit-learn,
joblib, scipy, xgboost-cpu) for the competition machine (Python 3.12, x86-64
Linux, glibc 2.28+), then verifies all expected files are present.

Usage:
    python prepare_submission.py
    python prepare_submission.py --skip-vendor   # reuse existing vendor/

Edit data/smtcomp26/scripts/z3alpha2.py; prepare_submission copies it into
submission/ alongside lib/ and selectors/.

Usage:
    python prepare_submission.py

Selector pinning is configured in SELECTOR_PINS below (group folder → training run).
Deployment settings (presolver, scheduler timeout) are in INFERENCE_DEFAULTS below
and merged into submission meta.json when copying selectors.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parents[3]
SCRIPTS_DIR     = Path(__file__).resolve().parent
SUBMISSION_DIR  = REPO_ROOT / "data/smtcomp26/submission"
SELECTION_RUNS  = REPO_ROOT / "data/smtcomp26/selection_runs"

ENTRYPOINT_SOURCE = SCRIPTS_DIR / "z3alpha2.py"

# ── Pin which training run to copy into submission/selectors/<group>/ ───────────
# Edit when promoting a new run. Logic routing is read from meta.json at runtime.
SELECTOR_PINS: dict[str, str] = {
    "QF_NRA": "20260602_103053",
    "LRA": "20260606_095841",
    "NIA": "20260606_100800",
    "NRA": "20260606_101800",
    "QF_DT": "20260606_103208",
    "QF_UFDT": "20260606_103552",
    "QF_NIA": "20260606_110320",
    "QF_ANIA": "20260606_113829",
    "QF_UFDTNIA": "20260606_114414",
    "QF_UFNIA": "20260606_115151",
    "QF_UFNRA": "20260606_115612",
    "QF_BV": "20260609_100826",
}

# ── z3alpha2 deployment settings (merged into submission meta.json on copy) ────
INFERENCE_DEFAULTS: dict[str, dict] = {
    "QF_NRA": {"presolver_seconds": 8.0, "scheduler_timeout": 960.0},
    "LRA": {"presolver_seconds": 0.0},
    "NIA": {"presolver_seconds": 0.0, "scheduler_timeout": 960.0},
    "NRA": {"presolver_seconds": 0.0, "scheduler_timeout": 960.0},
    "QF_DT": {"presolver_seconds": 0.0},
    "QF_UFDT": {"presolver_seconds": 0.0, "scheduler_timeout": 960.0},
    "QF_NIA": {"presolver_seconds": 8.0, "scheduler_timeout": 960.0},
    "QF_ANIA": {"presolver_seconds": 0.0, "scheduler_timeout": 960.0},
    "QF_UFDTNIA": {"presolver_seconds": 0.0, "scheduler_timeout": 960.0},
    "QF_UFNIA": {"presolver_seconds": 0.0, "scheduler_timeout": 960.0},
    "QF_UFNRA": {"presolver_seconds": 0.0, "scheduler_timeout": 960.0},
    "QF_BV": {"presolver_seconds": 8.0, "scheduler_timeout": 960.0},
}

# ── Fixed-tactic logics (meta.json only, no selector.joblib) ───────────────────
FIXED_LOGIC_CONFIGS: dict[str, dict] = {
    "LIA": {
        "logic": "LIA",
        "fixed_strategy": ["tactic.default_tactic=(then qe_rec smt)"],
    },
}

# ── Inference library files to bundle ─────────────────────────────────────────
INFER_SOURCE  = REPO_ROOT / "z3alpha/smt_select_infer.py"
SMTLIB_SOURCE = REPO_ROOT / "z3alpha/smtlib_features.py"

# ── Python packages to vendor for competition machine (Python 3.12, x86-64) ──
# xgboost-cpu avoids CUDA/NCCL deps bundled in the full xgboost wheel (~800 MB saved).
VENDOR_PACKAGES = ["numpy", "scikit-learn", "joblib", "scipy", "xgboost-cpu"]
VENDOR_PYTHON_VERSION = "3.12"
VENDOR_PLATFORM = "manylinux_2_28_x86_64"
VENDOR_ABI = "cp312"


def _venv_bin(name: str) -> Path | str:
    """Prefer .venv312 (cp312 target) over .venv."""
    for venv in (".venv312", ".venv"):
        path = REPO_ROOT / venv / "bin" / name
        if path.is_file():
            return path
    return name


def copy_entrypoint():
    if not ENTRYPOINT_SOURCE.is_file():
        sys.exit(f"Entry point not found: {ENTRYPOINT_SOURCE}")
    SUBMISSION_DIR.mkdir(exist_ok=True)
    dst = SUBMISSION_DIR / "z3alpha2.py"
    shutil.copy2(ENTRYPOINT_SOURCE, dst)
    dst.chmod(dst.stat().st_mode | 0o111)
    print(f"  copied {ENTRYPOINT_SOURCE.relative_to(REPO_ROOT)} → submission/z3alpha2.py")


def copy_lib():
    lib_dir = SUBMISSION_DIR / "lib"
    lib_dir.mkdir(exist_ok=True)
    shutil.copy2(SMTLIB_SOURCE, lib_dir / "smtlib_features.py")
    print(f"  copied {SMTLIB_SOURCE.relative_to(REPO_ROOT)} → submission/lib/smtlib_features.py")
    shutil.copy2(INFER_SOURCE, lib_dir / "smt_select.py")
    print(f"  copied {INFER_SOURCE.relative_to(REPO_ROOT)} → submission/lib/smt_select.py")


def vendor_packages():
    vendor_dir = SUBMISSION_DIR / "vendor"
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.mkdir()

    pip = _venv_bin("pip")

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

    size_mb = sum(f.stat().st_size for f in vendor_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  vendor/ size: {size_mb:.1f} MB")


def copy_selectors():
    for group, run_name in SELECTOR_PINS.items():
        src_dir = SELECTION_RUNS / group / run_name
        dst_dir = SUBMISSION_DIR / "selectors" / group
        if not src_dir.is_dir():
            sys.exit(f"Pinned run not found: {src_dir}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / "selector.joblib", dst_dir / "selector.joblib")
        meta = json.loads((src_dir / "meta.json").read_text())
        meta.update(INFERENCE_DEFAULTS.get(group, {}))
        (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  copied {group}/{run_name} → submission/selectors/{group}/")


def copy_fixed_logics():
    for group, meta in FIXED_LOGIC_CONFIGS.items():
        dst_dir = SUBMISSION_DIR / "selectors" / group
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  wrote fixed logic {group} → submission/selectors/{group}/")


def repackage_selectors():
    """Re-save selector.joblib files so the pickled class is smt_select.PairwiseSelector.

    Selectors are saved during training when PairwiseSelector lives in
    z3alpha.smt_select_infer.PairwiseSelector. Temporarily patching __module__
    before re-saving causes pickle to record the standalone module name instead,
    so the submission needs no z3alpha reference at inference time.

    Runs as a subprocess under .venv/bin/python which has z3alpha and all deps.
    """
    venv_python = _venv_bin("python")
    if not Path(venv_python).is_file():
        sys.exit("venv not found (.venv312 or .venv); create one first")

    paths = [
        str(SUBMISSION_DIR / "selectors" / group / "selector.joblib")
        for group in SELECTOR_PINS
    ]
    inline = (
        "import sys, joblib\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import z3alpha.smt_select_infer as _z3sel\n"
        # Register standalone name so pickle can verify the save lookup.
        "sys.modules['smt_select'] = _z3sel\n"
        "from z3alpha.smt_select_infer import PairwiseSelector\n"
        "PairwiseSelector.__module__ = 'smt_select'\n"
        "for p in sys.argv[2:]:\n"
        "    sel = joblib.load(p); joblib.dump(sel, p)\n"
        "    print(f'  re-saved {p.split(\"/selectors/\")[1].split(\"/\")[0]}"
        "/selector.joblib (class: smt_select.PairwiseSelector)')\n"
    )
    subprocess.run(
        [str(venv_python), "-c", inline, str(REPO_ROOT)] + paths,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": f"{SUBMISSION_DIR / 'vendor'}:{SUBMISSION_DIR / 'lib'}",
        },
    )


def verify_meta_json(meta_path: Path, *, fixed: bool = False) -> list[str]:
    """Return list of problems with a selector meta.json."""
    errors = []
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"{meta_path}: {e}"]

    if fixed:
        if "logic" not in meta:
            errors.append(f"{meta_path}: missing 'logic'")
        if "fixed_strategy" not in meta:
            errors.append(f"{meta_path}: missing 'fixed_strategy'")
        elif not isinstance(meta["fixed_strategy"], list):
            errors.append(f"{meta_path}: fixed_strategy must be a list")
        return errors

    for key in ("logic", "shortlist", "strategy_cli"):
        if key not in meta:
            errors.append(f"{meta_path}: missing '{key}'")

    logic = meta.get("logic")
    shortlist = meta.get("shortlist", [])
    strategy_cli = meta.get("strategy_cli", {})
    if logic and shortlist:
        missing = set(shortlist) - set(strategy_cli)
        if missing:
            errors.append(f"{meta_path}: strategy_cli missing {sorted(missing)}")
    return errors


def verify():
    errors = []

    vendor_dir = SUBMISSION_DIR / "vendor"
    vendor_checks = [vendor_dir / pkg for pkg in ("numpy", "sklearn", "joblib", "scipy", "xgboost")]

    expected = (
        [SUBMISSION_DIR / "z3alpha2.py", SUBMISSION_DIR / "bin" / "z3"]
        + [SUBMISSION_DIR / "lib" / name for name in ("smt_select.py", "smtlib_features.py")]
        + vendor_checks
        + [
            SUBMISSION_DIR / "selectors" / group / fname
            for group in SELECTOR_PINS
            for fname in ("selector.joblib", "meta.json")
        ]
        + [
            SUBMISSION_DIR / "selectors" / group / "meta.json"
            for group in FIXED_LOGIC_CONFIGS
        ]
    )

    for path in expected:
        if path.exists():
            print(f"  ok  {path.relative_to(SUBMISSION_DIR)}")
        else:
            print(f"  MISSING  {path.relative_to(SUBMISSION_DIR)}")
            errors.append(path)

    for group in SELECTOR_PINS:
        meta_path = SUBMISSION_DIR / "selectors" / group / "meta.json"
        if meta_path.is_file():
            for msg in verify_meta_json(meta_path):
                print(f"  INVALID  {msg}")
                errors.append(meta_path)

    for group in FIXED_LOGIC_CONFIGS:
        meta_path = SUBMISSION_DIR / "selectors" / group / "meta.json"
        if meta_path.is_file():
            for msg in verify_meta_json(meta_path, fixed=True):
                print(f"  INVALID  {msg}")
                errors.append(meta_path)

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-vendor",
        action="store_true",
        help="Skip re-downloading/extracting vendor/ (use when only code or selectors changed)",
    )
    args = parser.parse_args()

    print("── Copying entry point ──────────────────────────────────────────")
    copy_entrypoint()

    print("\n── Copying inference library ────────────────────────────────────")
    copy_lib()

    if args.skip_vendor:
        vendor_dir = SUBMISSION_DIR / "vendor"
        if not vendor_dir.is_dir() or not (vendor_dir / "xgboost").is_dir():
            sys.exit("vendor/ missing or incomplete; run without --skip-vendor first")
        print("\n── Vendoring Python dependencies ────────────────────────────────")
        print("  skipped (--skip-vendor)")
    else:
        print("\n── Vendoring Python dependencies ────────────────────────────────")
        vendor_packages()

    print("\n── Copying pinned selectors ─────────────────────────────────────")
    copy_selectors()

    print("\n── Writing fixed-logic configs ──────────────────────────────────")
    copy_fixed_logics()

    print("\n── Repackaging selectors (fixing pickle module path) ────────────")
    repackage_selectors()

    print("\n── Verifying submission directory ───────────────────────────────")
    errors = verify()

    if errors:
        sys.exit(f"\nSubmission incomplete: {len(errors)} file(s) missing.")
    else:
        print(f"\nSubmission ready at {SUBMISSION_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
