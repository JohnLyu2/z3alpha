"""Validate z3alpha/tactics/logic_configs/*.json against z3 and the tactic catalog.

For every tactic name and param declared in the logic config JSONs, checks:
  - the tactic name is registered in z3alpha.tactics.catalog.NAME_TO_ID (else
    z3alpha.tactics.logic_config.load_logic_config would raise KeyError)
  - the tactic name is a real z3 tactic (z3.tactics())
  - every declared param name is accepted by that z3 tactic
    (z3.Tactic(name).param_descrs())
  - the declared "default" matches the default z3 itself reports. The
    Python API's param_descrs() doesn't expose default values, so this is
    parsed from `z3 -tactics:<name>` CLI output instead.

Usage:
    python scripts/validate_logic_configs.py [--z3-path PATH] [--fix]

With --fix, mismatched/missing "default" values are rewritten in place to
match what z3 reports. Unknown tactic/param names are never auto-fixed
since there's no single correct fix to apply.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import z3

from z3alpha.tactics.catalog import NAME_TO_ID

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "z3alpha" / "tactics" / "logic_configs"
_DEFAULT_RE = re.compile(r"\(default:\s*([^)]+)\)")
_PARAM_LINE_RE = re.compile(r"^(\S+)\s*\(")


def _z3_param_defaults(z3_path: str, tactic: str) -> dict[str, str]:
    """Run `z3 -tactics:<tactic>` and parse {param_name: default_str_as_z3_prints_it}."""
    out = subprocess.run(
        [z3_path, f"-tactics:{tactic}"], capture_output=True, text=True, timeout=30
    ).stdout
    defaults: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        m = _PARAM_LINE_RE.match(line)
        if not m:
            continue
        dm = _DEFAULT_RE.search(line)
        if dm:
            defaults[m.group(1)] = dm.group(1).strip()
    return defaults


def _valid_params(tactic: str) -> set[str]:
    pd = z3.Tactic(tactic).param_descrs()
    return {pd.get_name(i) for i in range(pd.size())}


def _normalize(value) -> str:
    """Render a JSON default value the way z3's CLI default string looks."""
    if isinstance(value, bool):
        raise TypeError('tactic params use the strings "true"/"false", not JSON booleans')
    return str(value)


def _denormalize(z3_default: str):
    """Inverse of _normalize: turn a z3-reported default string back into a JSON value."""
    if re.fullmatch(r"-?\d+", z3_default):
        return int(z3_default)
    return z3_default  # "true" / "false" / "inf" / etc. stay as strings


def _compact_dump(obj) -> str:
    """json.dumps with short scalar-only arrays collapsed onto one line."""
    s = json.dumps(obj, indent=2)
    pattern = re.compile(r"\[\n(?:\s*(?:\"[^\"]*\"|-?\d+),?\n)+\s*\]")
    while True:
        new_s = pattern.sub(
            lambda m: "["
            + ", ".join(x.strip().rstrip(",") for x in m.group(0).strip("[]").splitlines() if x.strip())
            + "]",
            s,
        )
        if new_s == s:
            return s
        s = new_s


def validate(z3_path: str, fix: bool) -> int:
    valid_tactics = set(z3.tactics())
    param_name_cache: dict[str, set[str]] = {}
    default_cache: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    for path in sorted(_CONFIG_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        changed = False

        for tactic_name, entry in raw.items():
            loc = f"{path.name}:{tactic_name}"
            if tactic_name not in NAME_TO_ID:
                errors.append(f"{loc}: not registered in z3alpha.tactics.catalog.NAME_TO_ID")
            if tactic_name not in valid_tactics:
                errors.append(f"{loc}: not a real z3 tactic (see z3.tactics())")
                continue  # can't check params/defaults for a tactic z3 doesn't know

            params = entry.get("params")
            if not params:
                continue

            if tactic_name not in param_name_cache:
                param_name_cache[tactic_name] = _valid_params(tactic_name)
            valid_params = param_name_cache[tactic_name]

            for pname, spec in params.items():
                ploc = f"{loc}.{pname}"
                if pname not in valid_params:
                    errors.append(f"{ploc}: not a valid param for tactic {tactic_name!r}")
                    continue

                if tactic_name not in default_cache:
                    default_cache[tactic_name] = _z3_param_defaults(z3_path, tactic_name)
                z3_default = default_cache[tactic_name].get(pname)
                if z3_default is None:
                    errors.append(f"{ploc}: could not determine z3's default (CLI parse miss)")
                    continue

                declared = _normalize(spec["default"])
                if declared != z3_default:
                    msg = f"{ploc}: declared default {declared!r} != z3 default {z3_default!r}"
                    if fix:
                        spec["default"] = _denormalize(z3_default)
                        changed = True
                        print(f"FIXED: {msg}")
                    else:
                        errors.append(msg)

        if changed:
            path.write_text(_compact_dump(raw) + "\n")

    if errors:
        print(f"{len(errors)} problem(s) found:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("OK: all tactic names, param names, and defaults match z3.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--z3-path", default=shutil.which("z3"), help="Path to the z3 binary (default: $PATH)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite mismatched/missing default values in place to match z3",
    )
    args = parser.parse_args()
    if not args.z3_path:
        parser.error("z3 binary not found on PATH; pass --z3-path")
    sys.exit(validate(args.z3_path, args.fix))


if __name__ == "__main__":
    main()
