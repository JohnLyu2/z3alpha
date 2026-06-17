"""Machine-local environment config loaded from env_config.json at the repo root."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORKERS = 4

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EnvConfig:
    workers: int = DEFAULT_WORKERS
    z3_path: str | None = None
    z3_version: str | None = None
    machine_name: str | None = None


def load_env_config(path: str | Path | None = None) -> EnvConfig:
    """Load env_config.json; returns defaults for any missing field or missing file."""
    p = Path(path) if path is not None else _REPO_ROOT / "env_config.json"
    if not p.is_file():
        return EnvConfig()
    raw = json.loads(p.read_text())
    raw_workers = raw.get("workers", DEFAULT_WORKERS)
    try:
        workers = int(raw_workers)
    except (TypeError, ValueError):
        raise ValueError(f"env_config.json: 'workers' must be an integer, got {raw_workers!r}")
    return EnvConfig(
        workers=workers,
        z3_path=raw.get("z3_path"),
        z3_version=raw.get("z3_version"),
        machine_name=raw.get("machine_name"),
    )


def check_z3_version(env: EnvConfig) -> None:
    """If z3_version is set, verify the z3 binary reports that version.

    Raises RuntimeError on mismatch or if the binary cannot be found.
    """
    if env.z3_version is None:
        return
    z3 = env.z3_path or "z3"
    try:
        out = subprocess.check_output([z3, "--version"], text=True).strip()
    except FileNotFoundError:
        raise RuntimeError(f"z3 binary not found: {z3!r}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"z3 --version failed: {e}")
    if env.z3_version not in out:
        raise RuntimeError(
            f"z3 version mismatch: expected {env.z3_version!r}, got {out!r}"
        )
