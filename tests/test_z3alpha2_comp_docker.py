"""Optional smoke tests inside the SMT-COMP competition Docker image.

Set RUN_COMP_DOCKER=1 to enable (requires docker or podman and a built submission tree).

    RUN_COMP_DOCKER=1 python -m unittest tests.test_z3alpha2_comp_docker

Optional:
    COMP_DOCKER_PULL=1       pull the image before running
    COMP_DOCKER_IMAGE=...    override the default competition user image
    SUBMISSION_DIR=...       override submission path
"""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _PROJECT_ROOT / "data" / "smtcomp26" / "scripts" / "test_submission_in_comp_docker.sh"
_SUBMISSION = _PROJECT_ROOT / "data" / "smtcomp26" / "submission"
_Z3 = _SUBMISSION / "bin" / "z3"


def _comp_docker_enabled() -> bool:
    return os.environ.get("RUN_COMP_DOCKER") == "1"


@unittest.skipUnless(_comp_docker_enabled(), "set RUN_COMP_DOCKER=1 to run competition Docker smoke tests")
@unittest.skipUnless(_SCRIPT.is_file(), f"missing {_SCRIPT}")
@unittest.skipUnless(_Z3.is_file(), "submission bin/z3 not found")
class TestZ3alpha2CompDocker(unittest.TestCase):
    def test_submission_in_competition_image(self) -> None:
        env = os.environ.copy()
        proc = subprocess.run(
            ["bash", str(_SCRIPT)],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        combined = proc.stdout
        if proc.stderr:
            combined = f"{combined}\n--- stderr ---\n{proc.stderr}" if combined else proc.stderr
        self.assertEqual(
            proc.returncode,
            0,
            msg=combined or f"competition docker smoke test failed with code {proc.returncode}",
        )
        self.assertIn("all competition docker smoke tests passed", combined)


if __name__ == "__main__":
    unittest.main()
