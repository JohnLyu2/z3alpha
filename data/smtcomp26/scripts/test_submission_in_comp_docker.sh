#!/usr/bin/env bash
# Smoke-test z3alpha2 inside the SMT-COMP competition user image.
#
#   RUN_COMP_DOCKER=1 ./data/smtcomp26/scripts/test_submission_in_comp_docker.sh
#   python -m unittest tests.test_z3alpha2_comp_docker   # also needs RUN_COMP_DOCKER=1
#
# Requires SMT-LIB benchmarks: set SMTLIB_ROOT or configure env_config.json smtlib_root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SUBMISSION="${SUBMISSION_DIR:-$REPO_ROOT/data/smtcomp26/submission}"
BENCH="${BENCH_DIR:-$REPO_ROOT/data/smoke/benchmarks}"
IMAGE="${COMP_DOCKER_IMAGE:-registry.gitlab.com/sosy-lab/benchmarking/competition-scripts/user:latest}"

if [[ ! -f "$SUBMISSION/z3alpha2.py" ]]; then
  echo "submission not found: $SUBMISSION/z3alpha2.py" >&2
  exit 1
fi
if [[ ! -x "$SUBMISSION/bin/z3" ]]; then
  echo "submission Z3 not found or not executable: $SUBMISSION/bin/z3" >&2
  exit 1
fi
if [[ ! -f "$BENCH/cases.json" ]]; then
  echo "benchmark cases not found: $BENCH/cases.json" >&2
  exit 1
fi

SMTLIB_ROOT="${SMTLIB_ROOT:-}"
if [[ -z "$SMTLIB_ROOT" && -f "$REPO_ROOT/env_config.json" ]]; then
  SMTLIB_ROOT="$(python3 -c "import json; print(json.load(open('$REPO_ROOT/env_config.json')).get('smtlib_root',''))")"
fi
if [[ -z "$SMTLIB_ROOT" || ! -d "$SMTLIB_ROOT" ]]; then
  echo "SMT-LIB root not found; set SMTLIB_ROOT or env_config.json smtlib_root" >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1; then
  RUNTIME=docker
elif command -v podman >/dev/null 2>&1; then
  RUNTIME=podman
else
  echo "need docker or podman on PATH" >&2
  exit 1
fi

if [[ "${COMP_DOCKER_PULL:-0}" == "1" ]]; then
  echo "pulling $IMAGE ..."
  "$RUNTIME" pull "$IMAGE"
fi

SUBMISSION="$(cd "$SUBMISSION" && pwd)"
BENCH="$(cd "$BENCH" && pwd)"
SMTLIB_ROOT="$(cd "$SMTLIB_ROOT" && pwd)"

echo "runtime=$RUNTIME image=$IMAGE"
echo "submission=$SUBMISSION"
echo "benchmarks=$BENCH"
echo "smtlib_root=$SMTLIB_ROOT"

"$RUNTIME" run --rm -i \
  --volume="$SUBMISSION:/tool:ro" \
  --volume="$BENCH:/bench:ro" \
  --volume="$SMTLIB_ROOT:/smtlib25:ro" \
  --workdir=/tool \
  -e SMTLIB_ROOT=/smtlib25 \
  "$IMAGE" \
  bash -s <<'INNER'
set -euo pipefail

echo "=== environment ==="
python3 --version
test -x /tool/bin/z3
test -x /tool/z3alpha2.py

echo "=== logic behavior cases (SMT-LIB) ==="
python3 /bench/run_cases.py /tool /bench

echo "=== all competition docker smoke tests passed ==="
INNER
