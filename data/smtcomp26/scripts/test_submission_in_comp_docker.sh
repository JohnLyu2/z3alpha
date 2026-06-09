#!/usr/bin/env bash
# Smoke-test z3alpha2 inside the SMT-COMP competition user image.
#
#   RUN_COMP_DOCKER=1 ./data/smtcomp26/scripts/test_submission_in_comp_docker.sh
#   python -m unittest tests.test_z3alpha2_comp_docker   # also needs RUN_COMP_DOCKER=1
#
# Image docs: https://gitlab.com/sosy-lab/benchmarking/competition-scripts/-/blob/main/README.md

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

echo "runtime=$RUNTIME image=$IMAGE"
echo "submission=$SUBMISSION"
echo "benchmarks=$BENCH"

"$RUNTIME" run --rm -i \
  --volume="$SUBMISSION:/tool:ro" \
  --volume="$BENCH:/bench:ro" \
  --workdir=/tool \
  "$IMAGE" \
  bash -s <<'INNER'
set -euo pipefail

run_solver() {
  local bench="$1"
  local expect="${2:-}"
  local out rc
  out="$(python3 /tool/z3alpha2.py "$bench" 2>&1)" || rc=$?
  rc="${rc:-0}"
  local verdict
  verdict="$(printf '%s\n' "$out" | tail -1)"
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL $bench exit=$rc" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  if [[ "$verdict" != sat && "$verdict" != unsat ]]; then
    echo "FAIL $bench unexpected output: $verdict" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  if [[ -n "$expect" && "$verdict" != "$expect" ]]; then
    echo "FAIL $bench expected $expect got $verdict" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  echo "OK $bench -> $verdict"
}

echo "=== environment ==="
python3 --version
test -x /tool/bin/z3
test -x /tool/z3alpha2.py

echo "=== direct Z3 ==="
verdict="$(/tool/bin/z3 /bench/lia/jain_7_true-unreach-call_true-no-overflow_false-termination.i_10.smt2 | head -1)"
test "$verdict" = unsat
echo "OK direct z3 -> $verdict"

echo "=== LIA fixed strategy ==="
out="$(python3 /tool/z3alpha2.py --debug /bench/lia/jain_7_true-unreach-call_true-no-overflow_false-termination.i_10.smt2 2>&1)"
printf '%s\n' "$out" | grep -q 'logic=LIA'
printf '%s\n' "$out" | grep -q 'fixed_strategy='
run_solver /bench/lia/jain_7_true-unreach-call_true-no-overflow_false-termination.i_10.smt2 unsat

echo "=== LRA selector ==="
out="$(python3 /tool/z3alpha2.py --debug /bench/lra/formula_014.smt2 2>&1)"
printf '%s\n' "$out" | grep -q 'logic=LRA'
printf '%s\n' "$out" | grep -q 'ranked='
run_solver /bench/lra/formula_014.smt2 sat

echo "=== QF_NIA presolver ==="
out="$(python3 /tool/z3alpha2.py --debug /bench/0.smt2 2>&1)"
printf '%s\n' "$out" | grep -q 'presolver solved'
run_solver /bench/0.smt2 sat

echo "=== registry ==="
python3 - <<'PY'
import importlib.util
import sys
from pathlib import Path

sys.path[:0] = ["/tool/vendor", "/tool/lib"]
spec = importlib.util.spec_from_file_location("z3alpha2", "/tool/z3alpha2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
registry = mod.load_selector_registry(Path("/tool/selectors"))
required = [
    "LIA", "LRA", "QF_NIA", "QF_NRA", "QF_BV", "QF_LIA", "QF_IDL",
]
missing = [logic for logic in required if logic not in registry]
if missing:
    raise SystemExit(f"missing logics: {missing}")
print("OK registry:", ", ".join(sorted(registry)))
PY

cat >/tmp/qf_lia.smt2 <<'EOF'
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (> x 0))
(check-sat)
EOF
cat >/tmp/qf_idl.smt2 <<'EOF'
(set-logic QF_IDL)
(declare-fun x () Int)
(assert (<= x 5))
(check-sat)
EOF
cat >/tmp/qf_bv.smt2 <<'EOF'
(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(assert (bvugt x #x00))
(check-sat)
EOF

echo "=== QF presolver logics ==="
for f in /tmp/qf_lia.smt2 /tmp/qf_idl.smt2 /tmp/qf_bv.smt2; do
  out="$(python3 /tool/z3alpha2.py --debug "$f" 2>&1)"
  printf '%s\n' "$out" | grep -q 'presolver solved'
  run_solver "$f" sat
done

echo "=== all competition docker smoke tests passed ==="
INNER
