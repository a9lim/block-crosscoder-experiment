#!/bin/bash
# Phase-0.9 rehearsal matrix: verify the store, then train all four arms
# sequentially (they share the GPU). The first arm exercises the resume
# gate at production scale: --max-steps 500, then --resume to finish.
#
#   nohup bash scripts/run_phase09_matrix.sh > /data/runs/bcc-phase09/matrix.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
OUT=/data/runs/bcc-phase09
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "[matrix] start $name"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[matrix] done $name"
  else
    echo "[matrix] FAILED $name (exit $?, see $OUT/$name.log)"
    exit 1
  fi
}

run verify      python -u scripts/verify_phase09_store.py
run bsc_lam0_a  python -u scripts/run_phase09_rehearsal.py --arm bsc --lam 0 --max-steps 500
run bsc_lam0_b  python -u scripts/run_phase09_rehearsal.py --arm bsc --lam 0 --resume
run bsc_lam3e-4 python -u scripts/run_phase09_rehearsal.py --arm bsc --lam 3e-4
run bsc_lam1e-3 python -u scripts/run_phase09_rehearsal.py --arm bsc --lam 1e-3
run scalar      python -u scripts/run_phase09_rehearsal.py --arm scalar
echo "MATRIX DONE"
