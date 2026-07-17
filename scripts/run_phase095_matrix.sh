#!/bin/bash
# Phase-0.9.5 calibration addendum, unconditional arms (design v2.3.2,
# sol-corrected matrix; ratified 2026-07-17). Runs on the existing 0.9
# store; --out-root is separate so 0.9 artifacts are never clobbered.
#
# A: BSC lr x schedule ladder (lambda=1e-3, seed 0), full 3906 steps.
# B: scalar mirror ladder (identical tuning budget, finding 16).
# C: dead-dynamics arm — G=4096 at k=32 preserves the Phase-1
#    k/G = 0.78% block-frequency ratio (sol).
# D: site-renorm arm (F7 pre-4b-store decision data).
#
# Conditional arms (encoder-decay at best two, second seeds for winner +
# runner-up, lambda=0 confirmation, k=16 stress) depend on the ladder
# reports and are printed at the end.
#
#   nohup bash scripts/run_phase095_matrix.sh > /data/runs/bcc-phase095/matrix.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
OUT=/data/runs/bcc-phase095
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

REHEARSE="python -u scripts/run_phase09_rehearsal.py --out-root $OUT"

# A: BSC ladder
for lr in 1e-4 2e-4 3e-4 6e-4; do
  for sched in cosine linear_fifth; do
    run "bsc_lr${lr}_${sched}" $REHEARSE --arm bsc --lam 1e-3 --lr "$lr" --schedule "$sched"
  done
done

# B: scalar mirror ladder
for lr in 1e-4 2e-4 3e-4 6e-4; do
  for sched in cosine linear_fifth; do
    run "scalar_lr${lr}_${sched}" $REHEARSE --arm scalar --lr "$lr" --schedule "$sched"
  done
done

# C: dead-dynamics arm
run dead_G4096_k32 $REHEARSE --arm bsc --lam 1e-3 --blocks 4096 --k 32

# D: site-renorm arm
run renorm $REHEARSE --arm bsc --lam 1e-3 --site-renorm

echo "MATRIX DONE — unconditional arms complete."
cat <<'NEXT'
Conditional arms (read the ladder reports first):
  encoder-decay at the best two BSC settings:
    ... --arm bsc --lam 1e-3 --lr <LR> --schedule <SCHED> --encoder-wd 1e-3
  second seed for winner + runner-up (BSC and scalar):
    ... --seed 1 [winning flags]
  lambda=0 confirmation of the BSC winner (H3 regime):
    ... --arm bsc --lam 0 --lr <LR> --schedule <SCHED>
  k=16 stress arm ONLY if dead_G4096_k32 produced no dead blocks:
    ... --arm bsc --lam 1e-3 --blocks 4096 --k 16
NEXT
