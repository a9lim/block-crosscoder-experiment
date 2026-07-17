#!/bin/bash
# Phase 0.9.6 tier A: 1b consolidation & robustness matrix on the
# existing 0.9 store. All arms at the ratified winner optimizer
# (lr 1.2e-3 cosine, lambda=1e-3, enc-wd 0) unless tagged otherwise.
#
# Questions, per docs/runbook-phase096.md:
#   A1 seeds 2-5            — is single-block ring consolidation the rule?
#   A2 epoch ladder         — does more optimization consolidate seed 1 /
#                             merge the lr-3e-4 two-block split?
#   A3 renorm @ winner lr   — F7 deconfound (interim analysis flagged the
#                             renorm-vs-lr confound at 3e-4)
#   A4 G ladder 2048/8192   — consolidation vs splitting vs packing onset
#                             (8192 = Phase-1 stretch ratio at k=32)
#
#   nohup bash scripts/run_phase096_tier_a.sh > /data/runs/bcc-phase096/matrix.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
OUT=/data/runs/bcc-phase096
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "[096a] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[096a] done $name  $(date +%H:%M)"
  else
    echo "[096a] FAILED $name (exit $?, see $OUT/$name.log)"
    exit 1
  fi
}

WINNER="python -u scripts/run_phase09_rehearsal.py --out-root $OUT --arm bsc --lam 1e-3 --lr 1.2e-3"

# A1: seed fan-out at the winner config (~4 min each)
for seed in 2 3 4 5; do
  run "seed${seed}" $WINNER --seed "$seed"
done

# A3: renorm at the winner lr (F7 deconfound)
run renorm_lr12 $WINNER --site-renorm

# A4: G ladder at the winner optimizer
run G2048 $WINNER --blocks 2048
run G8192_k32 $WINNER --blocks 8192 --k 32

# A2: epoch ladder (4 and 8 passes over the 8M train split; ~8/16 min each)
run ep4_seed0 $WINNER --epochs 4
run ep8_seed0 $WINNER --epochs 8
run ep4_seed1 $WINNER --seed 1 --epochs 4
run ep8_seed1 $WINNER --seed 1 --epochs 8
run ep8_lr3e-4 python -u scripts/run_phase09_rehearsal.py \
  --out-root "$OUT" --arm bsc --lam 1e-3 --epochs 8

echo "TIER A DONE — reports under $OUT/*/report.json"
echo "Next: calendar probe + planarity screen across all new checkpoints"
echo "(see docs/runbook-phase096.md, Analysis pass)."
