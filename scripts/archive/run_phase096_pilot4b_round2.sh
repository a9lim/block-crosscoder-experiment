#!/bin/bash
# Pilot round 2 (2026-07-17): lr 1.2e-3 is over the 4b stability cliff
# (baseline spike at ~8e-4 during warmup, renorm at ~6.5e-4). Re-run the
# training arms at the pre-ratified 6e-4 fallback; scalar-at-1.2e-3 last
# as the is-it-BSC-specific diagnostic.
cd /home/jobe/Work/transformer-experiments/block-crosscoder-experiment || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-pilot4b
run() {
  local name=$1; shift
  echo "[round2] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[round2] done $name  $(date +%H:%M)"
  else
    echo "[round2] FAILED $name (exit $?, see $OUT/$name.log)"
    exit 1
  fi
}
BASE="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT --blocks 4096 --k 32 --lam 1e-3 --epochs 2"
run bsc_primary_6e4 $BASE --arm bsc --lr 6e-4
run bsc_renorm_6e4 $BASE --arm bsc --lr 6e-4 --site-renorm
run scalar_6e4 $BASE --arm scalar --lr 6e-4 --calib-batches 64
run bsc_seed1_6e4 $BASE --arm bsc --lr 6e-4 --seed 1
run scalar_lr12_diag $BASE --arm scalar --lr 1.2e-3 --calib-batches 64
echo "ROUND 2 DONE"
