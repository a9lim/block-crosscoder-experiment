#!/bin/bash
# Pilot round 4: scalar theta-calib OOMs even at 64 batches (61.7 GB
# anon-rss, 16384 latents at 4b) — resume into calibration at 16;
# finish the arms round 3 dropped.
cd /home/jobe/Work/transformer-experiments/block-crosscoder-experiment || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-pilot4b
run() {
  local name=$1; shift
  echo "[round4] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[round4] done $name  $(date +%H:%M)"
  else
    echo "[round4] FAILED $name (exit $?, see $OUT/$name.log)"
    exit 1
  fi
}
SCA="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT --blocks 4096 --k 32 --lam 0 --epochs 2 --arm scalar --calib-batches 16"
run scalar_3e4_finish $SCA --lr 3e-4 --resume
run bsc_seed1_3e4 python -u scripts/run_phase09_rehearsal.py \
  --store $STORE --out-root $OUT --blocks 4096 --k 32 --lam 1e-3 \
  --epochs 2 --arm bsc --lr 3e-4 --seed 1
run scalar_6e4_diag $SCA --lr 6e-4
echo "ROUND 4 DONE"
