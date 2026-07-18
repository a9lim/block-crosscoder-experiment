#!/bin/bash
# Pilot round 3 (2026-07-17): 6e-4 sits ON the 4b late-training stability
# edge (baseline spikes at peak, renorm destroyed at 6e-4). Clean
# baseline set at 3e-4 + the scalar cascade diagnostics (lam 0 — the
# runner refuses a rank penalty at b=1; round-2 scalar failed on that).
cd /home/jobe/Work/transformer-experiments/block-crosscoder-experiment || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-pilot4b
run() {
  local name=$1; shift
  echo "[round3] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[round3] done $name  $(date +%H:%M)"
  else
    echo "[round3] FAILED $name (exit $?, see $OUT/$name.log)"
    exit 1
  fi
}
BSC="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT --blocks 4096 --k 32 --lam 1e-3 --epochs 2 --arm bsc --lr 3e-4"
SCA="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT --blocks 4096 --k 32 --lam 0 --epochs 2 --arm scalar --calib-batches 64"
run bsc_primary_3e4 $BSC
run bsc_renorm_3e4 $BSC --site-renorm
run scalar_3e4 $SCA --lr 3e-4
run bsc_seed1_3e4 $BSC --seed 1
run scalar_6e4_diag $SCA --lr 6e-4
echo "ROUND 3 DONE"
