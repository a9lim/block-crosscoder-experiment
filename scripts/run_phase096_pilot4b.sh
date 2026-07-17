#!/bin/bash
# Phase 0.9.6 tier B: the D13 4b pilot on the existing /data disk.
# (a9 2026-07-17: tiers bundled as 0.9.6; pilot runs pre-NVMe — ~290 GB
# stored vs ~600 GB usable above the ShardWriter floor.)
#
# Sequence: harvest (the long pole, hours) -> store verify -> BSC primary
# with the checkpoint/resume gate split -> renorm arm -> matched scalar.
# Exact Phase-1 config: G=4096 x b=4, 8 sites (9,12,15,18,21,24,27,30),
# k=32, lr 1.2e-3 cosine, lambda=1e-3, SASA AuxK (trainer default),
# theta calibrated on the calibration split.
#
#   nohup bash scripts/run_phase096_pilot4b.sh > /data/runs/bcc-pilot4b/pilot.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-pilot4b
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "[pilot] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[pilot] done $name  $(date +%H:%M)"
  else
    echo "[pilot] FAILED $name (exit $?, see $OUT/$name.log)"
    exit 1
  fi
}

# 1. Harvest (skipped if the store manifest already exists — idempotent
#    relaunch after a mid-harvest failure requires clearing the store dir).
if [ ! -f "$STORE/train/split.json" ]; then
  run harvest python -u scripts/harvest_pilot4b_store.py
else
  echo "[pilot] store exists, skipping harvest"
fi

# 2. Store verification (checksums + whitening round trip).
run verify python -u scripts/verify_phase09_store.py --store "$STORE"

REHEARSE="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT --blocks 4096 --k 32 --lam 1e-3 --lr 1.2e-3"

# 3. BSC primary, split across the checkpoint/resume gate (D13): stop at
#    step 900 of 1953, then resume to completion bit-compatibly.
run bsc_primary_part1 $REHEARSE --arm bsc --max-steps 900
run bsc_primary_resume $REHEARSE --arm bsc --resume

# 4. F7 renorm arm (same config + per-site RMS renorm).
run bsc_renorm $REHEARSE --arm bsc --site-renorm

# 5. Matched scalar baseline (16384 latents, k=128; smaller calib cap —
#    the pooled score matrix at 128 batches would be ~34 GB host-side).
run scalar python -u scripts/run_phase09_rehearsal.py \
  --store "$STORE" --out-root "$OUT" --blocks 4096 --k 32 --lr 1.2e-3 \
  --arm scalar --calib-batches 64

echo "PILOT DONE — reports under $OUT/*/report.json"
echo "Next: calendar probe + planarity screen against these checkpoints"
echo "(see docs/runbook-phase096.md, Analysis pass)."
