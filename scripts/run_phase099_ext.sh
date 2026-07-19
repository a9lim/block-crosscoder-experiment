#!/bin/bash
# Phase 0.9.9 extensions (a9, 2026-07-19 morning): complete the renorm
# x k frontier grid (k16/k64 — the ratified gauge's R-D curve), price
# both points, then run the E6 +6M-token store extension harvest
# (tranche 6 enabler). Waits for the codec sweep to release the GPU
# (no concurrent checkpoint loads — 2026-07-19 OOM lesson).
#
#   nohup bash scripts/run_phase099_ext.sh > /data/runs/bcc-phase099/campaign_ext.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-phase099

while pgrep -f "run_phase099_codec_sweep.sh" > /dev/null; do sleep 30; done
echo "[ext] GPU free (codec sweep drained)  $(date +%H:%M)"

run() {
  local name=$1; shift
  echo "[ext] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[ext] done $name  $(date +%H:%M)"
  else
    echo "[ext] EXIT-NONZERO $name (exit $?, see $OUT/$name.log)"
  fi
}

REH="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT
     --blocks 4096 --epochs 2 --guard --theta-method streaming --prefetch 4"
RD="python -u scripts/validate_rd_codec.py --store $STORE"

run f_renorm_lam0_k16 $REH --arm bsc --lam 0 --k 16 --seed 0 --site-renorm
run f_renorm_lam0_k64 $REH --arm bsc --lam 0 --k 64 --seed 0 --site-renorm
run rd_f_renorm_lam0_k16 $RD --site-renorm \
  --ckpt "$OUT/bsc_lam0_seed0_G4096_renorm_guard/latest.pt" \
  --out "$OUT/rd/f_renorm_lam0_k16.json"
run rd_f_renorm_lam0_k64 $RD --site-renorm \
  --ckpt "$OUT/bsc_lam0_seed0_G4096_k64_renorm_guard/latest.pt" \
  --out "$OUT/rd/f_renorm_lam0_k64.json"

# ---- E6: +6M-token corpus-disjoint extension under the frozen whitener ----
run e6_extend_store python -u scripts/extend_pilot4b_store.py --device cuda

echo "[ext] extensions complete $(date)"
