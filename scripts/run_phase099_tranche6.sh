#!/bin/bash
# Phase 0.9.9 tranche 6 — the epochs-vs-fresh token-budget factorial
# (runbook §Tranche 6): 6M unique x 4 epochs vs 12M unique x 2 epochs
# ({train, train12m} splits), x {primary, renorm}, all at matched 24M
# optimizer tokens (5856 steps of the same cosine shape). Every run
# carries the full v2.4 pinned stack (lr 3e-4 cosine, lam 1e-3, guard,
# streaming theta, prefetch 4, aux-ratio-cap 1.0) — the campaign
# doubles as the longest-duration dogfood of the exact Phase-1 config.
# Codec passes at the tail price all four cells at fixed q.
#
#   nohup bash scripts/run_phase099_tranche6.sh > /data/runs/bcc-phase099/campaign_t6.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-phase099

run() {
  local name=$1; shift
  echo "[t6] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[t6] done $name  $(date +%H:%M)"
  else
    echo "[t6] EXIT-NONZERO $name (exit $?, see $OUT/$name.log)"
  fi
}

REH="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT
     --blocks 4096 --lam 1e-3 --seed 0 --guard --theta-method streaming
     --prefetch 4 --aux-ratio-cap 1.0"
RD="python -u scripts/validate_rd_codec.py --store $STORE"

# Cell A: 6M unique x 4 epochs (optimization-rich, data-poor)
run t6_ep4_primary $REH --arm bsc --epochs 4
run t6_ep4_renorm  $REH --arm bsc --epochs 4 --site-renorm

# Cell B: 12M unique x 2 epochs (data-rich, matched optimizer tokens)
run t6_12m_primary $REH --arm bsc --train-split train12m
run t6_12m_renorm  $REH --arm bsc --train-split train12m --site-renorm

# R-D positions for all four cells
run rd_t6_ep4_primary $RD \
  --ckpt "$OUT/bsc_lam0.001_seed0_G4096_ep4_guard_rcap1/latest.pt" \
  --out "$OUT/rd/t6_ep4_primary.json"
run rd_t6_ep4_renorm $RD --site-renorm \
  --ckpt "$OUT/bsc_lam0.001_seed0_G4096_renorm_ep4_guard_rcap1/latest.pt" \
  --out "$OUT/rd/t6_ep4_renorm.json"
run rd_t6_12m_primary $RD \
  --ckpt "$OUT/bsc_lam0.001_seed0_G4096_train12m_guard_rcap1/latest.pt" \
  --out "$OUT/rd/t6_12m_primary.json"
run rd_t6_12m_renorm $RD --site-renorm \
  --ckpt "$OUT/bsc_lam0.001_seed0_G4096_renorm_train12m_guard_rcap1/latest.pt" \
  --out "$OUT/rd/t6_12m_renorm.json"

echo "[t6] tranche 6 complete $(date)"
