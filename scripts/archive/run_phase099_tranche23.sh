#!/bin/bash
# Phase 0.9.9 tranches 2+3, GPU queue (launched after the tranche-1
# validation campaign drains).
#
# Codec validation (analysis-only, ~10 min each): the preregistered R-D
# codec on the three ratified pilot checkpoints — the first honest
# bits-vs-distortion numbers behind the 0.430/0.415/0.368 FVU story.
# NB the pilot bsc arms are lambda=1e-3 (ratified), the scalar is
# lambda=0 — these are H3 *preview* points; the frontier proper is
# lambda=0 both arms (later runs).
#
# Tranche 2 (one store pass each, ~2h each): the factorial's single-site
# cells at seed 0 — bsf = {block, single-site}, sae = {scalar,
# single-site} — exact per-site matching to the joint arms, dogfooding
# E4 (subset view) + E5 (prefetch) + E2 guard + E1 streaming theta.
#
#   nohup bash scripts/run_phase099_tranche23.sh > /data/runs/bcc-phase099/tranche23.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
PILOT=/data/runs/bcc-pilot4b
OUT=/data/runs/bcc-phase099
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "[t23] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[t23] done $name  $(date +%H:%M)"
  else
    echo "[t23] EXIT-NONZERO $name (exit $?, see $OUT/$name.log)"
  fi
}

# ---- R-D codec on the ratified pilot checkpoints ---------------------------
run rd_bsc_primary python -u scripts/validate_rd_codec.py \
  --ckpt "$PILOT/bsc_lam0.001_seed0_G4096_k32/latest.pt" \
  --store "$STORE" --out "$OUT/rd_bsc_primary.json"
run rd_bsc_renorm python -u scripts/validate_rd_codec.py \
  --ckpt "$PILOT/bsc_lam0.001_seed0_G4096_k32_renorm/latest.pt" \
  --store "$STORE" --site-renorm --out "$OUT/rd_bsc_renorm.json"
run rd_scalar python -u scripts/validate_rd_codec.py \
  --ckpt "$PILOT/scalar_lam0_seed0_G4096_k32/latest.pt" \
  --store "$STORE" --out "$OUT/rd_scalar.json"

# ---- tranche-2 factorial cells (seed 0) ------------------------------------
SS="python -u scripts/run_phase099_single_site.py --store $STORE --out-root $OUT
    --blocks 4096 --k 32 --guard --prefetch 4"
run t2_bsf $SS --arm bsf
run t2_sae $SS --arm sae

echo "[t23] queue complete $(date)"
