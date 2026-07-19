#!/bin/bash
# Phase 0.9.9 codec sweep: R-D positions for every tranche-3/4
# checkpoint (launched after run_phase099_tranche34.sh drains).
#
# Joint arms: one codec pass each (~5 min). Single-site cells: one
# pass per site with --sites L; per-cell plane placement sums per-site
# rates and pools FVU with store sq_tot weights at combine time.
#
#   nohup bash scripts/run_phase099_codec_sweep.sh > /data/runs/bcc-phase099/codecsweep.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-phase099
RD="python -u scripts/validate_rd_codec.py --store $STORE"

run() {
  local name=$1; shift
  if [ -f "$OUT/rd/$name.json" ]; then
    echo "[rd] skip $name (exists)"
    return
  fi
  echo "[rd] start $name  $(date +%H:%M)"
  if "$@" --out "$OUT/rd/$name.json" > "$OUT/rd/$name.log" 2>&1; then
    echo "[rd] done $name  $(date +%H:%M)"
  else
    echo "[rd] EXIT-NONZERO $name (exit $?, see $OUT/rd/$name.log)"
  fi
}
mkdir -p "$OUT/rd"

# ---- joint checkpoints: frontier + headline seeds --------------------------
joint() {  # name ckpt-dir [extra args]
  local name=$1 dir=$2; shift 2
  run "$name" $RD --ckpt "$dir/latest.pt" "$@"
}
joint f_bsc_lam0_k32       "$OUT/bsc_lam0_seed0_G4096_k32_guard"
# NB k=16 equals the rehearsal script's K default, so those run names
# carry no _k tag (bsc_lam0_seed0_G4096_guard is the k16 run).
joint f_bsc_lam0_k16       "$OUT/bsc_lam0_seed0_G4096_guard"
joint f_bsc_lam0_k64       "$OUT/bsc_lam0_seed0_G4096_k64_guard"
joint f_scalar_lam0_k16    "$OUT/scalar_lam0_seed0_G4096_guard"
joint f_scalar_lam0_k64    "$OUT/scalar_lam0_seed0_G4096_k64_guard"
joint f_bsc_lam0_k32_renorm "$OUT/bsc_lam0_seed0_G4096_k32_renorm_guard" --site-renorm
joint f_bsc_lam0_k32_seed1 "$OUT/bsc_lam0_seed1_G4096_k32_guard"
joint s_scalar_lam0_k32_seed1 "$OUT/scalar_lam0_seed1_G4096_k32_guard"
joint s_scalar_lam0_k32_seed2 "$OUT/scalar_lam0_seed2_G4096_k32_guard"
joint s_renorm_seed1       "$OUT/bsc_lam0.001_seed1_G4096_k32_renorm_guard" --site-renorm
joint s_renorm_seed2       "$OUT/bsc_lam0.001_seed2_G4096_k32_renorm_guard" --site-renorm
joint s_bsc_seed2          "$OUT/bsc_lam0.001_seed2_G4096_k32_guard"

# ---- single-site cells (seed 0), one pass per site -------------------------
for L in 9 12 15 18 21 24 27 30; do
  run "ss_bsf_seed0_site$L" $RD --ckpt "$OUT/bsf_lam0.001_seed0/site$L/latest.pt" --sites $L
done
for L in 9 12 15 18 21 24 27 30; do
  run "ss_sae_seed0_site$L" $RD --ckpt "$OUT/sae_lam0_seed0/site$L/latest.pt" --sites $L
done

echo "[rd] codec sweep complete $(date)"
