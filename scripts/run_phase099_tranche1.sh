#!/bin/bash
# Phase 0.9.9 tranche 1 validation campaign (runbook-phase099).
#
# E1: theta streaming-vs-exact on the three ratified pilot checkpoints
#     (the scalar arm is the 61 GB OOM case — exact capped at 16 batches,
#     streaming over the full 2M-token split).
# E2: guard regression suite at the known-failure operating points —
#     R2 3e-4 false-positive control (gate: zero skips), R1 6e-4 warmup
#     spike (gate: caught), R3 1.2e-3 catastrophe (gate: the run DIES on
#     the consecutive-skip RuntimeError — that crash is its pass
#     condition; the guard refuses to censor an unstable operating point).
# E3: cap candidates at 6e-4 where the AuxK cascade reproduces — R4
#     ratio cap, R5 frac cap, R6 renorm (hardest spike, rec 94.5) + ratio
#     cap. All runs theta-method streaming (dogfooding E1).
#
#   nohup bash scripts/run_phase099_tranche1.sh > /data/runs/bcc-phase099/campaign.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
PILOT=/data/runs/bcc-pilot4b
OUT=/data/runs/bcc-phase099
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "[t1] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[t1] done $name  $(date +%H:%M)"
  else
    echo "[t1] EXIT-NONZERO $name (exit $?, see $OUT/$name.log)"
  fi
}

# ---- E1 (analysis-only, minutes each) --------------------------------------
run e1_primary python -u scripts/validate_e1_theta.py \
  --ckpt "$PILOT/bsc_lam0.001_seed0_G4096_k32/latest.pt" \
  --store "$STORE" --out "$OUT/e1_primary.json"
run e1_renorm python -u scripts/validate_e1_theta.py \
  --ckpt "$PILOT/bsc_lam0.001_seed0_G4096_k32_renorm/latest.pt" \
  --store "$STORE" --site-renorm --out "$OUT/e1_renorm.json"
run e1_scalar python -u scripts/validate_e1_theta.py \
  --ckpt "$PILOT/scalar_lam0_seed0_G4096_k32/latest.pt" \
  --store "$STORE" --exact-batches 16 --out "$OUT/e1_scalar.json"

# ---- E2/E3 trainer runs (~35 min each, sequential on the one GPU) ----------
REH="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT
     --blocks 4096 --k 32 --lam 1e-3 --epochs 2 --theta-method streaming --guard"

run r2_guard_3e4  $REH --arm bsc --lr 3e-4
run r1_guard_6e4  $REH --arm bsc --lr 6e-4
run r3_guard_12e4 $REH --arm bsc --lr 1.2e-3
run r4_rcap_6e4   $REH --arm bsc --lr 6e-4 --aux-ratio-cap 1.0
run r5_fcap_6e4   $REH --arm bsc --lr 6e-4 --aux-frac-cap 0.5
run r6_renorm_rcap_6e4 $REH --arm bsc --lr 6e-4 --site-renorm --aux-ratio-cap 1.0

echo "[t1] campaign complete $(date)"
