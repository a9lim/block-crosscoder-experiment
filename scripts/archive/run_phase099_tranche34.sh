#!/bin/bash
# Phase 0.9.9 tranches 3+4, GPU queue (runbook execution-order step 3):
# the lambda=0 R-D frontier trainings, then balanced seeds for every
# headline cell. All runs carry the full production stack (guard +
# streaming theta + prefetch) — each is also a skip-rate datum at the
# ratified point.
#
# Frontier (tranche 3): lambda=0 both arms per protocol, k in
# {16,32,64} x {BSC, scalar}; scalar k32 seed0 already exists (the
# pilot scalar arm). The renorm lambda=0 k32 point is exploratory (F7
# gauge on the frontier plane; not in the ratified k x arm grid).
# Headline point k=32 gets 2 seeds (bsc + scalar seed 1).
#
# Seeds (tranche 4): >=3 seeds per headline cell — bsc primary needs
# seed 2 (0,1 exist), renorm needs 1,2, scalar k32 needs 1,2 (seed 1
# doubles as the frontier headline-point seed), bsf needs 1,2.
#
#   nohup bash scripts/run_phase099_tranche34.sh > /data/runs/bcc-phase099/campaign34.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-phase099
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "[t34] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[t34] done $name  $(date +%H:%M)"
  else
    echo "[t34] EXIT-NONZERO $name (exit $?, see $OUT/$name.log)"
  fi
}

REH="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT
     --blocks 4096 --epochs 2 --guard --theta-method streaming --prefetch 4"
SS="python -u scripts/run_phase099_single_site.py --store $STORE --out-root $OUT
    --blocks 4096 --k 32 --guard --prefetch 4"

# ---- tranche 3: lambda=0 frontier, seed 0 ----------------------------------
run f_bsc_lam0_k32 $REH --arm bsc --lam 0 --k 32 --seed 0
run f_bsc_lam0_k16 $REH --arm bsc --lam 0 --k 16 --seed 0
run f_bsc_lam0_k64 $REH --arm bsc --lam 0 --k 64 --seed 0
run f_scalar_lam0_k16 $REH --arm scalar --lam 0 --k 16 --seed 0
run f_scalar_lam0_k64 $REH --arm scalar --lam 0 --k 64 --seed 0
run f_bsc_lam0_k32_renorm $REH --arm bsc --lam 0 --k 32 --seed 0 --site-renorm

# ---- headline-point second seeds (tranche 3) + tranche 4 seeds -------------
run s_scalar_lam0_k32_seed1 $REH --arm scalar --lam 0 --k 32 --seed 1
run f_bsc_lam0_k32_seed1 $REH --arm bsc --lam 0 --k 32 --seed 1
run s_renorm_seed1 $REH --arm bsc --lam 1e-3 --k 32 --seed 1 --site-renorm
run s_renorm_seed2 $REH --arm bsc --lam 1e-3 --k 32 --seed 2 --site-renorm
run s_bsc_seed2 $REH --arm bsc --lam 1e-3 --k 32 --seed 2
run s_scalar_lam0_k32_seed2 $REH --arm scalar --lam 0 --k 32 --seed 2
run s_bsf_seed1 $SS --arm bsf --seed 1
run s_bsf_seed2 $SS --arm bsf --seed 2

echo "[t34] campaign complete $(date)"
