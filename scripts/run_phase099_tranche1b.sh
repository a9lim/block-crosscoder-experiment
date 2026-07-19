#!/bin/bash
# Phase 0.9.9 tranche 1b: cap-only (UNGUARDED) arms at the 6e-4
# reproduction point — the cascade-suppression axis proper.
#
# Why these exist: r1's forensics showed the 6e-4 blow-up is seed
# (main-loss wobble, steps 1009-1012, aux-free) + amplifier (SASA slam,
# aux -> 97-100% of gradient, dead 0.02% -> 13%). The guarded cap arms
# (r4/r5/r6) refuse at the blown state regardless of cap — guard and
# cap conflate. These unguarded arms isolate the cap: the unguarded
# baseline rode the amplified excursion to FVU 0.553 (primary) / 1.105
# destroyed (renorm); a cap that suppresses the amplifier should ride
# the same seed wobble to a materially better endpoint with a bounded
# dead fraction. Registered prediction (2026-07-19, pre-run): the ratio
# cap's clamp TIGHTENS as a cascade grows (aux grad up -> alpha_eff
# down) while the frac cap WEAKENS (dead set up -> allowed slam up), so
# rcap should dominate fcap here.
#
#   nohup bash scripts/run_phase099_tranche1b.sh > /data/runs/bcc-phase099/campaign1b.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STORE=/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
OUT=/data/runs/bcc-phase099
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "[t1b] start $name  $(date +%H:%M)"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[t1b] done $name  $(date +%H:%M)"
  else
    echo "[t1b] EXIT-NONZERO $name (exit $?, see $OUT/$name.log)"
  fi
}

REH="python -u scripts/run_phase09_rehearsal.py --store $STORE --out-root $OUT
     --blocks 4096 --k 32 --lam 1e-3 --epochs 2 --theta-method streaming
     --prefetch 4"

run r4b_rcap_6e4_noguard $REH --arm bsc --lr 6e-4 --aux-ratio-cap 1.0
run r5b_fcap_6e4_noguard $REH --arm bsc --lr 6e-4 --aux-frac-cap 0.5
run r6b_renorm_rcap_6e4_noguard $REH --arm bsc --lr 6e-4 --site-renorm --aux-ratio-cap 1.0

echo "[t1b] campaign complete $(date)"
