#!/bin/bash
# Regenerate the canonical figure set (figures/phase0/) from the current
# winner (data/phase0/winner.json — see scripts/analysis/winner.py).
# Promote a better run by editing winner.json and re-running this.
#
# Two tiers:
#   local  — R-D plane figures from committed payloads (no GPU, any machine)
#   jobe   — the showcase pipeline against the winner checkpoint + store:
#            probe scan -> capture tests -> showcase derivation -> geometry
#            extraction -> figures. Run on jobe AFTER training drains
#            (checkpoint loads beside a training residency OOM the box).
#
#   bash scripts/analysis/regen_figures.sh local
#   bash scripts/analysis/regen_figures.sh jobe          # full pipeline
#   bash scripts/analysis/regen_figures.sh jobe-figs     # figures only
#                                                        # (artifacts exist)
set -e
cd "$(dirname "$0")/../.." || exit 1
TIER=${1:-local}

if [ "$TIER" = "local" ] || [ "$TIER" = "all" ]; then
  python scripts/analysis/fig_rd_frontier.py \
    --inputs 'data/phase0/rd_*.json' 'data/phase0/f_*.json' 'data/phase0/s_*.json' \
    --out figures/phase0/rd_frontier.png
  python scripts/analysis/fig_rd_tying.py
fi

if [ "$TIER" = "jobe" ] || [ "$TIER" = "all" ]; then
  # ---- winner-scoped paths, read from the pointer -----------------------
  eval "$(python - <<'EOF'
import sys
sys.path.insert(0, "scripts/analysis")
from pathlib import Path
from winner import analysis_dir, load_winner
w = load_winner()
print(f'ADIR={analysis_dir(w)}')
print(f'WIN_DIR={Path(w["ckpt"]).parent}')
print(f'PRI_DIR={w["counterpart_primary"]}')
print(f'RUN_ROOT={Path(w["ckpt"]).parent.parent}')
print(f'STORE={w["store"]}')
print(f'MODEL={w["model"]}')
EOF
)"
  echo "winner: $WIN_DIR"
  echo "primary counterpart: $PRI_DIR"
  echo "artifacts -> $ADIR"
  mkdir -p "$ADIR"

  # 1. One combined family scan (weekday/month first: their fam indices
  #    are position-sensitive in downstream cap-filtered tests).
  #    ~8M scanned tokens, per-class cap keeps the concat in host RAM.
  python scripts/analysis/probe_calendar.py --device cuda \
    --model "$MODEL" --store "$STORE" --out "$ADIR" --tag "" \
    --families weekday month ordinal cardinal digit season compass \
               color country element planet \
    --per-class-cap 600 --runs

  # 2. Capture + consolidation tests (both arms), calendar code exports.
  python scripts/analysis/probe_ring_consolidation.py \
    --out-root "$RUN_ROOT" --only "$(basename "$WIN_DIR")" "$(basename "$PRI_DIR")" \
    --acts "$ADIR/calendar_probe_acts.npz" --store "$STORE" \
    --tokenizer "$MODEL" --device cuda --out "$ADIR/ring_tests.json"
  python scripts/analysis/probe_depth_scalar.py \
    --out-root "$RUN_ROOT" --acts "$ADIR/calendar_probe_acts.npz" \
    --store "$STORE" --tokenizer "$MODEL" --device cuda \
    --out "$ADIR/depth_scalar.json"
  python scripts/analysis/export_block_codes.py \
    --out-root "$RUN_ROOT" --only "$(basename "$WIN_DIR")" "$(basename "$PRI_DIR")" \
    --acts "$ADIR/calendar_probe_acts.npz" --store "$STORE" \
    --tokenizer "$MODEL" --device cuda --out-dir "$ADIR" --tag ""

  # 3. Family capture over all 12 families, both arms -> zoo artifacts.
  python scripts/analysis/probe_families.py \
    --acts "$ADIR/calendar_probe_acts.npz" --store "$STORE" \
    --tokenizer "$MODEL" --device cuda --out-dir "$ADIR" --tag "" \
    --runs "winner=$WIN_DIR" "primary=$PRI_DIR"

  # 4. Showcase-block derivation (the mega-block rule, made mechanical).
  python scripts/analysis/derive_showcase.py

  # 5. Weight-space geometry + eval activation stats + qualified-block
  #    frame dumps (blocks read back from the showcase map per arm).
  python scripts/analysis/extract_geometry.py --device cuda --out "$ADIR" \
    --runs "winner=$WIN_DIR" "primary=$PRI_DIR" \
    --sites $(python -c "
import sys; sys.path.insert(0, 'scripts/analysis')
from winner import load_winner
print(' '.join(str(s) for s in load_winner()['sites']))")
  python scripts/analysis/eval_activation_stats.py --device cuda \
    --store "$STORE" --out "$ADIR" \
    --runs "winner=$WIN_DIR" "primary=$PRI_DIR"
  for ARM in winner primary; do
    RUN_DIR=$([ "$ARM" = winner ] && echo "$WIN_DIR" || echo "$PRI_DIR")
    BLOCKS=$(python -c "
import json, sys
show = json.load(open('$ADIR/showcase_blocks.json'))
bs = sorted({e['arms']['$ARM']['block'] for e in show['families'].values()
             if '$ARM' in e['arms']})
print(' '.join(map(str, bs)))")
    [ -n "$BLOCKS" ] && python scripts/analysis/dump_block_frames.py \
      --run "$RUN_DIR" --blocks $BLOCKS --out "$ADIR/frames_$ARM.npz"
  done
  python scripts/analysis/probe_crossarm.py
fi

if [ "$TIER" = "jobe" ] || [ "$TIER" = "jobe-figs" ] || [ "$TIER" = "all" ]; then
  # ---- figures (need the artifacts above in the winner analysis dir) ----
  python scripts/analysis/fig_capture.py            # capture, rings, allocation
  python scripts/analysis/fig_geometry.py           # share/rotation/packing/census
  python scripts/analysis/fig_zoo_3d.py             # per-family 3D stacks
  python scripts/analysis/fig_geometry_3d.py        # interactive geometry
  python scripts/analysis/fig_block_anatomy_3d.py   # block-anatomy explainers
  python scripts/analysis/fig_bsc_atlas.py          # hoverable all-block atlas
  python scripts/analysis/fig_worldmap_3d.py        # stacked world-map decode
fi

echo "regen ($TIER) complete -> figures/phase0/"
