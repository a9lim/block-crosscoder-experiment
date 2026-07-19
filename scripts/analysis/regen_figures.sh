#!/bin/bash
# Regenerate the canonical figure set (figures/phase0/) from the current
# winner (data/phase0/winner.json — see scripts/analysis/winner.py).
# Promote a better run by editing winner.json and re-running this.
#
# Two tiers:
#   local  — R-D plane figures from committed payloads (no GPU, any machine)
#   jobe   — showcase figures needing encode passes against the winner
#            checkpoint + store (run on jobe after training drains)
#
#   bash scripts/analysis/regen_figures.sh local
#   bash scripts/analysis/regen_figures.sh jobe
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
  # Showcase set, drawn from the winner checkpoint (encode passes on CUDA).
  # NOTE: adaptation of these scripts to the winner pointer is the
  # post-tranche-6 work item (task: figures/phase0 canonical set) —
  # until it lands, each still carries its pilot-era run wiring.
  python scripts/analysis/probe_calendar.py          # capture + ring stats
  python scripts/analysis/probe_families.py          # zoo/number-line capture
  python scripts/analysis/fig_capture.py             # capture maps, rings, allocation
  python scripts/analysis/fig_geometry.py            # shear zone, packing, census
  python scripts/analysis/fig_geometry_3d.py         # interactive geometry
  python scripts/analysis/fig_zoo_3d.py              # per-family 3D stacks
  python scripts/analysis/fig_block_anatomy_3d.py    # block-anatomy explainers
  python scripts/analysis/fig_bsc_atlas.py           # hoverable all-block atlas
  python scripts/analysis/fig_worldmap_3d.py         # stacked world-map decode
fi

echo "regen ($TIER) complete -> figures/phase0/"
