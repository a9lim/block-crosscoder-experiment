# scripts/archive — ladder-era entry points

Historical one-shot scripts from the Phase-0 pilot program, preserved
verbatim: the SAE-era discovery/harvest tools, the 1b-era harvester
and toy export, and every campaign shell (0.9 → 0.9.9 tranches).
They regenerate archived results against the run layout of their day;
the findings they produced live in `docs/archive/`.

**These shells reference the live tools by their old names.** The
2026-07-19 script restructure renamed the phase-neutral tools they
call — apply this map when re-running anything here:

| old (referenced here) | live tool |
|---|---|
| `scripts/run_phase09_rehearsal.py` | `scripts/train_bsc.py` |
| `scripts/run_phase099_single_site.py` | `scripts/train_single_site.py` |
| `scripts/harvest_pilot4b_store.py` | `scripts/harvest_store.py` |
| `scripts/extend_pilot4b_store.py` | `scripts/extend_store.py` |
| `scripts/verify_phase09_store.py` | `scripts/verify_store.py` |
| `scripts/validate_rd_codec.py` | `scripts/validate_codec.py` |
| `scripts/validate_e1_theta.py` | `scripts/validate_theta.py` |
| `scripts/validate_e3_revival.py` | `scripts/validate_revival.py` |
| `scripts/run_phase_minus1.py` | `scripts/run_battery.py` |
| `scripts/run_{bundle,capture}_sweep.py` | `scripts/sweep_{bundle,capture}.py` |
| `scripts/analysis/calendar_probe.py` | `scripts/analysis/probe_calendar.py` |
| `scripts/analysis/zoo_block_tests.py` | `scripts/analysis/probe_families.py` |
| `scripts/analysis/atlas_stream_tests.py` | `scripts/analysis/probe_stream.py` |
| `scripts/analysis/crossarm_tests.py` | `scripts/analysis/probe_crossarm.py` |
| `scripts/analysis/tier_a_ring_tests.py` | `scripts/analysis/probe_ring_consolidation.py` |
| `scripts/analysis/depth_scalar_tests.py` | `scripts/analysis/probe_depth_scalar.py` |
| `scripts/analysis/fig_pilot4b{,_3d}.py` | `scripts/analysis/fig_capture.py` / `fig_zoo_3d.py` |
| `scripts/analysis/fig_geometry4b{,_3d}.py` | `scripts/analysis/fig_geometry{,_3d}.py` |
| `scripts/analysis/plot_rd_{frontier,tying}.py` | `scripts/analysis/fig_rd_{frontier,tying}.py` |

The five `fig_*`/`extract_*` scripts archived alongside
(`fig_probe`, `fig_rings`, `fig_geometry` (1b), `fig_calibration`,
`extract_phase0_geometry`) drew the deleted 1b/SAE-era figure sets;
they are superseded by the winner-pointer regeneration pass
(`scripts/analysis/regen_figures.sh`), not renamed.

**Pilot-era showcase figures, verbatim.** When the live `fig_*` set was
rewired to the dynamic winner pointer (2026-07-19 evening), the
pilot-campaign versions — hardcoded run names, hand-curated block
identities (b595/b862/b2146/…), and the pilot-only panels (lr-cliff
instability traces, the 8-run capture lottery, scalar/destroyed-arm
comparisons, the `_names.py` identity/oddball catalog) — were preserved
here under their original names: `fig_pilot4b.py`, `fig_pilot4b_3d.py`,
`fig_geometry4b.py`, `fig_geometry4b_3d.py`, `fig_pilot4b_atlas.py`,
`fig_pilot4b_anatomy_3d.py`, `fig_worldmap4b_3d.py`. They regenerate the
figures cited in the archived findings docs against the pilot artifacts
(`data/analysis/*_pilot4b*`, `/data/runs/bcc-pilot4b` checkpoints); the
live scripts regenerate the canonical set from whatever
`data/phase0/winner.json` points at, with block identities derived per
checkpoint (`derive_showcase.py`).

`run_phase099_tranche6.sh` ran the epochs-vs-fresh factorial on
2026-07-19 from the pre-restructure layout (jobe pulled the rename
only after it drained).
