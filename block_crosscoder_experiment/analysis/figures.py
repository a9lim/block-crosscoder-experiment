"""Generate the canonical, winner-scoped figure catalog.

Each descriptive zoo family gets exactly three views:

``stream.html``
    Raw shrinkage-whitened class means across the eight sites.
``frames.html``
    The same means seen through the current winner's best block frame.
``flow.html``
    That frame-space geometry in one fixed joint-PCA gauge, so its motion
    through depth is visible without a per-site refit.

The zoo is burned descriptive evidence. Every page therefore carries the
winner FVU, block identity, top-1 capture, ordering statistic, and
qualification status instead of presenting a best-looking block alone.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from scipy.stats import spearmanr

from .artifacts import FIGURES_DIR, analysis_dir, load_winner
from .catalog import FamilySpec, ZOO, ZOO_FAMILIES
from .style import INK, INK2, SURFACE, cyclic_colors

Z_STEP = 1.2


def _procrustes(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(moving.T @ reference)
    return u @ vt


def _harmonic_basis(x: np.ndarray) -> tuple[np.ndarray, float]:
    spectrum = np.fft.fft(x, axis=0)
    power = (np.abs(spectrum[1 : len(x) // 2 + 1]) ** 2).sum(1)
    stat = float(power[0] / max(power.sum(), 1e-12))
    u = np.real(spectrum[1])
    v = -np.imag(spectrum[1])
    u /= max(np.linalg.norm(u), 1e-12)
    v -= u * (v @ u)
    v /= max(np.linalg.norm(v), 1e-12)
    return np.stack([u, v], axis=1), stat


def _pca_basis(x: np.ndarray) -> tuple[np.ndarray, float]:
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:2].T
    rho = spearmanr(np.arange(len(x)), x @ basis[:, 0]).statistic
    return basis, float(abs(rho)) if np.isfinite(rho) else 0.0


def _cloud_basis(x: np.ndarray) -> tuple[np.ndarray, float]:
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    variance = float(
        (singular_values[:2] ** 2).sum()
        / max((singular_values**2).sum(), 1e-12)
    )
    return vt[:2].T, variance


def _stack_planes(
    means: np.ndarray, spec: FamilySpec
) -> list[tuple[np.ndarray, float]]:
    """Map ``[site, class, dim]`` means to aligned 2-D display planes."""

    planes: list[tuple[np.ndarray, float]] = []
    previous: np.ndarray | None = None
    for site_means in means:
        centered = site_means - site_means.mean(0)
        fitted = centered[: spec.fit_count] if spec.fit_count else centered
        if spec.topology == "ring":
            basis, stat = _harmonic_basis(fitted)
        elif spec.topology == "cloud":
            basis, stat = _cloud_basis(fitted)
        else:
            basis, stat = _pca_basis(fitted)
        projected = centered @ basis
        projected /= max(np.sqrt((projected**2).mean()), 1e-12)
        if previous is not None:
            projected = projected @ _procrustes(previous, projected)
        planes.append((projected, stat))
        previous = projected
    return planes


def _path(spec: FamilySpec) -> list[int]:
    if spec.topology == "cloud":
        return []
    count = spec.fit_count or len(spec.labels)
    path = list(range(count))
    if spec.topology == "ring":
        path.append(0)
    return path


def _status(entry: dict, n_classes: int, alpha: float = 0.01) -> dict:
    consolidated = entry["top1_claimed"] * 2 > n_classes
    ordered = entry["order"]["perm_p"] <= alpha
    return {
        "consolidated": consolidated,
        "ordered": ordered,
        "qualified": consolidated and ordered,
    }


def _order_text(order: dict) -> str:
    if order["kind"] == "ring":
        value = f"ring {order['hits']}/{order['max']}"
    elif order["kind"] == "geo":
        value = f"geo R² {order['r2']:.3f}"
    else:
        value = f"|rho| {order['spearman']:.3f}"
    return f"{value}, permutation p={order['perm_p']:.3g}"


def _subtitle(
    family: str, entry: dict, fvu: float, n_classes: int
) -> tuple[str, str]:
    state = _status(entry, n_classes)
    verdict = "qualifies" if state["qualified"] else "does not qualify"
    return (
        f"winner b{entry['best_block']}; top-1 {entry['top1_claimed']}/"
        f"{n_classes}; pooled FVU {fvu:.4f}; {verdict}.",
        f"{_order_text(entry['order'])}. Burned descriptive family; never a "
        "selection endpoint.",
    )


def _stack_figure(
    planes: list[tuple[np.ndarray, float]],
    spec: FamilySpec,
    sites: list[int],
    title: str,
    subtitle: tuple[str, str],
    metric: str,
) -> go.Figure:
    colors = cyclic_colors(len(spec.labels))
    z_pos = np.arange(len(sites)) * Z_STEP
    fig = go.Figure()

    for class_index, label in enumerate(spec.labels):
        fig.add_trace(
            go.Scatter3d(
                x=[plane[class_index, 0] for plane, _ in planes],
                y=[plane[class_index, 1] for plane, _ in planes],
                z=z_pos,
                mode="lines",
                line={"color": "rgba(137,135,129,0.40)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    path = _path(spec)
    for site_index, ((plane, stat), layer) in enumerate(zip(planes, sites)):
        if path:
            fig.add_trace(
                go.Scatter3d(
                    x=plane[path, 0],
                    y=plane[path, 1],
                    z=[z_pos[site_index]] * len(path),
                    mode="lines",
                    line={"color": INK2, "width": 3},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        show_text = site_index == len(sites) - 1 and len(spec.labels) <= 20
        fig.add_trace(
            go.Scatter3d(
                x=plane[:, 0],
                y=plane[:, 1],
                z=[z_pos[site_index]] * len(spec.labels),
                mode="markers+text" if show_text else "markers",
                marker={"size": 5, "color": colors},
                text=list(spec.labels) if show_text else None,
                textposition="top center",
                textfont={"size": 9, "color": INK2},
                hovertext=list(spec.labels),
                hovertemplate=f"L{layer}<br>%{{hovertext}}<extra></extra>",
                name=f"L{layer} ({metric} {stat:.0%})",
            )
        )

    fig.update_layout(
        title=(
            f"{html.escape(title)}<br><sup>{html.escape(subtitle[0])}</sup>"
            f"<br><sup>{html.escape(subtitle[1])}</sup>"
        ),
        height=760,
        width=1060,
        paper_bgcolor=SURFACE,
        font={"family": "system-ui, sans-serif", "color": INK},
        legend={"font": {"size": 10}},
        scene={
            "xaxis_title": "plane 1",
            "yaxis_title": "plane 2",
            "zaxis": {
                "title": "depth",
                "ticktext": [f"L{site}" for site in sites],
                "tickvals": z_pos.tolist(),
            },
            "camera": {"eye": {"x": 1.7, "y": 1.5, "z": 0.9}},
        },
        margin={"t": 125},
    )
    return fig


def _flow_figure(
    framed_means: np.ndarray,
    spec: FamilySpec,
    sites: list[int],
    title: str,
    subtitle: tuple[str, str],
) -> go.Figure:
    site_count, class_count, _ = framed_means.shape
    centered = framed_means - framed_means.mean(1, keepdims=True)
    flat = centered.reshape(site_count * class_count, -1)
    _, singular_values, vt = np.linalg.svd(flat, full_matrices=False)
    components = min(3, vt.shape[0])
    projected = flat @ vt[:components].T
    if components < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - components)))
    projected = projected.reshape(site_count, class_count, 3)
    variance = float(
        (singular_values[:components] ** 2).sum()
        / max((singular_values**2).sum(), 1e-12)
    )
    colors = cyclic_colors(class_count)
    fig = go.Figure()

    for class_index, label in enumerate(spec.labels):
        fig.add_trace(
            go.Scatter3d(
                x=projected[:, class_index, 0],
                y=projected[:, class_index, 1],
                z=projected[:, class_index, 2],
                mode="lines+markers",
                line={"color": colors[class_index], "width": 4},
                marker={"size": np.linspace(3, 6, site_count), "color": colors[class_index]},
                name=label,
                showlegend=class_count <= 20,
                hovertext=[f"{label}, L{site}" for site in sites],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    path = _path(spec)
    if path:
        for site_index in range(site_count):
            fig.add_trace(
                go.Scatter3d(
                    x=projected[site_index, path, 0],
                    y=projected[site_index, path, 1],
                    z=projected[site_index, path, 2],
                    mode="lines",
                    line={"color": "rgba(137,135,129,0.35)", "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=(
            f"{html.escape(title)}<br><sup>{html.escape(subtitle[0])}</sup>"
            f"<br><sup>{html.escape(subtitle[1])} Joint PCs retain "
            f"{variance:.0%} of class-mean variance; marker size increases "
            "with depth.</sup>"
        ),
        height=760,
        width=1060,
        paper_bgcolor=SURFACE,
        font={"family": "system-ui, sans-serif", "color": INK},
        scene={
            "xaxis_title": "joint PC1",
            "yaxis_title": "joint PC2",
            "zaxis_title": "joint PC3",
            "camera": {"eye": {"x": 1.6, "y": 1.4, "z": 1.0}},
        },
        margin={"t": 125},
    )
    return fig


def _write_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        path,
        include_plotlyjs="../assets/plotly.min.js",
        full_html=True,
    )


def _write_catalog(out: Path, manifest: dict) -> None:
    families = manifest["families"]
    run_name = manifest["winner"]["run_name"]
    rows = []
    readme_rows = []
    for family in sorted(families):
        entry = families[family]
        links = " · ".join(
            f'<a href="{family}/{view}.html">{view}</a>'
            for view in ("frames", "flow", "stream")
        )
        rows.append(
            f"<tr><td><strong>{html.escape(family)}</strong></td>"
            f"<td>b{entry['block']}</td><td>{entry['top1_claimed']}/"
            f"{entry['n_classes']}</td><td>{html.escape(entry['order'])}</td>"
            f"<td>{'yes' if entry['qualified'] else 'no'}</td><td>{links}</td></tr>"
        )
        readme_rows.append(
            f"| {family} | b{entry['block']} | {entry['top1_claimed']}/"
            f"{entry['n_classes']} | {entry['order']} | "
            f"{'yes' if entry['qualified'] else 'no'} | "
            f"[frames]({family}/frames.html) · [flow]({family}/flow.html) "
            f"· [stream]({family}/stream.html) |"
        )

    summaries = sorted((out / "summary").glob("*.png"))
    summary_html = "".join(
        f'<li><a href="summary/{p.name}">{html.escape(p.stem.replace("_", " "))}</a></li>'
        for p in summaries
    )
    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BSC figure catalog</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1180px;margin:3rem auto;padding:0 1.5rem;color:#0b0b0b;background:#fcfcfb}}
table{{border-collapse:collapse;table-layout:fixed;width:100%}}th,td{{padding:.55rem .7rem;border-bottom:1px solid #e1e0d9;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{position:sticky;top:0;background:#fcfcfb}}th:nth-child(1){{width:11%}}th:nth-child(2){{width:8%}}th:nth-child(3){{width:8%}}th:nth-child(4){{width:34%}}th:nth-child(5){{width:9%}}th:nth-child(6){{width:30%}}td:last-child{{white-space:nowrap}}code{{font-size:.9em;overflow-wrap:anywhere}}a{{color:#185fa5}}
@media(max-width:760px){{body{{margin:1.5rem auto;padding:0 .8rem}}table{{font-size:12px}}th,td{{padding:.4rem .3rem}}td:last-child{{white-space:normal}}}}
</style></head><body><h1>BSC figure catalog</h1>
<p>All zoo artifacts are generated from <code>{html.escape(run_name)}</code>. Families are burned descriptive probes; qualification is shown for context and is not a model-selection endpoint.</p>
<table><thead><tr><th>family</th><th>block</th><th>top-1</th><th>order</th><th>qualifies</th><th>views</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Cross-family summaries</h2><ul>{summary_html}</ul></body></html>
"""
    (out / "index.html").write_text(index)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = "\n".join(
        [
            "# Figure catalog",
            "",
            f"Generated from `{run_name}`. The family zoo is descriptive and burned;",
            "confirmatory capture remains sealed until Phase-1 config freeze.",
            "",
            "| family | winner block | top-1 | order | qualifies | views |",
            "|---|---:|---:|---|:---:|---|",
            *readme_rows,
            "",
            "Cross-family figures live in [`summary/`](summary/). Full refresh: "
            "`bsc refresh-analysis`; render-only: `bsc figures`.",
            "",
        ]
    )
    (out / "README.md").write_text(readme)


def generate(
    artifact_dir: Path,
    out: Path = FIGURES_DIR,
    families: tuple[str, ...] = ZOO_FAMILIES,
) -> dict:
    winner = load_winner()
    means_npz = np.load(artifact_dir / "zoo_means.npz")
    frames_npz = np.load(artifact_dir / "frames_winner.npz")
    tests = json.loads((artifact_dir / "zoo_block_tests.json").read_text())["winner"]
    frame_meta = json.loads(str(frames_npz["meta"]))
    frame_run = Path(frame_meta["run"]).name
    if frame_run != winner["run_name"]:
        raise ValueError(
            f"frames belong to {frame_run}, not promoted winner {winner['run_name']}"
        )

    sites = list(winner["sites"])
    site_scales = np.asarray(
        np.ones(len(sites))
        if winner.get("site_renorm_folded")
        else winner.get("site_renorm_scalars", np.ones(len(sites))),
        dtype=np.float32,
    )
    if site_scales.shape != (len(sites),):
        raise ValueError("winner site_renorm_scalars do not match its sites")
    blocks = frames_npz["blocks"].tolist()
    manifest: dict = {
        "winner": winner,
        "artifact_dir": str(artifact_dir),
        "families": {},
    }
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "plotly.min.js").write_text(get_plotlyjs())

    for family in families:
        if family not in ZOO:
            raise ValueError(f"unknown zoo family: {family}")
        spec = ZOO[family]
        entry = tests[family]
        block = entry["best_block"]
        if block not in blocks:
            raise ValueError(f"winner frame dump is missing {family} block b{block}")
        means = means_npz[f"{family}_means"].transpose(1, 0, 2)
        if means.shape[:2] != (len(sites), len(spec.labels)):
            raise ValueError(
                f"{family} means shape {means.shape} does not match registry"
            )
        frames = frames_npz["frames"][:, blocks.index(block)]
        framed = np.stack(
            [
                (means[site] - means[site].mean(0))
                * site_scales[site]
                @ frames[site].T
                for site in range(len(sites))
            ]
        )
        subtitle = _subtitle(family, entry, tests["fvu_pooled"], len(spec.labels))
        metric = (
            "1st harmonic"
            if spec.topology == "ring"
            else "top-2 var"
            if spec.topology == "cloud"
            else "|rho|"
        )
        family_out = out / family
        _write_figure(
            _stack_figure(
                _stack_planes(means, spec),
                spec,
                sites,
                f"{family}: residual-stream geometry across depth",
                subtitle,
                metric,
            ),
            family_out / "stream.html",
        )
        _write_figure(
            _stack_figure(
                _stack_planes(framed, spec),
                spec,
                sites,
                f"{family}: the stream through winner block b{block}'s frames",
                subtitle,
                metric,
            ),
            family_out / "frames.html",
        )
        _write_figure(
            _flow_figure(
                framed,
                spec,
                sites,
                f"{family}: winner block b{block}'s frame-space flow",
                subtitle,
            ),
            family_out / "flow.html",
        )
        state = _status(entry, len(spec.labels))
        manifest["families"][family] = {
            "block": block,
            "top1_claimed": entry["top1_claimed"],
            "n_classes": len(spec.labels),
            "order": _order_text(entry["order"]),
            **state,
            "files": [
                f"{family}/frames.html",
                f"{family}/flow.html",
                f"{family}/stream.html",
            ],
        }
        print(f"{family}: b{block} -> frames, flow, stream", flush=True)

    _write_catalog(out, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    parser.add_argument("--families", nargs="*", choices=ZOO_FAMILIES)
    args = parser.parse_args()
    selected = tuple(args.families) if args.families else ZOO_FAMILIES
    generate(args.analysis_dir or analysis_dir(), args.out, selected)


if __name__ == "__main__":
    main()
