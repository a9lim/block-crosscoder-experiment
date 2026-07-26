"""Controlled month-manifold analysis for a trained Phase-2 BSC.

This module is deliberately downstream of scientific selection.  It captures
GPT-2 residual streams for a balanced prompt panel, applies the exact scalar-RMS
transform bundled with each deployable codec, and inspects the most
month-responsive learned BSC block.  Per-site encoder contributions remain in
one shared block-coordinate system, so layer panels never receive independent
rotations.  Raw-residual controls use an explicit orthogonal Procrustes
alignment to the final captured layer.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTH_LABELS = tuple(month[:3] for month in MONTHS)
HOOKS = (
    "blocks.3.hook_resid_pre",
    "blocks.5.hook_resid_pre",
    "blocks.7.hook_resid_pre",
    "blocks.9.hook_resid_pre",
)
LAYER_LABELS = ("block 3", "block 5", "block 7", "block 9")
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROMPT_TEMPLATES = (
    "The month on the calendar is {month}.",
    "They scheduled the appointment for {month}.",
    "She wrote {month} in the date field.",
    "Our next review takes place in {month}.",
    "The document was labeled {month}.",
    "He circled {month} on the calendar.",
    "We expect the shipment during {month}.",
    "The archive folder is named {month}.",
    "A reminder was added for {month}.",
    "Their plan begins in {month}.",
    "The report covers the month of {month}.",
    "Please select {month} from the menu.",
)


@dataclass(frozen=True)
class SeedGeometry:
    seed: int
    selected_block: int
    top_blocks: tuple[dict[str, float | int], ...]
    site_points_2d: np.ndarray
    site_points_3d: np.ndarray
    joint_points_2d: np.ndarray
    joint_points_3d: np.ndarray
    site_rms: np.ndarray
    joint_rms: float
    site_metrics: tuple[dict[str, float | str], ...]
    joint_metrics: dict[str, float | str]
    gallery: tuple[dict[str, Any], ...]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _pca_basis(rows: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = rows - rows.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    n = min(n_components, vh.shape[0])
    basis = vh[:n].T
    if n < n_components:
        basis = np.pad(basis, ((0, 0), (0, n_components - n)))
    variance = np.square(singular)
    explained = variance / max(float(variance.sum()), np.finfo(float).eps)
    explained = np.pad(explained[:n_components], (0, max(0, n_components - n)))
    return basis, explained


def _shape_normalize(points: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0, keepdims=True)
    rms = math.sqrt(float(np.mean(np.sum(np.square(centered), axis=-1))))
    if rms <= np.finfo(float).eps:
        return centered, 0.0
    return centered / rms, rms


def _orthogonal_procrustes(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    points, _ = _shape_normalize(points)
    reference, _ = _shape_normalize(reference)
    u, _, vt = np.linalg.svd(points.T @ reference, full_matrices=False)
    return points @ (u @ vt)


def _circle_radial_cv(points: np.ndarray) -> float:
    if points.shape != (12, 2):
        raise ValueError("month circle fit requires [12,2] points")
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    rhs = np.square(x) + np.square(y)
    center_x, center_y, _ = np.linalg.lstsq(design, rhs, rcond=None)[0]
    radii = np.sqrt(np.square(x - center_x) + np.square(y - center_y))
    return float(radii.std() / max(float(radii.mean()), np.finfo(float).eps))


def _calendar_metrics(points: np.ndarray, *, permutations: int = 20_000) -> dict[str, float | str]:
    points, _ = _shape_normalize(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    calendar_edges = np.array([distances[i, (i + 1) % 12] for i in range(12)])
    noncalendar = [
        distances[i, j]
        for i in range(12)
        for j in range(i + 1, 12)
        if (j - i) not in {1, 11}
    ]
    neighbor_hits = 0
    for index in range(12):
        nearest = set(np.argsort(distances[index])[1:3].tolist())
        expected = {(index - 1) % 12, (index + 1) % 12}
        neighbor_hits += len(nearest & expected)
    cyclic_steps = np.fromfunction(
        lambda i, j: np.minimum(np.abs(i - j), 12 - np.abs(i - j)),
        (12, 12),
        dtype=int,
    )
    tri = np.triu_indices(12, 1)
    distance_correlation = float(
        np.corrcoef(distances[tri], cyclic_steps[tri])[0, 1]
    )
    observed_length = float(calendar_edges.sum())
    rng = np.random.default_rng(20260726)
    random_lengths = np.empty(permutations, dtype=np.float64)
    for item in range(permutations):
        order = rng.permutation(12)
        random_lengths[item] = sum(
            distances[order[index], order[(index + 1) % 12]]
            for index in range(12)
        )
    permutation_p = float((1 + np.sum(random_lengths <= observed_length)) / (permutations + 1))
    closure_ratio = float(calendar_edges[-1] / max(float(calendar_edges[:-1].mean()), np.finfo(float).eps))
    radial_cv = _circle_radial_cv(points)
    neighbor_recall = neighbor_hits / 24.0
    edge_ratio = float(calendar_edges.mean() / max(float(np.mean(noncalendar)), np.finfo(float).eps))
    if (
        neighbor_recall >= 0.625
        and permutation_p <= 0.05
        and closure_ratio <= 1.75
        and radial_cv <= 0.35
    ):
        read = "ring"
    elif permutation_p <= 0.10 and closure_ratio > 1.75:
        read = "arc"
    elif permutation_p <= 0.10:
        read = "distorted cycle"
    else:
        read = "no resolved cycle"
    return {
        "calendar_neighbor_recall": float(neighbor_recall),
        "calendar_edge_ratio": edge_ratio,
        "closure_ratio": closure_ratio,
        "cyclic_distance_correlation": distance_correlation,
        "circle_radial_cv": radial_cv,
        "cycle_length_permutation_p": permutation_p,
        "shape_read": read,
    }


def _capture_month_activations(device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    from sae_lens import HookedSAETransformer

    model = HookedSAETransformer.from_pretrained_no_processing(
        "gpt2",
        revision=GPT2_REVISION,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise RuntimeError("GPT-2 loader did not expose its pinned tokenizer")
    month_token_ids: dict[str, int] = {}
    for month in MONTHS:
        encoded = tokenizer.encode(" " + month, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"{month} is not one leading-space token: {encoded}")
        month_token_ids[month] = int(encoded[0])

    batches: list[torch.Tensor] = []
    positions: list[list[int]] = []
    with torch.inference_mode():
        for template in PROMPT_TEMPLATES:
            texts = [template.format(month=month) for month in MONTHS]
            tokens = model.to_tokens(texts, prepend_bos=True).to(device)
            target_positions = []
            for row, month in enumerate(MONTHS):
                matches = torch.nonzero(
                    tokens[row] == month_token_ids[month],
                    as_tuple=False,
                ).flatten()
                if len(matches) != 1:
                    raise RuntimeError(
                        f"target token for {month} occurs {len(matches)} times in "
                        f"{texts[row]!r}"
                    )
                target_positions.append(int(matches.item()))
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in HOOKS,
                return_type=None,
                stop_at_layer=10,
            )
            per_site = []
            rows = torch.arange(12, device=device)
            for hook in HOOKS:
                per_site.append(cache[hook][rows, target_positions].float().cpu())
            batches.append(torch.stack(per_site, dim=1))
            positions.append(target_positions)
            del cache, tokens
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(batches, dim=0), {
        "model": "openai-community/gpt2",
        "revision": GPT2_REVISION,
        "hooks": HOOKS,
        "months": MONTHS,
        "month_token_ids": month_token_ids,
        "prompt_templates": PROMPT_TEMPLATES,
        "target_positions": positions,
        "forward_dtype": "bfloat16",
    }


def _load_deployable(
    path: Path,
    device: torch.device,
) -> tuple[BlockCrosscoder, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg_payload = dict(payload["model_cfg"])
    if cfg_payload.get("site_dims") is not None:
        cfg_payload["site_dims"] = tuple(cfg_payload["site_dims"])
    model = BlockCrosscoder(BSCConfig(**cfg_payload), device=device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    normalization = payload["normalization"]
    if normalization.get("kind") != "frozen_transform":
        raise RuntimeError("geometry analysis requires a frozen scalar-RMS transform")
    if normalization.get("mode") != "scalar_rms":
        raise RuntimeError(
            f"expected scalar_rms, got {normalization.get('mode')!r}"
        )
    return model, normalization


def _normalize(raw: torch.Tensor, normalization: dict[str, Any], device: torch.device) -> torch.Tensor:
    mean = normalization["mean"].to(device=device, dtype=torch.float32)
    diagonal = torch.diagonal(
        normalization["W"].to(device=device, dtype=torch.float32),
        dim1=-2,
        dim2=-1,
    )
    return (raw.to(device=device, dtype=torch.float32) - mean) * diagonal


def _response_ranking(
    joint: np.ndarray,
    selection_frequency: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    centered = joint - joint.mean(axis=1, keepdims=True)
    month_means = centered.mean(axis=0)
    between = np.mean(np.square(month_means), axis=(0, 2))
    within = np.mean(
        np.square(centered - month_means[None, ...]),
        axis=(0, 1, 3),
    )
    contrast = between / np.maximum(between + within, np.finfo(float).eps)
    positive_between = between[between > 0]
    scale = float(np.median(positive_between)) if len(positive_between) else 1.0
    prominence = np.log1p(between / max(scale, np.finfo(float).eps))
    max_frequency = max(float(selection_frequency.max()), np.finfo(float).eps)
    activity_weight = 0.25 + np.sqrt(selection_frequency / max_frequency)
    score = contrast * prominence * activity_weight
    candidates = np.flatnonzero(selection_frequency >= 0.02)
    if len(candidates) < 6:
        candidates = np.flatnonzero(selection_frequency >= 0.005)
    if len(candidates) < 6:
        candidates = np.arange(len(score))
    order = candidates[np.argsort(score[candidates])[::-1]]
    records: list[dict[str, float | int]] = []
    for block in order[:12]:
        records.append(
            {
                "block": int(block),
                "response_score": float(score[block]),
                "selection_frequency": float(selection_frequency[block]),
                "between_month_energy": float(between[block]),
                "within_month_energy": float(within[block]),
                "month_contrast": float(contrast[block]),
            }
        )
    return order, records


def _seed_geometry(
    *,
    seed: int,
    deployable: Path,
    raw: torch.Tensor,
    device: torch.device,
) -> SeedGeometry:
    model, normalization = _load_deployable(deployable, device)
    templates, months, sites, width = raw.shape
    x = _normalize(raw.reshape(-1, sites, width), normalization, device)
    with torch.inference_mode():
        encoder = model.encoder_tensor()
        site = torch.einsum("nsd,sgbd->nsgb", x, encoder)
        joint = model.encode(x)
        selection = model.select(joint, mode="topk", x=x)
    site_np = site.float().cpu().numpy().reshape(
        templates,
        months,
        sites,
        model.cfg.n_blocks,
        model.cfg.block_dim,
    )
    joint_np = joint.float().cpu().numpy().reshape(
        templates,
        months,
        model.cfg.n_blocks,
        model.cfg.block_dim,
    )
    selection_frequency = selection.float().mean(dim=0).cpu().numpy()
    order, top_blocks = _response_ranking(joint_np, selection_frequency)
    selected_block = int(order[0])

    site_block = site_np[..., selected_block, :]
    joint_block = joint_np[..., selected_block, :]
    site_centered = site_block - site_block.mean(axis=1, keepdims=True)
    joint_centered = joint_block - joint_block.mean(axis=1, keepdims=True)
    site_centroids = site_centered.mean(axis=0).transpose(1, 0, 2)
    joint_centroids = joint_centered.mean(axis=0)
    combined = np.concatenate((site_centroids.reshape(-1, model.cfg.block_dim), joint_centroids))
    basis, explained = _pca_basis(combined, 3)
    site_projected = np.einsum("smb,bk->smk", site_centroids, basis)
    joint_projected = joint_centroids @ basis
    normalized_sites = []
    site_rms = []
    site_metrics = []
    for layer in range(sites):
        normalized, rms = _shape_normalize(site_projected[layer])
        normalized_sites.append(normalized)
        site_rms.append(rms)
        metric = _calendar_metrics(normalized[:, :2])
        metric["projection_variance_2d"] = float(explained[:2].sum())
        site_metrics.append(metric)
    normalized_joint, joint_rms = _shape_normalize(joint_projected)
    joint_metrics = _calendar_metrics(normalized_joint[:, :2])
    joint_metrics["projection_variance_2d"] = float(explained[:2].sum())

    gallery: list[dict[str, Any]] = []
    for block in order[:3]:
        values = joint_np[..., int(block), :]
        values = values - values.mean(axis=1, keepdims=True)
        centroids = values.mean(axis=0)
        gallery_basis, gallery_explained = _pca_basis(centroids, 2)
        points, rms = _shape_normalize(centroids @ gallery_basis)
        metrics = _calendar_metrics(points)
        metrics["projection_variance_2d"] = float(gallery_explained[:2].sum())
        gallery.append(
            {
                "block": int(block),
                "points": points,
                "rms": rms,
                "metrics": metrics,
                "selection_frequency": float(selection_frequency[int(block)]),
            }
        )

    del model, encoder, x, site, joint, selection
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return SeedGeometry(
        seed=seed,
        selected_block=selected_block,
        top_blocks=tuple(top_blocks),
        site_points_2d=np.stack(normalized_sites)[:, :, :2],
        site_points_3d=np.stack(normalized_sites),
        joint_points_2d=normalized_joint[:, :2],
        joint_points_3d=normalized_joint,
        site_rms=np.asarray(site_rms),
        joint_rms=joint_rms,
        site_metrics=tuple(site_metrics),
        joint_metrics=joint_metrics,
        gallery=tuple(gallery),
    )


def _raw_geometry(raw: torch.Tensor) -> dict[str, Any]:
    values = raw.float().numpy()
    centered = values - values.mean(axis=1, keepdims=True)
    centroids = centered.mean(axis=0).transpose(1, 0, 2)
    points_2d = []
    points_3d = []
    explained_2d = []
    explained_3d = []
    for layer in range(len(HOOKS)):
        basis, explained = _pca_basis(centroids[layer], 3)
        projected = centroids[layer] @ basis
        points_2d.append(projected[:, :2])
        points_3d.append(projected)
        explained_2d.append(float(explained[:2].sum()))
        explained_3d.append(float(explained[:3].sum()))
    reference_2d, _ = _shape_normalize(points_2d[-1])
    reference_3d, _ = _shape_normalize(points_3d[-1])
    aligned_2d = np.stack(
        [_orthogonal_procrustes(points, reference_2d) for points in points_2d]
    )
    aligned_3d = np.stack(
        [_orthogonal_procrustes(points, reference_3d) for points in points_3d]
    )
    metrics = []
    for layer in range(len(HOOKS)):
        item = _calendar_metrics(aligned_2d[layer])
        item["projection_variance_2d"] = explained_2d[layer]
        item["projection_variance_3d"] = explained_3d[layer]
        metrics.append(item)
    return {
        "points_2d": aligned_2d,
        "points_3d": aligned_3d,
        "metrics": metrics,
        "alignment": "per-layer PCA, unit-RMS shape normalization, then orthogonal Procrustes to block 9",
    }


def _month_colors() -> np.ndarray:
    return plt.colormaps["twilight_shifted"](np.linspace(0.03, 0.97, 12))


def _draw_cycle_2d(ax: plt.Axes, points: np.ndarray, *, annotate: bool = True) -> None:
    colors = _month_colors()
    closed = np.vstack((points, points[:1]))
    ax.plot(closed[:, 0], closed[:, 1], color="#59636e", linewidth=1.2, alpha=0.72)
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=52, edgecolor="white", linewidth=0.7, zorder=3)
    if annotate:
        for label, (x, y) in zip(MONTH_LABELS, points, strict=True):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#d9dde2", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#d9dde2", linewidth=0.6, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_cycle_3d(ax: Any, points: np.ndarray) -> None:
    colors = _month_colors()
    closed = np.vstack((points, points[:1]))
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="#59636e", linewidth=1.1, alpha=0.74)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=38, depthshade=False, edgecolor="white", linewidth=0.5)
    for label, point in zip(MONTH_LABELS, points, strict=True):
        ax.text(point[0], point[1], point[2], label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect((1, 1, 0.85))


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_bsc_layers(results: Iterable[SeedGeometry], output: Path, *, dimensions: int) -> None:
    results = tuple(results)
    if dimensions == 2:
        fig, axes = plt.subplots(len(results), 4, figsize=(15.5, 7.4), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row, result in enumerate(results):
            for layer in range(4):
                ax = axes[row, layer]
                _draw_cycle_2d(ax, result.site_points_2d[layer])
                metrics = result.site_metrics[layer]
                ax.set_title(
                    f"{LAYER_LABELS[layer]} · {metrics['shape_read']}\n"
                    f"neighbor {metrics['calendar_neighbor_recall']:.2f} · "
                    f"p={metrics['cycle_length_permutation_p']:.3f}",
                    fontsize=10,
                )
                if layer == 0:
                    ax.set_ylabel(
                        f"seed {result.seed}\nblock {result.selected_block}",
                        fontsize=10,
                    )
        fig.suptitle(
            "Month geometry in the most responsive BSC block\n"
            "Per-layer encoder contributions; one shared rotation per seed, calendar order connected",
            fontsize=15,
        )
        _save_figure(fig, output)
        return
    fig = plt.figure(figsize=(17, 8.4), constrained_layout=True)
    for row, result in enumerate(results):
        for layer in range(4):
            ax = fig.add_subplot(len(results), 4, row * 4 + layer + 1, projection="3d")
            _draw_cycle_3d(ax, result.site_points_3d[layer])
            ax.set_title(
                f"seed {result.seed} · {LAYER_LABELS[layer]}\n"
                f"{result.site_metrics[layer]['shape_read']}",
                fontsize=10,
            )
    fig.suptitle(
        "3D month geometry in the most responsive BSC block\n"
        "Per-layer encoder contributions; panel RMS normalized, calendar order connected",
        fontsize=15,
    )
    _save_figure(fig, output)


def _plot_joint(results: Iterable[SeedGeometry], output: Path, *, dimensions: int) -> None:
    results = tuple(results)
    if dimensions == 2:
        fig, axes = plt.subplots(1, len(results), figsize=(11, 5.2), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, result in zip(axes, results, strict=True):
            _draw_cycle_2d(ax, result.joint_points_2d)
            metrics = result.joint_metrics
            ax.set_title(
                f"seed {result.seed} · block {result.selected_block} · {metrics['shape_read']}\n"
                f"neighbor={metrics['calendar_neighbor_recall']:.2f}, "
                f"closure={metrics['closure_ratio']:.2f}, "
                f"p={metrics['cycle_length_permutation_p']:.3f}",
                fontsize=10,
            )
        fig.suptitle(
            "Joint all-layer BSC month geometry\n"
            "Most month-responsive learned block in each independent seed",
            fontsize=15,
        )
        _save_figure(fig, output)
        return
    fig = plt.figure(figsize=(11, 5.6), constrained_layout=True)
    for index, result in enumerate(results):
        ax = fig.add_subplot(1, len(results), index + 1, projection="3d")
        _draw_cycle_3d(ax, result.joint_points_3d)
        ax.set_title(
            f"seed {result.seed} · block {result.selected_block}\n"
            f"{result.joint_metrics['shape_read']}",
            fontsize=10,
        )
    fig.suptitle("3D joint all-layer BSC month geometry", fontsize=15)
    _save_figure(fig, output)


def _plot_raw(raw_geometry: dict[str, Any], output: Path, *, dimensions: int) -> None:
    points = raw_geometry[f"points_{dimensions}d"]
    if dimensions == 2:
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), constrained_layout=True)
        for layer, ax in enumerate(axes):
            _draw_cycle_2d(ax, points[layer])
            metrics = raw_geometry["metrics"][layer]
            ax.set_title(
                f"{LAYER_LABELS[layer]} · {metrics['shape_read']}\n"
                f"neighbor {metrics['calendar_neighbor_recall']:.2f} · "
                f"2D variance {metrics['projection_variance_2d']:.1%}",
                fontsize=10,
            )
        fig.suptitle(
            "Raw GPT-2 residual month geometry\n"
            "Layerwise PCA, shape normalized, orthogonally aligned to block 9",
            fontsize=15,
        )
        _save_figure(fig, output)
        return
    fig = plt.figure(figsize=(16, 4.4), constrained_layout=True)
    for layer in range(4):
        ax = fig.add_subplot(1, 4, layer + 1, projection="3d")
        _draw_cycle_3d(ax, points[layer])
        metrics = raw_geometry["metrics"][layer]
        ax.set_title(
            f"{LAYER_LABELS[layer]} · {metrics['shape_read']}\n"
            f"3D variance {metrics['projection_variance_3d']:.1%}",
            fontsize=10,
        )
    fig.suptitle(
        "3D raw GPT-2 residual month geometry\n"
        "Layerwise PCA, shape normalized, orthogonally aligned to block 9",
        fontsize=15,
    )
    _save_figure(fig, output)


def _plot_gallery(results: Iterable[SeedGeometry], output: Path) -> None:
    results = tuple(results)
    fig, axes = plt.subplots(len(results), 3, figsize=(12.5, 7.3), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, result in enumerate(results):
        for column, feature in enumerate(result.gallery):
            ax = axes[row, column]
            _draw_cycle_2d(ax, feature["points"])
            metrics = feature["metrics"]
            ax.set_title(
                f"block {feature['block']} · {metrics['shape_read']}\n"
                f"selected {feature['selection_frequency']:.1%} · "
                f"neighbor {metrics['calendar_neighbor_recall']:.2f}",
                fontsize=10,
            )
            if column == 0:
                ax.set_ylabel(f"seed {result.seed}", fontsize=10)
    fig.suptitle(
        "Top month-responsive joint BSC blocks\n"
        "Independent ranking within each seed; calendar order connected",
        fontsize=15,
    )
    _save_figure(fig, output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw, capture = _capture_month_activations(device)
    raw_result = _raw_geometry(raw)
    seed_results = []
    for seed, deployable in enumerate(args.deployable):
        seed_results.append(
            _seed_geometry(
                seed=seed,
                deployable=Path(deployable),
                raw=raw,
                device=device,
            )
        )

    _plot_bsc_layers(seed_results, output / "month_bsc_layers_2d.png", dimensions=2)
    _plot_bsc_layers(seed_results, output / "month_bsc_layers_3d.png", dimensions=3)
    _plot_joint(seed_results, output / "month_bsc_joint_2d.png", dimensions=2)
    _plot_joint(seed_results, output / "month_bsc_joint_3d.png", dimensions=3)
    _plot_raw(raw_result, output / "month_raw_residual_layers_2d.png", dimensions=2)
    _plot_raw(raw_result, output / "month_raw_residual_layers_3d.png", dimensions=3)
    _plot_gallery(seed_results, output / "month_bsc_feature_gallery_2d.png")

    report = {
        "schema": "bsc-month-geometry-v1",
        "analysis_contract": {
            "prompt_balance": "12 templates x 12 one-token month names",
            "template_control": "subtract each template's across-month mean before averaging",
            "block_selection": (
                "within-seed month contrast weighted by absolute between-month energy "
                "and BatchTopK selection frequency"
            ),
            "bsc_layer_alignment": (
                "per-site encoder contributions projected through one common PCA basis "
                "per seed; no independent layer rotation"
            ),
            "plot_scaling": "each panel centered and RMS-normalized for shape comparison",
            "cycle_test": "20,000 fixed-seed label permutations of closed calendar path length",
            "shape_read": (
                "exploratory: ring requires neighbor recall >=0.625, permutation p<=0.05, "
                "closure<=1.75, and radial CV<=0.35"
            ),
        },
        "capture": capture,
        "raw_residual": raw_result,
        "seeds": [
            {
                "seed": result.seed,
                "deployable": str(args.deployable[index]),
                "selected_block": result.selected_block,
                "top_blocks": result.top_blocks,
                "site_points_2d": result.site_points_2d,
                "site_points_3d": result.site_points_3d,
                "joint_points_2d": result.joint_points_2d,
                "joint_points_3d": result.joint_points_3d,
                "site_contribution_rms": result.site_rms,
                "joint_rms": result.joint_rms,
                "site_metrics": result.site_metrics,
                "joint_metrics": result.joint_metrics,
                "gallery": result.gallery,
            }
            for index, result in enumerate(seed_results)
        ],
        "figures": sorted(path.name for path in output.glob("*.png")),
    }
    report_path = output / "month_geometry_metrics.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployable",
        action="append",
        required=True,
        help="Deployable-codec path, once per independent seed.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    for seed in report["seeds"]:
        metrics = seed["joint_metrics"]
        print(
            f"seed {seed['seed']}: block {seed['selected_block']} "
            f"{metrics['shape_read']} "
            f"(neighbor={metrics['calendar_neighbor_recall']:.3f}, "
            f"closure={metrics['closure_ratio']:.3f}, "
            f"p={metrics['cycle_length_permutation_p']:.5f})"
        )


if __name__ == "__main__":
    main()
