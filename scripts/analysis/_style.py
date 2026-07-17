"""Shared figure style for the interim artifact analysis (dataviz defaults).

Reference palette (validated): categorical slots in fixed order; ordinal
blue ramp for depth-ordered series; light surface. Figures are static PNG
(matplotlib) with a companion interactive HTML (plotly) where 3D helps.
"""

from __future__ import annotations

import matplotlib as mpl

# Categorical slots, fixed order (never cycled past what's assigned).
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834",
       "#4a3aa7", "#e34948"]
# Ordinal blue ramp, light-mode band 250..650 (depth-ordered series).
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#104281"]
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SITES = [7, 10, 13, 17, 20, 22]
SITE_COLOR = dict(zip(SITES, BLUE_RAMP))

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })


def cyclic_colors(n: int):
    """Perceptually ordered cyclic colors for ring classes (identity is
    additionally carried by direct text labels, never color alone)."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np

    return [mcolors.to_hex(cm.twilight(x)) for x in (np.arange(n) / n + 0.03)]
