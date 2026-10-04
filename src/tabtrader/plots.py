"""Matplotlib styling for the notebooks (same validated palette as the web report)."""

from __future__ import annotations

import matplotlib.pyplot as plt

INK, MUTED, GRID = "#14161a", "#5a5f66", "#e4e6e2"
TABPFN, FAST, THINK = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (CVD-validated)
CLASSIC = "#9aa0a6"                                      # every classic model: gray, labelled
POS, NEG = "#2a78d6", "#e34948"

COLORS = {"TabPFN-3.5": TABPFN, "TabPFN-3.5-Fast": FAST, "TabPFN-3.5-Thinking": THINK,
          "Market price": INK}


def color(model: str) -> str:
    return COLORS.get(model, CLASSIC)


def style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4), "figure.dpi": 110, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.titleweight": "bold", "axes.titlesize": 11,
        "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2,
    })
