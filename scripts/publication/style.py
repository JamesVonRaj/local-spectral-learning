"""Shared style for all publication figures.

Single source of truth for physical figure sizes, typography, the
initial/learned/uniform/window encoding table, panel labels, in-panel
annotation tags, and a width-asserting save helper.

Geometry is measured from the REVTeX 4.2 PRL reprint layout
(\\columnwidth = 246 pt, \\textwidth = 510 pt, TeX points):
every figure is created at its final printed size and included by LaTeX
at natural width, so fonts render exactly at the sizes set here.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# --- Physical sizes (inches; TeX pt / 72.27) --------------------------------
COL_W = 246.0 / 72.27   # 3.404 in, single column
TEXT_W = 510.0 / 72.27  # 7.057 in, full text width (figure*)

# --- Typography floors (points at print size) --------------------------------
BASE_SIZE = 8.5   # axis labels, legends, annotations
TICK_SIZE = 8.0   # tick labels; nothing anywhere may render below 8 pt
PANEL_SIZE = 9.0  # bold panel letters

# --- Encoding table (sampled from the existing figures) ----------------------
GRAY = "#9ca3af"    # initial spectrum/state: light gray, solid, under everything
GRAY_DARK = "#6b7280"  # text color when direct-labeling the gray series
BLUE = "#2b6cb0"    # learned: solid, on top
DARK = "#111111"    # uniform control: black, dashed
WINDOW = "#e76f51"  # prescribed spectral-window shading ...
WINDOW_ALPHA = 0.18  # ... at this alpha, identical in every figure
RED = "#c53030"     # markers/guides (t_s, t_ell, lambda*)
GREEN = "#2f855a"
PURPLE = "#6b46c1"

INITIAL_KW = dict(color=GRAY)
LEARNED_KW = dict(color=BLUE)
UNIFORM_KW = dict(color=DARK, ls="--")

# One annotation-tag style for every in-panel tag in every figure:
# a borderless white knockout so text stays legible over plotted ink
# without reading as a framed UI chip.
TAG_BOX = dict(boxstyle="round,pad=0.25", facecolor="white",
               edgecolor="none", alpha=0.85)


def style() -> None:
    """Apply the manuscript rc style. Call once per figure function."""
    plt.rcParams.update({
        "font.family": "serif",
        # Keep the non-TeX fallback aligned with the manuscript typography.
        "font.serif": ["LM Roman 10", "Latin Modern Roman",
                       "Computer Modern Roman"],
        # Render plot text with the same Computer Modern TeX stack as REVTeX.
        # This also avoids Matplotlib selecting the wrong Latin Modern optical
        # master when several sizes share one internal family name.
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        "mathtext.fontset": "cm",
        "font.size": BASE_SIZE,
        "axes.labelsize": BASE_SIZE,
        "axes.titlesize": BASE_SIZE,
        "legend.fontsize": BASE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "axes.linewidth": 0.6,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    })


def shade_window(ax, lo: float, hi: float, *, axis: str = "x",
                 zorder: float = 0, hatch: str = "////") -> None:
    """Shade a target interval and outline both bounds.

    The fine boundary lines preserve the spectral-window encoding in grayscale
    and when a dense curve or histogram visually overwhelms the pale fill.
    """
    if axis == "x":
        ax.axvspan(lo, hi, facecolor=WINDOW, alpha=WINDOW_ALPHA,
                   edgecolor=WINDOW, hatch=hatch, lw=0.45, zorder=zorder)
        for value in (lo, hi):
            ax.axvline(value, color=WINDOW, alpha=0.85, lw=0.55,
                       ls=":", zorder=zorder + 0.1)
    elif axis == "y":
        ax.axhspan(lo, hi, facecolor=WINDOW, alpha=WINDOW_ALPHA,
                   edgecolor=WINDOW, hatch=hatch, lw=0.45, zorder=zorder)
        for value in (lo, hi):
            ax.axhline(value, color=WINDOW, alpha=0.85, lw=0.55,
                       ls=":", zorder=zorder + 0.1)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def panel_label(ax, letter: str, *, x_ref=None) -> None:
    """Bold lowercase '(a)' at a consistent offset outside the top-left corner.

    `x_ref`: optional axes whose left edge supplies the horizontal position
    (for aspect-locked panels whose box does not span the column).
    """
    trans = ax.transAxes if x_ref is None else \
        blended_transform_factory(x_ref.transAxes, ax.transAxes)
    ax.annotate(rf"\textbf{{({letter})}}", xy=(0.0, 1.0), xycoords=trans,
                xytext=(-4.0, 3.0), textcoords="offset points",
                fontsize=PANEL_SIZE, fontweight="bold",
                ha="right", va="bottom", annotation_clip=False)


def panel_tag(ax, text: str, *, loc: str = "upper left", color=None,
              fontsize: float = BASE_SIZE):
    """In-panel annotation tag in the single unified style."""
    positions = {
        "upper left": (0.03, 0.96, "left", "top"),
        "upper right": (0.97, 0.96, "right", "top"),
        "lower left": (0.03, 0.05, "left", "bottom"),
        "lower right": (0.97, 0.05, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=fontsize, color=color or DARK, bbox=TAG_BOX)


def _pdf_width_in(path: Path) -> float:
    out = subprocess.run(["pdfinfo", str(path)], check=True,
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size:"):
            return float(line.split()[2]) / 72.0  # PDF pt -> in
    raise RuntimeError(f"no page size in pdfinfo output for {path}")


def savefig(fig, out_dir: Path, name: str) -> None:
    """Save PDF+PNG tightly cropped and assert the width survived within 1%."""
    intended = fig.get_size_inches()[0]
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    actual = _pdf_width_in(pdf_path)
    if abs(actual - intended) / intended > 0.01:
        raise RuntimeError(
            f"{name}.pdf width {actual:.3f} in deviates >1% from intended "
            f"{intended:.3f} in; adjust the layout instead of letting LaTeX scale"
        )
