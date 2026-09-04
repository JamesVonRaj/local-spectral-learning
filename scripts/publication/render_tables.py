"""Render supplemental tables and their lightweight validation checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mechanical.learners import build_K
from mechanical.topology import make_network
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply

from publication.bloch_gap import (
    PeriodicVectorCell,
    band_frequencies,
)
from publication.paths import FIGURE_DIR, dataset
from publication.two_time import (
    phi_values,
    positive_rates,
    target_times,
    target_window,
    window_count_and_ratio,
)

FIG_DIR = FIGURE_DIR
OUT_DIR = dataset("prl_v2")
E1_DIR = dataset("prl_e1_two_time")
READINESS_DIR = dataset("prl_readiness")
BANDGAP_DIR = dataset("prl_bandgap")
VECTOR_DIR = dataset("prl_vector_periodic")
NONSPATIAL_DIR = dataset("prl_nonspatial")

GRAY = "#9ca3af"
BLUE = "#2b6cb0"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"
DARK = "#111111"
BAND = "#e76f51"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def style() -> None:
    plt.rcParams.update({
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.4,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "figure.dpi": 180,
        "savefig.dpi": 360,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="top", ha="left")


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def savefig(fig, name: str, *, tight=True) -> None:
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.015} if tight else {}
    fig.savefig(FIG_DIR / f"{name}.pdf", **kwargs)
    fig.savefig(FIG_DIR / f"{name}.png", **kwargs)
    plt.close(fig)


def draw_arrow(ax, xy0, xy1, color=DARK, lw=0.9, style="-|>", ms=8) -> None:
    ax.add_patch(FancyArrowPatch(
        xy0, xy1, arrowstyle=style, mutation_scale=ms, lw=lw,
        color=color, shrinkA=2, shrinkB=2,
    ))


def add_node(ax, xy, text="", fc="white", ec=DARK, r=0.035, color_text=DARK) -> None:
    ax.add_patch(Circle(xy, r, facecolor=fc, edgecolor=ec, lw=0.8))
    if text:
        ax.text(*xy, text, ha="center", va="center", fontsize=5.6, color=color_text)


def figure1_principle() -> None:
    style()
    fig = plt.figure(figsize=(3.375, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    # Panel a: schematics.
    ax_a.set_axis_off()
    panel_label(ax_a, "a")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.text(0.24, 0.98, "teacher-sourced", ha="center", va="top", fontweight="bold")
    ax_a.text(0.74, 0.98, "self-sourced", ha="center", va="top", fontweight="bold")
    ax_a.plot([0.50, 0.50], [0.05, 0.95], color="#dddddd", lw=0.8)

    # Teacher side.
    for xy in [(0.10, 0.72), (0.18, 0.62), (0.26, 0.72), (0.34, 0.62),
               (0.18, 0.42), (0.30, 0.42)]:
        add_node(ax_a, xy, fc="#f7fafc")
    for p, q in [((0.10, 0.72), (0.18, 0.62)), ((0.18, 0.62), (0.26, 0.72)),
                 ((0.26, 0.72), (0.34, 0.62)), ((0.18, 0.62), (0.18, 0.42)),
                 ((0.26, 0.72), (0.30, 0.42)), ((0.18, 0.42), (0.30, 0.42))]:
        ax_a.plot([p[0], q[0]], [p[1], q[1]], color=GRAY, lw=0.8)
    ax_a.text(0.08, 0.84, "input", color=BLUE, fontsize=6)
    ax_a.text(0.33, 0.84, "output", color=GREEN, fontsize=6, ha="right")
    ax_a.text(0.37, 0.62, r"target $y$", color=RED, fontsize=6, va="center")
    draw_arrow(ax_a, (0.35, 0.61), (0.30, 0.50), color=RED)
    ax_a.text(0.33, 0.36, "error", color=RED, fontsize=6, ha="center")
    draw_arrow(ax_a, (0.31, 0.40), (0.20, 0.58), color=RED, style="<|-")
    ax_a.text(0.245, 0.27, "second channel carries\nteacher error inward",
              ha="center", va="top", fontsize=5.7)

    # Self-sourced side.
    pts = [(0.60, 0.73), (0.69, 0.61), (0.79, 0.72), (0.88, 0.60),
           (0.65, 0.43), (0.82, 0.43)]
    for xy in pts:
        add_node(ax_a, xy, fc="#f7fafc")
    for p, q in [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]),
                 (pts[1], pts[4]), (pts[2], pts[5]), (pts[4], pts[5])]:
        ax_a.plot([p[0], q[0]], [p[1], q[1]], color=GRAY, lw=0.8)
    ax_a.text(0.56, 0.86, "noise probe\nall nodes", color=BLUE, fontsize=5.4,
              ha="left")
    draw_arrow(ax_a, (0.62, 0.80), (0.67, 0.65), color=BLUE)
    ax_a.text(0.74, 0.88, r"measure $u$", color=BLUE, fontsize=5.8, ha="center")
    draw_arrow(ax_a, (0.73, 0.80), (0.75, 0.69), color=BLUE)
    ax_a.text(0.92, 0.82, r"re-drive with $u$", color=PURPLE, fontsize=5.4,
              ha="right")
    draw_arrow(ax_a, (0.86, 0.77), (0.80, 0.66), color=PURPLE)
    ax_a.text(0.89, 0.50, r"measure $v$", color=PURPLE, fontsize=6, ha="right")
    ax_a.text(0.74, 0.31, "edge-local update",
              ha="center", va="center", fontsize=5.7)

    # Edge inset on self-sourced side.
    y = 0.17
    ax_a.plot([0.58, 0.90], [y, y], color=DARK, lw=1.0)
    add_node(ax_a, (0.58, y), "i", r=0.027)
    add_node(ax_a, (0.90, y), "j", r=0.027)
    ax_a.annotate("", xy=(0.69, y + 0.06), xytext=(0.60, y + 0.06),
                  arrowprops=dict(arrowstyle="<->", lw=0.7, color=BLUE))
    ax_a.annotate("", xy=(0.88, y - 0.06), xytext=(0.75, y - 0.06),
                  arrowprops=dict(arrowstyle="<->", lw=0.7, color=PURPLE))
    ax_a.text(0.645, y + 0.08, r"$u_i-u_j$", color=BLUE, fontsize=5.5, ha="center")
    ax_a.text(0.815, y - 0.10, r"$v_i-v_j$", color=PURPLE, fontsize=5.5, ha="center")
    ax_a.text(0.74, y - 0.15,
              r"$g_e=-\frac{4r_e}{L_e}(v_i-v_j)(u_i-u_j)$",
              ha="center", fontsize=5.3)

    # Panel b: parity plot.
    panel_label(ax_b, "b")
    w2 = 1.0
    lam = np.linspace(0.25, 1.75, 1600)
    mask = np.abs(lam - w2) > 0.018
    lam_m = lam[mask]
    phi1 = 1.0 / (lam_m - w2)
    phi2 = 1.0 / (lam_m - w2) ** 2
    ax_b.axvspan(0.90, 1.10, color=BAND, alpha=0.16, lw=0)
    ax_b.plot(lam_m, np.clip(phi1 / 9.0, -1.15, 1.15),
              color=DARK, ls="--", lw=1.2, label=r"$p=1$")
    ax_b.plot(lam_m, np.clip(phi2 / 80.0, 0.0, 1.2),
              color=BLUE, lw=1.5, label=r"$p=2$")
    ax_b.axhline(0, color="#cccccc", lw=0.6)
    ax_b.axvline(w2, color=RED, lw=0.8, ls=":")
    ax_b.set_xlim(0.25, 1.75)
    ax_b.set_ylim(-1.15, 1.22)
    ax_b.set_xlabel(r"eigenvalue $\lambda$")
    ax_b.set_ylabel("spectral potential (scaled)")
    ax_b.legend(frameon=False, loc="upper right", ncol=2)
    ax_b.annotate("repel", xy=(0.84, 0.48), xytext=(0.58, 0.82),
                  arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
                  color=BLUE, fontsize=6.2)
    ax_b.annotate("repel", xy=(1.16, 0.48), xytext=(1.34, 0.82),
                  arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
                  color=BLUE, fontsize=6.2)
    ax_b.annotate("below mode\npulled in", xy=(0.90, -0.56), xytext=(0.48, -0.92),
                  arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
                  color=RED, fontsize=6.0, ha="left")
    ax_b.text(0.30, 1.05, "forward-only rule = p=1", fontsize=6.2,
              bbox=dict(facecolor="white", edgecolor="#dddddd", pad=1.5))
    for x in [1.40, 1.48, 1.56, 1.64]:
        ax_b.add_patch(Rectangle((x - 0.015, -1.02), 0.03, 0.16,
                                 facecolor=GRAY, edgecolor=DARK, lw=0.4))
        draw_arrow(ax_b, (x, -1.08), (x, -0.86), color=DARK, lw=0.55, ms=6)
    ax_b.text(1.52, -0.74, "all edges\nstiffen", ha="center", fontsize=5.8)
    savefig(fig, "fig1_principle")


def figure2_scalar_gaps() -> None:
    style()
    ex = load_npz(BANDGAP_DIR / "fig1_exemplar.npz")
    controls = load_npz(READINESS_DIR / "expanded_controls_real.npz")

    fig = plt.figure(figsize=(3.375, 5.55), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.12, 0.82, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    f0 = positive_rates(ex["f0"])
    ff = positive_rates(ex["ff"])
    wlo, whi = float(ex["wlo"]), float(ex["whi"])
    bins = np.linspace(0, max(f0.max(), ff.max()) * 1.02, 95)
    panel_label(ax_a, "a")
    ax_a.axvspan(wlo, whi, color=BAND, alpha=0.18, lw=0)
    ax_a.hist(f0, bins=bins, color=GRAY, alpha=0.40, label="initial")
    ax_a.hist(ff, bins=bins, histtype="stepfilled", color=BLUE, alpha=0.68,
              label="learned")
    ax_a.set_xlim(0, max(4.2, whi * 1.35))
    ax_a.set_ylabel("mode count")
    ax_a.set_xlabel(r"frequency $\omega_n$")
    ax_a.set_title(rf"spectral window cleared: {int(ex['n_in_initial'])}$\to${int(ex['n_in_final'])}")
    ax_a.legend(frameon=False, loc="upper right")

    panel_label(ax_b, "b")
    samples = np.asarray(ex["nin_samples"], dtype=float)
    steps = np.r_[0.0, samples[:, 0], float(ex["n_steps"])]
    counts = np.r_[float(ex["n_in_initial"]), samples[:, 1], float(ex["n_in_final"])]
    ax_b.plot(steps, counts, color=BLUE, lw=1.5)
    ax_b.scatter(steps[::8], counts[::8], s=8, color=BLUE, zorder=3)
    ax_b.set_xlabel("training step")
    ax_b.set_ylabel("modes in window")
    ax_b.set_ylim(-5, max(counts) * 1.08)

    panel_label(ax_c, "c")
    records = controls["records"]
    order = [
        ("adjoint", "full\nrule"),
        ("forward_only", "p=1\nill-posed"),
        ("shuffled", "edge\nshuffled"),
        ("uniform_same_mean_stiffness", "uniform"),
        ("inflation_only", "inflation\nonly"),
    ]
    vals, errs = [], []
    for mode, _ in order:
        if mode == "uniform_same_mean_stiffness":
            sample = np.asarray(controls["uniform_same_mean_stiffness_nin_samples"], dtype=float)[:, 1]
        else:
            sample = records[records["mode"] == mode]["n_in_final"].astype(float)
        vals.append(float(np.mean(sample)))
        errs.append(float(np.std(sample)))
    x = np.arange(len(order))
    colors = [BLUE] + ["white"] * (len(order) - 1)
    bars = ax_c.bar(x, vals, yerr=errs, width=0.66, color=colors,
                    edgecolor=[BLUE, DARK, DARK, DARK, DARK], capsize=2, lw=0.9)
    for b in bars[1:]:
        b.set_linestyle("--")
        b.set_hatch("//")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([lab for _, lab in order])
    ax_c.set_ylabel("final modes in window")
    ax_c.set_ylim(0, max(vals) * 1.35)
    ax_c.text(1, vals[1] + errs[1] + 1.2, "theory:\nill-posed",
              ha="center", va="bottom", fontsize=5.6, color=RED)
    savefig(fig, "fig2_scalar_gaps")


def cell_from_npz(data: dict) -> PeriodicVectorCell:
    return PeriodicVectorCell(
        size=int(data["size"]),
        seed=int(data["seed"]),
        pos=np.asarray(data["pos"], dtype=float),
        box=np.asarray(data["box"], dtype=float),
        edges=np.asarray(data["edges"], dtype=int),
        offsets=np.asarray(data["offsets"], dtype=int),
        vectors=np.asarray(data["vectors"], dtype=float),
        lengths=np.asarray(data["lengths"], dtype=float),
        directions=np.asarray(data["directions"], dtype=float),
    )


def gamma_x_m_gamma(n_per_segment: int = 70):
    pts = [np.array([0.0, 0.0]), np.array([np.pi, 0.0]),
           np.array([np.pi, np.pi]), np.array([0.0, 0.0])]
    labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    kpts, xpos, xticks = [], [], [0.0]
    dist = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = np.linspace(0.0, 1.0, n_per_segment, endpoint=False)
        for s in seg:
            k = (1 - s) * a + s * b
            if kpts:
                dist += float(np.linalg.norm(k - kpts[-1]))
            kpts.append(k)
            xpos.append(dist)
        dist += float(np.linalg.norm(b - kpts[-1]))
        kpts.append(b)
        xpos.append(dist)
        xticks.append(dist)
    return np.asarray(kpts), np.asarray(xpos), xticks, labels


def draw_cell(ax, cell: PeriodicVectorCell, radii: np.ndarray) -> None:
    pos = cell.pos
    segs, vals, widths = [], [], []
    for e, (i, _j) in enumerate(cell.edges):
        p = pos[i]
        q = p + cell.vectors[e]
        if np.all(q >= -0.15) and np.all(q <= cell.box + 0.15):
            segs.append([p, q])
            vals.append(radii[e])
            widths.append(0.4 + 1.25 * (radii[e] - radii.min()) / (radii.max() - radii.min() + 1e-12))
    lc = LineCollection(segs, array=np.asarray(vals), cmap="viridis",
                        linewidths=widths, alpha=0.95)
    ax.add_collection(lc)
    ax.scatter(pos[:, 0], pos[:, 1], s=4, color=DARK, zorder=3)
    ax.add_patch(Rectangle((0, 0), cell.box[0], cell.box[1],
                           facecolor="none", edgecolor=DARK, lw=0.7))
    ax.set_xlim(-0.1, cell.box[0] + 0.1)
    ax.set_ylim(-0.1, cell.box[1] + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("learned cell radii")


def figure3_bloch() -> None:
    style()
    data = load_npz(VECTOR_DIR / "vector_periodic_s5_net0_train0_c50_w10.npz")
    cell = cell_from_npz(data)
    radii = np.asarray(data["radii"], dtype=float)
    kpath, xpos, xticks, labels = gamma_x_m_gamma()
    f0 = band_frequencies(cell, np.ones(cell.n_edges), kpath)
    ff = band_frequencies(cell, radii, kpath)
    bz = json.loads((VECTOR_DIR / "bz_convergence.json").read_text())
    size_scan = json.loads((VECTOR_DIR / "size_scan_summary.json").read_text())
    gap = bz["records"][0]["band_gap"]
    gap_lo = float(gap["lower_edge"])
    gap_hi = float(gap["upper_edge"])
    norm_gap = float(gap["normalized_gap"])
    lower_band = int(gap["lower_band_1based"])
    upper_band = int(gap["upper_band_1based"])
    gap_band_indices = {lower_band - 1, upper_band - 1}

    fig = plt.figure(figsize=(7.05, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[0.92, 1.48],
                          height_ratios=[1.18, 0.78, 0.78])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_b = fig.add_subplot(gs[:, 1])

    panel_label(ax_a, "a")
    draw_cell(ax_a, cell, radii)

    panel_label(ax_b, "b")
    ax_b.axhspan(gap_lo, gap_hi, color=BAND, alpha=0.16, lw=0)
    for n in range(f0.shape[1]):
        ax_b.plot(xpos, f0[:, n], color=GRAY, lw=0.45, alpha=0.35)
        lw = 0.75 if n in gap_band_indices else 0.48
        alpha = 0.95 if n in gap_band_indices else 0.55
        ax_b.plot(xpos, ff[:, n], color=BLUE, lw=lw, alpha=alpha)
    for t in xticks:
        ax_b.axvline(t, color="#dddddd", lw=0.45)
    ax_b.set_xticks(xticks)
    ax_b.set_xticklabels(labels)
    ax_b.set_xlim(xpos.min(), xpos.max())
    ax_b.set_ylim(0, min(4.2, np.nanmax(ff[:, :42]) * 1.08))
    ax_b.set_ylabel(r"frequency $\omega$")
    ax_b.set_title(
        rf"bands {lower_band}--{upper_band} gap, "
        rf"$\Delta\omega/\omega_{{\rm mid}}={norm_gap:.3f}$"
    )

    panel_label(ax_c, "c")
    grids = np.array([r["grid"] for r in bz["records"]], dtype=int)
    lows = np.array([r["band_gap"]["lower_edge"] for r in bz["records"]], dtype=float)
    highs = np.array([r["band_gap"]["upper_edge"] for r in bz["records"]], dtype=float)
    ax_c.plot(grids, lows, "o-", color=GRAY, ms=3, lw=0.9,
              label=rf"$\omega_{{{lower_band}}}^{{\max}}$")
    ax_c.plot(grids, highs, "o-", color=BLUE, ms=3, lw=0.9,
              label=rf"$\omega_{{{upper_band}}}^{{\min}}$")
    ax_c.fill_between(grids, lows, highs, color=BAND, alpha=0.13)
    ax_c.set_xticks(grids)
    ax_c.set_xticklabels([rf"${g}^2$" for g in grids])
    ax_c.set_xlabel("BZ grid")
    ax_c.set_ylabel("gap edges")
    ax_c.legend(frameon=False, loc="center left", fontsize=5.4)

    panel_label(ax_d, "d")
    sizes = np.array([r["size"] for r in size_scan], dtype=int)
    meds = np.array([r["median"] for r in size_scan], dtype=float)
    mins = np.array([r["min"] for r in size_scan], dtype=float)
    maxs = np.array([r["max"] for r in size_scan], dtype=float)
    ax_d.errorbar(sizes, meds, yerr=[meds - mins, maxs - meds],
                  fmt="s-", color=GREEN, ms=3, lw=0.9, capsize=2)
    ax_d.set_xticks(sizes)
    ax_d.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax_d.set_xlabel("cell size")
    ax_d.set_ylabel(r"median $\Delta\omega/\omega_{\rm mid}$", color=GREEN)
    ax_d.tick_params(axis="y", colors=GREEN)
    ax_d.spines["left"].set_color(GREEN)
    savefig(fig, "fig3_bloch")


def load_e1_seed(seed: int, variant: str = "boundslocal") -> dict:
    path = E1_DIR / f"e1_{variant}_size20_seed{seed:03d}.npz"
    return load_npz(path)


def select_median_e1_seed() -> int:
    summary = json.loads((E1_DIR / "e1_boundslocal_summary.json").read_text())
    rows = sorted(summary["rows"], key=lambda r: float(r["ratio_final"]))
    return int(rows[len(rows) // 2]["seed"])


def draw_twotime_protocol(ax, d: dict) -> None:
    panel_label(ax, "a")
    ts, tl = float(d["t_s"]), float(d["t_l"])
    t = np.linspace(0, 1.15 * tl, 250)
    trace = np.exp(-t / (0.36 * tl)) * (0.75 + 0.18 * np.cos(8 * np.pi * t / tl))
    ax.plot(t, trace, color=BLUE, lw=1.45)
    ax.axvline(ts, color=RED, lw=0.9)
    ax.axvline(tl, color=RED, lw=0.9)
    ax.scatter([ts, tl], np.interp([ts, tl], t, trace), color=RED, s=18, zorder=3)
    ax.text(ts, 1.03 * trace.max(), r"$t_s$", ha="center", va="bottom", color=RED)
    ax.text(tl, 1.03 * trace.max(), r"$t_\ell$", ha="center", va="bottom", color=RED)
    ax.set_ylim(-0.40, 1.10 * trace.max())
    ax.set_yticks([0, 0.4, 0.8])
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.set_xlabel("time")
    ax.set_ylabel(r"edge strain $\Delta_e x(t)$")
    ax.set_title("two local reads")
    ax.text(
        0.03, 0.04,
        r"$dr_e \propto \frac{4r_e}{L_e}$" "\n"
        r"$\times[t_s A_e(t_s)$" "\n"
        r"$-t_\ell A_e(t_\ell)]$" "\n"
        r"$A_e(t)=\langle(\Delta_e x(t))^2\rangle$" "\n"
        "one free relaxation\ntwo clock times",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=4.7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#dddddd"),
    )


def figure4_twotime() -> None:
    style()
    seed = select_median_e1_seed()
    d = load_e1_seed(seed)
    fig, axes = plt.subplots(2, 2, figsize=(3.375, 5.05), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    draw_twotime_protocol(ax_a, d)

    panel_label(ax_b, "b")
    lam_star, ts, tl = float(d["lambda_star"]), float(d["t_s"]), float(d["t_l"])
    wlo, whi = float(d["window_lo"]), float(d["window_hi"])
    lam = np.geomspace(max(lam_star / 25, 1e-4), lam_star * 25, 500)
    phi = phi_values(lam, ts, tl)
    ax_b.plot(lam, phi / phi.max(), color=PURPLE, lw=1.6)
    ax_b.axvspan(wlo, whi, color=BAND, alpha=0.18, lw=0)
    ax_b.axvline(lam_star, color=RED, lw=0.9, ls="--")
    ax_b.text(lam_star, 1.03, r"$\lambda^\ast$", ha="center", va="bottom", color=RED)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0, 1.14)
    ax_b.set_xlabel(r"rate $\lambda$")
    ax_b.set_ylabel(r"$\varphi/\max\varphi$")
    ax_b.set_title(r"$e^{-2t_s\lambda}-e^{-2t_\ell\lambda}$")

    panel_label(ax_c, "c")
    mu0 = positive_rates(d["mu_initial"])
    muf = positive_rates(d["mu_final"])
    muc = positive_rates(d["mu_control"])
    bins = np.geomspace(max(min(mu0.min(), muf.min(), muc.min()) * 0.8, 1e-7),
                        max(mu0.max(), muf.max(), muc.max()) * 1.08, 70)
    ax_c.axvspan(wlo, whi, color=BAND, alpha=0.22, lw=0)
    ax_c.hist(mu0, bins=bins, color=GRAY, alpha=0.36, label="initial")
    ax_c.hist(muc, bins=bins, histtype="step", color=DARK, lw=1.0,
              ls="--", label="uniform")
    ax_c.hist(muf, bins=bins, color=BLUE, alpha=0.66, label="learned")
    ax_c.axvline(lam_star, color=RED, lw=0.8, ls=":")
    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"rate $\lambda_n$")
    ax_c.set_ylabel("count")
    n_i, n_f = int(d["n_initial"]), int(d["n_final"])
    ax_c.text(0.02, 0.98,
              rf"seed {seed}: ${n_i}\to {n_f}$" "\n"
              rf"$\lambda_+/\lambda_-={float(d['ratio_final']):.2g}$",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=5.4,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8))
    ax_c.legend(frameon=False, loc="upper right", fontsize=5.7)

    panel_label(ax_d, "d")
    t = np.asarray(d["curve_times"], dtype=float)
    p0 = np.asarray(d["power_initial"], dtype=float)
    pf = np.asarray(d["power_final"], dtype=float)
    pc = np.asarray(d["power_control"], dtype=float)
    ax_d.loglog(t, p0 / p0[0], color=GRAY, lw=1.2, label="initial")
    ax_d.loglog(t, pc / pc[0], color=DARK, lw=1.0, ls="--", label="uniform")
    ax_d.loglog(t, pf / pf[0], color=BLUE, lw=1.65, label="learned")
    ax_d.axvline(ts, color=RED, lw=0.8, alpha=0.75)
    ax_d.axvline(tl, color=RED, lw=0.8, alpha=0.75)
    ax_d.text(ts, ax_d.get_ylim()[1] / 1.7, r"$t_s$", color=RED, ha="center", va="top")
    ax_d.text(tl, ax_d.get_ylim()[1] / 1.7, r"$t_\ell$", color=RED, ha="center", va="top")
    width = np.log(float(d["gap_final_hi"]) / float(d["gap_final_lo"]))
    ax_d.text(0.05, 0.06,
              rf"$\Delta\ln t=\ln(\lambda_+/\lambda_-)$" "\n"
              rf"$={width:.2g}$",
              transform=ax_d.transAxes, ha="left", va="bottom", fontsize=5.2)
    ax_d.set_xlabel("time")
    ax_d.set_ylabel(r"$\mathbb{E}|x(t)|^2$ norm.")
    ax_d.legend(frameon=False, loc="upper right", fontsize=5.6)
    ax_d.grid(alpha=0.18, which="both", linewidth=0.45)
    savefig(fig, "fig4_twotime")


def write_tex(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def fmt(x, digits=3):
    if np.isinf(x):
        return r"$\infty$"
    return f"{float(x):.{digits}f}"


def load_budget_seed(seed: int) -> dict:
    return load_npz(E1_DIR / f"e1_size20_seed{seed:03d}.npz")


def write_twotime_tables() -> dict:
    report = {}

    # Main local-bounds E1 table.
    summary = json.loads((E1_DIR / "e1_boundslocal_summary.json").read_text())
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"seed & arm & $N_{\rm in}^{0}$ & $N_{\rm in}^{f}$ & $\lambda_-$ & $\lambda_+$ & $\lambda_+/\lambda_-$ \\",
        r"\midrule",
    ]
    learned_counts = []
    ratios = []
    control_counts = []
    control_ratios = []
    for rec in summary["rows"]:
        seed = int(rec["seed"])
        d = load_e1_seed(seed)
        learned_counts.append(int(d["n_final"]))
        ratios.append(float(d["ratio_final"]))
        lines.append(
            f"{seed} & learned & {int(d['n_initial'])} & {int(d['n_final'])} & "
            f"{fmt(float(d['gap_final_lo']), 3)} & {fmt(float(d['gap_final_hi']), 3)} & "
            f"{fmt(float(d['ratio_final']), 3)} \\\\"
        )
        nctrl, rctrl, clo, chi = window_count_and_ratio(
            d["mu_control"], float(d["lambda_star"]),
            float(d["window_lo"]), float(d["window_hi"])
        )
        control_counts.append(nctrl)
        control_ratios.append(rctrl)
        lines.append(
            f"{seed} & uniform & {int(d['n_initial'])} & {nctrl} & "
            f"{fmt(clo, 3)} & {fmt(chi, 3)} & {fmt(rctrl, 3)} \\\\"
        )
    lines += [
        r"\midrule",
        f"median & learned & 88 & {int(np.median(learned_counts))} & -- & -- & {fmt(np.median(ratios), 3)} \\\\",
        f"median & uniform & 88 & {int(np.median(control_counts))} & -- & -- & {fmt(np.median(control_ratios), 3)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_tex(FIG_DIR / "table_twotime_perseed.tex", "\n".join(lines))
    report["twotime_perseed"] = {
        "median_initial": 88,
        "median_final": float(np.median(learned_counts)),
        "median_ratio": float(np.median(ratios)),
        "weakest_ratio": float(np.min(ratios)),
        "control_median_count": float(np.median(control_counts)),
        "control_median_ratio": float(np.median(control_ratios)),
    }

    # Stiffness-budget table.
    summary_b = json.loads((E1_DIR / "e1_summary.json").read_text())
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"seed & arm & $N_{\rm in}^{0}$ & $N_{\rm in}^{f}$ & $\lambda_-$ & $\lambda_+$ & $\lambda_+/\lambda_-$ \\",
        r"\midrule",
    ]
    b_counts, b_ratios, b_ctrl_counts, b_ctrl_ratios = [], [], [], []
    for rec in summary_b["rows"]:
        seed = int(rec["seed"])
        d = load_budget_seed(seed)
        b_counts.append(int(d["n_final"]))
        b_ratios.append(float(d["ratio_final"]))
        lines.append(
            f"{seed} & learned & {int(d['n_initial'])} & {int(d['n_final'])} & "
            f"{fmt(float(d['gap_final_lo']), 3)} & {fmt(float(d['gap_final_hi']), 3)} & "
            f"{fmt(float(d['ratio_final']), 3)} \\\\"
        )
        nctrl, rctrl, clo, chi = window_count_and_ratio(
            d["mu_control"], float(d["lambda_star"]),
            float(d["window_lo"]), float(d["window_hi"])
        )
        b_ctrl_counts.append(nctrl)
        b_ctrl_ratios.append(rctrl)
        lines.append(
            f"{seed} & conserved control & {int(d['n_initial'])} & {nctrl} & "
            f"{fmt(clo, 3)} & {fmt(chi, 3)} & {fmt(rctrl, 3)} \\\\"
        )
    lines += [
        r"\midrule",
        f"median & learned & 10 & {int(np.median(b_counts))} & -- & -- & {fmt(np.median(b_ratios), 3)} \\\\",
        f"median & control & 10 & {int(np.median(b_ctrl_counts))} & -- & -- & {fmt(np.median(b_ctrl_ratios), 3)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_tex(FIG_DIR / "table_twotime_budget.tex", "\n".join(lines))
    report["twotime_budget"] = {
        "median_initial": 10,
        "median_final": float(np.median(b_counts)),
        "median_ratio": float(np.median(b_ratios)),
        "control_median_ratio": float(np.median(b_ctrl_ratios)),
    }
    return report


def spectrum_rates(edges, radii, lengths, n):
    vals = eigh(build_K(edges, radii, lengths, n).toarray(), eigvals_only=True)
    return np.maximum(vals, 0.0)


def run_single_time_control(force=False, n_seeds=5) -> dict:
    out = E1_DIR / "e1_single_time_summary.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    rows = []
    for seed in range(int(n_seeds)):
        path = E1_DIR / f"e1_single_time_size20_seed{seed:03d}.npz"
        if path.exists() and not force:
            d = load_npz(path)
        else:
            pos, edges, lengths, _box = make_network("rand-del", 20, seed)
            n = len(pos)
            radii = np.ones(len(edges))
            mu0 = spectrum_rates(edges, radii, lengths, n)
            lam_star, ts, tl = target_times(mu0, 0.60, 8.0)
            wlo, whi = target_window(lam_star, 0.15)
            rng = np.random.RandomState(40000 + seed)
            i, j = edges[:, 0], edges[:, 1]
            for _step in range(1500):
                K = build_K(edges, radii, lengths, n).tocsc()
                X0 = rng.randn(n, 16)
                X0 -= X0.mean(axis=0, keepdims=True)
                xs = expm_multiply(-ts * K, X0)
                var_s = np.mean((xs[i] - xs[j]) ** 2, axis=1)
                # Descent on tr exp(-2tK): sign-definite stiffening.
                direction = (4.0 * ts * radii / lengths) * var_s
                step = np.clip(0.1 * direction, -0.05, 0.05)
                radii = np.clip(radii + step, 0.5, 2.0)
            muf = spectrum_rates(edges, radii, lengths, n)
            n0, ratio0, _, _ = window_count_and_ratio(mu0, lam_star, wlo, whi)
            nf, ratiof, glo, ghi = window_count_and_ratio(muf, lam_star, wlo, whi)
            d = {
                "seed": np.int64(seed),
                "mu_initial": mu0,
                "mu_final": muf,
                "lambda_star": np.float64(lam_star),
                "window_lo": np.float64(wlo),
                "window_hi": np.float64(whi),
                "n_initial": np.int64(n0),
                "n_final": np.int64(nf),
                "ratio_initial": np.float64(ratio0),
                "ratio_final": np.float64(ratiof),
                "gap_final_lo": np.float64(glo),
                "gap_final_hi": np.float64(ghi),
                "radii": radii.astype(float),
            }
            np.savez_compressed(path, **d)
        rows.append({
            "seed": int(d["seed"]),
            "n_initial": int(d["n_initial"]),
            "n_final": int(d["n_final"]),
            "ratio_final": float(d["ratio_final"]),
            "frac_stiffened": float(np.mean(np.asarray(d["radii"]) > 1.0 + 1e-9)),
            "radius_min": float(np.min(d["radii"])),
            "radius_max": float(np.max(d["radii"])),
        })
    summary = {
        "rows": rows,
        "median_final_count": float(np.median([r["n_final"] for r in rows])),
        "median_frac_stiffened": float(np.median([r["frac_stiffened"] for r in rows])),
    }
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def write_twotime_control_table(force=False) -> dict:
    summary = run_single_time_control(force=force)
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"seed & initial $N_{\rm in}$ & final $N_{\rm in}$ & stiffened edges \\",
        r"\midrule",
    ]
    for r in summary["rows"]:
        lines.append(
            f"{r['seed']} & {r['n_initial']} & {r['n_final']} & "
            f"{100.0 * r['frac_stiffened']:.1f}\\% \\\\"
        )
    lines += [
        r"\midrule",
        f"median & 88 & {summary['median_final_count']:.1f} & "
        f"{100.0 * summary['median_frac_stiffened']:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_tex(FIG_DIR / "table_twotime_controls.tex", "\n".join(lines))
    return summary


def write_nonspatial_table() -> dict:
    records = load_npz(NONSPATIAL_DIR / "ws_screen.npz")["records"]
    lines = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"seed & $N$ & initial $N_{\rm in}$ & final $N_{\rm in}$ & "
        r"$\Delta\omega/\omega_{\rm mid}$ \\",
        r"\midrule",
    ]
    for row in records:
        gap_text = (
            f"{float(row['gap_ratio']):.3f}"
            if int(row["n_in_final"]) == 0 else "--"
        )
        lines.append(
            f"{int(row['seed'])} & {int(row['N'])} & "
            f"{int(row['n_in_initial'])} & {int(row['n_in_final'])} & "
            f"{gap_text} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex(FIG_DIR / "table_nonspatial_gap.tex", "\n".join(lines))
    return {
        "n_cleared": int(np.sum(records["n_in_final"] == 0)),
        "n_total": int(len(records)),
    }


def heat_trace_objective(edges, radii, lengths, n, ts, tl):
    mu = spectrum_rates(edges, radii, lengths, n)
    return float(np.sum(np.exp(-2.0 * ts * mu) - np.exp(-2.0 * tl * mu)))


def twotime_gradient_fd_check(force=False) -> dict:
    out = OUT_DIR / "twotime_fd_check.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    pos, edges, lengths, _box = make_network("rand-del", 5, 7)
    n = len(pos)
    rng = np.random.RandomState(123)
    radii = 0.75 + 0.5 * rng.rand(len(edges))
    mu, psi = eigh(build_K(edges, radii, lengths, n).toarray())
    mu = np.maximum(mu, 0.0)
    rates = positive_rates(mu)
    lam_star = float(np.quantile(rates, 0.60))
    rho = 8.0
    ts = float(np.log(rho) / (2.0 * lam_star * (rho - 1.0)))
    tl = rho * ts
    dphi = -2.0 * ts * np.exp(-2.0 * ts * mu) + 2.0 * tl * np.exp(-2.0 * tl * mu)
    i, j = edges[:, 0], edges[:, 1]
    edge_modes = psi[i, :] - psi[j, :]
    grad = (2.0 * radii / lengths) * (edge_modes ** 2 @ dphi)
    candidates = rng.choice(len(edges), size=min(30, len(edges)), replace=False)
    h = 1e-5
    rel = []
    for e in candidates:
        rp = radii.copy()
        rm = radii.copy()
        rp[e] += h
        rm[e] = max(1e-7, rm[e] - h)
        fd = (
            heat_trace_objective(edges, rp, lengths, n, ts, tl)
            - heat_trace_objective(edges, rm, lengths, n, ts, tl)
        ) / (rp[e] - rm[e])
        rel.append(abs(float(grad[e]) - fd) / (abs(fd) + abs(float(grad[e])) + 1e-14))
    result = {
        "n_edges_checked": int(len(candidates)),
        "median_relative_error": float(np.median(rel)),
        "max_relative_error": float(np.max(rel)),
        "ts": ts,
        "tl": tl,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_all(force=False, tables_only=False) -> dict:
    ensure_dirs()
    if not tables_only:
        figure1_principle()
        figure2_scalar_gaps()
        figure3_bloch()
        figure4_twotime()
    twotime = write_twotime_tables()
    controls = write_twotime_control_table(force=force)
    nonspatial = write_nonspatial_table()
    fd = twotime_gradient_fd_check(force=force)
    produced = ([] if tables_only else [
        "figures/fig1_principle.pdf",
        "figures/fig2_scalar_gaps.pdf",
        "figures/fig3_bloch.pdf",
        "figures/fig4_twotime.pdf",
    ]) + [
        "figures/table_twotime_perseed.tex",
        "figures/table_twotime_controls.tex",
        "figures/table_twotime_budget.tex",
        "figures/table_nonspatial_gap.tex",
    ]
    log = {
        "produced": produced,
        "twotime": twotime,
        "single_time_control": controls,
        "nonspatial": nonspatial,
        "twotime_fd_check": fd,
    }
    (OUT_DIR / "prl_v2_run_log.json").write_text(json.dumps(log, indent=2, sort_keys=True))
    return log


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    p.add_argument("--tables-only", action="store_true",
                   help="write current supplemental tables without legacy v2 figures")
    args = p.parse_args()
    log = run_all(force=args.force, tables_only=args.tables_only)
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
