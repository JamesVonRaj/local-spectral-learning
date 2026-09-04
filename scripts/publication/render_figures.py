"""Generate the primary analysis figures from archived numerical data.

The active figure groups are ``scalar`` (``fig1_flagship``) and ``bloch``
(``fig2_bloch``).  The adaptive-control renderer produces ``fig3_adaptive``.
Older diagnostic groups remain callable for provenance, but are not part of
the canonical artifact rebuild.

The numerical conventions are:
    k_e = r_e^2 / L_e
    K = sum_e k_e q_e q_e^T
    H = K - omega^2 M
    C = u^T u
    g_e = -(4 r_e/L_e) (v_i-v_j) (u_i-u_j)
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from mechanical.learners import build_K, eigenfreqs
from mechanical.objectives import default_objective_terms
from mechanical.regularizers import inflation
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from publication import style as ps
from publication.paths import dataset
from publication.render_tables import (
    BANDGAP_DIR,
    E1_DIR,
    FIG_DIR,
    READINESS_DIR,
    VECTOR_DIR,
    PeriodicVectorCell,
    band_frequencies,
    cell_from_npz,
    gamma_x_m_gamma,
    load_npz,
    select_median_e1_seed,
)
from publication.style import (
    BLUE,
    DARK,
    GRAY,
    GRAY_DARK,
    PURPLE,
    RED,
    panel_label,
    panel_tag,
)
from publication.two_time import phi_values, positive_rates


def savefig(fig, name: str) -> None:
    ps.savefig(fig, FIG_DIR, name)

OUT_DIR = dataset("prl_v3")
EPS_REG = 0.0  # the claimed reciprocal protocol uses the bare operator


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_scalar_network(ax, data: dict) -> LineCollection:
    pos = np.asarray(data["pos"], dtype=float)
    edges = np.asarray(data["edges"], dtype=int)
    radii = np.asarray(data["radii"], dtype=float)
    box = np.asarray(data["box"], dtype=float)
    # Periodic network: draw every edge at its minimum-image representation.
    # Wrap-around edges become two short stubs at the boundary instead of
    # spurious chords across the whole cell.
    d = pos[edges[:, 1]] - pos[edges[:, 0]]
    shift = np.round(d / box) * box
    d_mi = d - shift
    wrap = np.any(shift != 0.0, axis=1)
    # Thinnest edge >= 0.4 pt at print size (figure is included at natural width).
    widths = 0.4 + 1.0 * (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
    segs, vals, ws = [], [], []
    for e, (i, j) in enumerate(edges):
        if wrap[e]:
            stubs = [[pos[i], pos[i] + d_mi[e]], [pos[j], pos[j] - d_mi[e]]]
        else:
            stubs = [[pos[i], pos[j]]]
        for s in stubs:
            segs.append(s)
            vals.append(radii[e])
            ws.append(widths[e])
    lc = LineCollection(segs, array=np.asarray(vals), cmap="viridis",
                        linewidths=ws, alpha=0.95)
    ax.add_collection(lc)
    ax.scatter(pos[:, 0], pos[:, 1], s=0.8, color=DARK, zorder=3)
    frame = Rectangle((0, 0), box[0], box[1], facecolor="none",
                      edgecolor=DARK, lw=0.6, zorder=4)
    ax.add_patch(frame)
    # Boundary stubs end exactly at the cell frame.
    lc.set_clip_path(frame)
    ax.set_xlim(-0.02 * box[0], 1.02 * box[0])
    ax.set_ylim(-0.02 * box[1], 1.02 * box[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return lc


def draw_probe_schematic(ax) -> None:
    """Panel (a): the paired-response protocol as a three-stage strip.

    Stage 1: random forces drive every node; each node reads its response u.
    Stage 2: each node is re-driven by its phase-conjugated measured
    displacement and reads the second response w.  Stage 3: one edge multiplies its two
    measured strains to form its exact raw local gradient signal.
    """
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    # Toy planar graph reused for both driven stages.
    pts0 = np.array([[0.05, 0.45], [0.42, 0.95], [0.95, 0.75],
                     [0.35, 0.05], [0.85, 0.10], [1.30, 0.42]])
    edges = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 5), (3, 4), (4, 5), (2, 4)]
    scale = 1.30
    angles = [150.0, 80.0, 25.0, 215.0, 300.0, 335.0]

    def draw_net(origin, color, lengths):
        pts = np.asarray(origin) + pts0 * scale
        for i, j in edges:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=GRAY, lw=0.9, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=DARK, zorder=3)
        for p, ang, length in zip(pts, angles, lengths):
            d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            ax.annotate("", xy=p + 0.10 * d, xytext=p + (length + 0.16) * d,
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.9, mutation_scale=6))

    y0 = 1.15
    draw_net((0.10, y0), BLUE, [0.30, 0.22, 0.28, 0.20, 0.30, 0.24])
    draw_net((4.30, y0), PURPLE, [0.20, 0.30, 0.18, 0.28, 0.22, 0.30])
    for x_from, x_to in [(2.55, 3.65), (6.60, 7.30)]:
        ax.annotate("", xy=(x_to, 1.95), xytext=(x_from, 1.95),
                    arrowprops=dict(arrowstyle="->", color=GRAY_DARK,
                                    lw=0.8))

    ax.text(0.95, 0.58, "random drive", ha="center", va="center",
            fontsize=7)
    ax.text(0.95, 0.16, r"measure $u_i$", ha="center", va="center",
            fontsize=7)
    ax.text(5.15, 0.57, "re-drive", ha="center", va="center", fontsize=7)
    ax.text(5.15, 0.10, r"$F_i^{(2)}{=}\Lambda u_i^*$", ha="center",
            va="center", fontsize=7)

    # Stage 3: one edge and its two measured strains.
    pi, pj = np.array([7.80, 2.0]), np.array([10.35, 2.0])
    ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color=DARK, lw=1.8,
            solid_capstyle="round", zorder=1)
    ax.scatter([pi[0], pj[0]], [pi[1], pj[1]], s=14, color=DARK, zorder=3)
    ax.text(pi[0] - 0.28, 2.0, r"$i$", ha="right", va="center", fontsize=ps.BASE_SIZE)
    ax.text(pj[0] + 0.28, 2.0, r"$j$", ha="left", va="center", fontsize=ps.BASE_SIZE)
    ax.annotate("", xy=(9.85, 2.55), xytext=(8.30, 2.55),
                arrowprops=dict(arrowstyle="<|-|>", color=BLUE, lw=0.9,
                                mutation_scale=6))
    ax.text(9.08, 2.90, r"$\Delta u_e$", ha="center", va="center",
            color=BLUE, fontsize=ps.BASE_SIZE)
    ax.annotate("", xy=(9.85, 1.45), xytext=(8.30, 1.45),
                arrowprops=dict(arrowstyle="<|-|>", color=PURPLE, lw=0.9,
                                mutation_scale=6))
    ax.text(9.08, 1.10, r"$\Delta w_e$", ha="center", va="center",
            color=PURPLE, fontsize=ps.BASE_SIZE)
    ax.text(9.08, 0.40, r"$g_e\propto-\,\mathrm{Re}(\Delta w_e\Delta u_e)$",
            ha="center", va="center", fontsize=7)


def render_scalar_gap_figure() -> None:
    ps.style()
    ex = load_npz(BANDGAP_DIR / "fig1_exemplar.npz")
    # Protocol schematic strip on top; square network lower left,
    # (c) over (d) on the lower right.
    fig = plt.figure(figsize=(ps.COL_W, 3.7), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.42, 1.0],
                          height_ratios=[1.15, 1.0, 1.0])
    ax_s = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1:, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, 1])

    panel_label(ax_s, "a")
    draw_probe_schematic(ax_s)

    panel_label(ax_a, "b")
    lc = draw_scalar_network(ax_a, ex)
    cbar = fig.colorbar(lc, ax=ax_a, orientation="horizontal",
                        fraction=0.07, pad=0.03, shrink=0.9, aspect=22)
    cbar.set_label(r"radius $r_e$", labelpad=2)
    cbar.ax.tick_params(length=2)

    panel_label(ax_b, "c")
    f0 = positive_rates(ex["f0"])
    ff = positive_rates(ex["ff"])
    wlo, whi = float(ex["wlo"]), float(ex["whi"])
    # This compact panel is intentionally a low-frequency detail around the
    # prescribed spectral window, not a silently clipped rendering of the full spectra.
    detail_hi = max(4.2, whi * 1.35)
    bins = np.linspace(0, detail_hi, 48)
    ps.shade_window(ax_b, wlo, whi, axis="x")
    n0, _, _ = ax_b.hist(f0[f0 <= detail_hi], bins=bins,
                         color=GRAY, alpha=0.45)
    nf, _, _ = ax_b.hist(ff[ff <= detail_hi], bins=bins,
                         color=BLUE, alpha=0.70)
    ax_b.set_xlim(0, detail_hi)
    ax_b.set_ylim(0, 1.22 * max(n0.max(), nf.max()))
    ax_b.set_xticks([0, 2, 4])
    ax_b.set_ylabel("mode count")
    ax_b.set_xlabel(r"frequency $\omega_n$")
    # Direct labels in series colors; the count sits over the window it counts.
    ax_b.text(0.02, 0.92, "learned", color=BLUE, ha="left", fontsize=ps.BASE_SIZE,
              transform=ax_b.transAxes, bbox=ps.TAG_BOX)
    ax_b.text(0.99, 0.74, "initial", color=GRAY_DARK, ha="right", fontsize=ps.BASE_SIZE,
              transform=ax_b.transAxes)
    ax_b.text(0.5 * (wlo + whi), 0.90,
              rf"${int(ex['n_in_initial'])}\to{int(ex['n_in_final'])}$",
              ha="center", color=DARK, fontsize=ps.BASE_SIZE,
              transform=blended_transform_factory(ax_b.transData,
                                                  ax_b.transAxes))

    panel_label(ax_c, "d")
    samples = np.asarray(ex["nin_samples"], dtype=float)
    steps = np.r_[0.0, samples[:, 0], float(ex["n_steps"])]
    counts = np.r_[float(ex["n_in_initial"]), samples[:, 1], float(ex["n_in_final"])]
    ax_c.plot(steps, counts, color=BLUE, lw=1.2)
    # Mark the informative nonzero samples and the cleared endpoint, without
    # laying a distracting row of markers over the long zero-valued tail.
    mark = np.unique(np.r_[np.flatnonzero(counts > 0), len(counts) - 1])
    ax_c.scatter(steps[mark], counts[mark], s=4, color=BLUE, zorder=3)
    n_steps = int(ex["n_steps"])
    ax_c.set_xticks([0, n_steps // 2, n_steps])
    ax_c.set_xlabel("training step")
    ax_c.set_ylabel("window count")
    ax_c.set_ylim(-5, max(counts) * 1.08)
    savefig(fig, "fig1_flagship")


def parity_potential_panel(ax) -> None:
    w2 = 1.0
    lam = np.linspace(0.25, 1.75, 1600)
    delta = lam - w2
    # Measure the two potentials in units set by the half-width of the
    # schematic spectral window.  NaNs at the pole keep Matplotlib from
    # spuriously joining the negative and positive branches.
    scale = 0.10
    phi1 = scale / delta
    phi2 = (scale / delta) ** 2
    pole = np.abs(delta) <= 0.018
    phi1[pole] = np.nan
    phi2[pole] = np.nan
    ps.shade_window(ax, 0.90, 1.10, axis="x")
    # The disconnected branches run naturally out of the frame at the pole.
    ax.plot(lam, phi1, color=DARK, ls="--", lw=1.1)
    ax.plot(lam, phi2, color=BLUE, lw=1.3)
    ax.axhline(0, color="#cccccc", lw=0.6)
    ax.axvline(w2, color=RED, lw=0.7, ls=":")
    ax.set_xlim(0.25, 1.75)
    ax.set_ylim(-1.15, 1.22)
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xlabel(r"eigenvalue $\omega^2$")
    ax.set_ylabel("scaled potential")
    # Direct curve labels replace the legend and the forward-only tag;
    # the caption identifies p=1 with the forward-only rule.
    ax.text(0.29, 0.16, r"$p=2$", color=BLUE, ha="left", fontsize=7)
    ax.text(1.55, 0.30, r"$p=1$", color=DARK, ha="center",
            va="bottom", fontsize=7)


def vectorized_response_step(
    edges: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    n_nodes: int,
    omegas: np.ndarray,
    rng: np.random.RandomState,
    *,
    batch: int,
    mode: str,
) -> tuple[float, np.ndarray]:
    """One deterministic scalar training step with the manuscript gradient."""
    K = build_K(edges, radii, lengths, n_nodes)
    M_sp = diags(np.ones(n_nodes))
    i_e, j_e = edges[:, 0], edges[:, 1]
    two_r_over_L = 2.0 * radii / lengths
    inv_bn = 1.0 / (int(batch) * len(omegas))
    forces = rng.randn(int(batch), n_nodes)
    cost = 0.0
    grad = np.zeros(len(edges), dtype=np.float64)
    for omega in omegas:
        H = (K - omega**2 * M_sp).tocsc()
        solv = splu(H)
        u = solv.solve(forces.T)
        if mode == "adjoint":
            v = solv.solve(u)
        elif mode == "forward_only":
            v = u
        else:
            raise ValueError(f"unknown mode {mode!r}")
        cost += float(np.sum(u * u)) * inv_bn
        du = u[i_e] - u[j_e]
        dv = v[i_e] - v[j_e]
        grad += (-2.0 * inv_bn) * two_r_over_L * np.sum(dv * du, axis=1)
    return cost, grad


def run_scalar_parity_trajectories(force: bool = False) -> dict:
    out = OUT_DIR / "scalar_parity_trajectories.npz"
    if out.exists() and not force:
        data = load_npz(out)
        sampling = str(np.asarray(data.get("frequency_sampling", "all")).item())
        per_step = int(np.asarray(data.get("frequencies_per_step", 0)).item())
        clip = float(np.asarray(data.get("grad_clip", np.nan)).item())
        material_update = str(np.asarray(data.get("material_update", "")).item())
        if (
            sampling == "random_grid" and per_step == 1 and np.isnan(clip)
            and material_update == "radius_euler"
        ):
            check_forward_refill(data)
            return data

    ex = load_npz(BANDGAP_DIR / "fig1_exemplar.npz")
    edges = np.asarray(ex["edges"], dtype=np.int64)
    lengths = np.asarray(ex["lengths"], dtype=np.float64)
    n_nodes = len(np.asarray(ex["pos"]))
    n_steps = 3000
    checkpoint_every = 25
    batch = 10
    n_freq = 8
    alpha = 0.05
    r_min, r_max = 0.5, 2.0
    checkpoint_steps = np.arange(0, n_steps + 1, checkpoint_every, dtype=np.int64)

    r0 = np.ones(len(edges), dtype=np.float64)
    f0 = eigenfreqs(edges, r0, lengths, n_nodes, 1.0)
    terms, wlo, whi = default_objective_terms(f0, n_freq=n_freq)
    omegas = np.array([t.omega for t in terms], dtype=np.float64)
    reg = inflation(0.03)

    spectra = {}
    counts = {}
    costs = {}
    final_radii = {}
    t0 = time.perf_counter()
    for mode in ("adjoint", "forward_only"):
        rng = np.random.RandomState(0)
        frequency_rng = np.random.RandomState(918273)
        radii = r0.copy()
        mode_spectra = np.zeros((len(checkpoint_steps), n_nodes), dtype=np.float64)
        mode_counts = np.zeros(len(checkpoint_steps), dtype=np.int64)
        mode_costs = np.zeros(n_steps, dtype=np.float64)
        cp_idx = 0
        mode_spectra[cp_idx] = f0
        mode_counts[cp_idx] = int(np.sum((f0 > wlo) & (f0 < whi)))
        cp_idx += 1
        for step in range(n_steps):
            step_omegas = omegas[[frequency_rng.randint(len(omegas))]]
            cost, grad = vectorized_response_step(
                edges, radii, lengths, n_nodes, step_omegas, rng,
                batch=batch, mode=mode,
            )
            mode_costs[step] = cost
            grad = grad + reg(radii, lengths)
            radii = np.clip(radii - alpha * grad, r_min, r_max)
            if cp_idx < len(checkpoint_steps) and step + 1 == checkpoint_steps[cp_idx]:
                freqs = eigenfreqs(edges, radii, lengths, n_nodes, 1.0)
                mode_spectra[cp_idx] = freqs
                mode_counts[cp_idx] = int(np.sum((freqs > wlo) & (freqs < whi)))
                cp_idx += 1
        spectra[mode] = mode_spectra
        counts[mode] = mode_counts
        costs[mode] = mode_costs
        final_radii[mode] = radii.copy()

    payload = {
        "steps": checkpoint_steps,
        "adjoint_spectra": spectra["adjoint"],
        "forward_only_spectra": spectra["forward_only"],
        "adjoint_counts": counts["adjoint"],
        "forward_only_counts": counts["forward_only"],
        "adjoint_cost": costs["adjoint"],
        "forward_only_cost": costs["forward_only"],
        "adjoint_radii": final_radii["adjoint"],
        "forward_only_radii": final_radii["forward_only"],
        "f0": f0,
        "wlo": np.float64(wlo),
        "whi": np.float64(whi),
        "omegas": omegas,
        "n_steps": np.int64(n_steps),
        "checkpoint_every": np.int64(checkpoint_every),
        "batch": np.int64(batch),
        "n_freq": np.int64(n_freq),
        "alpha": np.float64(alpha),
        "grad_clip": np.float64(np.nan),
        "gradient_clip_mode": np.asarray("none"),
        "frequency_sampling": np.asarray("random_grid"),
        "frequencies_per_step": np.int64(1),
        "material_update": np.asarray("radius_euler"),
        "regularizer_arm": np.asarray("inflation"),
        "eps_reg": np.float64(EPS_REG),
        "runtime_seconds": np.float64(time.perf_counter() - t0),
    }
    np.savez_compressed(out, **payload)
    data = load_npz(out)
    check_forward_refill(data)
    return data


def check_forward_refill(data: dict) -> None:
    f0 = np.asarray(data["f0"], dtype=float)
    wlo, whi = float(data["wlo"]), float(data["whi"])
    width = whi - wlo
    margin = 1.5 * width
    below = (f0 >= wlo - margin) & (f0 < wlo)
    forward = np.asarray(data["forward_only_spectra"], dtype=float)
    entered = np.any((forward[1:, below] > wlo) & (forward[1:, below] < whi))
    if not bool(entered):
        raise RuntimeError(
            "forward-only trajectory did not show below-window modes refilling "
            "the prescribed spectral window; stopping without tuning"
        )


def _spaced_subset(indices: np.ndarray, n: int) -> np.ndarray:
    """Return at most n deterministic, evenly spaced entries."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return indices
    take = np.rint(np.linspace(0, len(indices) - 1, n)).astype(int)
    return indices[np.unique(take)]


def trajectory_panel(ax, data: dict, key: str, tag: str, *, color: str,
                     x_max: float, x_ticks: list[float],
                     dashed: bool = False) -> None:
    steps = np.asarray(data["steps"], dtype=float)
    spectra = np.asarray(data[key], dtype=float)
    f0 = np.asarray(data["f0"], dtype=float)
    wlo, whi = float(data["wlo"]), float(data["whi"])
    width = whi - wlo
    margin = 1.5 * width
    keep = (f0 > wlo - margin) & (f0 < whi + margin)
    stop = int(np.searchsorted(steps, x_max, side="right"))
    shown_steps = steps[:stop]
    shown_spectra = spectra[:stop]
    ps.shade_window(ax, wlo, whi, axis="y")
    # Keep the full near-window spectrum visible, but sufficiently light that
    # a few representative spectral ranks can carry the mechanism visually.
    segs = [
        np.column_stack([shown_steps, series])
        for series in shown_spectra[:, keep].T
    ]
    lc = LineCollection(segs, colors=color, alpha=0.10, linewidths=0.25,
                        linestyles="--" if dashed else "-", rasterized=True)
    ax.add_collection(lc)

    if key.startswith("adjoint"):
        # Three ranks that begin inside the window, plus the two ranks that
        # bracket the final gap.  These are spectral-rank paths, not claims of
        # eigenvector identity through crossings.
        initially_inside = np.flatnonzero((f0 > wlo) & (f0 < whi))
        highlighted = list(_spaced_subset(initially_inside, 3))
        final = shown_spectra[-1]
        below = np.flatnonzero(final < wlo)
        above = np.flatnonzero(final > whi)
        if len(below):
            highlighted.append(int(below[-1]))
        if len(above):
            highlighted.append(int(above[0]))
    else:
        # Highlight exactly the below-window ranks that enter the target in
        # this forward-only run; this makes the observed refill explicit.
        initially_below = (f0 >= wlo - margin) & (f0 < wlo)
        enters = np.any(
            (shown_spectra > wlo) & (shown_spectra < whi), axis=0
        )
        highlighted = list(
            _spaced_subset(np.flatnonzero(initially_below & enters), 5)
        )

    for idx in dict.fromkeys(highlighted):
        ax.plot(
            shown_steps, shown_spectra[:, idx], color=color,
            lw=0.9, ls="--" if dashed else "-", alpha=0.95, zorder=3,
        )
        ax.scatter(
            [shown_steps[0], shown_steps[-1]],
            [shown_spectra[0, idx], shown_spectra[-1, idx]],
            s=5, color=color, linewidths=0, zorder=4,
        )

    counts_key = "adjoint_counts" if key.startswith("adjoint") else "forward_only_counts"
    counts = np.asarray(data[counts_key], dtype=int)
    panel_tag(ax, rf"{tag}: ${int(counts[0])}\to{int(counts[-1])}$",
              loc="upper left", fontsize=7)
    ax.set_xlim(steps.min(), x_max)
    ax.set_xticks(x_ticks)


def render_parity_figure(force: bool = False) -> dict:
    ps.style()
    traj = run_scalar_parity_trajectories(force=force)
    # Square potential panel on the left, (b) over (c) on the right.
    fig = plt.figure(figsize=(0.62 * ps.TEXT_W, 3.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)

    panel_label(ax_a, "a")
    parity_potential_panel(ax_a)
    ax_a.set_box_aspect(1)
    panel_label(ax_b, "b")
    trajectory_panel(ax_b, traj, "adjoint_spectra", r"$p=2$",
                     color=BLUE, x_max=700, x_ticks=[0, 300, 600],
                     dashed=False)
    panel_label(ax_c, "c")
    trajectory_panel(ax_c, traj, "forward_only_spectra", r"$p=1$",
                     color=DARK, x_max=75, x_ticks=[0, 25, 50, 75],
                     dashed=True)
    ylo = float(traj["wlo"]) - 1.60 * (float(traj["whi"]) - float(traj["wlo"]))
    yhi = float(traj["whi"]) + 1.45 * (float(traj["whi"]) - float(traj["wlo"]))
    ax_b.set_ylim(max(0, ylo), yhi)
    # The panels intentionally use different time ranges, so both must show
    # their own x ticks and labels rather than implying a shared scale.
    ax_b.set_xlabel("training step")
    ax_c.set_xlabel("training step")
    # One y-label centered on the shared axis of the (b,c) pair.
    ax_c.set_ylabel(r"frequency $\omega_n$")
    ax_c.yaxis.set_label_coords(-0.26, 1.02)
    savefig(fig, "fig2_parity")
    return traj


def draw_vector_cell(ax, cell: PeriodicVectorCell, radii: np.ndarray) -> LineCollection:
    """Learned periodic cell, minimum-image: wrap edges drawn as two stubs."""
    pos = cell.pos
    widths = 0.4 + 1.0 * (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
    segs, vals, ws = [], [], []
    for e, (i, j) in enumerate(cell.edges):
        vec = cell.vectors[e]
        if np.any(cell.offsets[e] != 0):
            stubs = [[pos[i], pos[i] + vec], [pos[j], pos[j] - vec]]
        else:
            stubs = [[pos[i], pos[j]]]
        for s in stubs:
            segs.append(s)
            vals.append(radii[e])
            ws.append(widths[e])
    lc = LineCollection(segs, array=np.asarray(vals), cmap="viridis",
                        linewidths=ws, alpha=0.95)
    ax.add_collection(lc)
    ax.scatter(pos[:, 0], pos[:, 1], s=3.5, color=DARK, zorder=3)
    frame = Rectangle((0, 0), cell.box[0], cell.box[1], facecolor="none",
                      edgecolor=DARK, lw=0.6, zorder=4)
    ax.add_patch(frame)
    # Boundary stubs end exactly at the cell frame.
    lc.set_clip_path(frame)
    ax.set_xlim(-0.02 * cell.box[0], 1.02 * cell.box[0])
    ax.set_ylim(-0.02 * cell.box[1], 1.02 * cell.box[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return lc


def render_bloch_gap_figure() -> None:
    ps.style()
    data = load_npz(VECTOR_DIR / "vector_periodic_s5_net0_train0_c50_w10.npz")
    cell = cell_from_npz(data)
    radii = np.asarray(data["radii"], dtype=float)
    kpath, xpos, xticks, labels = gamma_x_m_gamma()
    f0 = band_frequencies(cell, np.ones(cell.n_edges), kpath)
    ff = band_frequencies(cell, radii, kpath)
    bz = json.loads((VECTOR_DIR / "bz_convergence.json").read_text())
    size_scan = json.loads((VECTOR_DIR / "size_scan_summary.json").read_text())
    ensemble = json.loads((VECTOR_DIR / "vector_gap_ensemble.json").read_text())
    gap = bz["records"][0]["band_gap"]
    gap_lo = float(gap["lower_edge"])
    gap_hi = float(gap["upper_edge"])
    norm_gap = float(gap["normalized_gap"])
    target_lo = float(data["wlo"])
    target_hi = float(data["whi"])
    lower_band = int(gap["lower_band_1based"])
    upper_band = int(gap["upper_band_1based"])
    gap_band_indices = {lower_band - 1, upper_band - 1}

    # 2x2 at column width, mirroring Fig. 4's footprint: the material and
    # its robustness on top, the before/after band comparison side by side
    # (shared y) on the bottom.
    # The colorbar is vertical: a horizontal one under the aspect-locked
    # cell costs ~0.5 in of height in a half-column slot and starves the
    # cell; a vertical bar spends spare width instead.
    fig = plt.figure(figsize=(ps.COL_W, 3.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)

    # (a) The learned material itself, mirroring Fig. 1(a).
    panel_label(ax_a, "a")
    lc = draw_vector_cell(ax_a, cell, radii)
    cbar = fig.colorbar(lc, ax=ax_a, orientation="vertical",
                        fraction=0.08, pad=0.04, shrink=0.9, aspect=16)
    cbar.ax.set_title(r"$r_e$", pad=2)
    cbar.ax.tick_params(length=2)

    panel_label(ax_b, "b")
    sizes = np.array([r["size"] for r in size_scan], dtype=int)
    meds = np.array([r["median"] for r in size_scan], dtype=float)
    mins = np.array([r["min"] for r in size_scan], dtype=float)
    maxs = np.array([r["max"] for r in size_scan], dtype=float)
    rng = np.random.default_rng(0)
    for size in sizes:
        vals = np.asarray([
            r["band_gap"]["normalized_gap"] for r in ensemble["records"]
            if int(r["size"]) == int(size)
        ], dtype=float)
        jitter = rng.uniform(-0.10, 0.10, len(vals))
        ax_b.scatter(size + jitter, vals, s=7, color=BLUE, alpha=0.35,
                     linewidths=0, zorder=2)
    ax_b.errorbar(sizes, meds, yerr=[meds - mins, maxs - meds],
                  fmt="s", color=BLUE, ms=3.5, lw=0.9, capsize=2.5,
                  capthick=0.9, zorder=3)
    ax_b.set_xticks(sizes)
    ax_b.set_xticklabels([str(s) for s in sizes])
    ax_b.set_xlabel(r"cell size $N_{\mathrm{side}}$")
    ax_b.set_ylabel(r"$\Delta\omega/\omega_{\mathrm{mid}}$")

    # (c) Initial and (d) learned band structures, separated so the dense
    # initial spectrum is legible; they share the y-axis and the shaded
    # prescribed spectral window, making the before/after comparison direct.
    def band_axes(ax):
        ps.shade_window(ax, target_lo, target_hi, axis="y")
        for t in xticks:
            ax.axvline(t, color="#dddddd", lw=0.45, zorder=0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(xpos.min(), xpos.max())

    panel_label(ax_c, "c")
    band_axes(ax_c)
    n_show = min(31, f0.shape[1], ff.shape[1])
    band_ymax = 1.04 * max(np.max(f0[:, :n_show]), np.max(ff[:, :n_show]))
    for n in range(n_show):
        emphasized = n in gap_band_indices
        ax_c.plot(xpos, f0[:, n],
                  color=GRAY_DARK if emphasized else GRAY,
                  lw=0.75 if emphasized else 0.45,
                  ls="--",
                  alpha=0.90 if emphasized else 0.68,
                  rasterized=True)
    ax_c.set_ylim(0, band_ymax)
    ax_c.set_ylabel(r"frequency $\omega$")
    ax_c.text(0.05, 0.955, "initial", color=GRAY_DARK,
              fontsize=ps.BASE_SIZE, transform=ax_c.transAxes, va="top",
              zorder=10, bbox=ps.TAG_BOX)

    panel_label(ax_d, "d")
    ax_d.axhspan(gap_lo, gap_hi, facecolor=BLUE, edgecolor=BLUE,
                 hatch=r"\\\\", alpha=0.10, lw=0.45, zorder=-0.5)
    band_axes(ax_d)
    for n in range(n_show):
        lw = 0.78 if n in gap_band_indices else 0.45
        alpha = 0.95 if n in gap_band_indices else 0.68
        ax_d.plot(xpos, ff[:, n], color=BLUE, lw=lw, alpha=alpha,
                  rasterized=True)
    ax_d.tick_params(labelleft=False)
    ax_d.text(0.05, 0.08, "learned", color=BLUE,
              fontsize=ps.BASE_SIZE, transform=ax_d.transAxes, va="bottom",
              zorder=10, bbox=ps.TAG_BOX)
    # Keep the spectral window continuous: the single-line gap label sits in
    # the clear portion of the learned gap above that window.
    label_y = target_hi + 0.52 * (gap_hi - target_hi)
    ax_d.text(0.5 * xpos.max(), label_y,
              rf"$\Delta\omega/\omega_{{\mathrm{{mid}}}}={norm_gap:.3f}$",
              ha="center", va="center", fontsize=ps.BASE_SIZE, color=DARK,
              bbox=ps.TAG_BOX)
    savefig(fig, "fig2_bloch")


def load_e1_seed(seed: int, variant: str = "boundslocal") -> dict:
    return load_npz(E1_DIR / f"e1_{variant}_size20_seed{seed:03d}.npz")


def measured_relaxation_trace(d: dict, n_t: int = 300):
    """One real free relaxation of the initial (uniform-radius) network.

    x(t) = e^{-Kt} x0 computed by eigendecomposition; the reconstructed K is
    checked against the cached initial rate spectrum before use.  The plotted
    edge is chosen by a fixed criterion: strong strain at t_s and
    small-but-visible strain at t_l -- the signal the two-time rule reads.
    """
    edges = np.asarray(d["edges"])
    lengths = np.asarray(d["lengths"], dtype=float)
    n = len(np.asarray(d["pos"]))
    ts, tl = float(d["t_s"]), float(d["t_l"])
    K = build_K(edges, np.ones(len(edges)), lengths, n).toarray()
    lam, vecs = np.linalg.eigh(K)
    mu0 = np.sort(np.asarray(d["mu_initial"], dtype=float))
    if np.max(np.abs(np.sort(lam) - mu0)) > 1e-9:
        raise RuntimeError("reconstructed K does not match cached mu_initial")
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(n)
    x0 -= x0.mean()
    t = np.linspace(0.0, 1.15 * tl, n_t)
    coef = vecs.T @ x0
    x_t = vecs @ (coef[:, None] * np.exp(-lam[:, None] * t[None, :]))
    strains = x_t[edges[:, 0]] - x_t[edges[:, 1]]
    peak = np.abs(strains).max(axis=1)
    i_ts, i_tl = np.searchsorted(t, ts), np.searchsorted(t, tl)
    r_ts = np.abs(strains[:, i_ts]) / peak
    r_tl = np.abs(strains[:, i_tl]) / peak
    ok = ((r_ts > 0.55) & (r_tl > 0.04) & (r_tl < 0.18)
          & (np.abs(strains[:, 0]) / peak > 0.8))
    e = int(np.flatnonzero(ok)[0])
    trace = strains[e] / peak[e]
    if trace[i_ts] < 0:
        trace = -trace
    return t, trace


def draw_twotime_protocol_clear(ax, d: dict) -> None:
    panel_label(ax, "a")
    ts, tl = float(d["t_s"]), float(d["t_l"])
    t, trace = measured_relaxation_trace(d)
    ax.plot(t, trace, color=BLUE, lw=1.2)
    # Marker lines stop below the tag band so nothing is drawn under the box.
    ax.axvline(ts, color=RED, lw=0.8, ymax=0.60)
    ax.axvline(tl, color=RED, lw=0.8, ymax=0.60)
    ax.scatter([ts, tl], np.interp([ts, tl], t, trace), color=RED, s=12, zorder=3)
    ax.text(ts, 0.92 * trace.max(), r"$t_s$", ha="center", va="bottom", color=RED)
    ax.text(tl, 0.92 * trace.max(), r"$t_\ell$", ha="center", va="bottom", color=RED)
    ax.set_ylim(-0.06, 1.45 * trace.max())
    ax.set_yticks([0, 0.5, 1.0])
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.set_xlabel("time")
    ax.set_ylabel("normalized edge strain")
    tag = panel_tag(ax, "one free relaxation\ntwo clock times",
                    loc="upper left", fontsize=7)
    # Centered in the headroom band above the trace and the marker lines.
    tag.set_position((0.50, 0.985))
    tag.set_ha("center")


def select_median_e1_seed_checked() -> int:
    """Median seed by bracketing ratio, asserted against the SM summary."""
    seed = select_median_e1_seed()
    summary = json.loads((E1_DIR / "e1_boundslocal_summary.json").read_text())
    ratios = sorted(float(r["ratio_final"]) for r in summary["rows"])
    median_ratio = ratios[len(ratios) // 2]
    d = load_e1_seed(seed)
    if abs(float(d["ratio_final"]) - median_ratio) > 1e-9:
        raise RuntimeError(
            f"seed {seed} is not the median seed by bracketing ratio "
            f"({float(d['ratio_final'])} != {median_ratio})"
        )
    return seed


def render_two_time_figure() -> None:
    ps.style()
    seed = select_median_e1_seed_checked()
    d = load_e1_seed(seed)
    fig, axes = plt.subplots(2, 2, figsize=(ps.COL_W, 3.3),
                             constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    draw_twotime_protocol_clear(ax_a, d)

    panel_label(ax_b, "b")
    lam_star, ts, tl = float(d["lambda_star"]), float(d["t_s"]), float(d["t_l"])
    wlo, whi = float(d["window_lo"]), float(d["window_hi"])
    lam = np.geomspace(max(lam_star / 25, 1e-4), lam_star * 25, 500)
    phi = phi_values(lam, ts, tl)
    phi_star = float(phi_values(np.asarray([lam_star]), ts, tl)[0])
    ax_b.plot(lam, phi / phi_star, color=PURPLE, lw=1.3)
    ps.shade_window(ax_b, wlo, whi, axis="x")
    ax_b.axvline(lam_star, color=RED, lw=0.8, ls="--")
    ax_b.text(lam_star * 1.25, 0.06, r"$\lambda^\ast$", ha="left", va="bottom",
              color=RED, fontsize=7)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0, 1.34)
    ax_b.set_yticks([0, 0.5, 1.0])
    ax_b.set_xlabel(r"rate $\lambda$")
    ax_b.set_ylabel(r"$\varphi(\lambda)/\varphi(\lambda^\ast)$")
    ax_b.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax_b.xaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs="auto", numticks=10))
    ax_b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # Formula in the curve's own color acts as its direct label.
    panel_tag(ax_b,
              r"$\varphi(\lambda)=$" "\n"
              r"$e^{-2t_s\lambda}-e^{-2t_\ell\lambda}$",
              loc="upper left", color=PURPLE, fontsize=7)

    panel_label(ax_c, "c")
    mu0 = positive_rates(d["mu_initial"])
    muf = positive_rates(d["mu_final"])
    muc = positive_rates(d["mu_control"])
    bins = np.geomspace(max(min(mu0.min(), muf.min(), muc.min()) * 0.8, 1e-7),
                        max(mu0.max(), muf.max(), muc.max()) * 1.08, 70)
    ps.shade_window(ax_c, wlo, whi, axis="x")
    n_a, _, _ = ax_c.hist(mu0, bins=bins, color=GRAY, alpha=0.45)
    n_b, _, _ = ax_c.hist(muc, bins=bins, histtype="step", color=DARK, lw=0.8,
                          ls="--")
    n_c, _, _ = ax_c.hist(muf, bins=bins, color=BLUE, alpha=0.70)
    ax_c.axvline(lam_star, color=RED, lw=0.7, ls=":", ymax=0.70)
    ax_c.set_xscale("log")
    ax_c.set_ylim(0, 1.30 * max(n_a.max(), n_b.max(), n_c.max()))
    ax_c.set_xlabel(r"rate $\lambda_n$")
    ax_c.set_ylabel("count")
    ax_c.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax_c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    n_i, n_f = int(d["n_initial"]), int(d["n_final"])
    panel_tag(ax_c,
              rf"seed {seed}: ${n_i}\to{n_f}$" "\n"
              rf"$\lambda_+/\lambda_-={float(d['ratio_final']):.2f}$",
              loc="upper left", fontsize=7)
    # A compact key in empty high-frequency headroom avoids obscuring any
    # histogram bars, unlike the former lower-left legend.
    for y, text, color in [
        (0.93, "learned", BLUE),
        (0.82, "initial", GRAY_DARK),
        (0.71, "uniform", DARK),
    ]:
        ax_c.text(0.97, y, text, transform=ax_c.transAxes, ha="right",
                  va="top", fontsize=7, color=color, bbox=ps.TAG_BOX)

    panel_label(ax_d, "d")
    t = np.asarray(d["curve_times"], dtype=float)
    p0 = np.asarray(d["power_initial"], dtype=float)
    pf = np.asarray(d["power_final"], dtype=float)
    pc = np.asarray(d["power_control"], dtype=float)
    # At t=0 the projected random initial condition has one unit of variance
    # per positive mode, so the heat-trace normalization is the positive-mode
    # count rather than the first stored (strictly positive-time) sample.
    p0n = p0 / len(positive_rates(d["mu_initial"]))
    pcn = pc / len(positive_rates(d["mu_control"]))
    pfn = pf / len(positive_rates(d["mu_final"]))
    ax_d.loglog(t, p0n, color=GRAY, lw=1.0)
    ax_d.loglog(t, pcn, color=DARK, lw=0.9, ls="--")
    ax_d.loglog(t, pfn, color=BLUE, lw=1.3)
    # Offset labels from their anchor points and knock out the underlying ink,
    # so no response curve passes through its own name at print scale.
    i_l = int(0.62 * (len(t) - 1))
    i_i = int(0.75 * (len(t) - 1))
    i_u = int(0.68 * (len(t) - 1))
    ax_d.annotate("learned", (t[i_l], pfn[i_l]), xytext=(5, 5),
                  textcoords="offset points", color=BLUE, ha="left", va="bottom",
                  fontsize=7, bbox=ps.TAG_BOX)
    ax_d.annotate("initial", (t[i_i], p0n[i_i]), xytext=(5, -4),
                  textcoords="offset points", color=GRAY_DARK, ha="left", va="top",
                  fontsize=7, bbox=ps.TAG_BOX)
    ax_d.annotate("uniform", (t[i_u], pcn[i_u]), xytext=(-5, -5),
                  textcoords="offset points", color=DARK, ha="right", va="top",
                  fontsize=7, bbox=ps.TAG_BOX)
    ax_d.set_ylim(bottom=pcn[-1] / 4.0)
    # Marker lines span only the band where they cross the curves, staying
    # clear of the tag block below.
    ax_d.axvline(ts, color=RED, lw=0.7, alpha=0.75, ymin=0.60)
    ax_d.axvline(tl, color=RED, lw=0.7, alpha=0.75, ymin=0.60)
    ax_d.text(ts, ax_d.get_ylim()[1] / 1.7, r"$t_s$", color=RED,
              ha="center", va="top", fontsize=7)
    ax_d.text(tl * 1.12, ax_d.get_ylim()[1] / 1.7, r"$t_\ell$", color=RED,
              ha="left", va="top", fontsize=7)
    # Plateau width computed from the plotted seed's window edges, never typed in.
    width = np.log(float(d["gap_final_hi"]) / float(d["gap_final_lo"]))
    panel_tag(ax_d,
              rf"$\Delta\ln t=\ln(\lambda_+/\lambda_-)$" "\n"
              rf"$={width:.2g}$",
              loc="lower left", fontsize=7)
    ax_d.set_xlabel("time")
    ax_d.set_ylabel(r"$\mathbb{E}|x(t)|^2/\mathbb{E}|x(0)|^2$")
    ax_d.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax_d.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    savefig(fig, "fig3_twotime")


def render_scalar_controls_figure() -> None:
    ps.style()
    controls = load_npz(READINESS_DIR / "expanded_controls_real.npz")
    records = controls["records"]
    order = [
        ("adjoint", "paired\np=2"),
        ("forward_only", "p=1"),
        ("shuffled", "edge\nshuffled"),
        ("uniform_same_mean_stiffness", "uniform"),
        ("inflation_only", "inflation\nonly"),
    ]
    samples = [
        records[records["mode"] == mode]["n_in_final"].astype(float)
        for mode, _ in order
    ]
    if any(len(sample) != 8 for sample in samples):
        raise RuntimeError("scalar comparison figure requires eight records per arm")

    fig, ax = plt.subplots(
        figsize=(0.60 * ps.TEXT_W, 2.50), constrained_layout=True
    )
    x = np.arange(len(order))
    for i, sample in enumerate(samples):
        color = BLUE if i == 0 else GRAY_DARK
        mean = float(np.mean(sample))
        lo, hi = float(np.min(sample)), float(np.max(sample))
        offsets = np.linspace(-0.15, 0.15, len(sample))
        ax.vlines(i, lo, hi, color=color, lw=1.0, zorder=2)
        ax.hlines([lo, hi], i - 0.08, i + 0.08, color=color, lw=1.0,
                  zorder=2)
        ax.scatter(i + offsets, sample, s=12, color=color, alpha=0.55,
                   linewidths=0, zorder=3)
        ax.scatter(i, mean, s=28, marker="D", facecolor=color,
                   edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in order])
    ax.set_ylabel("final modes in window")
    ax.set_ylim(-0.8, max(float(np.max(sample)) for sample in samples) + 1.5)
    ax.set_yticks([0, 5, 10, 15, 20])
    savefig(fig, "fig_scalar_controls")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--figures",
        nargs="+",
        default=["scalar", "parity", "bloch", "two-time", "scalar-controls"],
        choices=["scalar", "parity", "bloch", "two-time", "scalar-controls"],
        help="named figure groups to render",
    )
    p.add_argument("--force", action="store_true",
                   help="recompute cached parity trajectories")
    args = p.parse_args()

    ensure_dirs()
    produced = []
    parity_summary = None
    if "scalar" in args.figures:
        render_scalar_gap_figure()
        produced.append("figures/fig1_flagship.pdf")
    if "parity" in args.figures:
        traj = render_parity_figure(force=args.force)
        produced.append("figures/fig2_parity.pdf")
        parity_summary = {
            "n_steps": int(traj["n_steps"]),
            "checkpoint_every": int(traj["checkpoint_every"]),
            "initial_window_count": int(traj["adjoint_counts"][0]),
            "adjoint_final_count": int(traj["adjoint_counts"][-1]),
            "forward_only_final_count": int(traj["forward_only_counts"][-1]),
            "runtime_seconds": float(traj["runtime_seconds"]),
            "forward_refill_from_below": True,
        }
    if "bloch" in args.figures:
        render_bloch_gap_figure()
        produced.append("figures/fig2_bloch.pdf")
    if "two-time" in args.figures:
        render_two_time_figure()
        produced.append("figures/fig3_twotime.pdf")
    if "scalar-controls" in args.figures:
        render_scalar_controls_figure()
        produced.append("figures/fig_scalar_controls.pdf")
    log = {
        "produced": produced,
    }
    if parity_summary is not None:
        log["parity_trajectory"] = parity_summary
    (OUT_DIR / "prl_v3_run_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
