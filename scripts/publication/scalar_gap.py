"""Generate scalar training data for the current PRL spectral-gap draft.

The script has two responsibilities:

1. Load or run one strictly-local reciprocal spatial exemplar and convert
   it into spectral learning diagnostics.
2. Run a reciprocal finite-difference check of the local paired-response gradient.

Numerical outputs are written to ``scripts/outputs/prl_bandgap/``. Production
figures are assembled by :mod:`publication.render_figures` after all experiment blocks
finish.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMPI_MCA_btl", "^openib")
os.environ.setdefault("OMPI_MCA_btl_openib_warn_no_device_params_found", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch
from mechanical.generate import generate_from_config
from mechanical.learners import build_K, eigenfreqs
from mechanical.learners_local import train_local
from mechanical.objectives import default_objective_terms
from mechanical.regularizers import regularizers_from_specs
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from publication.paths import FIGURE_DIR, REPO_ROOT, dataset

DATA_DIR = dataset("prl_bandgap")
FIG_DIR = FIGURE_DIR

REAL_SHIFT = "real_shift"
COMPLEX_DAMPING = "complex"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _gap_metrics(freqs, wlo, whi):
    freqs = np.asarray(freqs, dtype=np.float64)
    n_in = int(np.sum((freqs > wlo) & (freqs < whi)))
    positive = np.sort(freqs[freqs > 0])
    below = positive[positive <= wlo]
    above = positive[positive >= whi]
    if n_in == 0 and len(below) and len(above):
        gap_lo = float(below[-1])
        gap_hi = float(above[0])
        gap_ratio = (gap_hi - gap_lo) / ((gap_hi + gap_lo) / 2.0)
    else:
        gap_lo = 0.0
        gap_hi = 0.0
        gap_ratio = 0.0
    return n_in, gap_lo, gap_hi, gap_ratio


def _save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _load_npz_dict(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def run_exemplar(force=False):
    """Run or load the single local-rule reciprocal example."""
    out = DATA_DIR / "fig1_exemplar.npz"
    if out.exists() and not force:
        data = _load_npz_dict(out)
        damping = str(np.asarray(data.get("damping", "")).item())
        material_update = str(np.asarray(data.get("material_update", "")).item())
        response_parameters_match = (
            int(np.asarray(data.get("n_steps", -1)).item()) == 3000
            and np.isclose(float(np.asarray(data.get("response_metric_eta", np.nan)).item()), 0.02)
            and np.isclose(float(np.asarray(data.get("response_metric_lambda_ratio", np.nan)).item()), 0.025)
        )
        if (
            damping == REAL_SHIFT
            and material_update == "response_conditioned_log"
            and response_parameters_match
        ):
            return data

    config = {
        "network": {"topology": "rand-del", "size": 20, "seed": 0},
        "training": {
            "mode": "local",
            "M_node": 1.0,
            "R_INIT": 1.0,
            "R_MIN": 0.5,
            "R_MAX": 2.0,
            "n_steps": 3000,
            "batch": 10,
            "n_freq": 8,
            "frequency_sampling": "random_grid",
            "frequencies_per_step": 1,
            "alpha": 0.05,
            "grad_clip": None,
            "material_update": "response_conditioned_log",
            "response_metric_eta": 0.02,
            "response_metric_lambda_ratio": 0.025,
            "damping": REAL_SHIFT,
            "force_distribution": "gaussian",
            "regularizers": [],
            "eval_every": 25,
            "snapshot_every": 50,
            "snapshot_dtype": "float32",
            "train_seed": 0,
            "frequency_seed": 918273,
        },
    }
    network = generate_from_config(config)
    train_cfg = config["training"]
    regs = regularizers_from_specs(train_cfg["regularizers"])
    t0 = time.perf_counter()
    result = train_local(
        edges=network["edges"],
        lengths=network["lengths"],
        N=len(network["pos"]),
        M_node=train_cfg["M_node"],
        R_INIT=train_cfg["R_INIT"],
        R_MIN=train_cfg["R_MIN"],
        R_MAX=train_cfg["R_MAX"],
        n_steps=train_cfg["n_steps"],
        batch=train_cfg["batch"],
        n_freq=train_cfg["n_freq"],
        alpha=train_cfg["alpha"],
        grad_clip=train_cfg["grad_clip"],
        frequency_sampling=train_cfg["frequency_sampling"],
        frequencies_per_step=train_cfg["frequencies_per_step"],
        damping=train_cfg["damping"],
        force_distribution=train_cfg["force_distribution"],
        regularizers=regs,
        material_update=train_cfg["material_update"],
        response_metric_eta=train_cfg["response_metric_eta"],
        response_metric_lambda_ratio=train_cfg["response_metric_lambda_ratio"],
        train_seed=train_cfg["train_seed"],
        frequency_seed=train_cfg["frequency_seed"],
        eval_every=train_cfg["eval_every"],
        snapshot_every=train_cfg["snapshot_every"],
        snapshot_dtype=train_cfg["snapshot_dtype"],
    )
    elapsed = time.perf_counter() - t0

    spectra = np.vstack([
        eigenfreqs(
            network["edges"], radii.astype(np.float64), network["lengths"],
            len(network["pos"]), train_cfg["M_node"],
        )
        for radii in result["radii_snapshots"]
    ])

    payload = {
        "topology": network["topology"],
        "size": np.int64(network["size"]),
        "net_seed": np.int64(network["net_seed"]),
        "train_seed": np.int64(train_cfg["train_seed"]),
        "pos": network["pos"],
        "edges": network["edges"],
        "lengths": network["lengths"],
        "box": network["box"],
        "radii": result["radii"],
        "f0": result["f0"],
        "ff": result["ff"],
        "wlo": np.float64(result["wlo"]),
        "whi": np.float64(result["whi"]),
        "n_in_initial": np.int64(result["n_in_initial"]),
        "n_in_final": np.int64(result["n_in_final"]),
        "gap_ratio": np.float64(result["gap_ratio"]),
        "gap_lo": np.float64(result["gap_lo"]),
        "gap_hi": np.float64(result["gap_hi"]),
        "cost_history": result["cost_history"],
        "nin_samples": result["nin_samples"],
        "radius_snapshot_steps": result["radius_snapshot_steps"],
        "radii_snapshots": result["radii_snapshots"],
        "spectra_snapshots": spectra,
        "n_steps": np.int64(train_cfg["n_steps"]),
        "batch": np.int64(train_cfg["batch"]),
        "n_freq": np.int64(train_cfg["n_freq"]),
        "alpha": np.float64(train_cfg["alpha"]),
        "grad_clip": np.float64(np.nan),
        "gradient_clip_mode": "none",
        "material_update": train_cfg["material_update"],
        "response_metric_eta": np.float64(train_cfg["response_metric_eta"]),
        "response_metric_lambda_ratio": np.float64(
            train_cfg["response_metric_lambda_ratio"]
        ),
        "frequency_sampling": train_cfg["frequency_sampling"],
        "frequencies_per_step": np.int64(train_cfg["frequencies_per_step"]),
        "frequency_seed": np.int64(train_cfg["frequency_seed"]),
        "damping": train_cfg["damping"],
        "regularizer_arm": "response-conditioned-log",
        "t_train": np.float64(elapsed),
    }
    np.savez_compressed(out, **payload)
    _save_json(DATA_DIR / "fig1_exemplar_config.json", config)
    return _load_npz_dict(out)


def response_cost_grad_fixed(
    edges,
    radii,
    lengths,
    N,
    M_node,
    omegas,
    forces,
    eps_reg=1e-6,
    damping=REAL_SHIFT,
    mode="adjoint",
):
    """Squared-response cost and local edge gradient on a fixed force batch."""
    edges = np.asarray(edges, dtype=np.int64)
    radii = np.asarray(radii, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != N:
        raise ValueError(f"forces must have shape (batch, {N})")

    K = build_K(edges, radii, lengths, N)
    M_sp = diags(np.full(N, float(M_node)))
    I_sp = diags(np.ones(N))
    i_e, j_e = edges[:, 0], edges[:, 1]
    two_r_over_L = 2.0 * radii / lengths
    inv_bn = 1.0 / (len(forces) * len(omegas))

    cost = 0.0
    grad = np.zeros(len(edges), dtype=np.float64)
    for omega in omegas:
        if damping == REAL_SHIFT:
            H = (K - omega**2 * M_sp).tocsc()
            solv = splu(H)
            solv_adj = None
        elif damping == COMPLEX_DAMPING:
            H = (
                K - omega**2 * M_sp
                + (1j * eps_reg * omega**2) * I_sp
            ).tocsc()
            solv = splu(H)
            solv_adj = splu(H.conjugate().transpose().tocsc())
        else:
            raise ValueError(f"unknown damping mode {damping!r}")

        for F in forces:
            u = solv.solve(F)
            if damping == REAL_SHIFT:
                cost += float(np.sum(u * u)) * inv_bn
                if mode == "cost_only":
                    continue
                if mode == "adjoint":
                    v = solv.solve(u)
                elif mode == "forward_only":
                    v = u
                else:
                    raise ValueError(f"unknown gradient mode {mode!r}")
                du = u[i_e] - u[j_e]
                dv = v[i_e] - v[j_e]
                grad += (-2.0 * inv_bn) * two_r_over_L * dv * du
            else:
                cost += float(np.vdot(u, u).real) * inv_bn
                if mode == "cost_only":
                    continue
                if mode == "adjoint":
                    v = solv_adj.solve(u)
                elif mode == "forward_only":
                    v = u
                else:
                    raise ValueError(f"unknown gradient mode {mode!r}")
                du = u[i_e] - u[j_e]
                dv = v[i_e] - v[j_e]
                grad += (
                    (-2.0 * inv_bn)
                    * two_r_over_L
                    * np.real(np.conj(dv) * du)
                )
    return cost, grad


def run_gradient_check(force=False):
    out = DATA_DIR / "fig2_gradient_check.npz"
    if out.exists() and not force:
        data = _load_npz_dict(out)
        damping = str(np.asarray(data.get("damping", "")).item())
        if damping == REAL_SHIFT:
            return data

    rng = np.random.RandomState(11)
    network = generate_from_config({
        "network": {"topology": "ws", "size": 7, "seed": 2},
        "training": {},
    })
    edges = network["edges"]
    lengths = network["lengths"]
    N = len(network["pos"])
    radii = np.clip(1.0 + 0.08 * rng.randn(len(edges)), 0.65, 1.35)
    f0 = eigenfreqs(edges, radii, lengths, N, 1.0)
    terms, wlo, whi = default_objective_terms(f0, n_freq=5)
    omegas = np.array([t.omega for t in terms], dtype=np.float64)
    forces = rng.randn(6, N)

    _, grad = response_cost_grad_fixed(
        edges, radii, lengths, N, 1.0, omegas, forces,
        damping=REAL_SHIFT, mode="adjoint",
    )
    candidate = np.flatnonzero(np.abs(grad) > 1e-10)
    if len(candidate) < 35:
        candidate = np.arange(len(edges))
    check_edges = rng.choice(candidate, size=min(35, len(candidate)), replace=False)
    eps = 1e-5
    fd = np.zeros(len(check_edges), dtype=np.float64)
    for n, edge_idx in enumerate(check_edges):
        rp = radii.copy()
        rm = radii.copy()
        rp[edge_idx] += eps
        rm[edge_idx] -= eps
        cp, _ = response_cost_grad_fixed(
            edges, rp, lengths, N, 1.0, omegas, forces,
            damping=REAL_SHIFT, mode="cost_only",
        )
        cm, _ = response_cost_grad_fixed(
            edges, rm, lengths, N, 1.0, omegas, forces,
            damping=REAL_SHIFT, mode="cost_only",
        )
        fd[n] = (cp - cm) / (2.0 * eps)

    adj = grad[check_edges]
    denom = np.maximum(1.0, np.maximum(np.abs(fd), np.abs(adj)))
    rel_errors = np.abs(fd - adj) / denom
    np.savez_compressed(
        out,
        check_edges=check_edges,
        finite_difference=fd,
        adjoint=adj,
        relative_error=rel_errors,
        median_relative_error=np.float64(np.median(rel_errors)),
        max_relative_error=np.float64(np.max(rel_errors)),
        wlo=np.float64(wlo),
        whi=np.float64(whi),
        omegas=omegas,
        damping=REAL_SHIFT,
        topology=network["topology"],
        size=np.int64(network["size"]),
    )
    return _load_npz_dict(out)


def _panel_label(ax, label):
    ax.text(
        -0.12, 1.06, label, transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="top", ha="left",
    )


def plot_protocol_panel(ax):
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.text(0.06, 0.91, r"force $F$", fontsize=5.7, color="#2b6cb0")
    ax.text(0.58, 0.91, r"response $u$", fontsize=5.7, color="#2b6cb0")
    ax.add_patch(FancyArrowPatch(
        (0.34, 0.92), (0.54, 0.92),
        arrowstyle="->", mutation_scale=7.0, lw=0.85, color="#4b5563",
    ))
    ax.text(0.06, 0.79, r"$H(\omega)u=F$", fontsize=5.7)
    ax.text(0.06, 0.62, r"force $u$", fontsize=5.7, color="#2f855a")
    ax.text(0.58, 0.62, r"response $v$", fontsize=5.7, color="#2f855a")
    ax.add_patch(FancyArrowPatch(
        (0.34, 0.63), (0.54, 0.63),
        arrowstyle="->", mutation_scale=7.0, lw=0.85, color="#4b5563",
    ))
    ax.text(0.06, 0.50, r"$H(\omega)v=u$", fontsize=5.7)

    # Local edge readout: the edge only needs the two response differences
    # measured across its own endpoints.
    xi, xj, y = 0.18, 0.82, 0.28
    ax.plot([0.08, xi], [0.42, y], color="#c1c7cd", lw=0.8)
    ax.plot([0.08, xi], [0.15, y], color="#c1c7cd", lw=0.8)
    ax.plot([xj, 0.92], [y, 0.42], color="#c1c7cd", lw=0.8)
    ax.plot([xj, 0.92], [y, 0.15], color="#c1c7cd", lw=0.8)
    ax.plot([xi, xj], [y, y], color="#3f4b54", lw=2.1, solid_capstyle="round")
    ax.add_patch(Circle((xi, y), 0.035, fc="white", ec="#222222", lw=0.9, zorder=3))
    ax.add_patch(Circle((xj, y), 0.035, fc="white", ec="#222222", lw=0.9, zorder=3))
    ax.text(xi - 0.05, y - 0.10, r"$i$", fontsize=5.7, ha="center")
    ax.text(xj + 0.05, y - 0.10, r"$j$", fontsize=5.7, ha="center")
    ax.add_patch(FancyArrowPatch(
        (xi + 0.06, y + 0.10), (xj - 0.06, y + 0.10),
        arrowstyle="<->", mutation_scale=7.0, lw=0.9, color="#2b6cb0",
    ))
    ax.add_patch(FancyArrowPatch(
        (xi + 0.06, y - 0.10), (xj - 0.06, y - 0.10),
        arrowstyle="<->", mutation_scale=7.0, lw=0.9, color="#2f855a",
    ))
    ax.text(0.50, y + 0.135, r"$u_i-u_j$", fontsize=5.4, color="#2b6cb0", ha="center")
    ax.text(0.50, y - 0.18, r"$v_i-v_j$", fontsize=5.4, color="#2f855a", ha="center")
    ax.text(0.50, 0.02, r"$\Delta r_e\propto (u_i-u_j)(v_i-v_j)$",
            fontsize=5.15, color="#444444", ha="center", va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def plot_network(ax, pos, edges, radii, box):
    pos = np.asarray(pos)
    edges = np.asarray(edges, dtype=np.int64)
    radii = np.asarray(radii, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    segs = []
    vals = []
    for idx, (i, j) in enumerate(edges):
        p = pos[i]
        q = pos[j]
        if len(box) >= 2 and np.linalg.norm(p - q) > 0.45 * float(np.min(box)):
            continue
        segs.append([p, q])
        vals.append(radii[idx])
    vals = np.asarray(vals, dtype=np.float64)
    lw = 0.15 + 1.75 * (vals - 0.5) / 1.5
    lc = LineCollection(
        segs, cmap="viridis", array=vals, linewidths=lw,
        capstyle="round", alpha=0.95,
    )
    ax.add_collection(lc)
    ax.scatter(pos[:, 0], pos[:, 1], s=2.0, color="#111111", alpha=0.35, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, float(box[0]) + 0.2)
    ax.set_ylim(-0.2, float(box[1]) + 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("learned radii")
    return lc


def plot_fig1(data):
    _style()
    # Single-column 2x2 layout: learned geometry, learning protocol, spectrum,
    # and training trajectory.
    fig = plt.figure(figsize=(3.4, 3.55))
    gs = fig.add_gridspec(
        2, 2,
        left=0.045,
        right=1.0,
        bottom=0.12,
        top=0.985,
        hspace=0.48,
        wspace=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    lc = plot_network(ax_a, data["pos"], data["edges"], data["radii"], data["box"])
    ax_a.set_title("learned radius pattern")
    _panel_label(ax_a, "a")
    cax = ax_a.inset_axes([0.18, -0.16, 0.64, 0.045])
    cbar = fig.colorbar(
        lc, cax=cax, orientation="horizontal",
    )
    cbar.set_ticks([0.5, 1.0, 1.5, 2.0])
    cbar.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar.set_label(r"$r_e$", fontsize=6, labelpad=0)

    plot_protocol_panel(ax_b)
    _panel_label(ax_b, "b")

    wlo = float(data["wlo"])
    whi = float(data["whi"])

    f0 = np.sort(np.asarray(data["f0"], dtype=np.float64))
    ff = np.sort(np.asarray(data["ff"], dtype=np.float64))
    f0 = f0[f0 > 1e-9]
    ff = ff[ff > 1e-9]
    xmax = np.percentile(np.concatenate([f0, ff]), 98.5) * 1.05
    bins = np.linspace(0.0, xmax, 64)
    ax_c.axvspan(wlo, whi, color="#e76f51", alpha=0.17, lw=0)
    ax_c.hist(
        f0, bins=bins, histtype="stepfilled",
        color="#9ca3af", edgecolor="#4b5563", linewidth=0.7,
        alpha=0.55, label="initial",
    )
    ax_c.hist(
        ff, bins=bins, histtype="step",
        color="#2b6cb0", linewidth=1.4, label="learned",
    )
    ax_c.set_xlim(0, xmax)
    ax_c.set_xlabel(r"frequency $\omega$")
    ax_c.set_ylabel("modes", labelpad=1)
    ax_c.set_title(
        rf"global spectral exclusion: {int(data['n_in_initial'])}$\to${int(data['n_in_final'])} modes"
    )
    ax_c.legend(
        frameon=False, loc="upper right", handlelength=1.6,
        handletextpad=0.5, borderaxespad=0.3,
    )
    _panel_label(ax_c, "c")

    samples = np.asarray(data["nin_samples"], dtype=np.float64)
    steps = np.concatenate(([0.0], samples[:, 0]))
    nin = np.concatenate(([float(data["n_in_initial"])], samples[:, 1]))
    ax_d.plot(steps, nin, color="#2b6cb0", lw=1.3, marker="o", ms=2.2)
    ax_d.set_ylabel(r"$N_{\rm in}$")
    ax_d.set_ylim(-5, max(float(data["n_in_initial"]) * 1.08, 10.0))
    zero = steps[nin <= 0]
    ax_d.set_xlim(0, max(120.0, float(zero[0]) * 1.6) if len(zero) else float(data["n_steps"]))
    ax_d.set_xlabel("training step")
    ax_d.set_title("training")
    _panel_label(ax_d, "d")
    # panels are described in the caption; no per-panel titles (PRL style)
    for ax in fig.axes:
        ax.set_title("")
    fig.savefig(FIG_DIR / "fig1_learning_dynamics.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIG_DIR / "fig1_learning_dynamics.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def write_run_summary(exemplar, grad_data):
    payload = {
        "fig1": {
            "topology": str(exemplar["topology"]),
            "size": int(exemplar["size"]),
            "net_seed": int(exemplar["net_seed"]),
            "train_seed": int(exemplar["train_seed"]),
            "n_steps": int(exemplar["n_steps"]),
            "n_in_initial": int(exemplar["n_in_initial"]),
            "n_in_final": int(exemplar["n_in_final"]),
            "gap_ratio": float(exemplar["gap_ratio"]),
            "target_window": [float(exemplar["wlo"]), float(exemplar["whi"])],
            "t_train_seconds": float(exemplar["t_train"]),
            "damping": str(np.asarray(exemplar["damping"]).item()),
        },
        "gradient_check": {
            "gradient_median_relative_error": float(grad_data["median_relative_error"]),
            "gradient_max_relative_error": float(grad_data["max_relative_error"]),
            "damping": str(np.asarray(grad_data["damping"]).item()),
        },
    }
    _save_json(DATA_DIR / "prl_figure_summary.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate cached PRL data even if the NPZ files already exist.",
    )
    ap.add_argument(
        "--skip-experiments", action="store_true",
        help="Only replot from existing cached PRL data.",
    )
    args = ap.parse_args()

    _ensure_dirs()
    _style()

    force = bool(args.force)
    if args.skip_experiments:
        force = False

    print("Running/loading Fig. 1 exemplar...", flush=True)
    exemplar = run_exemplar(force=force and not args.skip_experiments)
    print(
        "  Fig. 1: "
        f"n_in {int(exemplar['n_in_initial'])}->{int(exemplar['n_in_final'])}, "
        f"gap_ratio={float(exemplar['gap_ratio']):.3f}",
        flush=True,
    )

    print("Running/loading Fig. 2 gradient check...", flush=True)
    grad_data = run_gradient_check(force=force and not args.skip_experiments)
    print(
        "  gradient median relative error "
        f"{float(grad_data['median_relative_error']):.2e}",
        flush=True,
    )

    payload = write_run_summary(exemplar, grad_data)
    print("Wrote:")
    for path in [
        DATA_DIR / "prl_figure_summary.json",
    ]:
        print(f"  {path.relative_to(REPO_ROOT)}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
