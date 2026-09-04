"""Matched diagnostic for local per-edge gradient saturation.

This script compares the Figure 1 scalar protocol with

    r_e <- clip(r_e - alpha * g_e)

against the otherwise identical locally saturated update

    r_e <- clip(r_e - alpha * sat_c(g_e)),
    sat_c(g_e) = clip(g_e, -c, c).

Both arms use the same network, force seed, frequency-sampling seed, target
window, regularizer, and number of steps.  Outputs are diagnostic artifacts in
``scripts/outputs/prl_bandgap/``; they do not replace the manuscript figure.
"""
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
from matplotlib.patches import Rectangle
from mechanical.generate import generate_from_config
from mechanical.learners_local import saturate_components, train_local
from mechanical.regularizers import inflation
from publication.paths import dataset

OUT_DIR = dataset("prl_bandgap")
R_MIN = 0.5
R_MAX = 2.0
BLUE = "#3274a1"
ORANGE = "#d06b32"
DARK = "#252525"
GRAY = "#8b8b8b"
WINDOW = "#e8c65b"


def _tag(c: float) -> str:
    return f"c{c:g}".replace(".", "p")


def _run_arm(network: dict, saturation_c: float | None, steps: int) -> dict:
    return train_local(
        edges=network["edges"],
        lengths=network["lengths"],
        N=len(network["pos"]),
        M_node=1.0,
        R_INIT=1.0,
        R_MIN=R_MIN,
        R_MAX=R_MAX,
        n_steps=steps,
        batch=10,
        n_freq=8,
        alpha=0.05,
        grad_clip=saturation_c,
        damping="real_shift",
        force_distribution="gaussian",
        regularizers=[inflation(strength=0.03)],
        train_seed=0,
        frequency_sampling="random_grid",
        frequencies_per_step=1,
        frequency_seed=918273,
        eval_every=25,
        snapshot_every=50,
        snapshot_dtype="float32",
    )


def _prefixed(payload: dict, prefix: str, result: dict) -> None:
    for key in (
        "radii", "f0", "ff", "wlo", "whi", "n_in_initial",
        "n_in_final", "gap_lo", "gap_hi", "gap_ratio", "cost_history",
        "nin_samples", "radius_snapshot_steps", "radii_snapshots",
    ):
        payload[f"{prefix}_{key}"] = result[key]


def run_comparison(c: float, steps: int, force: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"gradient_saturation_comparison_{_tag(c)}_{steps}steps.npz"
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}

    config = {"network": {"topology": "rand-del", "size": 20, "seed": 0}}
    network = generate_from_config(config)
    unsaturated = _run_arm(network, saturation_c=None, steps=steps)
    saturated = _run_arm(network, saturation_c=c, steps=steps)

    if not np.allclose(unsaturated["f0"], saturated["f0"]):
        raise RuntimeError("comparison arms did not start from the same spectrum")
    if not np.allclose(
        [unsaturated["wlo"], unsaturated["whi"]],
        [saturated["wlo"], saturated["whi"]],
    ):
        raise RuntimeError("comparison arms did not use the same prescribed spectral window")

    payload = {
        "pos": network["pos"],
        "edges": network["edges"],
        "lengths": network["lengths"],
        "box": network["box"],
        "saturation_c": np.float64(c),
        "steps": np.int64(steps),
        "alpha": np.float64(0.05),
        "train_seed": np.int64(0),
        "frequency_seed": np.int64(918273),
    }
    _prefixed(payload, "unsaturated", unsaturated)
    _prefixed(payload, "saturated", saturated)
    np.savez_compressed(cache, **payload)
    return payload


def _side_counts(freqs: np.ndarray, wlo: float, whi: float) -> tuple[int, int, int]:
    positive = _positive_modes(freqs)
    return (
        int(np.sum(positive <= wlo)),
        int(np.sum((positive > wlo) & (positive < whi))),
        int(np.sum(positive >= whi)),
    )


def _positive_modes(freqs: np.ndarray) -> np.ndarray:
    """Remove the translational zero mode with a scale-aware tolerance."""
    freqs = np.asarray(freqs, dtype=float)
    tolerance = 1e-8 * max(float(np.max(freqs)), 1.0)
    return np.sort(freqs[freqs > tolerance])


def _trajectory(data: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(data[f"{prefix}_nin_samples"], dtype=float)
    steps = np.r_[0.0, samples[:, 0], float(data["steps"])]
    counts = np.r_[
        float(data[f"{prefix}_n_in_initial"]),
        samples[:, 1],
        float(data[f"{prefix}_n_in_final"]),
    ]
    return steps, counts


def _periodic_segments(data: dict, radii: np.ndarray):
    pos = np.asarray(data["pos"], dtype=float)
    edges = np.asarray(data["edges"], dtype=int)
    box = np.asarray(data["box"], dtype=float)
    raw = pos[edges[:, 1]] - pos[edges[:, 0]]
    shift = np.round(raw / box) * box
    disp = raw - shift
    wraps = np.any(shift != 0.0, axis=1)
    segments, values, widths = [], [], []
    for edge_i, (i, j) in enumerate(edges):
        edge_segments = (
            [[pos[i], pos[i] + disp[edge_i]], [pos[j], pos[j] - disp[edge_i]]]
            if wraps[edge_i]
            else [[pos[i], pos[j]]]
        )
        for segment in edge_segments:
            segments.append(segment)
            values.append(radii[edge_i])
            widths.append(0.35 + 1.0 * (radii[edge_i] - R_MIN) / (R_MAX - R_MIN))
    return segments, values, widths


def _draw_network(ax, data: dict, radii: np.ndarray, title: str):
    segments, values, widths = _periodic_segments(data, radii)
    pos = np.asarray(data["pos"], dtype=float)
    box = np.asarray(data["box"], dtype=float)
    collection = LineCollection(
        segments, array=np.asarray(values), cmap="viridis", norm=plt.Normalize(R_MIN, R_MAX),
        linewidths=widths, alpha=0.95,
    )
    ax.add_collection(collection)
    ax.scatter(pos[:, 0], pos[:, 1], s=0.8, color=DARK, zorder=3)
    frame = Rectangle((0, 0), box[0], box[1], facecolor="none", edgecolor=DARK, lw=0.6)
    ax.add_patch(frame)
    collection.set_clip_path(frame)
    ax.set(xlim=(-0.02 * box[0], 1.02 * box[0]), ylim=(-0.02 * box[1], 1.02 * box[1]))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return collection


def make_figure(data: dict, c: float, output_stem: Path) -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 240,
    })
    fig = plt.figure(figsize=(9.0, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.08])
    ax_sat = fig.add_subplot(grid[0, 0])
    ax_spectrum = fig.add_subplot(grid[0, 1:])
    ax_net_u = fig.add_subplot(grid[1, 0])
    ax_net_s = fig.add_subplot(grid[1, 1])
    right = grid[1, 2].subgridspec(2, 1, hspace=0.42)
    ax_count = fig.add_subplot(right[0])
    ax_radii = fig.add_subplot(right[1])

    # (a) The scalar saturation map itself.
    g = np.linspace(-2.2 * c, 2.2 * c, 600)
    ax_sat.plot(g, g, color=GRAY, lw=1.0, ls="--", label="no saturation")
    ax_sat.plot(g, saturate_components(g, c), color=BLUE, lw=1.8,
                label=rf"$\operatorname{{sat}}_{{{c:g}}}$")
    ax_sat.axvline(-c, color=DARK, lw=0.6, alpha=0.4)
    ax_sat.axvline(c, color=DARK, lw=0.6, alpha=0.4)
    ax_sat.set_xlabel(r"raw local gradient $g_e$")
    ax_sat.set_ylabel(r"applied gradient")
    ax_sat.set_title(rf"(a) Local cap: $|\Delta r_e|\leq\alpha c={0.05*c:g}$")
    ax_sat.legend(frameon=False, loc="upper left")

    # (b) Sorted spectra: low-frequency detail plus complete side counts.
    wlo = float(data["unsaturated_wlo"])
    whi = float(data["unsaturated_whi"])
    f0 = _positive_modes(data["unsaturated_f0"])
    fu = _positive_modes(data["unsaturated_ff"])
    fs = _positive_modes(data["saturated_ff"])
    ax_spectrum.axhspan(wlo, whi, color=WINDOW, alpha=0.28, label="spectral window")
    ax_spectrum.plot(np.arange(len(f0)), f0, color=GRAY, lw=1.0, label="initial")
    ax_spectrum.plot(np.arange(len(fu)), fu, color=ORANGE, lw=1.15, label="no saturation")
    ax_spectrum.plot(np.arange(len(fs)), fs, color=BLUE, lw=1.15,
                     label=rf"local saturation, $c={c:g}$")
    ax_spectrum.set_ylim(0, max(4.3, 1.35 * whi))
    ax_spectrum.set_xlabel("sorted positive-mode index")
    ax_spectrum.set_ylabel(r"frequency $\omega_n$")
    ax_spectrum.set_title("(b) Matched final spectra (curves above the axis limit are still counted)")
    ax_spectrum.legend(frameon=False, ncol=4, loc="upper left")
    u_counts = _side_counts(fu, wlo, whi)
    s_counts = _side_counts(fs, wlo, whi)
    ax_spectrum.text(
        0.99, 0.04,
        "below / inside / above\n"
        f"no saturation: {u_counts[0]} / {u_counts[1]} / {u_counts[2]}\n"
        f"local saturation: {s_counts[0]} / {s_counts[1]} / {s_counts[2]}",
        ha="right", va="bottom", transform=ax_spectrum.transAxes,
        bbox={"facecolor": "white", "edgecolor": "0.85", "pad": 3.0},
    )

    # (c,d) Same network and color scale.
    ru = np.asarray(data["unsaturated_radii"], dtype=float)
    rs = np.asarray(data["saturated_radii"], dtype=float)
    _draw_network(ax_net_u, data, ru, "(c) No saturation")
    collection = _draw_network(ax_net_s, data, rs, rf"(d) Local saturation, $c={c:g}$")
    colorbar = fig.colorbar(collection, ax=[ax_net_u, ax_net_s], orientation="horizontal",
                            fraction=0.055, pad=0.025, aspect=28)
    colorbar.set_label(r"edge radius $r_e$", labelpad=2)

    # (e) Gap-clearing trajectories.
    xu, yu = _trajectory(data, "unsaturated")
    xs, ys = _trajectory(data, "saturated")
    ax_count.plot(xu, yu, color=ORANGE, lw=1.25, label="no saturation")
    ax_count.plot(xs, ys, color=BLUE, lw=1.25, label=rf"$c={c:g}$")
    ax_count.set_ylabel("modes in target")
    ax_count.set_xlabel("training step")
    ax_count.set_title("(e) Window clearing")
    ax_count.legend(frameon=False)
    ax_count.set_ylim(bottom=-4)

    # (f) Final material distribution.
    bins = np.linspace(R_MIN, R_MAX, 31)
    ax_radii.hist(ru, bins=bins, histtype="step", color=ORANGE, lw=1.4,
                  label=f"no sat.; median {np.median(ru):.3f}")
    ax_radii.hist(rs, bins=bins, histtype="step", color=BLUE, lw=1.4,
                  label=rf"$c={c:g}$; median {np.median(rs):.3f}")
    ax_radii.set_xlabel(r"final edge radius $r_e$")
    ax_radii.set_ylabel("edge count")
    ax_radii.set_title("(f) Radius distribution")
    ax_radii.legend(frameon=False)

    fig.suptitle(
        "Componentwise saturation caps resonant local gradients without a global readout",
        fontsize=11,
    )
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _summary(data: dict, c: float) -> dict:
    wlo = float(data["unsaturated_wlo"])
    whi = float(data["unsaturated_whi"])
    summary = {
        "saturation": {
            "c": c,
            "alpha": float(data["alpha"]),
            "maximum_per_step_radius_change": float(data["alpha"]) * c,
            "operation": "componentwise clip(g_e, -c, c)",
            "network_wide_readout": False,
        },
        "target_window": [wlo, whi],
    }
    for prefix in ("unsaturated", "saturated"):
        radii = np.asarray(data[f"{prefix}_radii"], dtype=float)
        summary[prefix] = {
            "below_inside_above": list(_side_counts(data[f"{prefix}_ff"], wlo, whi)),
            "gap_ratio": float(data[f"{prefix}_gap_ratio"]),
            "radius_mean": float(np.mean(radii)),
            "radius_median": float(np.median(radii)),
            "fraction_at_lower_bound": float(np.mean(np.isclose(radii, R_MIN))),
            "fraction_at_upper_bound": float(np.mean(np.isclose(radii, R_MAX))),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=float, default=3.0,
                        help="componentwise saturation threshold (default: 3)")
    parser.add_argument("--steps", type=int, default=3000,
                        help="training steps in each matched arm (default: 3000)")
    parser.add_argument("--force", action="store_true", help="overwrite cached comparison data")
    args = parser.parse_args()
    if not np.isfinite(args.c) or args.c <= 0:
        parser.error("--c must be finite and positive")
    if args.steps < 1:
        parser.error("--steps must be positive")

    data = run_comparison(args.c, args.steps, force=args.force)
    stem = OUT_DIR / f"gradient_saturation_comparison_{_tag(args.c)}_{args.steps}steps"
    make_figure(data, args.c, stem)
    summary = _summary(data, args.c)
    summary_path = stem.with_name(stem.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"figure: {stem.with_suffix('.png')}")
    print(f"data:   {stem.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
