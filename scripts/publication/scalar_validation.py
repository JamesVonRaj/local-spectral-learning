"""Run scalar gap ensembles, controls, and robustness checks.

The scalar exemplar lives in :mod:`publication.scalar_gap`; this module carries
the ensemble statistics, ablations, spectral-window tests, finite-size checks,
and centralized-response reference used by the supplement.
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
from mechanical.generate import generate_from_config
from mechanical.io import local_regularizer_arm_to_specs
from mechanical.learners import eigenfreqs
from mechanical.learners_local import train_local
from mechanical.objectives import default_objective_terms
from mechanical.regularizers import inflation, regularizers_from_specs

from publication.paths import FIGURE_DIR, dataset
from publication.scalar_gap import _gap_metrics, response_cost_grad_fixed

DATA_DIR = dataset("prl_readiness")
FIG_DIR = FIGURE_DIR
CURRENT_DATA_DIR = dataset("prl_bandgap")

REAL_SHIFT = "real_shift"
COMPLEX_DAMPING = "complex"
EPS_REG = 0.0  # claimed reciprocal protocol: bare operator
COMPLEX_EPS_REG = 1e-6  # damping amplitude of the complex-damped control arms
CONTROL_ORDER = (
    "adjoint",
    "shuffled",
    "forward_only",
    "random_matched",
    "inflation_only",
    "uniform_stiffness",
)
CONTROL_LABELS = {
    "adjoint": "paired",
    "shuffled": "shuffled",
    "forward_only": "forward",
    "random_matched": "random",
    "inflation_only": "inflation",
    "uniform_stiffness": "uniform",
}


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


def _load_npz_dict(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _structured_has_fields(array, fields):
    names = array.dtype.names or ()
    return all(field in names for field in fields)


def edge_material_diagnostics(radii, lengths, r_min=0.5, r_max=2.0, tol=1e-6):
    radii = np.asarray(radii, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    k0_mean = float(np.mean(1.0 / lengths))
    kf_mean = float(np.mean(radii**2 / lengths))
    material0 = float(np.sum(lengths))
    materialf = float(np.sum(radii**2 * lengths))
    frac_min = float(np.mean(radii <= float(r_min) + float(tol)))
    frac_max = float(np.mean(radii >= float(r_max) - float(tol)))
    return {
        "mean_radius": float(np.mean(radii)),
        "mean_stiffness_ratio": kf_mean / k0_mean,
        "material_ratio": materialf / material0,
        "frac_rmin": frac_min,
        "frac_rmax": frac_max,
        "frac_at_bounds": frac_min + frac_max,
    }


def _pack_run(network, result, t_train):
    return {
        "topology": str(network["topology"]),
        "size": int(network["size"]),
        "net_seed": int(network["net_seed"]),
        "pos": np.asarray(network["pos"], dtype=np.float64),
        "edges": np.asarray(network["edges"], dtype=np.int64),
        "lengths": np.asarray(network["lengths"], dtype=np.float64),
        "box": np.asarray(network["box"], dtype=np.float64),
        "radii": np.asarray(result["radii"], dtype=np.float64),
        "f0": np.asarray(result["f0"], dtype=np.float64),
        "ff": np.asarray(result["ff"], dtype=np.float64),
        "wlo": float(result["wlo"]),
        "whi": float(result["whi"]),
        "n_in_initial": int(result["n_in_initial"]),
        "n_in_final": int(result["n_in_final"]),
        "gap_lo": float(result["gap_lo"]),
        "gap_hi": float(result["gap_hi"]),
        "gap_ratio": float(result["gap_ratio"]),
        "t_train": float(t_train),
    }


def _mean_success_gap(rows):
    """Mean gap ratio over runs that actually clear the prescribed spectral window."""
    successful = rows[rows["n_in_final"] == 0]
    if len(successful) == 0:
        return None
    return float(np.mean(successful["gap_ratio"]))


def local_spacing(freqs, wlo, whi):
    f = np.sort(np.asarray(freqs, dtype=np.float64))
    f = f[f > 1e-9]
    width = float(whi - wlo)
    near = f[(f >= wlo - width) & (f <= whi + width)]
    diffs = np.diff(near)
    diffs = diffs[diffs > 1e-12]
    if len(diffs):
        return float(np.median(diffs))
    n_in = int(np.sum((f > wlo) & (f < whi)))
    return width / max(n_in + 1, 1)


def spectral_nontrivial_metrics(run):
    f0 = np.asarray(run["f0"], dtype=np.float64)
    ff = np.asarray(run["ff"], dtype=np.float64)
    radii = np.asarray(run["radii"], dtype=np.float64)
    lengths = np.asarray(run["lengths"], dtype=np.float64)
    wlo = float(run["wlo"])
    whi = float(run["whi"])

    spacing = local_spacing(f0, wlo, whi)
    gap_width = max(0.0, float(run["gap_hi"]) - float(run["gap_lo"]))
    target_width = whi - wlo
    f0_pos = f0[f0 > 1e-9]
    ff_pos = ff[ff > 1e-9]
    mean_f0 = float(np.mean(f0_pos))
    mean_ff = float(np.mean(ff_pos))
    k0_mean = float(np.mean(1.0 / lengths))
    kf_mean = float(np.mean(radii**2 / lengths))
    material0 = float(np.sum(lengths))
    materialf = float(np.sum(radii**2 * lengths))

    c_stiff = np.sqrt(kf_mean / k0_mean)
    c_mat = np.sqrt(materialf / material0)
    c_freq = mean_ff / mean_f0
    uniform = {
        "same_mean_stiffness": f0 * c_stiff,
        "same_material": f0 * c_mat,
        "same_mean_frequency": f0 * c_freq,
    }
    uniform_n = {
        key: int(np.sum((vals > wlo) & (vals < whi)))
        for key, vals in uniform.items()
    }

    ff_norm_mean = ff / mean_ff
    wlo_norm_mean = wlo / mean_f0
    whi_norm_mean = whi / mean_f0
    n_final_norm_mean = int(np.sum(
        (ff_norm_mean > wlo_norm_mean) & (ff_norm_mean < whi_norm_mean)
    ))
    f0_norm_k = f0 / np.sqrt(k0_mean)
    ff_norm_k = ff / np.sqrt(kf_mean)
    wlo_norm_k = wlo / np.sqrt(k0_mean)
    whi_norm_k = whi / np.sqrt(k0_mean)
    n_final_norm_k = int(np.sum(
        (ff_norm_k > wlo_norm_k) & (ff_norm_k < whi_norm_k)
    ))
    gap_lo_norm_k = float(run["gap_lo"]) / np.sqrt(kf_mean)
    gap_hi_norm_k = float(run["gap_hi"]) / np.sqrt(kf_mean)
    n_initial_in_final_norm_gap = int(np.sum(
        (f0_norm_k > gap_lo_norm_k) & (f0_norm_k < gap_hi_norm_k)
    ))
    n_final_in_final_norm_gap = int(np.sum(
        (ff_norm_k > gap_lo_norm_k) & (ff_norm_k < gap_hi_norm_k)
    ))
    norm_gap_spacing = local_spacing(
        f0_norm_k, gap_lo_norm_k, gap_hi_norm_k,
    )
    material = edge_material_diagnostics(radii, lengths)

    return {
        "size": int(run["size"]),
        "N": int(len(f0)),
        "n_in_initial": int(run["n_in_initial"]),
        "n_in_final": int(run["n_in_final"]),
        "gap_width": gap_width,
        "target_width": target_width,
        "gap_ratio": float(run["gap_ratio"]),
        "local_spacing": spacing,
        "gap_over_spacing": gap_width / spacing if spacing > 0 else np.nan,
        "target_over_spacing": target_width / spacing if spacing > 0 else np.nan,
        "mean_frequency_scale": c_freq,
        "mean_stiffness_scale": c_stiff,
        "material_scale": c_mat,
        "uniform_n_same_mean_stiffness": uniform_n["same_mean_stiffness"],
        "uniform_n_same_material": uniform_n["same_material"],
        "uniform_n_same_mean_frequency": uniform_n["same_mean_frequency"],
        "n_final_norm_mean": n_final_norm_mean,
        "n_final_norm_stiffness": n_final_norm_k,
        "final_gap_lo_norm_stiffness": gap_lo_norm_k,
        "final_gap_hi_norm_stiffness": gap_hi_norm_k,
        "n_initial_in_final_norm_gap": n_initial_in_final_norm_gap,
        "n_final_in_final_norm_gap": n_final_in_final_norm_gap,
        "final_norm_gap_over_spacing": (
            (gap_hi_norm_k - gap_lo_norm_k) / norm_gap_spacing
            if norm_gap_spacing > 0 else np.nan
        ),
        **material,
    }


def _bins_with_markers(lo, hi, n_bins, markers):
    bins = np.linspace(float(lo), float(hi), int(n_bins))
    markers = [float(x) for x in markers if float(lo) < float(x) < float(hi)]
    return np.unique(np.sort(np.concatenate([bins, markers])))


def _scalar_ensemble_runs(force=False):
    """Generate the 20-run scalar ensemble specified in the supplement."""
    runs = []
    exemplar_path = CURRENT_DATA_DIR / "fig1_exemplar.npz"
    for net_seed in range(20):
        if net_seed == 0 and exemplar_path.exists() and not force:
            data = _load_npz_dict(exemplar_path)
            run = {
                key: data[key]
                for key in (
                    "topology", "size", "net_seed", "pos", "edges", "lengths",
                    "box", "radii", "f0", "ff", "wlo", "whi", "n_in_initial",
                    "n_in_final", "gap_lo", "gap_hi", "gap_ratio", "t_train",
                )
            }
        else:
            network = generate_from_config({
                "network": {"topology": "rand-del", "size": 20, "seed": net_seed},
                "training": {},
            })
            t0 = time.perf_counter()
            result = train_local(
                edges=network["edges"],
                lengths=network["lengths"],
                N=len(network["pos"]),
                n_steps=3000,
                batch=10,
                n_freq=8,
                alpha=0.05,
                grad_clip=None,
                material_update="response_conditioned_log",
                response_metric_eta=0.02,
                response_metric_lambda_ratio=0.025,
                frequency_sampling="random_grid",
                frequencies_per_step=1,
                damping=REAL_SHIFT,
                force_distribution="gaussian",
                regularizers=[],
                train_seed=0,
                eval_every=500,
            )
            run = _pack_run(network, result, time.perf_counter() - t0)
        print(
            f"  scalar ensemble net={net_seed:02d} "
            f"n_in={int(run['n_in_initial'])}->{int(run['n_in_final'])}",
            flush=True,
        )
        runs.append(run)
    return runs


def run_complex_nontrivial_gap(force=False):
    out = DATA_DIR / "reciprocal_nontrivial_gap_runs.npz"
    if out.exists() and not force:
        cached = _load_npz_dict(out)
        if (
            "eps_reg" in cached
            and "damping" in cached
            and str(np.asarray(cached.get("material_update", "")).item())
            == "response_conditioned_log"
            and np.isclose(float(np.asarray(cached.get("response_metric_eta", np.nan)).item()), 0.02)
            and np.isclose(float(np.asarray(cached.get("response_metric_lambda_ratio", np.nan)).item()), 0.025)
            and int(np.asarray(cached.get("n_steps", -1)).item()) == 3000
            and int(np.asarray(cached.get("batch", -1)).item()) == 10
            and int(np.asarray(cached.get("n_freq", -1)).item()) == 8
            and _structured_has_fields(
                cached["metrics"],
                ("material_ratio", "frac_at_bounds"),
            )
        ):
            write_complex_nontrivial_table(cached["metrics"])
            return cached
        print("  cached reciprocal nontrivial metrics are stale; regenerating", flush=True)

    runs = _scalar_ensemble_runs(force=force)
    metrics = np.array(
        [
            (
                int(run["net_seed"]),
                *tuple(spectral_nontrivial_metrics(run).values()),
            )
            for run in runs
        ],
        dtype=[
            ("net_seed", "i4"),
            ("size", "i4"),
            ("N", "i4"),
            ("n_in_initial", "i4"),
            ("n_in_final", "i4"),
            ("gap_width", "f8"),
            ("target_width", "f8"),
            ("gap_ratio", "f8"),
            ("local_spacing", "f8"),
            ("gap_over_spacing", "f8"),
            ("target_over_spacing", "f8"),
            ("mean_frequency_scale", "f8"),
            ("mean_stiffness_scale", "f8"),
            ("material_scale", "f8"),
            ("uniform_n_same_mean_stiffness", "i4"),
            ("uniform_n_same_material", "i4"),
            ("uniform_n_same_mean_frequency", "i4"),
            ("n_final_norm_mean", "i4"),
            ("n_final_norm_stiffness", "i4"),
            ("final_gap_lo_norm_stiffness", "f8"),
            ("final_gap_hi_norm_stiffness", "f8"),
            ("n_initial_in_final_norm_gap", "i4"),
            ("n_final_in_final_norm_gap", "i4"),
            ("final_norm_gap_over_spacing", "f8"),
            ("mean_radius", "f8"),
            ("mean_stiffness_ratio", "f8"),
            ("material_ratio", "f8"),
            ("frac_rmin", "f8"),
            ("frac_rmax", "f8"),
            ("frac_at_bounds", "f8"),
        ],
    )
    save = {
        "metrics": metrics,
        "damping": REAL_SHIFT,
        "eps_reg": np.float64(EPS_REG),
        "material_update": np.asarray("response_conditioned_log"),
        "response_metric_eta": np.float64(0.02),
        "response_metric_lambda_ratio": np.float64(0.025),
        "n_steps": np.int64(3000),
        "batch": np.int64(10),
        "n_freq": np.int64(8),
    }
    for idx, run in enumerate(runs):
        prefix = f"run{idx}_"
        for key in (
            "size", "pos", "edges", "lengths", "box", "radii", "f0", "ff",
            "wlo", "whi", "gap_lo", "gap_hi", "gap_ratio",
            "n_in_initial", "n_in_final", "net_seed",
        ):
            save[prefix + key] = run[key]
    np.savez_compressed(out, **save)
    write_complex_nontrivial_table(metrics)
    return _load_npz_dict(out)


def write_complex_nontrivial_table(metrics):
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & Median & IQR & Range \\\\",
        "\\midrule",
    ]
    rows = [
        (
            "initial $n_{\\rm in}$",
            metrics["n_in_initial"],
            "{:.1f}",
        ),
        (
            "final $n_{\\rm in}$",
            metrics["n_in_final"],
            "{:.1f}",
        ),
        (
            "$\\Delta\\omega/\\langle\\delta\\omega\\rangle$",
            metrics["gap_over_spacing"],
            "{:.1f}",
        ),
        (
            "uniform-stiffness $n_{\\rm in}$",
            metrics["uniform_n_same_mean_stiffness"],
            "{:.1f}",
        ),
        (
            "stiffness-normalized $G_{\\rm stiff}$",
            metrics["final_norm_gap_over_spacing"],
            "{:.1f}",
        ),
        (
            "$\\langle r_e\\rangle$",
            metrics["mean_radius"],
            "{:.3f}",
        ),
        (
            "mean stiffness ratio",
            metrics["mean_stiffness_ratio"],
            "{:.3f}",
        ),
        (
            "material ratio",
            metrics["material_ratio"],
            "{:.3f}",
        ),
        (
            "edges at $R_{\\min}$ (\\%)",
            100.0 * metrics["frac_rmin"],
            "{:.1f}",
        ),
        (
            "edges at $R_{\\max}$ (\\%)",
            100.0 * metrics["frac_rmax"],
            "{:.1f}",
        ),
    ]
    for label, values, fmt in rows:
        vals = np.asarray(values, dtype=float)
        lines.append(
            f"{label} & {fmt.format(float(np.median(vals)))} & "
            f"{fmt.format(float(np.percentile(vals, 25)))}--"
            f"{fmt.format(float(np.percentile(vals, 75)))} & "
            f"{fmt.format(float(np.min(vals)))}--{fmt.format(float(np.max(vals)))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (FIG_DIR / "table_nontrivial_gap.tex").write_text("\n".join(lines))


def plot_complex_nontrivial_gap(data):
    _style()
    idx = 0
    f0 = data[f"run{idx}_f0"]
    ff = data[f"run{idx}_ff"]
    lengths = data[f"run{idx}_lengths"]
    radii = data[f"run{idx}_radii"]
    wlo = float(data[f"run{idx}_wlo"])
    whi = float(data[f"run{idx}_whi"])
    k0_mean = float(np.mean(1.0 / lengths))
    kf_mean = float(np.mean(radii**2 / lengths))

    # Single-column: spectral-window DOS (a) and normalized DOS (b) on top,
    # finite-size scaling (c, dual-axis) spanning the bottom.
    fig = plt.figure(figsize=(3.4, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    bins = _bins_with_markers(
        0, np.percentile(ff[ff > 0], 98) * 1.05, 36, [wlo, whi],
    )
    ax_a.hist(f0[f0 > 1e-9], bins=bins, histtype="step", color="#6b7280", lw=1.2, label="initial")
    ax_a.hist(ff[ff > 1e-9], bins=bins, histtype="stepfilled", color="#2b6cb0", alpha=0.35, label="learned")
    ax_a.axvspan(wlo, whi, color="#e76f51", alpha=0.16, lw=0)
    ax_a.set_xlabel(r"$\omega$")
    ax_a.set_ylabel("modes")
    ax_a.set_title("spectral-window DOS")
    ax_a.legend(frameon=False, loc="upper right")
    _panel_label(ax_a, "a")

    f0n = f0 / np.sqrt(k0_mean)
    ffn = ff / np.sqrt(kf_mean)
    gap_lo_norm = float(data[f"run{idx}_gap_lo"]) / np.sqrt(kf_mean)
    gap_hi_norm = float(data[f"run{idx}_gap_hi"]) / np.sqrt(kf_mean)
    bins_n = _bins_with_markers(
        0, np.percentile(ffn[ffn > 0], 98) * 1.05, 36,
        [gap_lo_norm, gap_hi_norm],
    )
    ax_b.hist(f0n[f0n > 1e-9], bins=bins_n, histtype="step", color="#6b7280", lw=1.2)
    ax_b.hist(ffn[ffn > 1e-9], bins=bins_n, histtype="step", color="#2b6cb0", lw=1.4)
    ax_b.axvspan(gap_lo_norm, gap_hi_norm, color="#e76f51", alpha=0.16, lw=0)
    ax_b.set_xlabel(r"$\omega/\sqrt{\langle k_e\rangle}$")
    ax_b.set_ylabel("modes")
    ax_b.set_title("normalized learned gap")
    _panel_label(ax_b, "b")

    fs = _load_npz_dict(DATA_DIR / "finite_size_scaling_real.npz")["records"]
    fs_sizes = np.unique(fs["size"])
    norm_mean = np.array([np.nanmean(fs[fs["size"] == s]["final_norm_gap_over_spacing"]) for s in fs_sizes])
    norm_std = np.array([np.nanstd(fs[fs["size"] == s]["final_norm_gap_over_spacing"]) for s in fs_sizes])
    abs_mean = np.array([np.mean(fs[fs["size"] == s]["abs_gap"]) for s in fs_sizes])
    abs_std = np.array([np.std(fs[fs["size"] == s]["abs_gap"]) for s in fs_sizes])
    ax_c.errorbar(
        fs_sizes, norm_mean, yerr=norm_std, fmt="o-", color="#2b6cb0",
        ms=4, lw=1.1, capsize=2,
    )
    ax_c.set_xlabel(r"system size $L$")
    ax_c.set_ylabel(r"$G_{\rm stiff}$", color="#2b6cb0")
    ax_c.tick_params(axis="y", labelcolor="#2b6cb0")
    ax_c.set_ylim(0, norm_mean.max() * 1.25)
    ax_c.set_xticks(fs_sizes)
    ax_c2 = ax_c.twinx()
    ax_c2.errorbar(
        fs_sizes, abs_mean, yerr=abs_std, fmt="s--", color="#bc6c25",
        ms=4, lw=1.1, capsize=2,
    )
    ax_c2.set_ylabel(r"absolute gap $\Delta\omega$", color="#bc6c25")
    ax_c2.tick_params(axis="y", labelcolor="#bc6c25")
    ax_c2.set_ylim(0, abs_mean.max() * 1.7)
    _panel_label(ax_c, "c")

    # panels are described in the caption; no per-panel titles (PRL style)
    for ax in fig.axes:
        ax.set_title("")
    fig.savefig(FIG_DIR / "fig2_nontrivial_gap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_nontrivial_gap.png", bbox_inches="tight")
    plt.close(fig)


def _control_train_ready(network, mode, train_seed, n_steps=850,
                         batch=5, n_freq=5, damping=COMPLEX_DAMPING):
    edges = network["edges"]
    lengths = network["lengths"]
    N = len(network["pos"])
    M_node = 1.0
    R_INIT = 1.0
    R_MIN = 0.5
    R_MAX = 2.0
    alpha = 0.05
    eval_every = 100

    radii = np.full(len(edges), R_INIT, dtype=np.float64)
    f0 = eigenfreqs(edges, radii, lengths, N, M_node)
    terms, wlo, whi = default_objective_terms(f0, n_freq=n_freq)
    omegas = np.array([t.omega for t in terms], dtype=np.float64)
    n_in_initial = int(np.sum((f0 > wlo) & (f0 < whi)))
    rng = np.random.RandomState(train_seed)
    frequency_rng = np.random.RandomState(int(train_seed) + 918273)
    reg = inflation(0.03)
    nin_samples = [(0, n_in_initial)]
    cost_samples = []

    for step in range(int(n_steps)):
        forces = rng.randn(batch, N)
        step_omegas = omegas[[frequency_rng.randint(len(omegas))]]
        if mode == "inflation_only":
            cost = np.nan
            grad = np.zeros(len(edges), dtype=np.float64)
        elif mode == "forward_only":
            cost, grad = response_cost_grad_fixed(
                edges, radii, lengths, N, M_node, step_omegas, forces,
                damping=damping, mode="forward_only",
            )
        elif mode in ("adjoint", "shuffled", "random_matched"):
            cost, grad = response_cost_grad_fixed(
                edges, radii, lengths, N, M_node, step_omegas, forces,
                damping=damping, mode="adjoint",
            )
            if mode == "shuffled":
                grad = rng.permutation(grad)
            elif mode == "random_matched":
                signs = rng.choice(np.array([-1.0, 1.0]), size=len(grad))
                grad = rng.permutation(np.abs(grad)) * signs
        else:
            raise ValueError(f"unknown control mode {mode!r}")

        grad = grad + reg(radii, lengths)
        radii = np.clip(radii - alpha * grad, R_MIN, R_MAX)

        if (step + 1) % eval_every == 0:
            freqs = eigenfreqs(edges, radii, lengths, N, M_node)
            n_in, _, _, _ = _gap_metrics(freqs, wlo, whi)
            nin_samples.append((step + 1, n_in))
            cost_samples.append((step + 1, cost))

    ff = eigenfreqs(edges, radii, lengths, N, M_node)
    n_in_final, gap_lo, gap_hi, gap_ratio = _gap_metrics(ff, wlo, whi)
    material = edge_material_diagnostics(radii, lengths, R_MIN, R_MAX)
    return {
        "mode": mode,
        "train_seed": int(train_seed),
        "n_steps": int(n_steps),
        "n_in_initial": n_in_initial,
        "n_in_final": n_in_final,
        "gap_ratio": float(gap_ratio),
        "gap_lo": float(gap_lo),
        "gap_hi": float(gap_hi),
        "wlo": float(wlo),
        "whi": float(whi),
        "f0": f0,
        "ff": ff,
        "radii": radii,
        "nin_samples": np.array(nin_samples, dtype=np.float64),
        "cost_samples": np.array(cost_samples, dtype=np.float64),
        **material,
    }


def _uniform_reference(network, adjoint_result, mode="same_mean_stiffness"):
    edges = network["edges"]
    lengths = network["lengths"]
    N = len(network["pos"])
    f0 = np.asarray(adjoint_result["f0"], dtype=np.float64)
    wlo = float(adjoint_result["wlo"])
    whi = float(adjoint_result["whi"])
    radii_ref = np.asarray(adjoint_result["radii"], dtype=np.float64)
    k0_mean = float(np.mean(1.0 / lengths))
    kf_mean = float(np.mean(radii_ref**2 / lengths))
    material0 = float(np.sum(lengths))
    materialf = float(np.sum(radii_ref**2 * lengths))
    if mode == "same_mean_stiffness":
        scale = np.sqrt(kf_mean / k0_mean)
    elif mode == "same_material":
        scale = np.sqrt(materialf / material0)
    else:
        raise ValueError(f"unknown uniform mode {mode!r}")
    radii = np.full(len(edges), scale, dtype=np.float64)
    ff = eigenfreqs(edges, radii, lengths, N, 1.0)
    n_in_final, gap_lo, gap_hi, gap_ratio = _gap_metrics(ff, wlo, whi)
    return {
        "mode": f"uniform_{mode}",
        "train_seed": int(adjoint_result["train_seed"]),
        "n_steps": 0,
        "n_in_initial": int(adjoint_result["n_in_initial"]),
        "n_in_final": int(n_in_final),
        "gap_ratio": float(gap_ratio),
        "gap_lo": float(gap_lo),
        "gap_hi": float(gap_hi),
        "wlo": float(wlo),
        "whi": float(whi),
        "f0": f0,
        "ff": ff,
        "radii": radii,
        "nin_samples": np.array([[0, n_in_final]], dtype=np.float64),
        "cost_samples": np.zeros((0, 2), dtype=np.float64),
    }


def _damping_suffix(damping):
    return "complex" if damping == COMPLEX_DAMPING else "real"


def run_expanded_controls(force=False, damping=COMPLEX_DAMPING):
    out = DATA_DIR / f"expanded_controls_{_damping_suffix(damping)}.npz"
    if out.exists() and not force:
        cached = _load_npz_dict(out)
        if (
            "eps_reg" in cached
            and _structured_has_fields(
                cached["records"],
                ("material_ratio", "frac_at_bounds"),
            )
        ):
            write_expanded_controls_table(cached["records"])
            return cached
        print("  cached expanded controls are stale; regenerating", flush=True)

    records = []
    trajectories = {}
    spectra = {}
    modes = (
        "adjoint", "shuffled", "forward_only", "random_matched",
        "inflation_only",
    )
    n_reps = 8
    size = 8
    for rep in range(n_reps):
        network = generate_from_config({
            "network": {"topology": "rand-del", "size": size, "seed": rep},
            "training": {},
        })
        adjoint_result = None
        for mode in modes:
            t0 = time.perf_counter()
            result = _control_train_ready(
                network, mode, train_seed=2000 + 19 * rep,
                n_steps=850, batch=5, n_freq=5, damping=damping,
            )
            dt = time.perf_counter() - t0
            if mode == "adjoint":
                adjoint_result = result
            print(
                f"  controls rep={rep} mode={mode} "
                f"n_in={result['n_in_initial']}->{result['n_in_final']} "
                f"gap={result['gap_ratio']:.3f}",
                flush=True,
            )
            records.append((
                rep, mode, result["n_in_initial"], result["n_in_final"],
                result["gap_ratio"], result["mean_radius"],
                result["mean_stiffness_ratio"], result["material_ratio"],
                result["frac_rmin"], result["frac_rmax"],
                result["frac_at_bounds"], dt,
            ))
            if rep == 0:
                trajectories[mode] = result["nin_samples"]
                spectra[f"{mode}_f0"] = result["f0"]
                spectra[f"{mode}_ff"] = result["ff"]
                spectra[f"{mode}_wlo"] = np.float64(result["wlo"])
                spectra[f"{mode}_whi"] = np.float64(result["whi"])

        for uniform_mode in ("same_mean_stiffness", "same_material"):
            result = _uniform_reference(network, adjoint_result, mode=uniform_mode)
            label = result["mode"]
            records.append((
                rep, label, result["n_in_initial"], result["n_in_final"],
                result["gap_ratio"],
                float(np.mean(result["radii"])),
                float(np.mean(result["radii"]**2 / network["lengths"]))
                / float(np.mean(1.0 / network["lengths"])),
                float(np.sum(result["radii"]**2 * network["lengths"]))
                / float(np.sum(network["lengths"])),
                float(np.mean(result["radii"] <= 0.5 + 1e-6)),
                float(np.mean(result["radii"] >= 2.0 - 1e-6)),
                float(
                    np.mean(result["radii"] <= 0.5 + 1e-6)
                    + np.mean(result["radii"] >= 2.0 - 1e-6)
                ),
                0.0,
            ))
            if rep == 0:
                trajectories[label] = result["nin_samples"]
                spectra[f"{label}_f0"] = result["f0"]
                spectra[f"{label}_ff"] = result["ff"]
                spectra[f"{label}_wlo"] = np.float64(result["wlo"])
                spectra[f"{label}_whi"] = np.float64(result["whi"])

    records = np.array(
        records,
        dtype=[
            ("rep", "i4"),
            ("mode", "U32"),
            ("n_in_initial", "i4"),
            ("n_in_final", "i4"),
            ("gap_ratio", "f8"),
            ("mean_radius", "f8"),
            ("mean_stiffness_ratio", "f8"),
            ("material_ratio", "f8"),
            ("frac_rmin", "f8"),
            ("frac_rmax", "f8"),
            ("frac_at_bounds", "f8"),
            ("seconds", "f8"),
        ],
    )
    save = {
        "records": records,
        "n_reps": np.int64(n_reps),
        "size": np.int64(size),
        "damping": damping,
        "eps_reg": np.float64(
            COMPLEX_EPS_REG if damping == COMPLEX_DAMPING else EPS_REG
        ),
    }
    for mode, arr in trajectories.items():
        save[f"{mode}_nin_samples"] = arr
    save.update(spectra)
    np.savez_compressed(out, **save)
    write_expanded_controls_table(records)
    return _load_npz_dict(out)


def write_expanded_controls_table(records):
    order = [
        "adjoint", "shuffled", "forward_only", "random_matched",
        "inflation_only", "uniform_same_mean_stiffness", "uniform_same_material",
    ]
    label = {
        "adjoint": "paired response",
        "shuffled": "shuffled",
        "forward_only": "forward-only",
        "random_matched": "random matched",
        "inflation_only": "inflation only",
        "uniform_same_mean_stiffness": "uniform stiffness",
        "uniform_same_material": "uniform material",
    }
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Control & success & final $n_{\\rm in}$ & "
        "$\\Delta\\omega/\\omega_{\\rm mid}^{\\rm gap}$ (successes) & "
        "material ratio & bound edges \\\\",
        "\\midrule",
    ]
    for mode in order:
        rows = records[records["mode"] == mode]
        if len(rows) == 0:
            continue
        success = 100.0 * np.mean(rows["n_in_final"] == 0)
        nin = float(np.mean(rows["n_in_final"]))
        successful = rows[rows["n_in_final"] == 0]
        gap = f"{float(np.mean(successful['gap_ratio'])):.3f}" if len(successful) else "--"
        material = float(np.mean(rows["material_ratio"]))
        bound = 100.0 * float(np.mean(rows["frac_at_bounds"]))
        lines.append(
            f"{label[mode]} & {success:.0f}\\% & {nin:.1f} & {gap} & "
            f"{material:.2f} & {bound:.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (FIG_DIR / "table_expanded_controls.tex").write_text("\n".join(lines))


def run_forward_inflation_diagnostic(force=False, damping=COMPLEX_DAMPING):
    out = DATA_DIR / f"forward_inflation_diagnostic_{_damping_suffix(damping)}.npz"
    if out.exists() and not force:
        cached = _load_npz_dict(out)
        if "eps_reg" in cached:
            write_forward_inflation_table(cached["records"])
            return cached
        print("  cached forward/inflation diagnostic is stale; regenerating", flush=True)

    records = []
    size = 8
    n_reps = 8
    for rep in range(n_reps):
        network = generate_from_config({
            "network": {"topology": "rand-del", "size": size, "seed": rep},
            "training": {},
        })
        edges = network["edges"]
        lengths = network["lengths"]
        N = len(network["pos"])
        radii = np.ones(len(edges), dtype=np.float64)
        f0 = eigenfreqs(edges, radii, lengths, N, 1.0)
        terms, _, _ = default_objective_terms(f0, n_freq=5)
        omegas = np.array([t.omega for t in terms], dtype=np.float64)
        rng = np.random.RandomState(2000 + 19 * rep)
        forces = rng.randn(5, N)
        _, forward_grad = response_cost_grad_fixed(
            edges, radii, lengths, N, 1.0, omegas, forces,
            damping=damping, mode="forward_only",
        )
        inflation_grad = inflation(0.03)(radii, lengths)
        total_grad = forward_grad + inflation_grad
        records.append((
            rep,
            float(np.mean(forward_grad < 0.0)),
            float(np.mean(total_grad < 0.0)),
            float(np.mean(np.abs(forward_grad))),
            float(np.mean(np.abs(inflation_grad))),
            float(np.mean(np.abs(forward_grad)) / np.mean(np.abs(inflation_grad))),
            float(np.max(np.abs(forward_grad))),
        ))

    records = np.array(
        records,
        dtype=[
            ("rep", "i4"),
            ("forward_stiffening_fraction", "f8"),
            ("total_stiffening_fraction", "f8"),
            ("mean_abs_forward_grad", "f8"),
            ("mean_abs_inflation_grad", "f8"),
            ("forward_to_inflation_ratio", "f8"),
            ("max_abs_forward_grad", "f8"),
        ],
    )
    np.savez_compressed(
        out,
        records=records,
        size=np.int64(size),
        n_reps=np.int64(n_reps),
        damping=damping,
        eps_reg=np.float64(
            COMPLEX_EPS_REG if damping == COMPLEX_DAMPING else EPS_REG
        ),
    )
    write_forward_inflation_table(records)
    return _load_npz_dict(out)


def write_forward_inflation_table(records):
    rows = [
        (
            "raw forward gradients that stiffen",
            100.0 * records["forward_stiffening_fraction"],
            "{:.1f}\\%",
        ),
        (
            "forward+inflation total updates that stiffen",
            100.0 * records["total_stiffening_fraction"],
            "{:.1f}\\%",
        ),
        (
            "$\\langle |g_{\\rm fwd}|\\rangle/|g_{\\rm infl}|$",
            records["forward_to_inflation_ratio"],
            "{:.3f}",
        ),
        (
            "$\\max |g_{\\rm fwd}|$",
            records["max_abs_forward_grad"],
            "{:.3f}",
        ),
    ]
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Diagnostic & mean & range \\\\",
        "\\midrule",
    ]
    for label, values, fmt in rows:
        vals = np.asarray(values, dtype=np.float64)
        lines.append(
            f"{label} & {fmt.format(float(np.mean(vals)))} & "
            f"{fmt.format(float(np.min(vals)))}--{fmt.format(float(np.max(vals)))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (FIG_DIR / "table_forward_inflation_diagnostic.tex").write_text("\n".join(lines))


def plot_expanded_controls(data):
    _style()
    records = data["records"]
    order = [
        "adjoint", "shuffled", "forward_only", "random_matched",
        "inflation_only", "uniform_same_mean_stiffness", "uniform_same_material",
    ]
    labels = ["paired", "shuffled", "forward", "random", "inflation", "uniform k", "uniform V"]
    colors = {
        "adjoint": "#2b6cb0",
        "shuffled": "#7c3aed",
        "forward_only": "#cc7722",
        "random_matched": "#a23e48",
        "inflation_only": "#6b7280",
        "uniform_same_mean_stiffness": "#5f6f52",
        "uniform_same_material": "#8b5e34",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    ax0, ax1, ax2 = axes

    for mode in order[:5]:
        key = f"{mode}_nin_samples"
        if key not in data:
            continue
        arr = data[key]
        ax0.plot(arr[:, 0], arr[:, 1], label=labels[order.index(mode)],
                 color=colors[mode], lw=1.2)
    ax0.set_xlabel("step")
    ax0.set_ylabel(r"modes in target")
    ax0.set_title("same network trajectories")
    ax0.legend(frameon=False, ncol=1)
    _panel_label(ax0, "a")

    success = []
    means = []
    errs = []
    for mode in order:
        rows = records[records["mode"] == mode]
        success.append(100.0 * np.mean(rows["n_in_final"] == 0))
        means.append(float(np.mean(rows["n_in_final"])))
        errs.append(float(np.std(rows["n_in_final"])))
    x = np.arange(len(order))
    ax1.bar(x, success, color=[colors[m] for m in order], width=0.72)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("success (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("success over seeds")
    _panel_label(ax1, "b")

    ax2.bar(x, means, yerr=errs, color=[colors[m] for m in order], width=0.72, capsize=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel(r"final $N_{\rm in}$")
    ax2.set_title("residual modes")
    _panel_label(ax2, "c")

    fig.savefig(FIG_DIR / "fig3_expanded_controls.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_expanded_controls.png", bbox_inches="tight")
    plt.close(fig)


def _percentile_window(f0, center, width):
    pos = np.asarray(f0, dtype=np.float64)
    pos = pos[pos > 1e-9]
    lo_p = max(1.0, float(center) - 0.5 * float(width))
    hi_p = min(99.0, float(center) + 0.5 * float(width))
    return float(np.percentile(pos, lo_p)), float(np.percentile(pos, hi_p))


def _window_train_ready(network, train_seed, window, n_steps=1000,
                        batch=6, n_freq=6, damping=REAL_SHIFT):
    if damping != REAL_SHIFT:
        raise ValueError('response-conditioned target screen requires real_shift')
    result = train_local(
        edges=network["edges"],
        lengths=network["lengths"],
        N=len(network["pos"]),
        n_steps=int(n_steps),
        batch=int(batch),
        n_freq=int(n_freq),
        grad_clip=None,
        damping=damping,
        force_distribution="gaussian",
        regularizers=[],
        material_update="response_conditioned_log",
        response_metric_eta=0.02,
        response_metric_lambda_ratio=0.025,
        target_window=tuple(float(x) for x in window),
        frequency_sampling="random_grid",
        frequencies_per_step=1,
        train_seed=int(train_seed),
        eval_every=100,
    )
    result["train_seed"] = int(train_seed)
    result["cost_samples"] = np.column_stack((
        np.arange(1, int(n_steps) + 1, dtype=np.float64),
        np.asarray(result["cost_history"], dtype=np.float64),
    ))
    return result


def run_target_robustness(force=False, damping=REAL_SHIFT):
    out = DATA_DIR / f"target_robustness_{_damping_suffix(damping)}.npz"
    if out.exists() and not force:
        cached = _load_npz_dict(out)
        if (
            "eps_reg" in cached
            and "damping" in cached
            and str(np.asarray(cached.get("material_update", "")).item())
            == "response_conditioned_log"
            and np.isclose(float(np.asarray(cached.get("response_metric_eta", np.nan)).item()), 0.02)
            and np.isclose(float(np.asarray(cached.get("response_metric_lambda_ratio", np.nan)).item()), 0.025)
            and int(np.asarray(cached.get("n_steps", -1)).item()) == 3000
            and int(np.asarray(cached.get("batch", -1)).item()) == 6
            and int(np.asarray(cached.get("n_freq", -1)).item()) == 6
            and bool(np.asarray(cached.get("matched_streams_across_centers", False)).item())
        ):
            write_target_robustness_table(cached["records"])
            return cached
        print("  cached target robustness is stale; regenerating", flush=True)

    records = []
    centers = (25, 35, 45, 55, 65, 75)
    width = 18
    size = 8
    n_reps = 10
    example_saved = False
    save = {"centers": np.array(centers, dtype=np.float64)}
    for rep in range(n_reps):
        base_network = generate_from_config({
            "network": {"topology": "rand-del", "size": size, "seed": rep},
            "training": {},
        })
        base_f0 = eigenfreqs(
            base_network["edges"], np.ones(len(base_network["edges"])),
            base_network["lengths"], len(base_network["pos"]), 1.0,
        )
        for center in centers:
            window = _percentile_window(base_f0, center, width)
            t0 = time.perf_counter()
            result = _window_train_ready(
                base_network,
                train_seed=3000 + 31 * rep,
                window=window,
                n_steps=3000,
                batch=6,
                n_freq=6,
                damping=damping,
            )
            elapsed = time.perf_counter() - t0
            network = base_network
            metrics = spectral_nontrivial_metrics(_pack_run(network, result, elapsed))
            success = int(result["n_in_final"] == 0)
            records.append((
                rep, center, window[0], window[1],
                int(result["n_in_initial"]), int(result["n_in_final"]),
                float(result["gap_ratio"]), float(metrics["gap_over_spacing"]),
                success, elapsed,
            ))
            print(
                f"  targets rep={rep} p={center} "
                f"n_in={result['n_in_initial']}->{result['n_in_final']} "
                f"gap={result['gap_ratio']:.3f}",
                flush=True,
            )
            if not example_saved and int(center) == 55:
                save.update({
                    "example_f0": result["f0"],
                    "example_ff": result["ff"],
                    "example_wlo": np.float64(result["wlo"]),
                    "example_whi": np.float64(result["whi"]),
                })
                example_saved = True

    records = np.array(
        records,
        dtype=[
            ("rep", "i4"),
            ("center_percentile", "f8"),
            ("wlo", "f8"),
            ("whi", "f8"),
            ("n_in_initial", "i4"),
            ("n_in_final", "i4"),
            ("gap_ratio", "f8"),
            ("gap_over_spacing", "f8"),
            ("success", "i4"),
            ("seconds", "f8"),
        ],
    )
    save["records"] = records
    save["size"] = np.int64(size)
    save["width_percentile"] = np.float64(width)
    save["damping"] = damping
    save["eps_reg"] = np.float64(EPS_REG)
    save["material_update"] = np.asarray("response_conditioned_log")
    save["response_metric_eta"] = np.float64(0.02)
    save["response_metric_lambda_ratio"] = np.float64(0.025)
    save["n_steps"] = np.int64(3000)
    save["batch"] = np.int64(6)
    save["n_freq"] = np.int64(6)
    save["matched_streams_across_centers"] = np.bool_(True)
    np.savez_compressed(out, **save)
    write_target_robustness_table(records)
    return _load_npz_dict(out)


def write_target_robustness_table(records):
    lines = [
        "\\begin{tabular}{rccc}",
        "\\toprule",
        "Center percentile & success & final $n_{\\rm in}$ & "
        "$\\Delta\\omega/\\omega_{\\rm mid}^{\\rm gap}$ (successes) \\\\",
        "\\midrule",
    ]
    for center in np.unique(records["center_percentile"]):
        rows = records[records["center_percentile"] == center]
        successful = rows[rows["n_in_final"] == 0]
        gap = f"{float(np.mean(successful['gap_ratio'])):.3f}" if len(successful) else "--"
        lines.append(
            f"{center:.0f} & {100.0 * np.mean(rows['success']):.0f}\\% & "
            f"{float(np.mean(rows['n_in_final'])):.1f} & "
            f"{gap} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (FIG_DIR / "table_target_robustness.tex").write_text("\n".join(lines))


def _centralized_reference_one(network, train_seed, n_steps=700,
                               batch=24, n_freq=6,
                               damping=COMPLEX_DAMPING):
    edges = network["edges"]
    lengths = network["lengths"]
    N = len(network["pos"])
    M_node = 1.0
    R_MIN = 0.5
    R_MAX = 2.0
    alpha = 0.05
    grad_clip = 3.0
    eval_every = 100
    radii = np.ones(len(edges), dtype=np.float64)
    f0 = eigenfreqs(edges, radii, lengths, N, M_node)
    terms, wlo, whi = default_objective_terms(f0, n_freq=n_freq)
    omegas = np.array([t.omega for t in terms], dtype=np.float64)
    rng = np.random.RandomState(train_seed)
    forces = rng.randn(batch, N)
    reg = inflation(0.03)
    n_in_initial = int(np.sum((f0 > wlo) & (f0 < whi)))
    nin_samples = [(0, n_in_initial)]
    cost_samples = []
    for step in range(int(n_steps)):
        cost, grad = response_cost_grad_fixed(
            edges, radii, lengths, N, M_node, omegas, forces,
            damping=damping, mode="adjoint",
        )
        grad = grad + reg(radii, lengths)
        grad = np.clip(grad, -grad_clip, grad_clip)
        radii = np.clip(radii - alpha * grad, R_MIN, R_MAX)
        if (step + 1) % eval_every == 0:
            freqs = eigenfreqs(edges, radii, lengths, N, M_node)
            n_in, _, _, _ = _gap_metrics(freqs, wlo, whi)
            nin_samples.append((step + 1, n_in))
            cost_samples.append((step + 1, cost))
    ff = eigenfreqs(edges, radii, lengths, N, M_node)
    n_in_final, gap_lo, gap_hi, gap_ratio = _gap_metrics(ff, wlo, whi)
    return {
        "n_in_initial": n_in_initial,
        "n_in_final": int(n_in_final),
        "gap_ratio": float(gap_ratio),
        "gap_lo": float(gap_lo),
        "gap_hi": float(gap_hi),
        "wlo": float(wlo),
        "whi": float(whi),
        "f0": f0,
        "ff": ff,
        "radii": radii,
        "nin_samples": np.array(nin_samples, dtype=np.float64),
        "cost_samples": np.array(cost_samples, dtype=np.float64),
    }


def run_centralized_reference(force=False, damping=COMPLEX_DAMPING):
    out = DATA_DIR / f"centralized_reference_{_damping_suffix(damping)}.npz"
    if out.exists() and not force:
        cached = _load_npz_dict(out)
        if "eps_reg" in cached and "damping" in cached:
            write_centralized_reference_table(cached["records"])
            return cached
        print("  cached centralized reference is stale; regenerating", flush=True)

    records = []
    trajectories = {}
    size = 8
    n_reps = 4
    for rep in range(n_reps):
        network = generate_from_config({
            "network": {"topology": "rand-del", "size": size, "seed": rep},
            "training": {},
        })
        t0 = time.perf_counter()
        local = _control_train_ready(
            network, "adjoint", train_seed=4000 + 23 * rep,
            n_steps=850, batch=5, n_freq=5, damping=damping,
        )
        local_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        central = _centralized_reference_one(
            network, train_seed=5000 + 23 * rep,
            n_steps=700, batch=24, n_freq=6, damping=damping,
        )
        central_t = time.perf_counter() - t0
        for label, result, elapsed in (
            ("local_stochastic", local, local_t),
            ("centralized_response", central, central_t),
        ):
            records.append((
                rep, label, result["n_in_initial"], result["n_in_final"],
                result["gap_ratio"], elapsed,
            ))
        if rep == 0:
            trajectories["local_stochastic"] = local["nin_samples"]
            trajectories["centralized_response"] = central["nin_samples"]
        print(
            f"  central rep={rep} local {local['n_in_initial']}->{local['n_in_final']} "
            f"central {central['n_in_initial']}->{central['n_in_final']}",
            flush=True,
        )

    records = np.array(
        records,
        dtype=[
            ("rep", "i4"),
            ("mode", "U32"),
            ("n_in_initial", "i4"),
            ("n_in_final", "i4"),
            ("gap_ratio", "f8"),
            ("seconds", "f8"),
        ],
    )
    save = {
        "records": records,
        "size": np.int64(size),
        "n_reps": np.int64(n_reps),
        "damping": damping,
        "eps_reg": np.float64(
            COMPLEX_EPS_REG if damping == COMPLEX_DAMPING else EPS_REG
        ),
    }
    for key, value in trajectories.items():
        save[f"{key}_nin_samples"] = value
    np.savez_compressed(out, **save)
    write_centralized_reference_table(records)
    return _load_npz_dict(out)


def write_centralized_reference_table(records):
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & success & final $n_{\\rm in}$ & "
        "$\\Delta\\omega/\\omega_{\\rm mid}$ (successes) \\\\",
        "\\midrule",
    ]
    labels = {
        "local_stochastic": "local stochastic",
        "centralized_response": "fixed-probe response",
    }
    for mode in ("local_stochastic", "centralized_response"):
        rows = records[records["mode"] == mode]
        successful = rows[rows["n_in_final"] == 0]
        gap = f"{float(np.mean(successful['gap_ratio'])):.3f}" if len(successful) else "--"
        lines.append(
            f"{labels[mode]} & {100.0 * np.mean(rows['n_in_final'] == 0):.0f}\\% & "
            f"{float(np.mean(rows['n_in_final'])):.1f} & "
            f"{gap} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (FIG_DIR / "table_centralized_reference.tex").write_text("\n".join(lines))


def plot_robustness_reference(target_data, central_data):
    _style()
    records = target_data["records"]
    central = central_data["records"]
    centers = np.unique(records["center_percentile"])
    success = []
    success_err = []
    gap = []
    gap_err = []
    for center in centers:
        rows = records[records["center_percentile"] == center]
        p = np.mean(rows["success"])
        success.append(100.0 * p)
        success_err.append(100.0 * np.sqrt(p * (1 - p) / max(len(rows), 1)))
        gap.append(float(np.mean(rows["gap_ratio"])))
        gap_err.append(float(np.std(rows["gap_ratio"])))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 2.45),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.16], "wspace": 0.55},
    )
    ax0, ax1, ax2 = axes
    ax0.errorbar(centers, success, yerr=success_err, marker="o", color="#2b6cb0", lw=1.2, capsize=2)
    ax0.set_xlabel("target center percentile")
    ax0.set_ylabel("success (%)")
    ax0.set_ylim(-5, 105)
    ax0.set_title("target robustness")
    _panel_label(ax0, "a")

    ax1.errorbar(centers, gap, yerr=gap_err, marker="s", color="#5f6f52", lw=1.2, capsize=2)
    ax1.set_xlabel("target center percentile")
    ax1.set_ylabel("gap ratio")
    ax1.set_title("gap size")
    _panel_label(ax1, "b")

    mode_order = ["local_stochastic", "centralized_response"]
    labels = ["local", "central"]
    means = [float(np.mean(central[central["mode"] == m]["gap_ratio"])) for m in mode_order]
    errs = [float(np.std(central[central["mode"] == m]["gap_ratio"])) for m in mode_order]
    succ = [100.0 * float(np.mean(central[central["mode"] == m]["n_in_final"] == 0)) for m in mode_order]
    x = np.arange(len(mode_order))
    ax2.bar(x - 0.17, succ, width=0.34, color="#2b6cb0", label="success")
    ax2b = ax2.twinx()
    ax2b.bar(x + 0.17, means, yerr=errs, width=0.34, color="#cc7722", label="gap", capsize=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 105)
    ax2b.set_ylim(0, 0.75)
    ax2.set_ylabel("success (%)", labelpad=2)
    ax2b.set_ylabel("gap ratio", labelpad=2)
    ax2.set_title("central reference")
    _panel_label(ax2, "c")

    handles, labels0 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(
        handles + handles2, labels0 + labels2,
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, handlelength=1.2,
    )
    fig.savefig(FIG_DIR / "fig4_robustness_reference.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_robustness_reference.png", bbox_inches="tight")
    plt.close(fig)


def _finite_size_training_steps(size):
    return 3000


def _finite_size_run(size, rep, damping=REAL_SHIFT):
    network = generate_from_config({
        "network": {"topology": "rand-del", "size": int(size), "seed": int(rep)},
        "training": {},
    })
    n_steps = _finite_size_training_steps(size)
    t0 = time.perf_counter()
    result = train_local(
        edges=network["edges"],
        lengths=network["lengths"],
        N=len(network["pos"]),
        n_steps=n_steps,
        batch=10,
        n_freq=8,
        alpha=0.05,
        grad_clip=None,
        material_update="response_conditioned_log",
        response_metric_eta=0.02,
        response_metric_lambda_ratio=0.025,
        frequency_sampling="random_grid",
        frequencies_per_step=1,
        damping=damping,
        force_distribution="gaussian",
        regularizers=[],
        train_seed=0,
        frequency_seed=918273,
        eval_every=max(100, n_steps // 5),
    )
    elapsed = time.perf_counter() - t0
    metrics = spectral_nontrivial_metrics(_pack_run(network, result, elapsed))
    return result, metrics, elapsed


def run_finite_size_scaling(force=False, damping=REAL_SHIFT):
    out = DATA_DIR / f"finite_size_scaling_{_damping_suffix(damping)}.npz"
    if out.exists() and not force:
        cached = _load_npz_dict(out)
        if (
            "eps_reg" in cached
            and str(np.asarray(cached.get("material_update", "")).item())
            == "response_conditioned_log"
            and np.isclose(float(np.asarray(cached.get("response_metric_eta", np.nan)).item()), 0.02)
            and np.isclose(float(np.asarray(cached.get("response_metric_lambda_ratio", np.nan)).item()), 0.025)
            and int(np.asarray(cached.get("n_steps", -1)).item()) == 3000
            and int(np.asarray(cached.get("batch", -1)).item()) == 10
            and int(np.asarray(cached.get("n_freq", -1)).item()) == 8
            and _structured_has_fields(
                cached["records"],
                ("material_ratio", "frac_at_bounds"),
            )
        ):
            write_finite_size_table(cached["records"])
            return cached
        print("  cached finite-size scaling is stale; regenerating", flush=True)

    records = []
    sizes = (6, 8, 10, 12)
    n_reps = 10
    for size in sizes:
        for rep in range(n_reps):
            result, metrics, elapsed = _finite_size_run(size, rep, damping=damping)
            records.append((
                int(size), int(size) * int(size), int(rep),
                int(result["n_in_initial"]), int(result["n_in_final"]),
                float(result["gap_ratio"]), float(metrics["gap_width"]),
                float(metrics["gap_over_spacing"]),
                float(metrics["final_norm_gap_over_spacing"]),
                int(metrics["uniform_n_same_mean_stiffness"]),
                float(metrics["mean_radius"]),
                float(metrics["mean_stiffness_ratio"]),
                float(metrics["material_ratio"]),
                float(metrics["frac_rmin"]),
                float(metrics["frac_rmax"]),
                float(metrics["frac_at_bounds"]),
                elapsed,
                int(result["n_in_final"] == 0),
            ))
            print(
                f"  finite size={size} rep={rep} "
                f"n_in={result['n_in_initial']}->{result['n_in_final']} "
                f"gap={result['gap_ratio']:.3f}",
                flush=True,
            )

    # Reuse the cached 20x20 reciprocal ensemble instead of rerunning the
    # expensive large-system trajectories.
    reciprocal20 = run_complex_nontrivial_gap(force=False)
    for row in reciprocal20["metrics"]:
        records.append((
            int(row["size"]), int(row["N"]), int(row["net_seed"]),
            int(row["n_in_initial"]), int(row["n_in_final"]),
            float(row["gap_ratio"]), float(row["gap_width"]),
            float(row["gap_over_spacing"]),
            float(row["final_norm_gap_over_spacing"]),
            int(row["uniform_n_same_mean_stiffness"]),
            float(row["mean_radius"]),
            float(row["mean_stiffness_ratio"]),
            float(row["material_ratio"]),
            float(row["frac_rmin"]),
            float(row["frac_rmax"]),
            float(row["frac_at_bounds"]),
            np.nan,
            int(row["n_in_final"] == 0),
        ))

    records = np.array(
        records,
        dtype=[
            ("size", "i4"),
            ("N", "i4"),
            ("rep", "i4"),
            ("n_in_initial", "i4"),
            ("n_in_final", "i4"),
            ("gap_ratio", "f8"),
            ("abs_gap", "f8"),
            ("gap_over_spacing", "f8"),
            ("final_norm_gap_over_spacing", "f8"),
            ("uniform_n_same_mean_stiffness", "i4"),
            ("mean_radius", "f8"),
            ("mean_stiffness_ratio", "f8"),
            ("material_ratio", "f8"),
            ("frac_rmin", "f8"),
            ("frac_rmax", "f8"),
            ("frac_at_bounds", "f8"),
            ("seconds", "f8"),
            ("success", "i4"),
        ],
    )
    np.savez_compressed(
        out,
        records=records,
        damping=damping,
        eps_reg=np.float64(EPS_REG),
        material_update=np.asarray("response_conditioned_log"),
        response_metric_eta=np.float64(0.02),
        response_metric_lambda_ratio=np.float64(0.025),
        n_steps=np.int64(3000),
        batch=np.int64(10),
        n_freq=np.int64(8),
    )
    write_finite_size_table(records)
    return _load_npz_dict(out)


def write_finite_size_table(records):
    lines = [
        "\\begin{tabular}{rcccccccc}",
        "\\toprule",
        "\\shortstack{network side length $L$\\\\($L\\times L$ nodes)} & "
        "runs & success & \\shortstack{mean final\\\\$n_{\\rm in}$} & "
        "$G_{\\rm stiff}$ & "
        "\\shortstack{absolute learned-gap\\\\width $\\Delta\\omega$} & "
        "\\shortstack{mean uniform-rescaling\\\\$n_{\\rm in}^{\\rm uni}$} & "
        "\\shortstack{mean material\\\\ratio} & "
        "\\shortstack{edges at either\\\\radius bound (\\%)} \\\\",
        "\\midrule",
    ]
    for size in np.unique(records["size"]):
        rows = records[records["size"] == size]
        successful = rows[rows["success"] == 1]
        g_stiff = (
            f"{float(np.mean(successful['final_norm_gap_over_spacing'])):.1f}"
            if len(successful) else "--"
        )
        abs_gap = (
            f"{float(np.mean(successful['abs_gap'])):.3f}"
            if len(successful) else "--"
        )
        lines.append(
            f"{int(size)} & {len(rows)} & "
            f"{100.0 * np.mean(rows['success']):.0f}\\% & "
            f"{float(np.mean(rows['n_in_final'])):.1f} & "
            f"{g_stiff} & "
            f"{abs_gap} & "
            f"{float(np.mean(rows['uniform_n_same_mean_stiffness'])):.1f} & "
            f"{float(np.mean(rows['material_ratio'])):.2f} & "
            f"{100.0 * float(np.mean(rows['frac_at_bounds'])):.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (FIG_DIR / "table_finite_size_scaling.tex").write_text("\n".join(lines))


def _regularizer_train_ready(network, arm, train_seed, damping=COMPLEX_DAMPING,
                             n_steps=850, batch=5, n_freq=5):
    specs = local_regularizer_arm_to_specs(arm)
    regs = regularizers_from_specs(specs)
    t0 = time.perf_counter()
    result = train_local(
        edges=network["edges"],
        lengths=network["lengths"],
        N=len(network["pos"]),
        n_steps=int(n_steps),
        batch=int(batch),
        n_freq=int(n_freq),
        alpha=0.05,
        grad_clip=None,
        frequency_sampling="random_grid",
        frequencies_per_step=1,
        damping=damping,
        force_distribution="gaussian",
        regularizers=regs,
        train_seed=int(train_seed),
        eval_every=max(100, int(n_steps) // 5),
    )
    elapsed = time.perf_counter() - t0
    metrics = spectral_nontrivial_metrics(_pack_run(network, result, elapsed))
    return result, metrics, elapsed, specs


def run_regularizer_robustness(force=False, damping=COMPLEX_DAMPING):
    out = DATA_DIR / f"regularizer_robustness_{_damping_suffix(damping)}.npz"
    if out.exists() and not force:
        cached = _load_npz_dict(out)
        if (
            "eps_reg" in cached
            and _structured_has_fields(
                cached["records"],
                ("material_ratio", "frac_at_bounds"),
            )
        ):
            write_regularizer_table(cached["records"])
            return cached
        print("  cached regularizer robustness is stale; regenerating", flush=True)

    arms = (
        "none", "inflate", "inflate-l1", "inflate-l2",
        "l1", "l2", "material", "stiffness", "binary",
    )
    n_reps = 10
    size = 8
    records = []
    for rep in range(n_reps):
        network = generate_from_config({
            "network": {"topology": "rand-del", "size": size, "seed": rep},
            "training": {},
        })
        for arm in arms:
            result, metrics, elapsed, specs = _regularizer_train_ready(
                network,
                arm,
                train_seed=7000 + 103 * rep + len(records),
                damping=damping,
            )
            records.append((
                rep, arm, int(result["n_in_initial"]), int(result["n_in_final"]),
                float(result["gap_ratio"]), float(metrics["final_norm_gap_over_spacing"]),
                float(metrics["mean_radius"]),
                float(metrics["mean_stiffness_ratio"]),
                float(metrics["material_ratio"]),
                float(metrics["frac_rmin"]),
                float(metrics["frac_rmax"]),
                float(metrics["frac_at_bounds"]),
                int(result["n_in_final"] == 0), elapsed, json.dumps(specs),
            ))
            print(
                f"  regularizer rep={rep} arm={arm} "
                f"n_in={result['n_in_initial']}->{result['n_in_final']} "
                f"gap={result['gap_ratio']:.3f}",
                flush=True,
            )

    records = np.array(
        records,
        dtype=[
            ("rep", "i4"),
            ("arm", "U32"),
            ("n_in_initial", "i4"),
            ("n_in_final", "i4"),
            ("gap_ratio", "f8"),
            ("final_norm_gap_over_spacing", "f8"),
            ("mean_radius", "f8"),
            ("mean_stiffness_ratio", "f8"),
            ("material_ratio", "f8"),
            ("frac_rmin", "f8"),
            ("frac_rmax", "f8"),
            ("frac_at_bounds", "f8"),
            ("success", "i4"),
            ("seconds", "f8"),
            ("specs", "U256"),
        ],
    )
    np.savez_compressed(
        out,
        records=records,
        size=np.int64(size),
        n_reps=np.int64(n_reps),
        damping=damping,
        eps_reg=np.float64(EPS_REG),
    )
    write_regularizer_table(records)
    return _load_npz_dict(out)


def write_regularizer_table(records):
    order = (
        "none", "inflate", "inflate-l1", "inflate-l2",
        "l1", "l2", "material", "stiffness", "binary",
    )
    labels = {
        "none": "none",
        "inflate": "inflation",
        "inflate-l1": "inflation + L1",
        "inflate-l2": "inflation + L2",
        "l1": "L1 only",
        "l2": "L2 only",
        "material": "material",
        "stiffness": "stiffness",
        "binary": "binary",
    }
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Local regularizer & success & final $n_{\\rm in}$ & "
        "$\\Delta\\omega/\\omega_{\\rm mid}^{\\rm gap}$ & "
        "material ratio & bound edges \\\\",
        "\\midrule",
    ]
    for arm in order:
        rows = records[records["arm"] == arm]
        if len(rows) == 0:
            continue
        lines.append(
            f"{labels[arm]} & {100.0 * np.mean(rows['success']):.0f}\\% & "
            f"{float(np.mean(rows['n_in_final'])):.1f} & "
            f"{float(np.mean(rows['gap_ratio'])):.3f} & "
            f"{float(np.mean(rows['material_ratio'])):.2f} & "
            f"{100.0 * float(np.mean(rows['frac_at_bounds'])):.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (FIG_DIR / "table_regularizer_robustness.tex").write_text("\n".join(lines))


def plot_scaling_regularizers(finite_data, target_data, regularizer_data):
    _style()
    finite = finite_data["records"]
    targets = target_data["records"]
    regs = regularizer_data["records"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 4.9),
        gridspec_kw={"hspace": 0.68, "wspace": 0.46},
    )
    ax0, ax1, ax2, ax3 = axes.ravel()

    sizes = np.unique(finite["size"])
    succ = []
    succ_err = []
    norm_gap = []
    norm_gap_err = []
    for size in sizes:
        rows = finite[finite["size"] == size]
        p = float(np.mean(rows["success"]))
        succ.append(100.0 * p)
        succ_err.append(100.0 * np.sqrt(p * (1.0 - p) / max(len(rows), 1)))
        norm_gap.append(float(np.nanmean(rows["final_norm_gap_over_spacing"])))
        norm_gap_err.append(float(np.nanstd(rows["final_norm_gap_over_spacing"])))

    ax0.errorbar(sizes, succ, yerr=succ_err, marker="o", color="#2b6cb0", lw=1.2, capsize=2)
    ax0.set_xlabel("linear size")
    ax0.set_ylabel("success (%)")
    ax0.set_ylim(-5, 105)
    ax0.set_title("finite-size success")
    _panel_label(ax0, "a")

    ax1.errorbar(sizes, norm_gap, yerr=norm_gap_err, marker="^", color="#5f6f52", lw=1.2, capsize=2)
    ax1.set_xlabel("linear size")
    ax1.set_ylabel(r"$G_{\rm stiff}$")
    ax1.set_title("normalized learned gap")
    _panel_label(ax1, "b")

    centers = np.unique(targets["center_percentile"])
    gap = []
    gap_err = []
    for center in centers:
        rows = targets[targets["center_percentile"] == center]
        gap.append(float(np.mean(rows["gap_ratio"])))
        gap_err.append(float(np.std(rows["gap_ratio"])))
    ax2.errorbar(centers, gap, yerr=gap_err, marker="s", color="#5f6f52", lw=1.2, capsize=2)
    ax2.set_xlabel("spectral-window center percentile")
    ax2.set_ylabel("gap ratio")
    ax2.set_title("spectral-window robustness")
    _panel_label(ax2, "c")

    arm_order = [
        "none", "inflate", "inflate-l1", "inflate-l2",
        "l1", "l2", "material", "stiffness", "binary",
    ]
    arm_labels = ["none", "infl.", "infl.+L1", "infl.+L2", "L1", "L2", "mat.", "stiff.", "binary"]
    reg_success = []
    reg_gap = []
    reg_gap_err = []
    for arm in arm_order:
        rows = regs[regs["arm"] == arm]
        reg_success.append(100.0 * float(np.mean(rows["success"])))
        reg_gap.append(float(np.mean(rows["gap_ratio"])))
        reg_gap_err.append(float(np.std(rows["gap_ratio"])))
    x = np.arange(len(arm_order))
    ax3.bar(x, reg_success, color="#2b6cb0", width=0.72)
    ax3.set_ylabel("success (%)")
    ax3.set_ylim(0, 105)
    ax3.set_xticks(x)
    ax3.set_xticklabels(arm_labels, rotation=45, ha="right")
    ax3.set_title("legacy local-regularizer screen")
    _panel_label(ax3, "d")
    ax3b = ax3.twinx()
    ax3b.errorbar(x, reg_gap, yerr=reg_gap_err, fmt="o", color="#a23e48", ms=3, capsize=2, lw=1.0)
    ax3b.set_ylabel("gap ratio", labelpad=2)
    ax3b.set_ylim(0.45, 0.78)

    fig.savefig(FIG_DIR / "fig4_scaling_regularizers.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_scaling_regularizers.png", bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax, label):
    ax.text(
        -0.14, 1.08, label, transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="top", ha="left",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument(
        "--only",
        choices=[
            "all", "nontrivial", "controls", "targets", "central",
            "finite", "regularizers", "mechanism",
        ],
        default="all",
        help="Run or replot one readiness block.",
    )
    args = ap.parse_args()
    _ensure_dirs()
    force = bool(args.force) and not bool(args.skip_experiments)
    summary = {}
    if args.only in ("all", "nontrivial"):
        print("Running/loading reciprocal nontrivial gap analysis...", flush=True)
        nontrivial = run_complex_nontrivial_gap(force=force)

        metrics = nontrivial["metrics"]
        summary["nontrivial"] = {
            "min_gap_over_spacing": float(np.min(metrics["gap_over_spacing"])),
            "max_gap_over_spacing": float(np.max(metrics["gap_over_spacing"])),
            "min_final_norm_gap_over_spacing": float(
                np.nanmin(metrics["final_norm_gap_over_spacing"])
            ),
            "sizes": [int(x) for x in metrics["size"]],
            "final_n_in": [int(x) for x in metrics["n_in_final"]],
            "uniform_same_stiffness_n_in": [
                int(x) for x in metrics["uniform_n_same_mean_stiffness"]
            ],
            "initial_modes_in_final_normalized_gap": [
                int(x) for x in metrics["n_initial_in_final_norm_gap"]
            ],
            "mean_material_ratio": float(np.mean(metrics["material_ratio"])),
            "max_frac_at_bounds": float(np.max(metrics["frac_at_bounds"])),
        }

    if args.only in ("all", "controls"):
        print("Running/loading reciprocal expanded controls...", flush=True)
        controls = run_expanded_controls(force=force, damping=REAL_SHIFT)
        records = controls["records"]
        summary["controls"] = {
            str(mode): {
                "success_rate": float(np.mean(records[records["mode"] == mode]["n_in_final"] == 0)),
                "mean_final_n_in": float(np.mean(records[records["mode"] == mode]["n_in_final"])),
                "mean_success_gap_ratio": _mean_success_gap(
                    records[records["mode"] == mode]
                ),
                "mean_material_ratio": float(np.mean(records[records["mode"] == mode]["material_ratio"])),
                "mean_frac_at_bounds": float(np.mean(records[records["mode"] == mode]["frac_at_bounds"])),
            }
            for mode in np.unique(records["mode"])
        }

    if args.only in ("all", "mechanism"):
        print("Running/loading forward-only mechanism diagnostic...", flush=True)
        mechanism = run_forward_inflation_diagnostic(
            force=force,
            damping=REAL_SHIFT,
        )
        records = mechanism["records"]
        summary["forward_inflation_diagnostic"] = {
            "mean_forward_stiffening_fraction": float(
                np.mean(records["forward_stiffening_fraction"])
            ),
            "mean_total_stiffening_fraction": float(
                np.mean(records["total_stiffening_fraction"])
            ),
            "mean_forward_to_inflation_ratio": float(
                np.mean(records["forward_to_inflation_ratio"])
            ),
        }

    target_data = None
    central_data = None
    if args.only in ("all", "targets"):
        print("Running/loading reciprocal spectral-window robustness...", flush=True)
        target_data = run_target_robustness(force=force, damping=REAL_SHIFT)
        records = target_data["records"]
        summary["targets"] = {
            str(int(center)): {
                "success_rate": float(np.mean(records[records["center_percentile"] == center]["success"])),
                "mean_final_n_in": float(np.mean(records[records["center_percentile"] == center]["n_in_final"])),
                "mean_success_gap_ratio": _mean_success_gap(
                    records[records["center_percentile"] == center]
                ),
            }
            for center in np.unique(records["center_percentile"])
        }

    if args.only in ("all", "central"):
        print("Running/loading reciprocal centralized response reference...", flush=True)
        central_data = run_centralized_reference(force=force, damping=REAL_SHIFT)
        records = central_data["records"]
        summary["central"] = {
            str(mode): {
                "success_rate": float(np.mean(records[records["mode"] == mode]["n_in_final"] == 0)),
                "mean_final_n_in": float(np.mean(records[records["mode"] == mode]["n_in_final"])),
                "mean_success_gap_ratio": _mean_success_gap(
                    records[records["mode"] == mode]
                ),
            }
            for mode in np.unique(records["mode"])
        }

    finite_data = None
    regularizer_data = None
    if args.only in ("all", "finite"):
        print("Running/loading reciprocal finite-size scaling...", flush=True)
        finite_data = run_finite_size_scaling(force=force, damping=REAL_SHIFT)
        records = finite_data["records"]
        summary["finite_size"] = {
            str(int(size)): {
                "runs": int(len(records[records["size"] == size])),
                "success_rate": float(np.mean(records[records["size"] == size]["success"])),
                "mean_final_n_in": float(np.mean(records[records["size"] == size]["n_in_final"])),
                "mean_norm_gap_spacing": float(np.nanmean(records[records["size"] == size]["final_norm_gap_over_spacing"])),
                "mean_material_ratio": float(np.mean(records[records["size"] == size]["material_ratio"])),
                "mean_frac_at_bounds": float(np.mean(records[records["size"] == size]["frac_at_bounds"])),
            }
            for size in np.unique(records["size"])
        }

    if args.only in ("all", "regularizers"):
        print("Running/loading reciprocal regularizer robustness...", flush=True)
        regularizer_data = run_regularizer_robustness(force=force, damping=REAL_SHIFT)
        records = regularizer_data["records"]
        summary["regularizers"] = {
            str(arm): {
                "success_rate": float(np.mean(records[records["arm"] == arm]["success"])),
                "mean_final_n_in": float(np.mean(records[records["arm"] == arm]["n_in_final"])),
                "mean_success_gap_ratio": _mean_success_gap(
                    records[records["arm"] == arm]
                ),
                "mean_material_ratio": float(np.mean(records[records["arm"] == arm]["material_ratio"])),
                "mean_frac_at_bounds": float(np.mean(records[records["arm"] == arm]["frac_at_bounds"])),
            }
            for arm in np.unique(records["arm"])
        }

    # During an active-asset render, report only the requested block.  Full
    # experiment runs still merge blocks into the persistent readiness summary.
    if args.only != "all" and not args.skip_experiments:
        summary_path = DATA_DIR / "readiness_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                previous = json.load(f)
            merged = dict(previous.get("blocks", {}))
            merged.update(summary)
            summary = merged

    summary_path = DATA_DIR / "readiness_summary.json"
    generated_at = float(time.time())
    if args.skip_experiments and summary_path.exists():
        generated_at = float(json.loads(summary_path.read_text())["generated_at_unix"])
    summary = {
        "generated_at_unix": generated_at,
        "damping": REAL_SHIFT,
        "eps_reg": EPS_REG,
        "blocks": summary,
    }
    if not args.skip_experiments:
        _save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
