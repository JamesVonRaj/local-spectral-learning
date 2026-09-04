"""Run the supplemental one-field, two-time trace-learning experiment.

This implements the two-clock relaxational rule described in
``prl_rebuild_outline.tex``:

    J = E|x(t_s)|^2 - E|x(t_l)|^2
      = sum_n [exp(-2 t_s lambda_n) - exp(-2 t_l lambda_n)]

with local signal

    Delta r_e propto (4 r_e / L_e)
        [t_s <(Delta_e x(t_s))^2> - t_l <(Delta_e x(t_l))^2>].

The main bounds-only experiment applies the two-time local rule with independent
per-edge step and radius bounds, without a network-wide rescaling.  A separate,
explicitly labeled fixed-budget diagnostic optionally rescales total stiffness
after each update to remove the uniform-stiffening direction and make its
stiffness-matched control meaningful.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mechanical.learners import build_K
from mechanical.topology import make_network
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply

from publication.paths import FIGURE_DIR, dataset

DATA_DIR = dataset("prl_e1_two_time")
FIG_DIR = FIGURE_DIR

BLUE = "#2b6cb0"
GREEN = "#2f855a"
RED = "#c53030"
AMBER = "#b7791f"
PURPLE = "#6b46c1"
GRAY = "#9ca3af"
DARK = "#111111"
BAND = "#e76f51"


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
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _panel_label(ax, label):
    ax.text(-0.13, 1.08, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="top")


def spectrum(edges, radii, lengths, N):
    K = build_K(edges, radii, lengths, N).toarray()
    mu = eigh(K, eigvals_only=True)
    return np.maximum(mu, 0.0)


def positive_rates(mu):
    mu = np.asarray(mu, dtype=np.float64)
    tol = 1e-8 * max(float(mu.max()), 1.0)
    return np.sort(mu[mu > tol])


def target_times(mu0, target_quantile, rho):
    rates = positive_rates(mu0)
    lam_star = float(np.quantile(rates, float(target_quantile)))
    ts = float(np.log(float(rho)) / (2.0 * lam_star * (float(rho) - 1.0)))
    tl = float(rho) * ts
    return lam_star, ts, tl


def target_window(lam_star, width_log):
    return (
        float(lam_star * np.exp(-float(width_log))),
        float(lam_star * np.exp(float(width_log))),
    )


def phi_values(lam, ts, tl):
    lam = np.asarray(lam, dtype=np.float64)
    return np.exp(-2.0 * float(ts) * lam) - np.exp(-2.0 * float(tl) * lam)


def window_count_and_ratio(rates, lam_star, wlo, whi):
    rates = positive_rates(rates)
    n = int(np.sum((rates >= wlo) & (rates <= whi)))
    below = rates[rates < lam_star]
    above = rates[rates > lam_star]
    if len(below) and len(above):
        ratio = float(above[0] / below[-1])
        gap_lo = float(below[-1])
        gap_hi = float(above[0])
    elif len(above):
        ratio = np.inf
        gap_lo = 0.0
        gap_hi = float(above[0])
    else:
        ratio = np.nan
        gap_lo = np.nan
        gap_hi = np.nan
    return n, ratio, gap_lo, gap_hi


def residual_power(mu, times):
    rates = positive_rates(mu)
    t = np.asarray(times, dtype=np.float64)
    return np.sum(np.exp(-2.0 * t[:, None] * rates[None, :]), axis=1)


def run_one(seed, *, size=20, train_seed_offset=10000, n_steps=800, batch=12,
            alpha=0.5, grad_clip=0.05, target_quantile=0.15, rho=3.0,
            width_log=0.08, radius_floor=1e-6, constraint="budget",
            r_min=0.5, r_max=2.0, bounds_step_normalization="mean",
            force=False):
    _ensure_dirs()
    constraint = str(constraint)
    if constraint not in ("budget", "bounds", "free"):
        raise ValueError("constraint must be 'budget', 'bounds', or 'free'")
    bounds_step_normalization = str(bounds_step_normalization)
    if bounds_step_normalization not in ("mean", "none"):
        raise ValueError("bounds_step_normalization must be 'mean' or 'none'")
    if constraint == "budget":
        path = DATA_DIR / f"e1_size{size}_seed{seed:03d}.npz"
    elif constraint == "bounds":
        suffix = "bounds" if bounds_step_normalization == "mean" else "boundslocal"
        path = DATA_DIR / f"e1_{suffix}_size{size}_seed{seed:03d}.npz"
    else:
        path = DATA_DIR / f"e1_free_size{size}_seed{seed:03d}.npz"
    if path.exists() and not force:
        return load_npz(path)

    pos, edges, lengths, box = make_network("rand-del", int(size), int(seed))
    N = len(pos)
    radii = np.ones(len(edges), dtype=np.float64)
    target_total_stiffness = float(np.sum(radii * radii / lengths))
    rng = np.random.RandomState(int(seed) + int(train_seed_offset))

    mu0 = spectrum(edges, radii, lengths, N)
    lam_star, ts, tl = target_times(mu0, target_quantile, rho)
    wlo, whi = target_window(lam_star, width_log)
    i, j = edges[:, 0], edges[:, 1]

    n_hist = np.zeros(int(n_steps) + 1, dtype=np.int64)
    ratio_hist = np.zeros(int(n_steps) + 1, dtype=np.float64)
    n_hist[0], ratio_hist[0], _, _ = window_count_and_ratio(mu0, lam_star, wlo, whi)

    for step in range(int(n_steps)):
        K = build_K(edges, radii, lengths, N).tocsc()
        X0 = rng.randn(N, int(batch))
        X0 -= X0.mean(axis=0, keepdims=True)
        Xt = expm_multiply(-K, X0, start=ts, stop=tl, num=2, endpoint=True)
        xs, xl = Xt[0], Xt[1]
        var_s = np.mean((xs[i] - xs[j]) ** 2, axis=1)
        var_l = np.mean((xl[i] - xl[j]) ** 2, axis=1)

        descent_direction = (4.0 * radii / lengths) * (ts * var_s - tl * var_l)
        if constraint == "budget":
            gmax = float(np.abs(descent_direction).max())
            if gmax > float(grad_clip):
                descent_direction *= float(grad_clip) / gmax
            radii = np.maximum(
                float(radius_floor), radii + float(alpha) * descent_direction
            )
            # Remove the uniform-stiffening direction for the fixed-budget
            # diagnostic.
            radii *= np.sqrt(
                target_total_stiffness / float(np.sum(radii * radii / lengths))
            )
        elif constraint == "bounds":
            if bounds_step_normalization == "mean":
                scale = float(np.mean(np.abs(descent_direction))) + 1e-16
            else:
                scale = 1.0
            radius_step = float(alpha) * descent_direction / scale
            np.clip(
                radius_step, -float(grad_clip), float(grad_clip), out=radius_step
            )
            radii = np.clip(radii + radius_step, float(r_min), float(r_max))
        else:
            radius_step = float(alpha) * descent_direction
            np.clip(
                radius_step, -float(grad_clip), float(grad_clip), out=radius_step
            )
            radii = np.maximum(float(radius_floor), radii + radius_step)

        if (step + 1) % 25 == 0 or step + 1 == int(n_steps):
            mu_step = spectrum(edges, radii, lengths, N)
            n_hist[step + 1], ratio_hist[step + 1], _, _ = (
                window_count_and_ratio(mu_step, lam_star, wlo, whi)
            )
        else:
            n_hist[step + 1] = n_hist[step]
            ratio_hist[step + 1] = ratio_hist[step]

    muf = spectrum(edges, radii, lengths, N)
    # Stiffness-matched uniform control.  Because the run is stiffness
    # conserved, this is the initial spectrum; keep the scale explicit.
    control_scale = float(np.sum(radii * radii / lengths) / target_total_stiffness)
    mu_control = mu0 * control_scale

    n0, ratio0, gap0_lo, gap0_hi = window_count_and_ratio(mu0, lam_star, wlo, whi)
    nf, ratiof, gapf_lo, gapf_hi = window_count_and_ratio(muf, lam_star, wlo, whi)
    nc, ratioc, gapc_lo, gapc_hi = window_count_and_ratio(mu_control, lam_star, wlo, whi)

    times = np.geomspace(max(1e-4, 0.02 / lam_star), 50.0 / lam_star, 280)
    payload = {
        "pos": pos.astype(np.float64),
        "edges": edges.astype(np.int64),
        "lengths": lengths.astype(np.float64),
        "box": np.asarray(box, dtype=np.float64),
        "radii": radii.astype(np.float64),
        "mu_initial": mu0.astype(np.float64),
        "mu_final": muf.astype(np.float64),
        "mu_control": mu_control.astype(np.float64),
        "lambda_star": np.float64(lam_star),
        "t_s": np.float64(ts),
        "t_l": np.float64(tl),
        "rho": np.float64(rho),
        "window_lo": np.float64(wlo),
        "window_hi": np.float64(whi),
        "n_initial": np.int64(n0),
        "n_final": np.int64(nf),
        "n_control": np.int64(nc),
        "ratio_initial": np.float64(ratio0),
        "ratio_final": np.float64(ratiof),
        "ratio_control": np.float64(ratioc),
        "gap_final_lo": np.float64(gapf_lo),
        "gap_final_hi": np.float64(gapf_hi),
        "control_scale": np.float64(control_scale),
        "n_hist": n_hist,
        "ratio_hist": ratio_hist,
        "curve_times": times.astype(np.float64),
        "power_initial": residual_power(mu0, times).astype(np.float64),
        "power_final": residual_power(muf, times).astype(np.float64),
        "power_control": residual_power(mu_control, times).astype(np.float64),
        "seed": np.int64(seed),
        "size": np.int64(size),
        "n_steps": np.int64(n_steps),
        "batch": np.int64(batch),
        "alpha": np.float64(alpha),
        "grad_clip": np.float64(grad_clip),
        "target_quantile": np.float64(target_quantile),
        "width_log": np.float64(width_log),
        "constraint": np.asarray(constraint),
        "r_min": np.float64(r_min),
        "r_max": np.float64(r_max),
        "bounds_step_normalization": np.asarray(bounds_step_normalization),
    }
    np.savez_compressed(path, **payload)
    return load_npz(path)


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def dataset_constraint(d):
    return str(np.asarray(d.get("constraint", "budget")).item())


def dataset_variant(d):
    constraint = dataset_constraint(d)
    if constraint == "budget":
        return "budget"
    if constraint == "free":
        return "free"
    norm = str(np.asarray(d.get("bounds_step_normalization", "mean")).item())
    return "bounds" if norm == "mean" else "boundslocal"


def run_ensemble(n_seeds=5, force=False, **kwargs):
    rows = []
    datasets = []
    constraint = str(kwargs.get("constraint", "budget"))
    bounds_step_normalization = str(kwargs.get("bounds_step_normalization", "mean"))
    for seed in range(int(n_seeds)):
        print(f"E1 seed {seed}/{int(n_seeds)-1}", flush=True)
        d = run_one(seed, force=force, **kwargs)
        r_min = float(np.asarray(d.get("r_min", 0.5)).item())
        r_max = float(np.asarray(d.get("r_max", 2.0)).item())
        radii = np.asarray(d["radii"], dtype=np.float64)
        datasets.append(d)
        rows.append({
            "seed": int(seed),
            "n_initial": int(d["n_initial"]),
            "n_final": int(d["n_final"]),
            "n_control": int(d["n_control"]),
            "ratio_final": float(d["ratio_final"]),
            "ratio_control": float(d["ratio_control"]),
            "radius_min": float(np.min(radii)),
            "radius_max": float(np.max(radii)),
            "mean_stiffness_ratio": float(d["control_scale"]),
            "frac_at_bounds": float(
                np.mean((radii <= r_min + 1e-9) | (radii >= r_max - 1e-9))
            ) if constraint == "bounds" else 0.0,
        })
    summary = summarize(rows)
    summary["rows"] = rows
    if constraint == "budget":
        summary_name = "e1_summary.json"
    elif constraint == "free":
        summary_name = "e1_free_summary.json"
    elif bounds_step_normalization == "mean":
        summary_name = "e1_bounds_summary.json"
    else:
        summary_name = "e1_boundslocal_summary.json"
    (DATA_DIR / summary_name).write_text(json.dumps(summary, indent=2, sort_keys=True))
    return datasets, summary


def summarize(rows):
    def finite_stats(key):
        x = np.array([r[key] for r in rows], dtype=float)
        xf = x[np.isfinite(x)]
        n_posinf = int(np.sum(np.isposinf(x)))
        n_neginf = int(np.sum(np.isneginf(x)))
        if len(xf) == 0:
            if n_posinf and not n_neginf:
                return {
                    "median": float("inf"),
                    "mean": float("inf"),
                    "min": float("inf"),
                    "max": float("inf"),
                    "n_finite": 0,
                    "n_posinf": n_posinf,
                    "n_neginf": n_neginf,
                }
            return {
                "median": float("nan"),
                "mean": float("nan"),
                "n_finite": 0,
                "n_posinf": n_posinf,
                "n_neginf": n_neginf,
            }
        return {
            "median": float(np.median(xf)),
            "mean": float(np.mean(xf)),
            "min": float(np.min(xf)),
            "max": float(np.max(xf)),
            "n_finite": int(len(xf)),
            "n_posinf": n_posinf,
            "n_neginf": n_neginf,
        }
    return {
        "n_seeds": len(rows),
        "initial_window_count_median": float(np.median([r["n_initial"] for r in rows])),
        "final_window_count_median": float(np.median([r["n_final"] for r in rows])),
        "control_window_count_median": float(np.median([r["n_control"] for r in rows])),
        "ratio_final": finite_stats("ratio_final"),
        "ratio_control": finite_stats("ratio_control"),
        "success_count": int(sum(r["n_final"] < r["n_initial"] for r in rows)),
    }


def choose_representative(datasets):
    finite = [d for d in datasets if np.isfinite(float(d["ratio_final"]))]
    if not finite:
        return datasets[0]
    # Prefer a cleared/depleted run with finite bracketing ratio.
    finite.sort(key=lambda d: (int(d["n_final"]), -float(d["ratio_final"])))
    return finite[0]


def make_figure(datasets, summary):
    _style()
    d = choose_representative(datasets)
    variant = dataset_variant(d)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    draw_protocol(ax_a, d)
    draw_potential(ax_b, d)
    draw_spectra(ax_c, d, summary)
    draw_relaxation(ax_d, d)
    title_suffix = {
        "budget": "fixed budget",
        "bounds": "bounds only",
        "boundslocal": "local bounds only",
        "free": "no radius bounds",
    }[variant]
    fig.suptitle(f"E1: one field, two times ({title_suffix})", fontsize=11)
    suffix = {
        "budget": "",
        "bounds": "_bounds",
        "boundslocal": "_bounds_local",
        "free": "_free",
    }[variant]
    out_pdf = FIG_DIR / f"fig_E1_two_time{suffix}.pdf"
    out_png = FIG_DIR / f"fig_E1_two_time{suffix}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def draw_protocol(ax, d):
    _panel_label(ax, "a")
    ts, tl = float(d["t_s"]), float(d["t_l"])
    t = np.linspace(0, 1.15 * tl, 250)
    trace = np.exp(-t / (0.36 * tl)) * (0.75 + 0.18 * np.cos(8 * np.pi * t / tl))
    ax.plot(t, trace, color=BLUE, lw=1.8)
    ax.axvline(ts, color=RED, lw=1.0)
    ax.axvline(tl, color=RED, lw=1.0)
    ax.scatter([ts, tl], np.interp([ts, tl], t, trace), color=RED, s=25, zorder=3)
    ax.text(ts, 1.03 * trace.max(), r"$t_s$", ha="center", va="bottom", color=RED)
    ax.text(tl, 1.03 * trace.max(), r"$t_\ell$", ha="center", va="bottom", color=RED)
    ax.set_ylim(-0.38, 1.10 * trace.max())
    ax.set_yticks([0.0, 0.4, 0.8])
    ax.axhline(0.0, color="#dddddd", lw=0.7, zorder=0)
    ax.set_xlabel("time")
    ax.set_ylabel(r"one edge strain $\Delta_e x(t)$")
    ax.set_title("free relaxation, two local reads")
    ax.text(
        0.03, 0.04,
        r"$\Delta r_e \propto \frac{4r_e}{L_e}$" "\n"
        r"$\times\,[t_s\langle(\Delta_e x_s)^2\rangle"
        r"-t_\ell\langle(\Delta_e x_\ell)^2\rangle]$" "\n"
        "one free relaxation  |  two clock times",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd"),
    )


def draw_potential(ax, d):
    _panel_label(ax, "b")
    lam_star, ts, tl = float(d["lambda_star"]), float(d["t_s"]), float(d["t_l"])
    wlo, whi = float(d["window_lo"]), float(d["window_hi"])
    lam = np.geomspace(max(lam_star / 20, 1e-4), lam_star * 20, 600)
    phi = phi_values(lam, ts, tl)
    ax.plot(lam, phi / phi.max(), color=PURPLE, lw=2.0)
    ax.axvspan(wlo, whi, color=BAND, alpha=0.18, lw=0)
    ax.axvline(lam_star, color=RED, lw=1.0, ls="--")
    ax.text(lam_star, 1.02, r"$\lambda^\ast$", ha="center", va="bottom", color=RED)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r"relaxation rate $\lambda$")
    ax.set_ylabel(r"$\varphi(\lambda)/\max\varphi$")
    ax.set_title(r"$e^{-2t_s\lambda}-e^{-2t_\ell\lambda}$")


def draw_spectra(ax, d, summary):
    _panel_label(ax, "c")
    mu0 = positive_rates(d["mu_initial"])
    muf = positive_rates(d["mu_final"])
    muc = positive_rates(d["mu_control"])
    wlo, whi, lam_star = float(d["window_lo"]), float(d["window_hi"]), float(d["lambda_star"])
    lo = max(min(mu0.min(), muf.min(), muc.min()) * 0.8, 1e-6)
    hi = max(mu0.max(), muf.max(), muc.max()) * 1.1
    bins = np.geomspace(lo, hi, 80)
    ax.axvspan(wlo, whi, color=BAND, alpha=0.24, lw=0)
    ax.hist(mu0, bins=bins, histtype="stepfilled", color=GRAY, alpha=0.35, label="initial")
    ax.hist(muc, bins=bins, histtype="step", color=DARK, lw=1.0, ls="--",
            label="uniform control")
    ax.hist(muf, bins=bins, histtype="stepfilled", color=BLUE, alpha=0.65, label="learned")
    ax.axvline(lam_star, color=RED, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"rate $\lambda_n$")
    ax.set_ylabel("mode count")
    ratio = float(d["ratio_final"])
    ratio_text = (rf"$\lambda_+/\lambda_-={ratio:.2g}$"
                  if np.isfinite(ratio) else r"$\lambda_+/\lambda_-=\infty$")
    ax.set_title(
        rf"window count {int(d['n_initial'])}$\to${int(d['n_final'])}; "
        + ratio_text
    )
    ax.legend(frameon=False, loc="upper right")


def draw_relaxation(ax, d):
    _panel_label(ax, "d")
    t = np.asarray(d["curve_times"], dtype=np.float64)
    p0 = np.asarray(d["power_initial"], dtype=np.float64)
    pf = np.asarray(d["power_final"], dtype=np.float64)
    pc = np.asarray(d["power_control"], dtype=np.float64)
    ts, tl = float(d["t_s"]), float(d["t_l"])
    ax.loglog(t, p0 / p0[0], color=GRAY, lw=1.5, label="initial")
    ax.loglog(t, pc / pc[0], color=DARK, lw=1.0, ls="--", label="uniform control")
    ax.loglog(t, pf / pf[0], color=BLUE, lw=2.0, label="learned")
    ax.axvline(ts, color=RED, lw=0.9, alpha=0.75)
    ax.axvline(tl, color=RED, lw=0.9, alpha=0.75)
    ax.text(ts, ax.get_ylim()[1] / 1.5, r"$t_s$", color=RED, ha="center", va="top")
    ax.text(tl, ax.get_ylim()[1] / 1.5, r"$t_\ell$", color=RED, ha="center", va="top")
    gap_lo = float(d["gap_final_lo"])
    gap_hi = float(d["gap_final_hi"])
    if np.isfinite(gap_lo) and gap_lo > 0 and np.isfinite(gap_hi):
        width = float(np.log(gap_hi / gap_lo))
        ax.text(0.04, 0.06, rf"$\Delta\ln t\simeq\ln(\lambda_+/\lambda_-)={width:.2g}$",
                transform=ax.transAxes, ha="left", va="bottom")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\mathbb{E}|x(t)|^2$ (normalized)")
    ax.set_title("residual power and relaxation plateau")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.18, which="both", linewidth=0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=0.05)
    p.add_argument("--target-quantile", type=float, default=0.15)
    p.add_argument("--rho", type=float, default=3.0)
    p.add_argument("--width-log", type=float, default=0.08)
    p.add_argument("--constraint", choices=("budget", "bounds", "free"),
                   default="budget")
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=2.0)
    p.add_argument("--bounds-step-normalization", choices=("mean", "none"),
                   default="mean")
    args = p.parse_args()
    datasets, summary = run_ensemble(
        n_seeds=args.n_seeds,
        force=args.force,
        size=args.size,
        n_steps=args.steps,
        batch=args.batch,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        target_quantile=args.target_quantile,
        rho=args.rho,
        width_log=args.width_log,
        constraint=args.constraint,
        r_min=args.r_min,
        r_max=args.r_max,
        bounds_step_normalization=args.bounds_step_normalization,
    )
    fig_path = make_figure(datasets, summary)
    print(f"wrote {fig_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
