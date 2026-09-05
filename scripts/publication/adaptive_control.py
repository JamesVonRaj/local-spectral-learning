"""Damping, calibration-disorder, absolute-window, and retargeting experiments.

All scalar absolute-window and retargeting runs disable spectral diagnostics
inside :func:`mechanical.learners_local.train_local`.  Eigenspectra in the
saved outputs are computed only afterward from stored material snapshots.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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
from mechanical.learners import eigenfreqs
from mechanical.learners_local import train_local
from mechanical.topology import make_network

from publication import style as ps
from publication.bloch_gap import (
    band_frequencies,
    k_grid,
    make_periodic_vector_cell,
    train_periodic_vector,
    window_metrics,
)
from publication.paths import FIGURE_DIR, dataset
from publication.style import BLUE, DARK, GRAY, GRAY_DARK, PURPLE, panel_label

DATA_DIR = dataset("prl_v7_adaptive")
DAMPING_RATIOS = (0.0, 0.001, 0.01, 0.05, 0.1)
ABSOLUTE_WINDOWS = ((2.10, 2.45), (2.45, 2.80), (2.80, 3.15))
RETARGET_A = (2.10, 2.45)
RETARGET_B = (2.80, 3.15)
RETARGET_SCHEDULE = (
    (0, *RETARGET_A),
    (1500, *RETARGET_B),
    (2050, *RETARGET_A),
)
RETARGET_END = 2350
CALIBRATION_RANGE = (0.5, 2.0)
VECTOR_CONTROLS = dataset("prl_vector_periodic") / "vector_controls.json"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def _scalar_metrics(freqs: np.ndarray, window) -> dict:
    wlo, whi = (float(value) for value in window)
    freqs = np.sort(np.asarray(freqs, dtype=np.float64))
    n_in = int(np.sum((freqs > wlo) & (freqs < whi)))
    below = freqs[(freqs > 1.0e-10) & (freqs <= wlo)]
    above = freqs[freqs >= whi]
    gap_ratio = 0.0
    gap_lo = gap_hi = 0.0
    if n_in == 0 and len(below) and len(above):
        gap_lo = float(below[-1])
        gap_hi = float(above[0])
        gap_ratio = (gap_hi - gap_lo) / (0.5 * (gap_hi + gap_lo))
    return {
        "n_in": n_in,
        "success": n_in == 0,
        "gap_lo": gap_lo,
        "gap_hi": gap_hi,
        "gap_ratio": float(gap_ratio),
    }


def _scalar_train(size, net_seed, train_seed, window, damping_ratio, *,
                  n_steps=3000, snapshot_every=0):
    pos, edges, lengths, box = make_network("rand-del", size=size, seed=net_seed)
    midpoint = 0.5 * (float(window[0]) + float(window[1]))
    result = train_local(
        edges, lengths, len(pos),
        n_steps=n_steps,
        batch=10,
        n_freq=8,
        material_update="response_conditioned_log",
        regularizers=[],
        target_window=window,
        damping="real_shift" if float(damping_ratio) == 0.0 else "viscous",
        damping_gamma=float(damping_ratio) * midpoint,
        spectrum_diagnostics=False,
        snapshot_every=snapshot_every,
        eval_every=max(n_steps, 1),
        train_seed=train_seed,
        frequency_seed=train_seed + 918273,
    )
    return pos, edges, lengths, box, result


def _run_scalar_damping_case(case):
    net_seed, ratio = case
    pos, edges, lengths, _ = make_network("rand-del", size=8, seed=net_seed)
    # Percentiles are used only to match initial task difficulty across the
    # damping ensemble.  Restart from the identical uniform material and pass
    # the resulting absolute interval directly to the eigensolve-free learner.
    initial = eigenfreqs(edges, np.ones(len(edges)), lengths, len(pos), 1.0)
    window = tuple(float(value) for value in np.percentile(initial, [35.0, 65.0]))
    _, _, _, _, result = _scalar_train(
        8, net_seed, net_seed, window, ratio, n_steps=3000,
    )
    final = eigenfreqs(edges, result["radii"], lengths, len(pos), 1.0)
    return {
        "net_seed": int(net_seed),
        "train_seed": int(net_seed),
        "damping_ratio": float(ratio),
        "damping_gamma": float(ratio) * float(np.mean(window)),
        "window": window,
        "initial": _scalar_metrics(initial, window),
        "final": _scalar_metrics(final, window),
        "online_eigensolves": 0,
        "cost_log_std_last_half": float(
            np.std(np.log10(np.maximum(result["cost_history"][1500:], 1.0e-300)))
        ),
    }


def run_scalar_damping(workers: int, force: bool = False) -> dict:
    output = DATA_DIR / "scalar_damping_scan.json"
    if output.exists() and not force:
        return json.loads(output.read_text())
    cases = [(seed, ratio) for seed in range(10) for ratio in DAMPING_RATIOS]
    with cf.ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        rows = list(executor.map(_run_scalar_damping_case, cases))
    summary = []
    for ratio in DAMPING_RATIOS:
        selected = [row for row in rows if row["damping_ratio"] == ratio]
        gap_ratios = np.array([row["final"]["gap_ratio"] for row in selected])
        summary.append({
            "damping_ratio": ratio,
            "n": len(selected),
            "success": int(sum(row["final"]["success"] for row in selected)),
            "success_rate": float(np.mean([row["final"]["success"] for row in selected])),
            "gap_ratio_median": float(np.median(gap_ratios)),
            "gap_ratio_q10": float(np.percentile(gap_ratios, 10)),
            "gap_ratio_q90": float(np.percentile(gap_ratios, 90)),
            "cost_log_std_median": float(np.median([
                row["cost_log_std_last_half"] for row in selected
            ])),
        })
    payload = {
        "description": "Matched scalar damping scan; percentile positioning is offline benchmark normalization.",
        "damping_model": "mass proportional: H=K-omega^2 M+i gamma omega M",
        "damping_ratios": DAMPING_RATIOS,
        "rows": rows,
        "summary": summary,
    }
    _save_json(output, payload)
    return payload


def _run_calibration_case(net_seed: int) -> dict:
    """Train with fixed unknown positive edgewise stiffness prefactors.

    Passing ``lengths / alpha_e`` to the physical response solver realizes
    ``k_e = alpha_e r_e**2 / L_e``.  The response-conditioned drive itself
    uses only measured edge responses; its multiplicative log-stiffness step
    and the actuator-radius bounds do not require ``alpha_e``.
    """
    pos, edges, lengths, _ = make_network("rand-del", size=8, seed=net_seed)
    calibration_rng = np.random.RandomState(700000 + int(net_seed))
    alpha_e = calibration_rng.uniform(
        CALIBRATION_RANGE[0], CALIBRATION_RANGE[1], size=len(edges),
    )
    effective_lengths = lengths / alpha_e
    initial = eigenfreqs(
        edges, np.ones(len(edges)), effective_lengths, len(pos), 1.0,
    )
    window = tuple(float(value) for value in np.percentile(initial, [35.0, 65.0]))
    midpoint = float(np.mean(window))
    result = train_local(
        edges, effective_lengths, len(pos),
        n_steps=3000,
        batch=10,
        n_freq=8,
        material_update="response_conditioned_log",
        regularizers=[],
        target_window=window,
        damping="viscous",
        damping_gamma=0.01 * midpoint,
        spectrum_diagnostics=False,
        eval_every=3000,
        train_seed=int(net_seed),
        frequency_seed=int(net_seed) + 918273,
    )
    final = eigenfreqs(
        edges, result["radii"], effective_lengths, len(pos), 1.0,
    )
    return {
        "net_seed": int(net_seed),
        "train_seed": int(net_seed),
        "calibration_seed": 700000 + int(net_seed),
        "alpha_min": float(np.min(alpha_e)),
        "alpha_max": float(np.max(alpha_e)),
        "window": window,
        "initial": _scalar_metrics(initial, window),
        "final": _scalar_metrics(final, window),
        "damping_ratio": 0.01,
        "online_eigensolves": 0,
        "calibration_supplied_to_controller": False,
    }


def run_calibration_disorder(workers: int, force: bool = False) -> dict:
    output = DATA_DIR / "stiffness_calibration_disorder.json"
    if output.exists() and not force:
        return json.loads(output.read_text())
    seeds = list(range(10))
    with cf.ProcessPoolExecutor(
        max_workers=max(1, min(int(workers), len(seeds)))
    ) as executor:
        rows = list(executor.map(_run_calibration_case, seeds))
    gap_ratios = np.asarray([row["final"]["gap_ratio"] for row in rows])
    payload = {
        "description": (
            "Fixed unknown stiffness calibration k_e=alpha_e r_e^2/L_e; "
            "alpha_e is used by the physical response solver but not the local controller."
        ),
        "alpha_distribution": "independent uniform",
        "alpha_range": CALIBRATION_RANGE,
        "n": len(rows),
        "success": int(sum(row["final"]["success"] for row in rows)),
        "gap_ratio_median": float(np.median(gap_ratios)),
        "gap_ratio_min": float(np.min(gap_ratios)),
        "gap_ratio_max": float(np.max(gap_ratios)),
        "rows": rows,
    }
    _save_json(output, payload)
    return payload


def _run_absolute_case(case):
    net_seed, window_index = case
    window = ABSOLUTE_WINDOWS[window_index]
    pos, edges, lengths, _, result = _scalar_train(
        8, net_seed, net_seed, window, 0.01, n_steps=3000,
    )
    initial = eigenfreqs(edges, np.ones(len(edges)), lengths, len(pos), 1.0)
    final = eigenfreqs(edges, result["radii"], lengths, len(pos), 1.0)
    return {
        "net_seed": int(net_seed),
        "train_seed": int(net_seed),
        "window_index": int(window_index),
        "window": window,
        "initial": _scalar_metrics(initial, window),
        "final": _scalar_metrics(final, window),
        "damping_ratio": 0.01,
        "online_eigensolves": 0,
    }


def run_absolute_windows(workers: int, force: bool = False) -> dict:
    output = DATA_DIR / "absolute_windows.json"
    if output.exists() and not force:
        return json.loads(output.read_text())
    cases = [(seed, index) for seed in range(10) for index in range(len(ABSOLUTE_WINDOWS))]
    with cf.ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        rows = list(executor.map(_run_absolute_case, cases))
    summary = []
    for index, window in enumerate(ABSOLUTE_WINDOWS):
        selected = [row for row in rows if row["window_index"] == index]
        summary.append({
            "window_index": index,
            "window": window,
            "n": len(selected),
            "success": int(sum(row["final"]["success"] for row in selected)),
            "success_rate": float(np.mean([row["final"]["success"] for row in selected])),
            "initial_count_mean": float(np.mean([row["initial"]["n_in"] for row in selected])),
            "final_count_mean": float(np.mean([row["final"]["n_in"] for row in selected])),
        })
    payload = {
        "description": "Shared absolute windows passed directly to the learner, with all eigensolves deferred until training finished.",
        "windows": ABSOLUTE_WINDOWS,
        "rows": rows,
        "summary": summary,
    }
    _save_json(output, payload)
    return payload


def run_retargeting(force: bool = False) -> dict:
    output = DATA_DIR / "retargeting.npz"
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    pos, edges, lengths, box = make_network("rand-del", size=10, seed=3)
    result = train_local(
        edges, lengths, len(pos),
        n_steps=RETARGET_END,
        batch=10,
        n_freq=8,
        material_update="response_conditioned_log",
        regularizers=[],
        target_window_schedule=RETARGET_SCHEDULE,
        damping="viscous",
        damping_gamma=0.01 * np.mean([*RETARGET_A, *RETARGET_B]),
        spectrum_diagnostics=False,
        snapshot_every=25,
        snapshot_dtype="float64",
        eval_every=RETARGET_END,
        train_seed=4,
        frequency_seed=918277,
    )
    spectra = np.vstack([
        eigenfreqs(edges, radii, lengths, len(pos), 1.0)
        for radii in result["radii_snapshots"]
    ])
    counts_a = np.sum(
        (spectra > RETARGET_A[0]) & (spectra < RETARGET_A[1]), axis=1,
    )
    counts_b = np.sum(
        (spectra > RETARGET_B[0]) & (spectra < RETARGET_B[1]), axis=1,
    )
    payload = {
        "pos": pos,
        "edges": edges,
        "lengths": lengths,
        "box": box,
        "radii": result["radii"],
        "radius_snapshot_steps": result["radius_snapshot_steps"],
        "radii_snapshots": result["radii_snapshots"],
        "spectra_snapshots": spectra,
        "counts_a": counts_a,
        "counts_b": counts_b,
        "window_a": np.asarray(RETARGET_A),
        "window_b": np.asarray(RETARGET_B),
        "window_schedule": np.asarray(RETARGET_SCHEDULE),
        "cost_history": result["cost_history"],
        "damping_ratio_nominal": np.float64(0.01),
        "damping_gamma": result["damping_gamma"],
        "online_eigensolves": np.int64(0),
        "schedule_protocol": np.asarray(
            "fixed before the displayed run from a separate offline pilot"
        ),
        "size": np.int64(10),
        "net_seed": np.int64(3),
        "train_seed": np.int64(4),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "schedule": RETARGET_SCHEDULE,
        "counts": {
            "initial": {"A": int(counts_a[0]), "B": int(counts_b[0])},
            "after_A": {
                "A": int(counts_a[np.searchsorted(result["radius_snapshot_steps"], 1500)]),
                "B": int(counts_b[np.searchsorted(result["radius_snapshot_steps"], 1500)]),
            },
            "after_B": {
                "A": int(counts_a[np.searchsorted(result["radius_snapshot_steps"], 2050)]),
                "B": int(counts_b[np.searchsorted(result["radius_snapshot_steps"], 2050)]),
            },
            "after_return_A": {"A": int(counts_a[-1]), "B": int(counts_b[-1])},
        },
        "online_eigensolves": 0,
        "schedule_protocol": (
            "phase lengths calibrated from the first empty 25-step snapshot "
            "in an offline pilot, then fixed before the displayed run"
        ),
    }
    _save_json(DATA_DIR / "retargeting_summary.json", summary)
    return payload


def run_vector_damping(force: bool = False) -> dict:
    output = DATA_DIR / "vector_damping_scan.json"
    if output.exists() and not force:
        return json.loads(output.read_text())
    cell = make_periodic_vector_cell(5, 0)
    initial = band_frequencies(cell, np.ones(cell.n_edges), k_grid(9))
    positive = initial[initial > 1.0e-8]
    window = tuple(float(value) for value in np.percentile(positive, [45.0, 55.0]))
    rows = []
    radii = []
    for ratio in DAMPING_RATIOS:
        result = train_periodic_vector(
            cell, *window,
            n_steps=3000,
            k_eval_grid=9,
            eval_every=3000,
            response_mode="reciprocal" if ratio == 0.0 else "viscous",
            damping_gamma_ratio=ratio,
            seed=0,
        )
        dense = band_frequencies(cell, result["radii"], k_grid(17))
        metrics = window_metrics(dense, *window)
        rows.append({
            "damping_ratio": ratio,
            "n_in": metrics["n_in"],
            "success": bool(metrics["n_in"] == 0),
            "gap_ratio": metrics["gap_ratio"],
            "gap_contains_window": bool(metrics["n_in"] == 0),
        })
        radii.append(result["radii"])
    payload = {
        "description": "Matched periodic-vector exemplar over mass-proportional damping.",
        "window": window,
        "size": 5,
        "net_seed": 0,
        "train_seed": 0,
        "rows": rows,
    }
    _save_json(output, payload)
    np.savez_compressed(
        DATA_DIR / "vector_damping_radii.npz",
        damping_ratios=np.asarray(DAMPING_RATIOS),
        radii=np.asarray(radii),
        window=np.asarray(window),
    )
    return payload


def _categorical_damping_axis(ax):
    positions = np.arange(len(DAMPING_RATIOS))
    ax.set_xticks(positions[[0, 2, 4]])
    ax.set_xticklabels(
        ["0", r"$10^{-2}$", "0.1"],
    )
    ax.set_xlabel(r"damping $\gamma/\omega_{\rm mid}^{\rm win}$")
    return positions


def render_main_figure() -> None:
    ps.style()
    retarget = run_retargeting(force=False)
    scalar = run_scalar_damping(workers=1, force=False)
    vector = run_vector_damping(force=False)
    if not VECTOR_CONTROLS.exists():
        raise FileNotFoundError(VECTOR_CONTROLS)
    controls = json.loads(VECTOR_CONTROLS.read_text())
    steps = np.asarray(retarget["radius_snapshot_steps"], dtype=float)
    spectra = np.asarray(retarget["spectra_snapshots"], dtype=float)

    fig = plt.figure(figsize=(ps.COL_W, 4.55), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :])
    ax_c = fig.add_subplot(grid[2, 0])
    ax_d = fig.add_subplot(grid[2, 1])

    for start, stop, window, color, hatch in (
        (0, 1500, RETARGET_A, BLUE, "////"),
        (1500, 2050, RETARGET_B, PURPLE, r"\\\\"),
        (2050, RETARGET_END, RETARGET_A, BLUE, "////"),
    ):
        ax_a.fill_between(
            [start, stop], window[0], window[1], facecolor=color,
            edgecolor=color, hatch=hatch, lw=0.45, alpha=0.16,
        )
    for mode in spectra.T:
        ax_a.plot(steps, mode, color=GRAY_DARK, lw=0.22, alpha=0.55)
    for boundary in (1500, 2050):
        ax_a.axvline(boundary, color=DARK, lw=0.6, ls="--")
    ax_a.set_xlim(0, RETARGET_END)
    ax_a.set_ylim(1.75, 3.55)
    ax_a.set_ylabel(r"eigenfrequency $\omega_n$")
    ax_a.set_xticklabels([])
    ax_a.text(750, 3.45, "probe A", color=BLUE, ha="center", fontsize=ps.BASE_SIZE)
    ax_a.text(1775, 3.45, "probe B", color=PURPLE, ha="center", fontsize=ps.BASE_SIZE)
    ax_a.text(2200, 3.45, "probe A", color=BLUE, ha="center", fontsize=ps.BASE_SIZE)
    panel_label(ax_a, "a", fontsize=10.0)

    ax_b.plot(
        steps, retarget["counts_a"], color=BLUE, lw=1.3, ls="-",
        marker="o", ms=3.0, markevery=10, label="window A",
    )
    ax_b.plot(
        steps, retarget["counts_b"], color=PURPLE, lw=1.3, ls="--",
        marker="s", ms=3.0, markevery=10, label="window B",
    )
    for boundary in (1500, 2050):
        ax_b.axvline(boundary, color=DARK, lw=0.6, ls="--")
    ax_b.set_xlim(0, RETARGET_END)
    ax_b.set_xticks([0, 750, 1500, 2050, RETARGET_END])
    ax_b.set_ylim(bottom=-0.7)
    ax_b.set_xlabel("adaptation step")
    ax_b.set_ylabel("window count")
    ax_b.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        fontsize=ps.BASE_SIZE,
        handlelength=1.25,
        handletextpad=0.45,
        labelspacing=0.2,
        borderaxespad=0.25,
    )
    panel_label(ax_b, "b", fontsize=10.0)

    scalar_summary = scalar["summary"]
    vector_rows = vector["rows"]
    control_order = (
        "paired-response", "forward-only", "shuffled-response",
        "random-matched", "uniform-material", "uniform-stiffness",
    )
    control_ticks = (
        "paired", "forward", "shuffled", "random", "uniform\n$r$",
        "uniform\n$k$",
    )
    control_summary = {
        str(row["control"]): row for row in controls["summary"]
    }
    rates = np.asarray([
        float(control_summary[label]["success_rate"]) for label in control_order
    ])
    sample_sizes = np.asarray([
        int(control_summary[label]["n"]) for label in control_order
    ])
    xpos = np.arange(len(control_order))
    colors = [BLUE] + [GRAY_DARK] * (len(control_order) - 1)
    ax_c.vlines(xpos, 0.0, rates, color=GRAY, lw=1.0, zorder=1)
    ax_c.scatter(xpos, rates, color=colors, s=24, zorder=2)
    ax_c.set_xlim(-0.55, len(control_order) - 0.45)
    ax_c.set_ylim(-0.08, 1.16)
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels(control_ticks, rotation=42, ha="right")
    ax_c.set_yticks([0.0, 0.5, 1.0])
    ax_c.set_ylabel("success fraction")
    paired_count = int(round(float(rates[0]) * int(sample_sizes[0])))
    ax_c.annotate(
        f"{paired_count}/{int(sample_sizes[0])}", (xpos[0], rates[0]),
        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
        fontsize=ps.TICK_SIZE,
    )
    ax_c.text(
        float(np.mean(xpos[1:])), 0.08, "each 0/8", ha="center", va="bottom",
        fontsize=ps.TICK_SIZE,
    )
    panel_label(ax_c, "c", fontsize=10.0)

    positions = _categorical_damping_axis(ax_d)
    medians = np.asarray([row["gap_ratio_median"] for row in scalar_summary])
    q10 = np.asarray([row["gap_ratio_q10"] for row in scalar_summary])
    q90 = np.asarray([row["gap_ratio_q90"] for row in scalar_summary])
    ax_d.fill_between(positions, q10, q90, color=BLUE, alpha=0.16, lw=0)
    ax_d.plot(positions, medians, color=BLUE, marker="o", ms=4.2,
              lw=1.2, ls="-")
    ax_d.plot(
        positions, [row["gap_ratio"] for row in vector_rows],
        color=PURPLE, marker="s", ms=4.2, lw=1.2, ls="--",
    )
    ax_d.set_ylim(bottom=0.0)
    ax_d.set_ylabel(r"relative gap $\Delta\omega/\omega_{\rm mid}^{\rm gap}$")
    panel_label(ax_d, "d", fontsize=10.0)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.xaxis.label.set_size(9.0)
        ax.yaxis.label.set_size(9.0)
        ax.tick_params(labelsize=8.5)

    ps.savefig(fig, FIGURE_DIR, "fig3_adaptive")
    plt.close(fig)


def render_absolute_figure() -> None:
    ps.style()
    data = run_absolute_windows(workers=1, force=False)
    summary = data["summary"]
    positions = np.arange(len(summary))
    widths = [f"{row['window'][0]:.2f}--{row['window'][1]:.2f}" for row in summary]
    initial = [row["initial_count_mean"] for row in summary]
    final = [row["final_count_mean"] for row in summary]
    success = [row["success_rate"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(ps.COL_W, 1.65), constrained_layout=True)
    width = 0.34
    axes[0].bar(positions - width / 2, initial, width, color=GRAY, label="initial")
    axes[0].bar(positions + width / 2, final, width, color=BLUE, label="learned")
    axes[0].set_ylabel("mean mode count")
    axes[0].legend(frameon=False)
    panel_label(axes[0], "a")
    axes[1].bar(positions, success, 0.55, color=BLUE)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("success fraction")
    panel_label(axes[1], "b")
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(widths, rotation=25, ha="right")
        ax.set_xlabel("absolute window")
    ps.savefig(fig, FIGURE_DIR, "figS_absolute_windows")
    plt.close(fig)


def write_tables() -> None:
    scalar = run_scalar_damping(workers=1, force=False)
    calibration = run_calibration_disorder(workers=1, force=False)
    absolute = run_absolute_windows(workers=1, force=False)
    vector = run_vector_damping(force=False)
    vector_by_ratio = {row["damping_ratio"]: row for row in vector["rows"]}
    lines = [
        r"\begin{tabular}{@{}rrrrr@{}}",
        r"\toprule",
        r"$\gamma/\omega_{\rm mid}^{\rm win}$ & scalar success & scalar $\Delta\omega/\omega_{\rm mid}^{\rm gap}$ & Bloch success & Bloch $\Delta\omega/\omega_{\rm mid}^{\rm gap}$ \\",
        r"\midrule",
    ]
    for row in scalar["summary"]:
        ratio = row["damping_ratio"]
        vrow = vector_by_ratio[ratio]
        lines.append(
            f"{ratio:g} & {row['success']}/{row['n']} & "
            f"{row['gap_ratio_median']:.3f} & "
            f"{int(vrow['success'])}/1 & {vrow['gap_ratio']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (FIGURE_DIR / "table_damping_scan.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{tabular}{@{}rrrr@{}}",
        r"\toprule",
        r"absolute window & initial count & final count & success \\",
        r"\midrule",
    ]
    for row in absolute["summary"]:
        window = row["window"]
        lines.append(
            f"{window[0]:.2f}--{window[1]:.2f} & "
            f"{row['initial_count_mean']:.1f} & {row['final_count_mean']:.1f} & "
            f"{row['success']}/{row['n']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (FIGURE_DIR / "table_absolute_windows.tex").write_text("\n".join(lines) + "\n")

    ideal = next(
        row for row in scalar["summary"]
        if np.isclose(float(row["damping_ratio"]), 0.01)
    )
    lines = [
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"calibration factors & success & median $\Delta\omega/\omega_{\rm mid}^{\rm gap}$ \\",
        r"\midrule",
        rf"$\alpha_e=1$ & {ideal['success']}/{ideal['n']} & "
        rf"{ideal['gap_ratio_median']:.3f} \\",
        rf"$\alpha_e\sim\operatorname{{Unif}}[0.5,2]$ & "
        rf"{calibration['success']}/{calibration['n']} & "
        rf"{calibration['gap_ratio_median']:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (FIGURE_DIR / "table_stiffness_calibration.tex").write_text(
        "\n".join(lines) + "\n"
    )


def render() -> None:
    render_main_figure()
    render_absolute_figure()
    write_tables()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("all", "experiments", "scalar-damping", "vector-damping", "absolute", "calibration", "retarget", "render"),
        default="all",
        nargs="?",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in {"all", "experiments", "scalar-damping"}:
        run_scalar_damping(args.workers, force=args.force)
    if args.mode in {"all", "experiments", "absolute"}:
        run_absolute_windows(args.workers, force=args.force)
    if args.mode in {"all", "experiments", "calibration"}:
        run_calibration_disorder(args.workers, force=args.force)
    if args.mode in {"all", "experiments", "retarget"}:
        run_retargeting(force=args.force)
    if args.mode in {"all", "experiments", "vector-damping"}:
        run_vector_damping(force=args.force)
    if args.mode in {"all", "render"}:
        render()


if __name__ == "__main__":
    main()
