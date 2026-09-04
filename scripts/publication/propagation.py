"""Source-agnostic propagation diagnostics for the learned Bloch band gap.

This script evaluates a learned periodic vector cell after training.  A point
force localized to one repeated cell has a k-independent Bloch transform, so
the real-space response is the inverse Bloch transform of

    G(k, omega) = [K(k) - omega**2 I + i gamma omega I]**-1.

The viscous term is an offline diagnostic only; it is not present during
training.  The main result compares the identical localized drive before and
after learning, then removes source-selection ambiguity by evaluating all
coordinate point forces (every node, x and y polarizations).

The pipeline is resumable: every expensive target-frequency cell calculation
is cached separately.  ``smoke`` makes a quick draft, while ``full`` runs the
publication-grade exemplar, ten-cell ensemble, and convergence matrix.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.transforms import blended_transform_factory

from publication import style as ps
from publication.bloch_gap import (
    PeriodicVectorCell,
    band_frequencies,
    best_gap_near_window,
    bloch_stiffness,
    k_grid,
    load_npz_result,
    make_periodic_vector_cell,
    pack_npz,
    target_window_from_percentiles,
    train_periodic_vector,
)
from publication.paths import FIGURE_DIR, REPO_ROOT, dataset

VECTOR_DIR = dataset("prl_vector_periodic")
OUT_DIR = dataset("prl_propagation")
TRAINED_DIR = OUT_DIR / "trained_cells"
FIG_DIR = FIGURE_DIR
FILE_TEMPLATE = "vector_periodic_s5_net{seed}_train0_c50_w10.npz"
FIGURE_NAME = "fig3_propagation"
INITIAL_RADIUS = 1.0
DB_FLOOR = -140.0


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def gamma_tag(gamma_fraction: float) -> str:
    return f"{float(gamma_fraction):.5f}".replace(".", "p")


def cell_path(seed: int) -> Path:
    prepared = TRAINED_DIR / FILE_TEMPLATE.format(seed=int(seed))
    if prepared.exists():
        return prepared
    # The manuscript exemplar already has the current protocol metadata and
    # lets the smoke test run without retraining.  Ensemble members are always
    # prepared into TRAINED_DIR so legacy caches cannot silently enter a run.
    if int(seed) == 0:
        return VECTOR_DIR / FILE_TEMPLATE.format(seed=0)
    return prepared


def prepared_cell_path(seed: int) -> Path:
    return TRAINED_DIR / FILE_TEMPLATE.format(seed=int(seed))


def target_cache_path(seed: int, nk: int, gamma_fraction: float) -> Path:
    return OUT_DIR / (
        f"target_s5_net{int(seed)}_nk{int(nk)}_g{gamma_tag(gamma_fraction)}.npz"
    )


def spectrum_cache_path(seed: int, nk: int, n_freq: int,
                        gamma_fraction: float) -> Path:
    return OUT_DIR / (
        f"spectrum_s5_net{int(seed)}_nk{int(nk)}_nf{int(n_freq)}_"
        f"g{gamma_tag(gamma_fraction)}.npz"
    )


def convergence_cache_path(nk: int, gamma_fractions: list[float]) -> Path:
    joined = "-".join(gamma_tag(x) for x in gamma_fractions)
    return OUT_DIR / f"convergence_net0_nk{int(nk)}_g{joined}.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def prepared_cell_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        _, result = load_npz_result(path)
    except Exception:
        return False
    params = result["params"]
    return bool(
        params.get("response_mode") == "reciprocal"
        and params.get("frequency_sampling") == "random_grid"
        and int(params.get("frequencies_per_step", -1)) == 1
        and int(params.get("n_steps", -1)) == 3000
        and params.get("material_update") == "response_conditioned_log"
        and params.get("gradient_clip_mode") == "none"
        and params.get("grad_clip") is None
        and params.get("regularizer") == "none"
        and np.isclose(float(params.get("inflation_strength", np.nan)), 0.0)
        and np.isclose(float(params.get("response_metric_eta", np.nan)), 0.02)
        and np.isclose(
            float(params.get("response_metric_lambda_ratio", np.nan)), 0.025,
        )
        and params.get("response_bound_mode") == "clip"
        and np.isclose(float(params.get("eps_reg", np.nan)), 0.0)
    )


def train_current_cell(seed: int, force: bool = False) -> Path:
    """Train one independently generated size-5 material with main-text settings."""
    ensure_dirs()
    out = prepared_cell_path(seed)
    if prepared_cell_is_current(out) and not force:
        log(f"prepared-cell cache hit: {out.name}")
        return out
    cell = make_periodic_vector_cell(5, int(seed), "rand-del")
    initial_eval = band_frequencies(
        cell, np.full(cell.n_edges, INITIAL_RADIUS), k_grid(9),
    )
    wlo, whi = target_window_from_percentiles(initial_eval, 50.0, 10.0)
    log(f"training current material seed={seed}")
    result = train_periodic_vector(
        cell, wlo, whi,
        n_steps=3000,
        alpha=0.035,
        grad_clip=None,
        inflation_strength=0.0,
        n_freq=5,
        k_train_grid=5,
        k_eval_grid=9,
        k_batch=4,
        force_batch=2,
        seed=0,
        eval_every=100,
        update_rule="adjoint",
        regularizer="none",
        response_mode="reciprocal",
        material_update="response_conditioned_log",
        response_metric_eta=0.02,
        response_metric_lambda_ratio=0.025,
        response_bound_mode="clip",
        frequency_sampling="random_grid",
        frequencies_per_step=1,
    )
    pack_npz(cell, result, out)
    if not prepared_cell_is_current(out):
        raise RuntimeError(f"prepared cell failed metadata validation: {out}")
    log(f"wrote trained material {out} ({float(result['elapsed_s']):.1f} s)")
    return out


def _training_job(args: tuple[int, bool]) -> str:
    seed, force = args
    return str(train_current_cell(seed, force))


def prepare_materials(seeds: list[int], workers: int,
                      force: bool = False) -> list[Path]:
    jobs = [(int(seed), bool(force)) for seed in seeds]
    n_workers = max(1, min(int(workers), len(jobs)))
    log(f"prepare materials: {len(jobs)} cells with {n_workers} workers")
    if n_workers == 1:
        return [Path(_training_job(job)) for job in jobs]
    with cf.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn"),
    ) as pool:
        return [Path(path) for path in pool.map(_training_job, jobs, chunksize=1)]


def fft_k_values(nk: int) -> np.ndarray:
    """Reduced wavevectors in the ordering expected by numpy's FFT."""
    return 2.0 * np.pi * np.fft.fftfreq(int(nk))


def eigensystem(cell: PeriodicVectorCell, radii: np.ndarray,
                nk: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense Bloch eigensystem on an FFT-compatible square k grid."""
    d = cell.n_dof
    values = np.empty((nk, nk, d), dtype=np.float64)
    vectors = np.empty((nk, nk, d, d), dtype=np.complex128)
    kvals = fft_k_values(nk)
    t0 = time.perf_counter()
    for ix, kx in enumerate(kvals):
        for iy, ky in enumerate(kvals):
            vals, vecs = np.linalg.eigh(
                bloch_stiffness(cell, radii, np.array([kx, ky])),
            )
            values[ix, iy] = vals.real
            vectors[ix, iy] = vecs
    log(f"eigensystem nk={nk}, dof={d}: {time.perf_counter() - t0:.1f} s")
    return values, vectors


def spectral_gap(values: np.ndarray, wlo: float, whi: float,
                 nk: int) -> dict:
    kvals = fft_k_values(nk)
    kpts = np.array([(kx, ky) for kx in kvals for ky in kvals], dtype=float)
    freqs = np.sqrt(np.maximum(values.reshape(-1, values.shape[-1]), 0.0))
    return best_gap_near_window(freqs, wlo, whi, kpts)


def centered_coordinates(nk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.arange(nk, dtype=int) - nk // 2
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx.astype(float) ** 2 + yy.astype(float) ** 2)
    return xx, yy, radius


def ring_masks(nk: int, max_radius: int) -> list[np.ndarray]:
    _, _, radius = centered_coordinates(nk)
    masks = []
    for r in range(max_radius + 1):
        mask = radius < 0.5 if r == 0 else (radius >= r - 0.5) & (radius < r + 0.5)
        if not np.any(mask):
            raise RuntimeError(f"empty radial shell r={r} on nk={nk}")
        masks.append(mask)
    return masks


def green_real_space(values: np.ndarray, vectors: np.ndarray,
                     omega: float, gamma: float) -> np.ndarray:
    """Return G_R with R=0 shifted to the center of the first two axes."""
    inverse = 1.0 / (values - omega**2 + 1j * gamma * omega)
    green_k = np.einsum(
        "...ia,...a,...ja->...ij",
        vectors,
        inverse,
        vectors.conjugate(),
        optimize=True,
    )
    green_r = np.fft.ifft2(green_k, axes=(0, 1))
    # FFT convention check: the R=0 block must equal the direct BZ average.
    scale = max(float(np.max(np.abs(green_k))), np.finfo(float).tiny)
    direct_error = float(np.max(np.abs(
        green_r[0, 0] - np.mean(green_k, axis=(0, 1))
    ))) / scale
    if direct_error > 5e-12:
        raise RuntimeError(f"inverse-Bloch R=0 check failed: {direct_error:.3e}")
    shifted = np.fft.fftshift(green_r, axes=(0, 1))
    # Reciprocal real-space Green blocks obey G_R = G_{-R}^T.  The comparison
    # below is exact for the odd grids used by the production calculations.
    if shifted.shape[0] % 2 == 1:
        reciprocal = np.swapaxes(shifted[::-1, ::-1], 2, 3)
        reciprocity_error = float(np.max(np.abs(shifted - reciprocal))) / scale
        if reciprocity_error > 2e-10:
            raise RuntimeError(
                f"real-space reciprocity check failed: {reciprocity_error:.3e}"
            )
    return shifted


def all_source_diagnostics(green_r: np.ndarray, max_radius: int,
                           source_dof: int) -> dict[str, np.ndarray]:
    """Radial transfer for every coordinate point force and one map source."""
    nk = green_r.shape[0]
    center = nk // 2
    # Sum receiver displacement power within each cell.  The last dimension
    # indexes every coordinate point force in the source cell.
    cell_power = np.sum(np.abs(green_r) ** 2, axis=2)
    local_amplitude = np.sqrt(np.maximum(cell_power[center, center], 0.0))
    if np.any(local_amplitude <= 0.0):
        raise RuntimeError("zero local response encountered")

    masks = ring_masks(nk, max_radius)
    radial = np.empty((max_radius + 1, green_r.shape[-1]), dtype=float)
    absolute = np.empty_like(radial)
    for r, mask in enumerate(masks):
        shell_amplitude = np.sqrt(np.mean(cell_power[mask], axis=0))
        absolute[r] = shell_amplitude
        radial[r] = shell_amplitude / local_amplitude

    map_amplitude = np.sqrt(np.maximum(cell_power[:, :, int(source_dof)], 0.0))
    map_relative = map_amplitude / local_amplitude[int(source_dof)]
    map_db = 20.0 * np.log10(np.maximum(map_relative, 10.0 ** (DB_FLOOR / 20.0)))
    return {
        "radial_transfer": radial,
        "radial_absolute": absolute,
        "local_amplitude": local_amplitude,
        "map_db": map_db,
    }


def fixed_source(cell: PeriodicVectorCell) -> tuple[int, int]:
    """Predeclared source: node nearest cell center, x polarization."""
    center = 0.5 * cell.box
    node = int(np.argmin(np.linalg.norm(cell.pos - center[None, :], axis=1)))
    return node, 2 * node


def compute_target_payload(seed: int, nk: int, gamma_fraction: float,
                           max_radius: int) -> dict[str, np.ndarray]:
    path = cell_path(seed)
    if not path.exists():
        raise FileNotFoundError(path)
    cell, result = load_npz_result(path)
    wlo = float(result["wlo"])
    whi = float(result["whi"])
    omega_target = 0.5 * (wlo + whi)
    gamma = float(gamma_fraction) * omega_target
    source_node, source_dof = fixed_source(cell)

    payload: dict[str, np.ndarray] = {
        "seed": np.int64(seed),
        "nk": np.int64(nk),
        "gamma_fraction": np.float64(gamma_fraction),
        "gamma": np.float64(gamma),
        "wlo": np.float64(wlo),
        "whi": np.float64(whi),
        "omega_target": np.float64(omega_target),
        "source_node": np.int64(source_node),
        "source_dof": np.int64(source_dof),
        "max_radius": np.int64(max_radius),
    }

    gap = None
    for label, radii in (
        ("initial", np.full(cell.n_edges, INITIAL_RADIUS, dtype=float)),
        ("learned", np.asarray(result["radii"], dtype=float)),
    ):
        log(f"target seed={seed} state={label} nk={nk} gamma/omega_t={gamma_fraction:g}")
        values, vectors = eigensystem(cell, radii, nk)
        if label == "learned":
            gap = spectral_gap(values, wlo, whi, nk)
        green_r = green_real_space(values, vectors, omega_target, gamma)
        diag = all_source_diagnostics(green_r, max_radius, source_dof)
        for key, value in diag.items():
            payload[f"{label}_{key}"] = value
        del green_r, values, vectors

    assert gap is not None
    payload.update({
        "gap_lo": np.float64(gap["lower_edge"]),
        "gap_hi": np.float64(gap["upper_edge"]),
        "normalized_gap": np.float64(gap["normalized_gap"]),
        "gap_lower_band": np.int64(gap["lower_band_1based"]),
        "gap_upper_band": np.int64(gap["upper_band_1based"]),
    })
    return payload


def run_target_cell(seed: int, nk: int, gamma_fraction: float,
                    max_radius: int, force: bool = False) -> Path:
    ensure_dirs()
    out = target_cache_path(seed, nk, gamma_fraction)
    if out.exists() and not force:
        log(f"cache hit: {out.name}")
        return out
    t0 = time.perf_counter()
    payload = compute_target_payload(seed, nk, gamma_fraction, max_radius)
    payload["elapsed_s"] = np.float64(time.perf_counter() - t0)
    np.savez_compressed(out, **payload)
    log(f"wrote {out} ({float(payload['elapsed_s']):.1f} s)")
    return out


def single_source_transfer(values: np.ndarray, vectors: np.ndarray,
                           omega: float, gamma: float, source_dof: int,
                           receiver_radius: int) -> tuple[float, float, float]:
    """Remote/local transfer for the fixed coordinate point force."""
    inverse = 1.0 / (values - omega**2 + 1j * gamma * omega)
    projection = vectors[..., int(source_dof), :].conjugate()
    response_k = np.einsum(
        "...ia,...a->...i", vectors, inverse * projection, optimize=True,
    )
    response_r = np.fft.fftshift(
        np.fft.ifft2(response_k, axes=(0, 1)), axes=(0, 1),
    )
    power = np.sum(np.abs(response_r) ** 2, axis=-1)
    center = values.shape[0] // 2
    local = float(np.sqrt(max(power[center, center], 0.0)))
    mask = ring_masks(values.shape[0], receiver_radius)[receiver_radius]
    remote = float(np.sqrt(np.mean(power[mask])))
    return remote / local, remote, local


_SPECTRUM_CONTEXT: dict | None = None


def _spectrum_worker(index_and_omega: tuple[int, float]) -> tuple:
    if _SPECTRUM_CONTEXT is None:
        raise RuntimeError("spectrum worker context is not initialized")
    idx, omega = index_and_omega
    ctx = _SPECTRUM_CONTEXT
    row = [idx, omega]
    for label in ("initial", "learned"):
        transfer, remote, local = single_source_transfer(
            ctx[f"{label}_values"],
            ctx[f"{label}_vectors"],
            omega,
            ctx["gamma"],
            ctx["source_dof"],
            ctx["receiver_radius"],
        )
        row.extend([transfer, remote, local])
    return tuple(row)


def compute_spectrum(seed: int, nk: int, n_freq: int,
                     gamma_fraction: float, receiver_radius: int,
                     workers: int) -> dict[str, np.ndarray]:
    global _SPECTRUM_CONTEXT
    cell, result = load_npz_result(cell_path(seed))
    wlo = float(result["wlo"])
    whi = float(result["whi"])
    omega_target = 0.5 * (wlo + whi)
    gamma = float(gamma_fraction) * omega_target
    source_node, source_dof = fixed_source(cell)

    eig = {}
    for label, radii in (
        ("initial", np.full(cell.n_edges, INITIAL_RADIUS, dtype=float)),
        ("learned", np.asarray(result["radii"], dtype=float)),
    ):
        log(f"spectrum eigensystem seed={seed} state={label} nk={nk}")
        eig[f"{label}_values"], eig[f"{label}_vectors"] = eigensystem(
            cell, radii, nk,
        )
    gap = spectral_gap(eig["learned_values"], wlo, whi, nk)
    gap_width = float(gap["upper_edge"] - gap["lower_edge"])
    freq_lo = max(0.05, float(gap["lower_edge"]) - 0.85 * gap_width)
    freq_hi = float(gap["upper_edge"]) + 0.70 * gap_width
    frequencies = np.linspace(freq_lo, freq_hi, int(n_freq))

    _SPECTRUM_CONTEXT = {
        **eig,
        "gamma": gamma,
        "source_dof": source_dof,
        "receiver_radius": receiver_radius,
    }
    tasks = list(enumerate(frequencies.tolist()))
    n_workers = max(1, min(int(workers), len(tasks)))
    t0 = time.perf_counter()
    if n_workers == 1:
        rows = [_spectrum_worker(task) for task in tasks]
    else:
        # Fork preserves the large read-only eigensystems as copy-on-write
        # shared pages instead of serializing them into every worker.
        with cf.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("fork"),
        ) as pool:
            rows = list(pool.map(_spectrum_worker, tasks, chunksize=1))
    rows.sort(key=lambda row: row[0])
    array = np.asarray(rows, dtype=float)
    log(f"spectrum responses: {time.perf_counter() - t0:.1f} s with {n_workers} workers")
    _SPECTRUM_CONTEXT = None
    return {
        "seed": np.int64(seed),
        "nk": np.int64(nk),
        "n_freq": np.int64(n_freq),
        "gamma_fraction": np.float64(gamma_fraction),
        "gamma": np.float64(gamma),
        "wlo": np.float64(wlo),
        "whi": np.float64(whi),
        "omega_target": np.float64(omega_target),
        "source_node": np.int64(source_node),
        "source_dof": np.int64(source_dof),
        "receiver_radius": np.int64(receiver_radius),
        "gap_lo": np.float64(gap["lower_edge"]),
        "gap_hi": np.float64(gap["upper_edge"]),
        "normalized_gap": np.float64(gap["normalized_gap"]),
        "frequencies": array[:, 1],
        "initial_transfer": array[:, 2],
        "initial_remote": array[:, 3],
        "initial_local": array[:, 4],
        "learned_transfer": array[:, 5],
        "learned_remote": array[:, 6],
        "learned_local": array[:, 7],
    }


def run_spectrum(seed: int, nk: int, n_freq: int, gamma_fraction: float,
                 receiver_radius: int, workers: int,
                 force: bool = False) -> Path:
    ensure_dirs()
    out = spectrum_cache_path(seed, nk, n_freq, gamma_fraction)
    if out.exists() and not force:
        log(f"cache hit: {out.name}")
        return out
    t0 = time.perf_counter()
    payload = compute_spectrum(
        seed, nk, n_freq, gamma_fraction, receiver_radius, workers,
    )
    payload["elapsed_s"] = np.float64(time.perf_counter() - t0)
    np.savez_compressed(out, **payload)
    log(f"wrote {out} ({float(payload['elapsed_s']):.1f} s)")
    return out


def _target_job(args: tuple) -> str:
    seed, nk, gamma_fraction, max_radius, force = args
    return str(run_target_cell(seed, nk, gamma_fraction, max_radius, force))


def run_ensemble(seeds: list[int], nk: int, gamma_fraction: float,
                 max_radius: int, workers: int, force: bool) -> list[Path]:
    jobs = [(seed, nk, gamma_fraction, max_radius, force) for seed in seeds]
    n_workers = max(1, min(int(workers), len(jobs)))
    log(f"ensemble: {len(jobs)} cells with {n_workers} workers")
    if n_workers == 1:
        return [Path(_target_job(job)) for job in jobs]
    with cf.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn"),
    ) as pool:
        return [Path(path) for path in pool.map(_target_job, jobs, chunksize=1)]


def convergence_for_grid(nk: int, gamma_fractions: list[float],
                         max_radius: int, force: bool) -> Path:
    out = convergence_cache_path(nk, gamma_fractions)
    if out.exists() and not force:
        log(f"cache hit: {out.name}")
        return out
    cell, result = load_npz_result(cell_path(0))
    wlo, whi = float(result["wlo"]), float(result["whi"])
    omega = 0.5 * (wlo + whi)
    _, source_dof = fixed_source(cell)
    payload: dict[str, np.ndarray] = {
        "nk": np.int64(nk),
        "gamma_fractions": np.asarray(gamma_fractions, dtype=float),
        "omega_target": np.float64(omega),
        "max_radius": np.int64(max_radius),
    }
    eig = {}
    for label, radii in (
        ("initial", np.full(cell.n_edges, INITIAL_RADIUS, dtype=float)),
        ("learned", np.asarray(result["radii"], dtype=float)),
    ):
        log(f"convergence nk={nk} state={label}")
        eig[label] = eigensystem(cell, radii, nk)
    gap = spectral_gap(eig["learned"][0], wlo, whi, nk)
    payload["gap_lo"] = np.float64(gap["lower_edge"])
    payload["gap_hi"] = np.float64(gap["upper_edge"])

    for label in ("initial", "learned"):
        radial_rows = []
        for fraction in gamma_fractions:
            gamma = float(fraction) * omega
            green = green_real_space(*eig[label], omega, gamma)
            diag = all_source_diagnostics(green, max_radius, source_dof)
            radial_rows.append(diag["radial_transfer"])
            del green
        payload[f"{label}_radial_transfer"] = np.stack(radial_rows)
    np.savez_compressed(out, **payload)
    log(f"wrote {out}")
    return out


def _convergence_job(args: tuple) -> str:
    nk, gammas, max_radius, force = args
    return str(convergence_for_grid(nk, gammas, max_radius, force))


def run_convergence(grids: list[int], gamma_fractions: list[float],
                    max_radius: int, workers: int, force: bool) -> list[Path]:
    jobs = [(nk, gamma_fractions, max_radius, force) for nk in grids]
    n_workers = max(1, min(int(workers), len(jobs)))
    log(f"convergence: {len(jobs)} grids with {n_workers} workers")
    if n_workers == 1:
        return [Path(_convergence_job(job)) for job in jobs]
    with cf.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn"),
    ) as pool:
        return [Path(path) for path in pool.map(_convergence_job, jobs)]


def finite_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(values, 10.0 ** (DB_FLOOR / 20.0)))


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(np.asarray(values, dtype=float), [0, 10, 25, 50, 75, 90, 100])
    return {
        "min": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
        "median": float(q[3]), "p75": float(q[4]), "p90": float(q[5]),
        "max": float(q[6]),
    }


def build_summary(target_paths: list[Path], spectrum_path: Path,
                  convergence_paths: list[Path], primary_radius: int) -> dict:
    targets = [load_npz(path) for path in target_paths]
    initial = np.concatenate([
        d["initial_radial_transfer"][primary_radius] for d in targets
    ])
    learned = np.concatenate([
        d["learned_radial_transfer"][primary_radius] for d in targets
    ])
    delta_db = finite_db(learned) - finite_db(initial)
    per_cell = []
    for data in targets:
        cell_delta = (
            finite_db(data["learned_radial_transfer"][primary_radius])
            - finite_db(data["initial_radial_transfer"][primary_radius])
        )
        per_cell.append({
            "seed": int(data["seed"]),
            "median_delta_db": float(np.median(cell_delta)),
            "worst_delta_db": float(np.max(cell_delta)),
            "best_delta_db": float(np.min(cell_delta)),
        })

    spectrum = load_npz(spectrum_path)
    freq = spectrum["frequencies"]
    in_gap = (freq >= float(spectrum["gap_lo"])) & (freq <= float(spectrum["gap_hi"]))
    outside = ~in_gap
    convergence = []
    for path in convergence_paths:
        data = load_npz(path)
        for j, gamma_fraction in enumerate(data["gamma_fractions"]):
            init = data["initial_radial_transfer"][j, primary_radius]
            learn = data["learned_radial_transfer"][j, primary_radius]
            convergence.append({
                "nk": int(data["nk"]),
                "gamma_fraction": float(gamma_fraction),
                "gap_lo": float(data["gap_lo"]),
                "gap_hi": float(data["gap_hi"]),
                "median_delta_db": float(np.median(finite_db(learn) - finite_db(init))),
                "worst_delta_db": float(np.max(finite_db(learn) - finite_db(init))),
            })

    summary = {
        "claim": (
            "No localized source-receiver path or transmission loss is optimized; "
            "the learned intrinsic gap suppresses responses to subsequently chosen point forces."
        ),
        "primary_radius_cells": int(primary_radius),
        "n_materials": len(targets),
        "n_coordinate_sources": int(len(initial)),
        "initial_transfer_db": percentile_dict(finite_db(initial)),
        "learned_transfer_db": percentile_dict(finite_db(learned)),
        "learned_minus_initial_db": percentile_dict(delta_db),
        "per_cell": sorted(per_cell, key=lambda x: x["seed"]),
        "spectrum": {
            "gap_lo": float(spectrum["gap_lo"]),
            "gap_hi": float(spectrum["gap_hi"]),
            "target_lo": float(spectrum["wlo"]),
            "target_hi": float(spectrum["whi"]),
            "median_learned_transfer_db_inside_gap": float(
                np.median(finite_db(spectrum["learned_transfer"][in_gap]))
            ),
            "median_learned_transfer_db_outside_gap": float(
                np.median(finite_db(spectrum["learned_transfer"][outside]))
            ),
        },
        "convergence": convergence,
        "files": {
            "targets": [str(path.relative_to(REPO_ROOT)) for path in target_paths],
            "spectrum": str(spectrum_path.relative_to(REPO_ROOT)),
            "convergence": [str(path.relative_to(REPO_ROOT)) for path in convergence_paths],
        },
    }
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(f"wrote {out}")
    return summary


def crop_map(map_db: np.ndarray, extent: int) -> np.ndarray:
    center = map_db.shape[0] // 2
    return map_db[
        center - extent:center + extent + 1,
        center - extent:center + extent + 1,
    ]


def shade_gap_and_target(ax, gap_lo: float, gap_hi: float,
                         wlo: float, whi: float) -> None:
    ax.axvspan(gap_lo, gap_hi, color=ps.BLUE, alpha=0.10, lw=0, zorder=0)
    for x in (gap_lo, gap_hi):
        ax.axvline(x, color=ps.BLUE, alpha=0.65, lw=0.55, zorder=0.1)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([wlo, whi], [0.965, 0.965], color=ps.WINDOW, lw=2.0,
            solid_capstyle="butt", transform=trans, clip_on=False)


def make_figure(exemplar_path: Path, spectrum_path: Path,
                ensemble_paths: list[Path], map_extent: int,
                radial_max: int) -> Path:
    ps.style()
    exemplar = load_npz(exemplar_path)
    spectrum = load_npz(spectrum_path)
    ensemble = [load_npz(path) for path in ensemble_paths]

    fig = plt.figure(figsize=(ps.COL_W, 3.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=-100.0, vmax=0.0, clip=True)
    image_extent = [-map_extent - 0.5, map_extent + 0.5,
                    -map_extent - 0.5, map_extent + 0.5]
    for ax, label, key, title in (
        (ax_a, "a", "initial_map_db", "initial"),
        (ax_b, "b", "learned_map_db", "learned"),
    ):
        ps.panel_label(ax, label)
        shown = crop_map(exemplar[key], map_extent).T
        im = ax.imshow(shown, origin="lower", extent=image_extent,
                       cmap=cmap, norm=norm, interpolation="nearest",
                       rasterized=True)
        ax.scatter([0], [0], marker="*", s=19, color="#23d5d5",
                   edgecolors="white", linewidths=0.25, zorder=4)
        ps.panel_tag(ax, title, color=ps.GRAY_DARK if title == "initial" else ps.BLUE)
        ax.set_aspect("equal")
        ax.set_xticks([-map_extent, 0, map_extent])
        ax.set_yticks([-map_extent, 0, map_extent])
        ax.set_xlabel(r"cell $R_x$")
    ax_a.set_ylabel(r"cell $R_y$")
    ax_b.tick_params(labelleft=False)
    cbar = fig.colorbar(im, ax=[ax_a, ax_b], orientation="vertical",
                        fraction=0.052, pad=0.025, shrink=0.92, aspect=18)
    cbar.set_label(r"relative response (dB)")
    cbar.set_ticks([-100, -50, 0])

    ps.panel_label(ax_c, "c")
    freq = spectrum["frequencies"]
    initial_db = finite_db(spectrum["initial_transfer"])
    learned_db = finite_db(spectrum["learned_transfer"])
    shade_gap_and_target(
        ax_c, float(spectrum["gap_lo"]), float(spectrum["gap_hi"]),
        float(spectrum["wlo"]), float(spectrum["whi"]),
    )
    ax_c.plot(freq, initial_db, color=ps.GRAY_DARK, lw=0.9, label="initial")
    ax_c.plot(freq, learned_db, color=ps.BLUE, lw=1.0, label="learned")
    ax_c.set_xlabel(r"frequency $\omega$")
    ax_c.set_ylabel(r"shell-averaged response $\mathcal{R}_3$ (dB)")
    ax_c.set_ylim(DB_FLOOR, 8)
    ax_c.legend(loc="lower left", frameon=False, handlelength=1.4)

    ps.panel_label(ax_d, "d")
    radii = np.arange(min(radial_max, int(ensemble[0]["max_radius"])) + 1)
    initial_all = np.concatenate([
        data["initial_radial_transfer"][radii].T for data in ensemble
    ], axis=0)
    learned_all = np.concatenate([
        data["learned_radial_transfer"][radii].T for data in ensemble
    ], axis=0)
    for values, color, label in (
        (initial_all, ps.GRAY_DARK, "initial"),
        (learned_all, ps.BLUE, "learned"),
    ):
        db = finite_db(values)
        p10, median, p90 = np.percentile(db, [10, 50, 90], axis=0)
        ax_d.fill_between(radii, p10, p90, color=color, alpha=0.16, lw=0)
        ax_d.plot(radii, median, color=color, lw=1.05, marker="o",
                  ms=2.5, label=label)
    ax_d.set_xlabel(r"shell distance $d$ (cells)")
    ax_d.set_ylabel(r"shell-averaged response $\mathcal{R}_d$ (dB)")
    ax_d.set_xticks(radii)
    ax_d.set_ylim(DB_FLOOR, 8)
    ax_d.legend(loc="lower left", frameon=False, handlelength=1.4)
    # The material/source counts belong in the caption; keeping them off the
    # axes leaves the attenuation slopes and their ensemble envelopes legible
    # at final PRL column width.

    ps.savefig(fig, FIG_DIR, FIGURE_NAME)
    out = FIG_DIR / f"{FIGURE_NAME}.pdf"
    log(f"wrote {out} and PNG")
    return out


def validate_payload(path: Path) -> None:
    data = load_npz(path)
    required = {
        "initial_radial_transfer", "learned_radial_transfer",
        "initial_map_db", "learned_map_db", "gap_lo", "gap_hi",
    }
    missing = required.difference(data)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    for key in required:
        if (
            key in data
            and isinstance(data[key], np.ndarray)
            and not np.all(np.isfinite(data[key]))
        ):
            raise RuntimeError(f"non-finite values in {path}:{key}")
    if not (float(data["gap_lo"]) < float(data["wlo"])
            and float(data["gap_hi"]) > float(data["whi"])):
        raise RuntimeError(f"learned gap in {path} does not contain the prescribed spectral window")


def smoke(args) -> None:
    ensure_dirs()
    target = run_target_cell(0, args.smoke_nk, args.gamma_fraction,
                             args.smoke_radius, args.force)
    validate_payload(target)
    spectrum = run_spectrum(0, args.smoke_nk, args.smoke_nfreq,
                            args.gamma_fraction, args.primary_radius,
                            args.workers, args.force)
    make_figure(target, spectrum, [target],
                min(args.map_extent, args.smoke_radius),
                min(args.radial_max, args.smoke_radius))
    summary = build_summary([target], spectrum, [], args.primary_radius)
    log(
        "smoke primary median learned-initial = "
        f"{summary['learned_minus_initial_db']['median']:.1f} dB"
    )


def full(args) -> None:
    ensure_dirs()
    t0 = time.perf_counter()
    prepare_materials(list(range(args.n_materials)), args.workers, args.force)
    exemplar = run_target_cell(0, args.exemplar_nk, args.gamma_fraction,
                               args.max_radius, args.force)
    validate_payload(exemplar)
    spectrum = run_spectrum(0, args.exemplar_nk, args.n_freq,
                            args.gamma_fraction, args.primary_radius,
                            args.workers, args.force)
    ensemble_paths = run_ensemble(
        list(range(args.n_materials)), args.ensemble_nk,
        args.gamma_fraction, args.max_radius, args.workers, args.force,
    )
    for path in ensemble_paths:
        validate_payload(path)
    convergence_paths = run_convergence(
        args.convergence_grids, args.convergence_gammas,
        args.max_radius, args.workers, args.force,
    )
    summary = build_summary(
        ensemble_paths, spectrum, convergence_paths, args.primary_radius,
    )
    make_figure(exemplar, spectrum, ensemble_paths,
                args.map_extent, args.radial_max)
    log(
        "full primary median learned-initial = "
        f"{summary['learned_minus_initial_db']['median']:.1f} dB; "
        f"worst = {summary['learned_minus_initial_db']['max']:.1f} dB"
    )
    log(f"full pipeline complete in {(time.perf_counter() - t0) / 60.0:.1f} min")


def render(args) -> None:
    exemplar = target_cache_path(0, args.exemplar_nk, args.gamma_fraction)
    spectrum = spectrum_cache_path(0, args.exemplar_nk, args.n_freq,
                                   args.gamma_fraction)
    ensemble = [
        target_cache_path(seed, args.ensemble_nk, args.gamma_fraction)
        for seed in range(args.n_materials)
    ]
    missing = [path for path in [exemplar, spectrum, *ensemble] if not path.exists()]
    if missing:
        raise FileNotFoundError("missing caches:\n" + "\n".join(map(str, missing)))
    make_figure(exemplar, spectrum, ensemble, args.map_extent, args.radial_max)


def status() -> None:
    ensure_dirs()
    files = sorted(OUT_DIR.glob("*.npz"))
    log(f"{len(files)} cached NPZ files in {OUT_DIR}")
    for path in files:
        print(f"  {path.name:72s} {path.stat().st_size / 1024**2:8.2f} MiB")
    summary = OUT_DIR / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        print(json.dumps({
            "n_materials": data.get("n_materials"),
            "n_coordinate_sources": data.get("n_coordinate_sources"),
            "learned_minus_initial_db": data.get("learned_minus_initial_db"),
        }, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("smoke", "full", "render", "status"))
    p.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 1))
    p.add_argument("--force", action="store_true")
    p.add_argument("--gamma-fraction", type=float, default=0.01,
                   help="viscous gamma / target-midpoint frequency")
    p.add_argument("--primary-radius", type=int, default=3)
    p.add_argument("--max-radius", type=int, default=10)
    p.add_argument("--map-extent", type=int, default=8)
    p.add_argument("--radial-max", type=int, default=4)
    p.add_argument("--exemplar-nk", type=int, default=129)
    p.add_argument("--ensemble-nk", type=int, default=81)
    p.add_argument("--n-freq", type=int, default=301)
    p.add_argument("--n-materials", type=int, default=10)
    p.add_argument("--convergence-grids", nargs="+", type=int,
                   default=[33, 49, 65, 81, 97, 129])
    p.add_argument("--convergence-gammas", nargs="+", type=float,
                   default=[0.0025, 0.005, 0.01, 0.02, 0.04])
    p.add_argument("--smoke-nk", type=int, default=25)
    p.add_argument("--smoke-nfreq", type=int, default=61)
    p.add_argument("--smoke-radius", type=int, default=8)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.primary_radius > args.max_radius:
        raise ValueError("primary radius cannot exceed max radius")
    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "full":
        full(args)
    elif args.mode == "render":
        render(args)
    else:
        status()


if __name__ == "__main__":
    main()
