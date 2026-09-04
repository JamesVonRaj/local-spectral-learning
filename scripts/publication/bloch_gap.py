"""Train and validate the periodic vector/Bloch-gap model.

This script is intentionally separate from the scalar graph-Laplacian PRL
pipeline.  It trains a periodically repeated 2D central-force spring cell using
two response fields and then checks whether the trained cell opens a spectral
exclusion window over sampled Bloch wavevectors.

The model is still a training protocol, not an autonomous material
implementation: the drive band, Bloch wavevectors, and random probes specify
the task.  The edge update itself is local in the same sense as contrastive
physical learning: each edge uses only its radius, length, edge direction, and
the two response extensions across that edge.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMPI_MCA_btl", "^openib")
os.environ.setdefault("OMPI_MCA_btl_openib_warn_no_device_params_found", "0")

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mechanical.topology import make_network
from scipy.linalg import lu_factor, lu_solve

from publication import style as ps
from publication.paths import FIGURE_DIR, REPO_ROOT, dataset

DATA_DIR = dataset("prl_vector_periodic")
FIG_DIR = FIGURE_DIR

EPS_REG = 1e-6  # legacy complex-shift control only; reciprocal runs are bare
R_INIT = 1.0
R_MIN = 0.5
R_MAX = 2.0
RESPONSE_MODES = ("reciprocal", "viscous", "complex")


@dataclass
class PeriodicVectorCell:
    size: int
    seed: int
    pos: np.ndarray
    box: np.ndarray
    edges: np.ndarray
    offsets: np.ndarray
    vectors: np.ndarray
    lengths: np.ndarray
    directions: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.pos.shape[0])

    @property
    def n_dim(self) -> int:
        return int(self.pos.shape[1])

    @property
    def n_dof(self) -> int:
        return self.n_dim * self.n_nodes

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)


def json_ready(obj):
    """Convert numpy-heavy result dictionaries into strict JSON values."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, np.generic):
        return json_ready(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def make_periodic_vector_cell(size: int, seed: int,
                              topology: str = "rand-del") -> PeriodicVectorCell:
    """Build a periodic central-force cell from an existing spatial topology.

    The topology generator gives periodic edge lengths but not explicit image
    offsets.  For a periodic Delaunay/Gabriel edge, the minimum-image offset is
    reconstructed from the endpoint coordinates and the periodic box.
    """
    pos, edges, _, box = make_network(topology, size=int(size), seed=int(seed))
    pos = np.asarray(pos, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("periodic vector validation requires 2D positions")
    if np.any(box <= 0):
        raise ValueError(f"invalid periodic box {box}")

    raw = pos[edges[:, 1]] - pos[edges[:, 0]]
    offsets = -np.rint(raw / box).astype(np.int64)
    vectors = raw + offsets * box
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("zero-length periodic edge encountered")
    directions = vectors / lengths[:, None]
    return PeriodicVectorCell(
        size=int(size),
        seed=int(seed),
        pos=pos,
        box=box,
        edges=edges,
        offsets=offsets,
        vectors=vectors,
        lengths=lengths,
        directions=directions,
    )


def bloch_phases(cell: PeriodicVectorCell, kred: np.ndarray) -> np.ndarray:
    """Bloch phases for reduced wavevector components in radians per cell."""
    return np.exp(1j * (cell.offsets @ np.asarray(kred, dtype=np.float64)))


def edge_extensions(cell: PeriodicVectorCell, field: np.ndarray,
                    kred: np.ndarray) -> np.ndarray:
    """Compute central-force edge extensions for one or more fields.

    Parameters
    ----------
    field
        Complex displacement array with shape ``(dN,)`` or ``(dN, batch)``,
        where ``d`` is the spatial dimension.

    Returns
    -------
    ndarray
        Complex edge extensions with shape ``(M,)`` or ``(M, batch)``.
    """
    field = np.asarray(field)
    was_1d = field.ndim == 1
    if was_1d:
        field = field[:, None]
    if field.shape[0] != cell.n_dof:
        raise ValueError(
            f"field has {field.shape[0]} rows; expected {cell.n_dof}"
        )
    i = cell.edges[:, 0]
    j = cell.edges[:, 1]
    n = cell.directions
    nodal = field.reshape(cell.n_nodes, cell.n_dim, field.shape[1])
    ui = np.einsum("ed,edb->eb", n, nodal[i], optimize=True)
    uj = np.einsum("ed,edb->eb", n, nodal[j], optimize=True)
    ext = bloch_phases(cell, kred)[:, None] * uj - ui
    return ext[:, 0] if was_1d else ext


def bloch_stiffness(cell: PeriodicVectorCell, radii: np.ndarray,
                    kred: np.ndarray) -> np.ndarray:
    """Hermitian Bloch stiffness matrix for central-force springs."""
    radii = np.asarray(radii, dtype=np.float64)
    if radii.shape != (cell.n_edges,):
        raise ValueError(f"radii must have shape {(cell.n_edges,)}, got {radii.shape}")
    K = np.zeros((cell.n_dof, cell.n_dof), dtype=np.complex128)
    phases = bloch_phases(cell, kred)
    stiffness = radii**2 / cell.lengths
    for e, (i, j) in enumerate(cell.edges):
        n = cell.directions[e]
        nn = np.outer(n, n)
        ke = stiffness[e]
        phase = phases[e]
        si = slice(cell.n_dim * i, cell.n_dim * (i + 1))
        sj = slice(cell.n_dim * j, cell.n_dim * (j + 1))
        K[si, si] += ke * nn
        K[sj, sj] += ke * nn
        K[si, sj] += -ke * phase * nn
        K[sj, si] += -ke * np.conjugate(phase) * nn
    return 0.5 * (K + K.conjugate().T)


def band_frequencies(cell: PeriodicVectorCell, radii: np.ndarray,
                     kpoints: np.ndarray) -> np.ndarray:
    freqs = np.empty((len(kpoints), cell.n_dof), dtype=np.float64)
    for q, kred in enumerate(kpoints):
        vals = np.linalg.eigvalsh(bloch_stiffness(cell, radii, kred))
        freqs[q] = np.sqrt(np.maximum(vals.real, 0.0))
    return freqs


def band_index_gaps(freqs: np.ndarray, kpoints: np.ndarray | None = None) -> list[dict]:
    """Compute indirect gaps between adjacent Bloch bands.

    Band index ``m`` is zero-based and denotes the gap between bands ``m`` and
    ``m + 1``.  For a true periodic gap, ``gap`` must be positive.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.ndim != 2:
        raise ValueError(f"freqs must have shape (n_k, n_bands), got {freqs.shape}")
    if kpoints is not None:
        kpoints = np.asarray(kpoints, dtype=np.float64)
        if kpoints.shape[0] != freqs.shape[0]:
            raise ValueError("kpoints and freqs have incompatible lengths")

    gaps = []
    for band in range(freqs.shape[1] - 1):
        lower_idx = int(np.argmax(freqs[:, band]))
        upper_idx = int(np.argmin(freqs[:, band + 1]))
        lower_edge = float(freqs[lower_idx, band])
        upper_edge = float(freqs[upper_idx, band + 1])
        gap = upper_edge - lower_edge
        midpoint = 0.5 * (upper_edge + lower_edge)
        record = {
            "band_index": int(band),
            "lower_band_1based": int(band + 1),
            "upper_band_1based": int(band + 2),
            "lower_edge": lower_edge,
            "upper_edge": upper_edge,
            "gap": float(gap),
            "normalized_gap": float(gap / midpoint) if midpoint > 0 else float("nan"),
            "lower_k_index": lower_idx,
            "upper_k_index": upper_idx,
        }
        if kpoints is not None:
            record["lower_k"] = kpoints[lower_idx].astype(float)
            record["upper_k"] = kpoints[upper_idx].astype(float)
        gaps.append(record)
    return gaps


def best_gap_near_window(
    freqs: np.ndarray,
    wlo: float,
    whi: float,
    kpoints: np.ndarray | None = None,
) -> dict:
    """Return the positive adjacent-band gap that best covers the prescribed spectral window.

    If no positive gap exists, return the least-bad adjacent-band separation so
    downstream diagnostics can still report why the test failed.
    """
    target_mid = 0.5 * (float(wlo) + float(whi))
    target_width = float(whi) - float(wlo)
    records = []
    for gap in band_index_gaps(freqs, kpoints):
        lower = float(gap["lower_edge"])
        upper = float(gap["upper_edge"])
        positive = float(gap["gap"]) > 0.0
        overlap = max(0.0, min(upper, float(whi)) - max(lower, float(wlo))) if positive else 0.0
        if upper < wlo:
            target_distance = float(wlo) - upper
        elif lower > whi:
            target_distance = lower - float(whi)
        else:
            target_distance = 0.0
        enriched = dict(gap)
        enriched.update({
            "positive": bool(positive),
            "overlaps_target_window": bool(overlap > 0.0),
            "contains_target_window": bool(positive and lower <= wlo and upper >= whi),
            "target_overlap_width": float(overlap),
            "target_overlap_fraction": float(overlap / target_width) if target_width > 0 else float("nan"),
            "target_distance": float(target_distance),
            "target_mid_distance": float(abs(0.5 * (lower + upper) - target_mid)),
        })
        records.append(enriched)

    positive = [r for r in records if r["positive"]]
    if positive:
        positive.sort(
            key=lambda r: (
                not r["contains_target_window"],
                not r["overlaps_target_window"],
                r["target_distance"],
                r["target_mid_distance"],
                -r["gap"],
            )
        )
        selected = dict(positive[0])
        selected["selected_by"] = "positive_gap_nearest_target"
        return selected

    records.sort(key=lambda r: (r["target_distance"], r["target_mid_distance"], -r["gap"]))
    selected = dict(records[0])
    selected["selected_by"] = "no_positive_gap"
    return selected


def k_grid(n: int, include_gamma: bool = True, dimension: int = 2) -> np.ndarray:
    """Uniform endpoint-excluding grid in a reduced cubic Brillouin zone."""
    import itertools

    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    vals = np.linspace(-np.pi, np.pi, int(n), endpoint=False)
    pts = np.asarray(list(itertools.product(vals, repeat=dimension)), dtype=np.float64)
    if include_gamma and not np.any(np.all(np.isclose(pts, 0.0), axis=1)):
        pts = np.vstack([pts, np.zeros(dimension)])
    return pts


def k_path(n_segment: int = 35) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    points = [
        (np.array([0.0, 0.0]), r"$\Gamma$"),
        (np.array([np.pi, 0.0]), "X"),
        (np.array([np.pi, np.pi]), "M"),
        (np.array([0.0, 0.0]), r"$\Gamma$"),
    ]
    kpts = []
    dist = []
    ticks = [0.0]
    labels = [points[0][1]]
    running = 0.0
    for (a, _), (b, label_b) in zip(points[:-1], points[1:]):
        for idx in range(int(n_segment)):
            t = idx / float(n_segment)
            p = (1.0 - t) * a + t * b
            if kpts:
                running += float(np.linalg.norm(p - kpts[-1]))
            kpts.append(p)
            dist.append(running)
        running += float(np.linalg.norm(b - kpts[-1]))
        kpts.append(b)
        dist.append(running)
        ticks.append(running)
        labels.append(label_b)
    return np.asarray(kpts), np.asarray(dist), labels, np.asarray(ticks)


def target_window_from_percentiles(freqs: np.ndarray, center_percentile: float,
                                   width_percentile: float,
                                   zero_tol: float = 1e-7) -> tuple[float, float]:
    vals = np.asarray(freqs, dtype=np.float64).ravel()
    vals = vals[vals > zero_tol]
    half = 0.5 * float(width_percentile)
    lo_p = max(0.0, float(center_percentile) - half)
    hi_p = min(100.0, float(center_percentile) + half)
    lo, hi = np.percentile(vals, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"invalid percentile window {lo_p}-{hi_p}: {lo}, {hi}")
    return float(lo), float(hi)


def window_metrics(freqs: np.ndarray, wlo: float, whi: float,
                   zero_tol: float = 1e-7) -> dict[str, float]:
    vals = np.sort(np.asarray(freqs, dtype=np.float64).ravel())
    vals = vals[vals > zero_tol]
    n_in = int(np.sum((vals > wlo) & (vals < whi)))
    below = vals[vals <= wlo]
    above = vals[vals >= whi]
    if n_in == 0 and len(below) and len(above):
        gap_lo = float(below[-1])
        gap_hi = float(above[0])
        gap_width = max(0.0, gap_hi - gap_lo)
        gap_mid = 0.5 * (gap_hi + gap_lo)
        gap_ratio = gap_width / gap_mid if gap_mid > 0 else 0.0
    else:
        gap_lo = gap_hi = gap_width = gap_ratio = 0.0
    if len(vals):
        unique_tol = 1e-5 * max(1.0, float(np.max(vals)))
        vals_unique = vals[np.r_[True, np.diff(vals) > unique_tol]]
    else:
        vals_unique = vals
    local = vals_unique[
        (vals_unique > wlo - 0.5 * (whi - wlo))
        & (vals_unique < whi + 0.5 * (whi - wlo))
    ]
    if len(local) >= 3:
        spacing = float(np.median(np.diff(local)))
    else:
        spacing = float(np.median(np.diff(vals_unique))) if len(vals_unique) >= 3 else float("nan")
    return {
        "n_in": n_in,
        "gap_lo": gap_lo,
        "gap_hi": gap_hi,
        "gap_width": gap_width,
        "gap_ratio": gap_ratio,
        "gap_over_target_width": gap_width / (whi - wlo),
        "gap_over_spacing": gap_width / spacing if spacing > 0 else float("nan"),
        "local_spacing": spacing,
    }


def material_diagnostics(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    radius_min: float | None = R_MIN,
    radius_max: float | None = R_MAX,
) -> dict[str, float]:
    radii = np.asarray(radii, dtype=np.float64)
    k0 = 1.0 / cell.lengths
    kf = radii**2 / cell.lengths
    mat0 = np.sum(cell.lengths)
    matf = np.sum(radii**2 * cell.lengths)
    at_min = (
        np.zeros_like(radii, dtype=bool)
        if radius_min is None
        else radii <= float(radius_min) + 1e-6
    )
    at_max = (
        np.zeros_like(radii, dtype=bool)
        if radius_max is None
        else radii >= float(radius_max) - 1e-6
    )
    return {
        "mean_radius": float(np.mean(radii)),
        "mean_stiffness_ratio": float(np.mean(kf) / np.mean(k0)),
        "material_ratio": float(matf / mat0),
        "frac_rmin": float(np.mean(at_min)),
        "frac_rmax": float(np.mean(at_max)),
        "frac_at_bounds": float(np.mean(at_min | at_max)),
    }


def random_complex_forces(rng: np.random.RandomState, n_dof: int,
                          batch: int) -> np.ndarray:
    """Complex Gaussian probes with unit expected total power.

    Each component has covariance ``1 / n_dof``, so force averaging estimates
    the per-degree-of-freedom trace rather than the unnormalized trace.
    """
    f = rng.normal(size=(n_dof, int(batch))) + 1j * rng.normal(size=(n_dof, int(batch)))
    f /= np.sqrt(2.0 * n_dof)
    return f


def response_matrix(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    kred: np.ndarray,
    omega: float,
    eps_reg: float,
    response_mode: str,
    damping_gamma: float = 0.0,
) -> np.ndarray:
    K = bloch_stiffness(cell, radii, kred)
    identity = np.eye(cell.n_dof, dtype=np.complex128)
    base = K - (float(omega) ** 2) * identity
    if response_mode == "reciprocal":
        # Bare Hermitian Bloch operator; eps_reg applies only to the
        # complex-damped control mode.
        return base
    if response_mode == "complex":
        return base + 1j * float(eps_reg) * (float(omega) ** 2) * identity
    if response_mode == "viscous":
        gamma = float(damping_gamma)
        if gamma < 0.0:
            raise ValueError("damping_gamma must be nonnegative")
        return base + 1j * gamma * float(omega) * identity
    raise ValueError(f"unknown response mode {response_mode}")


def local_bloch_step(cell: PeriodicVectorCell, radii: np.ndarray,
                     kpoints: np.ndarray, omegas: np.ndarray,
                     rng: np.random.RandomState, force_batch: int,
                     eps_reg: float = EPS_REG,
                     update_rule: str = "adjoint",
                     response_mode: str = "reciprocal",
                     damping_gamma: float = 0.0,
                     return_moments: bool = False):
    grad = np.zeros(cell.n_edges, dtype=np.float64)
    moment_u = np.zeros(cell.n_edges, dtype=np.float64)
    moment_v = np.zeros(cell.n_edges, dtype=np.float64)
    moment_d = np.zeros(cell.n_edges, dtype=np.float64)
    cost = 0.0
    inv = 1.0 / (len(kpoints) * len(omegas) * int(force_batch))
    two_r_over_l = 2.0 * radii / cell.lengths
    if update_rule not in {"adjoint", "forward-only", "shuffled-adjoint", "random-matched"}:
        raise ValueError(f"unknown update rule {update_rule}")

    for kred in kpoints:
        for omega in omegas:
            H = response_matrix(
                cell, radii, kred, float(omega), eps_reg, response_mode,
                damping_gamma=damping_gamma,
            )
            F = random_complex_forces(rng, cell.n_dof, force_batch)
            factor = lu_factor(H, check_finite=False)
            U = lu_solve(factor, F, check_finite=False)
            cost += float(np.sum(np.abs(U) ** 2).real) * inv
            du = edge_extensions(cell, U, kred)
            if return_moments:
                moment_u += inv * np.sum(np.abs(du) ** 2, axis=1).real
            if update_rule == "forward-only":
                edge_signal = np.sum(np.abs(du) ** 2, axis=1).real
            else:
                if response_mode == "viscous":
                    # At fixed Bloch wavevector H(k) is not complex symmetric;
                    # reciprocity gives H(k)^T = H(-k).  Conjugating the
                    # conventional adjoint equation therefore produces a
                    # physical second experiment at -k in the same damped cell.
                    # H(-k)=H(k)^T, so the transpose solve reuses the same
                    # numerical factorization.  Physically this is the second
                    # experiment in the reciprocal cell at -k.
                    W = lu_solve(
                        factor, np.conjugate(U), trans=1,
                        check_finite=False,
                    )
                    dsecond = edge_extensions(cell, W, -np.asarray(kred))
                    bilinear_product = True
                else:
                    V = lu_solve(factor, U, trans=2, check_finite=False)
                    dsecond = edge_extensions(cell, V, kred)
                    bilinear_product = False
                dsecond_signal = dsecond
                if return_moments and update_rule == "shuffled-adjoint":
                    dsecond_signal = dsecond[rng.permutation(cell.n_edges)]
                elif return_moments and update_rule == "random-matched":
                    permutation = rng.permutation(cell.n_edges)
                    signs = rng.choice(
                        np.array([-1.0, 1.0]), size=(cell.n_edges, 1),
                    )
                    dsecond_signal = signs * dsecond[permutation]
                if bilinear_product:
                    edge_signal = np.sum(
                        np.real(dsecond_signal * du), axis=1,
                    )
                else:
                    edge_signal = np.sum(
                        np.real(np.conjugate(dsecond_signal) * du), axis=1,
                    )
                if return_moments:
                    moment_v += inv * np.sum(
                        np.abs(dsecond_signal) ** 2, axis=1,
                    ).real
                    moment_d += inv * edge_signal
            grad += (-2.0 * inv) * two_r_over_l * edge_signal
    if return_moments:
        return cost, grad, {"U": moment_u, "V": moment_v, "D": moment_d}
    return cost, grad


def local_regularizer_gradient(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    regularizer: str,
    strength: float,
) -> np.ndarray:
    """Gradient contribution from local-only radius regularizers."""
    strength = float(strength)
    if regularizer in {"none", ""} or strength == 0.0:
        return np.zeros_like(radii)
    if regularizer == "inflation":
        return -strength * np.ones_like(radii)
    if regularizer == "l1":
        return strength * np.sign(radii - R_INIT)
    if regularizer == "l2":
        return strength * (radii - R_INIT)
    if regularizer == "inflation_l1":
        return -strength + strength * np.sign(radii - R_INIT)
    if regularizer == "inflation_l2":
        return -strength + strength * (radii - R_INIT)
    if regularizer == "material_penalty":
        return strength * 2.0 * radii * cell.lengths / float(np.mean(cell.lengths))
    if regularizer == "stiffness_penalty":
        return strength * 2.0 * radii / cell.lengths / float(np.mean(1.0 / cell.lengths))
    if regularizer == "binary_prior":
        x = radii
        return 0.1 * strength * 2.0 * (x - R_MIN) * (x - R_MAX) * (2.0 * x - R_MIN - R_MAX)
    raise ValueError(f"unknown regularizer {regularizer}")


def postprocess_control_gradient(
    grad: np.ndarray,
    rng: np.random.RandomState,
    update_rule: str,
) -> np.ndarray:
    if update_rule == "shuffled-adjoint":
        return grad[rng.permutation(len(grad))]
    if update_rule == "random-matched":
        magnitudes = rng.permutation(np.abs(grad))
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(grad))
        return signs * magnitudes
    return grad


def train_periodic_vector(
    cell: PeriodicVectorCell,
    wlo: float,
    whi: float,
    *,
    n_steps: int = 3000,
    alpha: float = 0.035,
    grad_clip: float | None = None,
    inflation_strength: float = 0.0,
    radius_min: float | None = R_MIN,
    radius_max: float | None = R_MAX,
    n_freq: int = 5,
    k_train_grid: int = 5,
    k_eval_grid: int = 9,
    k_batch: int = 4,
    force_batch: int = 2,
    seed: int = 0,
    eval_every: int = 100,
    update_rule: str = "adjoint",
    regularizer: str = "none",
    response_mode: str = "reciprocal",
    damping_gamma_ratio: float = 0.0,
    material_update: str = "response_conditioned_log",
    response_metric_eta: float = 0.02,
    response_metric_lambda_ratio: float = 0.025,
    response_bound_mode: str = "clip",
    frequency_sampling: str = "random_grid",
    frequencies_per_step: int = 1,
    frequency_seed: int | None = None,
) -> dict:
    if response_mode not in RESPONSE_MODES:
        raise ValueError(f"response_mode must be one of {RESPONSE_MODES}, got {response_mode}")
    rng = np.random.RandomState(int(seed))
    frequency_sampling = str(frequency_sampling)
    if frequency_sampling not in {"random_grid", "all"}:
        raise ValueError("frequency_sampling must be 'random_grid' or 'all'")
    frequencies_per_step = int(frequencies_per_step)
    if frequencies_per_step < 1:
        raise ValueError("frequencies_per_step must be positive")
    material_update = str(material_update)
    if material_update not in {"radius_euler", "response_conditioned_log"}:
        raise ValueError(
            "material_update must be 'radius_euler' or 'response_conditioned_log'"
        )
    response_conditioned_update = material_update == "response_conditioned_log"
    response_bound_mode = str(response_bound_mode)
    if response_conditioned_update:
        if update_rule not in {
            "adjoint", "forward-only", "shuffled-adjoint", "random-matched",
        }:
            raise ValueError(
                "response_conditioned_log requires a paired or matched lesion rule"
            )
        if regularizer not in {"none", ""} or float(inflation_strength) != 0.0:
            raise ValueError("response_conditioned_log uses no additive regularizer")
        if grad_clip is not None:
            raise ValueError(
                "response_conditioned_log supplies an analytic local rate bound; "
                "grad_clip must be None"
            )
        if response_mode not in {"reciprocal", "viscous"}:
            raise ValueError(
                "response_conditioned_log requires reciprocal undamped or "
                "mass-proportional viscous response"
            )
        if response_bound_mode not in {"clip", "none"}:
            raise ValueError("response_bound_mode must be 'clip' or 'none'")
        response_metric_eta = float(response_metric_eta)
        response_metric_lambda_ratio = float(response_metric_lambda_ratio)
        if response_metric_eta <= 0.0 or response_metric_lambda_ratio <= 0.0:
            raise ValueError("response metric eta and lambda ratio must be positive")
        if response_bound_mode == "clip" and (
            radius_min is None
            or radius_max is None
            or float(radius_min) <= 0.0
        ):
            raise ValueError("clipped response-conditioned updates require positive radius bounds")
    clip_min = -np.inf if radius_min is None else float(radius_min)
    clip_max = np.inf if radius_max is None else float(radius_max)
    if clip_min >= clip_max:
        raise ValueError("radius_min must be smaller than radius_max")
    frequency_rng = np.random.RandomState(
        int(seed) + 918273 if frequency_seed is None else int(frequency_seed)
    )
    radii = np.full(cell.n_edges, R_INIT, dtype=np.float64)
    train_k = k_grid(k_train_grid, dimension=cell.n_dim)
    eval_k = k_grid(k_eval_grid, dimension=cell.n_dim)
    omegas = np.linspace(wlo, whi, int(n_freq) + 2)[1:-1]
    damping_gamma_ratio = float(damping_gamma_ratio)
    if damping_gamma_ratio < 0.0:
        raise ValueError("damping_gamma_ratio must be nonnegative")
    if response_mode != "viscous" and damping_gamma_ratio != 0.0:
        raise ValueError(
            "damping_gamma_ratio is only valid with response_mode='viscous'"
        )
    damping_gamma = (
        damping_gamma_ratio * 0.5 * (float(wlo) + float(whi))
        if response_mode == "viscous"
        else 0.0
    )
    response_metric_lambda = (
        response_metric_lambda_ratio * float(whi**2 - wlo**2)
        if response_conditioned_update
        else float("nan")
    )
    if response_conditioned_update:
        theta_state = np.log(radii**2 / cell.lengths)
        if response_bound_mode == "clip":
            theta_min = np.log(float(radius_min) ** 2 / cell.lengths)
            theta_max = np.log(float(radius_max) ** 2 / cell.lengths)

    f0_eval = band_frequencies(cell, radii, eval_k)
    m0 = window_metrics(f0_eval, wlo, whi)
    history = []
    cost_history = []
    t0 = time.perf_counter()
    for step in range(int(n_steps)):
        edge_moments = None
        if update_rule == "inflation-only":
            cost = 0.0
            grad = np.zeros(cell.n_edges, dtype=np.float64)
        else:
            idx = rng.choice(len(train_k), size=min(int(k_batch), len(train_k)), replace=False)
            step_omegas = omegas
            if frequency_sampling == "random_grid":
                omega_idx = frequency_rng.choice(
                    len(omegas),
                    size=min(frequencies_per_step, len(omegas)),
                    replace=False,
                )
                step_omegas = omegas[omega_idx]
            step_result = local_bloch_step(
                cell, radii, train_k[idx], step_omegas, rng,
                force_batch=int(force_batch), eps_reg=EPS_REG,
                update_rule=update_rule, response_mode=response_mode,
                damping_gamma=damping_gamma,
                return_moments=response_conditioned_update,
            )
            if response_conditioned_update:
                cost, grad, edge_moments = step_result
            else:
                cost, grad = step_result
            if not response_conditioned_update:
                grad = postprocess_control_gradient(grad, rng, update_rule)
        if response_conditioned_update:
            moment_u = edge_moments["U"]
            if update_rule == "forward-only":
                # The forward-only trace derivative is sign-definite.  Its
                # response-conditioned lesion uses the same unit local rate
                # bound but contains no paired-response sign information.
                local_drive = np.where(moment_u > 0.0, 1.0, 0.0)
            else:
                moment_v_scaled = response_metric_lambda**2 * edge_moments["V"]
                moment_d_scaled = response_metric_lambda * edge_moments["D"]
                denominator = moment_u + moment_v_scaled
                local_drive = np.zeros_like(moment_d_scaled)
                np.divide(
                    2.0 * moment_d_scaled,
                    denominator,
                    out=local_drive,
                    where=denominator > 0.0,
                )
            if np.max(np.abs(local_drive)) > 1.0 + 5e-12:
                raise AssertionError(
                    "periodic response-conditioned drive exceeded its "
                    "Cauchy--Schwarz bound"
                )
            theta_state = theta_state + response_metric_eta * local_drive
            if response_bound_mode == "clip":
                theta_state = np.clip(theta_state, theta_min, theta_max)
            radii = np.sqrt(np.exp(theta_state) * cell.lengths)
            if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise FloatingPointError(
                    f"nonfinite response-conditioned material at step {step + 1}"
                )
        else:
            grad = grad + local_regularizer_gradient(
                cell, radii, regularizer, inflation_strength,
            )
            if grad_clip is not None:
                grad = np.clip(grad, -float(grad_clip), float(grad_clip))
            radii = np.clip(radii - float(alpha) * grad, clip_min, clip_max)
        cost_history.append(float(cost))
        if (step + 1) % int(eval_every) == 0 or step == 0:
            ff_eval = band_frequencies(cell, radii, eval_k)
            mf = window_metrics(ff_eval, wlo, whi)
            history.append((step + 1, mf["n_in"], mf["gap_ratio"], mf["gap_over_spacing"]))
            print(
                f"step {step + 1:5d}: n_in={mf['n_in']:4d}, "
                f"gap_ratio={mf['gap_ratio']:.3f}, "
                f"bounds={material_diagnostics(cell, radii, radius_min, radius_max)['frac_at_bounds']:.2f}",
                flush=True,
            )
    ff_eval = band_frequencies(cell, radii, eval_k)
    mf = window_metrics(ff_eval, wlo, whi)
    return {
        "radii": radii,
        "wlo": float(wlo),
        "whi": float(whi),
        "omegas": omegas,
        "eval_k": eval_k,
        "train_k": train_k,
        "f0_eval": f0_eval,
        "ff_eval": ff_eval,
        "initial_metrics": m0,
        "final_metrics": mf,
        "history": np.asarray(history, dtype=np.float64),
        "cost_history": np.asarray(cost_history, dtype=np.float64),
        "material": material_diagnostics(cell, radii, radius_min, radius_max),
        "elapsed_s": float(time.perf_counter() - t0),
        "params": {
            "n_steps": int(n_steps),
            "alpha": float(alpha),
            "grad_clip": None if grad_clip is None else float(grad_clip),
            "gradient_clip_mode": "none" if grad_clip is None else "componentwise",
            "frequency_sampling": frequency_sampling,
            "frequencies_per_step": (
                len(omegas) if frequency_sampling == "all" else
                min(frequencies_per_step, len(omegas))
            ),
            "frequency_seed": (
                int(seed) + 918273 if frequency_seed is None else int(frequency_seed)
            ),
            "inflation_strength": float(inflation_strength),
            "radius_min": None if radius_min is None else float(radius_min),
            "radius_max": None if radius_max is None else float(radius_max),
            "n_freq": int(n_freq),
            "k_train_grid": int(k_train_grid),
            "k_eval_grid": int(k_eval_grid),
            "k_batch": int(k_batch),
            "force_batch": int(force_batch),
            "force_covariance": "I/n_dof",
            "expected_force_power": 1.0,
            "train_seed": int(seed),
            "eps_reg": EPS_REG if response_mode == "complex" else 0.0,
            "response_mode": str(response_mode),
            "damping": str(response_mode),
            "damping_gamma_ratio": float(damping_gamma_ratio),
            "damping_gamma": float(damping_gamma),
            "update_rule": str(update_rule),
            "regularizer": str(regularizer),
            "material_update": material_update,
            "response_metric_eta": (
                float(response_metric_eta) if response_conditioned_update else None
            ),
            "response_metric_lambda_ratio": (
                float(response_metric_lambda_ratio) if response_conditioned_update else None
            ),
            "response_metric_lambda": (
                float(response_metric_lambda) if response_conditioned_update else None
            ),
            "response_bound_mode": (
                response_bound_mode if response_conditioned_update else None
            ),
        },
    }


def pack_npz(cell: PeriodicVectorCell, result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        size=np.int64(cell.size),
        seed=np.int64(cell.seed),
        pos=cell.pos.astype(np.float64),
        box=cell.box.astype(np.float64),
        edges=cell.edges.astype(np.int64),
        offsets=cell.offsets.astype(np.int64),
        vectors=cell.vectors.astype(np.float64),
        lengths=cell.lengths.astype(np.float64),
        directions=cell.directions.astype(np.float64),
        radii=result["radii"].astype(np.float64),
        wlo=np.float64(result["wlo"]),
        whi=np.float64(result["whi"]),
        omegas=result["omegas"].astype(np.float64),
        eval_k=result["eval_k"].astype(np.float64),
        train_k=result["train_k"].astype(np.float64),
        f0_eval=result["f0_eval"].astype(np.float64),
        ff_eval=result["ff_eval"].astype(np.float64),
        history=result["history"].astype(np.float64),
        cost_history=result["cost_history"].astype(np.float64),
        eps_reg=np.float64(result["params"].get("eps_reg", 0.0)),
        damping=np.array(result["params"].get("damping", "reciprocal")),
        response_mode=np.array(result["params"].get("response_mode", result["params"].get("damping", "reciprocal"))),
        damping_gamma_ratio=np.float64(
            result["params"].get("damping_gamma_ratio", 0.0)
        ),
        damping_gamma=np.float64(result["params"].get("damping_gamma", 0.0)),
        grad_clip=np.float64(
            np.nan if result["params"].get("grad_clip") is None
            else result["params"]["grad_clip"]
        ),
        gradient_clip_mode=np.array(result["params"].get("gradient_clip_mode", "unknown")),
        frequency_sampling=np.array(result["params"].get("frequency_sampling", "all")),
        frequencies_per_step=np.int64(result["params"].get("frequencies_per_step", len(result["omegas"]))),
        frequency_seed=np.int64(result["params"].get("frequency_seed", -1)),
        n_steps=np.int64(result["params"].get("n_steps", -1)),
        inflation_strength=np.float64(result["params"].get("inflation_strength", 0.0)),
        regularizer=np.array(result["params"].get("regularizer", "none")),
        update_rule=np.array(result["params"].get("update_rule", "adjoint")),
        material_update=np.array(result["params"].get("material_update", "radius_euler")),
        response_metric_eta=np.float64(
            np.nan if result["params"].get("response_metric_eta") is None
            else result["params"]["response_metric_eta"]
        ),
        response_metric_lambda_ratio=np.float64(
            np.nan if result["params"].get("response_metric_lambda_ratio") is None
            else result["params"]["response_metric_lambda_ratio"]
        ),
        response_bound_mode=np.array(
            result["params"].get("response_bound_mode") or "none"
        ),
    )


def load_npz_result(path: Path) -> tuple[PeriodicVectorCell, dict]:
    with np.load(path, allow_pickle=True) as z:
        cell = PeriodicVectorCell(
            size=int(z["size"]),
            seed=int(z["seed"]),
            pos=z["pos"],
            box=z["box"],
            edges=z["edges"],
            offsets=z["offsets"],
            vectors=z["vectors"],
            lengths=z["lengths"],
            directions=z["directions"],
        )
        result = {
            "radii": z["radii"],
            "wlo": float(z["wlo"]),
            "whi": float(z["whi"]),
            "omegas": z["omegas"],
            "eval_k": z["eval_k"],
            "train_k": z["train_k"],
            "f0_eval": z["f0_eval"],
            "ff_eval": z["ff_eval"],
            "history": z["history"],
            "cost_history": z["cost_history"],
        }
        result["params"] = {
            "eps_reg": float(z["eps_reg"]) if "eps_reg" in z.files else EPS_REG,
            "damping": str(z["damping"]) if "damping" in z.files else "complex",
            "response_mode": str(z["response_mode"]) if "response_mode" in z.files else (
                str(z["damping"]) if "damping" in z.files else "complex"
            ),
            "damping_gamma_ratio": (
                float(z["damping_gamma_ratio"])
                if "damping_gamma_ratio" in z.files else 0.0
            ),
            "damping_gamma": (
                float(z["damping_gamma"])
                if "damping_gamma" in z.files else 0.0
            ),
            "grad_clip": (
                None if "grad_clip" not in z.files or np.isnan(float(z["grad_clip"]))
                else float(z["grad_clip"])
            ),
            "gradient_clip_mode": str(z["gradient_clip_mode"]) if "gradient_clip_mode" in z.files else "unknown",
            "frequency_sampling": str(z["frequency_sampling"]) if "frequency_sampling" in z.files else "all",
            "frequencies_per_step": int(z["frequencies_per_step"]) if "frequencies_per_step" in z.files else len(z["omegas"]),
            "frequency_seed": int(z["frequency_seed"]) if "frequency_seed" in z.files else -1,
            "n_steps": int(z["n_steps"]) if "n_steps" in z.files else -1,
            "inflation_strength": float(z["inflation_strength"]) if "inflation_strength" in z.files else float("nan"),
            "regularizer": str(z["regularizer"]) if "regularizer" in z.files else "unknown",
            "update_rule": str(z["update_rule"]) if "update_rule" in z.files else "unknown",
            "material_update": str(z["material_update"]) if "material_update" in z.files else "unknown",
            "response_metric_eta": (
                None if "response_metric_eta" not in z.files or np.isnan(float(z["response_metric_eta"]))
                else float(z["response_metric_eta"])
            ),
            "response_metric_lambda_ratio": (
                None if "response_metric_lambda_ratio" not in z.files or np.isnan(float(z["response_metric_lambda_ratio"]))
                else float(z["response_metric_lambda_ratio"])
            ),
            "response_bound_mode": str(z["response_bound_mode"]) if "response_bound_mode" in z.files else "unknown",
        }
        result["initial_metrics"] = window_metrics(result["f0_eval"], result["wlo"], result["whi"])
        result["final_metrics"] = window_metrics(result["ff_eval"], result["wlo"], result["whi"])
        result["material"] = material_diagnostics(cell, result["radii"])
        return cell, result


def draw_periodic_vector_cell(
    ax: plt.Axes,
    cell: PeriodicVectorCell,
    radii: np.ndarray,
) -> LineCollection:
    """Draw every periodic edge, using paired clipped stubs at cell boundaries."""
    widths = 0.5 + 1.1 * (radii - R_MIN) / (R_MAX - R_MIN)
    segments: list[list[np.ndarray]] = []
    values: list[float] = []
    line_widths: list[float] = []
    for e, (i, j) in enumerate(cell.edges):
        if np.any(cell.offsets[e] != 0):
            edge_segments = [
                [cell.pos[i], cell.pos[i] + cell.vectors[e]],
                [cell.pos[j], cell.pos[j] - cell.vectors[e]],
            ]
        else:
            edge_segments = [[cell.pos[i], cell.pos[j]]]
        for segment in edge_segments:
            segments.append(segment)
            values.append(float(radii[e]))
            line_widths.append(float(widths[e]))

    collection = LineCollection(
        segments,
        array=np.asarray(values),
        cmap="viridis",
        linewidths=line_widths,
        alpha=0.95,
    )
    collection.set_clim(R_MIN, R_MAX)
    ax.add_collection(collection)
    ax.scatter(cell.pos[:, 0], cell.pos[:, 1], s=4.0, color=ps.DARK, zorder=3)
    frame = Rectangle(
        (0.0, 0.0), cell.box[0], cell.box[1],
        facecolor="none", edgecolor=ps.DARK, lw=0.6, zorder=4,
    )
    ax.add_patch(frame)
    collection.set_clip_path(frame)
    ax.set_xlim(-0.02 * cell.box[0], 1.02 * cell.box[0])
    ax.set_ylim(-0.02 * cell.box[1], 1.02 * cell.box[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return collection


def plot_validation(cell: PeriodicVectorCell, result: dict, out_base: Path) -> None:
    ps.style()
    path_k, path_x, labels, ticks = k_path(n_segment=32)
    f0_path = band_frequencies(cell, np.full(cell.n_edges, R_INIT), path_k)
    ff_path = band_frequencies(cell, result["radii"], path_k)
    wlo, whi = result["wlo"], result["whi"]

    # Compact 2x2 layout: (a) unit cell, (b) merged initial|learned bands,
    # (c) training, (d) gap-vs-size.  The BZ-sampled DOS panel is dropped --
    # the band structure and training curve already carry the gap/clearing.
    fig = plt.figure(figsize=(ps.TEXT_W, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.08], height_ratios=[1.0, 0.92],
    )
    ax_cell = fig.add_subplot(gs[0, 0])
    gs_bands = gs[0, 1].subgridspec(1, 2, wspace=0.07)
    ax_bi = fig.add_subplot(gs_bands[0, 0])
    ax_bf = fig.add_subplot(gs_bands[0, 1], sharey=ax_bi)
    ax_train = fig.add_subplot(gs[1, 0])
    ax_size = fig.add_subplot(gs[1, 1])

    # (a) Learned unit cell. Periodic crossings appear as paired, clipped
    # boundary stubs, so all physical springs are represented exactly once.
    radii = result["radii"]
    cell_lines = draw_periodic_vector_cell(ax_cell, cell, radii)
    cbar = fig.colorbar(
        cell_lines, ax=ax_cell, orientation="vertical",
        fraction=0.065, pad=0.035, shrink=0.88, aspect=16,
    )
    cbar.set_label(r"radius $r_e$", labelpad=2)
    cbar.set_ticks([R_MIN, R_INIT, R_MAX])
    cbar.ax.tick_params(length=2)

    # (b) merged Bloch bands: initial (grey) | learned (blue), shared omega
    for band in range(cell.n_dof):
        ax_bi.plot(path_x, f0_path[:, band], color=ps.GRAY, lw=0.5, alpha=0.68)
        ax_bf.plot(path_x, ff_path[:, band], color=ps.BLUE, lw=0.5, alpha=0.78)
    for ax in (ax_bi, ax_bf):
        ps.shade_window(ax, wlo, whi, axis="y")
        for guide in ticks[1:-1]:
            ax.axvline(guide, color=ps.GRAY, lw=0.45, alpha=0.65, zorder=0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(path_x[0], path_x[-1])
    ax_bi.set_ylabel(r"$\omega$")
    ax_bf.tick_params(labelleft=False)
    ax_bi.set_xlabel("Bloch path")
    ax_bi.xaxis.set_label_coords(1.04, -0.17)
    ps.panel_tag(ax_bi, "initial", color=ps.GRAY_DARK, fontsize=ps.TICK_SIZE)
    ps.panel_tag(ax_bf, "learned", color=ps.BLUE, fontsize=ps.TICK_SIZE)

    # (c) Ensemble trajectory over the full reported 3000-step experiment.
    # A small negative lower margin keeps the sustained zero plateau visible.
    ens_path = DATA_DIR / "figS_training_ensemble.npz"
    if ens_path.exists():
        ens = np.load(ens_path)
        es, em, esd = ens["steps"], ens["mean"], ens["std"]
        ax_train.fill_between(es, np.maximum(em - esd, 0.0), em + esd,
                              color=ps.BLUE, alpha=0.2, lw=0)
        ax_train.plot(es, em, color=ps.BLUE, marker="o", ms=2.5, lw=1.1)
        ymax = float(np.max(em + esd))
        ax_train.set_xlim(0, float(es[-1]))
        ax_train.set_ylim(-0.025 * ymax, 1.05 * ymax)
        ax_train.set_yticks(np.arange(0.0, ymax + 1.0, 100.0))
    else:
        hist = result["history"]
        if len(hist):
            ax_train.plot(hist[:, 0], hist[:, 1], color=ps.BLUE,
                          marker="o", ms=2.5, lw=1.0)
            ymax = max(float(np.max(hist[:, 1])), 1.0)
            ax_train.set_xlim(0, float(hist[-1, 0]))
            ax_train.set_ylim(-0.025 * ymax, 1.05 * ymax)
    ax_train.set_xlabel("training step")
    ax_train.set_ylabel("BZ-grid modes in window")

    # (d) gap vs unit-cell size (kept)
    size_scan_path = DATA_DIR / "size_scan_summary.json"
    if size_scan_path.exists():
        ss = sorted(json.loads(size_scan_path.read_text()), key=lambda r: r["size"])
        xs = [r["size"] for r in ss]
        meds = [r["median"] for r in ss]
        lo = [m - r["min"] for m, r in zip(meds, ss)]
        hi = [r["max"] - m for m, r in zip(meds, ss)]
        ax_size.errorbar(xs, meds, yerr=[lo, hi], fmt="o", color=ps.BLUE,
                         ms=4, capsize=3, capthick=0.9, ecolor=ps.BLUE,
                         elinewidth=0.9)
        ax_size.set_xticks(xs)
        ax_size.set_xticklabels([f"${s}\\times{s}$" for s in xs])
        ax_size.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
        ax_size.set_ylim(0, 0.70)
        ax_size.set_xlabel("unit cell")
        ax_size.set_ylabel(r"$\Delta\omega/\omega_{\rm mid}$")
        ps.panel_tag(ax_size, "all 10 clear", loc="upper right",
                     fontsize=ps.TICK_SIZE)
    else:
        mat = result["material"]
        ax_size.bar(["mat.", "stiff.", "bounds"],
                    [mat["material_ratio"], mat["mean_stiffness_ratio"], mat["frac_at_bounds"]],
                    color=[ps.GREEN, ps.PURPLE, ps.BLUE])
        ax_size.axhline(1.0, color=ps.DARK, lw=0.7)

    for label, ax in zip("abcd", (ax_cell, ax_bi, ax_train, ax_size)):
        ps.panel_label(ax, label)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    ps.savefig(fig, out_base.parent, out_base.name)


def run_single(args: argparse.Namespace) -> dict:
    ensure_dirs()
    cell = make_periodic_vector_cell(args.size, args.net_seed, args.topology)
    radii0 = np.full(cell.n_edges, R_INIT)
    initial_eval = band_frequencies(cell, radii0, k_grid(args.k_eval_grid))
    wlo, whi = target_window_from_percentiles(
        initial_eval, args.center_percentile, args.width_percentile,
    )
    print(
        f"cell: size={cell.size}, N={cell.n_nodes}, edges={cell.n_edges}, "
        f"target=[{wlo:.4f}, {whi:.4f}]",
        flush=True,
    )
    result = train_periodic_vector(
        cell, wlo, whi,
        n_steps=args.steps,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        inflation_strength=args.inflation,
        n_freq=args.n_freq,
        k_train_grid=args.k_train_grid,
        k_eval_grid=args.k_eval_grid,
        k_batch=args.k_batch,
        force_batch=args.force_batch,
        seed=args.train_seed,
        eval_every=args.eval_every,
        response_mode=args.response_mode,
        damping_gamma_ratio=args.damping_gamma_ratio,
        regularizer=args.regularizer,
        material_update=args.material_update,
        response_metric_eta=args.response_metric_eta,
        response_metric_lambda_ratio=args.response_metric_lambda_ratio,
        response_bound_mode=args.response_bound_mode,
        frequency_sampling=args.frequency_sampling,
        frequencies_per_step=args.frequencies_per_step,
    )
    stem = (
        f"vector_periodic_s{args.size}_net{args.net_seed}_train{args.train_seed}"
        f"_c{args.center_percentile:g}_w{args.width_percentile:g}"
    )
    npz_path = DATA_DIR / f"{stem}.npz"
    pack_npz(cell, result, npz_path)
    fig_base = FIG_DIR / "figS_vector_periodic_validation"
    plot_validation(cell, result, fig_base)
    dense_k = k_grid(args.k_dense_grid)
    dense_initial = band_frequencies(cell, np.full(cell.n_edges, R_INIT), dense_k)
    dense_final = band_frequencies(cell, result["radii"], dense_k)
    eval_initial_gap = best_gap_near_window(
        result["f0_eval"], result["wlo"], result["whi"], result["eval_k"],
    )
    eval_final_gap = best_gap_near_window(
        result["ff_eval"], result["wlo"], result["whi"], result["eval_k"],
    )
    dense_initial_gap = best_gap_near_window(
        dense_initial, result["wlo"], result["whi"], dense_k,
    )
    dense_final_gap = best_gap_near_window(
        dense_final, result["wlo"], result["whi"], dense_k,
    )
    band_gap = dict(dense_final_gap)
    band_gap.update({
        "source": "dense_grid",
        "k_grid": int(args.k_dense_grid),
        "eval_grid_band_index": int(eval_final_gap["band_index"]),
    })
    payload = {
        "npz": str(npz_path.relative_to(REPO_ROOT)),
        "figure_pdf": str(fig_base.with_suffix(".pdf").relative_to(REPO_ROOT)),
        "figure_png": str(fig_base.with_suffix(".png").relative_to(REPO_ROOT)),
        "size": cell.size,
        "net_seed": cell.seed,
        "train_seed": int(args.train_seed),
        "n_nodes": cell.n_nodes,
        "n_dof": cell.n_dof,
        "n_edges": cell.n_edges,
        "target": {"wlo": result["wlo"], "whi": result["whi"]},
        "initial": result["initial_metrics"],
        "final": result["final_metrics"],
        "band_gap": band_gap,
        "eval_band_gap": {
            "initial": eval_initial_gap,
            "final": eval_final_gap,
        },
        "dense_grid_check": {
            "k_grid": int(args.k_dense_grid),
            "initial": window_metrics(dense_initial, result["wlo"], result["whi"]),
            "final": window_metrics(dense_final, result["wlo"], result["whi"]),
            "initial_band_gap": dense_initial_gap,
            "final_band_gap": dense_final_gap,
        },
        "material": result["material"],
        "params": result["params"],
        "success": bool(result["final_metrics"]["n_in"] == 0),
    }
    table = write_single_validation_table(payload, FIG_DIR / "table_vector_periodic_validation.tex")
    payload["table"] = str(table.relative_to(REPO_ROOT))
    save_json(DATA_DIR / f"{stem}.json", payload)
    save_json(DATA_DIR / "latest_vector_periodic_validation.json", payload)
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    return payload


def run_search(args: argparse.Namespace) -> None:
    ensure_dirs()
    rows = []
    out_csv = DATA_DIR / "search_summary.csv"
    t0 = time.perf_counter()
    for size in args.search_sizes:
        for net_seed in range(args.search_net_seeds):
            cell = make_periodic_vector_cell(size, net_seed, args.topology)
            initial_eval = band_frequencies(cell, np.full(cell.n_edges, R_INIT), k_grid(args.k_eval_grid))
            for center in args.search_centers:
                for width in args.search_widths:
                    try:
                        wlo, whi = target_window_from_percentiles(initial_eval, center, width)
                    except ValueError:
                        continue
                    for train_seed in range(args.search_train_seeds):
                        print(
                            f"search size={size} net={net_seed} train={train_seed} "
                            f"center={center} width={width}",
                            flush=True,
                        )
                        result = train_periodic_vector(
                            cell, wlo, whi,
                            n_steps=args.steps,
                            alpha=args.alpha,
                            grad_clip=args.grad_clip,
                            inflation_strength=args.inflation,
                            n_freq=args.n_freq,
                            k_train_grid=args.k_train_grid,
                            k_eval_grid=args.k_eval_grid,
                            k_batch=args.k_batch,
                            force_batch=args.force_batch,
                            seed=train_seed,
                            eval_every=max(args.eval_every, args.steps // 4),
                            response_mode=args.response_mode,
                            damping_gamma_ratio=args.damping_gamma_ratio,
                            regularizer=args.regularizer,
                            material_update=args.material_update,
                            response_metric_eta=args.response_metric_eta,
                            response_metric_lambda_ratio=args.response_metric_lambda_ratio,
                            response_bound_mode=args.response_bound_mode,
                            frequency_sampling=args.frequency_sampling,
                            frequencies_per_step=args.frequencies_per_step,
                        )
                        row = {
                            "size": size,
                            "net_seed": net_seed,
                            "train_seed": train_seed,
                            "center_percentile": center,
                            "width_percentile": width,
                            "wlo": wlo,
                            "whi": whi,
                            "initial_n_in": result["initial_metrics"]["n_in"],
                            "final_n_in": result["final_metrics"]["n_in"],
                            "gap_ratio": result["final_metrics"]["gap_ratio"],
                            "gap_over_spacing": result["final_metrics"]["gap_over_spacing"],
                            "gap_over_target_width": result["final_metrics"]["gap_over_target_width"],
                            "material_ratio": result["material"]["material_ratio"],
                            "frac_at_bounds": result["material"]["frac_at_bounds"],
                            "success": int(result["final_metrics"]["n_in"] == 0),
                        }
                        rows.append(row)
                        with open(out_csv, "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                            writer.writeheader()
                            writer.writerows(rows)
                        if row["success"] and not args.no_stop_on_success:
                            print(f"Found success: {row}", flush=True)
                            single_args = argparse.Namespace(**vars(args))
                            single_args.size = size
                            single_args.net_seed = net_seed
                            single_args.train_seed = train_seed
                            single_args.center_percentile = center
                            single_args.width_percentile = width
                            run_single(single_args)
                            print(f"search elapsed {time.perf_counter() - t0:.1f}s", flush=True)
                            return
    print(f"search complete, wrote {out_csv}", flush=True)


def band_label(gap: dict) -> str:
    return f"{int(gap['lower_band_1based'])}-{int(gap['upper_band_1based'])}"


def write_bz_convergence_table(records: list[dict]) -> Path:
    path = FIG_DIR / "table_vector_bz_convergence.tex"
    lines = [
        r"\begin{tabular}{rcrrrrr}",
        r"\toprule",
        r"$N_k$ & Bands & $N_{\rm in}$ & $\max_{\bm k}\omega_m$ & $\min_{\bm k}\omega_{m+1}$ & $\Delta\omega$ & $\Delta\omega/\omega_{\rm mid}^{\rm gap}$ \\",
        r"\midrule",
    ]
    for rec in records:
        gap = rec["band_gap"]
        lines.append(
            f"{int(rec['grid'])} & {band_label(gap)} & {int(rec['final']['n_in'])} & "
            f"{gap['lower_edge']:.4f} & {gap['upper_edge']:.4f} & "
            f"{gap['gap']:.4f} & {gap['normalized_gap']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    return path


def write_size_scan_table(size_scan: list[dict]) -> Path:
    """Render the retained finite-cell ensemble summary as a TeX table."""
    lines = [
        r"\begin{tabular}{lrrccc}",
        r"\toprule",
        r"unit-cell size $N_{\rm side}\times N_{\rm side}$ & $N_{\rm nodes}$ & cells & success & median $\Delta\omega/\omega_{\rm mid}^{\rm gap}$ & range of $\Delta\omega/\omega_{\rm mid}^{\rm gap}$ \\",
        r"\midrule",
    ]
    for row in size_scan:
        lines.append(
            f"${row['size']}\\times{row['size']}$ & {row['nodes']} & {row['n']} & "
            f"{row['success']}/{row['n']} & {row['median']:.3f} & "
            f"{row['min']:.2f}--{row['max']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path = FIG_DIR / "table_vector_size_scan.tex"
    path.write_text("\n".join(lines))
    return path


def render_active_tables() -> list[Path]:
    """Render the three periodic-vector tables cited by the supplement."""
    ensure_dirs()
    bz = load_output_json("bz_convergence.json")
    size_scan = load_output_json("size_scan_summary.json")
    controls = load_output_json("vector_controls.json")
    paths = [
        write_bz_convergence_table(bz["records"]),
        write_size_scan_table(size_scan),
        write_vector_controls_table(
            controls["records"], FIG_DIR / "table_vector_controls.tex",
        ),
    ]
    print("rendered active vector tables:", flush=True)
    for path in paths:
        print(f"  {path.relative_to(REPO_ROOT)}", flush=True)
    return paths


def run_bz_convergence(args: argparse.Namespace) -> dict:
    if args.input is None:
        raise SystemExit("--mode bz-convergence requires --input")
    ensure_dirs()
    cell, result = load_npz_result(args.input)
    wlo, whi = result["wlo"], result["whi"]
    records = []
    for grid in args.bz_grids:
        t0 = time.perf_counter()
        kpts = k_grid(int(grid))
        initial = band_frequencies(cell, np.full(cell.n_edges, R_INIT), kpts)
        final = band_frequencies(cell, result["radii"], kpts)
        initial_metrics = window_metrics(initial, wlo, whi)
        final_metrics = window_metrics(final, wlo, whi)
        initial_gap = best_gap_near_window(initial, wlo, whi, kpts)
        final_gap = best_gap_near_window(final, wlo, whi, kpts)
        record = {
            "grid": int(grid),
            "n_k": int(len(kpts)),
            "initial": initial_metrics,
            "final": final_metrics,
            "initial_band_gap": initial_gap,
            "band_gap": final_gap,
            "elapsed_s": float(time.perf_counter() - t0),
        }
        records.append(record)
        print(
            f"BZ {grid}x{grid}: band {band_label(final_gap)}, "
            f"gap={final_gap['gap']:.6f}, n_in={final_metrics['n_in']}",
            flush=True,
        )

    payload = {
        "input": str(args.input),
        "target": {"wlo": wlo, "whi": whi},
        "records": records,
        "all_positive": bool(all(r["band_gap"]["gap"] > 0.0 for r in records)),
        "band_indices": [int(r["band_gap"]["band_index"]) for r in records],
        "table": str(write_bz_convergence_table(records).relative_to(REPO_ROOT)),
    }
    out_json = DATA_DIR / "bz_convergence.json"
    save_json(out_json, payload)

    np.savez_compressed(
        DATA_DIR / "bz_convergence.npz",
        grids=np.asarray([r["grid"] for r in records], dtype=np.int64),
        n_k=np.asarray([r["n_k"] for r in records], dtype=np.int64),
        initial_n_in=np.asarray([r["initial"]["n_in"] for r in records], dtype=np.int64),
        final_n_in=np.asarray([r["final"]["n_in"] for r in records], dtype=np.int64),
        band_index=np.asarray([r["band_gap"]["band_index"] for r in records], dtype=np.int64),
        lower_edge=np.asarray([r["band_gap"]["lower_edge"] for r in records], dtype=np.float64),
        upper_edge=np.asarray([r["band_gap"]["upper_edge"] for r in records], dtype=np.float64),
        gap=np.asarray([r["band_gap"]["gap"] for r in records], dtype=np.float64),
        normalized_gap=np.asarray([r["band_gap"]["normalized_gap"] for r in records], dtype=np.float64),
        lower_k=np.asarray([r["band_gap"]["lower_k"] for r in records], dtype=np.float64),
        upper_k=np.asarray([r["band_gap"]["upper_k"] for r in records], dtype=np.float64),
        target=np.asarray([wlo, whi], dtype=np.float64),
    )
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True), flush=True)
    return payload


def evaluate_radii(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    wlo: float,
    whi: float,
    k_dense_grid: int,
) -> dict:
    kpts = k_grid(int(k_dense_grid))
    freqs = band_frequencies(cell, radii, kpts)
    metrics = window_metrics(freqs, wlo, whi)
    band_gap = best_gap_near_window(freqs, wlo, whi, kpts)
    success = bool(
        metrics["n_in"] == 0
        and band_gap["gap"] > 0.0
        and band_gap["contains_target_window"]
    )
    return {
        "k_dense_grid": int(k_dense_grid),
        "metrics": metrics,
        "band_gap": band_gap,
        "material": material_diagnostics(cell, radii),
        "success": success,
    }


def scalar_field(records: list[dict], field: str) -> list[float]:
    vals = []
    for rec in records:
        cur = rec
        for part in field.split("."):
            cur = cur[part]
        vals.append(float(cur))
    return vals


def summarize_records(records: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(str(rec[group_key]), []).append(rec)
    summary = []
    for key, group in groups.items():
        success = np.asarray([bool(r["success"]) for r in group], dtype=bool)
        summary.append({
            group_key: key,
            "n": int(len(group)),
            "success_rate": float(np.mean(success)) if len(success) else float("nan"),
            "median_final_n_in": float(np.median(scalar_field(group, "metrics.n_in"))),
            "median_gap": float(np.median(scalar_field(group, "band_gap.gap"))),
            "median_normalized_gap": float(np.median(scalar_field(group, "band_gap.normalized_gap"))),
            "median_material_ratio": float(np.median(scalar_field(group, "material.material_ratio"))),
            "median_bound_fraction": float(np.median(scalar_field(group, "material.frac_at_bounds"))),
        })
    return summary


def latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def write_summary_table(
    records: list[dict],
    group_key: str,
    path: Path,
    first_col_label: str,
    count_label: str,
) -> Path:
    summary = summarize_records(records, group_key)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        rf"{first_col_label} & {count_label} & success & median $N_{{\rm in}}$ & median $\Delta\omega$ & median $\Delta\omega/\omega_{{\rm mid}}$ \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{latex_escape(row[group_key])} & {row['n']} & {row['success_rate']:.2f} & "
            f"{row['median_final_n_in']:.0f} & {row['median_gap']:.4f} & "
            f"{row['median_normalized_gap']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    return path


def write_vector_controls_table(records: list[dict], path: Path) -> Path:
    """Render target-gap metrics only for arms that actually succeed."""
    order = [
        "paired-response", "paired-no-bounds", "forward-only",
        "shuffled-response", "random-matched", "uniform-material",
        "uniform-stiffness",
    ]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"control & cells & success & median $N_{\rm in}$ & median $\Delta\omega$ & median $\Delta\omega/\omega_{\rm mid}^{\rm gap}$ \\",
        r"\midrule",
    ]
    for label in order:
        group = [record for record in records if str(record["control"]) == label]
        if not group:
            continue
        successful = [record for record in group if bool(record["success"])]
        median_n_in = float(np.median(scalar_field(group, "metrics.n_in")))
        if successful:
            gap = f"{float(np.median(scalar_field(successful, 'band_gap.gap'))):.4f}"
            relative = (
                f"{float(np.median(scalar_field(successful, 'band_gap.normalized_gap'))):.3f}"
            )
        else:
            gap = "--"
            relative = "--"
        lines.append(
            f"{latex_escape(label)} & {len(group)} & "
            f"{len(successful)}/{len(group)} & {median_n_in:.0f} & "
            f"{gap} & {relative} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    return path


def write_single_validation_table(payload: dict, path: Path) -> Path:
    mat = payload["material"]
    rows = [
        (
            "exemplar",
            "9\\times9",
            payload["initial"]["n_in"],
            payload["final"]["n_in"],
            payload["final"]["gap_over_target_width"],
        ),
        (
            "dense check",
            f"{payload['dense_grid_check']['k_grid']}\\times{payload['dense_grid_check']['k_grid']}",
            payload["dense_grid_check"]["initial"]["n_in"],
            payload["dense_grid_check"]["final"]["n_in"],
            payload["dense_grid_check"]["final"]["gap_over_target_width"],
        ),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"case & grid & initial $N_{\rm in}$ & final $N_{\rm in}$ & gap / target width & material ratio & bound edges \\",
        r"\midrule",
    ]
    for label, grid, n0, nf, gap_over_target in rows:
        lines.append(
            f"{label} & ${grid}$ & {int(n0)} & {int(nf)} & "
            f"{float(gap_over_target):.2f} & {float(mat['material_ratio']):.3f} & "
            f"{100.0 * float(mat['frac_at_bounds']):.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    return path


def save_records_npz(path: Path, records: list[dict], group_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        records_json=np.asarray(json.dumps(json_ready(records))),
        group=np.asarray([str(r[group_key]) for r in records]),
        size=np.asarray([r["size"] for r in records], dtype=np.int64),
        net_seed=np.asarray([r["net_seed"] for r in records], dtype=np.int64),
        success=np.asarray([int(r["success"]) for r in records], dtype=np.int64),
        final_n_in=np.asarray([r["metrics"]["n_in"] for r in records], dtype=np.int64),
        gap=np.asarray([r["band_gap"]["gap"] for r in records], dtype=np.float64),
        normalized_gap=np.asarray([r["band_gap"]["normalized_gap"] for r in records], dtype=np.float64),
        material_ratio=np.asarray([r["material"]["material_ratio"] for r in records], dtype=np.float64),
        frac_at_bounds=np.asarray([r["material"]["frac_at_bounds"] for r in records], dtype=np.float64),
    )


def make_run_record(
    *,
    cell: PeriodicVectorCell,
    wlo: float,
    whi: float,
    radii: np.ndarray,
    k_dense_grid: int,
    label_key: str,
    label: str,
    train_seed: int,
    elapsed_s: float = 0.0,
    params: dict | None = None,
) -> dict:
    evaluation = evaluate_radii(cell, radii, wlo, whi, k_dense_grid)
    record = {
        label_key: label,
        "size": cell.size,
        "net_seed": cell.seed,
        "train_seed": int(train_seed),
        "n_nodes": cell.n_nodes,
        "n_edges": cell.n_edges,
        "target": {"wlo": float(wlo), "whi": float(whi)},
        "elapsed_s": float(elapsed_s),
        "params": params or {},
    }
    record.update(evaluation)
    return record


def write_ensemble_derived_outputs(
    records: list[dict],
    histories: dict[int, list[np.ndarray]],
) -> dict:
    """Write the size scan and training curve consumed by the paper figures."""
    size_scan = []
    for size in sorted({int(r["size"]) for r in records}):
        group = [r for r in records if int(r["size"]) == size]
        gaps = np.asarray(
            [r["band_gap"]["normalized_gap"] for r in group], dtype=np.float64,
        )
        size_scan.append({
            "size": size,
            "nodes": int(group[0]["n_nodes"]),
            "n": len(group),
            "success": int(sum(bool(r["success"]) for r in group)),
            "median": float(np.median(gaps)),
            "min": float(np.min(gaps)),
            "max": float(np.max(gaps)),
        })

    size_scan_path = DATA_DIR / "size_scan_summary.json"
    save_json(size_scan_path, size_scan)
    size_table_path = write_size_scan_table(size_scan)

    training_path = DATA_DIR / "figS_training_ensemble.npz"
    training_size = min(histories)
    curves = histories[training_size]
    if not curves:
        raise RuntimeError("ensemble produced no training histories")
    steps = curves[0][:, 0]
    if any(not np.array_equal(curve[:, 0], steps) for curve in curves[1:]):
        raise RuntimeError("ensemble histories use inconsistent evaluation steps")
    counts = np.stack([curve[:, 1] for curve in curves])
    np.savez_compressed(
        training_path,
        steps=steps,
        mean=np.mean(counts, axis=0),
        std=np.std(counts, axis=0),
        counts=counts,
        size=np.int64(training_size),
    )
    return {
        "size_scan": str(size_scan_path.relative_to(REPO_ROOT)),
        "size_table": str(size_table_path.relative_to(REPO_ROOT)),
        "training_ensemble": str(training_path.relative_to(REPO_ROOT)),
    }


def run_ensemble(args: argparse.Namespace) -> dict:
    ensure_dirs()
    jobs = [
        (int(size), int(net_seed), vars(args))
        for size in args.ensemble_sizes
        for net_seed in range(int(args.ensemble_net_seeds))
    ]
    n_workers = max(1, min(int(args.workers), len(jobs)))
    print(f"ensemble: {len(jobs)} cells with {n_workers} workers", flush=True)
    if n_workers == 1:
        completed = [_ensemble_cell_job(job) for job in jobs]
    else:
        with cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            completed = list(pool.map(_ensemble_cell_job, jobs, chunksize=1))
    records = [item[0] for item in completed]
    histories: dict[int, list[np.ndarray]] = {
        int(size): [] for size in args.ensemble_sizes
    }
    for record, history in completed:
        histories[int(record["size"])].append(history)
        print(
            f"ensemble size={record['size']} net={record['net_seed']}: "
            f"success={record['success']} gap={record['band_gap']['gap']:.4f} "
            f"n_in={record['metrics']['n_in']}",
            flush=True,
        )

    primary_size = min(int(size) for size in args.ensemble_sizes)
    primary_records = [r for r in records if int(r["size"]) == primary_size]
    table = write_summary_table(
        primary_records, "size_group", FIG_DIR / "table_vector_ensemble.tex", "size", "cells",
    )
    out_json = DATA_DIR / "vector_gap_ensemble.json"
    payload = {
        "records": records,
        "summary": summarize_records(records, "size_group"),
        "table": str(table.relative_to(REPO_ROOT)),
        "derived_outputs": write_ensemble_derived_outputs(records, histories),
    }
    save_json(out_json, payload)
    save_records_npz(DATA_DIR / "vector_gap_ensemble.npz", records, "size_group")
    print(json.dumps(json_ready(payload["summary"]), indent=2, sort_keys=True), flush=True)
    return payload


def _ensemble_cell_job(job: tuple[int, int, dict]) -> tuple[dict, np.ndarray]:
    size, net_seed, config = job
    args = argparse.Namespace(**config)
    cell = make_periodic_vector_cell(size, net_seed, args.topology)
    initial_eval = band_frequencies(
        cell, np.full(cell.n_edges, R_INIT), k_grid(args.k_eval_grid),
    )
    wlo, whi = target_window_from_percentiles(
        initial_eval, args.center_percentile, args.width_percentile,
    )
    result = train_periodic_vector(
        cell, wlo, whi,
        n_steps=args.steps,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        inflation_strength=args.inflation,
        n_freq=args.n_freq,
        k_train_grid=args.k_train_grid,
        k_eval_grid=args.k_eval_grid,
        k_batch=args.k_batch,
        force_batch=args.force_batch,
        seed=args.train_seed,
        eval_every=args.eval_every,
        update_rule="adjoint",
        regularizer=args.regularizer,
        response_mode=args.response_mode,
        damping_gamma_ratio=args.damping_gamma_ratio,
        material_update=args.material_update,
        response_metric_eta=args.response_metric_eta,
        response_metric_lambda_ratio=args.response_metric_lambda_ratio,
        response_bound_mode=args.response_bound_mode,
        frequency_sampling=args.frequency_sampling,
        frequencies_per_step=args.frequencies_per_step,
    )
    record = make_run_record(
        cell=cell,
        wlo=wlo,
        whi=whi,
        radii=result["radii"],
        k_dense_grid=args.k_dense_grid,
        label_key="size_group",
        label=str(size),
        train_seed=args.train_seed,
        elapsed_s=result["elapsed_s"],
        params=result["params"],
    )
    return record, result["history"]


CONTROL_SPECS = [
    ("paired-response", "adjoint", "none", "clip"),
    ("paired-no-bounds", "adjoint", "none", "none"),
    ("forward-only", "forward-only", "none", "clip"),
    ("shuffled-response", "shuffled-adjoint", "none", "clip"),
    ("random-matched", "random-matched", "none", "clip"),
]


def plot_vector_controls(records: list[dict]) -> Path:
    ps.style()
    summary = summarize_records(records, "control")
    labels = [s["control"] for s in summary]
    success = np.asarray([s["success_rate"] for s in summary], dtype=float)
    norm_gap = np.asarray([s["median_normalized_gap"] for s in summary], dtype=float)
    sample_sizes = np.asarray([s["n"] for s in summary], dtype=int)
    y = np.arange(len(labels))
    colors = [
        ps.BLUE if label == "paired-response"
        else ps.GREEN if label == "paired-no-bounds"
        else ps.GRAY_DARK
        for label in labels
    ]

    # Horizontal shared-category panels avoid repeating seven long rotated
    # labels. Dots keep zero-success controls visible; the log gap axis exposes
    # the roughly two-order-of-magnitude separation without a broken axis.
    fig, axes = plt.subplots(
        1, 2, figsize=(ps.TEXT_W, 2.5), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.22]}, constrained_layout=True,
    )
    axes[0].hlines(y, 0.0, success, color=ps.GRAY, lw=0.8, zorder=1)
    axes[0].scatter(success, y, color=colors, s=25, zorder=2)
    axes[0].set_xlim(-0.04, 1.14)
    axes[0].set_xlabel("success rate")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    for rate, n, yi in zip(success, sample_sizes, y):
        count = int(round(float(rate) * int(n)))
        axes[0].annotate(
            f"{count}/{int(n)}", xy=(rate, yi), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=ps.TICK_SIZE,
        )

    if np.any(norm_gap <= 0.0):
        raise ValueError("vector-control normalized gaps must be positive for log plotting")
    gap_floor = 0.72 * float(np.min(norm_gap))
    axes[1].hlines(y, gap_floor, norm_gap, color=ps.GRAY, lw=0.8, zorder=1)
    axes[1].scatter(norm_gap, y, color=colors, s=25, zorder=2)
    axes[1].set_xscale("log")
    axes[1].set_xlim(gap_floor, 1.55 * float(np.max(norm_gap)))
    axes[1].set_xlabel(r"median $\Delta\omega/\omega_{\rm mid}$")
    axes[1].tick_params(axis="y", labelleft=False)

    for label, ax in zip("ab", axes):
        ps.panel_label(ax, label)
    out = FIG_DIR / "fig_vector_controls"
    ps.savefig(fig, out.parent, out.name)
    return out.with_suffix(".pdf")


def run_controls(args: argparse.Namespace) -> dict:
    ensure_dirs()
    jobs = [
        (int(net_seed), vars(args))
        for net_seed in range(int(args.control_net_seeds))
    ]
    n_workers = max(1, min(int(args.workers), len(jobs)))
    print(f"controls: {len(jobs)} cells with {n_workers} workers", flush=True)
    if n_workers == 1:
        completed = [_control_cell_job(job) for job in jobs]
    else:
        with cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            completed = list(pool.map(_control_cell_job, jobs, chunksize=1))
    records = [record for group in completed for record in group]
    for rec in records:
        print(
            f"control net={rec['net_seed']} {rec['control']}: "
            f"success={rec['success']} gap={rec['band_gap']['gap']:.4f} "
            f"n_in={rec['metrics']['n_in']}",
            flush=True,
        )

    table = write_vector_controls_table(
        records, FIG_DIR / "table_vector_controls.tex",
    )
    fig = plot_vector_controls(records)
    payload = {
        "records": records,
        "summary": summarize_records(records, "control"),
        "table": str(table.relative_to(REPO_ROOT)),
        "figure": str(fig.relative_to(REPO_ROOT)),
    }
    save_json(DATA_DIR / "vector_controls.json", payload)
    save_records_npz(DATA_DIR / "vector_controls.npz", records, "control")
    print(json.dumps(json_ready(payload["summary"]), indent=2, sort_keys=True), flush=True)
    return payload


def _control_cell_job(job: tuple[int, dict]) -> list[dict]:
    net_seed, config = job
    args = argparse.Namespace(**config)
    cell = make_periodic_vector_cell(args.size, net_seed, args.topology)
    initial_eval = band_frequencies(
        cell, np.full(cell.n_edges, R_INIT), k_grid(args.k_eval_grid),
    )
    wlo, whi = target_window_from_percentiles(
        initial_eval, args.center_percentile, args.width_percentile,
    )
    records = []
    paired_material = None
    paired_stiffness = None
    for label, update_rule, regularizer, bound_mode in CONTROL_SPECS:
        result = train_periodic_vector(
            cell, wlo, whi,
            n_steps=args.steps,
            alpha=args.alpha,
            grad_clip=args.grad_clip,
            inflation_strength=args.inflation,
            n_freq=args.n_freq,
            k_train_grid=args.k_train_grid,
            k_eval_grid=args.k_eval_grid,
            k_batch=args.k_batch,
            force_batch=args.force_batch,
            seed=args.train_seed,
            eval_every=max(args.eval_every, max(1, args.steps // 4)),
            update_rule=update_rule,
            regularizer=regularizer,
            response_mode=args.response_mode,
            damping_gamma_ratio=args.damping_gamma_ratio,
            material_update=args.material_update,
            response_metric_eta=args.response_metric_eta,
            response_metric_lambda_ratio=args.response_metric_lambda_ratio,
            response_bound_mode=bound_mode,
            frequency_sampling=args.frequency_sampling,
            frequencies_per_step=args.frequencies_per_step,
        )
        record = make_run_record(
            cell=cell,
            wlo=wlo,
            whi=whi,
            radii=result["radii"],
            k_dense_grid=args.k_dense_grid,
            label_key="control",
            label=label,
            train_seed=args.train_seed,
            elapsed_s=result["elapsed_s"],
            params=result["params"],
        )
        records.append(record)
        if label == "paired-response":
            paired_material = record["material"]["material_ratio"]
            paired_stiffness = record["material"]["mean_stiffness_ratio"]
    for label, ratio in (
        ("uniform-material", paired_material),
        ("uniform-stiffness", paired_stiffness),
    ):
        radius = float(np.clip(np.sqrt(ratio), R_MIN, R_MAX))
        records.append(make_run_record(
            cell=cell,
            wlo=wlo,
            whi=whi,
            radii=np.full(cell.n_edges, radius),
            k_dense_grid=args.k_dense_grid,
            label_key="control",
            label=label,
            train_seed=args.train_seed,
            params={"matched_ratio": float(ratio), "uniform_radius": radius},
        ))
    return records


DEFAULT_REGULARIZERS = [
    "none",
    "inflation",
    "inflation_l1",
    "inflation_l2",
    "l1",
    "l2",
    "material_penalty",
    "stiffness_penalty",
    "binary_prior",
]


def run_regularizers(args: argparse.Namespace) -> dict:
    ensure_dirs()
    records = []
    for regularizer in args.regularizers:
        for net_seed in range(int(args.regularizer_net_seeds)):
            cell = make_periodic_vector_cell(args.size, net_seed, args.topology)
            initial_eval = band_frequencies(cell, np.full(cell.n_edges, R_INIT), k_grid(args.k_eval_grid))
            wlo, whi = target_window_from_percentiles(
                initial_eval, args.center_percentile, args.width_percentile,
            )
            print(f"regularizer={regularizer} net={net_seed}", flush=True)
            result = train_periodic_vector(
                cell, wlo, whi,
                n_steps=args.steps,
                alpha=args.alpha,
                grad_clip=args.grad_clip,
                inflation_strength=args.inflation,
                n_freq=args.n_freq,
                k_train_grid=args.k_train_grid,
                k_eval_grid=args.k_eval_grid,
                k_batch=args.k_batch,
                force_batch=args.force_batch,
                seed=args.train_seed,
                eval_every=max(args.eval_every, max(1, args.steps // 4)),
                update_rule="adjoint",
                regularizer=regularizer,
                response_mode=args.response_mode,
                damping_gamma_ratio=args.damping_gamma_ratio,
                material_update="radius_euler",
                frequency_sampling=args.frequency_sampling,
                frequencies_per_step=args.frequencies_per_step,
            )
            rec = make_run_record(
                cell=cell,
                wlo=wlo,
                whi=whi,
                radii=result["radii"],
                k_dense_grid=args.k_dense_grid,
                label_key="regularizer",
                label=regularizer,
                train_seed=args.train_seed,
                elapsed_s=result["elapsed_s"],
                params=result["params"],
            )
            records.append(rec)
            print(
                f"  success={rec['success']} gap={rec['band_gap']['gap']:.4f} "
                f"n_in={rec['metrics']['n_in']}",
                flush=True,
            )
    table = write_summary_table(
        records, "regularizer", FIG_DIR / "table_vector_regularizers.tex", "regularizer", "cells",
    )
    payload = {
        "records": records,
        "summary": summarize_records(records, "regularizer"),
        "table": str(table.relative_to(REPO_ROOT)),
    }
    save_json(DATA_DIR / "vector_regularizers.json", payload)
    save_records_npz(DATA_DIR / "vector_regularizers.npz", records, "regularizer")
    print(json.dumps(json_ready(payload["summary"]), indent=2, sort_keys=True), flush=True)
    return payload


def run_perturbation(args: argparse.Namespace) -> dict:
    if args.input is None:
        raise SystemExit("--mode perturbation requires --input")
    ensure_dirs()
    cell, result = load_npz_result(args.input)
    rng = np.random.RandomState(int(args.noise_seed))
    records = []
    for sigma in args.noise_levels:
        for rep in range(int(args.noise_reps)):
            radii = np.clip(
                result["radii"] * (1.0 + float(sigma) * rng.normal(size=cell.n_edges)),
                R_MIN,
                R_MAX,
            )
            rec = make_run_record(
                cell=cell,
                wlo=result["wlo"],
                whi=result["whi"],
                radii=radii,
                k_dense_grid=args.k_dense_grid,
                label_key="sigma",
                label=f"{float(sigma):.3g}",
                train_seed=rep,
                params={"noise_sigma": float(sigma), "noise_rep": int(rep)},
            )
            records.append(rec)
        print(f"noise sigma={sigma}: done {args.noise_reps} reps", flush=True)
    table = write_summary_table(
        records, "sigma", FIG_DIR / "table_vector_perturbation.tex", "radius noise", "perturbations",
    )
    payload = {
        "input": str(args.input),
        "records": records,
        "summary": summarize_records(records, "sigma"),
        "table": str(table.relative_to(REPO_ROOT)),
    }
    save_json(DATA_DIR / "vector_perturbation.json", payload)
    save_records_npz(DATA_DIR / "vector_perturbation.npz", records, "sigma")
    print(json.dumps(json_ready(payload["summary"]), indent=2, sort_keys=True), flush=True)
    return payload


def load_output_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(f"missing required output {path}")
    return json.loads(path.read_text())


def plot_vector_summary(args: argparse.Namespace) -> Path:
    ensure_dirs()
    ps.style()
    bz = load_output_json("bz_convergence.json")
    ensemble = load_output_json("vector_gap_ensemble.json")
    regularizers = load_output_json("vector_regularizers.json")
    perturbation = load_output_json("vector_perturbation.json")

    fig, axes = plt.subplots(
        2, 2, figsize=(ps.TEXT_W, 4.9), constrained_layout=True,
    )
    ax = axes[0, 0]
    grids = [r["grid"] for r in bz["records"]]
    gaps = [r["band_gap"]["normalized_gap"] for r in bz["records"]]
    ax.plot(grids, gaps, marker="o", color=ps.BLUE, ms=4, lw=1.1)
    ax.set_xticks(grids)
    ax.set_xlim(min(grids) - 3, max(grids) + 3)
    ax.set_xlabel(r"BZ grid $N_k$")
    ax.set_ylabel(r"$\Delta\omega/\omega_{\rm mid}$")

    ax = axes[0, 1]
    all_ens_records = ensemble["records"]
    primary_size = min(int(r["size"]) for r in all_ens_records)
    ens_records = sorted(
        [r for r in all_ens_records if int(r["size"]) == primary_size],
        key=lambda r: int(r["net_seed"]),
    )
    x = [int(r["net_seed"]) for r in ens_records]
    ax.bar(x, [r["band_gap"]["normalized_gap"] for r in ens_records],
           color=ps.GREEN)
    ax.set_xticks(x)
    ax.set_xlabel("network seed")
    ax.set_ylabel(r"$\Delta\omega/\omega_{\rm mid}$")

    ax = axes[1, 0]
    reg_summary = regularizers["summary"]
    short_labels = {
        "none": "none",
        "inflation": "infl.",
        "inflation_l1": r"infl.+$\ell_1$",
        "inflation_l2": r"infl.+$\ell_2$",
        "l1": r"$\ell_1$",
        "l2": r"$\ell_2$",
        "material_penalty": "material",
        "stiffness_penalty": "stiffness",
        "binary_prior": "binary",
    }
    labels = [short_labels.get(r["regularizer"], r["regularizer"].replace("_", " ")) for r in reg_summary]
    x = np.arange(len(labels))
    ax.bar(x, [r["median_normalized_gap"] for r in reg_summary], color=ps.PURPLE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"median $\Delta\omega/\omega_{\rm mid}$")

    ax = axes[1, 1]
    pert_summary = sorted(perturbation["summary"], key=lambda r: float(r["sigma"]))
    sigmas = [float(r["sigma"]) for r in pert_summary]
    med = [r["median_normalized_gap"] for r in pert_summary]
    # The BZ-convergence cache evaluates the same unperturbed exemplar. Use
    # its grid matched to the perturbation screen as the sigma=0 baseline.
    if sigmas and not np.isclose(sigmas[0], 0.0):
        perturbation_grid = int(perturbation["records"][0]["k_dense_grid"])
        baseline = next(
            (r for r in bz["records"] if int(r["grid"]) == perturbation_grid),
            None,
        )
        if baseline is not None:
            sigmas.insert(0, 0.0)
            med.insert(0, float(baseline["band_gap"]["normalized_gap"]))
    ax.plot(sigmas, med, marker="o", color=ps.RED, ms=4, lw=1.1)
    ax.set_xticks(sigmas)
    ax.set_xlim(-0.003, max(sigmas) * 1.08)
    ax.set_xlabel(r"radius-noise $\sigma$")
    ax.set_ylabel(r"median $\Delta\omega/\omega_{\rm mid}$")

    # All four panels report the same dimensionless gap. A common baseline
    # and range prevent tiny convergence changes from being visually inflated.
    for label, ax in zip("abcd", axes.ravel()):
        ax.set_ylim(0.0, 0.68)
        ps.panel_label(ax, label)
    out = FIG_DIR / "fig_vector_bz_ensemble"
    ps.savefig(fig, out.parent, out.name)
    print(f"wrote {out.with_suffix('.pdf')}", flush=True)
    return out.with_suffix(".pdf")


def wrap_reduced_k(kred: np.ndarray) -> np.ndarray:
    return (np.asarray(kred, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def band_frequency_at(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    kred: np.ndarray,
    band_index: int,
) -> float:
    return float(band_frequencies(cell, radii, wrap_reduced_k(kred)[None, :])[0, int(band_index)])


def band_neighborhood_validation(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    kred: np.ndarray,
    band_index: int,
    delta: float = 1e-3,
) -> dict:
    offsets = np.asarray([
        [0.0, 0.0],
        [delta, 0.0],
        [-delta, 0.0],
        [0.0, delta],
        [0.0, -delta],
    ])
    vals = []
    separations = []
    for offset in offsets:
        freqs = band_frequencies(cell, radii, wrap_reduced_k(kred + offset)[None, :])[0]
        band = int(band_index)
        vals.append(float(freqs[band]))
        if band > 0:
            separations.append(float(freqs[band] - freqs[band - 1]))
        if band + 1 < len(freqs):
            separations.append(float(freqs[band + 1] - freqs[band]))
    return {
        "delta": float(delta),
        "band_frequency_range": float(np.max(vals) - np.min(vals)),
        "min_adjacent_separation": float(np.min(separations)) if separations else float("nan"),
    }


def refine_band_extremum(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    band_index: int,
    starts: np.ndarray,
    *,
    maximize: bool,
    maxiter: int,
) -> dict:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise SystemExit("scipy is required for --mode refine-gap") from exc

    def objective(x: np.ndarray) -> float:
        val = band_frequency_at(cell, radii, x, band_index)
        return -val if maximize else val

    best = None
    for start in np.asarray(starts, dtype=np.float64):
        res = minimize(
            objective,
            wrap_reduced_k(start),
            method="Nelder-Mead",
            options={
                "maxiter": int(maxiter),
                "xatol": 1e-10,
                "fatol": 1e-11,
                "disp": False,
            },
        )
        k_opt = wrap_reduced_k(res.x)
        freq = band_frequency_at(cell, radii, k_opt, band_index)
        record = {
            "band_index": int(band_index),
            "band_1based": int(band_index + 1),
            "frequency": float(freq),
            "k": k_opt,
            "start_k": wrap_reduced_k(start),
            "optimizer_success": bool(res.success),
            "optimizer_message": str(res.message),
            "nfev": int(res.nfev),
            "nit": int(res.nit),
            "neighborhood": band_neighborhood_validation(cell, radii, k_opt, band_index),
        }
        if best is None or maximize and record["frequency"] > best["frequency"] or not maximize and record["frequency"] < best["frequency"]:
            best = record
    return best


def run_refine_gap(args: argparse.Namespace) -> dict:
    if args.input is None:
        raise SystemExit("--mode refine-gap requires --input")
    ensure_dirs()
    cell, result = load_npz_result(args.input)
    wlo, whi = result["wlo"], result["whi"]
    grid = int(args.refine_grid)
    kpts = k_grid(grid)
    freqs = band_frequencies(cell, result["radii"], kpts)
    grid_gap = best_gap_near_window(freqs, wlo, whi, kpts)
    band = int(grid_gap["band_index"])
    n_candidates = min(int(args.refine_candidates), len(kpts))
    lower_candidates = np.argsort(freqs[:, band])[-n_candidates:]
    upper_candidates = np.argsort(freqs[:, band + 1])[:n_candidates]
    lower = refine_band_extremum(
        cell, result["radii"], band, kpts[lower_candidates],
        maximize=True, maxiter=args.refine_maxiter,
    )
    upper = refine_band_extremum(
        cell, result["radii"], band + 1, kpts[upper_candidates],
        maximize=False, maxiter=args.refine_maxiter,
    )
    refined_gap = float(upper["frequency"] - lower["frequency"])
    midpoint = 0.5 * (upper["frequency"] + lower["frequency"])
    payload = {
        "input": str(args.input),
        "target": {"wlo": wlo, "whi": whi},
        "grid": grid,
        "grid_gap": grid_gap,
        "refined_gap": {
            "band_index": band,
            "lower_band_1based": int(band + 1),
            "upper_band_1based": int(band + 2),
            "lower_edge": float(lower["frequency"]),
            "upper_edge": float(upper["frequency"]),
            "gap": refined_gap,
            "normalized_gap": float(refined_gap / midpoint) if midpoint > 0 else float("nan"),
            "positive": bool(refined_gap > 0.0),
            "lower_k": lower["k"],
            "upper_k": upper["k"],
            "overlaps_target_window": bool(
                refined_gap > 0.0 and max(lower["frequency"], wlo) < min(upper["frequency"], whi)
            ),
            "contains_target_window": bool(
                refined_gap > 0.0 and lower["frequency"] <= wlo and upper["frequency"] >= whi
            ),
        },
        "lower_refinement": lower,
        "upper_refinement": upper,
        "refine_candidates": int(n_candidates),
    }
    save_json(DATA_DIR / "refined_gap.json", payload)
    print(
        f"refined band {band_label(payload['refined_gap'])}: "
        f"gap={refined_gap:.6f}, normalized={payload['refined_gap']['normalized_gap']:.3f}",
        flush=True,
    )
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=[
            "single", "search", "plot", "bz-convergence", "refine-gap",
            "ensemble", "controls", "regularizers", "perturbation",
            "summary-figure", "tables",
        ],
        default="single",
    )
    p.add_argument("--topology", default="rand-del")
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--net-seed", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--center-percentile", type=float, default=50.0)
    p.add_argument("--width-percentile", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=0.035)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--inflation", type=float, default=0.0)
    p.add_argument("--regularizer", default="none")
    p.add_argument("--response-mode", choices=RESPONSE_MODES, default="reciprocal")
    p.add_argument(
        "--damping-gamma-ratio",
        type=float,
        default=0.0,
        help="mass-damping gamma divided by the target-window midpoint",
    )
    p.add_argument(
        "--material-update",
        choices=("radius_euler", "response_conditioned_log"),
        default="response_conditioned_log",
    )
    p.add_argument("--response-metric-eta", type=float, default=0.02)
    p.add_argument("--response-metric-lambda-ratio", type=float, default=0.025)
    p.add_argument(
        "--response-bound-mode", choices=("clip", "none"), default="clip",
    )
    p.add_argument("--n-freq", type=int, default=5)
    p.add_argument(
        "--frequency-sampling", choices=["random_grid", "all"],
        default="random_grid",
    )
    p.add_argument("--frequencies-per-step", type=int, default=1)
    p.add_argument("--k-train-grid", type=int, default=5)
    p.add_argument("--k-eval-grid", type=int, default=9)
    p.add_argument("--k-dense-grid", type=int, default=17)
    p.add_argument("--k-batch", type=int, default=4)
    p.add_argument("--force-batch", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    p.add_argument("--input", type=Path, default=None, help="Existing NPZ for --mode plot")
    p.add_argument("--bz-grids", type=int, nargs="+", default=[17, 33, 65])
    p.add_argument("--refine-grid", type=int, default=33)
    p.add_argument("--refine-candidates", type=int, default=7)
    p.add_argument("--refine-maxiter", type=int, default=400)

    p.add_argument("--search-sizes", type=int, nargs="+", default=[4, 5])
    p.add_argument("--search-net-seeds", type=int, default=4)
    p.add_argument("--search-train-seeds", type=int, default=2)
    p.add_argument("--search-centers", type=float, nargs="+", default=[45.0, 50.0, 55.0])
    p.add_argument("--search-widths", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0])
    p.add_argument("--no-stop-on-success", action="store_true")

    p.add_argument("--ensemble-sizes", type=int, nargs="+", default=[5])
    p.add_argument("--ensemble-net-seeds", type=int, default=10)
    p.add_argument("--control-net-seeds", type=int, default=8)
    p.add_argument("--regularizer-net-seeds", type=int, default=3)
    p.add_argument("--regularizers", nargs="+", default=DEFAULT_REGULARIZERS)
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    p.add_argument("--noise-reps", type=int, default=50)
    p.add_argument("--noise-seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    elif args.mode == "search":
        run_search(args)
    elif args.mode == "plot":
        if args.input is None:
            raise SystemExit("--mode plot requires --input")
        cell, result = load_npz_result(args.input)
        plot_validation(cell, result, FIG_DIR / "figS_vector_periodic_validation")
    elif args.mode == "bz-convergence":
        run_bz_convergence(args)
    elif args.mode == "refine-gap":
        run_refine_gap(args)
    elif args.mode == "ensemble":
        run_ensemble(args)
    elif args.mode == "controls":
        run_controls(args)
    elif args.mode == "regularizers":
        run_regularizers(args)
    elif args.mode == "perturbation":
        run_perturbation(args)
    elif args.mode == "summary-figure":
        plot_vector_summary(args)
        controls = load_output_json("vector_controls.json")
        plot_vector_controls(controls["records"])
    elif args.mode == "tables":
        render_active_tables()
    else:
        raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()
