"""Test self-probed trace learning in a fully three-dimensional Bloch model.

This exploratory calculation is deliberately separate from the publication
workflow.  Each node has three translational displacement components, the
central-force network and Bloch wavevectors are three-dimensional, and gap
validation covers a cubic Brillouin zone rather than a high-symmetry path.

The experiment uses the same response-conditioned log-stiffness material law
as the two-dimensional periodic calculation.  Its purpose is to test whether
that dimension-independent local rule can open a complete 3D Bloch band gap;
it does not alter or regenerate any manuscript result.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
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
from publication.bloch_gap import (
    R_INIT,
    R_MAX,
    R_MIN,
    PeriodicVectorCell,
    band_frequencies,
    best_gap_near_window,
    bloch_stiffness,
    edge_extensions,
    k_grid,
    material_diagnostics,
    response_matrix,
    target_window_from_percentiles,
    train_periodic_vector,
    window_metrics,
)
from publication.paths import REPO_ROOT
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEFAULT_OUTPUT = REPO_ROOT / "scripts" / "outputs" / "experimental_3d"


def json_ready(value):
    """Convert numpy-rich records into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def _canonical_edge(i: int, j: int, offset: np.ndarray) -> tuple[int, ...]:
    forward = (int(i), int(j), *(int(x) for x in offset))
    reverse = (int(j), int(i), *(int(-x) for x in offset))
    return min(forward, reverse)


def make_periodic_delaunay_3d(
    size: int,
    seed: int,
    jitter: float = 0.30,
) -> PeriodicVectorCell:
    """Construct a periodic 3D Delaunay central-force cell.

    ``size**3`` points begin on a cubic grid and receive independent quenched
    displacements.  Delaunay tetrahedralization is performed on a 3x3x3
    tiling; edge orbits incident on the central tile define the periodic cell.
    The jitter removes lattice degeneracies while retaining a controlled point
    density and avoiding arbitrarily close random pairs.
    """
    size = int(size)
    if size < 2:
        raise ValueError("size must be at least two")
    jitter = float(jitter)
    if not 0.0 < jitter < 0.5:
        raise ValueError("jitter must lie strictly between zero and 0.5")

    rng = np.random.RandomState(int(seed))
    axes = [np.arange(size, dtype=float) + 0.5 for _ in range(3)]
    pos = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
    pos += rng.uniform(-jitter, jitter, size=pos.shape)
    box = np.full(3, float(size), dtype=np.float64)
    pos %= box

    shifts = np.asarray(
        list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64
    )
    tiled_pos = np.concatenate([pos + shift * box for shift in shifts], axis=0)
    tiled_base = np.tile(np.arange(len(pos), dtype=np.int64), len(shifts))
    tiled_shift = np.repeat(shifts, len(pos), axis=0)
    simplices = Delaunay(tiled_pos).simplices

    central = np.zeros(3, dtype=np.int64)
    edge_keys: set[tuple[int, ...]] = set()
    for simplex in simplices:
        for left, right in itertools.combinations(simplex, 2):
            shift_left = tiled_shift[left]
            shift_right = tiled_shift[right]
            left_central = bool(np.array_equal(shift_left, central))
            right_central = bool(np.array_equal(shift_right, central))
            if not (left_central or right_central):
                continue
            if left_central:
                i = int(tiled_base[left])
                j = int(tiled_base[right])
                offset = shift_right - shift_left
            else:
                i = int(tiled_base[right])
                j = int(tiled_base[left])
                offset = shift_left - shift_right
            if i == j and np.all(offset == 0):
                continue
            edge_keys.add(_canonical_edge(i, j, offset))

    ordered = sorted(edge_keys)
    edges = np.asarray([[item[0], item[1]] for item in ordered], dtype=np.int64)
    offsets = np.asarray([item[2:] for item in ordered], dtype=np.int64)
    vectors = pos[edges[:, 1]] - pos[edges[:, 0]] + offsets * box
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-10):
        raise RuntimeError("periodic Delaunay construction produced a zero-length edge")
    directions = vectors / lengths[:, None]

    cell = PeriodicVectorCell(
        size=size,
        seed=int(seed),
        pos=pos,
        box=box,
        edges=edges,
        offsets=offsets,
        vectors=vectors,
        lengths=lengths,
        directions=directions,
    )
    if cell.n_dim != 3 or cell.n_dof != 3 * cell.n_nodes:
        raise AssertionError("3D cell has an inconsistent degree-of-freedom count")

    gamma_values = np.linalg.eigvalsh(
        bloch_stiffness(cell, np.full(cell.n_edges, R_INIT), np.zeros(3))
    )
    scale = max(float(gamma_values[-1]), 1.0)
    if np.sum(gamma_values < 1e-9 * scale) != 3:
        raise RuntimeError(
            "initial 3D cell does not have exactly three translational zero modes"
        )
    return cell


def fixed_probe_gradient_check(cell: PeriodicVectorCell, seed: int = 0) -> dict:
    """Compare the 3D edge-local paired-response signal with finite differences."""
    rng = np.random.RandomState(int(seed))
    radii = rng.uniform(0.8, 1.2, size=cell.n_edges)
    kred = rng.uniform(-0.8 * np.pi, 0.8 * np.pi, size=3)
    eigenfrequencies = np.sqrt(np.maximum(
        np.linalg.eigvalsh(bloch_stiffness(cell, radii, kred)).real, 0.0
    ))
    omega = float(np.percentile(eigenfrequencies[eigenfrequencies > 1e-8], 43.0))
    omega *= 1.017
    force = rng.normal(size=cell.n_dof) + 1j * rng.normal(size=cell.n_dof)
    force /= np.linalg.norm(force)

    operator = response_matrix(cell, radii, kred, omega, 0.0, "reciprocal")
    response = np.linalg.solve(operator, force)
    redrive = np.linalg.solve(operator, response)
    du = edge_extensions(cell, response, kred)
    dv = edge_extensions(cell, redrive, kred)
    analytic = -4.0 * radii / cell.lengths * np.real(np.conjugate(dv) * du)

    chosen = rng.choice(cell.n_edges, size=min(8, cell.n_edges), replace=False)
    epsilon = 2e-6
    records = []
    for edge in chosen:
        costs = []
        for sign in (-1.0, 1.0):
            perturbed = radii.copy()
            perturbed[edge] += sign * epsilon
            h = response_matrix(cell, perturbed, kred, omega, 0.0, "reciprocal")
            u = np.linalg.solve(h, force)
            costs.append(float(np.vdot(u, u).real))
        finite_difference = (costs[1] - costs[0]) / (2.0 * epsilon)
        absolute_error = abs(float(analytic[edge]) - finite_difference)
        relative_error = absolute_error / max(
            1.0, abs(float(analytic[edge])), abs(finite_difference)
        )
        records.append({
            "edge": int(edge),
            "analytic": float(analytic[edge]),
            "finite_difference": float(finite_difference),
            "absolute_error": float(absolute_error),
            "relative_error": float(relative_error),
        })
    maximum = max(record["relative_error"] for record in records)
    return {
        "frequency": omega,
        "wavevector": kred,
        "finite_difference_step": epsilon,
        "records": records,
        "max_relative_error": float(maximum),
        "pass": bool(maximum < 2e-5),
    }


def _band_frequency(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    wavevector: np.ndarray,
    band_index: int,
) -> float:
    values = np.linalg.eigvalsh(bloch_stiffness(cell, radii, wavevector))
    return float(np.sqrt(max(float(values[int(band_index)].real), 0.0)))


def refine_gap_edges(
    cell: PeriodicVectorCell,
    radii: np.ndarray,
    dense_k: np.ndarray,
    dense_frequencies: np.ndarray,
    band_index: int,
    candidates: int = 8,
    maxiter: int = 250,
) -> dict:
    """Refine the upper/lower edge of one indirect 3D band gap."""
    band_index = int(band_index)
    candidates = min(int(candidates), len(dense_k))
    bounds = [(-np.pi, np.pi)] * 3

    def optimize_edge(which: str) -> dict:
        target_band = band_index if which == "lower" else band_index + 1
        values = dense_frequencies[:, target_band]
        ordering = np.argsort(values)
        if which == "lower":
            ordering = ordering[::-1]
        best_frequency = float(values[ordering[0]])
        best_k = dense_k[ordering[0]].copy()
        total_evaluations = 0
        for index in ordering[:candidates]:
            sign = -1.0 if which == "lower" else 1.0
            result = minimize(
                lambda point, sign=sign: sign * _band_frequency(
                    cell, radii, np.asarray(point), target_band
                ),
                dense_k[index],
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxiter), "ftol": 1e-13, "gtol": 1e-9},
            )
            total_evaluations += int(result.nfev)
            frequency = _band_frequency(cell, radii, result.x, target_band)
            improved = frequency > best_frequency if which == "lower" else frequency < best_frequency
            if improved:
                best_frequency = frequency
                best_k = np.asarray(result.x, dtype=float)
        return {
            "band_1based": int(target_band + 1),
            "frequency": best_frequency,
            "wavevector": best_k,
            "function_evaluations": total_evaluations,
        }

    lower = optimize_edge("lower")
    upper = optimize_edge("upper")
    gap = float(upper["frequency"] - lower["frequency"])
    midpoint = 0.5 * float(upper["frequency"] + lower["frequency"])
    return {
        "lower": lower,
        "upper": upper,
        "gap": gap,
        "normalized_gap": gap / midpoint if midpoint > 0.0 else float("nan"),
        "positive": bool(gap > 0.0),
        "optimizer": "multistart L-BFGS-B from dense-grid extrema",
        "candidates_per_edge": candidates,
    }


def evaluate_dense(
    cell: PeriodicVectorCell,
    initial_radii: np.ndarray,
    learned_radii: np.ndarray,
    wlo: float,
    whi: float,
    grid: int,
    *,
    refine: bool,
    refine_candidates: int,
    refine_maxiter: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate both materials on an independent uniform 3D k grid."""
    started = time.perf_counter()
    points = k_grid(int(grid), dimension=3)
    initial = band_frequencies(cell, initial_radii, points)
    learned = band_frequencies(cell, learned_radii, points)
    initial_gap = best_gap_near_window(initial, wlo, whi, points)
    learned_gap = best_gap_near_window(learned, wlo, whi, points)
    refined = None
    if refine:
        refined = refine_gap_edges(
            cell,
            learned_radii,
            points,
            learned,
            int(learned_gap["band_index"]),
            candidates=refine_candidates,
            maxiter=refine_maxiter,
        )
        refined["contains_target_window"] = bool(
            refined["positive"]
            and refined["lower"]["frequency"] <= wlo
            and refined["upper"]["frequency"] >= whi
        )
    learned_metrics = window_metrics(learned, wlo, whi)
    success = bool(
        learned_metrics["n_in"] == 0
        and learned_gap["gap"] > 0.0
        and learned_gap["contains_target_window"]
        and (refined is None or refined["contains_target_window"])
    )
    summary = {
        "grid_per_axis": int(grid),
        "wavevectors": int(len(points)),
        "initial_metrics": window_metrics(initial, wlo, whi),
        "learned_metrics": learned_metrics,
        "initial_band_gap": initial_gap,
        "learned_band_gap": learned_gap,
        "refined_learned_gap": refined,
        "success": success,
        "elapsed_s": float(time.perf_counter() - started),
    }
    return summary, points, initial, learned


def cubic_k_path(points_per_segment: int = 25):
    vertices = [
        (np.array([0.0, 0.0, 0.0]), r"$\Gamma$"),
        (np.array([np.pi, 0.0, 0.0]), "X"),
        (np.array([np.pi, np.pi, 0.0]), "M"),
        (np.array([0.0, 0.0, 0.0]), r"$\Gamma$"),
        (np.array([np.pi, np.pi, np.pi]), "R"),
        (np.array([np.pi, 0.0, 0.0]), "X"),
    ]
    wavevectors = []
    distances = []
    ticks = [0.0]
    labels = [vertices[0][1]]
    distance = 0.0
    for segment, ((left, _), (right, right_label)) in enumerate(
        zip(vertices[:-1], vertices[1:])
    ):
        samples = np.linspace(0.0, 1.0, int(points_per_segment) + 1)
        if segment:
            samples = samples[1:]
        for fraction in samples:
            point = (1.0 - fraction) * left + fraction * right
            if wavevectors:
                distance += float(np.linalg.norm(point - wavevectors[-1]))
            wavevectors.append(point)
            distances.append(distance)
        ticks.append(distance)
        labels.append(right_label)
    return np.asarray(wavevectors), np.asarray(distances), ticks, labels


def render_summary(
    path: Path,
    cell: PeriodicVectorCell,
    learned_radii: np.ndarray,
    result: dict,
    dense_summary: dict,
) -> None:
    """Render the physically relevant 3D gap diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wavevectors, distance, ticks, labels = cubic_k_path()
    initial_path = band_frequencies(
        cell, np.full(cell.n_edges, R_INIT), wavevectors
    )
    learned_path = band_frequencies(cell, learned_radii, wavevectors)
    wlo, whi = float(result["wlo"]), float(result["whi"])

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)
    for axis, frequencies, title, color in (
        (axes[0], initial_path, "initial", "0.50"),
        (axes[1], learned_path, "learned", "#1764ab"),
    ):
        axis.axhspan(wlo, whi, color="#f28e2b", alpha=0.22, lw=0)
        for band in range(frequencies.shape[1]):
            axis.plot(distance, frequencies[:, band], color=color, lw=0.55, alpha=0.70)
        axis.set_xticks(ticks, labels)
        axis.set_xlim(distance[0], distance[-1])
        axis.set_title(title)
        axis.set_xlabel("cubic Bloch path")
    axes[0].set_ylabel(r"frequency $\omega$")
    axes[1].set_ylim(axes[0].get_ylim())

    history = np.asarray(result["history"], dtype=float)
    axes[2].plot(history[:, 0], history[:, 1], color="#1764ab", lw=1.6)
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("sampled modes in window")
    axes[2].set_title(
        "dense success" if dense_summary["success"] else "dense check not yet cleared"
    )
    axes[2].set_ylim(bottom=0)
    figure.suptitle(
        "3D self-probed trace learning: complete Bloch-gap test",
        fontsize=11,
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def case_stem(config: dict, net_seed: int) -> str:
    return (
        f"bloch3d_s{config['size']}_net{int(net_seed)}_train"
        f"{int(config['train_seed_offset']) + int(net_seed)}"
    )


def save_case_npz(
    path: Path,
    cell: PeriodicVectorCell,
    result: dict,
    dense_k: np.ndarray,
    initial_dense: np.ndarray,
    learned_dense: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        size=np.int64(cell.size),
        seed=np.int64(cell.seed),
        pos=cell.pos,
        box=cell.box,
        edges=cell.edges,
        offsets=cell.offsets,
        vectors=cell.vectors,
        lengths=cell.lengths,
        directions=cell.directions,
        radii=result["radii"],
        wlo=np.float64(result["wlo"]),
        whi=np.float64(result["whi"]),
        history=result["history"],
        cost_history=result["cost_history"],
        dense_k=dense_k,
        initial_dense=initial_dense,
        learned_dense=learned_dense,
        params_json=np.asarray(json.dumps(json_ready(result["params"]))),
    )


def load_case_npz(path: Path) -> tuple[PeriodicVectorCell, np.ndarray, float, float]:
    """Load the geometry, learned radii, and target from a saved 3D case."""
    with np.load(path, allow_pickle=False) as data:
        cell = PeriodicVectorCell(
            size=int(data["size"]),
            seed=int(data["seed"]),
            pos=data["pos"],
            box=data["box"],
            edges=data["edges"],
            offsets=data["offsets"],
            vectors=data["vectors"],
            lengths=data["lengths"],
            directions=data["directions"],
        )
        return cell, data["radii"].copy(), float(data["wlo"]), float(data["whi"])


def verify_saved_case(
    path: Path,
    grids: list[int],
    *,
    refine_candidates: int,
    refine_maxiter: int,
) -> dict:
    """Run a Brillouin-zone convergence series without retraining."""
    if not grids:
        raise ValueError("at least one verification grid is required")
    grids = sorted(set(int(grid) for grid in grids))
    if grids[0] < 2:
        raise ValueError("verification grids must be at least two")
    cell, radii, wlo, whi = load_case_npz(path)
    records = []
    for grid in grids:
        print(f"verifying saved 3D case on {grid}^3 grid", flush=True)
        summary, _, _, _ = evaluate_dense(
            cell,
            np.full(cell.n_edges, R_INIT),
            radii,
            wlo,
            whi,
            grid,
            refine=grid == grids[-1],
            refine_candidates=refine_candidates,
            refine_maxiter=refine_maxiter,
        )
        records.append(summary)
        print(
            f"  n_in={summary['learned_metrics']['n_in']}, "
            f"gap={summary['learned_band_gap']['gap']:.6f}, "
            f"success={summary['success']}",
            flush=True,
        )
    payload = {
        "input": str(path.resolve()),
        "target": {"wlo": wlo, "whi": whi},
        "grids": grids,
        "records": records,
        "all_grids_pass": bool(all(record["success"] for record in records)),
        "largest_grid_refined": True,
    }
    output = path.with_name(path.stem + "_verification.json")
    payload["output"] = str(output.resolve())
    write_json(output, payload)
    return payload


def run_case(config: dict, net_seed: int, *, make_figure: bool, refine: bool) -> dict:
    started = time.perf_counter()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = case_stem(config, net_seed)
    print(f"[{stem}] constructing periodic 3D cell", flush=True)
    cell = make_periodic_delaunay_3d(
        config["size"], net_seed, jitter=config["jitter"]
    )
    gradient_check = fixed_probe_gradient_check(cell, seed=7000 + int(net_seed))
    if not gradient_check["pass"]:
        raise RuntimeError(
            f"[{stem}] 3D gradient check failed: "
            f"{gradient_check['max_relative_error']:.3e}"
        )

    window_points = k_grid(config["window_grid"], dimension=3)
    initial_window_spectrum = band_frequencies(
        cell, np.full(cell.n_edges, R_INIT), window_points
    )
    wlo, whi = target_window_from_percentiles(
        initial_window_spectrum,
        config["center_percentile"],
        config["width_percentile"],
    )
    print(
        f"[{stem}] nodes={cell.n_nodes}, edges={cell.n_edges}, dof={cell.n_dof}, "
        f"window=[{wlo:.4f}, {whi:.4f}]",
        flush=True,
    )

    result = train_periodic_vector(
        cell,
        wlo,
        whi,
        n_steps=config["steps"],
        radius_min=R_MIN,
        radius_max=R_MAX,
        n_freq=config["n_freq"],
        k_train_grid=config["train_grid"],
        k_eval_grid=config["eval_grid"],
        k_batch=config["k_batch"],
        force_batch=config["force_batch"],
        seed=config["train_seed_offset"] + int(net_seed),
        eval_every=config["eval_every"],
        update_rule="adjoint",
        regularizer="none",
        response_mode="reciprocal",
        material_update="response_conditioned_log",
        response_metric_eta=config["eta"],
        response_metric_lambda_ratio=config["lambda_ratio"],
        response_bound_mode="clip",
        frequency_sampling="random_grid",
        frequencies_per_step=config["frequencies_per_step"],
    )
    print(f"[{stem}] independent dense {config['dense_grid']}^3 verification", flush=True)
    dense, dense_k, initial_dense, learned_dense = evaluate_dense(
        cell,
        np.full(cell.n_edges, R_INIT),
        result["radii"],
        wlo,
        whi,
        config["dense_grid"],
        refine=refine,
        refine_candidates=config["refine_candidates"],
        refine_maxiter=config["refine_maxiter"],
    )
    summary = {
        "status": "PASS" if dense["success"] else "NO_COMPLETE_GAP",
        "scientific_scope": {
            "spatial_dimension": 3,
            "displacement_components_per_node": 3,
            "wavevector_dimension": 3,
            "interaction": "unstressed central-force springs",
            "gap_test": "complete indirect Bloch gap over a cubic Brillouin zone",
            "publication_workflow": False,
        },
        "cell": {
            "size": int(cell.size),
            "nodes": cell.n_nodes,
            "edges": cell.n_edges,
            "degrees_of_freedom": cell.n_dof,
            "mean_coordination": float(2.0 * cell.n_edges / cell.n_nodes),
            "net_seed": int(net_seed),
            "jitter": float(config["jitter"]),
        },
        "target": {"wlo": wlo, "whi": whi},
        "gradient_check": gradient_check,
        "training": {
            "params": result["params"],
            "initial_metrics": result["initial_metrics"],
            "final_metrics": result["final_metrics"],
            "material": material_diagnostics(cell, result["radii"]),
            "elapsed_s": result["elapsed_s"],
        },
        "dense_verification": dense,
        "elapsed_s": float(time.perf_counter() - started),
    }
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"
    save_case_npz(npz_path, cell, result, dense_k, initial_dense, learned_dense)
    summary["artifacts"] = {"npz": str(npz_path), "json": str(json_path)}
    if make_figure:
        figure_path = output_dir / f"{stem}.png"
        render_summary(figure_path, cell, result["radii"], result, dense)
        summary["artifacts"]["figure"] = str(figure_path)
    write_json(json_path, summary)
    print(
        f"[{stem}] {summary['status']}: dense n_in={dense['learned_metrics']['n_in']}, "
        f"gap={dense['learned_band_gap']['gap']:.6f}, "
        f"normalized={dense['learned_band_gap']['normalized_gap']:.4f}",
        flush=True,
    )
    return summary


def _ensemble_worker(payload: tuple[dict, int]) -> dict:
    config, seed = payload
    return run_case(config, seed, make_figure=False, refine=False)


def run_ensemble(config: dict, seeds: list[int], workers: int) -> dict:
    started = time.perf_counter()
    jobs = [(config, int(seed)) for seed in seeds]
    if int(workers) == 1:
        records = [_ensemble_worker(job) for job in jobs]
    else:
        with cf.ProcessPoolExecutor(max_workers=int(workers)) as pool:
            records = list(pool.map(_ensemble_worker, jobs, chunksize=1))
    successes = [record["dense_verification"]["success"] for record in records]
    payload = {
        "status": "COMPLETE",
        "seeds": [int(seed) for seed in seeds],
        "workers": int(workers),
        "successes": int(sum(successes)),
        "cases": int(len(records)),
        "success_rate": float(np.mean(successes)),
        "records": records,
        "elapsed_s": float(time.perf_counter() - started),
    }
    path = Path(config["output_dir"]) / "ensemble_summary.json"
    write_json(path, payload)
    print(
        f"3D ensemble complete: {payload['successes']}/{payload['cases']} dense-grid successes",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("single", "ensemble", "verify"), default="single"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--net-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(8)))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-seed-offset", type=int, default=1000)
    parser.add_argument("--jitter", type=float, default=0.30)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--center-percentile", type=float, default=50.0)
    parser.add_argument("--width-percentile", type=float, default=10.0)
    parser.add_argument("--window-grid", type=int, default=5)
    parser.add_argument("--n-freq", type=int, default=5)
    parser.add_argument("--train-grid", type=int, default=5)
    parser.add_argument("--eval-grid", type=int, default=5)
    parser.add_argument("--dense-grid", type=int, default=11)
    parser.add_argument("--k-batch", type=int, default=8)
    parser.add_argument("--force-batch", type=int, default=2)
    parser.add_argument("--frequencies-per-step", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--lambda-ratio", type=float, default=0.025)
    parser.add_argument("--refine-candidates", type=int, default=8)
    parser.add_argument("--refine-maxiter", type=int, default=250)
    parser.add_argument(
        "--verify-grids", type=int, nargs="+", default=[9, 13, 17, 25]
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict:
    return {
        "output_dir": str(args.output_dir.resolve()),
        "size": int(args.size),
        "train_seed_offset": int(args.train_seed_offset),
        "jitter": float(args.jitter),
        "steps": int(args.steps),
        "center_percentile": float(args.center_percentile),
        "width_percentile": float(args.width_percentile),
        "window_grid": int(args.window_grid),
        "n_freq": int(args.n_freq),
        "train_grid": int(args.train_grid),
        "eval_grid": int(args.eval_grid),
        "dense_grid": int(args.dense_grid),
        "k_batch": int(args.k_batch),
        "force_batch": int(args.force_batch),
        "frequencies_per_step": int(args.frequencies_per_step),
        "eval_every": int(args.eval_every),
        "eta": float(args.eta),
        "lambda_ratio": float(args.lambda_ratio),
        "refine_candidates": int(args.refine_candidates),
        "refine_maxiter": int(args.refine_maxiter),
    }


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.mode == "single":
        run_case(config, int(args.net_seed), make_figure=True, refine=True)
    elif args.mode == "ensemble":
        if not args.seeds:
            raise SystemExit("--mode ensemble requires at least one --seeds value")
        run_ensemble(config, list(args.seeds), int(args.workers))
    else:
        if args.input is None:
            raise SystemExit("--mode verify requires --input")
        verify_saved_case(
            args.input,
            list(args.verify_grids),
            refine_candidates=int(args.refine_candidates),
            refine_maxiter=int(args.refine_maxiter),
        )


if __name__ == "__main__":
    main()
