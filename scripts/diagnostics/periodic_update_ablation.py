#!/usr/bin/env python3
"""Matched ablations of auxiliary terms in periodic Bloch-gap learning.

The experiment varies three conceptually distinct parts of the material law:

* the uniform inflation contribution;
* componentwise saturation of the paired-response trace gradient; and
* hard radius bounds beyond the nonnegative physical domain.

Every arm for a given cell uses the same prescribed spectral window and random
probe schedules.  Results are written one run at a time so a large sweep is
safe to resume after interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

for variable in (
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from publication.bloch_gap import (  # noqa: E402
    R_INIT,
    band_frequencies,
    best_gap_near_window,
    k_grid,
    make_periodic_vector_cell,
    material_diagnostics,
    target_window_from_percentiles,
    train_periodic_vector,
    window_metrics,
)

DEFAULT_OUTPUT = REPO_ROOT / "scripts" / "outputs" / "periodic_update_ablation"


@dataclass(frozen=True)
class Variant:
    name: str
    inflation: float
    grad_clip: float | None
    radius_min: float | None
    radius_max: float | None
    alpha: float = 0.035
    family: str = "factorial"
    material_update: str = "radius_euler"
    response_metric_eta: float = 0.02
    response_metric_lambda_ratio: float = 0.025
    response_bound_mode: str = "clip"


def pilot_variants() -> list[Variant]:
    """Broad screen, deduplicated by the complete numerical protocol."""
    variants = [
        # Full 2 x 2 x 2 factorial comparison.
        Variant("infl_clip05_bounded", 0.02, 0.5, 0.5, 2.0),
        Variant("noinfl_clip05_bounded", 0.0, 0.5, 0.5, 2.0),
        Variant("infl_noclip_bounded", 0.02, None, 0.5, 2.0),
        Variant("noinfl_noclip_bounded", 0.0, None, 0.5, 2.0),
        Variant("infl_clip05_positive", 0.02, 0.5, 0.0, None),
        Variant("noinfl_clip05_positive", 0.0, 0.5, 0.0, None),
        Variant("infl_noclip_positive", 0.02, None, 0.0, None),
        Variant("noinfl_noclip_positive", 0.0, None, 0.0, None),
        # Saturation-strength sweep without inflation.
        Variant("noinfl_clip01_bounded", 0.0, 0.1, 0.5, 2.0, family="clip_sweep"),
        Variant("noinfl_clip025_bounded", 0.0, 0.25, 0.5, 2.0, family="clip_sweep"),
        Variant("noinfl_clip1_bounded", 0.0, 1.0, 0.5, 2.0, family="clip_sweep"),
        Variant("noinfl_clip2_bounded", 0.0, 2.0, 0.5, 2.0, family="clip_sweep"),
        Variant("noinfl_clip5_bounded", 0.0, 5.0, 0.5, 2.0, family="clip_sweep"),
        # Unsaturated step-size sweep without inflation.
        Variant("noinfl_noclip_a001_bounded", 0.0, None, 0.5, 2.0, 0.01, "alpha_sweep"),
        Variant("noinfl_noclip_a00035_bounded", 0.0, None, 0.5, 2.0, 0.0035, "alpha_sweep"),
        Variant("noinfl_noclip_a0001_bounded", 0.0, None, 0.5, 2.0, 0.001, "alpha_sweep"),
        Variant("noinfl_noclip_a000035_bounded", 0.0, None, 0.5, 2.0, 0.00035, "alpha_sweep"),
        # The same step-size sweep without an upper material bound.  These arms
        # distinguish a genuine need for local saturation from a large-step
        # instability hidden by projection onto the production bounds.
        Variant("noinfl_noclip_a001_positive", 0.0, None, 0.0, None, 0.01, "alpha_unbounded"),
        Variant("noinfl_noclip_a00035_positive", 0.0, None, 0.0, None, 0.0035, "alpha_unbounded"),
        Variant("noinfl_noclip_a0001_positive", 0.0, None, 0.0, None, 0.001, "alpha_unbounded"),
        Variant("noinfl_noclip_a000035_positive", 0.0, None, 0.0, None, 0.00035, "alpha_unbounded"),
        # Bound-width sweep under the simplest promising local rule.
        Variant("noinfl_clip05_midbounds", 0.0, 0.5, 0.25, 3.0, family="bounds_sweep"),
        Variant("noinfl_clip05_widebounds", 0.0, 0.5, 0.1, 4.0, family="bounds_sweep"),
    ]
    seen: set[tuple] = set()
    unique = []
    for variant in variants:
        key = (
            variant.inflation,
            variant.grad_clip,
            variant.radius_min,
            variant.radius_max,
            variant.alpha,
            variant.material_update,
            variant.response_metric_eta,
            variant.response_metric_lambda_ratio,
            variant.response_bound_mode,
        )
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def confirmation_variants() -> list[Variant]:
    """Default confirmation set; can be revised after inspecting the pilot."""
    return [
        Variant("infl_clip05_bounded", 0.02, 0.5, 0.5, 2.0),
        Variant("noinfl_clip05_bounded", 0.0, 0.5, 0.5, 2.0),
        Variant("noinfl_noclip_bounded", 0.0, None, 0.5, 2.0),
        Variant("noinfl_clip05_positive", 0.0, 0.5, 0.0, None),
        Variant(
            "response_log_bounded", 0.0, None, 0.5, 2.0,
            family="unified_rule", material_update="response_conditioned_log",
            response_bound_mode="clip",
        ),
        Variant(
            "response_log_unbounded", 0.0, None, None, None,
            family="unified_rule", material_update="response_conditioned_log",
            response_bound_mode="none",
        ),
    ]


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    temporary.replace(path)


def result_path(output: Path, stage: str, size: int, seed: int, name: str) -> Path:
    return output / stage / "runs" / f"size{size}_net{seed}_{name}.json"


def run_one(task: dict) -> dict:
    variant = Variant(**task["variant"])
    size = int(task["size"])
    net_seed = int(task["net_seed"])
    started = time.perf_counter()
    common = task["common"]
    try:
        cell = make_periodic_vector_cell(size, net_seed, common["topology"])
        initial = band_frequencies(
            cell,
            np.full(cell.n_edges, R_INIT),
            k_grid(common["k_eval_grid"]),
        )
        wlo, whi = target_window_from_percentiles(
            initial,
            common["center_percentile"],
            common["width_percentile"],
        )
        trained = train_periodic_vector(
            cell,
            wlo,
            whi,
            n_steps=common["steps"],
            alpha=variant.alpha,
            grad_clip=variant.grad_clip,
            inflation_strength=variant.inflation,
            radius_min=variant.radius_min,
            radius_max=variant.radius_max,
            n_freq=common["n_freq"],
            k_train_grid=common["k_train_grid"],
            k_eval_grid=common["k_eval_grid"],
            k_batch=common["k_batch"],
            force_batch=common["force_batch"],
            seed=common["train_seed"],
            eval_every=common["eval_every"],
            update_rule="adjoint",
            regularizer="inflation" if variant.inflation else "none",
            response_mode="reciprocal",
            material_update=variant.material_update,
            response_metric_eta=variant.response_metric_eta,
            response_metric_lambda_ratio=variant.response_metric_lambda_ratio,
            response_bound_mode=variant.response_bound_mode,
            frequency_sampling="random_grid",
            frequencies_per_step=1,
        )
        radii = np.asarray(trained["radii"], dtype=np.float64)
        finite = bool(np.all(np.isfinite(radii)))
        nonnegative = bool(np.all(radii >= 0.0))
        if not finite:
            raise FloatingPointError("training produced non-finite radii")
        dense_k = k_grid(common["k_dense_grid"])
        dense_frequencies = band_frequencies(cell, radii, dense_k)
        metrics = window_metrics(dense_frequencies, wlo, whi)
        gap = best_gap_near_window(dense_frequencies, wlo, whi, dense_k)
        success = bool(
            nonnegative
            and metrics["n_in"] == 0
            and gap["gap"] > 0.0
            and gap["contains_target_window"]
        )
        return {
            "status": "complete",
            "stage": task["stage"],
            "variant": asdict(variant),
            "size": size,
            "net_seed": net_seed,
            "train_seed": common["train_seed"],
            "target": {"wlo": wlo, "whi": whi},
            "success": success,
            "metrics": metrics,
            "band_gap": gap,
            "material": material_diagnostics(
                cell, radii, variant.radius_min, variant.radius_max,
            ),
            "radius_range": {
                "min": float(np.min(radii)),
                "median": float(np.median(radii)),
                "max": float(np.max(radii)),
                "finite": finite,
                "nonnegative": nonnegative,
            },
            "radii": radii,
            "history": trained["history"],
            "elapsed_s": float(time.perf_counter() - started),
            "protocol": common,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "stage": task["stage"],
            "variant": asdict(variant),
            "size": size,
            "net_seed": net_seed,
            "train_seed": common["train_seed"],
            "success": False,
            "elapsed_s": float(time.perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "protocol": common,
        }


def median(records: list[dict], path: tuple[str, ...]) -> float | None:
    values = []
    for record in records:
        value = record
        try:
            for key in path:
                value = value[key]
            value = float(value)
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.median(values)) if values else None


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        key = (record["variant"]["name"], int(record["size"]))
        groups.setdefault(key, []).append(record)
    rows = []
    for (name, size), group in sorted(groups.items()):
        complete = [record for record in group if record["status"] == "complete"]
        rows.append({
            "variant": name,
            "family": group[0]["variant"]["family"],
            "size": size,
            "attempted": len(group),
            "complete": len(complete),
            "successes": sum(bool(record["success"]) for record in complete),
            "success_rate": (
                sum(bool(record["success"]) for record in complete) / len(complete)
                if complete else None
            ),
            "median_n_in": median(complete, ("metrics", "n_in")),
            "median_normalized_gap": median(complete, ("band_gap", "normalized_gap")),
            "median_material_ratio": median(complete, ("material", "material_ratio")),
            "median_bound_fraction": median(complete, ("material", "frac_at_bounds")),
            "median_min_radius": median(complete, ("radius_range", "min")),
            "median_max_radius": median(complete, ("radius_range", "max")),
        })
    return rows


def load_stage_records(output: Path, stage: str) -> list[dict]:
    paths = sorted((output / stage / "runs").glob("*.json"))
    return [json.loads(path.read_text()) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "confirm"), default="pilot")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument(
        "--variants", nargs="+",
        help="Run only the named variants from the selected stage.",
    )
    args = parser.parse_args()

    if args.stage == "pilot":
        variants = pilot_variants()
        sizes = args.sizes or [5]
        seeds = args.seeds or 4
        steps = args.steps or 800
        dense_grid = 17
    else:
        variants = confirmation_variants()
        sizes = args.sizes or [5, 6, 8]
        seeds = args.seeds or 10
        steps = args.steps or 1200
        dense_grid = 33
    if args.variants:
        requested = set(args.variants)
        variants = [variant for variant in variants if variant.name in requested]
        missing = requested - {variant.name for variant in variants}
        if missing:
            raise SystemExit(f"unknown variants for stage {args.stage}: {sorted(missing)}")

    common = {
        "topology": "rand-del",
        "center_percentile": 50.0,
        "width_percentile": 10.0,
        "steps": int(steps),
        "n_freq": 5,
        "k_train_grid": 5,
        "k_eval_grid": 9,
        "k_dense_grid": dense_grid,
        "k_batch": 4,
        "force_batch": 2,
        "train_seed": 0,
        "eval_every": max(100, steps // 4),
    }
    tasks = []
    for size in sizes:
        for net_seed in range(int(seeds)):
            for variant in variants:
                path = result_path(args.output, args.stage, size, net_seed, variant.name)
                if args.force or not path.exists():
                    tasks.append({
                        "stage": args.stage,
                        "size": size,
                        "net_seed": net_seed,
                        "variant": asdict(variant),
                        "common": common,
                        "path": str(path),
                    })

    print(
        f"stage={args.stage} variants={len(variants)} sizes={sizes} "
        f"seeds={seeds} pending={len(tasks)} workers={args.workers}",
        flush=True,
    )
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {pool.submit(run_one, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                record = future.result()
                atomic_json(Path(task["path"]), record)
                print(
                    f"[{completed}/{len(tasks)}] size={record['size']} "
                    f"net={record['net_seed']} variant={record['variant']['name']} "
                    f"status={record['status']} success={record['success']} "
                    f"elapsed={record['elapsed_s']:.1f}s",
                    flush=True,
                )

    records = load_stage_records(args.output, args.stage)
    summary = summarize(records)
    payload = {
        "stage": args.stage,
        "protocol": common,
        "variants": [asdict(variant) for variant in variants],
        "records": len(records),
        "summary": summary,
    }
    atomic_json(args.output / args.stage / "summary.json", payload)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
