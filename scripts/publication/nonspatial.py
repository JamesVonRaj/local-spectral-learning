"""Reproduce the nonspatial Watts--Strogatz screen in the supplement."""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from mechanical.generate import generate_from_config
from mechanical.learners_local import train_local

from publication.paths import FIGURE_DIR, dataset

DATA_DIR = dataset("prl_nonspatial")
FIG_DIR = FIGURE_DIR
N_STEPS = 3000
N_SEEDS = 5
SIZE = 20
ETA = 0.02
LAMBDA_RATIO = 0.025


def _cache_is_current(data):
    return (
        str(np.asarray(data.get("topology", "")).item()) == "ws"
        and int(np.asarray(data.get("n_steps", -1)).item()) == N_STEPS
        and int(np.asarray(data.get("n_seeds", -1)).item()) == N_SEEDS
        and int(np.asarray(data.get("size", -1)).item()) == SIZE
        and np.isclose(float(data.get("eta", np.nan)), ETA)
        and np.isclose(float(data.get("lambda_ratio", np.nan)), LAMBDA_RATIO)
        and str(np.asarray(data.get("material_update", "")).item())
        == "response_conditioned_log"
    )


def _train(seed):
    network = generate_from_config({
        "network": {"topology": "ws", "size": SIZE, "seed": int(seed)},
        "training": {},
    })
    started = time.perf_counter()
    result = train_local(
        edges=network["edges"], lengths=network["lengths"],
        N=len(network["pos"]), n_steps=N_STEPS, batch=10, n_freq=8,
        grad_clip=None, damping="real_shift", regularizers=[],
        material_update="response_conditioned_log",
        response_metric_eta=ETA,
        response_metric_lambda_ratio=LAMBDA_RATIO,
        frequency_sampling="random_grid", frequencies_per_step=1,
        train_seed=0, frequency_seed=918273, eval_every=500,
    )
    elapsed = time.perf_counter() - started
    print(
        f"WS seed={seed}: n_in={result['n_in_initial']}"
        f"->{result['n_in_final']} gap={result['gap_ratio']:.3f} "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return (
        int(seed), len(network["pos"]), int(result["n_in_initial"]),
        int(result["n_in_final"]), float(result["gap_ratio"]), elapsed,
    )


def run_screen(force=False):
    path = DATA_DIR / "ws_screen.npz"
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as data:
            if _cache_is_current(data):
                return data["records"]
    records = np.asarray(
        [_train(seed) for seed in range(N_SEEDS)],
        dtype=[
            ("seed", "i4"), ("N", "i4"), ("n_in_initial", "i4"),
            ("n_in_final", "i4"), ("gap_ratio", "f8"),
            ("seconds", "f8"),
        ],
    )
    np.savez_compressed(
        path, records=records, topology=np.asarray("ws"),
        n_steps=np.int64(N_STEPS), n_seeds=np.int64(N_SEEDS),
        size=np.int64(SIZE), eta=np.float64(ETA),
        lambda_ratio=np.float64(LAMBDA_RATIO),
        material_update=np.asarray("response_conditioned_log"),
    )
    return records


def write_table(records):
    lines = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"seed & $N$ & initial $N_{\rm in}$ & final $N_{\rm in}$ & $\Delta\omega/\omega_{\rm mid}$ \\",
        r"\midrule",
    ]
    for row in records:
        gap = f"{float(row['gap_ratio']):.3f}" if int(row["n_in_final"]) == 0 else "--"
        lines.append(
            f"{int(row['seed'])} & {int(row['N'])} & "
            f"{int(row['n_in_initial'])} & {int(row['n_in_final'])} & {gap} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (FIG_DIR / "table_nonspatial_gap.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = run_screen(force=args.force)
    write_table(records)


if __name__ == "__main__":
    main()
