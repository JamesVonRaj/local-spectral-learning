"""Stress-test the fixed scalar material rule without parameter retuning.

The matrix varies network size, realization, target position, random stream,
and topology. Detailed outputs are regenerable diagnostics under
``scripts/outputs/response_metric/validation`` and are ignored by Git.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from mechanical.generate import generate_from_config
from mechanical.learners import eigenfreqs
from mechanical.learners_local import train_local
from publication.paths import dataset

OUT_DIR = dataset("response_metric") / "validation"
ETA = 0.02
LAMBDA_RATIO = 0.025
N_STEPS = 3000


def validation_configs():
    configs = []
    for size in (10, 15, 20, 25):
        for net_seed in range(3):
            configs.append({
                "family": "size", "topology": "rand-del",
                "network_size": size, "net_seed": net_seed,
                "train_seed": 0, "frequency_seed": 918273,
                "window_percentiles": (35.0, 65.0),
            })
    for window in ((15.0, 35.0), (35.0, 65.0), (65.0, 85.0)):
        for net_seed in range(5):
            configs.append({
                "family": "window", "topology": "rand-del",
                "network_size": 20, "net_seed": net_seed,
                "train_seed": 0, "frequency_seed": 918273,
                "window_percentiles": window,
            })
    for stream_index in range(5):
        configs.append({
            "family": "random_stream", "topology": "rand-del",
            "network_size": 20, "net_seed": 0,
            "train_seed": stream_index,
            "frequency_seed": 918273 + 7919 * stream_index,
            "window_percentiles": (35.0, 65.0),
        })
    for net_seed in range(5):
        configs.append({
            "family": "topology", "topology": "ws",
            "network_size": 20, "net_seed": net_seed,
            "train_seed": 0, "frequency_seed": 918273,
            "window_percentiles": (35.0, 65.0),
        })

    unique = {}
    for config in configs:
        key = (
            config["topology"], config["network_size"], config["net_seed"],
            config["train_seed"], config["frequency_seed"],
            tuple(config["window_percentiles"]),
        )
        if key in unique:
            unique[key]["family"] += "+" + config["family"]
        else:
            unique[key] = dict(config)
    return list(unique.values())


def _positive(freqs):
    ordered = np.sort(np.asarray(freqs, dtype=np.float64))
    tolerance = 1e-8 * max(float(np.max(ordered)), 1.0)
    positive = ordered[ordered > tolerance]
    if len(positive) == len(ordered):
        positive = ordered[1:]
    return positive


def _counts(freqs, wlo, whi):
    positive = _positive(freqs)
    return {
        "below": int(np.sum(positive <= wlo)),
        "inside": int(np.sum((positive > wlo) & (positive < whi))),
        "above": int(np.sum(positive >= whi)),
    }


def _tag(config):
    lo, hi = config["window_percentiles"]
    return (
        f"{config['family']}_{config['topology']}_s{config['network_size']}"
        f"_net{config['net_seed']}_train{config['train_seed']}"
        f"_freq{config['frequency_seed']}_p{lo:g}-{hi:g}"
    ).replace(".", "p").replace("+", "-")


def _evaluate(config):
    network = generate_from_config({
        "network": {
            "topology": config["topology"],
            "size": config["network_size"],
            "seed": config["net_seed"],
        },
        "training": {},
    })
    edges = network["edges"]
    lengths = network["lengths"]
    n_nodes = len(network["pos"])
    initial = eigenfreqs(edges, np.ones(len(edges)), lengths, n_nodes, 1.0)
    wlo, whi = (
        float(x) for x in np.percentile(
            _positive(initial), config["window_percentiles"],
        )
    )
    result = train_local(
        edges=edges, lengths=lengths, N=n_nodes,
        n_steps=N_STEPS, batch=10, n_freq=8,
        grad_clip=None, damping="real_shift", regularizers=[],
        material_update="response_conditioned_log",
        response_metric_eta=ETA,
        response_metric_lambda_ratio=LAMBDA_RATIO,
        target_window=(wlo, whi),
        frequency_sampling="random_grid", frequencies_per_step=1,
        train_seed=config["train_seed"],
        frequency_seed=config["frequency_seed"],
        eval_every=500,
    )

    final = _counts(result["ff"], wlo, whi)
    stiffness_ratio = float(
        np.sum(result["radii"] ** 2 / lengths) / np.sum(1.0 / lengths)
    )
    uniform = _counts(initial * np.sqrt(stiffness_ratio), wlo, whi)
    n_modes = sum(final.values())
    minimum_side = max(3, int(np.ceil(0.05 * n_modes)))
    checks = {
        "cleared": final["inside"] == 0,
        "two_sided": min(final["below"], final["above"]) >= minimum_side,
        "not_uniform_rescaling": uniform["inside"] > 0,
    }
    record = {
        "config": config,
        "target_window": [wlo, whi],
        "initial": _counts(initial, wlo, whi),
        "final": final,
        "stiffness_ratio": stiffness_ratio,
        "uniform_same_stiffness": uniform,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    output = OUT_DIR / f"{_tag(config)}.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    record["output"] = str(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        records = list(pool.map(_evaluate, validation_configs()))
    summary = {
        "fixed_parameters": {
            "eta": ETA,
            "lambda_ratio": LAMBDA_RATIO,
            "n_steps": N_STEPS,
        },
        "n_runs": len(records),
        "n_passed": int(sum(record["passed"] for record in records)),
        "records": records,
    }
    summary_path = OUT_DIR / "validation_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_runs": summary["n_runs"],
        "n_passed": summary["n_passed"],
        "summary": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
