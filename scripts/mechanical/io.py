"""Configuration helpers for the mechanical paper experiments."""
from __future__ import annotations

from common.io import load_json_config as _load_json_config

MECHANICAL_DEFAULT_CONFIG = {
    "pipeline": {"generate": True, "train": True, "plot": True},
    "network": {
        "topology": "rand-del",
        "size": 20,
        "seed": 0,
        "preset": None,
    },
    "training": {
        "mode": "local",
        "n_steps": 3000,
        "batch": 10,
        "n_freq": 8,
        "force_distribution": "gaussian",
        "monitor_batch": 0,
        "monitor_every": None,
        "monitor_seed": None,
        "monitor_force_distribution": None,
        "alpha": 0.05,
        "grad_clip": None,
        "material_update": "response_conditioned_log",
        "response_metric_eta": 0.02,
        "response_metric_lambda_ratio": 0.025,
        "damping": "real_shift",
        "M_node": 1.0,
        "R_INIT": 1.0,
        "R_MIN": 0.5,
        "R_MAX": 2.0,
        "radius_parameterization": "clip",
        "warmup_steps": 0,
        "eval_every": 500,
        "snapshot_every": 0,
        "snapshot_dtype": "float32",
        "train_seed": 0,
        "regularizers": None,
        "objective": None,
        "constraint": None,
    },
    "plotting": {"dos": True, "radii": True},
    "paths": {"output_dir": "../outputs/mechanical", "data_file": None},
}


def load_mechanical_config(filepath):
    """Load and validate a mechanical-pipeline JSON or YAML config."""
    config = _load_json_config(
        filepath, defaults=MECHANICAL_DEFAULT_CONFIG, validate=False,
    )
    network = config["network"]
    if network.get("preset") is None and network.get("topology") is None:
        raise ValueError(
            "Mechanical config must specify either network.topology or network.preset"
        )
    return config


def local_regularizer_arm_to_specs(arm):
    """Translate a paper regularizer-arm label into local JSON specs."""
    specs = {
        "none": [],
        "inflate": [{"type": "inflation", "strength": 0.03}],
        "l1": [{"type": "l1_sparsity", "strength": 0.02, "length_weighted": True}],
        "l2": [{"type": "l2_pull", "target": 1.0, "strength": 0.01}],
        "material": [{"type": "material_volume", "strength": 0.01}],
        "stiffness": [{"type": "stiffness_penalty", "strength": 0.01}],
        "binary": [{"type": "binary_radius", "strength": 0.01, "low": 0.5, "high": 2.0}],
        "inflate-l1": [
            {"type": "inflation", "strength": 0.03},
            {"type": "l1_sparsity", "strength": 0.01},
        ],
        "inflate-l2": [
            {"type": "inflation", "strength": 0.03},
            {"type": "l2_pull", "target": 1.0, "strength": 0.01},
        ],
    }
    try:
        return specs[arm]
    except KeyError as exc:
        raise ValueError(
            f"unknown local regularizer arm {arm!r}; expected one of {sorted(specs)}"
        ) from exc
