"""Small configuration loader shared by the paper scripts."""
from __future__ import annotations

import json
import os


def merge_config_defaults(config, defaults):
    """Recursively merge a user configuration with default values."""
    merged = {}
    for key, default_value in defaults.items():
        if key in config:
            if isinstance(default_value, dict) and isinstance(config[key], dict):
                merged[key] = merge_config_defaults(config[key], default_value)
            else:
                merged[key] = config[key]
        else:
            merged[key] = default_value
    for key in config:
        if key not in merged:
            merged[key] = config[key]
    return merged


def load_json_config(filepath, defaults=None, validate=True):
    """Load JSON/YAML, resolve repository-relative paths, and apply defaults."""
    if not os.path.isfile(filepath):
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        relative_path = filepath
        while relative_path.startswith("../"):
            relative_path = relative_path[3:]
        if relative_path.startswith("./"):
            relative_path = relative_path[2:]
        candidate = os.path.normpath(os.path.join(repo_root, relative_path))
        if os.path.isfile(candidate):
            filepath = candidate

    with open(filepath) as stream:
        if filepath.endswith((".yaml", ".yml")):
            import yaml

            config = yaml.safe_load(stream)
        else:
            config = json.load(stream)

    config = merge_config_defaults(config, defaults or {})
    if validate:
        network = config.get("network", {})
        if network.get("preset") is None and network.get("type") is None:
            raise ValueError("Config must specify either network.type or network.preset")
    return config
