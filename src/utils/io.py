"""Small filesystem helpers shared across the pipeline."""
import json
import os

import numpy as np


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_artifacts(artifacts: dict, path: str) -> None:
    """Save the nested {model_name: {field: array}} artifacts dict to one
    compressed .npz file. Keys are flattened as "model__field" since npz
    only supports a flat namespace; load_artifacts() below reverses this.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {
        f"{model_name}__{field_name}": np.asarray(value)
        for model_name, payload in artifacts.items()
        for field_name, value in payload.items()
    }
    np.savez_compressed(path, **flat)


def load_artifacts(path: str) -> dict:
    """Inverse of save_artifacts: .npz -> {model_name: {field: array}}."""
    data = np.load(path, allow_pickle=False)
    artifacts = {}
    for key in data.files:
        model_name, field_name = key.split("__", 1)
        artifacts.setdefault(model_name, {})[field_name] = data[key]
    return artifacts
