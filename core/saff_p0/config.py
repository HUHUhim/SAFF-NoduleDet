from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any


SUPPORTED_SAFF_ABLATIONS = (
    "no_distill",
    "no_morph_gate",
    "fixed_fusion",
    "cnn_only",
    "transformer_only",
    "global_norm",
    "full_attention",
)

SUPPORTED_SAFF_VARIANTS = ("full",) + SUPPORTED_SAFF_ABLATIONS


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg["_config_path"] = str(path)
    return cfg


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def set_seed(seed: int, deterministic: bool = False) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_yaml(path: str | Path, obj: Any) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def output_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["output_root"]).expanduser().resolve()


def resolve_luna_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["luna_root"]).expanduser().resolve()
