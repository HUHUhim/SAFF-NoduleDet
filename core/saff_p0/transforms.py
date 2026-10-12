from __future__ import annotations

from collections import OrderedDict

import numpy as np


class VolumeCache:
    def __init__(self, max_items: int = 2):
        self.max_items = max(0, int(max_items))
        self._cache: OrderedDict[str, tuple[np.ndarray, object]] = OrderedDict()

    def get(self, key: str):
        if self.max_items <= 0:
            return None
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def put(self, key: str, value):
        if self.max_items <= 0:
            return
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


def clip_and_normalize_hu(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    x = np.clip(volume.astype(np.float32), hu_min, hu_max)
    x = (x - hu_min) / max(1e-6, hu_max - hu_min)
    return x * 2.0 - 1.0


def _standardize_values(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std() + 1e-6)


def lung_window_standardize(
    volume: np.ndarray,
    hu_min: float,
    hu_max: float,
    mode: str = "global",
    lobe_mask: np.ndarray | None = None,
    lung_mask: np.ndarray | None = None,
    regional_regions: int = 5,
) -> np.ndarray:
    clipped = np.clip(volume.astype(np.float32), hu_min, hu_max)
    lung_bool = None
    if lung_mask is not None:
        lung_bool = lung_mask > 0
        if not lung_bool.any():
            lung_bool = None
    if mode == "lobe" and lobe_mask is not None:
        out = np.empty_like(clipped, dtype=np.float32)
        used = np.zeros(clipped.shape, dtype=bool)
        for label in [1, 2, 3, 4, 5]:
            mask = lobe_mask == label
            if not mask.any():
                continue
            vals = clipped[mask]
            mean, std = _standardize_values(vals)
            out[mask] = (clipped[mask] - mean) / std
            used |= mask
        if used.any():
            vals = clipped[used]
            mean, std = _standardize_values(vals)
            out[~used] = (clipped[~used] - mean) / std
            return out
        mode = "regional"
    elif mode == "lobe":
        mode = "regional"

    if mode == "global":
        mask = (clipped > hu_min + 1) & (clipped < hu_max - 1)
        if lung_bool is not None:
            mask &= lung_bool
        vals = clipped[mask] if mask.any() else clipped.reshape(-1)
        mean, std = _standardize_values(vals)
        return ((clipped - mean) / std).astype(np.float32)

    if mode == "regional":
        z = clipped.shape[0]
        out = np.empty_like(clipped, dtype=np.float32)
        regions = max(1, int(regional_regions))
        boundaries = np.linspace(0, z, regions + 1, dtype=int)
        for i in range(regions):
            lo, hi = int(boundaries[i]), int(boundaries[i + 1])
            slab = clipped[lo:hi]
            mask = (slab > hu_min + 1) & (slab < hu_max - 1)
            if lung_bool is not None:
                mask &= lung_bool[lo:hi]
            vals = slab[mask] if mask.any() else slab.reshape(-1)
            mean, std = _standardize_values(vals)
            out[lo:hi] = (slab - mean) / std
        return out

    if mode == "minmax":
        return clip_and_normalize_hu(volume, hu_min, hu_max)

    raise ValueError(f"Unknown normalization mode: {mode}")


def crop_patch(volume: np.ndarray, center_zyx: np.ndarray, patch_size: int) -> np.ndarray:
    patch_size = int(patch_size)
    half = patch_size // 2
    center = np.rint(center_zyx).astype(int)
    starts = center - half
    ends = starts + patch_size
    src_slices = []
    dst_slices = []
    for axis, (start, end, size) in enumerate(zip(starts, ends, volume.shape)):
        src_start = max(0, int(start))
        src_end = min(int(size), int(end))
        dst_start = max(0, -int(start))
        dst_end = dst_start + max(0, src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
    patch[tuple(dst_slices)] = volume[tuple(src_slices)]
    return patch


def augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = patch
    for axis in range(3):
        if rng.random() < 0.5:
            x = np.flip(x, axis=axis)
    if rng.random() < 0.5:
        k = int(rng.integers(0, 4))
        x = np.rot90(x, k=k, axes=(1, 2))
    if rng.random() < 0.35:
        x = x + rng.normal(0.0, 0.03, size=x.shape).astype(np.float32)
    if rng.random() < 0.35:
        x = x * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.05, 0.05))
    return np.ascontiguousarray(x.astype(np.float32))
