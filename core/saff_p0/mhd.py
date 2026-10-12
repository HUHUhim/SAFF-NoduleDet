from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Page cache control (mandatory on memory-constrained instances)
#
# The cgroup memory limit on this machine is only 120 GiB, and page cache is
# counted into memory.current. Linux lets page cache greedily fill free memory,
# so memory.current stays pinned at 100% -> the [host-side] OOM killer fires
# (this cgroup's oom_kill count is 0; the record of the kill lives in the parent
# cgroup and cannot be seen from inside the container). Measured 2026-09-16
# 02:53 was exactly such a kill:
#   the cgroup held at 99-100% for several minutes, then the training process
#   received SIGKILL (anon was in fact only 60 GB).
# `memory.high` is read-only in this container (Read-only file system), so no
# soft limit can be set; the only reliable remedy is to drop on read:
# POSIX_FADV_DONTNEED. The price is that page cache is no longer reused across
# epochs, but block sampling already reads each scan only once per epoch, and
# the measured hit rate in that regime was low anyway, so barely any speed is
# lost.
# ---------------------------------------------------------------------------
_DROP_PAGE_CACHE = True


def set_drop_page_cache(enabled: bool) -> None:
    global _DROP_PAGE_CACHE
    _DROP_PAGE_CACHE = bool(enabled)


def drop_page_cache_enabled() -> bool:
    return _DROP_PAGE_CACHE


def _read_bytes_no_cache(path: Path) -> bytes:
    """Read a file; if enabled, drop it from the page cache via posix_fadvise as soon as it has been read."""
    if not _DROP_PAGE_CACHE or not hasattr(os, "posix_fadvise"):
        return path.read_bytes()
    with path.open("rb") as f:
        data = f.read()
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    return data


@dataclass(frozen=True)
class MhdInfo:
    path: Path
    data_file: Path
    dim_size_xyz: np.ndarray
    spacing_xyz: np.ndarray
    origin_xyz: np.ndarray
    direction: np.ndarray
    element_type: str
    compressed: bool

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x, y, z = [int(v) for v in self.dim_size_xyz.tolist()]
        return z, y, x

    @property
    def spacing_zyx(self) -> np.ndarray:
        return self.spacing_xyz[::-1].astype(np.float32)


def _parse_vector(value: str, dtype=float) -> np.ndarray:
    return np.array([dtype(x) for x in value.strip().split()], dtype=dtype)


def parse_mhd(path: str | Path) -> MhdInfo:
    path = Path(path)
    meta: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    origin_key = "Offset" if "Offset" in meta else "Origin"
    data_file = path.parent / meta["ElementDataFile"]
    return MhdInfo(
        path=path,
        data_file=data_file,
        dim_size_xyz=_parse_vector(meta["DimSize"], int),
        spacing_xyz=_parse_vector(meta["ElementSpacing"], float),
        origin_xyz=_parse_vector(meta[origin_key], float),
        direction=_parse_vector(meta.get("TransformMatrix", "1 0 0 0 1 0 0 0 1"), float).reshape(3, 3),
        element_type=meta["ElementType"],
        compressed=meta.get("CompressedData", "False").lower() == "true",
    )


def dtype_from_mhd(element_type: str) -> np.dtype:
    mapping = {
        "MET_CHAR": np.int8,
        "MET_UCHAR": np.uint8,
        "MET_SHORT": np.int16,
        "MET_USHORT": np.uint16,
        "MET_INT": np.int32,
        "MET_UINT": np.uint32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }
    if element_type not in mapping:
        raise ValueError(f"Unsupported MHD ElementType: {element_type}")
    return np.dtype(mapping[element_type])


def load_mhd_volume(info: MhdInfo) -> np.ndarray:
    raw_bytes = _read_bytes_no_cache(info.data_file)
    if info.compressed or info.data_file.suffix.lower() == ".zraw":
        raw_bytes = zlib.decompress(raw_bytes)
    arr = np.frombuffer(raw_bytes, dtype=dtype_from_mhd(info.element_type))
    expected = int(np.prod(info.dim_size_xyz))
    if arr.size != expected:
        raise ValueError(f"Unexpected voxel count for {info.path}: got {arr.size}, expected {expected}")
    return arr.reshape(info.shape_zyx)


def world_to_voxel_xyz(world_xyz: np.ndarray, info: MhdInfo) -> np.ndarray:
    direction_inv = np.linalg.inv(info.direction)
    index_scaled = direction_inv.dot(world_xyz.astype(np.float64) - info.origin_xyz.astype(np.float64))
    return index_scaled / info.spacing_xyz.astype(np.float64)


def voxel_xyz_to_world(voxel_xyz: np.ndarray, info: MhdInfo) -> np.ndarray:
    return info.origin_xyz.astype(np.float64) + info.direction.dot(voxel_xyz * info.spacing_xyz.astype(np.float64))


def world_to_voxel_zyx(world_xyz: np.ndarray, info: MhdInfo) -> np.ndarray:
    return world_to_voxel_xyz(world_xyz, info)[::-1]


def voxel_zyx_to_world(voxel_zyx: np.ndarray, info: MhdInfo) -> np.ndarray:
    return voxel_xyz_to_world(voxel_zyx[::-1], info)

