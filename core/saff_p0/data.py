from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mhd import MhdInfo, load_mhd_volume, parse_mhd, world_to_voxel_zyx
from .transforms import VolumeCache, augment_patch, crop_patch, lung_window_standardize


# ---------------------------------------------------------------------------
# Process-level volume cache
#
# Motivation: every sample must read a .raw file and perform whole-volume regional
# standardization (about 0.5~1.5 s per sample). If each sample reloaded it, data loading
# would become the absolute bottleneck; yet a single CT volume is reused by tens of
# thousands of samples from the same fold. Placing these read-only volumes in a
# module-level dictionary lets DataLoader workers share the same physical memory through
# fork (a read only bumps the object header's reference count and never triggers
# copy-on-write of the volume pages), which both saves memory and costs almost no I/O.
# The key includes the normalization mode and the region count, so different ablation
# conditions cannot contaminate one another.
# ---------------------------------------------------------------------------
_VOLUME_STORE: dict[tuple[str, int, str], tuple[np.ndarray, Any]] = {}


def volume_store_size() -> int:
    return len(_VOLUME_STORE)


def volume_store_bytes() -> int:
    return int(sum(v[0].nbytes for v in _VOLUME_STORE.values()))


def _read_int_file(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as f:
            value = f.read().strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cgroup_memory_limit() -> int | None:
    """True container memory limit (cgroup v2 preferred, falling back to v1)."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read_int_file(path)
        if v is not None and 0 < v < (1 << 62):
            return v
    return None


def _cgroup_memory_used() -> int | None:
    """Memory already used by the cgroup that is [not reclaimable] = memory.current minus page cache."""
    current = _read_int_file("/sys/fs/cgroup/memory.current")
    if current is None:
        current = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if current is None:
        return None
    file_cache = None
    for path, key in (("/sys/fs/cgroup/memory.stat", "file"),
                      ("/sys/fs/cgroup/memory/memory.stat", "total_cache")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == key:
                        file_cache = int(parts[1])
                        break
        except OSError:
            continue
        if file_cache is not None:
            break
    return current if file_cache is None else max(0, current - file_cache)


def _mem_available_bytes() -> int | None:
    """Memory budget available for preloading (bytes).

    WARNING: /proc/meminfo and the cgroup must both be consulted: the AutoDL container
    does not mount lxcfs, so `/proc/meminfo` reports the [host] memory (measured
    1007 GiB), while the container's real limit lives in the cgroup (measured
    memory.max = 120 GiB). Using meminfo alone yields a 687 GB budget -> it blows
    straight through 120 GiB -> the process gets SIGKILLed (this actually happened once
    at 2026-09-16 00:31, killed at ~120 GB of preload).
    """
    available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        available = None

    limit = _cgroup_memory_limit()
    if limit is not None:
        used = _cgroup_memory_used() or 0
        headroom = max(0, limit - used)
        available = headroom if available is None else min(available, headroom)
    return available


def preload_volumes(datasets, max_volumes: int = 0, budget_fraction: float = 0.7, log=print) -> int:
    """Standardize all scans used by the given datasets in a single pass into the process-level cache; returns the number of cached entries."""
    available = _mem_available_bytes()
    budget = int(available * budget_fraction) if available else None
    ds = datasets[0]
    ids: list[str] = []
    seen: set[str] = set()
    for d in datasets:
        for s in d.samples:
            if s.seriesuid not in seen:
                seen.add(s.seriesuid)
                ids.append(s.seriesuid)
    if max_volumes and max_volumes > 0:
        ids = ids[:max_volumes]

    t0 = time.time()
    used = volume_store_bytes()
    for i, uid in enumerate(ids, 1):
        key = ds.store_key(uid)
        if key in _VOLUME_STORE:
            continue
        if budget is not None and used > budget:
            log(f"[preload] reached the memory budget of {budget/1e9:.0f} GB; stopping after {i-1}/{len(ids)} scans loaded")
            break
        vol, info = ds.load_scan(uid, bypass_store=True)
        _VOLUME_STORE[key] = (vol, info)
        used += int(vol.nbytes)
        if i % 100 == 0:
            log(f"[preload] {i}/{len(ids)}, cumulative {used/1e9:.1f} GB, elapsed {time.time()-t0:.0f} s")
    log(f"[preload] done: {len(_VOLUME_STORE)} scans, {volume_store_bytes()/1e9:.1f} GB, elapsed {time.time()-t0:.0f} s")
    return len(_VOLUME_STORE)


@dataclass
class Sample:
    seriesuid: str
    coord_x: float
    coord_y: float
    coord_z: float
    label: int
    diameter_mm: float
    source: str

    @property
    def world_xyz(self) -> np.ndarray:
        return np.array([self.coord_x, self.coord_y, self.coord_z], dtype=np.float64)


def csv_path(luna_root: Path, filename: str) -> Path:
    candidates = [
        luna_root / "CSVFILES" / filename,
        luna_root / filename,
        luna_root / "csvfiles" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} under {luna_root}")


def build_mhd_index(luna_root: Path, scan_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(scan_root).expanduser().resolve() if scan_root else luna_root
    if not root.exists():
        raise FileNotFoundError(f"Scan root does not exist: {root}")
    paths = sorted(root.rglob("*.mhd"))
    if not paths:
        raise FileNotFoundError(f"No .mhd files found under {root}")
    return {p.stem: p for p in paths}


def load_annotations(luna_root: Path, mhd_index: dict[str, Path]) -> pd.DataFrame:
    ann = pd.read_csv(csv_path(luna_root, "annotations.csv"))
    ann = ann[ann["seriesuid"].isin(mhd_index.keys())].copy()
    ann["diameter_mm"] = ann["diameter_mm"].astype(float)
    return ann.reset_index(drop=True)


def load_candidates(luna_root: Path, mhd_index: dict[str, Path]) -> pd.DataFrame:
    cand = pd.read_csv(csv_path(luna_root, "candidates.csv"))
    cand = cand[cand["seriesuid"].isin(mhd_index.keys())].copy()
    cand["class"] = cand["class"].astype(int)
    return cand.reset_index(drop=True)


def official10_subset_id(seriesuid: str, mhd_path: Path) -> int | None:
    for part in mhd_path.parts:
        low = part.lower()
        if low.startswith("subset"):
            tail = low.replace("subset", "")
            if tail.isdigit():
                return int(tail)
    return None


def make_folds(
    series: list[str],
    mhd_index: dict[str, Path],
    num_folds: int,
    strategy: str,
    seed: int,
) -> list[list[str]]:
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    folds = [[] for _ in range(num_folds)]
    # ``official10`` assigns each official LUNA16 subset to one held-out fold.
    # Refuse any other fold count so an accidental alternate split cannot be
    # mistaken for the official protocol.
    if strategy == "official10":
        if num_folds != 10:
            raise ValueError("fold_strategy=official10 requires num_folds=10")
        for sid in series:
            subset_id = official10_subset_id(sid, mhd_index[sid])
            if subset_id is None or not 0 <= subset_id < 10:
                raise ValueError(f"Could not determine official subset0-subset9 for {sid}")
            folds[subset_id].append(sid)
        if not all(folds):
            empty = [i for i, fold in enumerate(folds) if not fold]
            raise ValueError(f"Official ten-fold split has empty subsets: {empty}")
        return [sorted(f) for f in folds]

    rng = random.Random(seed)
    shuffled = list(series)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(num_folds)]
    for i, sid in enumerate(shuffled):
        folds[i % num_folds].append(sid)
    return [sorted(f) for f in folds]


def split_for_fold(
    folds: list[list[str]],
    fold_index: int,
    val_fraction_from_train_pool: float = 0.125,
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    test = sorted(folds[fold_index])
    train_pool = sorted([sid for i, fold in enumerate(folds) if i != fold_index for sid in fold])
    rng = random.Random(seed + fold_index * 1009)
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(train_pool) * float(val_fraction_from_train_pool))))
    val = sorted(shuffled[:val_count])
    val_set = set(val)
    train = sorted([sid for sid in train_pool if sid not in val_set])
    return train, val, test


class LunaCandidateDataset:
    def __init__(
        self,
        samples: list[Sample],
        mhd_index: dict[str, Path],
        patch_size: int,
        hu_min: float,
        hu_max: float,
        normalization_mode: str,
        lobe_mask_root: str | Path | None = None,
        lung_mask_root: str | Path | None = None,
        regional_regions: int = 5,
        cache_volumes: int = 2,
        augment: bool = False,
        center_jitter_mm: float = 0.0,
        seed: int = 0,
    ):
        self.samples = samples
        self.mhd_index = mhd_index
        self.patch_size = int(patch_size)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.normalization_mode = normalization_mode
        self.lobe_mask_root = Path(lobe_mask_root).expanduser().resolve() if lobe_mask_root else None
        self.lung_mask_root = Path(lung_mask_root).expanduser().resolve() if lung_mask_root else None
        self.regional_regions = max(1, int(regional_regions))
        self.cache = VolumeCache(cache_volumes)
        self.augment = augment
        self.center_jitter_mm = float(center_jitter_mm)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def load_lobe_mask(self, seriesuid: str, shape: tuple[int, int, int]) -> np.ndarray | None:
        if self.lobe_mask_root is None:
            return None
        for suffix in [".npy", ".npz", ".mhd"]:
            path = self.lobe_mask_root / f"{seriesuid}{suffix}"
            if not path.exists():
                continue
            if suffix == ".npy":
                mask = np.load(path)
            elif suffix == ".npz":
                data = np.load(path)
                key = data.files[0]
                mask = data[key]
            else:
                mask = load_mhd_volume(parse_mhd(path))
            if tuple(mask.shape) != tuple(shape):
                raise ValueError(f"Lobe mask shape mismatch for {seriesuid}: mask={mask.shape}, volume={shape}")
            return mask.astype(np.int16, copy=False)
        return None

    def load_lung_mask(self, seriesuid: str, shape: tuple[int, int, int]) -> np.ndarray | None:
        if self.lung_mask_root is None:
            return None
        for suffix in [".npy", ".npz", ".mhd"]:
            path = self.lung_mask_root / f"{seriesuid}{suffix}"
            if not path.exists():
                continue
            if suffix == ".npy":
                mask = np.load(path)
            elif suffix == ".npz":
                data = np.load(path)
                key = data.files[0]
                mask = data[key]
            else:
                mask = load_mhd_volume(parse_mhd(path))
            if tuple(mask.shape) != tuple(shape):
                raise ValueError(f"Lung mask shape mismatch for {seriesuid}: mask={mask.shape}, volume={shape}")
            return mask.astype(np.int16, copy=False)
        return None

    def store_key(self, seriesuid: str) -> tuple[str, int, str]:
        return (self.normalization_mode, int(self.regional_regions), seriesuid)

    def load_scan(self, seriesuid: str, bypass_store: bool = False) -> tuple[np.ndarray, MhdInfo]:
        if not bypass_store:
            hit = _VOLUME_STORE.get(self.store_key(seriesuid))
            if hit is not None:
                return hit
        cached = self.cache.get(seriesuid)
        if cached is not None:
            return cached
        info = parse_mhd(self.mhd_index[seriesuid])
        volume = load_mhd_volume(info)
        lobe_mask = self.load_lobe_mask(seriesuid, volume.shape)
        lung_mask = self.load_lung_mask(seriesuid, volume.shape)
        volume = lung_window_standardize(
            volume,
            self.hu_min,
            self.hu_max,
            self.normalization_mode,
            lobe_mask=lobe_mask,
            lung_mask=lung_mask,
            regional_regions=self.regional_regions,
        )
        self.cache.put(seriesuid, (volume, info))
        return volume, info

    def __getitem__(self, index: int):
        sample = self.samples[index]
        volume, info = self.load_scan(sample.seriesuid)
        center_zyx = world_to_voxel_zyx(sample.world_xyz, info)
        rng = np.random.default_rng(self.seed + index * 9973)
        if self.augment and self.center_jitter_mm > 0:
            jitter_mm = rng.normal(0.0, self.center_jitter_mm / 2.5, size=3)
            jitter_zyx = jitter_mm[::-1] / np.maximum(info.spacing_zyx, 1e-6)
            center_zyx = center_zyx + jitter_zyx
        patch = crop_patch(volume, center_zyx, self.patch_size)
        if self.augment:
            patch = augment_patch(patch, rng)
        x = patch[None, ...].astype(np.float32)
        y = np.array([sample.label], dtype=np.float32)
        meta = {
            "seriesuid": sample.seriesuid,
            "coordX": sample.coord_x,
            "coordY": sample.coord_y,
            "coordZ": sample.coord_z,
            "label": sample.label,
            "diameter_mm": sample.diameter_mm,
            "source": sample.source,
        }
        return x, y, meta


def collate_batch(batch):
    xs, ys, metas = zip(*batch)
    return np.stack(xs, axis=0), np.stack(ys, axis=0), list(metas)


def annotation_samples(ann: pd.DataFrame, series: list[str]) -> list[Sample]:
    series_set = set(series)
    rows = ann[ann["seriesuid"].isin(series_set)]
    return [
        Sample(
            seriesuid=str(r.seriesuid),
            coord_x=float(r.coordX),
            coord_y=float(r.coordY),
            coord_z=float(r.coordZ),
            label=1,
            diameter_mm=float(r.diameter_mm),
            source="annotation",
        )
        for r in rows.itertuples(index=False)
    ]


def sample_negative_candidates(
    cand: pd.DataFrame,
    series: list[str],
    target_count: int | None,
    seed: int,
    per_scan_limit: int | None = None,
) -> list[Sample]:
    series_set = set(series)
    rows = cand[(cand["seriesuid"].isin(series_set)) & (cand["class"] == 0)]
    if rows.empty:
        return []
    rng = np.random.default_rng(seed)
    groups = []
    if per_scan_limit is not None and per_scan_limit > 0:
        for _, group in rows.groupby("seriesuid"):
            take = min(len(group), per_scan_limit)
            groups.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        rows = pd.concat(groups, ignore_index=True) if groups else rows.iloc[0:0]
    if target_count is not None and target_count > 0 and len(rows) > target_count:
        rows = rows.sample(n=target_count, random_state=seed)
    return [
        Sample(
            seriesuid=str(r.seriesuid),
            coord_x=float(r.coordX),
            coord_y=float(r.coordY),
            coord_z=float(r.coordZ),
            label=0,
            diameter_mm=0.0,
            source="candidate_negative",
        )
        for r in rows.itertuples(index=False)
    ]


def positive_candidate_samples(cand: pd.DataFrame, ann: pd.DataFrame, series: list[str]) -> list[Sample]:
    series_set = set(series)
    rows = cand[(cand["seriesuid"].isin(series_set)) & (cand["class"] == 1)]
    if rows.empty:
        return annotation_samples(ann, series)
    ann_by_series = {sid: g for sid, g in ann[ann["seriesuid"].isin(series_set)].groupby("seriesuid")}
    samples: list[Sample] = []
    for r in rows.itertuples(index=False):
        diameter = 0.0
        group = ann_by_series.get(str(r.seriesuid))
        if group is not None and not group.empty:
            coords = group[["coordX", "coordY", "coordZ"]].to_numpy(dtype=float)
            p = np.array([float(r.coordX), float(r.coordY), float(r.coordZ)], dtype=float)
            idx = int(np.argmin(np.linalg.norm(coords - p[None, :], axis=1)))
            diameter = float(group.iloc[idx]["diameter_mm"])
        samples.append(
            Sample(
                seriesuid=str(r.seriesuid),
                coord_x=float(r.coordX),
                coord_y=float(r.coordY),
                coord_z=float(r.coordZ),
                label=1,
                diameter_mm=diameter,
                source="candidate_positive",
            )
        )
    return samples


def build_train_samples(
    ann: pd.DataFrame,
    cand: pd.DataFrame,
    series: list[str],
    negatives_per_positive: int,
    seed: int,
) -> list[Sample]:
    positives = annotation_samples(ann, series)
    neg_target = max(1, len(positives) * int(negatives_per_positive))
    negatives = sample_negative_candidates(cand, series, neg_target, seed)
    samples = positives + negatives
    random.Random(seed).shuffle(samples)
    return samples


def build_eval_samples(
    ann: pd.DataFrame,
    cand: pd.DataFrame,
    series: list[str],
    negatives_per_scan: int | None,
    include_positive_candidates: bool,
    seed: int,
    sort_by_scan: bool = False,
) -> list[Sample]:
    positives = positive_candidate_samples(cand, ann, series) if include_positive_candidates else annotation_samples(ann, series)
    # None or a non-positive value means the complete negative candidate pool.
    if negatives_per_scan is None or int(negatives_per_scan) <= 0:
        neg_target = None
        per_scan_limit = None
    else:
        neg_target = int(negatives_per_scan) * len(series)
        per_scan_limit = int(negatives_per_scan)
    negatives = sample_negative_candidates(cand, series, neg_target, seed, per_scan_limit=per_scan_limit)
    samples = positives + negatives
    if sort_by_scan:
        # Group candidates by scan so that one volume load serves all candidates of that scan
        # (the volume LRU cache can then be small). FROC/CPM thresholds are derived from the
        # score distribution and matching is order-independent, so this only changes speed,
        # not the candidate set or the resulting metrics.
        samples.sort(key=lambda s: s.seriesuid)
    else:
        random.Random(seed).shuffle(samples)
    return samples


def dataset_overview(luna_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    mhd_index = build_mhd_index(luna_root, cfg["data"].get("scan_root"))
    ann = load_annotations(luna_root, mhd_index)
    cand = load_candidates(luna_root, mhd_index)
    series = sorted(mhd_index.keys())
    folds = make_folds(
        series,
        mhd_index,
        int(cfg["data"]["num_folds"]),
        str(cfg["data"].get("fold_strategy", "official10")),
        int(cfg["seed"]),
    )
    fold_rows = []
    for i in range(len(folds)):
        train, val, test = split_for_fold(
            folds,
            i,
            float(cfg["data"].get("val_fraction_from_train_pool", 0.125)),
            int(cfg["seed"]),
        )
        fold_rows.append(
            {
                "fold": i,
                "train_scans": len(train),
                "val_scans": len(val),
                "test_scans": len(test),
                "train_nodules": int(ann["seriesuid"].isin(train).sum()),
                "val_nodules": int(ann["seriesuid"].isin(val).sum()),
                "test_nodules": int(ann["seriesuid"].isin(test).sum()),
            }
        )
    return {
        "luna_root": str(luna_root),
        "mhd_files": len(mhd_index),
        "annotation_rows": len(ann),
        "candidate_rows": len(cand),
        "series_indexed": len(series),
        "series_with_annotations": int(ann["seriesuid"].nunique()),
        "series_with_candidates": int(cand["seriesuid"].nunique()),
        "folds": fold_rows,
    }


def build_fold_datasets(cfg: dict[str, Any], fold_index: int, variant: str, max_train_samples: int = 0, max_eval_scans: int = 0):
    luna_root = Path(cfg["data"]["luna_root"]).expanduser().resolve()
    mhd_index = build_mhd_index(luna_root, cfg["data"].get("scan_root"))
    ann = load_annotations(luna_root, mhd_index)
    cand = load_candidates(luna_root, mhd_index)
    series = sorted(mhd_index.keys())
    folds = make_folds(
        series,
        mhd_index,
        int(cfg["data"]["num_folds"]),
        str(cfg["data"].get("fold_strategy", "official10")),
        int(cfg["seed"]),
    )
    train_series, val_series, test_series = split_for_fold(
        folds,
        fold_index,
        float(cfg["data"].get("val_fraction_from_train_pool", 0.125)),
        int(cfg["seed"]),
    )
    if max_eval_scans and max_eval_scans > 0:
        val_series = val_series[:max_eval_scans]
        test_series = test_series[:max_eval_scans]

    requested_normalization_mode = str(cfg["data"].get("normalization_mode", "regional"))
    normalization_mode = requested_normalization_mode
    if variant == "global_norm":
        normalization_mode = "global"
    elif normalization_mode == "lobe" and not cfg["data"].get("lobe_mask_root"):
        normalization_mode = "regional"

    train_samples = build_train_samples(
        ann,
        cand,
        train_series,
        int(cfg["data"]["train_negatives_per_positive"]),
        int(cfg["seed"]) + fold_index,
    )
    if max_train_samples and max_train_samples > 0:
        train_samples = train_samples[:max_train_samples]
    val_negatives = cfg["data"].get("val_negatives_per_scan", cfg["data"].get("eval_negatives_per_scan"))
    val_samples = build_eval_samples(
        ann,
        cand,
        val_series,
        val_negatives,
        bool(cfg["data"].get("eval_include_all_positive_candidates", True)),
        int(cfg["seed"]) + 1000 + fold_index + int(cfg["data"].get("eval_seed_offset", 0)),
        sort_by_scan=bool(cfg["data"].get("eval_sort_by_scan", False)),
    )
    test_samples = build_eval_samples(
        ann,
        cand,
        test_series,
        cfg["data"].get("eval_negatives_per_scan"),
        bool(cfg["data"].get("eval_include_all_positive_candidates", True)),
        int(cfg["seed"]) + 2000 + fold_index + int(cfg["data"].get("eval_seed_offset", 0)),
        sort_by_scan=bool(cfg["data"].get("eval_sort_by_scan", False)),
    )
    common = dict(
        mhd_index=mhd_index,
        patch_size=int(cfg["data"]["patch_size"]),
        hu_min=float(cfg["data"]["hu_min"]),
        hu_max=float(cfg["data"]["hu_max"]),
        normalization_mode=normalization_mode,
        lobe_mask_root=cfg["data"].get("lobe_mask_root"),
        lung_mask_root=cfg["data"].get("lung_mask_root"),
        regional_regions=int(cfg["data"].get("regional_regions", 5)),
        cache_volumes=int(cfg["data"].get("cache_volumes", 2)),
    )
    train_ds = LunaCandidateDataset(
        train_samples,
        augment=True,
        center_jitter_mm=float(cfg["data"].get("center_jitter_mm", 0.0)),
        seed=int(cfg["seed"]) + fold_index * 17,
        **common,
    )
    val_ds = LunaCandidateDataset(val_samples, augment=False, seed=int(cfg["seed"]) + fold_index * 19, **common)
    test_ds = LunaCandidateDataset(test_samples, augment=False, seed=int(cfg["seed"]) + fold_index * 23, **common)
    val_ds.scan_count = len(val_series)
    test_ds.scan_count = len(test_series)

    # Resample training negatives every epoch.
    #
    # Motivation (measured 2026-09-16): the training set holds only 911 positives + 3644
    # negatives = 4555 **fixed** samples, while the model has ~5.6M parameters -- after 80
    # epochs of looking at the same crops, train_loss drops to 0.003 by roughly epoch 30
    # (perfect memorization), yet validation CPM stays flat at 0.005 throughout, showing no
    # generalization at all. With a fresh batch of random negatives each epoch, the
    # cumulative number of negatives seen over 80 epochs grows from 3644 to about 290,000,
    # and **per-epoch compute does not increase** (the sample count is unchanged; only the
    # identities differ).
    if bool(cfg["data"].get("resample_train_each_epoch", False)):
        def _resample_train_samples(epoch_idx: int):
            return build_train_samples(
                ann,
                cand,
                train_series,
                int(cfg["data"]["train_negatives_per_positive"]),
                int(cfg["seed"]) + fold_index + 7919 * (int(epoch_idx) + 1),
            )

        train_ds.resample_fn = _resample_train_samples

    split_info = {
        "fold_index": fold_index,
        "num_folds": len(folds),
        "train_scans": train_series,
        "val_scans": val_series,
        "test_scans": test_series,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "normalization_mode_requested": requested_normalization_mode,
        "normalization_mode": normalization_mode,
        "regional_regions": int(cfg["data"].get("regional_regions", 5)),
        "eval_negatives_per_scan": cfg["data"].get("eval_negatives_per_scan"),
        "val_negatives_per_scan": val_negatives,
        "eval_seed_offset": int(cfg["data"].get("eval_seed_offset", 0)),
        "eval_sort_by_scan": bool(cfg["data"].get("eval_sort_by_scan", False)),
        "cache_volumes": int(cfg["data"].get("cache_volumes", 2)),
        "normalization_fallback": "binary lung-mask constrained axial regions" if normalization_mode == "regional" else "",
    }
    gt_ann = {
        "val": ann[ann["seriesuid"].isin(val_series)].copy(),
        "test": ann[ann["seriesuid"].isin(test_series)].copy(),
    }
    return train_ds, val_ds, test_ds, gt_ann, split_info
