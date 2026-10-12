#!/usr/bin/env python
"""Generate a **synthetic mini LUNA16 directory** for an end-to-end CPU smoke test locally or on a server.

It only creates the file structure the pipeline needs (.mhd/.raw, lung masks as .npy, annotations.csv, candidates.csv).
Its purpose is to verify that the whole chain "data pipeline + training + evaluation + packaging" runs through,
and it **must never be used for paper results**.

Usage:
  python tools/make_fake_luna.py --out /tmp/fake_luna --scans 8 --size 64 --num-folds 2
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def write_mhd(path: Path, raw_name: str, dim_xyz: tuple[int, int, int], spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)) -> None:
    path.write_text(
        "ObjectType = Image\n"
        "NDims = 3\n"
        f"DimSize = {dim_xyz[0]} {dim_xyz[1]} {dim_xyz[2]}\n"
        "ElementType = MET_SHORT\n"
        f"ElementSpacing = {spacing[0]} {spacing[1]} {spacing[2]}\n"
        f"Offset = {origin[0]} {origin[1]} {origin[2]}\n"
        "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        f"ElementDataFile = {raw_name}\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scans", type=int, default=8)
    ap.add_argument("--size", type=int, default=64, help="edge length of the volume (a cube, in voxels)")
    ap.add_argument("--num-folds", type=int, default=2)
    ap.add_argument("--nodules-per-scan", type=int, default=2)
    ap.add_argument("--candidates-per-scan", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    scan_root = out / "subset_data"
    mask_root = out / "seg-lungs-LUNA16"
    rng = np.random.default_rng(args.seed)
    S = int(args.size)

    ann_rows = ["seriesuid,coordX,coordY,coordZ,diameter_mm"]
    cand_rows = ["seriesuid,coordX,coordY,coordZ,class"]

    for i in range(args.scans):
        subset = i % max(1, args.num_folds)
        sdir = scan_root / f"subset{subset}"
        sdir.mkdir(parents=True, exist_ok=True)
        uid = f"1.3.6.1.4.1.14519.5.2.1.6279.6001.{100000000000000000000000000000 + i:030d}"

        # Volume: background + lung-like ellipsoid + a few spherical "nodules"
        z, y, x = np.meshgrid(np.arange(S), np.arange(S), np.arange(S), indexing="ij")
        cz, cy, cx = S / 2.0, S / 2.0, S / 2.0
        lung = (((z - cz) / (S * 0.38)) ** 2 + ((y - cy) / (S * 0.32)) ** 2 + ((x - cx) / (S * 0.32)) ** 2) <= 1.0
        vol = np.full((S, S, S), -1000, dtype=np.int16)
        vol[lung] = -800 + (rng.normal(0, 40, size=int(lung.sum()))).astype(np.int16)
        mask = lung.astype(np.uint8) * 3
        mask[lung] = 3

        for _ in range(args.nodules_per_scan):
            for _try in range(50):
                nz, ny, nx = rng.integers(S // 5, S - S // 5, size=3)
                if lung[nz, ny, nx]:
                    break
            else:
                continue
            r = float(rng.uniform(2.0, 5.0))
            blob = (((z - nz) ** 2 + (y - ny) ** 2 + (x - nx) ** 2) <= r * r)
            vol[blob] = rng.integers(20, 120)
            diameter = 2.0 * r
            ann_rows.append(f"{uid},{nx},{ny},{nz},{diameter:.4f}")
            # Positive candidate: jittered around the nodule center
            cand_rows.append(f"{uid},{nx + rng.integers(-1, 2)},{ny + rng.integers(-1, 2)},{nz + rng.integers(-1, 2)},1")

        for _ in range(args.candidates_per_scan):
            while True:
                nz, ny, nx = rng.integers(1, S - 1, size=3)
                if not lung[nz, ny, nx]:
                    break
            cand_rows.append(f"{uid},{nx},{ny},{nz},0")

        (sdir / f"{uid}.raw").write_bytes(np.ascontiguousarray(vol).tobytes())
        write_mhd(sdir / f"{uid}.mhd", f"{uid}.raw", (S, S, S))
        mask_root.mkdir(parents=True, exist_ok=True)
        np.save(mask_root / f"{uid}.npy", mask)

    (out / "annotations.csv").write_text("\n".join(ann_rows) + "\n", encoding="utf-8")
    (out / "candidates.csv").write_text("\n".join(cand_rows) + "\n", encoding="utf-8")

    # Also write a ready-to-use smoke config (small model / small patch / 2 folds / 2 epochs)
    cfg_path = out / "smoke_config.yaml"
    cfg_path.write_text(
        f"""# Auto-generated smoke config (synthetic data, pipeline verification only, must not be used for the paper)
seed: 20260612

data:
  luna_root: {out}
  scan_root: {scan_root}
  output_root: {out / 'outputs_smoke'}
  patch_size: 32
  hu_min: -1200
  hu_max: 600
  normalization_mode: regional
  regional_regions: 5
  lobe_mask_root:
  lung_mask_root: {mask_root}
  train_negatives_per_positive: 4
  eval_negatives_per_scan: 20
  eval_include_all_positive_candidates: true
  eval_sort_by_scan: true
  cache_volumes: 4
  num_folds: {args.num_folds}
  fold_strategy: random
  val_fraction_from_train_pool: 0.25
  center_jitter_mm: 1.0

training:
  epochs: 2
  batch_size: 1
  grad_accum_steps: 2
  lr: 0.0002
  weight_decay: 0.0001
  num_workers: 0
  cpu_threads: 4
  amp: false
  early_stop_patience: 18
  eval_every: 1

model:
  base_channels: 8
  local_channels: 32
  global_channels: 32
  fusion_channels: 32
  global_token_grid: 3
  full_attention_token_grid: 4
  transformer_depth: 2
  transformer_heads: 4
  swin_adapted_heads: 4
  dropout: 0.10
  distill_weight: 0.10
  gate_smooth_weight: 0.005

eval:
  fp_targets: [0.125, 0.25, 0.5, 1, 2, 4, 8]
  hit_radius_scale: 0.5
  min_hit_radius_mm: 3.0
  plot: false
""",
        encoding="utf-8",
    )

    print(f"Synthetic data generated: {out}")
    print(f"  scans={args.scans}  volume={S}^3  folds={args.num_folds}  annotations={len(ann_rows) - 1}  candidates={len(cand_rows) - 1}")
    print(f"  smoke config: {cfg_path}")
    print("  Note: this is synthetic data, usable only for a smoke test, never as paper results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
