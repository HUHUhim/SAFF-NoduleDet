"""Re-evaluate finished ten-fold checkpoints under candidate-pool settings.

This is intentionally evaluation-only: it never changes a checkpoint and never
re-trains a model. ``eval_negatives_per_scan <= 0`` means the complete negative
candidate pool for every held-out scan.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .config import load_config, save_json, set_seed
from .data import build_fold_datasets
from .models import build_model
from .train_eval import evaluate_model, make_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Base ten-fold YAML config")
    p.add_argument("--checkpoint-root", required=True, help="Root containing runs/<experiment>/...")
    p.add_argument("--checkpoint-experiment", default="p0_main")
    p.add_argument("--model", default="saff_noduledet")
    p.add_argument("--variant", default="full")
    p.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--counts", nargs="+", type=int, default=[1200, 2400, 0])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-root", required=True)
    return p.parse_args()


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    obj = torch.load(path, map_location=device)
    state = obj.get("model", obj)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)


def main() -> int:
    args = parse_args()
    load_config(args.config)  # Validate the requested base configuration early.
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    for count in args.counts:
        count_name = "full_pool" if count <= 0 else f"neg_{count}_per_scan"
        for repeat in range(args.repeats):
            # Repeat zero with 1200 negatives reproduces the manuscript run's
            # deterministic sampling seed; later repeats use independent seeds.
            seed_offset = repeat * 1_000_003
            for fold in args.folds:
                ckpt_dir = (
                    checkpoint_root
                    / "runs"
                    / args.checkpoint_experiment
                    / args.model
                    / args.variant
                    / f"fold_{fold}"
                )
                ckpt_path = ckpt_dir / "best.pt"
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
                cfg_path = ckpt_dir / "config_resolved.yaml"
                cfg = load_config(cfg_path if cfg_path.exists() else args.config)
                cfg["data"]["eval_negatives_per_scan"] = None if count <= 0 else int(count)
                cfg["data"]["eval_seed_offset"] = int(seed_offset)
                set_seed(int(cfg["seed"]) + fold * 101 + seed_offset, bool(cfg["training"].get("deterministic", False)))
                _, _, test_ds, gt_ann, split_info = build_fold_datasets(cfg, fold, args.variant)
                model = build_model(args.model, cfg, args.variant).to(device)
                load_checkpoint(model, ckpt_path, device)
                loader = make_loader(test_ds, cfg, train=False)
                run_dir = output_root / count_name / f"repeat_{repeat:02d}" / f"fold_{fold}"
                run_dir.mkdir(parents=True, exist_ok=True)
                save_json(run_dir / "split_info.json", split_info)
                metrics = evaluate_model(
                    model,
                    loader,
                    device,
                    cfg,
                    gt_ann["test"],
                    "test",
                    run_dir,
                    bool(cfg["training"].get("amp", True) and device.type == "cuda"),
                )
                row = {
                    "count": int(count),
                    "count_name": count_name,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "checkpoint": str(ckpt_path),
                    "n_test_samples": len(test_ds),
                    "n_test_scans": int(getattr(test_ds, "scan_count", 0)),
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
                rows.append(row)
                save_json(run_dir / "evaluation_meta.json", row)
                print(f"[done] {count_name} repeat={repeat} fold={fold} cpm={metrics.get('cpm')}")

    summary_path = output_root / "candidate_sampling_metrics.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
