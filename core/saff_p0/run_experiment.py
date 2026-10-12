from __future__ import annotations

import argparse
import copy
import sys
from contextlib import redirect_stderr, redirect_stdout

from .config import (
    SUPPORTED_SAFF_VARIANTS,
    ensure_dir,
    load_config,
    output_root,
    save_json,
    save_yaml,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--variants", nargs="*", default=["full"])
    parser.add_argument("--variants-from-config", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-negatives-per-positive", type=int, default=None)
    parser.add_argument("--eval-negatives-per-scan", type=int, default=None)
    parser.add_argument("--regional-regions", type=int, default=None)
    parser.add_argument("--distill-weight", type=float, default=None)
    parser.add_argument("--smooth-weight", type=float, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-scans", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolved_cfg(base_cfg, args, model_name: str, variant: str, fold: int):
    cfg = copy.deepcopy(base_cfg)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.train_negatives_per_positive is not None:
        cfg["data"]["train_negatives_per_positive"] = int(args.train_negatives_per_positive)
    if args.eval_negatives_per_scan is not None:
        cfg["data"]["eval_negatives_per_scan"] = int(args.eval_negatives_per_scan)
    if args.regional_regions is not None:
        cfg["data"]["regional_regions"] = int(args.regional_regions)
    if args.distill_weight is not None:
        cfg["model"]["distill_weight"] = float(args.distill_weight)
    if args.smooth_weight is not None:
        cfg["model"]["gate_smooth_weight"] = float(args.smooth_weight)
    if args.output_root is not None:
        cfg["data"]["output_root"] = str(args.output_root)
    cfg["run"] = {
        "experiment": args.experiment,
        "model": model_name,
        "variant": variant,
        "fold": fold,
        "max_train_samples": int(args.max_train_samples),
        "max_eval_scans": int(args.max_eval_scans),
    }
    return cfg


def run_one(base_cfg, args, model_name: str, variant: str, fold: int) -> int:
    import torch

    from .data import build_fold_datasets
    from .models import build_model, count_parameters
    from .train_eval import train_one_fold

    cfg = resolved_cfg(base_cfg, args, model_name, variant, fold)
    variant_name = variant if model_name == "saff_noduledet" else "full"
    run_dir = output_root(cfg) / "runs" / args.experiment / model_name / variant_name / f"fold_{fold}"
    final_metrics = run_dir / "final_metrics.json"
    if final_metrics.exists() and not args.force:
        print(f"[skip] {model_name}/{variant_name}/fold_{fold}: {final_metrics} exists")
        return 0
    ensure_dir(run_dir)
    save_yaml(run_dir / "config_resolved.yaml", cfg)
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log, redirect_stdout(log), redirect_stderr(log):
        print(f"run_dir={run_dir}")
        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device={torch.cuda.get_device_name(0)}")
        set_seed(int(cfg["seed"]) + fold * 101, bool(cfg["training"].get("deterministic", False)))
        train_ds, val_ds, test_ds, gt_ann, split_info = build_fold_datasets(
            cfg,
            fold,
            variant_name,
            max_train_samples=int(args.max_train_samples),
            max_eval_scans=int(args.max_eval_scans),
        )
        save_json(run_dir / "split_info.json", split_info)
        model = build_model(model_name, cfg, variant_name)
        save_json(
            run_dir / "model_info.json",
            {
                "model": model_name,
                "variant": variant_name,
                "trainable_parameters": count_parameters(model),
            },
        )
        train_one_fold(model, train_ds, val_ds, test_ds, gt_ann, cfg, run_dir)
    print(f"[done] {model_name}/{variant_name}/fold_{fold}: log={log_path}")
    return 0


def resolve_folds(base_cfg: dict) -> list[int]:
    num_folds = int(base_cfg["data"]["num_folds"])
    if num_folds < 2:
        raise ValueError("data.num_folds must be >= 2")
    return list(range(num_folds))


def resolve_variants(base_cfg: dict, args, model_name: str) -> list[str]:
    if model_name != "saff_noduledet":
        return ["full"]
    if args.variants_from_config:
        configured = base_cfg.get("experiments", {}).get("ablations", [])
        variants = list(configured) if configured else ["full"]
    else:
        variants = args.variants
    unknown = sorted(set(variants) - set(SUPPORTED_SAFF_VARIANTS))
    if unknown:
        allowed = ", ".join(SUPPORTED_SAFF_VARIANTS)
        raise ValueError(f"Unsupported SAFF ablation variant(s): {unknown}. Allowed: {allowed}")
    return list(variants)


def main() -> int:
    args = parse_args()
    base_cfg = load_config(args.config)
    if args.folds is None:
        args.folds = resolve_folds(base_cfg)
    failures = 0
    for model_name in args.models:
        variants = resolve_variants(base_cfg, args, model_name)
        for variant in variants:
            for fold in args.folds:
                try:
                    failures += int(run_one(base_cfg, args, model_name, variant, fold) != 0)
                except Exception as exc:
                    failures += 1
                    print(f"[fail] {model_name}/{variant}/fold_{fold}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
