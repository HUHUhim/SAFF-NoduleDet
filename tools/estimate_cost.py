#!/usr/bin/env python
"""Rerun cost estimation: first count the samples of each condition, then extrapolate hours and cost from the per-sample time.

Usage (server or local, only LUNA16's .mhd/annotations.csv/candidates.csv are needed):
  python tools/estimate_cost.py --config configs/rerun_4090_final.yaml --price 2.5
  python tools/estimate_cost.py --config configs/rerun_4090_final.yaml --tier 1          # first priority tier only
  python tools/estimate_cost.py --config configs/rerun_4090_final.yaml --train-sec 0.02 --eval-sec 0.008

First calibrate --train-sec / --eval-sec with a 1-fold 2-epoch smoke run, then estimate the total price of the full run from them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from saff_p0.config import load_config  # noqa: E402
from saff_p0.data import (  # noqa: E402
    build_mhd_index,
    build_train_samples,
    build_eval_samples,
    load_annotations,
    load_candidates,
    make_folds,
    split_for_fold,
)

# Priority plan: (tier, experiment name, model, variant, description)
PLAN: list[tuple[int, str, str, str, str]] = [
    (1, "p0_main", "saff_noduledet", "full", "Table 4 SAFF row + reference for the 4.3 paired tests"),
    (1, "table4_baselines", "cotr3d", "full", "Table 4 CoTr3D row + comparison arm for the 4.3 paired tests"),
    (2, "table4_baselines", "unet3d", "full", "Table 4 3D U-Net row"),
    (2, "table4_baselines", "convnext3d", "full", "Table 4 3D ConvNeXt row"),
    (2, "table4_baselines", "swin_unetr", "full", "Table 4 Swin UNETR row"),
    (3, "table6_ablation", "saff_noduledet", "no_distill", "Table 6 w/o cross-branch distillation"),
    (3, "table6_ablation", "saff_noduledet", "no_morph_gate", "Table 6 w/o morphology-aware gating"),
    (3, "table6_ablation", "saff_noduledet", "fixed_fusion", "Table 6 fixed fusion"),
    (3, "table6_ablation", "saff_noduledet", "global_norm", "Table 6 global normalization"),
    (3, "table6_ablation", "saff_noduledet", "cnn_only", "Table 6 CNN branch only"),
    (3, "table6_ablation", "saff_noduledet", "transformer_only", "Table 6 Transformer branch only"),
    (3, "table6_ablation", "saff_noduledet", "full_attention", "Table 6 full attention"),
    (4, "table7_sensitivity", "saff_noduledet", "full", "Table 7 r=3"),
    (4, "table7_sensitivity", "saff_noduledet", "full", "Table 7 r=7"),
    (4, "table7_sensitivity", "saff_noduledet", "full", "Table 7 λ_dist=0.05"),
    (4, "table7_sensitivity", "saff_noduledet", "full", "Table 7 λ_dist=0.20"),
    (4, "table7_sensitivity", "saff_noduledet", "full", "Table 7 λ_smooth=0"),
    (4, "table7_sensitivity", "saff_noduledet", "full", "Table 7 λ_smooth=0.010"),
]


def count_split(cfg: dict, fold: int) -> dict:
    luna_root = Path(cfg["data"]["luna_root"]).expanduser()
    mhd_index = build_mhd_index(luna_root, cfg["data"].get("scan_root"))
    ann = load_annotations(luna_root, mhd_index)
    cand = load_candidates(luna_root, mhd_index)
    series = sorted(mhd_index.keys())
    folds = make_folds(series, mhd_index, int(cfg["data"]["num_folds"]), str(cfg["data"].get("fold_strategy", "official10")), int(cfg["seed"]))
    train, val, test = split_for_fold(folds, fold, float(cfg["data"].get("val_fraction_from_train_pool", 0.125)), int(cfg["seed"]))
    tr = build_train_samples(ann, cand, train, int(cfg["data"]["train_negatives_per_positive"]), int(cfg["seed"]) + fold)
    va = build_eval_samples(ann, cand, val, cfg["data"].get("eval_negatives_per_scan"),
                            bool(cfg["data"].get("eval_include_all_positive_candidates", True)), 0)
    te = build_eval_samples(ann, cand, test, cfg["data"].get("eval_negatives_per_scan"),
                            bool(cfg["data"].get("eval_include_all_positive_candidates", True)), 0)
    return {"train": tr, "val": va, "test": te, "val_scans": len(val), "test_scans": len(test)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tier", type=int, default=0, help="only count conditions at or below this tier (0 = all)")
    ap.add_argument("--price", type=float, default=2.5, help="GPU rental price (CNY per hour)")
    ap.add_argument("--train-sec", type=float, default=0.020, help="seconds per training sample (forward + backward, batch=1)")
    ap.add_argument("--eval-sec", type=float, default=0.008, help="seconds per evaluated candidate (volumes cached per scan after sorting)")
    ap.add_argument("--volume-sec", type=float, default=0.6, help="seconds to load + standardize each scan for the first time")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    epochs = int(cfg["training"]["epochs"])
    eval_every = max(1, int(cfg["training"].get("eval_every", 1)))
    n_evals = 1 + len([e for e in range(2, epochs + 1) if e % eval_every == 0 or e == epochs])

    print("Counting samples per fold (reads only csvs and file indices, does not load images)…")
    per_fold = {}
    for fold in range(int(cfg["data"]["num_folds"])):
        s = count_split(cfg, fold)
        per_fold[fold] = s
    avg_tr = sum(len(v["train"]) for v in per_fold.values()) / len(per_fold)
    avg_va = sum(len(v["val"]) for v in per_fold.values()) / len(per_fold)
    avg_te = sum(len(v["test"]) for v in per_fold.values()) / len(per_fold)
    avg_vs = sum(v["val_scans"] for v in per_fold.values()) / len(per_fold)
    avg_ts = sum(v["test_scans"] for v in per_fold.values()) / len(per_fold)

    print(f"\nPer-fold averages: train {avg_tr:.0f} samples | val {avg_va:.0f} candidates / {avg_vs:.0f} scans | test {avg_te:.0f} candidates / {avg_ts:.0f} scans")
    print(f"Training {epochs} epochs, validation every {eval_every} epoch(s) → {n_evals} validation rounds + 1 test\n")

    rows = []
    cum_hours = 0.0
    for tier, exp, model, variant, desc in PLAN:
        if args.tier and tier > args.tier:
            continue
        train_h = avg_tr * epochs * args.train_sec / 3600
        val_h = (n_evals * (avg_va * args.eval_sec + avg_vs * args.volume_sec)) / 3600
        test_h = (avg_te * args.eval_sec + avg_ts * args.volume_sec) / 3600
        fold_h = train_h + val_h + test_h
        cond_h = fold_h * int(cfg["data"]["num_folds"])
        cum_hours += cond_h
        rows.append({"tier": tier, "experiment": exp, "model": model, "variant": variant, "desc": desc,
                     "hours_per_fold": round(fold_h, 2), "hours": round(cond_h, 1), "cost": round(cond_h * args.price, 1)})

    print(f"{'Tier':<3}{'Experiment':<18}{'Model/Variant':<32}{'Hours/fold':>8}{'Hours':>8}{'Cost (CNY)':>10}  Description")
    for r in rows:
        print(f"{r['tier']:<3}{r['experiment']:<18}{(r['model'] + '/' + r['variant']):<32}"
              f"{r['hours_per_fold']:>8.2f}{r['hours']:>8.1f}{r['cost']:>10.1f}  {r['desc']}")

    tier_hours = {}
    for r in rows:
        tier_hours[r["tier"]] = tier_hours.get(r["tier"], 0.0) + r["hours"]
    print("\nCumulative by tier:")
    running = 0.0
    for tier in sorted(tier_hours):
        running += tier_hours[tier]
        print(f"  Completing tiers 1~{tier}: {running:6.1f} GPU hours ≈ ¥{running * args.price:,.0f}")

    out = {"config": args.config, "epochs": epochs, "eval_every": eval_every, "n_eval_rounds": n_evals,
           "price_per_hour": args.price, "train_sec_per_sample": args.train_sec, "eval_sec_per_candidate": args.eval_sec,
           "avg_train_samples_per_fold": avg_tr, "avg_val_candidates": avg_va, "avg_test_candidates": avg_te,
           "rows": rows, "tier_hours": tier_hours, "total_hours": round(cum_hours, 1),
           "total_cost": round(cum_hours * args.price, 1)}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWritten {args.json_out}")
    print("\nHint: first measure the per-sample seconds with a 1-fold 2-epoch smoke run, then re-run this script to calibrate --train-sec/--eval-sec.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
