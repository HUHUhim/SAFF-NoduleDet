#!/usr/bin/env python
"""SAFF-NoduleDet vs CoTr3D: the ten-fold paired tests required by Section 4.3 of the paper.

Differences from analyze_revision.py:
  * it compares only the single CoTr3D condition, and only on the three metrics declared in the paper (CPM / Sens@0.125 / Sens@1.0);
  * the Holm correction is strictly restricted to these 3 tests (the *_holm_p columns of analyze_revision.py correct over
    all condition x metric tests in the whole table, so copying them directly would give far more conservative values that
    do not match the paper text);
  * it additionally prints a per-fold table that can be used directly as supplementary Table S2.

Usage (local or server, only final_metrics.json is needed, no GPU/CT data required):
  python analysis/paired_tests_cotr3d.py \
    --saff-root   /root/autodl-tmp/SAFF_ws/SAFF/outputs_p0            --saff-experiment   p0_main \
    --cotr3d-root /root/autodl-tmp/SAFF_ws/SAFF_revision/outputs_revision --cotr3d-experiment table4_baselines \
    --out-dir     /root/autodl-tmp/SAFF_ws/SAFF_revision/analysis_cotr3d
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

METRICS = [("cpm", "CPM"), ("sens_at_0.125", "Sens@0.125"), ("sens_at_1", "Sens@1.0")]


def read_folds(root: Path, experiment: str, model: str, variant: str = "full") -> dict[int, dict]:
    base = root / "runs" / experiment / model / variant
    out: dict[int, dict] = {}
    for path in sorted(base.glob("fold_*/final_metrics.json")):
        fold = int(path.parent.name.replace("fold_", ""))
        obj = json.loads(path.read_text(encoding="utf-8"))
        out[fold] = obj.get("test_metrics", obj)
    return out


def holm_over_three(pvals: list[float]) -> list[float]:
    """Holm correction with the number of tests fixed at len(pvals) (3 in this script)"""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [1.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        v = min(1.0, pvals[i] * (m - rank))
        running = max(running, v)
        adj[i] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saff-root", required=True)
    ap.add_argument("--saff-experiment", default="p0_main")
    ap.add_argument("--cotr3d-root", required=True)
    ap.add_argument("--cotr3d-experiment", default="table4_baselines")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    saff_root, cotr_root = Path(args.saff_root).expanduser(), Path(args.cotr3d_root).expanduser()
    saff = read_folds(saff_root, args.saff_experiment, "saff_noduledet", args.variant)
    cotr = read_folds(cotr_root, args.cotr3d_experiment, "cotr3d", args.variant)
    if not saff:
        raise SystemExit(f"SAFF results not found: {saff_root}/runs/{args.saff_experiment}/saff_noduledet/{args.variant}/fold_*/final_metrics.json")
    if not cotr:
        raise SystemExit(f"CoTr3D results not found: {cotr_root}/runs/{args.cotr3d_experiment}/cotr3d/{args.variant}/fold_*/final_metrics.json")

    common = sorted(set(saff) & set(cotr))
    print(f"SAFF folds={len(saff)}  CoTr3D folds={len(cotr)}  common folds={len(common)} -> {common}")
    if len(common) != 10:
        print("[warn] the number of common folds is not 10; p values may only be reported as paper conclusions when >=10 folds")

    rows, per_fold, pvals = [], {}, []
    for key, label in METRICS:
        x = np.array([float(saff[f].get(key, np.nan)) for f in common])
        y = np.array([float(cotr[f].get(key, np.nan)) for f in common])
        d = x - y
        if len(d) < 2 or np.all(np.isnan(d)):
            print(f"[skip] {label}: insufficient data")
            continue
        t_p = float(ttest_rel(x, y, nan_policy="omit").pvalue)
        try:
            w_p = float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            w_p = math.nan
        dz = float(np.mean(d) / np.std(d, ddof=1))
        se = float(np.std(d, ddof=1) / math.sqrt(len(d)))
        tcrit = float(pd.Series([2.262157]).iloc[0]) if len(d) == 10 else 1.96
        rows.append({
            "metric": key, "label": label, "n_folds": len(d),
            "saff_mean": float(np.mean(x)), "cotr3d_mean": float(np.mean(y)),
            "mean_diff": float(np.mean(d)), "std_diff": float(np.std(d, ddof=1)),
            "ci95_low": float(np.mean(d) - tcrit * se), "ci95_high": float(np.mean(d) + tcrit * se),
            "wilcoxon_p": w_p, "paired_t_p": t_p, "cohen_dz": dz,
        })
        per_fold[label] = (x, y, d)
        pvals.append(w_p if np.isfinite(w_p) else 1.0)

    if not rows:
        raise SystemExit("There is no metric that can be computed")

    holm = holm_over_three(pvals)
    for r, h in zip(rows, holm):
        r["wilcoxon_holm_p_3tests"] = h

    df = pd.DataFrame(rows)
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "paired_tests_cotr3d.csv", index=False)

    print("\n=== SAFF-NoduleDet vs CoTr3D (Holm correction restricted to these 3 tests) ===")
    for r in rows:
        print(f"  {r['label']:<12} mean diff {r['mean_diff']:+.4f}  95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
              f"  Wilcoxon p={r['wilcoxon_p']:.4f}  Holm p={r['wilcoxon_holm_p_3tests']:.4f}"
              f"  t p={r['paired_t_p']:.4f}  d_z={r['cohen_dz']:.2f}")

    pf = []
    for label, (x, y, d) in per_fold.items():
        pf.append({"Metric": label, **{f"fold_{f}": round(float(v), 3) for f, v in zip(common, x)},
                   "Mean": round(float(np.mean(x)), 3), "SD": round(float(np.std(x, ddof=1)), 3)})
        pf.append({"Metric": label + " (CoTr3D)", **{f"fold_{f}": round(float(v), 3) for f, v in zip(common, y)},
                   "Mean": round(float(np.mean(y)), 3), "SD": round(float(np.std(y, ddof=1)), 3)})
        pf.append({"Metric": label + " (difference)", **{f"fold_{f}": round(float(v), 3) for f, v in zip(common, d)},
                   "Mean": round(float(np.mean(d)), 4), "SD": round(float(np.std(d, ddof=1)), 4)})
    pd.DataFrame(pf).to_csv(out / "per_fold_table_cotr3d.csv", index=False, encoding="utf-8-sig")

    print(f"\nWritten: {out / 'paired_tests_cotr3d.csv'}")
    print(f"Written: {out / 'per_fold_table_cotr3d.csv'} (can serve as supplementary Table S2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
