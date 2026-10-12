"""Summarize ten-fold supplementary runs and paired reviewer-requested tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


METRICS = ["cpm", "sens_at_0.125", "sens_at_1", "sens_at_8"]


def read_final_runs(root: Path, experiments: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment in experiments:
        base = root / "runs" / experiment
        for path in sorted(base.glob("*/*/fold_*/final_metrics.json")):
            parts = path.relative_to(base).parts
            if len(parts) != 4:
                continue
            model, variant, fold_name = parts[:3]
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            metrics = obj.get("test_metrics", {})
            row = {
                "experiment": experiment,
                "model": model,
                "variant": variant,
                "condition": f"{experiment}/{model}/{variant}",
                "fold": int(fold_name.replace("fold_", "")),
                "run_dir": str(path.parent),
            }
            for metric in METRICS:
                row[metric] = metrics.get(metric, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_row(left: pd.Series, right: pd.Series, metric: str, min_folds: int = 10) -> dict:
    joined = pd.DataFrame({"left": left, "right": right}).dropna()
    x = joined["left"].to_numpy(dtype=float)
    y = joined["right"].to_numpy(dtype=float)
    diff = x - y
    result = {
        "n_folds": int(len(diff)),
        "left_mean": float(np.mean(x)) if len(x) else math.nan,
        "right_mean": float(np.mean(y)) if len(y) else math.nan,
        "mean_diff_left_minus_right": float(np.mean(diff)) if len(diff) else math.nan,
        "median_diff_left_minus_right": float(np.median(diff)) if len(diff) else math.nan,
        "metric": metric,
        "eligible_for_inference": bool(len(diff) >= min_folds),
    }
    if len(diff) >= min_folds:
        result["paired_t_p"] = float(ttest_rel(x, y, nan_policy="omit").pvalue)
        try:
            result["wilcoxon_p"] = float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            result["wilcoxon_p"] = math.nan
    else:
        result["paired_t_p"] = math.nan
        result["wilcoxon_p"] = math.nan
    if len(diff) >= 2:
        result["paired_cohen_dz"] = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else math.nan
        se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
        tcrit = 2.262157 if len(diff) == 10 else 1.96
        result["mean_diff_ci95_low"] = float(np.mean(diff) - tcrit * se)
        result["mean_diff_ci95_high"] = float(np.mean(diff) + tcrit * se)
    else:
        result["paired_cohen_dz"] = math.nan
        result["mean_diff_ci95_low"] = math.nan
        result["mean_diff_ci95_high"] = math.nan
    return result


def holm_adjust(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    out = np.full(len(arr), np.nan, dtype=float)
    idx = np.where(finite)[0]
    order = idx[np.argsort(arr[finite])]
    running = 0.0
    m = len(order)
    for rank, i in enumerate(order):
        adjusted = min(1.0, float(arr[i]) * (m - rank))
        running = max(running, adjusted)
        out[i] = running
    return pd.Series(out, index=values.index)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--experiments", nargs="+", default=["p0_main"])
    p.add_argument("--reference-root", default=None, help="Optional root containing the completed p0_main ten-fold runs")
    p.add_argument("--reference-experiment", default="p0_main")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sampling-csv", default=None)
    p.add_argument("--min-folds", type=int, default=10, help="Minimum paired folds for inferential p-values (default: 10)")
    args = p.parse_args()
    root = Path(args.results_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    folds = read_final_runs(root, args.experiments)
    if args.reference_root:
        reference_root = Path(args.reference_root).expanduser().resolve()
        reference = read_final_runs(reference_root, [args.reference_experiment])
        if not reference.empty and args.reference_experiment != "p0_main":
            reference["condition"] = reference["condition"].str.replace(
                f"^{args.reference_experiment}/", "p0_main/", regex=True
            )
        folds = pd.concat([reference, folds], ignore_index=True)
    folds.to_csv(out / "revision_fold_metrics.csv", index=False)
    if folds.empty:
        raise SystemExit("No final_metrics.json files found")

    # The ten-fold main SAFF result is the reference for baseline comparisons
    # and for conditions that were trained with the same official split.
    reference_name = "p0_main/saff_noduledet/full"
    tests: list[dict] = []
    ref = folds[folds["condition"] == reference_name].set_index("fold")
    if ref.empty and "condition" in folds:
        # Convenience fallback when the main and supplementary outputs share
        # one root and the SAFF main condition is under table4_baselines.
        alt = folds[folds["condition"] == "table4_baselines/saff_noduledet/full"]
        if len(alt) >= args.min_folds:
            ref = alt.set_index("fold")
            reference_name = "table4_baselines/saff_noduledet/full"
    if len(ref) != 10:
        print(f"[warn] reference has {len(ref)} folds, expected 10")
    for condition, group in folds.groupby("condition"):
        if condition == reference_name:
            continue
        other = group.set_index("fold")
        common = sorted(set(ref.index) & set(other.index))
        for metric in METRICS:
            r = paired_row(other.loc[common, metric], ref.loc[common, metric], metric, args.min_folds)
            r.update({"left_condition": condition, "right_condition": reference_name})
            tests.append(r)
    test_df = pd.DataFrame(tests)
    if not test_df.empty:
        eligible = test_df["eligible_for_inference"]
        test_df["wilcoxon_holm_p"] = np.nan
        test_df["paired_t_holm_p"] = np.nan
        test_df.loc[eligible, "wilcoxon_holm_p"] = holm_adjust(test_df.loc[eligible, "wilcoxon_p"])
        test_df.loc[eligible, "paired_t_holm_p"] = holm_adjust(test_df.loc[eligible, "paired_t_p"])
    test_df.to_csv(out / "paired_tests.csv", index=False)

    summary_rows = []
    for condition, group in folds.groupby("condition"):
        row = {"condition": condition, "finished_folds": int(group["fold"].nunique())}
        for metric in METRICS:
            vals = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else math.nan
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out / "revision_condition_summary.csv", index=False)

    if args.sampling_csv:
        sampling = pd.read_csv(args.sampling_csv)
        sampling.to_csv(out / "candidate_sampling_metrics.csv", index=False)
        group_cols = [c for c in ["count_name", "count", "repeat"] if c in sampling.columns]
        available_metrics = [c for c in METRICS if c in sampling.columns]
        if group_cols and available_metrics:
            sampling.groupby(group_cols, dropna=False)[available_metrics].agg(["mean", "std"]).to_csv(
                out / "candidate_sampling_summary.csv"
            )
    print(f"Saved {out / 'revision_fold_metrics.csv'}")
    print(f"Saved {out / 'paired_tests.csv'}")
    print(f"Saved {out / 'revision_condition_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
