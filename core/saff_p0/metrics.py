from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _match_predictions_at_threshold(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    threshold: float,
    hit_radius_scale: float,
    min_hit_radius_mm: float,
) -> tuple[int, int, int]:
    pred_t = pred[pred["probability"] >= threshold].sort_values("probability", ascending=False)
    matched: set[int] = set()
    tp = 0
    fp = 0
    gt_by_series = {sid: g.reset_index(drop=False) for sid, g in gt.groupby("seriesuid")}
    for row in pred_t.itertuples(index=False):
        group = gt_by_series.get(str(row.seriesuid))
        if group is None or group.empty:
            fp += 1
            continue
        coords = group[["coordX", "coordY", "coordZ"]].to_numpy(dtype=float)
        p = np.array([float(row.coordX), float(row.coordY), float(row.coordZ)], dtype=float)
        dists = np.linalg.norm(coords - p[None, :], axis=1)
        order = np.argsort(dists)
        found = False
        for local_idx in order:
            global_idx = int(group.iloc[int(local_idx)]["index"])
            if global_idx in matched:
                continue
            diameter = float(group.iloc[int(local_idx)].get("diameter_mm", 0.0))
            radius = max(float(min_hit_radius_mm), float(hit_radius_scale) * diameter)
            if float(dists[int(local_idx)]) <= radius:
                matched.add(global_idx)
                tp += 1
                found = True
                break
        if not found:
            fp += 1
    fn = max(0, len(gt) - tp)
    return tp, fp, fn


def _prefix_match_counts(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    hit_radius_scale: float,
    min_hit_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass greedy matching: process every prediction in descending probability order
    and return (tp, fp) for each prefix length k.

    Why this is equivalent to the original "recompute at every threshold":
      The prediction set left by threshold t is exactly the first k entries of the
      descending-probability sequence (k = #{prob >= t}), and the greedy assignment depends
      only on the higher-probability predictions processed before it, never on
      lower-probability ones, so the result for prefix k == the result of recomputing at
      threshold t.

    The original implementation ran a full greedy match for each of the ~(201+n) thresholds
    and used pandas scalar access (`.iloc` / row-wise Series) in the inner loop, making it
    O(n^2) overall: measured at 15 minutes for a validation set with n=3.3k, and extrapolated
    to n=61k for a full fold that is ~80 hours per validation point, which is entirely
    unaffordable. This implementation reduces it to O(n).
    """
    n = len(pred)
    prob = pred["probability"].to_numpy(dtype=float)
    # Descending probability; ties are broken by ascending original row index to guarantee determinism (the original sort_values was unstable for ties)
    order = np.lexsort((np.arange(n), -prob))

    gt_by_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid, g in gt.groupby("seriesuid", sort=False):
        if "diameter_mm" in g.columns:
            diams = pd.to_numeric(g["diameter_mm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            diams = np.zeros(len(g), dtype=float)
        gt_by_series[str(sid)] = (
            g[["coordX", "coordY", "coordZ"]].to_numpy(dtype=float),
            diams,
            # The original code took group.reset_index(drop=False)["index"]; gt has already been
            # reset_index(drop=True), so the group's index labels are exactly that global column.
            g.index.to_numpy(dtype=np.int64),
        )

    sid_all = pred["seriesuid"].astype(str).to_numpy()
    xyz_all = pred[["coordX", "coordY", "coordZ"]].to_numpy(dtype=float)

    tp_at = np.zeros(n + 1, dtype=np.int64)
    fp_at = np.zeros(n + 1, dtype=np.int64)
    matched: set[int] = set()
    tp = 0
    fp = 0
    for k, i in enumerate(order.tolist(), start=1):
        grp = gt_by_series.get(sid_all[i])
        if grp is None or grp[0].shape[0] == 0:
            fp += 1
        else:
            coords, diams, gidx = grp
            dists = np.linalg.norm(coords - xyz_all[i][None, :], axis=1)
            found = False
            for j in np.argsort(dists):
                j = int(j)
                gj = int(gidx[j])
                if gj in matched:
                    continue
                radius = max(float(min_hit_radius_mm), float(hit_radius_scale) * float(diams[j]))
                if float(dists[j]) <= radius:
                    matched.add(gj)
                    tp += 1
                    found = True
                    break
            if not found:
                fp += 1
        tp_at[k] = tp
        fp_at[k] = fp
    return order, tp_at, fp_at


def compute_froc(
    predictions: list[dict[str, Any]] | pd.DataFrame,
    gt_annotations: pd.DataFrame,
    fp_targets: list[float],
    hit_radius_scale: float,
    min_hit_radius_mm: float,
    scan_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred = pd.DataFrame(predictions).copy()
    gt = gt_annotations.copy().reset_index(drop=True)
    if pred.empty:
        curve = pd.DataFrame(columns=["threshold", "sensitivity", "fp_per_scan", "tp", "fp", "fn"])
        metrics = {f"sens_at_{t:g}": 0.0 for t in fp_targets}
        metrics.update({"cpm": 0.0, "n_scans": int(scan_count or 0), "n_gt": int(len(gt)), "n_predictions": 0})
        return curve, metrics
    scan_count = max(1, int(scan_count or pred["seriesuid"].nunique()))
    prob = pred["probability"].to_numpy(dtype=float)
    thresholds = np.unique(np.r_[np.linspace(0.0, 1.0, 201), prob])
    order, tp_at, fp_at = _prefix_match_counts(pred, gt, hit_radius_scale, min_hit_radius_mm)
    # The number of predictions at threshold t is k = #{prob >= t}; read it straight from the prefix table (-desc is ascending, equivalent to counting >= t in descending order)
    desc = prob[order]
    k_of = np.searchsorted(-desc, -thresholds, side="right")
    rows = []
    for idx in np.argsort(-thresholds):
        threshold = float(thresholds[idx])
        k = int(k_of[idx])
        tp = int(tp_at[k])
        fp = int(fp_at[k])
        fn = max(0, len(gt) - tp)
        sensitivity = tp / max(1, len(gt))
        fp_per_scan = fp / scan_count
        rows.append(
            {
                "threshold": threshold,
                "sensitivity": float(sensitivity),
                "fp_per_scan": float(fp_per_scan),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    curve = pd.DataFrame(rows).sort_values("fp_per_scan").reset_index(drop=True)
    metrics: dict[str, Any] = {
        "n_scans": scan_count,
        "n_gt": int(len(gt)),
        "n_predictions": int(len(pred)),
    }
    sens_values = []
    for target in fp_targets:
        eligible = curve[curve["fp_per_scan"] <= float(target)]
        sens = float(eligible["sensitivity"].max()) if not eligible.empty else 0.0
        metrics[f"sens_at_{target:g}"] = sens
        sens_values.append(sens)
    metrics["cpm"] = float(np.mean(sens_values)) if sens_values else math.nan
    best_idx = int(curve["sensitivity"].idxmax()) if not curve.empty else 0
    if not curve.empty:
        metrics["max_sensitivity"] = float(curve.loc[best_idx, "sensitivity"])
        metrics["fp_per_scan_at_max_sens"] = float(curve.loc[best_idx, "fp_per_scan"])
    return curve, metrics


def write_predictions(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_curve_plot(curve: pd.DataFrame, path: str | Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if not curve.empty:
        plt.plot(curve["fp_per_scan"], curve["sensitivity"], marker=".", linewidth=1.4)
    plt.xscale("symlog", linthresh=0.125)
    plt.xlim(0, max(8.0, float(curve["fp_per_scan"].max()) if not curve.empty else 8.0))
    plt.ylim(0, 1.02)
    plt.xlabel("False positives per scan")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
