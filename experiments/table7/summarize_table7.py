"""Create the manuscript-ready Table 7 from completed ten-fold runs.

The script never invents missing folds. A condition is marked ``complete`` only
when all requested folds (by default 0--9) have a finite final metric.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"CPM": "cpm", "Sens@0.125": "sens_at_0.125", "Sens@1.0": "sens_at_1"}
DEFAULT = {"r": 5, "lambda_dist": 0.10, "lambda_smooth": 0.005}


def _read(root: Path, experiment: str, folds: list[int]) -> pd.DataFrame:
    rows = []
    base = root / "runs" / experiment / "saff_noduledet" / "full"
    for fold in folds:
        path = base / f"fold_{fold}" / "final_metrics.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        m = obj.get("test_metrics", {})
        rows.append({"condition": experiment, "fold": fold, **{k: m.get(v, np.nan) for k, v in METRICS.items()}})
    return pd.DataFrame(rows)


def _label(exp: str) -> tuple[str, float]:
    if exp.startswith("region_r"):
        return "r", float(exp.removeprefix("region_r"))
    if exp.startswith("distill_w_"):
        return "lambda_dist", float(exp.removeprefix("distill_w_"))
    if exp.startswith("smooth_w_"):
        return "lambda_smooth", float(exp.removeprefix("smooth_w_"))
    return "unknown", math.nan


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experiments", nargs="+", default=[
        "region_r3", "region_r5", "region_r7",
        "distill_w_0.05", "distill_w_0.10", "distill_w_0.20",
        "smooth_w_0", "smooth_w_0.005", "smooth_w_0.010",
    ])
    p.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    args = p.parse_args()
    root, out = Path(args.results_root).expanduser(), Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    frames = [_read(root, e, args.folds) for e in args.experiments]
    fold_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    fold_df.to_csv(out / "table7_fold_metrics.csv", index=False)
    rows = []
    for exp in args.experiments:
        g = fold_df[fold_df.get("condition", pd.Series(dtype=str)) == exp] if not fold_df.empty else pd.DataFrame()
        key, value = _label(exp)
        row = {"condition": exp, "parameter": key, "parameter_value": value,
               "finished_folds": int(g["fold"].nunique()) if not g.empty else 0,
               "expected_folds": len(args.folds)}
        row["status"] = "complete" if row["finished_folds"] == row["expected_folds"] else "incomplete"
        for label, col in METRICS.items():
            vals = pd.to_numeric(g[label], errors="coerce").dropna().to_numpy(float) if not g.empty else np.array([])
            row[f"{label}_mean"] = float(vals.mean()) if len(vals) else math.nan
            row[f"{label}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
            row[f"{label}_mean_std"] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}" if len(vals) > 1 else ""
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Δ is calculated only within each parameter sweep against its own centre.
    for key, default_value in DEFAULT.items():
        ref = summary[(summary["parameter"] == key) & np.isclose(summary["parameter_value"], default_value)]
        if len(ref) != 1 or ref.iloc[0]["status"] != "complete":
            continue
        ref = ref.iloc[0]
        mask = summary["parameter"] == key
        for label in METRICS:
            summary.loc[mask, f"delta_{label}"] = summary.loc[mask, f"{label}_mean"] - ref[f"{label}_mean"]
    summary.to_csv(out / "table7_summary.csv", index=False)
    lines = ["# Table 7. Hyper-parameter sensitivity (ten-fold)", "", "Only rows marked `complete` (all requested folds) may be reported in the manuscript.", ""]
    cols = ["condition", "parameter", "parameter_value", "status"] + [f"{m}_mean_std" for m in METRICS] + [f"delta_{m}" for m in METRICS]
    if not summary.empty:
        lines.extend(summary[cols].to_markdown(index=False, floatfmt=".4f").splitlines())
    (out / "table7_markdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out / 'table7_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
