from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from pathlib import Path

import pandas as pd

from .config import ensure_dir, load_config, output_root


METRIC_KEYS = [
    "cpm",
    "sens_at_0.125",
    "sens_at_0.25",
    "sens_at_0.5",
    "sens_at_1",
    "sens_at_2",
    "sens_at_4",
    "sens_at_8",
    "loss",
    "elapsed_sec",
    "samples_per_sec",
    "peak_memory_bytes",
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return "", "", ""
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std, f"{mean:.4f} ± {std:.4f}"


def collect_runs(root: Path, experiment: str) -> list[dict]:
    base = root / "runs" / experiment
    rows = []
    if not base.exists():
        return rows
    for final_path in sorted(base.glob("*/*/fold_*/final_metrics.json")):
        parts = final_path.relative_to(base).parts
        model, variant, fold_name = parts[0], parts[1], parts[2]
        final = read_json(final_path)
        test = final.get("test_metrics", {})
        model_info_path = final_path.parent / "model_info.json"
        model_info = read_json(model_info_path) if model_info_path.exists() else {}
        row = {
            "experiment": experiment,
            "model": model,
            "variant": variant,
            "fold": int(fold_name.replace("fold_", "")),
            "best_epoch": final.get("best_epoch"),
            "best_val_cpm": final.get("best_val_cpm"),
            "trainable_parameters": model_info.get("trainable_parameters"),
            "run_dir": str(final_path.parent),
        }
        for key in METRIC_KEYS:
            row[key] = test.get(key)
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    out = []
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_key.setdefault((row["model"], row["variant"]), []).append(row)
    for (model, variant), group in sorted(by_key.items()):
        s = {"model": model, "variant": variant, "finished_folds": len(group)}
        for key in ["best_epoch", "best_val_cpm", "trainable_parameters"] + METRIC_KEYS:
            mean, std, text = mean_std([g.get(key) for g in group])
            s[f"{key}_mean"] = mean
            s[f"{key}_std"] = std
            s[f"{key}_mean_std"] = text
        out.append(s)
    return out


def make_paper_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    main_cols = [
        "model",
        "variant",
        "cpm_mean_std",
        "sens_at_0.5_mean_std",
        "sens_at_1_mean_std",
        "sens_at_2_mean_std",
        "sens_at_4_mean_std",
        "trainable_parameters_mean_std",
        "elapsed_sec_mean_std",
        "peak_memory_bytes_mean_std",
    ]
    cols = [c for c in main_cols if c in summary.columns]
    summary[cols].to_csv(out_dir / "paper_table_main.csv", index=False)
    ablation = summary[summary["model"] == "saff_noduledet"]
    if not ablation.empty:
        ablation[cols].to_csv(out_dir / "paper_table_ablation.csv", index=False)


def copy_froc_curves(root: Path, experiment: str, out_dir: Path) -> None:
    curve_dir = ensure_dir(out_dir / "froc_curves")
    base = root / "runs" / experiment
    if not base.exists():
        return
    for path in sorted(base.glob("*/*/fold_*/test_froc_curve.csv")):
        rel = path.relative_to(base)
        name = "__".join(rel.parts[:-1]) + "__test_froc_curve.csv"
        shutil.copy2(path, curve_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", default=None, help="Override config data.output_root")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_root:
        cfg["data"]["output_root"] = args.output_root
    root = output_root(cfg)
    out_dir = ensure_dir(root / "summaries" / args.experiment)
    rows = collect_runs(root, args.experiment)
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = summarize_rows(rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "model_summary.csv", index=False)
    make_paper_tables(summary_df, out_dir)
    copy_froc_curves(root, args.experiment, out_dir)
    print(f"Saved {out_dir / 'fold_metrics.csv'}")
    print(f"Saved {out_dir / 'model_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
