#!/usr/bin/env python
"""Turn the runs/ produced by the rerun into a results package that maps directly onto the paper tables (Tables 4/6/7 + paired tests + Figures 3/4/6 + self-check report).

It depends only on numpy / pandas / scipy / matplotlib, reads each fold's final_metrics.json,
and never modifies any training artifact.

Usage:
  python analysis/make_results_package.py \
      --runs-root /root/autodl-tmp/SAFF_ws/SAFF_rerun/outputs_rerun \
      --out-dir   /root/autodl-tmp/SAFF_ws/SAFF_rerun/results_package \
      --reference table4_baselines/saff_noduledet/full
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["cpm", "sens_at_0.125", "sens_at_0.25", "sens_at_0.5", "sens_at_1", "sens_at_2", "sens_at_4", "sens_at_8"]
PAPER_METRICS = ["cpm", "sens_at_0.125", "sens_at_1"]

MODEL_LABELS = {
    "saff_noduledet": "SAFF-NoduleDet",
    "unet3d": "3D U-Net",
    "convnext3d": "3D ConvNeXt-adapted",
    "swin_unetr": "Swin UNETR-adapted",
    "cotr3d": "CoTr3D",
}
# Each hyperparameter setting in Table 7 must live in its **own experiment directory**, otherwise same-named variants overwrite each other
T7_EXPERIMENTS = {
    "region_r3", "region_r5", "region_r7",
    "distill_w_0.05", "distill_w_0.10", "distill_w_0.20",
    "smooth_w_0", "smooth_w_0.005", "smooth_w_0.010",
}
VARIANT_LABELS = {
    "full": "Full model",
    "no_distill": "w/o cross-branch distillation",
    "no_morph_gate": "w/o morphology-aware gating",
    "fixed_fusion": "Fixed fusion",
    "global_norm": "Global normalization",
    "cnn_only": "CNN branch only",
    "transformer_only": "Transformer branch only",
    "full_attention": "Full attention",
}


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running, m = [1.0] * len(pvals), 0.0, len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvals[i] * (m - rank)))
        out[i] = running
    return out


def read_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*/*/*/fold_*/final_metrics.json")):
        parts = path.relative_to(runs_dir).parts
        experiment, model, variant, fold_name = parts[0], parts[1], parts[2], parts[3]
        obj = json.loads(path.read_text(encoding="utf-8"))
        metrics = obj.get("test_metrics", {})
        row = {"experiment": experiment, "model": model, "variant": variant,
               "condition": f"{experiment}/{model}/{variant}", "fold": int(fold_name.replace("fold_", "")),
               "run_dir": str(path.parent), "best_epoch": obj.get("best_epoch")}
        for m in METRICS:
            row[m] = metrics.get(m, np.nan)
        row["n_scans"] = metrics.get("n_scans")
        row["n_gt"] = metrics.get("n_gt")
        row["n_predictions"] = metrics.get("n_predictions")
        cfg_path = path.parent / "config_resolved.yaml"
        if cfg_path.exists():
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            row["regional_regions"] = cfg.get("data", {}).get("regional_regions")
            row["distill_weight"] = cfg.get("model", {}).get("distill_weight")
            row["gate_smooth_weight"] = cfg.get("model", {}).get("gate_smooth_weight")
            row["eval_every"] = cfg.get("training", {}).get("eval_every", 1)
            row["eval_sort_by_scan"] = cfg.get("data", {}).get("eval_sort_by_scan", False)
            row["eval_negatives_per_scan"] = cfg.get("data", {}).get("eval_negatives_per_scan")
        rows.append(row)
    return pd.DataFrame(rows)


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    wp = ["sens_at_0.125", "sens_at_0.25", "sens_at_0.5", "sens_at_1", "sens_at_2", "sens_at_4", "sens_at_8"]
    for r in df.to_dict("records"):
        vals = [r.get(m) for m in wp]
        cpm = r.get("cpm")
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals) or cpm is None or (isinstance(cpm, float) and math.isnan(cpm)):
            out.append({"condition": r["condition"], "fold": r["fold"], "cpm_recorded": cpm, "cpm_recomputed": np.nan,
                        "abs_diff": np.nan, "ok": False, "note": "missing working points"})
            continue
        rec = float(np.mean(vals))
        ok = abs(rec - float(cpm)) < 1e-3
        out.append({"condition": r["condition"], "fold": r["fold"], "cpm_recorded": float(cpm), "cpm_recomputed": rec,
                    "abs_diff": rec - float(cpm), "ok": ok, "note": "" if ok else "CPM is inconsistent with the mean of the 7 working points"})
    return pd.DataFrame(out)


def paired_tests(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    from scipy.stats import ttest_rel, wilcoxon
    ref = df[df.condition == reference].set_index("fold")
    rows = []
    if ref.empty:
        raise SystemExit(f"Reference condition {reference} has no fold results")
    for cond, group in df.groupby("condition"):
        if cond == reference:
            continue
        other = group.set_index("fold")
        common = sorted(set(ref.index) & set(other.index))
        pvals, entries = [], []
        for metric in PAPER_METRICS:
            x = other.loc[common, metric].to_numpy(dtype=float)
            y = ref.loc[common, metric].to_numpy(dtype=float)
            d = x - y
            if len(d) < 2 or np.all(np.isnan(d)):
                continue
            t_p = float(ttest_rel(x, y, nan_policy="omit").pvalue)
            try:
                w_p = float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                w_p = math.nan
            se = float(np.std(d, ddof=1) / math.sqrt(len(d)))
            tcrit = 2.262157 if len(d) == 10 else 1.96
            entries.append({"condition": cond, "reference": reference, "metric": metric, "n_folds": len(d),
                            "mean_diff_condition_minus_reference": float(np.mean(d)), "std_diff": float(np.std(d, ddof=1)),
                            "ci95_low": float(np.mean(d) - tcrit * se), "ci95_high": float(np.mean(d) + tcrit * se),
                            "wilcoxon_p": w_p, "paired_t_p": t_p,
                            "cohen_dz": float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else np.nan})
            pvals.append(w_p if np.isfinite(w_p) else 1.0)
        if not entries:
            continue
        adj = holm(pvals)
        for e, a in zip(entries, adj):
            e["wilcoxon_holm_p_3metrics"] = a
        rows.extend(entries)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition"):
        row = {"condition": cond, "finished_folds": int(g["fold"].nunique())}
        for m in METRICS:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cpm_mean", ascending=False)


def paper_tables(df: pd.DataFrame, summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s = summary.set_index("condition")
    def cell(cond, m):
        if cond not in s.index or math.isnan(s.loc[cond, f"{m}_mean"]):
            return "MISSING"
        return f"{s.loc[cond, f'{m}_mean']:.3f} ± {s.loc[cond, f'{m}_std']:.3f}"
    t4 = []
    for model in ["unet3d", "swin_unetr", "convnext3d", "cotr3d", "saff_noduledet"]:
        for exp in ["table4_baselines", "p0_main"]:
            c = f"{exp}/{model}/full"
            if c in s.index:
                t4.append({"Model": MODEL_LABELS.get(model, model), "condition": c,
                           **{m: cell(c, m) for m in ["cpm", "sens_at_0.125", "sens_at_0.25", "sens_at_0.5", "sens_at_1", "sens_at_2", "sens_at_4", "sens_at_8"]}})
                break
    t6 = []
    for exp in ["table6_ablation"]:
        for variant in ["full", "no_distill", "no_morph_gate", "fixed_fusion", "global_norm", "cnn_only", "transformer_only", "full_attention"]:
            c = f"{exp}/saff_noduledet/{variant}"
            if c in s.index:
                t6.append({"Variant": VARIANT_LABELS.get(variant, variant), "condition": c,
                           **{m: cell(c, m) for m in ["cpm", "sens_at_0.125", "sens_at_0.25", "sens_at_0.5", "sens_at_1", "sens_at_2", "sens_at_4", "sens_at_8"]}})
    t7 = []
    for c in s.index:
        if not any(c.startswith(f"{e}/") for e in T7_EXPERIMENTS):
            continue
        r = s.loc[c]
        t7.append({"condition": c, "r": r.get("regional_regions"), "lambda_dist": r.get("distill_weight"),
                   "lambda_smooth": r.get("gate_smooth_weight"),
                   "CPM": cell(c, "cpm"), "Sens@0.125": cell(c, "sens_at_0.125"), "Sens@1.0": cell(c, "sens_at_1")})
    return {"table4": pd.DataFrame(t4), "table6": pd.DataFrame(t6), "table7": pd.DataFrame(t7)}


def figures(df: pd.DataFrame, runs_dir: Path, out: Path) -> list[str]:
    made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable, skipping figures: {exc}"]
    fp = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    # Figure 4: FROC (using the relevant table4 conditions)
    t4 = df[df.experiment.isin(["table4_baselines", "p0_main"])]
    if not t4.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for i, (cond, g) in enumerate(t4.groupby("condition")):
            means = [pd.to_numeric(g[m], errors="coerce").mean() for m in METRICS[1:]]
            stds = [pd.to_numeric(g[m], errors="coerce").std(ddof=1) for m in METRICS[1:]]
            lw = 2.4 if cond.endswith("saff_noduledet/full") else 1.6
            color = plt.cm.tab10(i % 10)
            ax.plot(fp, means, marker="o", ms=4, lw=lw, color=color, label=cond.split("/")[-2])
            ax.fill_between(fp, np.array(means) - np.nan_to_num(stds), np.array(means) + np.nan_to_num(stds), color=color, alpha=0.1, lw=0)
        ax.set_xscale("symlog", linthresh=0.125); ax.set_xticks(fp); ax.set_xticklabels([str(v) for v in fp])
        ax.set_xlabel("False positives per scan"); ax.set_ylabel("Sensitivity"); ax.grid(alpha=.22, which="both"); ax.legend(frameon=False, ncol=2)
        fig.tight_layout(); fig.savefig(out / "figure4_froc.png", dpi=300, bbox_inches="tight"); plt.close(fig)
        made.append("figure4_froc.png")
    # Figure 6: ablation bar chart
    t6 = df[df.experiment == "table6_ablation"]
    if not t6.empty:
        g = t6.groupby("variant")["cpm"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(range(len(g)), g["mean"], yerr=g["std"].fillna(0), capsize=3,
               color=["#1f77b4" if v == "full" else "#8da0cb" for v in g.index], edgecolor="white")
        ax.set_ylabel("CPM"); ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=28, ha="right"); ax.grid(axis="y", alpha=.22)
        fig.tight_layout(); fig.savefig(out / "figure6_ablation.png", dpi=300, bbox_inches="tight"); plt.close(fig)
        made.append("figure6_ablation.png")
    # Figure 3: convergence curves (using the reference condition's history.csv)
    hist = runs_dir / "table4_baselines" / "saff_noduledet" / "full"
    files = sorted(hist.glob("fold_*/history.csv"))
    if files:
        frames = [pd.read_csv(f).assign(fold=f.parent.name) for f in files]
        h = pd.concat(frames, ignore_index=True)
        by = h.groupby("epoch")
        fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
        for col, color, label in [("train_loss", "#1f77b4", "Training loss"), ("val_loss", "#d95f02", "Validation loss")]:
            m, sd = by[col].mean(), by[col].std(ddof=1).fillna(0)
            ax1.plot(m.index, m, color=color, lw=2, label=label); ax1.fill_between(m.index, m - sd, m + sd, color=color, alpha=.12, lw=0)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(alpha=.22)
        ax2 = ax1.twinx(); m, sd = by["val_cpm"].mean(), by["val_cpm"].std(ddof=1).fillna(0)
        ax2.plot(m.index, m, color="#2ca25f", lw=2, label="Validation CPM")
        ax2.fill_between(m.index, m - sd, m + sd, color="#2ca25f", alpha=.14, lw=0); ax2.set_ylabel("CPM")
        lines = ax1.get_lines() + ax2.get_lines(); ax1.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=False)
        fig.tight_layout(); fig.savefig(out / "figure3_convergence.png", dpi=300, bbox_inches="tight"); plt.close(fig)
        made.append("figure3_convergence.png")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference", default="table4_baselines/saff_noduledet/full")
    args = ap.parse_args()

    runs_dir = Path(args.runs_root).expanduser() / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"Cannot find {runs_dir}")
    out = Path(args.out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)

    df = read_runs(runs_dir)
    if df.empty:
        raise SystemExit("No final_metrics.json was found")
    df.to_csv(out / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    cons = consistency(df)
    cons.to_csv(out / "consistency_report.csv", index=False, encoding="utf-8-sig")

    summary = summarize(df)
    summary.to_csv(out / "condition_summary.csv", index=False, encoding="utf-8-sig")

    reference = args.reference
    if reference not in set(df.condition):
        alt = [c for c in set(df.condition) if c.endswith("saff_noduledet/full")]
        if not alt:
            raise SystemExit("Cannot find a SAFF reference condition; specify one with --reference")
        reference = sorted(alt)[0]
        print(f"[info] reference condition automatically changed to {reference}")
    tests = paired_tests(df, reference)
    tests.to_csv(out / "paired_tests.csv", index=False, encoding="utf-8-sig")

    tables = paper_tables(df, summary)
    for name, t in tables.items():
        t.to_csv(out / f"paper_{name}.csv", index=False, encoding="utf-8-sig")

    figs = figures(df, Path(args.runs_root).expanduser(), out)

    bad = cons[~cons.ok.fillna(False)]
    lines = ["# Rerun results package\n",
             f"- Conditions: {df.condition.nunique()}; fold records: {len(df)}",
             f"- Folds per condition:\n{df.groupby('condition')['fold'].nunique().to_string()}",
             f"- Reference condition: `{reference}`",
             f"- Self-consistency check: {len(cons) - len(bad)}/{len(cons)} passed (CPM = mean of the 7 working points)",
             f"- Paired tests: `paired_tests.csv` (Holm correction restricted to the three tests CPM / Sens@0.125 / Sens@1.0)",
             f"- Figures: {', '.join(figs) if figs else 'none'}",
             "\n## Mapping to the paper tables\n",
             "- `paper_table4.csv` / `paper_table6.csv` / `paper_table7.csv`: rows with measured results are filled in; `MISSING` means that fold set has not been run yet.\n",
             "\n## Protocol flags (must stay consistent when writing the methodology)\n"]
    for col in ["eval_every", "eval_sort_by_scan", "eval_negatives_per_scan"]:
        if col in df.columns:
            lines.append(f"- {col}: {sorted(set(df[col].dropna().tolist()))}")
    if len(bad):
        lines.append("\n## ⚠️ Inconsistent folds (do not use in the paper)\n")
        lines.append(bad.to_string(index=False))
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Results package written to: {out}")
    print(f"  {df.condition.nunique()} conditions, {len(df)} fold records, self-consistency {len(cons) - len(bad)}/{len(cons)} passed")
    if len(bad):
        print("  ⚠️ Inconsistent folds present, see consistency_report.csv for details")
    for name in tables:
        t = tables[name]
        miss = int((t.astype(str) == "MISSING").any(axis=1).sum()) if not t.empty else 0
        print(f"  {name}: {0 if t.empty else len(t)} rows, {miss} of them missing results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
