"""Generate Figures 3, 4 and 6 from real experiment outputs (no demo data)."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":8})

def require(path: Path) -> Path:
    if not path.exists(): raise FileNotFoundError(f"Required result is missing: {path}")
    return path

def fig3(results: Path, experiment: str, out: Path) -> None:
    rows=[]
    for p in sorted((results/"runs"/experiment/"saff_noduledet"/"full").glob("fold_*/history.csv")):
        d=pd.read_csv(p); d["fold"]=p.parent.name; rows.append(d)
    if not rows: raise FileNotFoundError("No history.csv files found for Figure 3")
    g=pd.concat(rows, ignore_index=True).groupby("epoch"); e=g.size().index.to_numpy()
    fig,ax1=plt.subplots(figsize=(7.2,4.2),dpi=220)
    for col,color,label in [("train_loss","#1f77b4","Training loss"),("val_loss","#d95f02","Validation loss")]:
        m,s=g[col].mean(),g[col].std(ddof=1).fillna(0); ax1.plot(e,m,color=color,lw=2,label=label); ax1.fill_between(e,m-s,m+s,color=color,alpha=.12,lw=0)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(alpha=.22); ax2=ax1.twinx(); m,s=g["val_cpm"].mean(),g["val_cpm"].std(ddof=1).fillna(0); ax2.plot(e,m,color="#2ca25f",lw=2,label="Validation CPM"); ax2.fill_between(e,m-s,m+s,color="#2ca25f",alpha=.14,lw=0); ax2.set_ylabel("CPM")
    lines=ax1.get_lines()+ax2.get_lines(); ax1.legend(lines,[x.get_label() for x in lines],loc="center right",frameon=False); fig.tight_layout(); fig.savefig(out/"figure3_convergence_10fold.png",dpi=300,bbox_inches="tight"); plt.close(fig)

def fig4(results: Path, experiment: str, out: Path) -> None:
    df=pd.read_csv(require(results/"summaries"/experiment/"fold_metrics.csv")); fp=[(.125,"sens_at_0.125"),(.25,"sens_at_0.25"),(.5,"sens_at_0.5"),(1,"sens_at_1"),(2,"sens_at_2"),(4,"sens_at_4"),(8,"sens_at_8")]; fig,ax=plt.subplots(figsize=(7.2,4.5),dpi=220)
    for color,(model,g) in zip(plt.cm.tab10(np.linspace(0,.9,max(1,df.model.nunique()))),df.groupby("model")):
        means=[pd.to_numeric(g[c],errors="coerce").mean() for _,c in fp]; stds=[pd.to_numeric(g[c],errors="coerce").std(ddof=1) for _,c in fp]; x=[v for v,_ in fp]; ax.plot(x,means,marker="o",ms=4,lw=2.4 if model=="saff_noduledet" else 1.6,color=color,label=model); ax.fill_between(x,np.array(means)-np.nan_to_num(stds),np.array(means)+np.nan_to_num(stds),color=color,alpha=.1,lw=0)
    ax.set_xscale("symlog",linthresh=.125); ax.set_xticks([v for v,_ in fp]); ax.set_xticklabels([str(v) for v,_ in fp]); ax.set_xlabel("False positives per scan"); ax.set_ylabel("Sensitivity"); ax.grid(alpha=.22,which="both"); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(out/"figure4_froc_table4.png",dpi=300,bbox_inches="tight"); plt.close(fig)

def fig6(results: Path, experiment: str, out: Path) -> None:
    df=pd.read_csv(require(results/"summaries"/experiment/"model_summary.csv")); df=df[df.model=="saff_noduledet"].copy()
    if df.empty: raise ValueError("No SAFF ablation rows found for Figure 6")
    labels=df.variant.astype(str).tolist(); y=pd.to_numeric(df.cpm_mean,errors="coerce").to_numpy(); s=pd.to_numeric(df.cpm_std,errors="coerce").fillna(0).to_numpy(); fig,ax=plt.subplots(figsize=(7.5,4.2),dpi=220); ax.bar(np.arange(len(labels)),y,yerr=s,capsize=3,color=["#1f77b4" if x=="full" else "#8da0cb" for x in labels],edgecolor="white");
    if "full" in labels: ax.axhline(y[labels.index("full")],color="#1f77b4",ls="--",lw=1,alpha=.65)
    ax.set_ylabel("CPM"); ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels,rotation=28,ha="right"); ax.grid(axis="y",alpha=.22); fig.tight_layout(); fig.savefig(out/"figure6_ablation_table6.png",dpi=300,bbox_inches="tight"); plt.close(fig)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--results-root",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--main-experiment",default="table4_baselines"); p.add_argument("--ablation-experiment",default="table6_ablation"); p.add_argument("--skip-figure3",action="store_true"); a=p.parse_args(); results,out=Path(a.results_root).expanduser(),Path(a.out_dir).expanduser(); out.mkdir(parents=True,exist_ok=True)
    if not a.skip_figure3: fig3(results,a.main_experiment,out)
    fig4(results,a.main_experiment,out); fig6(results,a.ablation_experiment,out); print(f"Saved manuscript figures to {out}"); return 0
if __name__=="__main__": raise SystemExit(main())
