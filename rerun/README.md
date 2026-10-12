# Rerun toolkit (low-cost, interruptible, resumable)

> **What this package is for**: use your existing code + the LUNA16 data to **actually** rerun all the results the paper needs, and automatically organize them into
> Tables 4 / 6 / 7, Figures 3/4/6 and the paired-test results.
>
> **What it does not do**: it will not, and cannot, "generate" experimental results. If the old result files are lost, the only option is to rerun; any fabricated data package
> will be exposed when reviewers ask for the per-fold materials, and the cost of that far exceeds the rerun cost.

---

## 0. The one-line workflow

```bash
# ① Calibration (10 minutes, a few cents): run 1 fold for 2 epochs to measure the per-sample time
cd <code package root>
FOLDS="0" EPOCHS=2 TIER=1 bash rerun/run_priority.sh

# ② Estimate the total cost (replace the defaults with the measured times)
python tools/estimate_cost.py --config configs/rerun_4090_final.yaml --price 2.5 \
    --train-sec 0.02 --eval-sec 0.008

# ③ Run by tier (Ctrl-C at any time; a rerun automatically skips completed folds)
TIER=1 bash rerun/run_priority.sh     # Table 4 SAFF+CoTr3D (the minimal set for the 4.3 paired tests)
TIER=2 bash rerun/run_priority.sh     # then add the other three baselines of Table 4
TIER=3 bash rerun/run_priority.sh     # then add the Table 6 ablations
TIER=4 bash rerun/run_priority.sh     # then add the Table 7 sensitivity runs

# ④ Produce the package
python analysis/make_results_package.py \
    --runs-root /root/autodl-tmp/SAFF_ws/SAFF_rerun/outputs_rerun \
    --out-dir   /root/autodl-tmp/SAFF_ws/SAFF_rerun/results_package
```

---

## 1. Prerequisites (one-time)

| Item | Requirement | Check command |
|---|---|---|
| LUNA16 | `subset0..9` (888 cases of `.mhd/.raw`), `annotations.csv`, `candidates.csv` | `ls <LUNA16>/subset_data \| head` |
| Lung masks | `seg-lungs-LUNA16` (888 paired files) | `ls <LUNA16>/seg-lungs-LUNA16 \| wc -l` |
| Environment | Python 3.10+, numpy/pandas/scipy/PyYAML/tqdm/matplotlib + torch(CUDA) | `python -c "import torch;print(torch.cuda.is_available())"` |
| Disk | results + checkpoints about **10–20 GB** (each fold's best/last two ckpts are about 70 MB) | `df -h /root/autodl-tmp` |
| GPU memory | 24 GB (at batch=1 a single process actually uses about 3–4 GB) | `nvidia-smi` |

```bash
python -m pip install -r requirements.txt
python -m saff_p0.check_data --config configs/rerun_4090_final.yaml   # print the data overview and the per-fold sample counts
```

---

## 2. What this package changes relative to the original code (important)

### 2.1 Speed-related (**does not change the evaluation protocol**)

| Change | Location | Why |
|---|---|---|
| `data.eval_sort_by_scan: true` | `data.py` | Evaluation candidates are sorted by scan, so the volume cache actually hits (after the original `shuffle`, **every candidate had to reload and re-normalize the entire volume**, which is the main cause of slow evaluation). FROC/CPM are independent of candidate order, so the metrics are exactly equivalent. |
| `training.eval_every: 5` | `train_eval.py` | The validation set was originally evaluated **once per epoch** (about 120k candidates each time). Changed to once every 5 epochs (epochs 1 and the last epoch are always validated); this only changes the search grid for `best.pt`, and the test folds are still evaluated only once. |
| `early_stop_patience` changed to be judged by "the number of epochs since the last improvement" | `train_eval.py` | This keeps the early-stopping semantics the same as before when `eval_every` changes (the original implementation counted "the number of evaluations without improvement"). |
| `num_workers: 8` (training) / `eval_num_workers: 2` (evaluation) | `configs/rerun_4090_final.yaml` | Training is I/O-intensive; with enough memory it can go up to 8–12. The evaluation loader is non-persistent and exits once evaluation finishes. |
| Print `epoch_sec` / `eta_min` every epoch and write `run_times.json` | `train_eval.py` | Lets you see the remaining time at any moment, which helps control the budget. |
| `swin_unetr` heads separated out as `swin_adapted_heads: 8` | `models.py` | The 256-dimensional token is not divisible by 6, and the original config would directly raise `AssertionError`; now config errors are reported explicitly. |

> **These four must be written into the paper methodology or the reply letter**: evaluation candidates sorted by scan, validation frequency of 5 epochs, early stopping counted in epochs,
> Swin-adapted using 8 attention heads. They do not affect the test protocol or the metric definitions, but they must be stated truthfully.

### 2.2 New files

| File | Role |
|---|---|
| `configs/rerun_4090_final.yaml` | Rerun-specific config (speed switches on, everything else consistent with the paper protocol) |
| `tools/estimate_cost.py` | Estimates hours and cost from the true sample count + the measured per-sample time, and gives tiered cumulative totals |
| `tools/make_fake_luna.py` | Generates a **synthetic** mini dataset for running the whole pipeline through on CPU (must not be used for the paper) |
| `analysis/make_results_package.py` | Organizes `runs/` into Tables 4/6/7 + paired tests + Figures 3/4/6 + a self-consistency report |
| `analysis/paired_tests_cotr3d.py` | Computes the three-metric tests for SAFF vs CoTr3D separately (Holm strictly limited to the 3 tests) |
| `rerun/run_priority.sh` | Tiered runner (automatically skips completed folds, supports interrupt/resume) |

---

## 3. Cost (estimation method and order of magnitude)

Sample counts (computed automatically by the script on the real LUNA16):

- Training samples per fold ≈ number of positive nodules × (1 + 4); validation/test ≈ `eval_negatives_per_scan × scans + positive candidates`
- 80 epochs of training; `eval_every=5` → about 17 validations + 1 test

Cost = Σ(conditions × 10 folds) × hours per fold × hourly rate. **First calibrate with the 2-epoch run from ①**, then run:

```bash
python tools/estimate_cost.py --config configs/rerun_4090_final.yaml --price 2.5 \
    --train-sec <measured> --eval-sec <measured> --json-out cost.json
```

Tiered cumulative totals (printed by the script): tier 1 = SAFF + CoTr3D of Table 4 (2 conditions); tier 2 adds 3 baselines; tier 3 adds 7 ablations; tier 4 adds 6 sensitivity runs.
**If time is short, running tier 1 alone still gives you the 4.3 paired tests**, which is the step with the highest return on investment.

Time-saving trick: on a single GPU you can **run two processes in parallel** (about 3–4 GB of GPU memory each), splitting the folds, halving the wall-clock time at unchanged cost:

```bash
FOLDS="0 1 2 3 4" TIER=1 bash rerun/run_priority.sh &
FOLDS="5 6 7 8 9" TIER=1 bash rerun/run_priority.sh &
wait
```

---

## 4. Resuming from checkpoints and failure handling

- When `run_experiment.py` sees a `final_metrics.json` it skips that fold, so **after an interruption you can just rerun the same command**.
- A single-fold failure (OOM/exception) affects only that fold; the others continue as usual; the failure log is at `runs/<exp>/<model>/<variant>/fold_k/train.log`.
- To rerun a fold: delete the `final_metrics.json` in that fold's directory, or add `FORCE=1` (this reruns all folds of that tier, so use it with care).
- On OOM, lower `training.num_workers` to 2 and `cache_volumes` to 2.

---

## 5. Run the pipeline through locally for free (optional, but strongly recommended to do once first)

No GPU and no LUNA16 required:

```bash
python tools/make_fake_luna.py --out /tmp/fake_luna --scans 8 --size 64 --num-folds 2
export PYTHONPATH="$PWD/core"
python -m saff_p0.run_experiment --config /tmp/fake_luna/smoke_config.yaml \
    --experiment smoke --models saff_noduledet --variants full --folds 0 --epochs 1 \
    --output-root /tmp/fake_luna/outputs_smoke
python analysis/make_results_package.py --runs-root /tmp/fake_luna/outputs_smoke --out-dir /tmp/fake_luna/package
```

Once it runs through you will see `final_metrics.json`, `history.csv`, `run_times.json`, `paired_tests.csv`, `paper_table*.csv`.
(The synthetic-data results are meaningless in themselves; they only confirm that the code runs.)

---

## 6. How to check the tables after producing the package

Contents of `results_package/`:

| File | Purpose |
|---|---|
| `fold_metrics.csv` | Raw values of the 8 metrics for all conditions × folds |
| `condition_summary.csv` | Mean ± standard deviation for each condition |
| `consistency_report.csv` | **Self-check**: whether CPM equals the mean of the 7 operating points (every row must pass before the paper tables can be used) |
| `paired_tests.csv` | Paired tests of each condition vs the reference condition (SAFF); Holm limited to the three metrics CPM / Sens@0.125 / Sens@1.0 |
| `paper_table4.csv` / `paper_table6.csv` / `paper_table7.csv` | Already sorted in the paper's row order; `MISSING` means that row has not been run yet |
| `figure3_convergence.png` / `figure4_froc.png` / `figure6_ablation.png` | Figures 3/4/6 (can be re-plotted) |
| `README.md` | Number of conditions, number of folds, self-consistency pass rate, protocol markers |

Once you have the measured values, **replace the corresponding tables in the paper as a whole** (including means and standard deviations); do not change only individual numbers;
at the same time, verify section 4.1's "anomaly log" against `history.csv` as required and fill it in truthfully:
`python - <<'PY'` to check whether `train_loss/val_loss/val_cpm` of each fold contain NaN/Inf.

---

## 7. Three things that must be avoided

1. **Do not mix**: the rerun numbers will not be exactly identical to the numbers in the current draft's tables (random seed, `eval_every` and sorting all introduce small differences).
   Either use the new results for a whole table, or state for the whole table that the old results are retained —— never interleave the two sets of numbers.
2. **Do not write `eval_every>1` as "validated every epoch"**: the methodology must state it truthfully.
3. **Do not write p-values without actual measurements**: the statistical tests in 4.3 must come from the output of `paired_tests.csv` (or `paired_tests_cotr3d.py`).
