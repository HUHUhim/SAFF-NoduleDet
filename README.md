# SAFF-NoduleDet Complete Experiment Code Package (Ten Folds)

This directory contains only the code, configs and analysis tools for reproducible experiments; it **does not contain the LUNA16 dataset, main-experiment checkpoints, or any final experimental results**. All tables and figures must be computed from the `final_metrics.json`, `history.csv` and FROC CSVs actually produced on the server; the package contains no demo numbers.

## Server preparation

Python 3.10+, CUDA and a CUDA-matching PyTorch are recommended (the paper records the environment as PyTorch 2.8.0). Please install the CUDA-matching PyTorch yourself first, then run:

```bash
cd <package-root-directory>          # i.e. the folder containing this README
bash tools/prepare_server.sh
```

`requirements.txt` installs base dependencies such as NumPy/Pandas/SciPy/scikit-learn/PyYAML/tqdm/matplotlib/Pillow and does not overwrite an existing PyTorch. After modifying the data paths in `configs/*.yaml`, first run `bash tools/run_smoke.sh`.

## Directory layout

- `core/saff_p0/`: models, data splitting, training, evaluation, FROC/CPM metrics and statistics primitives.
- `configs/`: ten-fold official10 configs; `supplement_server_10fold.yaml` is used for the supplementary experiments.
- `experiments/table4/`: main comparison of SAFF against 3D U-Net, Swin UNETR, ConvNeXt3D and CoTr3D.
- `experiments/table6/`: the eight SAFF ablation settings.
- `experiments/table7/`: parameter sensitivity for `r=3,5,7`, `lambda_dist=0.05,0.10,0.20`, `lambda_smooth=0,0.005,0.010` plus the summary.
- `experiments/negative_sampling/`: evaluation of the 1200, 2400 and full candidate pools with fixed checkpoints, repeated three times.
- `analysis/`: per-fold aggregation, SAFF/CoTr3D paired t-tests and Wilcoxon tests, Holm correction, effect sizes and 95% CI.
- `figures/`: re-plots Figures 3/4/6 from real outputs; the example script for candidate-center and size stratification is used for Figure 5.
- `tools/`: server preparation and a short smoke test.

## Recommended order

```bash
bash experiments/table4/run_table4_baselines.sh
bash experiments/table6/run_table6_ablation.sh
bash experiments/table7/run_table7_sensitivity.sh
bash experiments/negative_sampling/run_negative_sampling.sh
bash analysis/run_statistics.sh
python figures/make_manuscript_figures.py --results-root /path/to/outputs_revision --out-dir /path/to/figures
```

The scripts default to ten folds `0 1 2 3 4 5 6 7 8 9` and 80 epochs, and support overriding via environment variables such as `CONFIG`, `OUT_ROOT`, `FOLDS` and `EPOCHS`. If your existing server main checkpoint directory is named `p0_main`, you can set `CHECKPOINT_EXPERIMENT=p0_main`.

## Result thresholds

Every formal condition must have ten `fold_*/final_metrics.json`. `summarize_table7.py` marks fewer than ten folds as `incomplete` and will not produce a mean±std that can be written directly into the paper; the statistics scripts do not emit inferential p-values for fewer than ten paired folds. Smoke outputs may only be used to check the pipeline and are forbidden from the main text, Tables 4/6/7, or Figures 3/4/6.

## Checkpoint resuming and auditing

The training scripts automatically skip a fold if a `final_metrics.json` already exists; only `--force` overwrites a single fold. Each fold saves the parsed config, data split, model parameter count, training log, history, checkpoint, test metrics and FROC curves. Please keep the logs and the SHA-256 manifest before and after a run, to ensure that the figures and tables come from the same output root.
