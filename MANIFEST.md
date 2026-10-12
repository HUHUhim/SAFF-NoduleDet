# File Manifest and Exclusions

Included: source code, Shell launchers, YAML configs, dependency lists, statistics and plotting scripts, README.

Explicitly excluded: LUNA16/CT data, lung/nodule annotations, main-experiment or supplementary-experiment checkpoints, `final_metrics.json`/FROC/history results, temporary caches, `__pycache__`, `.pyc`, AppleDouble `._*` files.

## Rerun toolkit (rerun/, for low-cost reruns with automatic packaging)

```
configs/rerun_4090_final.yaml      Rerun-specific config (eval_every=5, eval_sort_by_scan, cache_volumes=6)
rerun/README.md                  Operations manual (cost, resuming from checkpoints, protocol differences that must be written into the paper)
rerun/run_priority.sh            Tiered runner (TIER=1..4, automatically skips folds already completed)
tools/estimate_cost.py           Estimates hours and cost from the true sample count
tools/make_fake_luna.py          Generates a synthetic mini dataset for local CPU smoke tests
analysis/make_results_package.py Organizes runs/ into Tables 4/6/7 + paired tests + Figures 3/4/6 + a self-consistency report
analysis/paired_tests_cotr3d.py  Paired tests of SAFF vs CoTr3D on three metrics (Holm restricted to the 3 tests)
```

Speed-related changes (they do not change the evaluation protocol, but must be written into the methodology/reply letter): evaluation candidates sorted by scan, validation frequency changed to once every 5 epochs,
early stopping decided by "the number of epochs since the last improvement", Swin-adapted uses the dedicated key `swin_adapted_heads` (default 8).

## Detection-head pooling: all 5 models unified to "center-region pooling" (revised 2026-09-28)

This is a **candidate-level classification** task: the candidate point lies at the exact center of a 96³ patch, while the nodule voxels occupy only
0.02%–0.5% of that patch. Global average pooling over the entire feature map dilutes the central signal across 216 voxels — a point
that the comments on `DetectorHead` in `core/saff_p0/models.py` already made.

In the original implementation, only the detection head of `SAFFNoduleDet` read the config option `model.head_pool` (value `center`),
whereas `UNet3DBaseline` / `ConvNeXt3DBaseline` / `SwinUNETRAdapted` / `CoTr3DBaseline`
**did not read it**, and therefore fell back to the `DetectorHead` default value `"global"`.
This caused the comparison in Table 4 to be contaminated by the architectural difference of "pooling method" rather than being an isomorphic comparison.

This version has fixed it: the 4 baseline classes likewise read `head_pool` and `head_center_frac` from the config
(the modified places are marked with `# [headpool-fix]`, 4 in total). After the fix, for all 5 models
`build_model(name, cfg).head.pool_mode` is **`center`**, and the comparison in Table 4 is performed under the same detection-head structure.

- Verification command: `python tools/apply_headpool_fix.py --check`
- Revert command: `python tools/apply_headpool_fix.py --revert`
- This setting is driven by the **same single config**, so there is no structural privilege enjoyed only by this method;
  the fairness statement in §4.3 of the paper has been updated accordingly.

Before archiving, you may run:

```bash
python -m compileall -q core figures experiments analysis tools
bash -n rerun/run_priority.sh
```
