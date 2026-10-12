# Config notes

All configs use the ten-fold official10 protocol; the data paths must be modified according to the actual mount locations on your server.

## Which one to use to reproduce the paper's results

| Config | Purpose |
|---|---|
| **`rerun_4090_final.yaml`** | ✅ **The authoritative config for reproducing Tables 4/6/7 of the paper** (batch size 16, no gradient accumulation, AdamW 4e-4, cosine annealing, 80 epochs, validate once every 5 epochs) |
| `rerun_4090_fold6_nw2.yaml` / `_nw0.yaml` | **Degraded reproduction configs** for when a single fold crashes in a DataLoader worker (`DataLoader worker exited unexpectedly`): only `num_workers` is changed (2 / 0), everything else is line-for-line identical to the authoritative config |
| `p0_4090.yaml` / `p0_server.yaml` | Early-version configs (batch size 1 + gradient accumulation 4), **for reproducing early experiments only**; they do not represent the paper's final results |
| `supplement_server_10fold.yaml` | Used for the Table 7 parameter-sensitivity experiments |

The `/root/autodl-tmp/SAFF_ws/...` in each `output_root` and the `<OUTPUT_ROOT>` in `p0_local_onetouch.yaml`
are both **placeholder paths**; replace them with the actual mount locations on your own machine (or override them with the scripts' `--out-root` / `OUT_ROOT` environment variable).

Do not hard-code local Windows paths, dataset paths or checkpoint paths into the submission archive.
