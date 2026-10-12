#!/usr/bin/env python3
"""Make training support [resuming from the last completed epoch after a crash] -- patches core/saff_p0/train_eval.py.

Problem (why this is needed)
----------------------------
Every call to `train_one_fold`:
  * opens `history.csv` with `"w"` -> **truncates** the already completed epoch records
  * starts training from `epoch = 1`
  * only stores `{"model", "epoch"}` in `last.pt` -> **no optimizer state**

So in an 80 epoch x 10 fold long run, **any single crash amounts to retraining the whole fold from scratch**.
The 10-fold experiments of this project paid a real price for this: when one fold was killed by OOM,
all 41 completed epochs were discarded and that fold restarted from epoch 1 (about 7.5 hours wasted).

What the patch does
-------------------
1. At the end of every epoch it writes `resume_state.pt` **atomically**:
   `{epoch, model, optimizer, scheduler, scaler, best_cpm, best_epoch, stale}`
2. On startup, if that file exists and `epoch < epochs`:
   * restore the model / optimizer / scheduler / scaler state and the early-stop counter
   * restore the completed epoch records verbatim from `history.csv` (including per-epoch duration, for the ETA)
   * `history.csv` switches to **append** mode and the epoch loop continues from `resume_epoch + 1`
3. Once training has run to completion, delete `resume_state.pt`

Why this is **numerically equivalent** rather than approximate
-------------------------------------------------------------
* The randomness of the data augmentation comes from `np.random.default_rng(seed + index * 9973)` --
  one independent rng per sample, and the state is **not advanced across samples**, so a sample's content
  is determined solely by `(seed, index)`;
* optimizer momentum, the cosine learning-rate schedule, the AMP scaler and the early-stop counter are all explicitly saved to disk;
* therefore the new epoch sequence produced by resuming is numerically identical to "running on without ever having been interrupted".

It does **not change** the model definition, hyperparameters, data splits, preprocessing, losses or evaluation scripts.

Usage
----
    python tools/apply_resume_patch.py --check     # inspect only, does not modify files
    python tools/apply_resume_patch.py             # apply the patch (automatic backup)
    python tools/apply_resume_patch.py --revert    # restore from the backup

After patching, just use `python -m saff_p0.run_experiment ...` as usual;
if a fold is interrupted, **re-running the very same command** will automatically continue from the next epoch.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "core" / "saff_p0" / "train_eval.py"
MARK = "# [resume-patch]"
BACKUP_SUFFIX = ".orig_preresume"

OLD_BLOCK = '''    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_micro = len(train_ds)  # micro-steps per epoch = sample count (also true in batch_sampler mode)
    for epoch in range(1, epochs + 1):
'''

NEW_BLOCK = '''    # ===== [resume-patch] restore the training state from resume_state.pt (resume after a crash) =====
    # Seven items are stored: {model, optimizer, scheduler, scaler, best_cpm, best_epoch, stale},
    # so "restart after a crash" and "never crashed" are numerically identical (the RNG depends only on seed/epoch/index).
    _resume_path = output_dir / "resume_state.pt"
    _resume_epoch = 0
    if _resume_path.exists():
        try:
            _st = torch.load(_resume_path, map_location=device, weights_only=False)
            _resume_epoch = int(_st.get("epoch", 0))
            if _resume_epoch >= epochs:
                print(f"[resume] snapshot is already at epoch {_resume_epoch} (>= {epochs}), ignoring it")
                _resume_epoch = 0
            else:
                model.load_state_dict(_st["model"])
                optimizer.load_state_dict(_st["optimizer"])
                scheduler.load_state_dict(_st["scheduler"])
                if scaler is not None and _st.get("scaler") is not None:
                    scaler.load_state_dict(_st["scaler"])
                best_cpm = float(_st.get("best_cpm", -1.0))
                best_epoch = int(_st.get("best_epoch", 0))
                stale = int(_st.get("stale", 0))
                if history_path.exists():
                    with history_path.open(newline="", encoding="utf-8") as f:
                        for _row in csv.DictReader(f):
                            try:
                                epoch_times.append({
                                    "epoch": float(_row["epoch"]),
                                    "elapsed_sec": float(_row["elapsed_sec"]),
                                    "evaluated": float(
                                        _row.get("val_cpm") not in (None, "", "nan")
                                    ),
                                })
                            except (KeyError, TypeError, ValueError):
                                pass
                print(f"[resume] resuming from epoch {_resume_epoch} ({len(epoch_times)} history records already completed, "
                      f"best_cpm={best_cpm:.4f}@{best_epoch})")
        except Exception as _exc:
            print(f"[resume] failed to read {_resume_path} ({type(_exc).__name__}: {_exc}) -> training from scratch")
            _resume_epoch = 0
            best_cpm = -1.0
            best_epoch = 0
            stale = 0
            epoch_times.clear()

    _hdr = ["epoch", "train_loss", "val_loss", "val_cpm", "val_sens_at_0.125", "val_sens_at_0.25",
            "val_sens_at_0.5", "val_sens_at_1", "val_sens_at_2", "val_sens_at_4", "val_sens_at_8",
            "lr", "elapsed_sec"]
    _need_hdr = (
        not history_path.exists()
        or history_path.stat().st_size == 0
        or not history_path.read_text(encoding="utf-8", errors="replace").startswith("epoch,")
    )
    with history_path.open("a" if (_resume_epoch > 0 and not _need_hdr) else "w",
                           newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_hdr)
        if _resume_epoch == 0 or _need_hdr:
            writer.writeheader()
    # ===== [/resume-patch] =====

    def _save_resume_snapshot(ep: int) -> None:
        """[resume-patch] Persist the complete state once per epoch (write .tmp first, then replace, to avoid half-written files)."""
        _tmp = output_dir / "resume_state.pt.tmp"
        torch.save({
            "epoch": int(ep),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_cpm": float(best_cpm),
            "best_epoch": int(best_epoch),
            "stale": int(stale),
        }, _tmp)
        _tmp.replace(_resume_path)

    total_micro = len(train_ds)  # micro-steps per epoch = number of samples (also holds in batch_sampler mode)
    for epoch in range(1, epochs + 1):
        if epoch <= _resume_epoch:
            continue   # [resume-patch] skip epochs that are already completed (their state has been restored)
'''

OLD_SAVE = '''        save_every = int(cfg["training"].get("save_every", 0))
        if save_every > 0 and epoch % save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / f"epoch_{epoch:03d}.pt")
'''

NEW_SAVE = '''        save_every = int(cfg["training"].get("save_every", 0))
        if save_every > 0 and epoch % save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / f"epoch_{epoch:03d}.pt")
        _save_resume_snapshot(epoch)   # [resume-patch]
'''

OLD_TAIL = '''    save_json(
        output_dir / "run_times.json",
        {
            "eval_every": eval_every,
            "epochs_ran": len(epoch_times),
'''

NEW_TAIL = '''    if _resume_path.exists():
        _resume_path.unlink()   # [resume-patch] ran to completion, drop the snapshot so it cannot be misread
    save_json(
        output_dir / "run_times.json",
        {
            "eval_every": eval_every,
            "epochs_ran": len(epoch_times),
'''


def check() -> int:
    if not TARGET.exists():
        print(f"❌ Cannot find {TARGET}")
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARK in src:
        print("✅ Patch [applied]")
        return 0
    print("ℹ️ Patch [not applied]")
    for old, name in ((OLD_BLOCK, "history initialization"), (OLD_SAVE, "end-of-epoch save"), (OLD_TAIL, "final cleanup")):
        print(f"   anchor [{name}] occurs {src.count(old)} time(s) (expected 1)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="restore from the .orig_preresume backup")
    ap.add_argument("--check", action="store_true", help="inspect only")
    a = ap.parse_args()

    if a.check:
        return check()

    bak = TARGET.with_suffix(TARGET.suffix + BACKUP_SUFFIX)
    if a.revert:
        if not bak.exists():
            print(f"❌ No backup {bak.name}")
            return 1
        shutil.copy2(bak, TARGET)
        print(f"✅ Restored {TARGET} (from {bak.name})")
        return 0

    if not TARGET.exists():
        print(f"❌ Cannot find {TARGET}")
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARK in src:
        print("ℹ️ The patch is already present, nothing to re-apply")
        return 0

    for old, name in ((OLD_BLOCK, "history initialization"), (OLD_SAVE, "end-of-epoch save"), (OLD_TAIL, "final cleanup")):
        n = src.count(old)
        if n != 1:
            print(f"❌ anchor [{name}] occurs {n} time(s) (expected 1); the source may have changed, so no changes were made")
            return 1

    if not bak.exists():
        shutil.copy2(TARGET, bak)
        print(f"  Backed up → {bak.name}")

    out = src
    for old, new, name in ((OLD_BLOCK, NEW_BLOCK, "history initialization -> resumable restore"),
                           (OLD_SAVE, NEW_SAVE, "end-of-epoch snapshot save"),
                           (OLD_TAIL, NEW_TAIL, "end-of-training snapshot cleanup")):
        out = out.replace(old, new, 1)
        print(f"  ✅ {name}")
    TARGET.write_text(out, encoding="utf-8")
    print(f"✅ Wrote {TARGET}")
    print("   Now run run_experiment as usual; after an interruption, re-running the same command resumes from the next epoch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
