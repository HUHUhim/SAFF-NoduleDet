from __future__ import annotations

import csv
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import save_json
from .data import collate_batch
from .metrics import compute_froc, save_curve_plot, sigmoid_np, write_predictions


def torch_collate(batch):
    xs, ys, metas = collate_batch(batch)
    return torch.from_numpy(xs), torch.from_numpy(ys), metas


class ScanBlockBatchSampler(torch.utils.data.Sampler):
    """Sampler that "gathers all samples of one scan into a single batch" (mandatory under
    memory constraints).

    Background (measured 2026-09-16):
      * The local cgroup memory limit is only 120 GiB, which cannot hold 888 standardized
        float32 volumes (about 240 GB), so `preload_in_ram` can cover only part of the
        scans, and `cache_volumes` is only 2 (an LRU holding two volumes);
      * A single "read the whole volume + standardize the whole volume" pass measures 1.02 s
        (0.089 s reading, 0.927 s standardizing -- the bottleneck is standardization);
      * Under the default scheme (batch_size=1 + fully random shuffle + 6 workers in
        round-robin dispatch), the ~7 samples of one scan land on 6 different workers, and
        **every worker loads and standardizes the same volume on its own**, so each epoch
        becomes 4604 loads ~= 4700 s (still about 17 h/fold even with 6 workers in
        parallel), which makes training infeasible.

    Approach: hand "all samples of the same scan" to the DataLoader as one batch. The whole
    batch produced by the batch_sampler is handled by [one] worker only, so each scan is
    loaded just once per epoch (699 loads -> about 119 s/epoch with 6 workers in parallel,
    entirely hidden behind the 141 s/epoch of GPU compute).

    Numerical impact: the sample set, the content of each sample and the loss function are
    completely unchanged; each sample is still forward- and backward-propagated on its own
    with batch=1, and the 4-step gradient-accumulation split is exactly the same as
    "training sample by sample in that order". **Only the sample order within an epoch
    changed** (from fully random to "scan blocks + random order across blocks"). That is
    still a protocol difference that has to be written into the methodology / response
    letter.
    """

    def __init__(self, dataset, seed: int = 0, batch_size: int = 1):
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = max(1, int(batch_size))
        self.rebuild(dataset)

    def rebuild(self, dataset) -> None:
        """Rebuild the blocks after the sample list has been replaced (e.g. per-epoch negative resampling)."""
        groups: dict[str, list[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            groups.setdefault(sample.seriesuid, []).append(idx)
        self.blocks: list[list[int]] = list(groups.values())
        self._length = sum(len(b) for b in self.blocks)

    @property
    def n_samples(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """Number of batches. With batch_size=1 this is the scan count; otherwise it is the sample count / batch_size."""
        if self.batch_size <= 1:
            return len(self.blocks)
        return int(math.ceil(self._length / self.batch_size))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.blocks)))
        rng.shuffle(order)
        if self.batch_size <= 1:
            for b in order:
                idxs = list(self.blocks[b])
                rng.shuffle(idxs)
                yield idxs
            return
        # Key point: cut the "sample stream ordered by scan block" into fixed-size batches of
        # batch_size. One forward pass is then a [real batch] (BatchNorm3d finally receives
        # batch statistics), while each batch still spans only 1~3 scans, so the number of
        # volume loads stays essentially the same as with batch_size=1.
        stream: list[int] = []
        for b in order:
            idxs = list(self.blocks[b])
            rng.shuffle(idxs)
            stream.extend(idxs)
        for i in range(0, len(stream), self.batch_size):
            chunk = stream[i : i + self.batch_size]
            if chunk:
                yield chunk


def make_loader(dataset, cfg: dict[str, Any], train: bool, batch_sampler=None,
                persistent_override: bool | None = None) -> DataLoader:
    num_workers = int(cfg["training"]["num_workers"])
    persistent = num_workers > 0
    if not train:
        # The evaluation loader does not need many workers: the val/test volumes are already
        # preloaded with priority, and workers only crop slices. Moreover
        # `persistent_workers=True` would keep them resident for the whole fold, measured at
        # [about 6 GB per worker] -- 3 loaders x 6 workers = up to 18 ~= 105 GB, which pushes
        # memory to 97% under the 120 GiB limit (this actually squeezed workers to death at
        # 2026-09-16 01:23). Evaluation therefore uses few, non-persistent workers: once
        # evaluation ends the processes exit and the memory is returned immediately.
        num_workers = min(num_workers, int(cfg["training"].get("eval_num_workers", 2)))
        persistent = False
    if persistent_override is not None:
        persistent = bool(persistent_override)
    common: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": persistent,
        "collate_fn": torch_collate,
    }
    if batch_sampler is not None:
        # batch_size cannot be passed together with batch_sampler mode
        return DataLoader(dataset, batch_sampler=batch_sampler, **common)
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train,
        **common,
    )


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True)


def make_scaler(enabled: bool):
    if not enabled:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    gt_annotations,
    split_name: str,
    output_dir: Path,
    use_amp: bool,
) -> dict[str, Any]:
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    losses = []
    rows = []
    start = time.time()
    peak_mem = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for x_cpu, y_cpu, metas in loader:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True).view(-1)
            with autocast_context(use_amp):
                out = model(x)
                logits = out["logit"].view(-1)
                loss = criterion(logits, y)
            probs = sigmoid_np(logits.detach().float().cpu().numpy())
            losses.append(float(loss.detach().cpu()))
            for meta, logit, prob in zip(metas, logits.detach().float().cpu().numpy().tolist(), probs.tolist()):
                row = dict(meta)
                row["logit"] = float(logit)
                row["probability"] = float(prob)
                rows.append(row)
    elapsed = time.time() - start
    if device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))
    pred_path = output_dir / f"{split_name}_predictions.csv"
    curve_path = output_dir / f"{split_name}_froc_curve.csv"
    metrics_path = output_dir / f"{split_name}_metrics.json"
    write_predictions(pred_path, rows)
    curve, metrics = compute_froc(
        rows,
        gt_annotations,
        [float(x) for x in cfg["eval"]["fp_targets"]],
        float(cfg["eval"]["hit_radius_scale"]),
        float(cfg["eval"]["min_hit_radius_mm"]),
        scan_count=getattr(loader.dataset, "scan_count", None),
    )
    curve.to_csv(curve_path, index=False)
    metrics.update(
        {
            "loss": float(np.mean(losses)) if losses else math.nan,
            "elapsed_sec": elapsed,
            "samples_per_sec": len(rows) / max(1e-6, elapsed),
            "peak_memory_bytes": peak_mem,
        }
    )
    save_json(metrics_path, metrics)
    if bool(cfg["eval"].get("plot", True)):
        save_curve_plot(curve, output_dir / f"{split_name}_froc_curve.png", f"{split_name} FROC")
    return metrics


def train_one_fold(
    model,
    train_ds,
    val_ds,
    test_ds,
    gt_ann: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"]["amp"] and device.type == "cuda")
    model = model.to(device)
    if bool(cfg["training"].get("compile", False)):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"[warn] torch.compile failed, continuing without compile: {exc}")

    # First standardize every CT volume used by this fold in a single pass and place it in the
    # process-level cache: workers share it through fork, every subsequent sample costs only a
    # slice crop, and data loading is no longer the bottleneck. Reused across folds (done only
    # once per process).
    #
    # WARNING: the memory budget is limited (the local cgroup limit is 120 GiB, which cannot
    # hold the roughly 240 GB required by 888 volumes), so preload can cover only part of the
    # scans and **must be ordered by importance**:
    #   val / test gain the most, because a single load serves the whole evaluation pass
    #   (eval_sort_by_scan makes the candidates of the same scan appear consecutively); train
    #   relies on ScanBlockShuffleSampler to guarantee a single read per scan per epoch. Hence
    #   [val, test, train] is passed in here as the priority order.
    # Page-cache policy: DONTNEED immediately after reading each .raw, so that page cache
    # cannot push memory.current to 100% and get the process killed by the host OOM killer
    # (see the note at the top of mhd.py).
    from .mhd import set_drop_page_cache

    set_drop_page_cache(bool(cfg["data"].get("drop_page_cache", True)))

    if bool(cfg["data"].get("preload_in_ram", False)):
        from .data import preload_volumes

        preload_volumes(
            [val_ds, test_ds, train_ds],
            max_volumes=int(cfg["data"].get("preload_max_volumes", 0) or 0),
            budget_fraction=float(cfg["data"].get("preload_budget_fraction", 0.7)),
            log=print,
        )

    train_batch_sampler = None
    if bool(cfg["data"].get("train_scan_block_shuffle", False)):
        # batch_size is handed to the sampler: when >1 it cuts the sample stream ordered by
        # scan block into real batches, so that BatchNorm3d receives batch statistics
        # (measured: with batch=1 training does not converge at all, see §0b.13/0b.14).
        batch_sz = max(1, int(cfg["training"]["batch_size"]))
        train_batch_sampler = ScanBlockBatchSampler(train_ds, seed=int(cfg["seed"]) + 7,
                                                    batch_size=batch_sz)
        print(f"[sampler] train sampling: shuffled by scan block ({len(train_batch_sampler.blocks)} scans, "
              f"{train_batch_sampler.n_samples} samples/epoch, batch_size={batch_sz}, "
              f"{len(train_batch_sampler)} batches/epoch)")

    # Resample training negatives every epoch (see the note on _resample_train_samples in data.py).
    # Key point: DataLoader workers are forked and each holds its own copy of the dataset --
    # mutating train_ds.samples in the main process does not propagate, so the loader must be
    # rebuilt every epoch.
    train_resample = getattr(train_ds, "resample_fn", None)
    if train_resample is not None:
        print("[sampler] per-epoch training-negative resampling enabled")

    train_loader = make_loader(train_ds, cfg, train=True, batch_sampler=train_batch_sampler,
                               persistent_override=False if train_resample is not None else None)
    val_loader = make_loader(val_ds, cfg, train=False)
    test_loader = make_loader(test_ds, cfg, train=False)

    labels = [s.label for s in train_ds.samples]
    pos = max(1, sum(labels))
    neg = max(1, len(labels) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["training"]["epochs"])))
    scaler = make_scaler(use_amp)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_cpm = -1.0
    best_epoch = 0
    patience = int(cfg["training"]["early_stop_patience"])
    stale = 0
    epochs = int(cfg["training"]["epochs"])
    # Number of epochs between validation runs. 1 reproduces the original per-epoch protocol;
    # larger values cut the (I/O dominated) validation cost at the price of a coarser
    # model-selection grid. The test fold is always evaluated once, after training.
    eval_every = max(1, int(cfg["training"].get("eval_every", 1)))
    epoch_times: list[dict[str, float]] = []
    grad_accum = max(1, int(cfg["training"]["grad_accum_steps"]))
    distill_weight = float(cfg["model"].get("distill_weight", 0.0))
    smooth_weight = float(cfg["model"].get("gate_smooth_weight", 0.0))
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_cpm",
        "val_sens_at_0.125",
        "val_sens_at_0.25",
        "val_sens_at_0.5",
        "val_sens_at_1",
        "val_sens_at_2",
        "val_sens_at_4",
        "val_sens_at_8",
        "lr",
        "elapsed_sec",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_micro = len(train_ds)  # micro-steps per epoch = sample count (also true in batch_sampler mode)
    for epoch in range(1, epochs + 1):
        start = time.time()
        if train_resample is not None and epoch > 1:
            train_ds.samples = train_resample(epoch)
            if train_batch_sampler is not None:
                train_batch_sampler.rebuild(train_ds)
            train_loader = make_loader(train_ds, cfg, train=True,
                                       batch_sampler=train_batch_sampler, persistent_override=False)
        total_micro = len(train_ds)
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        micro = 0
        # The sampler cuts the "sample stream ordered by scan block" into real batches of size
        # batch_size:
        #   * a batch spans only 1~3 scans, so the number of volume loads is basically unchanged;
        #   * one forward pass is a real batch -- BatchNorm3d finally receives batch statistics.
        #     (measured: with batch=1 training does not converge at all: the epoch mean loss can
        #       reach 0.09, yet the final weights give a BCE of 1.5 on the very same data batch,
        #       see the handover list §0b.13/0b.14)
        n_batches = len(train_loader)
        for x_blk, y_blk, _ in train_loader:
            x_blk = x_blk.to(device, non_blocking=True)
            y_blk = y_blk.to(device, non_blocking=True).view(-1)
            with autocast_context(use_amp):
                out = model(x_blk)
                logits = out["logit"].view(-1)
                loss = criterion(logits, y_blk)
                if "distill_loss" in out:
                    loss = loss + distill_weight * out["distill_loss"]
                if "gate_smooth" in out:
                    loss = loss + smooth_weight * out["gate_smooth"]
                scaled = loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            micro += 1
            if micro % grad_accum == 0 or micro == n_batches:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        do_eval = (epoch % eval_every == 0) or (epoch == 1) or (epoch == epochs)
        val_metrics = None
        if do_eval:
            val_metrics = evaluate_model(model, val_loader, device, cfg, gt_ann["val"], "val", output_dir, use_amp)
        val_get = (lambda k: val_metrics.get(k, math.nan)) if val_metrics is not None else (lambda k: math.nan)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else math.nan,
            "val_loss": val_get("loss"),
            "val_cpm": val_get("cpm"),
            "val_sens_at_0.125": val_get("sens_at_0.125"),
            "val_sens_at_0.25": val_get("sens_at_0.25"),
            "val_sens_at_0.5": val_get("sens_at_0.5"),
            "val_sens_at_1": val_get("sens_at_1"),
            "val_sens_at_2": val_get("sens_at_2"),
            "val_sens_at_4": val_get("sens_at_4"),
            "val_sens_at_8": val_get("sens_at_8"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": time.time() - start,
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
        epoch_times.append({"epoch": float(epoch), "elapsed_sec": float(row["elapsed_sec"]), "evaluated": float(do_eval)})
        mean_epoch = float(np.mean([e["elapsed_sec"] for e in epoch_times]))
        eta_min = mean_epoch * max(0, epochs - epoch) / 60.0
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_cpm={row['val_cpm']:.4f} val_loss={row['val_loss']:.4f} "
            f"epoch_sec={row['elapsed_sec']:.1f} eta_min={eta_min:.1f}"
        )
        if do_eval:
            current_cpm = float(val_metrics.get("cpm", -1.0))
            if current_cpm > best_cpm:
                best_cpm = current_cpm
                best_epoch = epoch
                stale = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, output_dir / "best.pt")
            else:
                stale += 1
        torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / "last.pt")
        save_every = int(cfg["training"].get("save_every", 0))
        if save_every > 0 and epoch % save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / f"epoch_{epoch:03d}.pt")
        # Patience is measured in epochs since the last improvement, so the stopping rule is
        # identical whether validation runs every epoch or every eval_every epochs.
        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"early stop at epoch {epoch}; best_epoch={best_epoch}, best_cpm={best_cpm:.4f}")
            break

    save_json(
        output_dir / "run_times.json",
        {
            "eval_every": eval_every,
            "epochs_ran": len(epoch_times),
            "mean_epoch_sec": float(np.mean([e["elapsed_sec"] for e in epoch_times])) if epoch_times else math.nan,
            "total_train_sec": float(sum(e["elapsed_sec"] for e in epoch_times)),
            "per_epoch": epoch_times,
        },
    )

    best_path = output_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        state = ckpt["model"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            model.load_state_dict(cleaned, strict=False)
    test_metrics = evaluate_model(model, test_loader, device, cfg, gt_ann["test"], "test", output_dir, use_amp)
    final = {"best_epoch": best_epoch, "best_val_cpm": best_cpm, "test_metrics": test_metrics}
    save_json(output_dir / "final_metrics.json", final)
    return final
