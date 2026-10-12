#!/usr/bin/env python3
"""Make the detection heads of the 4 baselines use the `head_pool` from the config as well (unifying them with the detection head of this method).

Problem
-------
The default pooling of `DetectorHead` is `"global"` (global average pooling).
The config `configs/rerun_4090_final.yaml` sets `head_pool: center`,
but only the `SAFFNoduleDet` constructor reads it; the 4 baseline classes **do not read it**:

    saff_noduledet   center   <- reads the config
    cotr3d           global   <- does not read it
    unet3d           global   <- does not read it
    convnext3d       global   <- does not read it
    swin_unetr       global   <- does not read it

Why this makes the comparison unfair
------------------------------------
The task in this paper is **candidate-level classification**: the candidate point sits at the exact center
of a 96^3 patch, while the nodule voxels occupy only 0.02%–0.5% of that patch.
Global average pooling over the entire feature map dilutes the central signal across 216 voxels --
a point that the comment on `DetectorHead` in `models.py` already makes.

If this method uses center-region pooling while the baselines use global average pooling,
the comparison in Table 4 mixes in the architectural difference of "pooling scheme" and is no longer like-for-like.

Fix
---
Only the 4 baseline classes are changed, so that like `SAFFNoduleDet` they read
`head_pool` and `head_center_frac` from cfg. **The SAFF site is left untouched** (it was correct to begin with).

Measured effect: `build_model(...).head.pool_mode` becomes `center` for all 5 models.

Usage
----
    python tools/apply_headpool_fix.py --check     # inspect only
    python tools/apply_headpool_fix.py             # apply the patch (automatic backup)
    python tools/apply_headpool_fix.py --revert    # restore
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "core" / "saff_p0" / "models.py"
BACKUP = TARGET.with_suffix(TARGET.suffix + ".orig_preheadpool")
MARK = "# [headpool-fix]"

REPL = [
    ('        self.head = DetectorHead(base * 8, float(cfg["model"]["dropout"]))',
     '        self.head = DetectorHead(base * 8, float(cfg["model"]["dropout"]),\n'
     '                                  str(cfg["model"].get("head_pool", "global")),\n'
     '                                  float(cfg["model"].get("head_center_frac", 0.25)))  ' + MARK,
     "UNet3DBaseline"),

    ('        self.head = DetectorHead(chans[-1], float(cfg["model"]["dropout"]))',
     '        self.head = DetectorHead(chans[-1], float(cfg["model"]["dropout"]),\n'
     '                                  str(cfg["model"].get("head_pool", "global")),\n'
     '                                  float(cfg["model"].get("head_center_frac", 0.25)))  ' + MARK,
     "ConvNeXt3DBaseline"),

    ('        self.head = DetectorHead(ch, float(mcfg["dropout"]))',
     '        self.head = DetectorHead(ch, float(mcfg["dropout"]),\n'
     '                                  str(mcfg.get("head_pool", "global")),\n'
     '                                  float(mcfg.get("head_center_frac", 0.25)))  ' + MARK,
     "SwinUNETRAdapted"),

    ('        self.head = DetectorHead(int(mcfg["local_channels"]), float(mcfg["dropout"]))',
     '        self.head = DetectorHead(int(mcfg["local_channels"]), float(mcfg["dropout"]),\n'
     '                                  str(mcfg.get("head_pool", "global")),\n'
     '                                  float(mcfg.get("head_center_frac", 0.25)))  ' + MARK,
     "CoTr3DBaseline"),
]


def check() -> int:
    if not TARGET.exists():
        print(f"❌ Cannot find {TARGET}")
        return 1
    src = TARGET.read_text(encoding="utf-8")
    print(f"  Target: {TARGET}")
    print(f"  Status: {'✅ applied' if MARK in src else 'not applied'}")
    for old, _new, name in REPL:
        print(f"    anchor {name:20s} occurs {src.count(old)} time(s) (should be 1 when not applied)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if a.check:
        return check()

    if a.revert:
        if not BACKUP.exists():
            print(f"❌ No backup {BACKUP.name}")
            return 1
        shutil.copy2(BACKUP, TARGET)
        print(f"✅ Restored {TARGET} (from {BACKUP.name})")
        return 0

    src = TARGET.read_text(encoding="utf-8")
    if MARK in src:
        print("ℹ️ The patch is already present, nothing to redo")
        return 0
    for old, _new, name in REPL:
        n = src.count(old)
        if n != 1:
            print(f"❌ anchor {name} occurs {n} time(s) (expected 1); no changes were made")
            return 1

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"  Backed up → {BACKUP.name}")

    out = src
    for old, new, name in REPL:
        out = out.replace(old, new, 1)
        print(f"  ✅ Patched {name}")
    TARGET.write_text(out, encoding="utf-8")
    print(f"✅ Wrote {TARGET}")
    print("   From now on the detection heads of all 5 models read head_pool from the same config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
