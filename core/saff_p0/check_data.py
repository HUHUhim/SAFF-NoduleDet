from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, output_root, save_json
from .data import dataset_overview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    overview = dataset_overview(Path(cfg["data"]["luna_root"]).expanduser().resolve(), cfg)
    out = output_root(cfg) / "dataset_overview.json"
    save_json(out, overview)
    print(f"Saved {out}")
    print(f"mhd_files={overview['mhd_files']}")
    print(f"annotation_rows={overview['annotation_rows']}")
    print(f"candidate_rows={overview['candidate_rows']}")
    for row in overview["folds"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

