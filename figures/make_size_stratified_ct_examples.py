#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "core"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from saff_p0.config import load_config  # noqa: E402
from saff_p0.data import build_mhd_index, load_annotations  # noqa: E402
from saff_p0.mhd import load_mhd_volume, parse_mhd, world_to_voxel_zyx  # noqa: E402


@dataclass(frozen=True)
class SizeGroup:
    name: str
    label: str
    lower_mm: float
    upper_mm: float | None

    def contains(self, diameter_mm: float) -> bool:
        if diameter_mm < self.lower_mm:
            return False
        if self.upper_mm is None:
            return True
        return diameter_mm < self.upper_mm

    @property
    def range_text(self) -> str:
        if self.upper_mm is None:
            return f"d >= {self.lower_mm:g} mm"
        return f"{self.lower_mm:g} <= d < {self.upper_mm:g} mm"


@dataclass(frozen=True)
class PlaneSpec:
    name: str
    display: str
    fixed_axis: int
    row_axis: int
    col_axis: int


SIZE_GROUPS = (
    SizeGroup("small", "Small", 3.0, 6.0),
    SizeGroup("medium", "Medium", 6.0, 10.0),
    SizeGroup("large", "Large", 10.0, None),
)

PLANES = (
    PlaneSpec("axial", "Axial", fixed_axis=0, row_axis=1, col_axis=2),
    PlaneSpec("coronal", "Coronal", fixed_axis=1, row_axis=0, col_axis=2),
    PlaneSpec("sagittal", "Sagittal", fixed_axis=2, row_axis=0, col_axis=1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create true CT subfigures for size-stratified candidate-level "
            "nodule recognition examples from saved test_predictions.csv files."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "p0_server.yaml"),
        help="Experiment config. CLI paths override values in this file.",
    )
    parser.add_argument("--luna-root", default=None, help="LUNA16 root containing CSVFILES/.")
    parser.add_argument("--scan-root", default=None, help="Folder containing CT .mhd files; use subset_data if present.")
    parser.add_argument("--output-root", default=None, help="outputs_p0 root used to infer prediction paths.")
    parser.add_argument(
        "--pred-glob",
        action="append",
        default=[],
        help=(
            "Glob for saved test_predictions.csv. Can be repeated. If omitted, "
            "the script searches output_root/runs/<experiment>/<model>/<variant>/fold_*/test_predictions.csv."
        ),
    )
    parser.add_argument("--experiment", default=None, help="Experiment folder under outputs_p0/runs/.")
    parser.add_argument("--model", default="saff_noduledet", help="Model folder name under runs/<experiment>/.")
    parser.add_argument("--variant", default="full", help="Variant folder name, usually full.")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs_p0_demo" / "figures" / "size_stratified_ct_examples"),
        help="Directory for generated subfigures and metadata.",
    )
    parser.add_argument("--hu-min", type=float, default=None, help="Lower CT window, default from config.")
    parser.add_argument("--hu-max", type=float, default=None, help="Upper CT window, default from config.")
    parser.add_argument("--hit-radius-scale", type=float, default=None, help="TP match radius scale, default from config.")
    parser.add_argument("--min-hit-radius-mm", type=float, default=None, help="Minimum TP match radius, default from config.")
    parser.add_argument(
        "--allow-nearest-annotation",
        action="store_true",
        help=(
            "For manual visualization only: if a candidate is outside the configured hit radius, "
            "use the nearest unmatched annotation in the same scan so the provided candidate can still be drawn."
        ),
    )
    parser.add_argument("--min-probability", type=float, default=0.0, help="Ignore predictions below this score.")
    parser.add_argument(
        "--selection",
        choices=("highest-prob", "closest-to-bin-center"),
        default="highest-prob",
        help="How to choose one true positive from each size group.",
    )
    parser.add_argument("--crop-mm", type=float, default=48.0, help="Physical field of view for each CT panel.")
    parser.add_argument("--panel-px", type=int, default=720, help="Pixel size of each square output panel.")
    parser.add_argument("--scale-bar-mm", type=float, default=10.0, help="Scale bar length. Set 0 to hide.")
    parser.add_argument(
        "--draw-reference-circle",
        action="store_true",
        help="Optionally draw the LUNA16 annotation diameter as an amber reference circle.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI metadata saved into PNG files.")
    return parser.parse_args()


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def load_base_config(config_path: str | Path | None) -> dict:
    path = resolve_path(config_path)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return load_config(path)


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def infer_prediction_files(args: argparse.Namespace, cfg: dict) -> list[Path]:
    patterns: list[str] = list(args.pred_glob)
    if not patterns:
        output_root = resolve_path(args.output_root or cfg_get(cfg, "data", "output_root"))
        if output_root is None:
            raise ValueError("Provide --pred-glob or set data.output_root in the config.")
        experiment = args.experiment if args.experiment else "*"
        inferred = output_root / "runs" / experiment / args.model / args.variant / "fold_*" / "test_predictions.csv"
        patterns.append(str(inferred))

    files: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(str(resolve_glob_pattern(pattern)))
        files.extend(Path(p).resolve() for p in expanded)

    unique = sorted(set(files))
    if not unique:
        joined = "\n  ".join(patterns)
        raise FileNotFoundError(f"No prediction files matched:\n  {joined}")
    return unique


def resolve_glob_pattern(pattern: str | Path) -> str:
    pattern_text = str(pattern)
    if pattern_text.startswith("~"):
        pattern_text = str(Path(pattern_text).expanduser())
    if Path(pattern_text).is_absolute():
        return pattern_text
    return str((Path.cwd() / pattern_text).resolve())


def fold_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.fullmatch(r"fold_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def load_predictions(paths: list[Path], min_probability: float) -> pd.DataFrame:
    rows = []
    required = {"seriesuid", "coordX", "coordY", "coordZ", "probability"}
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = frame.copy()
        frame["seriesuid"] = frame["seriesuid"].astype(str)
        for col in ["coordX", "coordY", "coordZ", "probability"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["coordX", "coordY", "coordZ", "probability"])
        frame = frame[frame["probability"] >= float(min_probability)].copy()
        frame["fold"] = fold_from_path(path)
        frame["prediction_file"] = str(path)
        rows.append(frame)

    if not rows:
        raise ValueError("No prediction rows were loaded.")
    pred = pd.concat(rows, ignore_index=True)
    if pred.empty:
        raise ValueError("All predictions were filtered out by --min-probability.")
    return pred


def match_true_positive_candidates(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    hit_radius_scale: float,
    min_hit_radius_mm: float,
    allow_nearest_annotation: bool = False,
) -> pd.DataFrame:
    annotations = annotations.reset_index(drop=True).copy()
    annotations["ann_id"] = np.arange(len(annotations), dtype=int)
    annotations["seriesuid"] = annotations["seriesuid"].astype(str)
    gt_by_series = {sid: group.reset_index(drop=True) for sid, group in annotations.groupby("seriesuid")}

    matched_ann_ids: set[int] = set()
    matches: list[dict] = []
    pred_sorted = predictions.sort_values("probability", ascending=False).reset_index(drop=True)

    for row in pred_sorted.itertuples(index=False):
        seriesuid = str(row.seriesuid)
        group = gt_by_series.get(seriesuid)
        if group is None or group.empty:
            continue

        candidate_xyz = np.array([float(row.coordX), float(row.coordY), float(row.coordZ)], dtype=np.float64)
        ann_xyz = group[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)
        dists = np.linalg.norm(ann_xyz - candidate_xyz[None, :], axis=1)

        ordered_local_indices = np.argsort(dists)
        best_unmatched: tuple[int, int, float, float] | None = None
        for local_idx in ordered_local_indices:
            ann_row = group.iloc[int(local_idx)]
            ann_id = int(ann_row["ann_id"])
            if ann_id in matched_ann_ids:
                continue

            diameter_mm = float(ann_row["diameter_mm"])
            hit_radius_mm = max(float(min_hit_radius_mm), float(hit_radius_scale) * diameter_mm)
            distance_mm = float(dists[int(local_idx)])
            if best_unmatched is None:
                best_unmatched = (int(local_idx), ann_id, diameter_mm, hit_radius_mm)
            if distance_mm > hit_radius_mm:
                continue

            matched_ann_ids.add(ann_id)
            matches.append(
                {
                    "seriesuid": seriesuid,
                    "fold": getattr(row, "fold", None),
                    "probability": float(row.probability),
                    "candidate_coordX": float(row.coordX),
                    "candidate_coordY": float(row.coordY),
                    "candidate_coordZ": float(row.coordZ),
                    "ann_id": ann_id,
                    "ann_coordX": float(ann_row["coordX"]),
                    "ann_coordY": float(ann_row["coordY"]),
                    "ann_coordZ": float(ann_row["coordZ"]),
                    "diameter_mm": diameter_mm,
                    "distance_mm": distance_mm,
                    "hit_radius_mm": hit_radius_mm,
                    "matched_within_radius": True,
                    "prediction_file": getattr(row, "prediction_file", ""),
                }
            )
            break
        else:
            if allow_nearest_annotation and best_unmatched is not None:
                local_idx, ann_id, diameter_mm, hit_radius_mm = best_unmatched
                ann_row = group.iloc[int(local_idx)]
                distance_mm = float(dists[int(local_idx)])
                matched_ann_ids.add(ann_id)
                matches.append(
                    {
                        "seriesuid": seriesuid,
                        "fold": getattr(row, "fold", None),
                        "probability": float(row.probability),
                        "candidate_coordX": float(row.coordX),
                        "candidate_coordY": float(row.coordY),
                        "candidate_coordZ": float(row.coordZ),
                        "ann_id": ann_id,
                        "ann_coordX": float(ann_row["coordX"]),
                        "ann_coordY": float(ann_row["coordY"]),
                        "ann_coordZ": float(ann_row["coordZ"]),
                        "diameter_mm": diameter_mm,
                        "distance_mm": distance_mm,
                        "hit_radius_mm": hit_radius_mm,
                        "matched_within_radius": False,
                        "prediction_file": getattr(row, "prediction_file", ""),
                    }
                )

    if not matches:
        raise ValueError("No true-positive candidate matches were found.")
    return pd.DataFrame(matches)


def assign_size_groups(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    group_names = []
    group_labels = []
    group_ranges = []
    for diameter in matches["diameter_mm"].to_numpy(dtype=float):
        selected = None
        for group in SIZE_GROUPS:
            if group.contains(float(diameter)):
                selected = group
                break
        group_names.append(selected.name if selected else "")
        group_labels.append(selected.label if selected else "")
        group_ranges.append(selected.range_text if selected else "")
    matches["size_group"] = group_names
    matches["size_label"] = group_labels
    matches["size_range"] = group_ranges
    return matches[matches["size_group"] != ""].reset_index(drop=True)


def select_examples(matches: pd.DataFrame, selection: str) -> pd.DataFrame:
    selected = []
    missing = []
    for group in SIZE_GROUPS:
        group_rows = matches[matches["size_group"] == group.name].copy()
        if group_rows.empty:
            missing.append(f"{group.label} ({group.range_text})")
            continue

        if selection == "closest-to-bin-center" and group.upper_mm is not None:
            center = (group.lower_mm + group.upper_mm) / 2.0
            group_rows["diameter_distance_to_bin_center"] = (group_rows["diameter_mm"] - center).abs()
            group_rows = group_rows.sort_values(
                ["diameter_distance_to_bin_center", "probability", "distance_mm"],
                ascending=[True, False, True],
            )
        else:
            group_rows = group_rows.sort_values(["probability", "distance_mm"], ascending=[False, True])
        selected.append(group_rows.iloc[0])

    if missing:
        available = matches.groupby("size_group")["diameter_mm"].agg(["count", "min", "max"]).reset_index()
        raise ValueError(f"Missing size groups: {missing}\nAvailable TP groups:\n{available.to_string(index=False)}")
    return pd.DataFrame(selected).reset_index(drop=True)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def plane_slice(volume: np.ndarray, center_zyx: np.ndarray, plane: PlaneSpec) -> np.ndarray:
    fixed = int(np.clip(round(float(center_zyx[plane.fixed_axis])), 0, volume.shape[plane.fixed_axis] - 1))
    if plane.name == "axial":
        return volume[fixed, :, :]
    if plane.name == "coronal":
        return volume[:, fixed, :]
    if plane.name == "sagittal":
        return volume[:, :, fixed]
    raise ValueError(f"Unknown plane: {plane.name}")


def crop_physical(
    image: np.ndarray,
    center_rc: np.ndarray,
    row_spacing_mm: float,
    col_spacing_mm: float,
    crop_mm: float,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    half_rows = max(8, int(round((crop_mm / 2.0) / max(float(row_spacing_mm), 1e-6))))
    half_cols = max(8, int(round((crop_mm / 2.0) / max(float(col_spacing_mm), 1e-6))))
    center_int = np.rint(center_rc).astype(int)
    start_r = int(center_int[0] - half_rows)
    start_c = int(center_int[1] - half_cols)
    end_r = start_r + 2 * half_rows
    end_c = start_c + 2 * half_cols

    out = np.full((2 * half_rows, 2 * half_cols), fill_value, dtype=np.float32)
    src_r0 = max(0, start_r)
    src_c0 = max(0, start_c)
    src_r1 = min(int(image.shape[0]), end_r)
    src_c1 = min(int(image.shape[1]), end_c)
    dst_r0 = src_r0 - start_r
    dst_c0 = src_c0 - start_c
    dst_r1 = dst_r0 + max(0, src_r1 - src_r0)
    dst_c1 = dst_c0 + max(0, src_c1 - src_c0)
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = image[src_r0:src_r1, src_c0:src_c1].astype(np.float32)

    crop_origin_rc = np.array([start_r, start_c], dtype=np.float64)
    return out, crop_origin_rc


def ct_to_uint8(crop: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    crop = np.clip(crop.astype(np.float32), float(hu_min), float(hu_max))
    crop = (crop - float(hu_min)) / max(1e-6, float(hu_max) - float(hu_min))
    return np.rint(crop * 255.0).astype(np.uint8)


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, length: int, width: int) -> None:
    color = (30, 170, 80)
    shadow = (20, 20, 20)
    draw.line((x - length, y, x + length, y), fill=shadow, width=width + 2)
    draw.line((x, y - length, x, y + length), fill=shadow, width=width + 2)
    draw.line((x - length, y, x + length, y), fill=color, width=width)
    draw.line((x, y - length, x, y + length), fill=color, width=width)
    r = max(3, width + 2)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_scale_bar(draw: ImageDraw.ImageDraw, panel_px: int, crop_mm: float, scale_bar_mm: float, font: ImageFont.ImageFont) -> None:
    if scale_bar_mm <= 0:
        return
    bar_px = int(round(float(scale_bar_mm) / max(1e-6, float(crop_mm)) * panel_px))
    bar_px = max(24, min(panel_px // 2, bar_px))
    x0 = 34
    y0 = panel_px - 42
    x1 = x0 + bar_px
    draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=8)
    draw.line((x0, y0, x1, y0), fill=(255, 255, 255), width=5)
    text = f"{scale_bar_mm:g} mm"
    draw.text((x0, y0 - 33), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))


def draw_reference_ellipse(
    draw: ImageDraw.ImageDraw,
    ann_rc_panel: np.ndarray,
    diameter_mm: float,
    row_spacing_mm: float,
    col_spacing_mm: float,
    scale_y: float,
    scale_x: float,
) -> None:
    radius_row = (diameter_mm / 2.0) / max(1e-6, float(row_spacing_mm)) * scale_y
    radius_col = (diameter_mm / 2.0) / max(1e-6, float(col_spacing_mm)) * scale_x
    x = float(ann_rc_panel[1])
    y = float(ann_rc_panel[0])
    box = (x - radius_col, y - radius_row, x + radius_col, y + radius_row)
    draw.ellipse(box, outline=(230, 45, 45), width=4)


def render_panel(
    volume: np.ndarray,
    center_zyx: np.ndarray,
    ann_center_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    plane: PlaneSpec,
    diameter_mm: float,
    hu_min: float,
    hu_max: float,
    crop_mm: float,
    panel_px: int,
    scale_bar_mm: float,
    draw_reference_circle: bool,
) -> Image.Image:
    image2d = plane_slice(volume, center_zyx, plane)
    candidate_rc = np.array([center_zyx[plane.row_axis], center_zyx[plane.col_axis]], dtype=np.float64)
    annotation_rc = np.array([ann_center_zyx[plane.row_axis], ann_center_zyx[plane.col_axis]], dtype=np.float64)
    row_spacing = float(spacing_zyx[plane.row_axis])
    col_spacing = float(spacing_zyx[plane.col_axis])
    crop, crop_origin_rc = crop_physical(image2d, candidate_rc, row_spacing, col_spacing, crop_mm, fill_value=hu_min)

    gray = ct_to_uint8(crop, hu_min, hu_max)
    image = Image.fromarray(gray, mode="L").convert("RGB")
    image = image.resize((panel_px, panel_px), resample=Image.Resampling.LANCZOS)

    scale_y = panel_px / max(1, crop.shape[0])
    scale_x = panel_px / max(1, crop.shape[1])
    candidate_rc_panel = (candidate_rc - crop_origin_rc) * np.array([scale_y, scale_x], dtype=np.float64)
    annotation_rc_panel = (annotation_rc - crop_origin_rc) * np.array([scale_y, scale_x], dtype=np.float64)

    draw = ImageDraw.Draw(image)
    if draw_reference_circle:
        draw_reference_ellipse(
            draw,
            annotation_rc_panel,
            diameter_mm,
            row_spacing,
            col_spacing,
            scale_y,
            scale_x,
        )
    draw_cross(draw, float(candidate_rc_panel[1]), float(candidate_rc_panel[0]), length=18, width=4)
    draw_scale_bar(draw, panel_px, crop_mm, scale_bar_mm, load_font(26))
    return image


def save_png(image: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(dpi, dpi))


def example_title(row: pd.Series) -> str:
    return (
        f"{row['size_label']} ({row['size_range']}); "
        f"d={float(row['diameter_mm']):.1f} mm; "
        f"score={float(row['probability']):.3f}; "
        f"fold={int(row['fold']) if not pd.isna(row['fold']) else 'NA'}"
    )


def compose_mpr_row(panels: list[Image.Image], row: pd.Series, panel_px: int) -> Image.Image:
    gap = 24
    top = 88
    bottom = 24
    left = 24
    title_font = load_font(34)
    label_font = load_font(28)
    width = left * 2 + panel_px * 3 + gap * 2
    height = top + panel_px + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 14), example_title(row), fill=(20, 20, 20), font=title_font)
    for idx, (panel, plane) in enumerate(zip(panels, PLANES)):
        x = left + idx * (panel_px + gap)
        y = top
        draw.text((x, top - 34), plane.display, fill=(70, 70, 70), font=label_font)
        canvas.paste(panel, (x, y))
    return canvas


def compose_final_figure(row_images: list[Image.Image], selected: pd.DataFrame, panel_px: int) -> Image.Image:
    gap = 28
    left = 28
    top = 72
    bottom = 24
    title_font = load_font(38)
    width = max(img.width for img in row_images) + left * 2
    height = top + sum(img.height for img in row_images) + gap * (len(row_images) - 1) + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (left, 18),
        "Size-stratified candidate-level recognition examples on real CT",
        fill=(20, 20, 20),
        font=title_font,
    )
    y = top
    for image in row_images:
        canvas.paste(image, (left, y))
        y += image.height + gap
    return canvas


def render_examples(
    selected: pd.DataFrame,
    mhd_index: dict[str, Path],
    out_dir: Path,
    hu_min: float,
    hu_max: float,
    crop_mm: float,
    panel_px: int,
    scale_bar_mm: float,
    draw_reference_circle: bool,
    dpi: int,
) -> pd.DataFrame:
    rendered_rows = []
    row_images = []
    volume_cache: dict[str, tuple[np.ndarray, object]] = {}

    for row in selected.itertuples(index=False):
        row_data = row._asdict()
        seriesuid = str(row_data["seriesuid"])
        if seriesuid not in mhd_index:
            raise FileNotFoundError(f"seriesuid={seriesuid} was not found in the CT .mhd index.")

        if seriesuid not in volume_cache:
            info = parse_mhd(mhd_index[seriesuid])
            volume_cache[seriesuid] = (load_mhd_volume(info), info)
        volume, info = volume_cache[seriesuid]

        candidate_xyz = np.array(
            [row_data["candidate_coordX"], row_data["candidate_coordY"], row_data["candidate_coordZ"]],
            dtype=np.float64,
        )
        annotation_xyz = np.array(
            [row_data["ann_coordX"], row_data["ann_coordY"], row_data["ann_coordZ"]],
            dtype=np.float64,
        )
        center_zyx = world_to_voxel_zyx(candidate_xyz, info)
        ann_center_zyx = world_to_voxel_zyx(annotation_xyz, info)

        group_dir = out_dir / str(row_data["size_group"])
        panels = []
        panel_paths = {}
        for plane in PLANES:
            panel = render_panel(
                volume,
                center_zyx,
                ann_center_zyx,
                info.spacing_zyx,
                plane,
                float(row_data["diameter_mm"]),
                hu_min,
                hu_max,
                crop_mm,
                panel_px,
                scale_bar_mm,
                draw_reference_circle,
            )
            panel_path = group_dir / f"{row_data['size_group']}_{plane.name}.png"
            save_png(panel, panel_path, dpi)
            panels.append(panel)
            panel_paths[f"{plane.name}_png"] = str(panel_path)

        row_image = compose_mpr_row(panels, pd.Series(row_data), panel_px)
        row_path = group_dir / f"{row_data['size_group']}_mpr_three_views.png"
        save_png(row_image, row_path, dpi)
        row_images.append(row_image)

        row_data.update(panel_paths)
        row_data["mpr_png"] = str(row_path)
        row_data["ct_mhd_path"] = str(mhd_index[seriesuid])
        rendered_rows.append(row_data)

    final = compose_final_figure(row_images, selected, panel_px)
    save_png(final, out_dir / "figure_size_stratified_candidate_examples.png", dpi)
    return pd.DataFrame(rendered_rows)


def write_caption(path: Path, draw_reference_circle: bool) -> None:
    reference = (
        "The red circle shows the LUNA16 annotated diameter extent and is a reference annotation only."
        if draw_reference_circle
        else "No predicted or annotated boxes are drawn, so that candidate-level classification results are not presented as bounding-box regression results."
    )
    caption = (
        "Fig. X. Real-CT examples of size-stratified candidate-level nodule recognition. Each row shows the axial, "
        "coronal and sagittal slices of a small, "
        "medium and large nodule respectively. The green crosses mark the candidate centers that "
        "SAFF-NoduleDet output on the test folds and that hit an annotated nodule; "
        f"{reference} "
        "Below each example are the nodule diameter, the model prediction score and the test fold it belongs to; all images come from "
        "the held-out test cases corresponding to test_predictions.csv."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_base_config(args.config)

    luna_root = resolve_path(args.luna_root or cfg_get(cfg, "data", "luna_root"))
    if luna_root is None:
        raise ValueError("Provide --luna-root or set data.luna_root in the config.")
    scan_root = resolve_path(args.scan_root or cfg_get(cfg, "data", "scan_root"))
    out_dir = resolve_path(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    hu_min = float(args.hu_min if args.hu_min is not None else cfg_get(cfg, "data", "hu_min", -1200))
    hu_max = float(args.hu_max if args.hu_max is not None else cfg_get(cfg, "data", "hu_max", 600))
    hit_radius_scale = float(
        args.hit_radius_scale if args.hit_radius_scale is not None else cfg_get(cfg, "eval", "hit_radius_scale", 0.5)
    )
    min_hit_radius_mm = float(
        args.min_hit_radius_mm if args.min_hit_radius_mm is not None else cfg_get(cfg, "eval", "min_hit_radius_mm", 3.0)
    )

    prediction_files = infer_prediction_files(args, cfg)
    mhd_index = build_mhd_index(luna_root, scan_root)
    annotations = load_annotations(luna_root, mhd_index)
    predictions = load_predictions(prediction_files, args.min_probability)

    matches = match_true_positive_candidates(
        predictions,
        annotations,
        hit_radius_scale,
        min_hit_radius_mm,
        allow_nearest_annotation=bool(args.allow_nearest_annotation),
    )
    matches = assign_size_groups(matches)
    matches.to_csv(out_dir / "all_true_positive_matches_by_size.csv", index=False)

    selected = select_examples(matches, args.selection)
    selected.to_csv(out_dir / "selected_size_stratified_examples.csv", index=False)

    rendered = render_examples(
        selected,
        mhd_index,
        out_dir,
        hu_min,
        hu_max,
        float(args.crop_mm),
        int(args.panel_px),
        float(args.scale_bar_mm),
        bool(args.draw_reference_circle),
        int(args.dpi),
    )
    rendered.to_csv(out_dir / "selected_size_stratified_examples_with_paths.csv", index=False)
    write_caption(out_dir / "caption_en.txt", bool(args.draw_reference_circle))

    print(f"prediction_files={len(prediction_files)}")
    print(f"true_positive_matches={len(matches)}")
    print(f"out_dir={out_dir}")
    print(f"final_figure={out_dir / 'figure_size_stratified_candidate_examples.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
