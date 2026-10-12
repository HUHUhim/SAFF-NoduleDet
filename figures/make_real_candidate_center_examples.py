#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from saff_p0.mhd import load_mhd_volume, parse_mhd, world_to_voxel_zyx  # noqa: E402


@dataclass(frozen=True)
class SizeGroup:
    key: str
    label: str
    lower_mm: float
    upper_mm: float | None

    def contains(self, diameter_mm: float) -> bool:
        if diameter_mm < self.lower_mm:
            return False
        return self.upper_mm is None or diameter_mm < self.upper_mm

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


SIZE_GROUPS = {
    "small": SizeGroup("small", "Small", 3.0, 6.0),
    "medium": SizeGroup("medium", "Medium", 6.0, 10.0),
    "large": SizeGroup("large", "Large", 10.0, None),
}

PLANES = (
    PlaneSpec("axial", "Axial", fixed_axis=0, row_axis=1, col_axis=2),
    PlaneSpec("coronal", "Coronal", fixed_axis=1, row_axis=0, col_axis=2),
    PlaneSpec("sagittal", "Sagittal", fixed_axis=2, row_axis=0, col_axis=1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render real CT panels from candidate-center prediction rows.")
    parser.add_argument("--examples-csv", required=True, help="CSV with size_group, seriesuid, coordX, coordY, coordZ, probability, fold.")
    parser.add_argument("--luna-root", required=True, help="LUNA16 root containing CSVFILES/annotations.csv.")
    parser.add_argument("--scan-root", required=True, help="Folder containing CT .mhd files, e.g. LUNA16/subset_data.")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated panels.")
    parser.add_argument("--hu-min", type=float, default=-1200.0)
    parser.add_argument("--hu-max", type=float, default=600.0)
    parser.add_argument("--hit-radius-scale", type=float, default=0.5)
    parser.add_argument("--min-hit-radius-mm", type=float, default=3.0)
    parser.add_argument("--crop-mm", type=float, default=48.0)
    parser.add_argument("--panel-px", type=int, default=720)
    parser.add_argument("--scale-bar-mm", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def build_mhd_index(scan_root: Path) -> dict[str, Path]:
    paths = sorted(scan_root.rglob("*.mhd"))
    if not paths:
        raise FileNotFoundError(f"No .mhd files found under {scan_root}")
    return {path.stem: path for path in paths}


def load_annotations(luna_root: Path, mhd_index: dict[str, Path]) -> pd.DataFrame:
    path = luna_root / "CSVFILES" / "annotations.csv"
    if not path.exists():
        path = luna_root / "annotations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find annotations.csv under {luna_root}")
    ann = pd.read_csv(path)
    ann = ann[ann["seriesuid"].isin(mhd_index.keys())].copy()
    ann["seriesuid"] = ann["seriesuid"].astype(str)
    ann["diameter_mm"] = pd.to_numeric(ann["diameter_mm"], errors="coerce")
    return ann.dropna(subset=["diameter_mm"]).reset_index(drop=True)


def normalize_size_key(value: str) -> str:
    key = str(value).strip().lower()
    for prefix in SIZE_GROUPS:
        if key.startswith(prefix):
            return prefix
    raise ValueError(f"Unknown size_group={value!r}; expected Small, Medium, or Large.")


def read_examples(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {"size_group", "seriesuid", "coordX", "coordY", "coordZ", "probability", "fold"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rows = rows.copy()
    rows["seriesuid"] = rows["seriesuid"].astype(str)
    rows["size_key"] = rows["size_group"].map(normalize_size_key)
    for column in ["coordX", "coordY", "coordZ", "probability", "fold"]:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    return rows


def attach_nearest_annotation(
    examples: pd.DataFrame,
    annotations: pd.DataFrame,
    hit_radius_scale: float,
    min_hit_radius_mm: float,
) -> pd.DataFrame:
    output = []
    for row in examples.itertuples(index=False):
        group = annotations[annotations["seriesuid"] == str(row.seriesuid)].copy()
        if group.empty:
            raise ValueError(f"No annotation found for seriesuid={row.seriesuid}")

        size_group = SIZE_GROUPS[str(row.size_key)]
        in_size = group[group["diameter_mm"].map(size_group.contains)].copy()
        if not in_size.empty:
            group = in_size

        candidate_xyz = np.array([float(row.coordX), float(row.coordY), float(row.coordZ)], dtype=np.float64)
        ann_xyz = group[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)
        distances = np.linalg.norm(ann_xyz - candidate_xyz[None, :], axis=1)
        ann_row = group.iloc[int(np.argmin(distances))]
        distance_mm = float(np.min(distances))
        diameter_mm = float(ann_row["diameter_mm"])
        hit_radius_mm = max(float(min_hit_radius_mm), float(hit_radius_scale) * diameter_mm)

        data = row._asdict()
        data.update(
            {
                "size_label": size_group.label,
                "size_range": size_group.range_text,
                "ann_coordX": float(ann_row["coordX"]),
                "ann_coordY": float(ann_row["coordY"]),
                "ann_coordZ": float(ann_row["coordZ"]),
                "diameter_mm": diameter_mm,
                "distance_mm": distance_mm,
                "hit_radius_mm": hit_radius_mm,
                "matched_by_current_rule": bool(distance_mm <= hit_radius_mm),
            }
        )
        output.append(data)
    return pd.DataFrame(output)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def plane_slice(volume: np.ndarray, center_zyx: np.ndarray, plane: PlaneSpec) -> np.ndarray:
    fixed = int(np.clip(round(float(center_zyx[plane.fixed_axis])), 0, volume.shape[plane.fixed_axis] - 1))
    if plane.name == "axial":
        return volume[fixed, :, :]
    if plane.name == "coronal":
        return volume[:, fixed, :]
    if plane.name == "sagittal":
        return volume[:, :, fixed]
    raise ValueError(f"Unknown plane={plane.name}")


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
    return out, np.array([start_r, start_c], dtype=np.float64)


def ct_to_uint8(crop: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    crop = np.clip(crop.astype(np.float32), hu_min, hu_max)
    crop = (crop - hu_min) / max(1e-6, hu_max - hu_min)
    return np.rint(crop * 255.0).astype(np.uint8)


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, length: int = 18, width: int = 4) -> None:
    green = (30, 170, 80)
    shadow = (20, 20, 20)
    draw.line((x - length, y, x + length, y), fill=shadow, width=width + 2)
    draw.line((x, y - length, x, y + length), fill=shadow, width=width + 2)
    draw.line((x - length, y, x + length, y), fill=green, width=width)
    draw.line((x, y - length, x, y + length), fill=green, width=width)
    r = width + 2
    draw.ellipse((x - r, y - r, x + r, y + r), fill=green)


def draw_annotation(
    draw: ImageDraw.ImageDraw,
    ann_rc_panel: np.ndarray,
    diameter_mm: float,
    row_spacing_mm: float,
    col_spacing_mm: float,
    scale_y: float,
    scale_x: float,
) -> None:
    red = (230, 45, 45)
    radius_row = (diameter_mm / 2.0) / max(1e-6, row_spacing_mm) * scale_y
    radius_col = (diameter_mm / 2.0) / max(1e-6, col_spacing_mm) * scale_x
    x = float(ann_rc_panel[1])
    y = float(ann_rc_panel[0])
    draw.ellipse((x - radius_col, y - radius_row, x + radius_col, y + radius_row), outline=red, width=4)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=red)


def draw_scale_bar(draw: ImageDraw.ImageDraw, panel_px: int, crop_mm: float, scale_bar_mm: float) -> None:
    if scale_bar_mm <= 0:
        return
    font = load_font(26)
    bar_px = int(round(scale_bar_mm / max(1e-6, crop_mm) * panel_px))
    bar_px = max(24, min(panel_px // 2, bar_px))
    x0 = 34
    y0 = panel_px - 42
    x1 = x0 + bar_px
    draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=8)
    draw.line((x0, y0, x1, y0), fill=(255, 255, 255), width=5)
    draw.text((x0, y0 - 33), f"{scale_bar_mm:g} mm", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))


def render_panel(
    volume: np.ndarray,
    ann_center_zyx: np.ndarray,
    candidate_center_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    plane: PlaneSpec,
    diameter_mm: float,
    hu_min: float,
    hu_max: float,
    crop_mm: float,
    panel_px: int,
    scale_bar_mm: float,
) -> Image.Image:
    image2d = plane_slice(volume, ann_center_zyx, plane)
    ann_rc = np.array([ann_center_zyx[plane.row_axis], ann_center_zyx[plane.col_axis]], dtype=np.float64)
    candidate_rc = np.array([candidate_center_zyx[plane.row_axis], candidate_center_zyx[plane.col_axis]], dtype=np.float64)
    row_spacing = float(spacing_zyx[plane.row_axis])
    col_spacing = float(spacing_zyx[plane.col_axis])
    crop, crop_origin_rc = crop_physical(image2d, ann_rc, row_spacing, col_spacing, crop_mm, hu_min)

    image = Image.fromarray(ct_to_uint8(crop, hu_min, hu_max)).convert("RGB")
    image = image.resize((panel_px, panel_px), resample=Image.Resampling.LANCZOS)
    scale_y = panel_px / max(1, crop.shape[0])
    scale_x = panel_px / max(1, crop.shape[1])
    ann_rc_panel = (ann_rc - crop_origin_rc) * np.array([scale_y, scale_x], dtype=np.float64)
    candidate_rc_panel = (candidate_rc - crop_origin_rc) * np.array([scale_y, scale_x], dtype=np.float64)

    draw = ImageDraw.Draw(image)
    draw_annotation(draw, ann_rc_panel, diameter_mm, row_spacing, col_spacing, scale_y, scale_x)
    draw_cross(draw, float(candidate_rc_panel[1]), float(candidate_rc_panel[0]))
    draw_scale_bar(draw, panel_px, crop_mm, scale_bar_mm)
    return image


def save_png(image: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(dpi, dpi))


def title_for(row: pd.Series) -> str:
    return (
        f"{row['size_label']} ({row['size_range']}); d={float(row['diameter_mm']):.1f} mm; "
        f"score={float(row['probability']):.2f}; fold={int(row['fold'])}; "
        f"dist={float(row['distance_mm']):.1f} mm"
    )


def compose_row(panels: list[Image.Image], row: pd.Series, panel_px: int) -> Image.Image:
    gap = 24
    top = 92
    bottom = 24
    left = 24
    width = left * 2 + panel_px * 3 + gap * 2
    height = top + panel_px + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 14), title_for(row), fill=(20, 20, 20), font=load_font(32, bold=True))
    for idx, (panel, plane) in enumerate(zip(panels, PLANES)):
        x = left + idx * (panel_px + gap)
        draw.text((x, top - 34), plane.display, fill=(70, 70, 70), font=load_font(28))
        canvas.paste(panel, (x, top))
    return canvas


def compose_final(row_images: list[Image.Image]) -> Image.Image:
    gap = 28
    left = 28
    top = 88
    bottom = 28
    width = max(image.width for image in row_images) + left * 2
    height = top + sum(image.height for image in row_images) + gap * (len(row_images) - 1) + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 18), "Candidate centers on real CT images", fill=(20, 20, 20), font=load_font(38, bold=True))
    draw.text((left, 58), "Red = LUNA16 annotation; Green = model candidate center", fill=(80, 80, 80), font=load_font(25))
    y = top
    for image in row_images:
        canvas.paste(image, (left, y))
        y += image.height + gap
    return canvas


def render_examples(examples: pd.DataFrame, mhd_index: dict[str, Path], args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    row_images = []
    volume_cache: dict[str, tuple[np.ndarray, object]] = {}

    for row in examples.itertuples(index=False):
        data = row._asdict()
        seriesuid = str(data["seriesuid"])
        if seriesuid not in mhd_index:
            raise FileNotFoundError(f"seriesuid={seriesuid} not found in CT index")
        if seriesuid not in volume_cache:
            info = parse_mhd(mhd_index[seriesuid])
            volume_cache[seriesuid] = (load_mhd_volume(info), info)
        volume, info = volume_cache[seriesuid]

        ann_xyz = np.array([data["ann_coordX"], data["ann_coordY"], data["ann_coordZ"]], dtype=np.float64)
        candidate_xyz = np.array([data["coordX"], data["coordY"], data["coordZ"]], dtype=np.float64)
        ann_zyx = world_to_voxel_zyx(ann_xyz, info)
        candidate_zyx = world_to_voxel_zyx(candidate_xyz, info)
        group_dir = out_dir / str(data["size_key"])

        panels = []
        for plane in PLANES:
            panel = render_panel(
                volume,
                ann_zyx,
                candidate_zyx,
                info.spacing_zyx,
                plane,
                float(data["diameter_mm"]),
                float(args.hu_min),
                float(args.hu_max),
                float(args.crop_mm),
                int(args.panel_px),
                float(args.scale_bar_mm),
            )
            panel_path = group_dir / f"{data['size_key']}_{plane.name}_real_candidate.png"
            save_png(panel, panel_path, int(args.dpi))
            data[f"{plane.name}_png"] = str(panel_path)
            panels.append(panel)

        row_image = compose_row(panels, pd.Series(data), int(args.panel_px))
        row_path = group_dir / f"{data['size_key']}_mpr_real_candidate.png"
        save_png(row_image, row_path, int(args.dpi))
        data["mpr_png"] = str(row_path)
        data["ct_mhd_path"] = str(mhd_index[seriesuid])
        row_images.append(row_image)
        rendered.append(data)

    final = compose_final(row_images)
    save_png(final, out_dir / "figure_real_candidate_center_examples.png", int(args.dpi))
    return pd.DataFrame(rendered)


def main() -> int:
    args = parse_args()
    luna_root = Path(args.luna_root).expanduser().resolve()
    scan_root = Path(args.scan_root).expanduser().resolve()
    examples_csv = Path(args.examples_csv).expanduser().resolve()
    mhd_index = build_mhd_index(scan_root)
    annotations = load_annotations(luna_root, mhd_index)
    examples = read_examples(examples_csv)
    examples = attach_nearest_annotation(examples, annotations, float(args.hit_radius_scale), float(args.min_hit_radius_mm))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    examples.to_csv(out_dir / "real_candidate_examples_with_annotation_match.csv", index=False)
    rendered = render_examples(examples, mhd_index, args)
    rendered.to_csv(out_dir / "real_candidate_examples_with_paths.csv", index=False)

    with (out_dir / "caption_en.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "Fig. X. Examples of real CT candidate centers for pulmonary nodules of different sizes. Red circles mark the "
            "LUNA16 reference annotation extent of the nodules; green crosses mark the candidate centers output by the model. "
            "Each panel gives the nodule diameter, the candidate probability, the test fold and the distance from the candidate "
            "center to the annotation center. This figure illustrates the spatial relationship between the candidate-level model "
            "output and the annotated nodules; whether a candidate counts as a hit follows the FROC matching-radius rule in the text.\\n"
        )

    print(f"out_dir={out_dir}")
    print(f"final_figure={out_dir / 'figure_real_candidate_center_examples.png'}")
    print(rendered[["size_label", "seriesuid", "diameter_mm", "probability", "fold", "distance_mm", "hit_radius_mm", "matched_by_current_rule"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
