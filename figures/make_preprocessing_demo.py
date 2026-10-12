#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from saff_p0.mhd import load_mhd_volume, parse_mhd, world_to_voxel_zyx  # noqa: E402


DEFAULT_LUNA_ROOT = "/Volumes/One Touch/LUNA16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript-ready preprocessing demo images from one LUNA16 case. "
            "The script writes one PNG per preprocessing step plus a composite overview."
        )
    )
    parser.add_argument("--luna-root", default=DEFAULT_LUNA_ROOT, help="LUNA16 root directory.")
    parser.add_argument("--scan-root", default=None, help="Directory containing subset*/CT .mhd files.")
    parser.add_argument("--lung-mask-root", default=None, help="Directory containing lung mask .mhd files.")
    parser.add_argument("--seriesuid", default=None, help="Optional explicit LUNA16 seriesuid to visualize.")
    parser.add_argument("--annotation-index", type=int, default=0, help="Annotation index if seriesuid has multiple nodules.")
    parser.add_argument("--out-dir", default="outputs_preprocessing_demo", help="Output directory.")
    parser.add_argument("--hu-min", type=float, default=-1200.0, help="Lower HU clipping bound.")
    parser.add_argument("--hu-max", type=float, default=600.0, help="Upper HU clipping bound.")
    parser.add_argument(
        "--target-spacing-zyx",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("Z", "Y", "X"),
        help="Target spacing in mm for array axes z y x.",
    )
    parser.add_argument("--patch-size", type=int, default=96, help="Cubic patch size after resampling.")
    parser.add_argument(
        "--max-resampled-voxels",
        type=int,
        default=80_000_000,
        help="Skip automatic case choices whose 1 mm resampled volume would exceed this voxel count.",
    )
    parser.add_argument("--panel-size", type=int, default=512, help="Output size for each square step panel.")
    parser.add_argument("--paper-panel-size", type=int, default=720, help="Panel size for the paper-ready figure.")
    parser.add_argument(
        "--marker-style",
        choices=("circle", "circle_cross", "none"),
        default="circle",
        help="How to mark the annotated nodule center in output panels.",
    )
    parser.add_argument(
        "--marker-color",
        default="255,72,72",
        help="RGB color for the nodule marker, formatted as R,G,B.",
    )
    parser.add_argument("--keep-intermediate-npy", action="store_true", help="Also save selected arrays as .npy files.")
    return parser.parse_args()


def find_csv(luna_root: Path, name: str) -> Path:
    for candidate in (luna_root / "CSVFILES" / name, luna_root / name, luna_root / "csvfiles" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {name} under {luna_root}")


def build_mhd_index(root: Path) -> dict[str, Path]:
    paths = sorted(p for p in root.rglob("*.mhd") if ".ipynb_checkpoints" not in p.parts)
    if not paths:
        raise FileNotFoundError(f"No .mhd files found under {root}")
    return {p.stem: p for p in paths}


def read_annotations(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "seriesuid": row["seriesuid"],
                    "coordX": float(row["coordX"]),
                    "coordY": float(row["coordY"]),
                    "coordZ": float(row["coordZ"]),
                    "diameter_mm": float(row["diameter_mm"]),
                }
            )
    if not rows:
        raise ValueError(f"No annotations found in {path}")
    return rows


def target_shape_zyx(shape_zyx: tuple[int, int, int], spacing_zyx: np.ndarray, target_spacing_zyx: np.ndarray) -> tuple[int, int, int]:
    shape = np.asarray(shape_zyx, dtype=np.float64)
    new_shape = np.maximum(1, np.round(shape * spacing_zyx / target_spacing_zyx).astype(int))
    return tuple(int(v) for v in new_shape.tolist())


def choose_annotation(
    annotations: list[dict[str, object]],
    ct_index: dict[str, Path],
    mask_index: dict[str, Path],
    target_spacing_zyx: np.ndarray,
    max_resampled_voxels: int,
    seriesuid: str | None,
    annotation_index: int,
) -> dict[str, object]:
    if seriesuid:
        matches = [r for r in annotations if r["seriesuid"] == seriesuid and str(r["seriesuid"]) in ct_index]
        if not matches:
            raise ValueError(f"No annotated CT case found for seriesuid={seriesuid}")
        if annotation_index < 0 or annotation_index >= len(matches):
            raise ValueError(f"seriesuid={seriesuid} has {len(matches)} annotations; annotation-index is out of range")
        return matches[annotation_index]

    candidates = sorted(
        (r for r in annotations if str(r["seriesuid"]) in ct_index),
        key=lambda r: float(r["diameter_mm"]),
        reverse=True,
    )
    for row in candidates:
        sid = str(row["seriesuid"])
        if sid not in mask_index:
            continue
        info = parse_mhd(ct_index[sid])
        new_shape = target_shape_zyx(info.shape_zyx, info.spacing_zyx, target_spacing_zyx)
        if int(np.prod(new_shape)) <= max_resampled_voxels:
            return row
    if not candidates:
        raise ValueError("No annotated seriesuid matches the CT index")
    return candidates[0]


def resize_axis_linear(arr: np.ndarray, new_len: int, axis: int) -> np.ndarray:
    old_len = arr.shape[axis]
    if old_len == new_len:
        return arr.astype(np.float32, copy=False)
    moved = np.moveaxis(arr.astype(np.float32, copy=False), axis, 0)
    pos = np.linspace(0.0, old_len - 1.0, new_len, dtype=np.float32)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, old_len - 1)
    w = (pos - lo).reshape((new_len,) + (1,) * (moved.ndim - 1))
    out = moved[lo] * (1.0 - w) + moved[hi] * w
    return np.moveaxis(out.astype(np.float32, copy=False), 0, axis)


def resize_axis_nearest(arr: np.ndarray, new_len: int, axis: int) -> np.ndarray:
    old_len = arr.shape[axis]
    if old_len == new_len:
        return arr
    moved = np.moveaxis(arr, axis, 0)
    idx = np.rint(np.linspace(0.0, old_len - 1.0, new_len)).astype(np.int64)
    out = moved[idx]
    return np.moveaxis(out, 0, axis)


def resample_volume(arr: np.ndarray, new_shape: tuple[int, int, int], order: int) -> np.ndarray:
    out = arr
    for axis, new_len in enumerate(new_shape):
        if order == 0:
            out = resize_axis_nearest(out, int(new_len), axis)
        else:
            out = resize_axis_linear(out, int(new_len), axis)
    return out


def standardize_lung_regions(volume: np.ndarray, lung_mask: np.ndarray | None, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(volume.astype(np.float32), hu_min, hu_max)
    out = np.empty_like(clipped, dtype=np.float32)
    lung_bool = lung_mask > 0 if lung_mask is not None else None
    boundaries = np.linspace(0, clipped.shape[0], 6, dtype=int)
    for i in range(5):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        slab = clipped[lo:hi]
        valid = (slab > hu_min + 1.0) & (slab < hu_max - 1.0)
        if lung_bool is not None:
            valid &= lung_bool[lo:hi]
        vals = slab[valid] if valid.any() else slab.reshape(-1)
        mean = float(vals.mean())
        std = float(vals.std() + 1e-6)
        out[lo:hi] = (slab - mean) / std
    return out


def crop_patch(volume: np.ndarray, center_zyx: np.ndarray, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    center = np.rint(center_zyx).astype(int)
    starts = center - half
    ends = starts + patch_size
    src_slices = []
    dst_slices = []
    for start, end, size in zip(starts, ends, volume.shape):
        src_start = max(0, int(start))
        src_end = min(int(size), int(end))
        dst_start = max(0, -int(start))
        dst_end = dst_start + max(0, src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
    patch[tuple(dst_slices)] = volume[tuple(src_slices)]
    return patch


def axial_slice(arr: np.ndarray, z: float) -> np.ndarray:
    zi = int(np.clip(round(float(z)), 0, arr.shape[0] - 1))
    return arr[zi]


def to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip((arr.astype(np.float32) - vmin) / max(1e-6, vmax - vmin), 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)


def robust_uint8(arr: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    vmin, vmax = np.percentile(finite, [lower, upper])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return to_uint8(arr, float(vmin), float(vmax))


def colorize_mask(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    m = mask.astype(bool)
    base[m] = base[m] * (1.0 - alpha) + overlay[m] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def morphology_overlay(gray: np.ndarray, hu_slice: np.ndarray, lung_slice: np.ndarray | None) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    valid = np.ones(hu_slice.shape, dtype=bool) if lung_slice is None else lung_slice > 0
    channels = [
        ((hu_slice >= -700) & (hu_slice < -300) & valid, (70, 170, 255), "GGO"),
        ((hu_slice >= -300) & (hu_slice < 200) & valid, (255, 180, 70), "solid"),
        ((hu_slice >= 200) & valid, (255, 80, 80), "calcified"),
    ]
    for mask, color, _ in channels:
        overlay = np.zeros_like(base)
        overlay[..., 0] = color[0]
        overlay[..., 1] = color[1]
        overlay[..., 2] = color[2]
        base[mask] = base[mask] * 0.55 + overlay[mask] * 0.45
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_square(img: Image.Image, size: int, upscale_small: bool = True) -> Image.Image:
    img = img.convert("RGB")
    resample = Image.Resampling.NEAREST if upscale_small else Image.Resampling.LANCZOS
    if upscale_small and (img.width < size * 0.75 and img.height < size * 0.75):
        scale = min(size / img.width, size / img.height)
        img = img.resize((max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))), resample)
    else:
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (18, 20, 24))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_marker(
    img: Image.Image,
    center_yx: tuple[float, float],
    radius_yx: tuple[float, float],
    original_shape_yx: tuple[int, int],
    color: tuple[int, int, int] = (255, 72, 72),
    style: str = "circle",
) -> Image.Image:
    if style == "none":
        return img.convert("RGB").copy()
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    scale_x = out.width / float(original_shape_yx[1])
    scale_y = out.height / float(original_shape_yx[0])
    cy, cx = center_yx
    ry, rx = radius_yx
    box = [
        (cx - rx) * scale_x,
        (cy - ry) * scale_y,
        (cx + rx) * scale_x,
        (cy + ry) * scale_y,
    ]
    line_width = max(3, out.width // 150)
    draw.ellipse(box, outline=color, width=line_width)
    if style == "circle_cross":
        cross = max(5, out.width // 80)
        draw.line([(cx * scale_x - cross, cy * scale_y), (cx * scale_x + cross, cy * scale_y)], fill=color, width=max(2, line_width // 2))
        draw.line([(cx * scale_x, cy * scale_y - cross), (cx * scale_x, cy * scale_y + cross)], fill=color, width=max(2, line_width // 2))
    return out


def add_title(img: Image.Image, title: str, subtitle: str | None = None) -> Image.Image:
    title_font = font(24)
    subtitle_font = font(16)
    pad = 18
    header_h = 70 if subtitle else 48
    out = Image.new("RGB", (img.width, img.height + header_h), (245, 247, 250))
    draw = ImageDraw.Draw(out)
    draw.text((pad, 10), title, fill=(18, 24, 32), font=title_font)
    if subtitle:
        draw.text((pad, 42), subtitle, fill=(82, 92, 105), font=subtitle_font)
    out.paste(img, (0, header_h))
    return out


def save_panel(
    path: Path,
    img: Image.Image,
    title: str,
    subtitle: str | None,
    panel_size: int,
) -> Path:
    panel = fit_square(img, panel_size)
    panel = add_title(panel, title, subtitle)
    panel.save(path, quality=95)
    return path


def add_paper_label(img: Image.Image, label: str, subtitle: str | None = None) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    label_font = font(36)
    subtitle_font = font(24)
    pad = 16
    label_w = int(draw.textbbox((0, 0), label, font=label_font)[2]) + 2 * pad
    box_h = 56 if not subtitle else 92
    draw.rounded_rectangle([0, 0, max(label_w, 320), box_h], radius=0, fill=(245, 247, 250))
    draw.text((pad, 8), label, fill=(18, 24, 32), font=label_font)
    if subtitle:
        draw.text((pad, 56), subtitle, fill=(80, 88, 100), font=subtitle_font)
    return out


def save_paper_panel(
    path: Path,
    img: Image.Image,
    label: str,
    subtitle: str | None,
    panel_size: int,
) -> Path:
    panel = fit_square(img, panel_size)
    panel = add_paper_label(panel, label, subtitle)
    panel.save(path, quality=95)
    return path


def make_info_panel(path: Path, metadata: dict[str, object], panel_size: int) -> Path:
    body_font = font(18)
    mono_font = font(14)
    img = Image.new("RGB", (panel_size, panel_size), (18, 20, 24))
    draw = ImageDraw.Draw(img)
    lines = [
        "Case metadata",
        f"seriesuid: {metadata['seriesuid']}",
        f"original shape zyx: {tuple(metadata['original_shape_zyx'])}",
        f"original spacing zyx: {tuple(round(float(x), 3) for x in metadata['original_spacing_zyx_mm'])} mm",
        f"resampled shape zyx: {tuple(metadata['resampled_shape_zyx'])}",
        f"target spacing zyx: {tuple(float(x) for x in metadata['target_spacing_zyx_mm'])} mm",
        f"HU clip: [{metadata['hu_clip'][0]:.0f}, {metadata['hu_clip'][1]:.0f}]",
        "patch: 96 x 96 x 96 voxels",
        "mask: seg-lungs binary lung mask",
        "fallback: 5 axial lung regions",
    ]
    y = 20
    for i, line in enumerate(lines):
        fnt = body_font if i == 0 else mono_font
        fill = (245, 247, 250) if i == 0 else (207, 216, 226)
        wrapped = textwrap.wrap(line, width=42) or [line]
        for part in wrapped:
            draw.text((20, y), part, fill=fill, font=fnt)
            y += 29 if i == 0 else 24
        if i == 0:
            y += 8
    return save_panel(path, img, "8. Run metadata", "parameters recorded for reproducibility", panel_size)


def make_paper_info_panel(path: Path, metadata: dict[str, object], panel_size: int) -> Path:
    body_font = font(32)
    small_font = font(27)
    img = Image.new("RGB", (panel_size, panel_size), (18, 20, 24))
    draw = ImageDraw.Draw(img)
    lines = [
        ("Preprocessing parameters", body_font, (245, 247, 250)),
        (f"HU window: [{metadata['hu_clip'][0]:.0f}, {metadata['hu_clip'][1]:.0f}] HU", small_font, (214, 222, 232)),
        (f"Original spacing: {tuple(round(float(x), 3) for x in metadata['original_spacing_zyx_mm'])} mm", small_font, (214, 222, 232)),
        (f"Target spacing: {tuple(float(x) for x in metadata['target_spacing_zyx_mm'])} mm", small_font, (214, 222, 232)),
        (f"Patch size: 96 x 96 x 96 voxels", small_font, (214, 222, 232)),
        ("Mask: binary lung mask", small_font, (214, 222, 232)),
        ("Normalization: 5 axial lung regions", small_font, (214, 222, 232)),
    ]
    y = 104
    for text, fnt, fill in lines:
        wrapped = textwrap.wrap(text, width=34) or [text]
        for part in wrapped:
            draw.text((42, y), part, fill=fill, font=fnt)
            y += 48 if fnt is body_font else 42
        y += 16
    panel = add_paper_label(img, "(h) Parameters", None)
    panel.save(path, quality=95)
    return path


def make_overview(step_paths: list[Path], out_path: Path, panel_size: int) -> None:
    panels = [Image.open(p).convert("RGB") for p in step_paths]
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    gap = 18
    title_h = 56
    w = cols * panel_size + (cols + 1) * gap
    h = title_h + rows * (panel_size + 70) + (rows + 1) * gap
    canvas = Image.new("RGB", (w, h), (238, 241, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 14), "LUNA16 preprocessing demonstration", fill=(18, 24, 32), font=font(30))
    for i, panel in enumerate(panels):
        row = i // cols
        col = i % cols
        x = gap + col * (panel_size + gap)
        y = title_h + gap + row * (panel_size + 70 + gap)
        canvas.paste(panel, (x, y))
    canvas.save(out_path, quality=95)


def make_paper_figure(panel_paths: list[Path], out_path: Path, panel_size: int) -> None:
    panels = [Image.open(p).convert("RGB") for p in panel_paths]
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    gap = 14
    w = cols * panel_size + (cols + 1) * gap
    h = rows * panel_size + (rows + 1) * gap
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    for i, panel in enumerate(panels):
        row = i // cols
        col = i % cols
        x = gap + col * (panel_size + gap)
        y = gap + row * (panel_size + gap)
        canvas.paste(panel, (x, y))
    canvas.save(out_path, quality=95)


def write_caption(path: Path, seriesuid: str) -> None:
    text = f"""Fig. X. Example of the image preprocessing workflow. Using one annotated nodule in LUNA16 case {seriesuid} as an example, the panels show in order the original CT axial slice, HU clipping, lung-region mask overlay, 1.0 mm isotropic resampling for visualization, lung-region-based regional standardization, the 96 x 96 x 96 3D patch cropped around the nodule center, and the extent of the lung-region mask inside the patch. The red circle marks the annotated nodule location. The bottom-right panel lists the input size, voxel spacing and preprocessing parameters of this case. This example illustrates the preprocessing implementation and quality control; model evaluation is still based on the complete cross-validation results."""
    path.write_text(text, encoding="utf-8")


def write_paper_caption(path: Path) -> None:
    text = """Fig. X. Example preprocessing workflow for a LUNA16 CT scan. The pipeline includes (a) raw CT loading, (b) HU clipping, (c) lung-mask overlay, (d) isotropic resampling for visualization, (e) lung-region standardization, (f) nodule-centered patch extraction, (g) the lung-mask extent inside the patch, and (h) the preprocessing parameters used in this case. The red circle indicates the annotated nodule location. This figure illustrates preprocessing implementation and quality control; quantitative evaluation was performed using the full cross-validation protocol."""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    luna_root = Path(args.luna_root).expanduser().resolve()
    scan_root = Path(args.scan_root).expanduser().resolve() if args.scan_root else luna_root / "subset_data"
    lung_mask_root = Path(args.lung_mask_root).expanduser().resolve() if args.lung_mask_root else luna_root / "seg-lungs-LUNA16"
    out_root = Path(args.out_dir).expanduser().resolve()
    target_spacing_zyx = np.asarray(args.target_spacing_zyx, dtype=np.float32)
    marker_color = tuple(int(x.strip()) for x in args.marker_color.split(","))
    if len(marker_color) != 3 or any(v < 0 or v > 255 for v in marker_color):
        raise ValueError("--marker-color must be formatted as R,G,B with values from 0 to 255")

    annotations = read_annotations(find_csv(luna_root, "annotations.csv"))
    ct_index = build_mhd_index(scan_root)
    mask_index = build_mhd_index(lung_mask_root) if lung_mask_root.exists() else {}
    row = choose_annotation(
        annotations,
        ct_index,
        mask_index,
        target_spacing_zyx=target_spacing_zyx,
        max_resampled_voxels=args.max_resampled_voxels,
        seriesuid=args.seriesuid,
        annotation_index=args.annotation_index,
    )
    sid = str(row["seriesuid"])
    case_out = out_root / sid
    case_out.mkdir(parents=True, exist_ok=True)

    info = parse_mhd(ct_index[sid])
    volume = load_mhd_volume(info).astype(np.float32, copy=False)
    mask = None
    if sid in mask_index:
        mask = load_mhd_volume(parse_mhd(mask_index[sid]))
        if tuple(mask.shape) != tuple(volume.shape):
            print(f"Warning: lung mask shape {mask.shape} does not match CT shape {volume.shape}; ignoring mask", file=sys.stderr)
            mask = None

    world_xyz = np.array([row["coordX"], row["coordY"], row["coordZ"]], dtype=np.float64)
    center_zyx = world_to_voxel_zyx(world_xyz, info)
    diameter_mm = float(row["diameter_mm"])
    z_raw = float(np.clip(round(center_zyx[0]), 0, volume.shape[0] - 1))
    center_yx_raw = (float(center_zyx[1]), float(center_zyx[2]))
    radius_yx_raw = (
        max(3.0, diameter_mm / (2.0 * float(info.spacing_zyx[1]))),
        max(3.0, diameter_mm / (2.0 * float(info.spacing_zyx[2]))),
    )

    clipped = np.clip(volume, args.hu_min, args.hu_max)
    new_shape = target_shape_zyx(volume.shape, info.spacing_zyx, target_spacing_zyx)
    resampled = resample_volume(clipped, new_shape, order=1)
    resampled_mask = resample_volume(mask.astype(np.uint8), new_shape, order=0) if mask is not None else None
    center_zyx_resampled = center_zyx * (info.spacing_zyx / target_spacing_zyx)
    z_resampled = float(np.clip(round(center_zyx_resampled[0]), 0, resampled.shape[0] - 1))
    center_yx_resampled = (float(center_zyx_resampled[1]), float(center_zyx_resampled[2]))
    radius_yx_resampled = (
        max(3.0, diameter_mm / (2.0 * float(target_spacing_zyx[1]))),
        max(3.0, diameter_mm / (2.0 * float(target_spacing_zyx[2]))),
    )

    standardized = standardize_lung_regions(resampled, resampled_mask, args.hu_min, args.hu_max)
    patch = crop_patch(standardized, center_zyx_resampled, args.patch_size)
    patch_hu = crop_patch(resampled, center_zyx_resampled, args.patch_size)
    patch_mask = crop_patch(resampled_mask.astype(np.float32), center_zyx_resampled, args.patch_size) if resampled_mask is not None else None
    patch_center = (args.patch_size / 2.0, args.patch_size / 2.0)
    patch_radius = (max(3.0, diameter_mm / 2.0), max(3.0, diameter_mm / 2.0))

    raw_gray = to_uint8(axial_slice(volume, z_raw), -1000, 400)
    raw_img = Image.fromarray(raw_gray, mode="L").convert("RGB")
    raw_img = draw_marker(raw_img, center_yx_raw, radius_yx_raw, raw_gray.shape, color=marker_color, style=args.marker_style)

    clipped_gray = to_uint8(axial_slice(clipped, z_raw), args.hu_min, args.hu_max)
    clipped_img = Image.fromarray(clipped_gray, mode="L").convert("RGB")
    clipped_img = draw_marker(clipped_img, center_yx_raw, radius_yx_raw, clipped_gray.shape, color=marker_color, style=args.marker_style)

    mask_slice = axial_slice(mask, z_raw) > 0 if mask is not None else np.zeros(volume.shape[1:], dtype=bool)
    mask_img = colorize_mask(clipped_gray, mask_slice, color=(30, 180, 255), alpha=0.35)
    mask_img = draw_marker(mask_img, center_yx_raw, radius_yx_raw, clipped_gray.shape, color=marker_color, style=args.marker_style)

    resampled_gray = to_uint8(axial_slice(resampled, z_resampled), args.hu_min, args.hu_max)
    resampled_img = Image.fromarray(resampled_gray, mode="L").convert("RGB")
    resampled_img = draw_marker(resampled_img, center_yx_resampled, radius_yx_resampled, resampled_gray.shape, color=marker_color, style=args.marker_style)

    standardized_gray = robust_uint8(axial_slice(standardized, z_resampled), 1, 99)
    standardized_img = Image.fromarray(standardized_gray, mode="L").convert("RGB")
    standardized_img = draw_marker(standardized_img, center_yx_resampled, radius_yx_resampled, standardized_gray.shape, color=marker_color, style=args.marker_style)

    patch_gray = robust_uint8(axial_slice(patch, args.patch_size // 2), 1, 99)
    patch_img = Image.fromarray(patch_gray, mode="L").convert("RGB")
    patch_img = draw_marker(patch_img, patch_center, patch_radius, patch_gray.shape, color=marker_color, style=args.marker_style)

    patch_mask_gray = to_uint8(axial_slice(patch_hu, args.patch_size // 2), args.hu_min, args.hu_max)
    patch_mask_slice = axial_slice(patch_mask, args.patch_size // 2) > 0 if patch_mask is not None else np.zeros(patch_mask_gray.shape, dtype=bool)
    patch_mask_img = colorize_mask(patch_mask_gray, patch_mask_slice, color=(30, 180, 255), alpha=0.35)
    patch_mask_img = draw_marker(patch_mask_img, patch_center, patch_radius, patch_mask_gray.shape, color=marker_color, style=args.marker_style)

    subtitle_raw = f"spacing zyx={tuple(round(float(x), 3) for x in info.spacing_zyx)} mm"
    subtitle_resampled = f"target spacing zyx={tuple(float(x) for x in target_spacing_zyx)} mm"
    metadata = {
        "seriesuid": sid,
        "ct_path": str(ct_index[sid]),
        "lung_mask_path": str(mask_index[sid]) if sid in mask_index else None,
        "annotation": row,
        "original_shape_zyx": list(volume.shape),
        "original_spacing_zyx_mm": [float(x) for x in info.spacing_zyx],
        "target_spacing_zyx_mm": [float(x) for x in target_spacing_zyx],
        "resampled_shape_zyx": list(resampled.shape),
        "center_voxel_zyx_original": [float(x) for x in center_zyx],
        "center_voxel_zyx_resampled": [float(x) for x in center_zyx_resampled],
        "hu_clip": [float(args.hu_min), float(args.hu_max)],
        "normalization": "lung-mask regional standardization fallback; no five-lobe masks are present by default",
        "marker_style": args.marker_style,
        "marker_color_rgb": list(marker_color),
    }
    step_paths = [
        save_panel(case_out / "01_raw_ct.png", raw_img, "1. Raw CT slice", subtitle_raw, args.panel_size),
        save_panel(case_out / "02_hu_clipped.png", clipped_img, "2. HU clipping", f"[{args.hu_min:.0f}, {args.hu_max:.0f}] HU", args.panel_size),
        save_panel(case_out / "03_lung_mask_overlay.png", mask_img, "3. Lung mask overlay", "seg-lungs-LUNA16 binary mask", args.panel_size),
        save_panel(case_out / "04_resampled_1mm.png", resampled_img, "4. Isotropic resampling", subtitle_resampled, args.panel_size),
        save_panel(case_out / "05_lung_region_standardized.png", standardized_img, "5. Lung-region standardization", "5 axial regions when lobe masks are unavailable", args.panel_size),
        save_panel(case_out / "06_nodule_patch_96.png", patch_img, "6. Nodule-centered patch", f"{args.patch_size} x {args.patch_size} x {args.patch_size} voxels", args.panel_size),
        save_panel(case_out / "07_patch_lung_mask.png", patch_mask_img, "7. Patch lung mask", "binary lung-mask extent inside patch", args.panel_size),
        make_info_panel(case_out / "08_run_metadata.png", metadata, args.panel_size),
    ]
    make_overview(step_paths, case_out / "09_preprocessing_overview.png", args.panel_size)
    paper_dir = case_out / "paper_panels"
    paper_dir.mkdir(exist_ok=True)
    paper_paths = [
        save_paper_panel(paper_dir / "a_raw_ct.png", raw_img, "(a) Raw CT", None, args.paper_panel_size),
        save_paper_panel(paper_dir / "b_hu_clipping.png", clipped_img, "(b) HU clipping", f"[{args.hu_min:.0f}, {args.hu_max:.0f}] HU", args.paper_panel_size),
        save_paper_panel(paper_dir / "c_lung_mask.png", mask_img, "(c) Lung mask", None, args.paper_panel_size),
        save_paper_panel(paper_dir / "d_resampling.png", resampled_img, "(d) Resampling", "1.0 mm isotropic", args.paper_panel_size),
        save_paper_panel(paper_dir / "e_standardization.png", standardized_img, "(e) Standardization", "lung-region statistics", args.paper_panel_size),
        save_paper_panel(paper_dir / "f_patch.png", patch_img, "(f) Patch", "96 x 96 x 96 voxels", args.paper_panel_size),
        save_paper_panel(paper_dir / "g_patch_lung_mask.png", patch_mask_img, "(g) Patch mask", "binary lung region", args.paper_panel_size),
        make_paper_info_panel(paper_dir / "h_parameters.png", metadata, args.paper_panel_size),
    ]
    make_paper_figure(paper_paths, case_out / "10_preprocessing_paper_figure.png", args.paper_panel_size)
    write_paper_caption(case_out / "caption_en_paper.txt")
    metadata["outputs"] = [str(p) for p in step_paths] + [
        str(case_out / "09_preprocessing_overview.png"),
        str(case_out / "10_preprocessing_paper_figure.png"),
        str(case_out / "caption_en_paper.txt"),
    ]
    (case_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_caption(case_out / "caption_en.txt", sid)

    if args.keep_intermediate_npy:
        np.save(case_out / "raw_volume.npy", volume)
        np.save(case_out / "resampled_hu.npy", resampled)
        np.save(case_out / "standardized.npy", standardized)
        np.save(case_out / "patch.npy", patch)

    print(
        textwrap.dedent(
            f"""
            Wrote preprocessing demo images for:
              seriesuid: {sid}
              output:    {case_out}

            Main manuscript figure:
              {case_out / '10_preprocessing_paper_figure.png'}

            Demo overview:
              {case_out / '09_preprocessing_overview.png'}

            Step images:
              {chr(10).join('  ' + str(p) for p in step_paths)}
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
