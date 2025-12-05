#!/usr/bin/env python3
"""
tile_fullsize_images.py
-----------------------
Tile full-size images with proper YOLO(-Seg) label transformation.

Behavior:
- If INPUT label line is bbox-only (5 values): OUTPUT is bbox-only (5 values).
- If INPUT label line includes polygon coords: OUTPUT includes bbox + polygon coords (YOLO-Seg style).

Defaults:
- Tile size: 1280x1280
- Overlap: 30% (0.30)
- Robust polygon clipping and coordinate transformation
- All outputs clamped to [0,1] to avoid out-of-range labels
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from shapely.geometry import Polygon, box as shp_box
from utils.tiling_shared import load_tiling_config, make_tiles  # reuse helper

# Load shared config
CFG = load_tiling_config()

# --- helpers ----------------------------------------------------------
def _clamp01(v: float) -> float:
    """Clamp value to [0,1] range."""
    return max(0.0, min(1.0, float(v)))

def parse_yolo_seg_line(line: str):
    """
    Parse a YOLO(-Seg) label line; if no polygon, derive one from bbox (normalized).
    Returns: (cls, cx, cy, w, h, poly_norm, had_poly)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    coords = list(map(float, parts[5:])) if len(parts) > 5 else []
    if coords:
        poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        had_poly = True
    else:
        # Derive rectangle polygon from bbox (normalized)
        x0, y0 = cx - w / 2.0, cy - h / 2.0
        x1, y1 = cx + w / 2.0, cy + h / 2.0
        poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        had_poly = False
    return cls, cx, cy, w, h, poly, had_poly

def clip_poly_to_tile(poly_abs_xy, tx, ty, ts):
    """Clip polygon (absolute pixels) to tile bounds using robust polygon intersection."""
    if not poly_abs_xy or len(poly_abs_xy) < 3:
        return []
    try:
        pg = Polygon(poly_abs_xy).buffer(0)
        tile = shp_box(float(tx), float(ty), float(tx + ts), float(ty + ts))
        inter = pg.intersection(tile)
        if inter.is_empty:
            return []
        if inter.geom_type == "MultiPolygon":
            inter = max(list(inter.geoms), key=lambda g: g.area)
        if inter.geom_type != "Polygon" or inter.is_empty:
            return []
        pts = np.asarray(inter.exterior.coords, dtype=np.float32)[:-1]
        return [(float(x), float(y)) for x, y in pts]
    except Exception:
        return []

def bbox_from_poly(poly_abs_xy):
    """Axis-aligned bbox (absolute px) from polygon points."""
    if not poly_abs_xy:
        return 0.0, 0.0, 0.0, 0.0
    xs, ys = zip(*poly_abs_xy)
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

def transform_annotation_for_tile(line, tile_x, tile_y, tile_size, img_w, img_h):
    """
    Transform a YOLO(-Seg) annotation line for a specific tile (to tile-normalized coords).
    If input was bbox-only (5 fields), output is bbox-only; if input had polygon, output keeps polygon.
    """
    parsed = parse_yolo_seg_line(line)
    if parsed is None:
        return None
    cls, cx, cy, w, h, poly_n, had_poly = parsed

    # Convert normalized polygon → absolute pixels
    abs_poly = [(x * img_w, y * img_h) for x, y in poly_n]

    # Robust clip to tile bounds
    clipped_poly = clip_poly_to_tile(abs_poly, tile_x, tile_y, tile_size)
    if len(clipped_poly) < 3:
        return None

    # Re-compute bbox from clipped polygon (absolute), then normalize
    xmin, ymin, xmax, ymax = bbox_from_poly(clipped_poly)
    cx_abs = (xmin + xmax) * 0.5
    cy_abs = (ymin + ymax) * 0.5
    w_abs = (xmax - xmin)
    h_abs = (ymax - ymin)

    cx_norm = _clamp01((cx_abs - tile_x) / tile_size)
    cy_norm = _clamp01((cy_abs - tile_y) / tile_size)
    w_norm  = _clamp01(w_abs / tile_size)
    h_norm  = _clamp01(h_abs / tile_size)

    # Discard tiny masks/boxes (less than 0.15% of tile area)
    if w_norm * h_norm < 0.0015:
        return None

    if had_poly:
        # Shift polygon to tile-local coords, normalize, then clamp
        norm_poly = []
        for x, y in clipped_poly:
            nx = _clamp01((x - tile_x) / tile_size)
            ny = _clamp01((y - tile_y) / tile_size)
            norm_poly.extend([nx, ny])

        # YOLO-Seg style: class cx cy w h [poly...]
        label_line = (
            f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f} "
            + " ".join(f"{p:.6f}" for p in norm_poly)
        )
    else:
        # Bbox-only input → bbox-only output (5 fields)
        label_line = f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

    return label_line

def _parse_tile_label_line_area_and_center(label_line):
    """Return (area_frac, cx_norm, cy_norm) from a tile-space YOLO line."""
    try:
        parts = label_line.strip().split()
        if len(parts) < 5:
            return 0.0, 0.5, 0.5
        cx_n = float(parts[1]); cy_n = float(parts[2])
        w_n = float(parts[3]); h_n = float(parts[4])
        return max(0.0, w_n * h_n), cx_n, cy_n
    except Exception:
        return 0.0, 0.5, 0.5

# --- tiling core ------------------------------------------------------
def create_tiles_from_image(image_path, label_path, tile_size=1280, overlap=0.30,
                            min_panel_frac=0.20, retile_if_below=True, pre_scale=1.0):
    """Create tiles from an image with proper label transformation."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return []

    # Optional pre-scale (labels normalized → safe)
    if pre_scale and abs(float(pre_scale) - 1.0) > 1e-6:
        h0, w0 = img.shape[:2]
        new_w = max(1, int(round(w0 * float(pre_scale))))
        new_h = max(1, int(round(h0 * float(pre_scale))))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return []

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Use shared tiling logic
    tiles_xyxy = make_tiles(h, w, tile_size, overlap)
    tiles = []
    tile_count = 0

    # Generate tiles using shared logic
    for (tile_x, tile_y, x2, y2) in tiles_xyxy:
        tile_w = x2 - tile_x
        tile_h = y2 - tile_y

        tile_img = img[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]

        if tile_w < tile_size or tile_h < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:tile_h, :tile_w] = tile_img
            tile_img = padded

        # Transform annotations for this tile
        tile_labels = []
        for line in lines:
            tl = transform_annotation_for_tile(
                line, tile_x, tile_y, tile_size, w, h
            )
            if tl:
                tile_labels.append(tl)

        if tile_labels:
            # Occupancy check + optional re-tile
            areas, centers = [], []
            for ll in tile_labels:
                a, cxn, cyn = _parse_tile_label_line_area_and_center(ll)
                areas.append(a); centers.append((cxn, cyn))
            max_area = max(areas) if areas else 0.0
            chosen_center = centers[areas.index(max_area)] if areas else (0.5, 0.5)

            if retile_if_below and max_area < float(min_panel_frac):
                cx_abs = tile_x + chosen_center[0] * tile_size
                cy_abs = tile_y + chosen_center[1] * tile_size
                new_tx = int(max(0, min(w - tile_size, cx_abs - tile_size / 2)))
                new_ty = int(max(0, min(h - tile_size, cy_abs - tile_size / 2)))

                # Rebuild labels for centered tile
                retile_labels = []
                for line in lines:
                    tl = transform_annotation_for_tile(line, new_tx, new_ty, tile_size, w, h)
                    if tl:
                        retile_labels.append(tl)

                # New image crop (+ pad if needed)
                new_tw = min(tile_size, w - new_tx)
                new_th = min(tile_size, h - new_ty)
                new_img = img[new_ty:new_ty + new_th, new_tx:new_tx + new_tw]
                if new_tw < tile_size or new_th < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:new_th, :new_tw] = new_img
                    new_img = padded

                # Adopt if better (or passes threshold)
                if retile_labels:
                    new_areas = [_parse_tile_label_line_area_and_center(ll)[0] for ll in retile_labels]
                    new_max_area = max(new_areas) if new_areas else 0.0
                    if new_max_area >= max(max_area, float(min_panel_frac)):
                        tile_img = new_img
                        tile_labels = retile_labels
                        tile_x, tile_y = new_tx, new_ty

            tiles.append({
                "image": tile_img,
                "labels": tile_labels,
                "tile_id": f"r{tile_y}_c{tile_x}",
                "bounds": (tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)
            })
            tile_count += 1

    print(f"Created {tile_count} tiles with annotations")
    return tiles

def tile_split_images(split, base_dir="datasets/rcp_dual_seg", tile_size=1280, overlap=0.30,
                      min_panel_frac=0.20, retile_if_below=True, pre_scale=1.0):
    """Tile images in a split."""
    img_dir = f"{base_dir}/images/{split}/fullsize"
    lbl_dir = f"{base_dir}/labels/{split}/fullsize"
    out_img = f"{base_dir}/images/{split}/tiled1280"
    out_lbl = f"{base_dir}/labels/{split}/tiled1280"

    # Check dirs
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found: {img_dir}")
        return 0
    if not os.path.exists(lbl_dir):
        print(f"Warning: Label directory not found: {lbl_dir}")
        return 0

    # Clean output
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    for file in os.listdir(out_img):
        try: os.remove(os.path.join(out_img, file))
        except: pass
    for file in os.listdir(out_lbl):
        try: os.remove(os.path.join(out_lbl, file))
        except: pass

    total_tiles = 0

    # Process each image
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = f"{img_dir}/{fn}"
        lbl_path = f"{lbl_dir}/{Path(fn).stem}.txt"

        print(f"Processing {fn}...")
        tiles = create_tiles_from_image(img_path, lbl_path, tile_size, overlap,
                                        min_panel_frac, retile_if_below, pre_scale)

        if tiles:
            for tile in tiles:
                tile_name = f"{Path(fn).stem}_{tile['tile_id']}"
                cv2.imwrite(f"{out_img}/{tile_name}.jpg", tile["image"])
                with open(f"{out_lbl}/{tile_name}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(tile["labels"]))
                total_tiles += 1
            print(f"  Created {len(tiles)} tiles")
        else:
            print("  No tiles created")

    return total_tiles

# --- CLI --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tile full-size images with proper YOLO(-Seg) label transformation")
    parser.add_argument("--base_dir", default="datasets/rcp_dual_seg", help="Base directory for dataset")
    parser.add_argument("--tile_size", type=int, default=CFG.tile_size, help="Tile size in pixels")
    parser.add_argument("--overlap", type=float, default=CFG.overlap, help="Overlap ratio (0.0-1.0)")
    parser.add_argument("--min-panel-frac", type=float, default=CFG.min_panel_frac, help="Minimum largest-panel area fraction inside a tile before adaptive re-tiling")
    parser.add_argument("--retile-if-below", action="store_true", default=CFG.retile_if_below, help="Enable adaptive re-tiling around the largest panel if area is below threshold")
    parser.add_argument("--pre-scale", type=float, default=CFG.pre_scale, help="Uniformly scale the full image before tiling (e.g., 0.75)")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated list of splits to tile (e.g., train,val)")
    args = parser.parse_args()

    print("=== Full-Size Image Tiling ===")
    print(f"Base directory: {args.base_dir}")
    print(f"Tile size: {args.tile_size}px")
    print(f"Overlap: {args.overlap:.1%}\n")

    splits = [s.strip() for s in args.splits.split(",")]
    info = {}
    for split in splits:
        print(f"Processing {split} split...")
        info[split] = tile_split_images(split, args.base_dir, args.tile_size, args.overlap,
                                        args.min_panel_frac, args.retile_if_below, args.pre_scale)

    print(f"\nTiling Summary: {info}")
    print("Tiling complete!")

if __name__ == "__main__":
    main()
