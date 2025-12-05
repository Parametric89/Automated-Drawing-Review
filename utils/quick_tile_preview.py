#!/usr/bin/env python3
"""
Quick image-only tiler for visual inspection.

- No labels required
- Optional pre-scale (e.g., 0.75) applied before tiling
- Pads edge tiles to requested tile size

Outputs tiles as JPEGs named <stem>_r<y>_c<x>.jpg
"""

import os
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(src: str):
    pats = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for p in pats:
        files.extend(glob.glob(str(Path(src) / p)))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="Quick image-only tiling preview")
    ap.add_argument("--src", required=True, help="Folder with source images (fullsize)")
    ap.add_argument("--out", required=True, help="Output folder for tiles")
    ap.add_argument("--tile-size", type=int, default=1280, help="Tile size (pixels)")
    ap.add_argument("--overlap", type=float, default=0.30, help="Overlap ratio (0.0-1.0)")
    ap.add_argument("--pre-scale", type=float, default=1.0, help="Uniform pre-scale factor (e.g., 0.75)")
    ap.add_argument("--center-align", action="store_true", help="Center-align the tiling grid on image center")
    args = ap.parse_args()

    ensure_dir(args.out)
    stride = max(1, int(round(args.tile_size * (1.0 - float(args.overlap)))))

    image_paths = list_images(args.src)
    print(f"Tiling {len(image_paths)} images from {args.src} → {args.out}")

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[warn] failed to read: {img_path}")
            continue

        if abs(float(args.pre_scale) - 1.0) > 1e-6:
            h0, w0 = img.shape[:2]
            new_w = max(1, int(round(w0 * float(args.pre_scale))))
            new_h = max(1, int(round(h0 * float(args.pre_scale))))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]
        stem = Path(img_path).stem

        tile_count = 0
        # Optionally shift the grid so a tile is centered on the image center
        if args.center_align:
            x_center_origin = max(0, (w - args.tile_size) // 2)
            y_center_origin = max(0, (h - args.tile_size) // 2)
            sx = int(x_center_origin % stride)
            sy = int(y_center_origin % stride)
        else:
            sx, sy = 0, 0

        for y in range(sy, h, stride):
            for x in range(sx, w, stride):
                tile = img[y:min(y + args.tile_size, h), x:min(x + args.tile_size, w)]
                if tile.shape[0] < args.tile_size or tile.shape[1] < args.tile_size:
                    pad = np.zeros((args.tile_size, args.tile_size, 3), dtype=np.uint8)
                    pad[:tile.shape[0], :tile.shape[1]] = tile
                    tile = pad
                out_name = f"{stem}_r{y}_c{x}.jpg"
                cv2.imwrite(str(Path(args.out) / out_name), tile)
                tile_count += 1
        print(f"[{idx}/{len(image_paths)}] {Path(img_path).name}: {tile_count} tiles")

    print("Done.")


if __name__ == "__main__":
    main()


