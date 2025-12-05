"""
Quickly overlay YOLO labels (seg + bbox) on images in a folder and save previews.

Usage:
  python utils/overlay_labels_folder.py --images <dir> --labels <dir> --out <dir> [--limit 50]

Supports:
- Class 0: polygon masks (YOLO-seg line with >= 6 coords after bbox)
- Class 1: bbox-only (YOLO box line)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def parse_yolo_seg_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    vals = list(map(float, parts[1:]))
    cx, cy, w, h = vals[:4]
    poly = vals[4:]
    return cls, cx, cy, w, h, poly


def denorm_box(cx, cy, w, h, W, H):
    x = (cx - w / 2.0) * W
    y = (cy - h / 2.0) * H
    wpx = w * W
    hpx = h * H
    return int(round(x)), int(round(y)), int(round(wpx)), int(round(hpx))


def denorm_poly(poly, W, H):
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= W
    pts[:, 1] *= H
    return pts.astype(np.int32)


def draw_overlay(img, lines):
    H, W = img.shape[:2]
    overlay = img.copy()
    for line in lines:
        parsed = parse_yolo_seg_line(line)
        if not parsed:
            continue
        cls, cx, cy, w, h, poly = parsed
        if cls == 0:
            if len(poly) >= 6:
                pts = denorm_poly(poly, W, H)
                cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 180, 0), thickness=2, lineType=cv2.LINE_AA)
            # Always draw bbox for class 0 as well (handles bbox-only labels after cropping)
            x, y, bw, bh = denorm_box(cx, cy, w, h, W, H)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2, cv2.LINE_AA)
        elif cls == 1:
            x, y, bw, bh = denorm_box(cx, cy, w, h, W, H)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Images directory")
    ap.add_argument("--labels", required=True, help="Labels directory")
    ap.add_argument("--out", required=True, help="Output directory for overlays")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    count = 0
    for img_path in imgs:
        if count >= args.limit:
            break
        lbl_path = (lbl_dir / img_path.stem).with_suffix(".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        out = draw_overlay(img, lines)
        cv2.imwrite(str(out_dir / img_path.name), out)
        count += 1

    print(f"Saved {count} overlays to {out_dir}")


if __name__ == "__main__":
    main()


