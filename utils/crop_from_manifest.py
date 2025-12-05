#!/usr/bin/env python3
"""
Crop images and remap YOLO labels strictly from a JSONL manifest produced by apply_top_colors_mask.py.

Each JSON line must include:
- image (path to source image)
- roi_image (path to write ROI image)
- roi_label (path to write ROI label)
- bbox: {x0,y0,x1,y1} in full-resolution coords (or null for full image copy)

Usage:
  python utils/crop_from_manifest.py \
    --manifest inference/top_colors_masks/manifest.jsonl \
    --visible-min-frac 0.10

Notes:
- Only crops/labels are written; this tool does not generate masks or overlays.
- Paths in the manifest are used as-is. Missing directories will be created.
"""

from pathlib import Path
import argparse
import json
from typing import Dict, Any, List
import cv2


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_lines(path: Path, lines: List[str]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def remap_yolo_labels_for_bbox(src_label: Path, dst_label: Path, img_w: int, img_h: int, x0: int, y0: int, x1: int, y1: int, visible_min_frac: float, keep_if_center_in_roi: bool = True) -> int:
    if not src_label.exists():
        _write_lines(dst_label, [])
        return 0
    try:
        lines = [ln.strip() for ln in src_label.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        _write_lines(dst_label, [])
        return 0
    rw = float(max(1, int(x1) - int(x0) + 1))
    rh = float(max(1, int(y1) - int(y0) + 1))
    W = float(img_w)
    H = float(img_h)
    out: List[str] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        # Keep only class 0 (panels); drop tags or others in ROI labels
        if str(cls) != "0":
            continue
        # Detect segmentation format (class followed by x,y pairs)
        is_segmentation = (len(parts) > 5) and (((len(parts) - 5) % 2) == 0)
        if is_segmentation:
            try:
                poly_vals = [float(v) for v in parts[5:]]
                xs = poly_vals[0::2]
                ys = poly_vals[1::2]
            except Exception:
                xs, ys = [], []
            if not xs or not ys:
                continue
            minx = min(xs) * W
            maxx = max(xs) * W
            miny = min(ys) * H
            maxy = max(ys) * H
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            bw = max(0.0, maxx - minx)
            bh = max(0.0, maxy - miny)
        else:
            x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            cx = x * W; cy = y * H
            bw = w * W; bh = h * H
        x_left = cx - bw / 2.0; x_right = cx + bw / 2.0
        y_top = cy - bh / 2.0; y_bottom = cy + bh / 2.0
        inter_left = max(x_left, float(x0))
        inter_right = min(x_right, float(x1))
        inter_top = max(y_top, float(y0))
        inter_bottom = min(y_bottom, float(y1))
        inter_w = max(0.0, inter_right - inter_left)
        inter_h = max(0.0, inter_bottom - inter_top)
        vis_frac = (inter_w * inter_h) / max(1e-6, (bw * bh))
        if inter_w <= 1e-9 or inter_h <= 1e-9:
            if not (keep_if_center_in_roi and (float(x0) <= cx <= float(x1) and float(y0) <= cy <= float(y1))):
                continue
            inter_left = min(max(x_left, float(x0)), float(x1))
            inter_right = min(max(x_right, float(x0)), float(x1))
            inter_top = min(max(y_top, float(y0)), float(y1))
            inter_bottom = min(max(y_bottom, float(y0)), float(y1))
            inter_w = max(0.0, inter_right - inter_left)
            inter_h = max(0.0, inter_bottom - inter_top)
        elif vis_frac < float(visible_min_frac):
            if not (keep_if_center_in_roi and (float(x0) <= cx <= float(x1) and float(y0) <= cy <= float(y1))):
                continue
            inter_left = min(max(x_left, float(x0)), float(x1))
            inter_right = min(max(x_right, float(x0)), float(x1))
            inter_top = min(max(y_top, float(y0)), float(y1))
            inter_bottom = min(max(y_bottom, float(y0)), float(y1))
            inter_w = max(0.0, inter_right - inter_left)
            inter_h = max(0.0, inter_bottom - inter_top)
        new_cx = (inter_left + inter_right) / 2.0
        new_cy = (inter_top + inter_bottom) / 2.0
        new_w = inter_w
        new_h = inter_h
        nx = (new_cx - float(x0)) / rw
        ny = (new_cy - float(y0)) / rh
        nw = new_w / rw
        nh = new_h / rh
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        out.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    _write_lines(dst_label, out)
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop images and labels strictly from a manifest JSONL")
    ap.add_argument("--manifest", required=True, help="Path to manifest JSONL emitted by apply_top_colors_mask.py")
    ap.add_argument("--visible-min-frac", type=float, default=0.0, help="Min fraction of a box that must remain after crop to keep it")
    ap.add_argument("--force-keep-nearest", action="store_true", help="If remap yields 0 boxes, keep the source box with the largest intersection area with ROI")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec: Dict[str, Any] = json.loads(line)
        img_path = Path(rec["image"]) if rec.get("image") else None
        roi_img_path = Path(rec["roi_image"]) if rec.get("roi_image") else None
        roi_lbl_path = Path(rec["roi_label"]) if rec.get("roi_label") else None
        bbox = rec.get("bbox")
        if img_path is None or roi_img_path is None or roi_lbl_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            continue
        H, W = img.shape[:2]

        if bbox is None:
            # Full-image copy
            _ensure_parent(roi_img_path)
            cv2.imwrite(str(roi_img_path), img)
            # Mirror labels (copy or empty)
            src_lbl_path = Path(rec.get("src_label") or "")
            _ensure_parent(roi_lbl_path)
            if src_lbl_path.exists():
                try:
                    content = src_lbl_path.read_text(encoding="utf-8")
                except Exception:
                    content = ""
                roi_lbl_path.write_text(content, encoding="utf-8")
            else:
                roi_lbl_path.write_text("", encoding="utf-8")
            continue

        x0 = int(bbox.get("x0", 0)); y0 = int(bbox.get("y0", 0))
        x1 = int(bbox.get("x1", W - 1)); y1 = int(bbox.get("y1", H - 1))
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1))
        if x1 < x0 or y1 < y0:
            # invalid bbox; skip
            continue

        # Crop image
        crop = img[y0:y1 + 1, x0:x1 + 1].copy()
        _ensure_parent(roi_img_path)
        cv2.imwrite(str(roi_img_path), crop)

        # Remap labels
        src_lbl_path = Path(rec.get("src_label") or "")
        _ensure_parent(roi_lbl_path)
        kept = remap_yolo_labels_for_bbox(
            src_label=src_lbl_path,
            dst_label=roi_lbl_path,
            img_w=W,
            img_h=H,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            visible_min_frac=float(args.visible_min_frac),
        )
        if kept == 0 and bool(args.force_keep_nearest):
            # Fallback: keep the single source box with the largest intersection area
            def _largest_intersection_box() -> str:
                if not src_lbl_path.exists():
                    return ""
                try:
                    lines = [ln.strip() for ln in src_lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                except Exception:
                    return ""
                Wf = float(W); Hf = float(H)
                best = None  # (area, cls, inter_left, inter_right, inter_top, inter_bottom)
                for ln in lines:
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    cls = parts[0]
                    # consider only class 0 (panels)
                    if str(cls) != "0":
                        continue
                    # segmentation lines have polygon pairs after the first 5 tokens
                    is_seg = (len(parts) > 5) and (((len(parts) - 5) % 2) == 0)
                    if is_seg:
                        try:
                            poly_vals = [float(v) for v in parts[5:]]
                        except Exception:
                            continue
                        xs = poly_vals[0::2]; ys = poly_vals[1::2]
                        if not xs or not ys:
                            continue
                        minx = min(xs) * Wf; maxx = max(xs) * Wf
                        miny = min(ys) * Hf; maxy = max(ys) * Hf
                        x_left = minx; x_right = maxx; y_top = miny; y_bottom = maxy
                    else:
                        cx = float(parts[1]) * Wf; cy = float(parts[2]) * Hf
                        bw = float(parts[3]) * Wf; bh = float(parts[4]) * Hf
                        x_left = cx - bw / 2.0; x_right = cx + bw / 2.0
                        y_top = cy - bh / 2.0; y_bottom = cy + bh / 2.0
                    inter_left = max(x_left, float(x0))
                    inter_right = min(x_right, float(x1))
                    inter_top = max(y_top, float(y0))
                    inter_bottom = min(y_bottom, float(y1))
                    inter_w = max(0.0, inter_right - inter_left)
                    inter_h = max(0.0, inter_bottom - inter_top)
                    inter_area = inter_w * inter_h
                    if inter_area <= 0.0:
                        continue
                    if (best is None) or (inter_area > best[0]):
                        best = (inter_area, cls, inter_left, inter_right, inter_top, inter_bottom)
                if best is None:
                    return ""
                _, cls, il, ir, it, ib = best
                rw = float(max(1, int(x1) - int(x0) + 1))
                rh = float(max(1, int(y1) - int(y0) + 1))
                new_cx = (il + ir) / 2.0
                new_cy = (it + ib) / 2.0
                new_w = max(0.0, ir - il)
                new_h = max(0.0, ib - it)
                nx = (new_cx - float(x0)) / rw
                ny = (new_cy - float(y0)) / rh
                nw = new_w / rw
                nh = new_h / rh
                nx = min(max(nx, 0.0), 1.0)
                ny = min(max(ny, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)
                return f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"

            best_line = _largest_intersection_box()
            if best_line:
                try:
                    roi_lbl_path.write_text(best_line + "\n", encoding="utf-8")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
