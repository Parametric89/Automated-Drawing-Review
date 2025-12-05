#!/usr/bin/env python3
"""
Apply masks from a top-colors Excel report and write per-image mask and overlay.

Inputs:
- Excel produced by utils/report_top_colors.py (sheet 'top_colors')
- Uses columns: image, color1_rgb, color2_rgb (optionally color3_rgb if --use-topn 3)

Outputs:
- mask/: 8-bit mask where selected colors (with tolerance) are 255, else 0
- overlay/: original image with translucent color overlay on masked regions

Example:
  python utils/apply_top_colors_mask.py --excel reports/top_colors.xlsx --out inference/top_colors_masks --use-topn 2 --tol-frac 0.05 --overlay-alpha 0.4
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import shutil
import json

import cv2
import numpy as np

# Optional fallback import (outline-based detection when color masks fail)
try:
    from utils.panel_outline_fallback import outline_fallback_mask  # type: ignore
except Exception:
    try:
        from panel_outline_fallback import outline_fallback_mask  # type: ignore
    except Exception:
        outline_fallback_mask = None  # will guard at runtime


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_rgb_tuple(s: str) -> Tuple[int, int, int]:
    # expects formats like '(173,173,173)' or '173, 173, 173'
    if s is None:
        raise ValueError("empty color string")
    s2 = s.strip().replace("(", "").replace(")", "")
    parts = [p.strip() for p in s2.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"bad rgb: {s}")
    r, g, b = map(int, parts)
    return (r, g, b)


def make_mask_for_colors(img_bgr: np.ndarray, colors_rgb: List[Tuple[int, int, int]], tol_abs: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    img = img_bgr.astype(np.int16)
    for (r, g, b) in colors_rgb:
        target_bgr = np.array([b, g, r], dtype=np.int16)
        diff = np.abs(img - target_bgr)
        within = (diff[:, :, 0] <= tol_abs) & (diff[:, :, 1] <= tol_abs) & (diff[:, :, 2] <= tol_abs)
        mask[within] = 255
    return mask


def main():
    ap = argparse.ArgumentParser(description="Apply masks from top-colors Excel and save overlays")
    ap.add_argument("--excel", required=True, help="Path to Excel produced by report_top_colors.py")
    ap.add_argument("--out", required=True, help="Output directory root for masks/overlays")
    ap.add_argument("--use-topn", type=int, default=2, help="How many top colors to include (1..3)")
    ap.add_argument("--tol-frac", type=float, default=0.05, help="Per-channel tolerance as fraction of 255")
    ap.add_argument("--tol-abs", type=int, default=None, help="Override: per-channel absolute tolerance (0..255)")
    ap.add_argument("--pad-abs", type=int, default=0, help="Additional per-channel pad added to tolerance (0..255)")
    ap.add_argument("--pad-gray-only", action="store_true", help="Apply --pad-abs only for near-gray top colors")
    ap.add_argument("--gray-chroma-thresh", type=int, default=18, help="Max channel delta for a color to be considered near-gray")
    ap.add_argument("--twoof3", action="store_true", help="Mask a pixel if at least 2 of 3 channels are within tolerance")
    # Optional post-mask cap to avoid full-page fills
    ap.add_argument("--cap-mask-frac", type=float, default=None, help="If set and mask coverage exceeds this fraction, cap by keeping only edge-dense, low-sat components")
    ap.add_argument("--edge-min", type=float, default=10.0, help="Min mean Sobel magnitude inside a component to keep during capping")
    ap.add_argument("--sat-keep-max", type=int, default=60, help="Max mean saturation (HSV S) inside a component to keep during capping")
    ap.add_argument("--cap-strategy", type=str, default="score", choices=["score", "largest"], help="Capping strategy when over coverage")
    ap.add_argument("--overlay-alpha", type=float, default=0.45, help="Overlay alpha for the masked regions")
    ap.add_argument("--overlay-color", type=str, default="255,0,0", help="Overlay BGR color, e.g., '0,255,0'")
    ap.add_argument("--down", type=int, default=2048, help="Downscale images to this max side before masking (0 to keep original)")
    ap.add_argument("--skip-overlay", action="store_true", help="Skip writing overlay images for speed")
    ap.add_argument("--visible-min-frac", type=float, default=0.10, help="Min fraction of a box that must remain after crop to keep it")
    # Manifest output
    ap.add_argument("--manifest", type=str, default=None, help="Optional path to write JSONL manifest of crops")
    ap.add_argument("--manifest-overwrite", action="store_true", help="Overwrite manifest if it exists (default appends)")
    ap.add_argument("--manifest-only", action="store_true", help="Only write manifest/masks (no crops/labels/overlays)")
    # ROI crop output (always bbox when mask exists)
    ap.add_argument("--silhouette", action="store_true", help="(deprecated)")
    ap.add_argument("--crop-mode", type=str, default="bbox", choices=["bbox", "silhouette"], help="(deprecated)")
    ap.add_argument("--pad-frac", type=float, default=0.01, help="Padding fraction added around the tight bbox of the mask")
    # (reverted) label remap flags removed
    # Fallback removed for non-hatched drawings; flags above are deprecated.
    # Anti-aliased text suppression
    ap.add_argument("--black-cutoff", type=int, default=30, help="Drop masked pixels where gray < cutoff (exclude near-black text edges)")
    ap.add_argument("--nbhd-ksize", type=int, default=5, help="Neighborhood size (odd) for majority filter")
    ap.add_argument("--nbhd-frac", type=float, default=0.5, help="Keep pixels only if this fraction of the neighborhood is also masked")
    ap.add_argument("--min-comp-area-frac", type=float, default=0.0005, help="Drop connected components smaller than this fraction of image")
    # Strengthen/bridge missing bits
    ap.add_argument("--close-k", type=int, default=0, help="Final closing kernel (odd) to bridge small gaps (0 disables)")
    ap.add_argument("--dilate-k", type=int, default=0, help="Final dilation kernel (odd) to slightly expand mask (0 disables)")
    ap.add_argument("--fill-holes", action="store_true", help="Fill interior holes in the final mask")
    # Optional HSV constraints (disabled by default) for targeted suppression without changing defaults
    ap.add_argument("--limit-sat", type=int, default=None, help="If set, drop masked pixels with saturation > value (0-255)")
    ap.add_argument("--limit-hue", type=str, default=None, help="If set, keep only masked pixels with hue in 'hmin,hmax' (0-179)")
    ap.add_argument("--limit-v", type=str, default=None, help="If set, keep only masked pixels with value in 'vmin,vmax' (0-255)")
    # Mask inversion options
    ap.add_argument("--invert-mask", action="store_true", help="Force invert final mask (swap black/white)")
    ap.add_argument("--invert-on-fallback", action="store_true", help="Invert mask only when fallback was used")
    args = ap.parse_args()

    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("This script requires pandas and openpyxl. Please `pip install pandas openpyxl`.") from e

    df = pd.read_excel(args.excel, sheet_name="top_colors")
    out_root = Path(args.out)
    masks_dir = out_root / "mask"; ensure_dir(masks_dir)
    overlays_dir = out_root / "overlay"; ensure_dir(overlays_dir)
    roi_dir = out_root / "ROI_images"; ensure_dir(roi_dir)
    roi_labels_dir = out_root / "ROI_labels"; ensure_dir(roi_labels_dir)

    use_n = max(1, min(int(args.use_topn), 3))
    if args.tol_abs is not None:
        base_tol = max(0, min(255, int(args.tol_abs)))
    else:
        base_tol = max(0, min(255, int(round(float(args.tol_frac) * 255))))
    tol_abs = max(0, min(255, int(base_tol + max(0, int(args.pad_abs)))))

    try:
        bgr_parts = [int(x.strip()) for x in args.overlay_color.split(",")]
        overlay_bgr = np.array([bgr_parts[0], bgr_parts[1], bgr_parts[2]], dtype=np.uint8)
    except Exception:
        overlay_bgr = np.array([255, 0, 0], dtype=np.uint8)

    # Prepare manifest writer
    manifest_fp = None
    if args.manifest:
        manifest_path = Path(str(args.manifest))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if bool(args.manifest_overwrite) else "a"
        manifest_fp = manifest_path.open(mode, encoding="utf-8")

    for _, row in df.iterrows():
        img_path = Path(str(row.get("image", "")).strip())
        if not img_path.exists():
            # Try relative to excel folder
            alt = Path(args.excel).parent / img_path
            img_path = alt if alt.exists() else img_path
        base = img_path.stem
        img_full = cv2.imread(str(img_path))
        if img_full is None or img_full.size == 0:
            continue
        img = img_full.copy()
        # Downscale for faster QA masks/overlays
        s = 1.0
        if int(args.down) > 0:
            h0, w0 = img.shape[:2]
            m = max(h0, w0)
            if m > int(args.down):
                s = float(args.down) / float(m)
                nw = max(1, int(round(w0 * s)))
                nh = max(1, int(round(h0 * s)))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        colors_rgb: List[Tuple[int, int, int]] = []
        for i in range(1, use_n + 1):
            key = f"color{i}_rgb"
            if key in row and isinstance(row[key], str) and row[key].strip():
                try:
                    colors_rgb.append(parse_rgb_tuple(row[key]))
                except Exception:
                    pass
        if not colors_rgb:
            continue

        # Initial mask
        # Build mask with optional per-color gray-only pad and 2-of-3 rule
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        img_i16 = img.astype(np.int16)
        for (r, g, b) in colors_rgb:
            chroma = max(abs(int(r) - int(g)), abs(int(g) - int(b)), abs(int(b) - int(r)))
            extra_pad = 0
            if int(args.pad_abs) > 0:
                if bool(args.pad_gray_only):
                    if chroma <= int(args.gray_chroma_thresh):
                        extra_pad = int(args.pad_abs)
                else:
                    extra_pad = int(args.pad_abs)
            tol_use = max(0, min(255, int(tol_abs + extra_pad)))
            target_bgr = np.array([int(b), int(g), int(r)], dtype=np.int16)
            diff = np.abs(img_i16 - target_bgr)
            if bool(args.twoof3):
                within = ((diff[:, :, 0] <= tol_use).astype(np.int32) +
                          (diff[:, :, 1] <= tol_use).astype(np.int32) +
                          (diff[:, :, 2] <= tol_use).astype(np.int32)) >= 2
            else:
                within = (diff[:, :, 0] <= tol_use) & (diff[:, :, 1] <= tol_use) & (diff[:, :, 2] <= tol_use)
            mask[within] = 255
        # Optional HSV gating (disabled unless args are set)
        if args.limit_sat is not None or args.limit_hue is not None or args.limit_v is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gate = np.ones(mask.shape, dtype=bool)
            if args.limit_sat is not None:
                gate = gate & (hsv[:, :, 1] <= int(args.limit_sat))
            if args.limit_hue:
                try:
                    hmin, hmax = [int(x.strip()) for x in str(args.limit_hue).split(",")]
                    gate = gate & (hsv[:, :, 0] >= hmin) & (hsv[:, :, 0] <= hmax)
                except Exception:
                    pass
            if args.limit_v:
                try:
                    vmin, vmax = [int(x.strip()) for x in str(args.limit_v).split(",")]
                    gate = gate & (hsv[:, :, 2] >= vmin) & (hsv[:, :, 2] <= vmax)
                except Exception:
                    pass
            mask = np.where(gate, mask, 0).astype(np.uint8)
        # 1) Exclude near-black pixels (anti-aliased text borders)
        if int(args.black_cutoff) > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask[gray < int(args.black_cutoff)] = 0
        # 2) Neighborhood majority filter
        k = max(1, int(args.nbhd_ksize))
        if k % 2 == 0:
            k += 1
        if k > 1 and float(args.nbhd_frac) > 0:
            # fraction of masked pixels in kxk window
            m_float = (mask.astype(np.float32) / 255.0)
            nb = cv2.boxFilter(m_float, ddepth=-1, ksize=(k, k), normalize=True)
            mask = np.where(nb >= float(args.nbhd_frac), 255, 0).astype(np.uint8)
        # 3) Remove tiny components
        if float(args.min_comp_area_frac) > 0:
            h, w = mask.shape[:2]
            thr = float(args.min_comp_area_frac) * float(h * w)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = np.zeros(num, dtype=bool)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= thr:
                    keep[i] = True
            mask = np.where(keep[labels], 255, 0).astype(np.uint8)
        # 4) Optional closing/dilation to recover thin gaps
        def _odd(v: int) -> int:
            return v if v % 2 == 1 else v + 1
        if int(args.close_k) > 0:
            k = _odd(int(args.close_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
        if int(args.dilate_k) > 0:
            k = _odd(int(args.dilate_k))
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        # 5) Optional fill holes inside mask
        if bool(args.fill_holes):
            h, w = mask.shape[:2]
            inv = 255 - mask
            flood = inv.copy()
            ffmask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ffmask, (0, 0), 0)
            holes = inv - flood
            mask = cv2.bitwise_or(mask, holes)

        # Optional post-mask coverage cap
        if args.cap_mask_frac is not None and float(args.cap_mask_frac) > 0:
            cov = float(np.count_nonzero(mask)) / float(mask.size)
            cap_thr = float(args.cap_mask_frac)
            if cov > cap_thr:
                hh, ww = mask.shape[:2]
                total_allow = int(round(cap_thr * float(hh * ww)))
                num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if num > 1:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                    grad = np.abs(sx) + np.abs(sy)
                    comps = []
                    for i in range(1, num):
                        area = int(stats[i, cv2.CC_STAT_AREA])
                        if area <= 0:
                            continue
                        m_i = (labels == i)
                        mean_s = float(hsv[:, :, 1][m_i].mean()) if area > 0 else 255.0
                        mean_edge = float(grad[m_i].mean()) if area > 0 else 0.0
                        comps.append((i, area, mean_edge, mean_s))
                    # Filter by thresholds
                    kept = []
                    for (i, area, mean_edge, mean_s) in comps:
                        if args.cap_strategy == "score":
                            if mean_edge >= float(args.edge_min) and int(mean_s) <= int(args.sat_keep_max):
                                kept.append((i, area, mean_edge))
                        else:
                            kept.append((i, area, mean_edge))
                    # Sort by score/area
                    if args.cap_strategy == "score":
                        kept.sort(key=lambda x: (x[2], x[1]), reverse=True)
                    else:
                        kept.sort(key=lambda x: x[1], reverse=True)
                    # Accumulate until under cap (never add a component larger than remaining allowance)
                    new_mask = np.zeros_like(mask)
                    acc = 0
                    for (i, area, _) in kept:
                        if acc + area > total_allow:
                            continue
                        new_mask[labels == i] = 255
                        acc += area
                        if acc >= total_allow:
                            break
                    # Fallback: shrink top-scoring/largest component by erosion until under cap
                    if np.count_nonzero(new_mask) == 0:
                        # Choose candidate: best from 'kept' or from 'comps' if kept is empty
                        if len(kept) > 0:
                            cand_idx = kept[0][0]
                        elif len(comps) > 0:
                            comps.sort(key=lambda x: x[1], reverse=True)
                            cand_idx = comps[0][0]
                        else:
                            cand_idx = None
                        if cand_idx is not None:
                            cand_mask = (labels == cand_idx).astype(np.uint8) * 255
                            # Iteratively erode until area <= total_allow or limit iters
                            iters = 0
                            ker = np.ones((3, 3), np.uint8)
                            while np.count_nonzero(cand_mask) > total_allow and iters < 32:
                                cand_mask = cv2.erode(cand_mask, ker, iterations=1)
                                iters += 1
                            new_mask = cand_mask
                    mask = new_mask

        # Fallback disabled: do not alter mask here
        kmeans_pts = None

        # Optional inversion controls (fallback disabled); invert if explicitly requested
        if bool(args.invert_mask):
            mask = 255 - mask

        # Save mask
        cv2.imwrite(str(masks_dir / f"{base}_mask.png"), mask)

        # Overlay
        if (not bool(args.skip_overlay)) and (not bool(args.manifest_only)):
            overlay = img.copy()
            color_layer = np.zeros_like(img)
            color_layer[:, :] = overlay_bgr
            m3 = cv2.merge([mask, mask, mask])
            overlay = np.where(m3 > 0, (overlay * (1.0 - args.overlay_alpha) + color_layer * args.overlay_alpha).astype(np.uint8), overlay)
            cv2.imwrite(str(overlays_dir / f"{base}_overlay.jpg"), overlay)

        # ROI output: save bbox crop if mask exists; else save full-size image
        mh, mw = mask.shape[:2]
        H, W = img_full.shape[:2]
        if np.count_nonzero(mask) > 0:
            ys, xs = np.where(mask > 0)
            # Map bbox from downscaled mask space back to full resolution without resizing the mask
            inv_s = (1.0 / float(s)) if float(s) > 0 else 1.0
            x0 = int(max(0, min(W - 1, np.floor(xs.min() * inv_s))))
            x1 = int(max(0, min(W - 1, np.ceil(xs.max() * inv_s))))
            y0 = int(max(0, min(H - 1, np.floor(ys.min() * inv_s))))
            y1 = int(max(0, min(H - 1, np.ceil(ys.max() * inv_s))))
            pad_x = int(round(float(args.pad_frac) * float(x1 - x0 + 1)))
            pad_y = int(round(float(args.pad_frac) * float(y1 - y0 + 1)))
            x0 = max(0, x0 - pad_x); x1 = min(W - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y); y1 = min(H - 1, y1 + pad_y)
            roi_img = img_full[y0:y1 + 1, x0:y1 + 1].copy()
            roi_img_path = roi_dir / f"{base}_roi.png"
            # Cleanup any stale full-image variant from previous runs
            try:
                alt_full = roi_dir / f"{base}{img_path.suffix.lower()}"
                if alt_full.exists():
                    alt_full.unlink()
            except Exception:
                pass
            if not bool(args.manifest_only):
                cv2.imwrite(str(roi_img_path), roi_img)
            # Prepare destination label path and ensure a file exists regardless of remap outcome
            dst_lbl = roi_labels_dir / f"{base}_roi.txt"
            if not bool(args.manifest_only):
                try:
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    if not dst_lbl.exists():
                        dst_lbl.write_text("", encoding="utf-8")
                except Exception:
                    pass
            # Call standalone label remapper if a corresponding label exists
            try:
                # Candidates: global inference/fullsize_labels and sibling of the image folder
                cand1 = Path("inference/fullsize_labels") / f"{base}.txt"
                cand2 = (img_path.parent.parent / "fullsize_labels" / f"{base}.txt") if len(img_path.parents) >= 2 else cand1
                src_lbl = cand1 if cand1.exists() else cand2
                src_count = 0
                if src_lbl.exists() and (not bool(args.manifest_only)):
                    # Count source boxes
                    try:
                        src_lines = [ln.strip() for ln in src_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
                        src_count = sum(1 for ln in src_lines if len(ln.split()) >= 5)
                    except Exception:
                        src_count = 0
                    from utils.remap_labels_by_crop import remap_yolo_labels_for_crop  # type: ignore
                    remap_yolo_labels_for_crop(
                        src_label=src_lbl,
                        dst_label=dst_lbl,
                        img_w=W,
                        img_h=H,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        visible_min_frac=float(args.visible_min_frac),
                        keep_if_center_in_roi=True,
                    )
            except Exception:
                pass

            # Manifest entry for ROI crop
            if manifest_fp is not None:
                kept_count = 0
                try:
                    kept_lines = [ln.strip() for ln in dst_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    kept_count = sum(1 for ln in kept_lines if len(ln.split()) >= 5)
                except Exception:
                    kept_count = 0
                cov = float(np.count_nonzero(mask)) / float(mask.size) if mask.size > 0 else 0.0
                reason = None
                if (locals().get("src_count", 0) or 0) > 0 and kept_count == 0:
                    reason = "no_intersections_or_filtered_by_threshold"
                record: Dict[str, Any] = {
                    "image": str(img_path),
                    "src_label": str(src_lbl) if 'src_lbl' in locals() and src_lbl.exists() else None,
                    "roi_image": str(roi_img_path),
                    "roi_label": str(dst_lbl),
                    "bbox": {
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x1),
                        "y1": int(y1),
                    },
                    "scale": float(s),
                    "mask_coverage": cov,
                    "src_box_count": int(locals().get("src_count", 0) or 0),
                    "kept_box_count": int(kept_count),
                    "remap_note": reason,
                }
                manifest_fp.write(json.dumps(record) + "\n")
        else:
            ext = img_path.suffix.lower()
            roi_img_path = roi_dir / f"{base}{ext}"
            # Cleanup any stale ROI variant from previous runs
            try:
                alt_roi = roi_dir / f"{base}_roi.png"
                if alt_roi.exists():
                    alt_roi.unlink()
            except Exception:
                pass
            if not bool(args.manifest_only):
                cv2.imwrite(str(roi_img_path), img_full)
            # For full-image saves, mirror/copy the original labels (or create empty if none)
            try:
                cand1 = Path("inference/fullsize_labels") / f"{base}.txt"
                cand2 = (img_path.parent.parent / "fullsize_labels" / f"{base}.txt") if len(img_path.parents) >= 2 else cand1
                src_lbl = cand1 if cand1.exists() else cand2
                dst_lbl = roi_labels_dir / f"{base}.txt"
                # Cleanup any stale ROI label variant from previous runs
                try:
                    alt_roi_lbl = roi_labels_dir / f"{base}_roi.txt"
                    if alt_roi_lbl.exists():
                        alt_roi_lbl.unlink()
                except Exception:
                    pass
                src_count = 0
                if not bool(args.manifest_only):
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    if src_lbl.exists():
                        shutil.copyfile(src_lbl, dst_lbl)
                        try:
                            src_lines = [ln.strip() for ln in src_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
                            src_count = sum(1 for ln in src_lines if len(ln.split()) >= 5)
                        except Exception:
                            src_count = 0
                    else:
                        dst_lbl.write_text("", encoding="utf-8")
            except Exception:
                pass

            # Manifest entry for full-image case
            if manifest_fp is not None:
                kept_count = 0
                try:
                    kept_lines = [ln.strip() for ln in dst_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    kept_count = sum(1 for ln in kept_lines if len(ln.split()) >= 5)
                except Exception:
                    kept_count = 0
                record: Dict[str, Any] = {
                    "image": str(img_path),
                    "src_label": str(src_lbl) if 'src_lbl' in locals() and src_lbl.exists() else None,
                    "roi_image": str(roi_img_path),
                    "roi_label": str(dst_lbl),
                    "bbox": None,
                    "scale": float(s),
                    "mask_coverage": None,
                    "src_box_count": int(locals().get("src_count", 0) or 0),
                    "kept_box_count": int(kept_count),
                    "remap_note": None,
                }
                manifest_fp.write(json.dumps(record) + "\n")

    if manifest_fp is not None:
        manifest_fp.close()
    print(f"\nWrote masks to: {masks_dir}\nWrote overlays to: {overlays_dir}\nWrote ROI images to: {roi_dir}\nWrote ROI labels to: {roi_labels_dir}")


if __name__ == "__main__":
    main()


