#!/usr/bin/env python3
"""
Predict panels on PDFs, images, or folders using the trained panel-only YOLO model.

- Input: PDF file, image file, or directory of images (and/or PDFs)
- Output: Saved predictions (images with boxes) and YOLO bbox-only labels (5 tokens)
- Tiled inference defaults align with tile_fullsize_images.py:
  tile-size=1280, tile-overlap=0.30
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Add root directory to sys.path to allow importing from utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.tiling_shared import load_tiling_config, make_tiles, inner_window, iqr_band

# Load shared config
CFG = load_tiling_config()

# Keep OpenCV deterministic and avoid oversubscription on laptop GPU/CPU
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image as PILImage
from utils.eval_utils import append_eval_row
from utils.matching import match_and_count

# Avoid PIL DecompressionBomb on very large rendered pages
PILImage.MAX_IMAGE_PIXELS = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_default_weights() -> str:
    preferred = Path("runs/train/v6_panel_bbox_synth_yolov8m3/weights/best.pt")
    if preferred.exists():
        return str(preferred)
    return "yolov8m.pt"


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 300) -> List[Path]:
    """Convert a PDF into page images using PyMuPDF (fitz). Returns list of image paths."""
    ensure_dir(out_dir)
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF input. Install with: pip install pymupdf") from e

    image_paths: List[Path] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in tqdm(range(len(doc)), desc=f"Render {pdf_path.name}"):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"page_{i+1:03d}.png"
        pix.save(str(out_path))
        image_paths.append(out_path)
    doc.close()
    return image_paths


def list_images_in_dir(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    def is_output_dir(path: Path) -> bool:
        parts = set(path.parts)
        if "pred" in parts or "pdf_pages" in parts:
            return True
        for seg in path.parts:
            if seg.startswith("run_") or seg.startswith("pred_"):
                return True
        return False
    return [p for p in sorted(dir_path.rglob("*")) if p.suffix.lower() in exts and not is_output_dir(p)]


def list_pdfs_in_dir(dir_path: Path) -> List[Path]:
    return [p for p in sorted(dir_path.rglob("*.pdf"))]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Simple NMS for single-class boxes (xyxy). Returns kept indices."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def cluster_1d(vals: np.ndarray, delta: float) -> np.ndarray:
    """Cluster 1D values by distance threshold delta. Returns cluster ids per value."""
    n = vals.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(vals)
    clusters = np.zeros(n, dtype=int)
    cid = 0
    prev = vals[order[0]]
    clusters[order[0]] = cid
    for idx in order[1:]:
        if abs(vals[idx] - prev) > delta:
            cid += 1
        clusters[idx] = cid
        prev = vals[idx]
    return clusters


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix for two sets of xyxy boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def merge_overlaps(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster boxes by IoU and merge each cluster by score-weighted averaging."""
    if boxes.size == 0:
        return boxes, scores
    remaining = np.arange(boxes.shape[0])
    merged_boxes = []
    merged_scores = []
    while remaining.size > 0:
        idx = remaining[0]
        base = boxes[idx:idx+1, :]
        ious = iou_matrix(base, boxes[remaining])[0]
        cluster_mask = ious >= iou_thresh
        cluster_idx = remaining[cluster_mask]
        w = scores[cluster_idx]
        bb = boxes[cluster_idx]
        x1 = np.average(bb[:, 0], weights=w)
        y1 = np.average(bb[:, 1], weights=w)
        x2 = np.average(bb[:, 2], weights=w)
        y2 = np.average(bb[:, 3], weights=w)
        merged_boxes.append([x1, y1, x2, y2])
        merged_scores.append(float(w.max()))
        remaining = remaining[~cluster_mask]
    return np.array(merged_boxes, dtype=np.float32), np.array(merged_scores, dtype=np.float32)


def merge_overlaps_minmax(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster boxes by IoU and merge each cluster by taking min-max bounds (avoids shrinking)."""
    if boxes.size == 0:
        return boxes, scores
    remaining = np.arange(boxes.shape[0])
    merged_boxes: List[List[float]] = []
    merged_scores: List[float] = []
    while remaining.size > 0:
        idx = remaining[0]
        base = boxes[idx:idx+1, :]
        ious = iou_matrix(base, boxes[remaining])[0]
        cluster_mask = ious >= iou_thresh
        cluster_idx = remaining[cluster_mask]
        bb = boxes[cluster_idx]
        x1 = float(np.min(bb[:, 0]))
        y1 = float(np.min(bb[:, 1]))
        x2 = float(np.max(bb[:, 2]))
        y2 = float(np.max(bb[:, 3]))
        merged_boxes.append([x1, y1, x2, y2])
        merged_scores.append(float(np.max(scores[cluster_idx])))
        remaining = remaining[~cluster_mask]
    return np.array(merged_boxes, dtype=np.float32), np.array(merged_scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training-style tiler helpers
# ---------------------------------------------------------------------------


def _to_xyxy_from_xywh(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=int)
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


def _tile_image_training_style(img: np.ndarray, cfg) -> List[dict]:
    tile_size = int(cfg.tile_size)
    overlap = float(cfg.overlap)
    h, w = img.shape[:2]
    tiles_xyxy = make_tiles(h, w, tile_size, overlap)
    tiles = []
    for (x1, y1, x2, y2) in tiles_xyxy:
        tw = x2 - x1
        th = y2 - y1
        crop = img[y1:y1 + th, x1:x1 + tw]
        if tw < tile_size or th < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:th, :tw] = crop
            crop = padded
        tiles.append({
            "image": crop,
            "origin": (x1, y1),
            "size": (tw, th),
            "bounds": (x1, y1, x2, y2),
        })
    return tiles


def infer_with_training_tiler(
    image_path: Path,
    model,
    cfg,
    save_dir: Path,
    conf: float,
    iou: float,
    merge_iou: float,
    imgsz: int,
    max_det: int,
    center_merge_ratio: float,
    keep_inner_frac: float,
    line_thickness: int,
):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    tile_size = int(cfg.tile_size)
    overlap = float(cfg.overlap)
    pre_scale = float(getattr(cfg, "pre_scale", 1.0))

    h0, w0 = img.shape[:2]
    if abs(pre_scale - 1.0) > 1e-6:
        new_w = max(1, int(round(w0 * pre_scale)))
        new_h = max(1, int(round(h0 * pre_scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_h, new_w = h0, w0

    print(f"[TRAINING-TILER] {image_path.name}: original {w0}x{h0}, scaled {new_w}x{new_h}")
    tiles = _tile_image_training_style(img, cfg)
    print(f"[TRAINING-TILER] Created {len(tiles)} tiles (size={tile_size}, overlap={overlap:.2f})")

    tile_imgs = [t["image"] for t in tiles]
    results = model.predict(
        source=tile_imgs,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )

    full_boxes = []
    full_scores = []
    full_classes = []
    for t, res in zip(tiles, results):
        origin_x, origin_y = t["origin"]
        bounds = t["bounds"]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xywhn = res.boxes.xywhn.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        ix1 = iy1 = ix2 = iy2 = None
        if keep_inner_frac > 0:
            ix1, iy1, ix2, iy2 = inner_window(bounds[0], bounds[1], bounds[2], bounds[3], overlap, keep_inner_frac)
        for (cx_n, cy_n, w_n, h_n), score, cls_id in zip(xywhn, scores, cls_ids):
            cx_t = cx_n * tile_size
            cy_t = cy_n * tile_size
            w_t = w_n * tile_size
            h_t = h_n * tile_size
            x1_t, y1_t, x2_t, y2_t = _to_xyxy_from_xywh(cx_t, cy_t, w_t, h_t)
            x1_s = x1_t + origin_x
            y1_s = y1_t + origin_y
            x2_s = x2_t + origin_x
            y2_s = y2_t + origin_y
            if keep_inner_frac > 0:
                cx_s = (x1_s + x2_s) / 2.0
                cy_s = (y1_s + y2_s) / 2.0
                if not (ix1 <= cx_s <= ix2 and iy1 <= cy_s <= iy2):
                    continue
            if abs(pre_scale - 1.0) > 1e-6:
                x1 = x1_s / pre_scale
                y1 = y1_s / pre_scale
                x2 = x2_s / pre_scale
                y2 = y2_s / pre_scale
            else:
                x1, y1, x2, y2 = x1_s, y1_s, x2_s, y2_s
            x1 = max(0.0, min(float(x1), w0))
            y1 = max(0.0, min(float(y1), h0))
            x2 = max(0.0, min(float(x2), w0))
            y2 = max(0.0, min(float(y2), h0))
            if x2 <= x1 or y2 <= y1:
                continue
            full_boxes.append([x1, y1, x2, y2])
            full_scores.append(score)
            full_classes.append(cls_id)

    if full_boxes:
        full_boxes = np.array(full_boxes, dtype=np.float32)
        full_scores = np.array(full_scores, dtype=np.float32)
        full_classes = np.array(full_classes, dtype=np.int32)
    else:
        full_boxes = np.empty((0, 4), dtype=np.float32)
        full_scores = np.empty((0,), dtype=np.float32)
        full_classes = np.empty((0,), dtype=np.int32)

    print(f"[TRAINING-TILER] Raw detections: {len(full_boxes)}")
    if full_boxes.shape[0] > 0:
        keep_idx = _nms_xyxy(full_boxes, full_scores, iou_thr=merge_iou)
        full_boxes = full_boxes[keep_idx]
        full_scores = full_scores[keep_idx]
        full_classes = full_classes[keep_idx]
        print(f"[TRAINING-TILER] After global NMS: {len(full_boxes)}")

        full_boxes, full_scores = merge_overlaps_minmax(full_boxes, full_scores, iou_thresh=merge_iou)
        full_classes = np.zeros(full_boxes.shape[0], dtype=np.int32)
        print(f"[TRAINING-TILER] After minmax merge: {len(full_boxes)}")

        if center_merge_ratio > 0 and full_boxes.shape[0] > 0:
            widths = full_boxes[:, 2] - full_boxes[:, 0]
            heights = full_boxes[:, 3] - full_boxes[:, 1]
            short_sides = np.minimum(widths, heights)
            med_short = float(np.median(short_sides)) if short_sides.size else 0.0
            if med_short > 0:
                radius = float(center_merge_ratio) * med_short
                full_boxes, full_scores = merge_by_center(full_boxes, full_scores, radius)
                full_classes = np.zeros(full_boxes.shape[0], dtype=np.int32)
                print(f"[TRAINING-TILER] After center merge: {len(full_boxes)}")
    else:
        full_boxes = np.empty((0, 4), dtype=np.float32)
        full_scores = np.empty((0,), dtype=np.float32)
        full_classes = np.empty((0,), dtype=np.int32)

    if len(full_boxes) > max_det:
        order = np.argsort(full_scores)[::-1][:max_det]
        full_boxes = full_boxes[order]
        full_scores = full_scores[order]
        full_classes = full_classes[order]

    pred_dir = save_dir / "pred"
    labels_dir = pred_dir / "labels"
    ensure_dir(pred_dir)
    ensure_dir(labels_dir)

    vis = cv2.imread(str(image_path))
    if vis is None:
        vis = np.zeros((h0, w0, 3), dtype=np.uint8)
    thickness = max(1, int(round(line_thickness)))
    text_thickness = max(1, thickness // 2 + 1)
    for (x1, y1, x2, y2), score in zip(full_boxes, full_scores):
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), thickness)
        cv2.putText(
            vis,
            f"{score:.2f}",
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            text_thickness,
            cv2.LINE_AA,
        )
    base_name = Path(image_path).stem
    out_img_path = pred_dir / f"{base_name}.jpg"
    cv2.imwrite(str(out_img_path), vis)

    lines = []
    for (x1, y1, x2, y2), cls_id in zip(full_boxes, full_classes):
        cx = (x1 + x2) / 2.0 / w0
        cy = (y1 + y2) / 2.0 / h0
        bw = (x2 - x1) / w0
        bh = (y2 - y1) / h0
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (labels_dir / f"{base_name}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[TRAINING-TILER] Saved outputs to {pred_dir}")


def merge_by_center(boxes: np.ndarray, scores: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Merge boxes whose centers are within a radius (pixels) using score-weighted averaging."""
    if boxes.size == 0:
        return boxes, scores
    centers = np.stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), axis=1)
    remaining = np.arange(boxes.shape[0])
    merged_b, merged_s = [], []
    while remaining.size > 0:
        i = remaining[0]
        c0 = centers[i]
        d = np.linalg.norm(centers[remaining] - c0, axis=1)
        cluster_mask = d <= radius
        idxs = remaining[cluster_mask]
        w = scores[idxs]
        bb = boxes[idxs]
        x1 = np.average(bb[:, 0], weights=w)
        y1 = np.average(bb[:, 1], weights=w)
        x2 = np.average(bb[:, 2], weights=w)
        y2 = np.average(bb[:, 3], weights=w)
        merged_b.append([x1, y1, x2, y2])
        merged_s.append(float(w.max()))
        remaining = remaining[~cluster_mask]
    return np.array(merged_b, dtype=np.float32), np.array(merged_s, dtype=np.float32)


# make_tiles function now imported from utils.tiling_shared


def _odd(k: int) -> int:
    return int(k) if int(k) % 2 == 1 else int(k) + 1


def compute_content_roi(img: np.ndarray, down: int = 1536, blur_k: int = 7, morph_k: int = 11,
                        min_area_ratio: float = 0.02, margin: float = 0.06) -> Tuple[int, int, int, int]:
    """Compute a content-dense ROI using edge magnitude + Otsu thresholding."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return 0, 0, w, h
    max_side = max(h, w)
    scale = 1.0
    if max_side > down:
        scale = down / float(max_side)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(mag, (_odd(blur_k), _odd(blur_k)), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((_odd(morph_k), _odd(morph_k)), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h
    areas = [cv2.contourArea(c) for c in contours]
    k = int(np.argmax(areas))
    if areas[k] < (min_area_ratio * float(sw * sh)):
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(contours[k])
    mx = int(bw * max(0.0, float(margin)))
    my = int(bh * max(0.0, float(margin)))
    x1s = max(0, x - mx)
    y1s = max(0, y - my)
    x2s = min(sw, x + bw + mx)
    y2s = min(sh, y + bh + my)
    inv = 1.0 / scale if scale > 0 else 1.0
    x1 = int(round(x1s * inv)); y1 = int(round(y1s * inv))
    x2 = int(round(x2s * inv)); y2 = int(round(y2s * inv))
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def content_debug_artifacts(img: np.ndarray, down: int, blur_k: int, morph_k: int,
                            min_area_ratio: float, margin: float):
    """Return debug artifacts for content ROI."""
    h, w = img.shape[:2]
    max_side = max(h, w)
    scale = 1.0
    if max_side > down:
        scale = down / float(max_side)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()
    small_bgr = small.copy()
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(mag, (_odd(blur_k), _odd(blur_k)), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((_odd(morph_k), _odd(morph_k)), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h, small_bgr, mag, th, (0, 0, sw, sh)
    areas = [cv2.contourArea(c) for c in contours]
    k = int(np.argmax(areas))
    x, y, bw, bh = cv2.boundingRect(contours[k])
    x1s = max(0, x - int(bw * max(0.0, float(margin))))
    y1s = max(0, y - int(bh * max(0.0, float(margin))))
    x2s = min(sw, x + bw + int(bw * max(0.0, float(margin))))
    y2s = min(sh, y + bh + int(bh * max(0.0, float(margin))))
    inv = 1.0 / scale if scale > 0 else 1.0
    rx1 = int(round(x1s * inv)); ry1 = int(round(y1s * inv))
    rx2 = int(round(x2s * inv)); ry2 = int(round(y2s * inv))
    rx1 = max(0, min(w, rx1)); rx2 = max(0, min(w, rx2))
    ry1 = max(0, min(h, ry1)); ry2 = max(0, min(h, ry2))
    return rx1, ry1, rx2, ry2, small_bgr, mag, th, (x1s, y1s, x2s, y2s)


def predict_tiled(
    model: YOLO,
    img_path: Path,
    save_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    tile: int,
    overlap: float,
    max_det: int,
    merge_iou: float,
    min_area_ratio: float,
    center_merge_ratio: float,
    min_short_side_ratio: float,
    keep_inner_frac: float = 0.5,
    merge_mode: str = "iou_avg",
    refine_roi: bool = False,
    refine_imagesz: int = 1280,
    refine_margin: float = 0.08,
    refine_iou: float = 0.5,
    draw: bool = True,
    line_thickness: int = 2,
) -> None:
    orig = cv2.imread(str(img_path))
    if orig is None:
        print(f"Warning: failed to read {img_path}")
        return
    full_h, full_w = orig.shape[:2]

    # Optional content-aware crop
    x_off, y_off = 0, 0
    img = orig
    if globals().get("CONTENT_CROP", False):
        if globals().get("CONTENT_DEBUG", False):
            rx1, ry1, rx2, ry2, small_bgr, mag_u8, th_u8, small_roi = content_debug_artifacts(
                orig,
                down=globals().get("CONTENT_DOWN", 1536),
                blur_k=globals().get("CONTENT_BLUR", 7),
                morph_k=globals().get("CONTENT_MORPH", 11),
                min_area_ratio=globals().get("CONTENT_MIN_AREA", 0.02),
                margin=globals().get("CONTENT_MARGIN", 0.06),
            )
            debug_dir = (save_dir / "content_debug"); ensure_dir(debug_dir)
            x1s, y1s, x2s, y2s = small_roi
            small_vis = small_bgr.copy()
            cv2.rectangle(small_vis, (x1s, y1s), (x2s, y2s), (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_small.jpg"), small_vis)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_mag.jpg"), mag_u8)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_th.jpg"), th_u8)
            full_vis = orig.copy()
            cv2.rectangle(full_vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_full_roi.jpg"), full_vis)
        else:
            rx1, ry1, rx2, ry2 = compute_content_roi(
                orig,
                down=globals().get("CONTENT_DOWN", 1536),
                blur_k=globals().get("CONTENT_BLUR", 7),
                morph_k=globals().get("CONTENT_MORPH", 11),
                min_area_ratio=globals().get("CONTENT_MIN_AREA", 0.02),
                margin=globals().get("CONTENT_MARGIN", 0.06),
            )
        if (rx2 - rx1) > 0 and (ry2 - ry1) > 0 and (rx1 != 0 or ry1 != 0 or rx2 != full_w or ry2 != full_h):
            img = orig[ry1:ry2, rx1:rx2]
            x_off, y_off = rx1, ry1

    h, w = img.shape[:2]
    windows = make_tiles(h, w, tile, overlap)

    all_boxes = []
    all_scores = []
    for (x1, y1, x2, y2) in tqdm(windows, desc=f"Tiles {img_path.name}"):
        tile_img = img[y1:y2, x1:x2]
        res = model.predict(source=[tile_img], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                            verbose=False, save=False, augment=False)[0]
        if res.boxes is None or res.boxes.shape[0] == 0:
            continue
        b = res.boxes.xyxy.cpu().numpy()
        s = res.boxes.conf.cpu().numpy()
        # Map to full image coords
        b[:, [0, 2]] += x1
        b[:, [1, 3]] += y1
        # Keep only boxes whose centers lie inside tile inner window (avoid duplicates from overlap)
        ix1, iy1, ix2, iy2 = inner_window(x1, y1, x2, y2, overlap, keep_inner_frac)
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        keep = (cx >= ix1) & (cx <= ix2) & (cy >= iy1) & (cy <= iy2)
        if keep.any():
            all_boxes.append(b[keep])
            all_scores.append(s[keep])

    # Coarse full-image pass to catch misses
    resg = model.predict(source=[img], imgsz=imgsz, conf=max(0.03, conf * 0.7), iou=iou, max_det=max_det,
                         verbose=False, save=False, augment=False)[0]
    if resg.boxes is not None and resg.boxes.shape[0] > 0:
        bg = resg.boxes.xyxy.cpu().numpy()
        sg = resg.boxes.conf.cpu().numpy()
        all_boxes.append(bg)
        all_scores.append(sg)

    out_dir = save_dir / "pred"
    ensure_dir(out_dir)

    # Unique output name to avoid collisions across PDFs
    base_name = img_path.stem
    parts = img_path.parts
    if "pdf_pages" in parts:
        try:
            idx = parts.index("pdf_pages")
            if idx + 1 < len(parts):
                pdf_stem = parts[idx + 1]
                base_name = f"{pdf_stem}_{img_path.stem}"
        except ValueError:
            base_name = img_path.stem

    if not all_boxes:
        cv2.imwrite(str((out_dir / f"{base_name}.jpg")), img)
        (out_dir / f"{base_name}.txt").write_text("")
        return

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    # First pass: NMS
    keep = nms(boxes, scores, iou_thresh=merge_iou)
    boxes = boxes[keep]
    scores = scores[keep]

    # Second pass: IoU cluster-merge (mode)
    mm = (merge_mode or "iou_avg").lower()
    if mm == "minmax":
        boxes, scores = merge_overlaps_minmax(boxes, scores, iou_thresh=merge_iou)
    elif mm == "nms":
        pass
    else:
        boxes, scores = merge_overlaps(boxes, scores, iou_thresh=merge_iou)

    # --- Optional centre-based merge ---
    if boxes.shape[0] > 0 and center_merge_ratio > 0:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        short_sides = np.minimum(widths, heights)
        med_short = float(np.median(short_sides)) if short_sides.size else 0.0
        if med_short > 0:
            radius = float(center_merge_ratio) * med_short
            boxes, scores = merge_by_center(boxes, scores, radius)

    # --- Filters (safe shapes) ---
    n = boxes.shape[0]
    if n == 0:
        final_boxes, final_scores = boxes, scores
    else:
        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas   = widths * heights
        short_sides = np.minimum(widths, heights)

        valid = np.ones(n, dtype=bool)

        if n > 0:
            med_area = float(np.median(areas)) if areas.size else 0.0
            med_short = float(np.median(short_sides)) if short_sides.size else 0.0
            if med_area > 0:
                valid &= (areas >= (med_area * float(min_area_ratio)))
            if med_short > 0:
                valid &= (short_sides >= (med_short * float(min_short_side_ratio)))

        with np.errstate(divide='ignore', invalid='ignore'):
            aspect = np.where(heights > 0, widths / heights, 0)
        amn = float(globals().get("AR_MIN", 0.0))
        amx = float(globals().get("AR_MAX", 9999.0))
        valid &= (aspect >= amn) & (aspect <= amx)

        boxes, scores = boxes[valid], scores[valid]


    # Auto page-adaptive filtering (IQR banding on width, height, area, aspect)
    if globals().get("AUTO_FILTER", False) and boxes.shape[0] > 0:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        with np.errstate(divide='ignore', invalid='ignore'):
            aspect = np.where(heights > 0, widths / heights, 0)
        def iqr_band(x, k):
            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            iqr = q3 - q1
            lo = q1 - k * iqr
            hi = q3 + k * iqr
            return (x >= lo) & (x <= hi)
        k = globals().get("AUTO_K", 1.2)
        mask = iqr_band(widths, k) & iqr_band(heights, k) & iqr_band(areas, k) & iqr_band(aspect, k)
        boxes = boxes[mask]
        scores = scores[mask]

    # Grid-aware deduplication (optional)
    if globals().get("GRID_DEDUP", False) and boxes.shape[0] > 0:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        mw = float(np.median(widths)) if widths.size else 0.0
        mh = float(np.median(heights)) if heights.size else 0.0
        dx = max(1.0, mw * globals().get("GRID_X", 0.5))
        dy = max(1.0, mh * globals().get("GRID_Y", 0.5))
        col_ids = cluster_1d(cx, dx)
        row_ids = cluster_1d(cy, dy)
        keep_mask = np.zeros(boxes.shape[0], dtype=bool)
        best_for_cell = {}
        for i, (r, c) in enumerate(zip(row_ids, col_ids)):
            key = (int(r), int(c))
            sc = float(scores[i])
            if key not in best_for_cell or sc > best_for_cell[key][0]:
                best_for_cell[key] = (sc, i)
        for sc, i in best_for_cell.values():
            keep_mask[i] = True
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]

    # Apply ROI offset back to full image coords if cropped
    if (x_off != 0) or (y_off != 0):
        boxes[:, [0, 2]] += x_off
        boxes[:, [1, 3]] += y_off

    # Draw and save on full image
    overlay = orig.copy()
    color = (255, 0, 0)
    thickness = max(1, int(round(line_thickness)))
    text_thickness = max(1, thickness // 2 + 1)
    for (x1, y1, x2, y2), sc in zip(boxes.astype(int), scores):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        label = f"panel {sc:.2f}"
        ytxt = max(0, y1 - 5)
        cv2.putText(
            overlay,
            label,
            (x1, ytxt),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    out_dir = save_dir / "pred"
    ensure_dir(out_dir)
    cv2.imwrite(str(out_dir / f"{base_name}.jpg"), overlay)

    # Save YOLO txt (bbox-only, normalized cx, cy, w, h) against FULL image size
    lines = []
    for (x1, y1, x2, y2) in boxes:
        cxn = ((x1 + x2) / 2) / full_w
        cyn = ((y1 + y2) / 2) / full_h
        bwn = (x2 - x1) / full_w
        bhn = (y2 - y1) / full_h
        lines.append(f"0 {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f}")
    (out_dir / f"{base_name}.txt").write_text("\n".join(lines))


def run_prediction(
    source_paths: List[Path],
    weights: str,
    imgsz: int,
    conf: float,
    iou: float,
    save_dir: Path,
    augment: bool,
    max_det: int,
    tile_mode: bool,
    tile_size: int,
    tile_overlap: float,
    merge_iou: float,
    min_area_ratio: float,
    center_merge_ratio: float,
    min_short_side_ratio: float,
    keep_inner_frac: float,
    merge_mode: str,
    refine_roi: bool,
    refine_imagesz: int,
    refine_margin: float,
    refine_iou: float,
    line_thickness: int,
):
    ensure_dir(save_dir)
    model = YOLO(weights)

    if tile_mode:
        for p in tqdm(source_paths, desc="Images"):
            predict_tiled(
                model,
                p,
                save_dir,
                imgsz,
                conf,
                iou,
                tile_size,
                tile_overlap,
                max_det,
                merge_iou=merge_iou,
                min_area_ratio=min_area_ratio,
                center_merge_ratio=center_merge_ratio,
                min_short_side_ratio=min_short_side_ratio,
                keep_inner_frac=keep_inner_frac,
                merge_mode=merge_mode,
                refine_roi=refine_roi,
                refine_imagesz=refine_imagesz,
                refine_margin=refine_margin,
                refine_iou=refine_iou,
                line_thickness=line_thickness,
            )
        print(f"\nDone. Tiled predictions saved under: {save_dir / 'pred'}")
        return

    project = str(save_dir)
    name = "pred"
    model.predict(
        source=[str(p) for p in source_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=True,
        project=project,
        name=name,
        exist_ok=True,
        augment=augment,
        max_det=max_det,
        verbose=True,
    )
    print(f"\nDone. Predictions saved under: {save_dir / name}")


def main():
    ap = argparse.ArgumentParser(description="Predict panels on PDFs, images, or folders.")
    ap.add_argument("--input", required=True, help="Path to PDF, image, or directory of images/PDFs")
    ap.add_argument("--weights", default=find_default_weights(), help="Path to YOLO weights (.pt)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    ap.add_argument("--augment", action="store_true", help="Enable TTA for recall (non-tiled mode)")
    ap.add_argument("--max-det", type=int, default=CFG.max_det)

    # Align with tiler defaults
    ap.add_argument("--tile", action="store_true", help="Enable tiled inference for large pages")
    ap.add_argument("--tile-size", type=int, default=CFG.tile_size)
    ap.add_argument("--tile-overlap", type=float, default=CFG.overlap, help="Tile overlap fraction 0..0.5")
    ap.add_argument("--merge-iou", type=float, default=0.50, help="IoU for cross-tile de-duplication")

    # Post-merge filters
    ap.add_argument("--min-area-ratio", type=float, default=CFG.min_area_ratio, help="Drop boxes smaller than this × median area")
    ap.add_argument("--center-merge-ratio", type=float, default=0.5, help="Merge centers within this × median size (pixels)")
    ap.add_argument("--min-short-side-ratio", type=float, default=CFG.min_short_side_ratio, help="Drop if min(w,h) < this × median short side")
    ap.add_argument("--area-low", type=float, default=None, help="Keep if area >= this × median area (after merges)")
    ap.add_argument("--area-high", type=float, default=None, help="Keep if area <= this × median area (after merges)")
    ap.add_argument("--ar-min", type=float, default=CFG.ar_min, help="Keep if aspect ratio >= this (w/h)")
    ap.add_argument("--ar-max", type=float, default=CFG.ar_max, help="Keep if aspect ratio <= this (w/h)")
    ap.add_argument("--iqr", action="store_true", default=False, help="Enable IQR-based width/height filtering")
    ap.add_argument("--iqr-k", type=float, default=1.3, help="IQR multiplier")
    ap.add_argument("--keep-inner-frac", type=float, default=CFG.keep_inner_frac, help="Keep centers inside inner window reduced by overlap*frac")
    ap.add_argument("--merge-mode", type=str, default=CFG.merge_mode, choices=["iou_avg", "minmax", "nms"], help="Box merge strategy after NMS")

    # Optional ROI refinement
    ap.add_argument("--refine-roi", action="store_true", default=CFG.refine_enabled, help="Enable ROI refinement per detection")
    ap.add_argument("--refine-imgsz", type=int, default=CFG.refine_imgsz, help="Refinement ROI inference size")
    ap.add_argument("--refine-margin", type=float, default=CFG.refine_margin, help="Padding margin fraction around each box for ROI")
    ap.add_argument("--refine-iou", type=float, default=CFG.refine_iou, help="IOU for refinement pass")

    # Content-aware crop options
    ap.add_argument("--content-crop", action="store_true", help="Enable content-aware pre-crop before tiling")
    ap.add_argument("--content-down", type=int, default=1536, help="Downscale max side before heatmap")
    ap.add_argument("--content-blur", type=int, default=7, help="Gaussian blur kernel for heatmap")
    ap.add_argument("--content-morph", type=int, default=11, help="Morph kernel for closing/opening")
    ap.add_argument("--content-min-area", type=float, default=0.02, help="Min ROI area as fraction of page")
    ap.add_argument("--content-margin", type=float, default=0.06, help="Margin to expand ROI bbox")
    ap.add_argument("--content-debug", action="store_true", help="Save heatmap/threshold and ROI previews")

    ap.add_argument("--tag", type=str, default=None, help="Suffix to separate outputs (e.g., 'auto1')")
    ap.add_argument("--out", default=None, help="Output directory (defaults inside the input folder)")
    ap.add_argument("--use-training-tiler", action="store_true", help="Use training-style tiler (make_tiles + padding)")
    ap.add_argument("--line-thickness", type=float, default=2.0, help="Line thickness (pixels) for saved overlays")
    # Optional CSV eval logging
    ap.add_argument("--excel", action="store_true", help="If set, append a CSV row with summary counts for this run.")
    ap.add_argument("--gt-set", type=str, default="val", help="Label for this GT set in the CSV (e.g., 'val', 'stress', 'curated').")
    ap.add_argument("--excel-path", type=str, default=None, help="Optional explicit CSV path; defaults to <save_dir>/eval_summary.csv.")
    args = ap.parse_args()

    inp = Path(args.input)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out:
        save_root = Path(args.out)
    else:
        if inp.is_dir():
            save_root = inp / f"pred_{timestamp}"
        elif inp.is_file():
            save_root = inp.parent / f"{inp.stem}_pred_{timestamp}"
        else:
            raise FileNotFoundError(f"Input not found: {inp}")
    ensure_dir(save_root)

    line_thickness = max(1, int(round(args.line_thickness)))

    # Expose band-pass as module globals for simple access inside helper
    global AREA_LOW, AREA_HIGH, AR_MIN, AR_MAX
    AREA_LOW = args.area_low
    AREA_HIGH = args.area_high
    AR_MIN = args.ar_min
    AR_MAX = args.ar_max
    global IQR_ON, IQR_K
    IQR_ON = bool(args.iqr)
    IQR_K = args.iqr_k
    global GRID_DEDUP
    GRID_DEDUP = False  # off unless explicitly used below
    global GRID_X, GRID_Y
    GRID_X = 0.5
    GRID_Y = 0.5

    # Auto mode hook removed; keep explicit flags for reproducibility

    # Content crop globals
    global CONTENT_CROP, CONTENT_DOWN, CONTENT_BLUR, CONTENT_MORPH, CONTENT_MIN_AREA, CONTENT_MARGIN
    CONTENT_CROP = bool(args.content_crop)
    CONTENT_DOWN = args.content_down
    CONTENT_BLUR = args.content_blur
    CONTENT_MORPH = args.content_morph
    CONTENT_MIN_AREA = args.content_min_area
    CONTENT_MARGIN = args.content_margin
    global CONTENT_DEBUG
    CONTENT_DEBUG = bool(args.content_debug)

    sources: List[Path] = []
    pdfs_found = False

    if inp.is_file():
        if inp.suffix.lower() == ".pdf":
            pdfs_found = True
            pdf_imgs_dir = save_root / "pdf_pages"
            print(f"Converting PDF to images @ {args.dpi} DPI...")
            sources = pdf_to_images(inp, pdf_imgs_dir, dpi=args.dpi)
        else:
            sources = [inp]
    elif inp.is_dir():
        sources = list_images_in_dir(inp)
        pdfs = list_pdfs_in_dir(inp)
        if pdfs:
            pdfs_found = True
            print(f"Found {len(pdfs)} PDF(s). Converting to images @ {args.dpi} DPI...")
            for pdf in pdfs:
                pdf_out_dir = save_root / "pdf_pages" / pdf.stem
                pdf_imgs = pdf_to_images(pdf, pdf_out_dir, dpi=args.dpi)
                sources.extend(pdf_imgs)
    else:
        raise FileNotFoundError(f"Input not found: {inp}")

    if not sources:
        print("No images found to process.")
        return

    auto_tile = args.tile or pdfs_found
    if pdfs_found and not args.tile:
        print("Auto-enabled tiled inference for PDFs.")

    final_save_dir = save_root if not args.tag else (save_root / f"run_{args.tag}")
    if args.use_training_tiler:
        cfg = load_tiling_config()
        model = YOLO(args.weights)
        print(f"Processing {len(sources)} image(s) with training-style tiler.")
        for idx, src in enumerate(sources, 1):
            print(f"\n[{idx}/{len(sources)}] {Path(src).name}")
            infer_with_training_tiler(
                image_path=Path(src),
                model=model,
                cfg=cfg,
                save_dir=final_save_dir,
                conf=args.conf,
                iou=args.iou,
                merge_iou=args.merge_iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                center_merge_ratio=args.center_merge_ratio,
                keep_inner_frac=args.keep_inner_frac,
                line_thickness=line_thickness,
            )
        return

    print(f"Processing {len(sources)} image(s) with weights: {args.weights}")
    run_prediction(
        source_paths=sources,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_dir=final_save_dir,
        augment=args.augment,
        max_det=args.max_det,
        tile_mode=auto_tile,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        merge_iou=args.merge_iou,
        min_area_ratio=args.min_area_ratio,
        center_merge_ratio=args.center_merge_ratio,
        min_short_side_ratio=args.min_short_side_ratio,
        keep_inner_frac=args.keep_inner_frac,
        merge_mode=args.merge_mode,
        refine_roi=bool(args.refine_roi),
        refine_imagesz=args.refine_imgsz,
        refine_margin=args.refine_margin,
        refine_iou=args.refine_iou,
        line_thickness=line_thickness,
    )

    # ---------------- Optional one-line CSV summary ----------------
    if args.excel:
        # Determine prediction labels directory
        pred_dir_primary = final_save_dir / "pred" / "labels"
        pred_dir_alt = final_save_dir / "pred"
        pred_labels_dir = pred_dir_primary if pred_dir_primary.exists() else pred_dir_alt

        # Derive labels_root by swapping 'images' with 'labels' in the input path
        images_root = Path(args.input)
        if images_root.is_dir():
            parts = list(images_root.parts)
            try:
                idx = parts.index("images")
            except ValueError:
                raise RuntimeError(f"Cannot derive labels path from input {images_root}")
            parts[idx] = "labels"
            labels_root = Path(*parts)
        else:
            parts = list(images_root.parts)
            try:
                i = parts.index("images")
            except ValueError:
                raise RuntimeError(f"Input path does not contain an 'images' segment: {images_root}")
            labels_root = Path(*parts[:i], "labels", *parts[i+1:-1])

        def _load_gt_boxes_dict(path: Path):
            if not path.exists():
                return []
            out = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                        except Exception:
                            continue
                        if cls_id != 0:
                            continue
                        cx, cy, w, h = map(float, parts[1:5])
                        out.append({"x_center": cx, "y_center": cy, "width": w, "height": h})
            return out

        def _load_pred_boxes_dict(path: Path):
            if not path.exists():
                return []
            out = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                        except Exception:
                            continue
                        if cls_id != 0:
                            continue
                        cx, cy, w, h = map(float, parts[1:5])
                        out.append({"x_center": cx, "y_center": cy, "width": w, "height": h})
            return out

        total_gt = 0
        total_tp = 0
        total_fp = 0
        num_images = 0
        for pred_file in sorted(pred_labels_dir.glob("*.txt")):
            img_id = pred_file.stem
            gt_file = labels_root / f"{img_id}.txt"
            preds = _load_pred_boxes_dict(pred_file)
            gts = _load_gt_boxes_dict(gt_file)
            # Run shared matcher
            tp, fp, fn = match_and_count(preds, gts, float(args.iou))
            total_tp += tp
            total_fp += fp
            total_gt += (tp + fn)
            num_images += 1

        csv_path = Path(args.excel_path) if args.excel_path else (final_save_dir / "eval_summary.csv")
        row = {
            "gt_set": args.gt_set,
            "resolution": int(args.imgsz),
            "total_gt": int(total_gt),
            "matched_gt_0_4": int(total_tp),
            "total_fp": int(total_fp),
            "num_images": int(num_images),
        }
        append_eval_row(csv_path, row)
        print(f"[excel] Appended evaluation row to {csv_path}")


if __name__ == "__main__":
    main()
