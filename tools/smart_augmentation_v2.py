#!/usr/bin/env python3
import os, random, argparse, shutil, sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Add root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from shapely.geometry import Polygon, box as shp_box
except ModuleNotFoundError:
    raise SystemExit("Shapely not installed. Run: conda.exe run -n rhino_mcp pip install shapely")

# Load shared augmentation config
from utils.augmentation_shared import load_augmentation_config, load_augmentation_preset
CFG = load_augmentation_config()

# --------------------------- Helpers ---------------------------------
def cleanup_and_make(dir_path: Path, keep_existing: bool = False):
    """Recreate output directory and remove all existing contents (files and folders)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    if keep_existing:
        return
    for p in dir_path.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception:
            # best-effort cleanup; continue even if a file is locked
            pass

def parse_yolo_seg(label_path: Path):
    anns = []
    if not label_path.exists():
        return anns
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return anns
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
        except Exception:
            continue
        poly = []
        if len(parts) > 5:
            try:
                poly = [float(v) for v in parts[5:]]
            except Exception:
                poly = []
        anns.append({"cls": cls, "bbox": (cx, cy, w, h), "poly": poly})
    return anns

def bbox_to_poly_abs(bbox_norm, W, H):
    cx, cy, w, h = bbox_norm
    x, y, ww, hh = (cx - w/2) * W, (cy - h/2) * H, w * W, h * H
    return np.array([[x, y], [x+ww, y], [x+ww, y+hh], [x, y+hh]], dtype=np.float32)

def poly_norm_to_abs(poly_norm, W, H):
    if not poly_norm:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(poly_norm, dtype=np.float32)
    if arr.size % 2 != 0:
        return np.empty((0, 2), dtype=np.float32)
    return arr.reshape(-1, 2) * np.array([W, H], dtype=np.float32)

def clamp_and_clip_poly(points_abs, W, H, clip_quad=None):
    if points_abs is None or len(points_abs) < 3:
        return np.empty((0, 2), dtype=np.float32)

    poly = Polygon(points_abs).buffer(0)  # clean
    if poly.is_empty:
        return np.empty((0, 2), dtype=np.float32)

    clip_region = shp_box(0.0, 0.0, float(W), float(H))
    if clip_quad is not None and len(clip_quad) == 4:
        content_poly = Polygon(clip_quad).buffer(0)
        if content_poly.is_valid and not content_poly.is_empty:
            clip_region = clip_region.intersection(content_poly)

    # robust intersect with tiny buffer to avoid degeneracies
    inter = poly.intersection(clip_region)
    if inter.is_empty or inter.geom_type not in ("Polygon", "MultiPolygon"):
        inter = poly.buffer(1e-6).intersection(clip_region.buffer(1e-6))

    if inter.is_empty:
        return np.empty((0, 2), dtype=np.float32)

    if inter.geom_type == "MultiPolygon":
        inter = max(list(inter.geoms), key=lambda g: g.area)

    if inter.geom_type != "Polygon" or inter.is_empty:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(inter.exterior.coords, dtype=np.float32)[:-1]

def _clamp01(v: float) -> float:
    v = float(v)
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

def write_label(out_path: Path, items, W, H, tag_class_id: int):
    lines = []
    for it in items:
        if not it.get("bbox_abs"):
            continue
        cls = int(it["cls"])
        cx, cy, w, h = it["bbox_abs"]
        # clamp numerics
        cx = _clamp01(cx); cy = _clamp01(cy); w = _clamp01(w); h = _clamp01(h)
        line = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

        # if this class is NOT the tag class, include polygon if available
        if cls != tag_class_id:
            poly_abs = it.get("poly_abs", np.empty((0, 2), dtype=np.float32))
            if poly_abs.size >= 6:
                poly_norm = (poly_abs / np.array([W, H], dtype=np.float32)).reshape(-1)
                # clamp polygon coords
                poly_norm = np.clip(poly_norm, 0.0, 1.0)
                line += " " + " ".join(f"{v:.6f}" for v in poly_norm)
        lines.append(line)
    out_path.write_text("\n".join(lines), encoding="utf-8")

# ----------------------- Augmentation core ----------------------------
def augment_one(
    img, anns, target=None, pad_margin=None, rot_deg=None,
    scale_low=None, scale_high=None, trans_frac=None, tag_class_id: int = None
):
    # Use config defaults if not provided
    target = target or CFG.target_size
    pad_margin = pad_margin or CFG.pad_margin
    rot_deg = rot_deg or CFG.rotation_deg
    scale_low = scale_low or CFG.scale_low
    scale_high = scale_high or CFG.scale_high
    trans_frac = trans_frac or CFG.translation_frac
    tag_class_id = tag_class_id or CFG.tag_class_id
    H0, W0 = img.shape[:2]
    s_fit = min(target / H0, target / W0)
    newW, newH = int(W0 * s_fit), int(H0 * s_fit)
    img_rs = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)

    Wc = Hc = target + 2 * pad_margin
    canvas = np.full((Hc, Wc, 3), 255, np.uint8)
    x0 = pad_margin + (target - newW) // 2
    y0 = pad_margin + (target - newH) // 2
    canvas[y0:y0+newH, x0:x0+newW] = img_rs

    items = []
    for a in anns:
        # prefer polygon if present; otherwise build rect from bbox
        src_pts = (
            poly_norm_to_abs(a["poly"], W0, H0)
            if a["poly"] else
            bbox_to_poly_abs(a["bbox"], W0, H0)
        )
        items.append({"cls": a["cls"], "poly_abs": src_pts * s_fit + np.array([x0, y0], dtype=np.float32)})

    angle = random.uniform(-rot_deg, rot_deg)
    s_rand = random.uniform(scale_low, scale_high)
    M = cv2.getRotationMatrix2D((Wc/2, Hc/2), angle, s_rand)
    tmax = trans_frac * target
    M[:, 2] += [random.uniform(-tmax, tmax), random.uniform(-tmax, tmax)]

    xC, yC = (Wc - target) // 2, (Hc - target) // 2
    T_crop = np.array([[1.0, 0.0, -xC], [0.0, 1.0, -yC]], dtype=np.float32)
    A_full = T_crop @ np.vstack([M, [0, 0, 1]])
    A_2x3 = A_full[:2, :]

    img_out = cv2.warpAffine(canvas, A_2x3, (target, target), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

    pasted_corners = np.array([[x0, y0], [x0+newW, y0], [x0+newW, y0+newH], [x0, y0+newH]], dtype=np.float32)
    pasted_corners_h = np.hstack([pasted_corners, np.ones((4, 1), dtype=np.float32)])
    content_quad_tf = (pasted_corners_h @ A_full.T)[:, :2]

    min_area_px = 4.0  # keep tiny but non-zero
    kept = []
    for it in items:
        pts_h = np.hstack([it["poly_abs"], np.ones((it["poly_abs"].shape[0], 1), dtype=np.float32)])
        pts_tf = (pts_h @ A_full.T)[:, :2]

        # clipped polygon for panels; tags may degenerate
        poly_clip = clamp_and_clip_poly(pts_tf, target, target, clip_quad=content_quad_tf)
        it["poly_abs"] = poly_clip

        # compute bbox from transformed corners; then clip to image bounds
        xmin, ymin = np.clip(pts_tf[:,0].min(), 0, target), np.clip(pts_tf[:,1].min(), 0, target)
        xmax, ymax = np.clip(pts_tf[:,0].max(), 0, target), np.clip(pts_tf[:,1].max(), 0, target)
        bw, bh = max(0.0, xmax - xmin), max(0.0, ymax - ymin)

        if it["cls"] == tag_class_id:
            # tag: keep bbox even if polygon is empty (as long as area > min)
            if bw * bh >= min_area_px:
                it["bbox_abs"] = ((xmin+xmax)/2/target, (ymin+ymax)/2/target, bw/target, bh/target)
                kept.append(it)
        else:
            # panel: require a valid polygon; salvage with bbox only if polygon exists and non-trivial
            if poly_clip.size >= 6:
                pxmin, pymin = poly_clip[:,0].min(), poly_clip[:,1].min()
                pxmax, pymax = poly_clip[:,0].max(), poly_clip[:,1].max()
                pbw, pbh = pxmax - pxmin, pymax - pymin
                if pbw * pbh >= min_area_px:
                    it["bbox_abs"] = ((pxmin+pxmax)/2/target, (pymin+pymax)/2/target, pbw/target, pbh/target)
                    kept.append(it)

    items = kept
    # final filter: ensure bbox positivity
    items = [it for it in items if it.get("bbox_abs") and it["bbox_abs"][2] > 0 and it["bbox_abs"][3] > 0]
    return img_out, items

# ----------------------- Split processing ----------------------------
def process_split(
    split, base_dir: Path, target=None, augs_per=None, in_folder: str | None = None,
    out_folder: str = None, jpeg_quality: int = None, tag_class_id: int = None, keep_existing: bool = None,
    max_images: int | None = None, rotation_deg: float = None, scale_low: float = None, 
    scale_high: float = None, translation_frac: float = None, pad_margin: int = None
):
    # Use config defaults if not provided
    target = target or CFG.target_size
    augs_per = augs_per or CFG.augs_per_image
    out_folder = out_folder or CFG.out_folder
    jpeg_quality = jpeg_quality or CFG.jpeg_quality
    tag_class_id = tag_class_id or CFG.tag_class_id
    keep_existing = keep_existing if keep_existing is not None else CFG.keep_existing
    max_images = max_images or CFG.max_images
    # Resolve sources
    if in_folder:
        img_in = base_dir/"images"/split/in_folder
        lbl_in = base_dir/"labels"/split/in_folder
        if not img_in.exists() or not lbl_in.exists():
            print(f"[skip] {split}: missing {img_in} or {lbl_in}")
            return 0
    else:
        # Fallback order
        candidates = [
            ("gtcenter1k_11_22_33", "gtcenter1k_11_22_33"),
            ("gtcenter1k", "gtcenter1k"),
            ("tiled1k", "tiled1k"),
            ("cropped1k", "cropped1k"),
        ]
        img_in = lbl_in = None
        for a, b in candidates:
            _img = base_dir/"images"/split/a
            _lbl = base_dir/"labels"/split/b
            if _img.exists() and _lbl.exists():
                img_in, lbl_in = _img, _lbl
                break
        if img_in is None:
            print(f"[skip] {split}: missing all of gtcenter1k_11_22_33/gtcenter1k/tiled1k/cropped1k")
            return 0

    img_out = base_dir/"images"/split/out_folder
    lbl_out = base_dir/"labels"/split/out_folder

    cleanup_and_make(img_out, keep_existing=keep_existing)
    cleanup_and_make(lbl_out, keep_existing=keep_existing)

    # Collect files
    files = sorted(
        list(img_in.glob("*.jpg")) + list(img_in.glob("*.jpeg")) + list(img_in.glob("*.png")) +
        list(img_in.glob("*.JPG")) + list(img_in.glob("*.JPEG")) + list(img_in.glob("*.PNG"))
    )
    if max_images is not None:
        files = files[:max_images]
    if not files:
        print(f"[skip] {split}: no images in {img_in}")
        return 0

    count = 0
    kept_src = 0
    for fn in tqdm(files, desc=f"Augmenting {split}", unit="img"):
        img = cv2.imread(str(fn))
        anns = parse_yolo_seg(lbl_in / f"{fn.stem}.txt")
        if img is None or not anns:
            continue

        # Measure “tag density” to boost augs for tag-heavy samples
        tag_count = sum(1 for a in anns if int(a.get("cls", -1)) == tag_class_id)
        local_augs = augs_per + (1 if tag_count >= 2 else 0) + (2 if tag_count >= 4 else 0)

        for j in range(local_augs):
            out_img, out_items = augment_one(
                img, anns, 
                target=target, 
                pad_margin=pad_margin or CFG.pad_margin,
                rot_deg=rotation_deg or CFG.rotation_deg,
                scale_low=scale_low or CFG.scale_low,
                scale_high=scale_high or CFG.scale_high,
                trans_frac=translation_frac or CFG.translation_frac,
                tag_class_id=tag_class_id
            )
            out_name = f"{fn.stem}_aug{j:02d}"
            cv2.imwrite(str(img_out / f"{out_name}.jpg"), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            write_label(lbl_out / f"{out_name}.txt", out_items, target, target, tag_class_id)
            count += 1
        kept_src += 1

    print(f"[summary] {split}: source images used = {kept_src}, augmented outputs = {count}")
    return count

# ----------------------------- Main ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Smart augmentation for RCP-Dual-Seg")
    # Core arguments (use config defaults)
    ap.add_argument("--base-dir", default="datasets/rcp_dual_seg_v2")
    ap.add_argument("--target-size", type=int, default=CFG.target_size)
    ap.add_argument("--augs-per-image", type=int, default=CFG.augs_per_image)
    ap.add_argument("--splits", type=str, default=CFG.splits, help="Comma-separated splits to process")
    ap.add_argument("--in-folder", type=str, default=None, help="Explicit source folder under images/labels split (e.g., gtcenter1k_11_22_33)")
    ap.add_argument("--out-folder", type=str, default=CFG.out_folder, help="Output folder under images/labels split")
    
    # Augmentation parameters (use config defaults)
    ap.add_argument("--rotation-deg", type=float, default=CFG.rotation_deg, help="Rotation range in degrees")
    ap.add_argument("--scale-low", type=float, default=CFG.scale_low, help="Minimum scale factor")
    ap.add_argument("--scale-high", type=float, default=CFG.scale_high, help="Maximum scale factor")
    ap.add_argument("--translation-frac", type=float, default=CFG.translation_frac, help="Translation as fraction of target size")
    ap.add_argument("--pad-margin", type=int, default=CFG.pad_margin, help="Padding margin around images")
    
    # Other parameters (use config defaults)
    ap.add_argument("--tag-class-id", type=int, default=CFG.tag_class_id, help="Class id that should be treated as bbox-only")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--jpeg-quality", type=int, default=CFG.jpeg_quality, help="JPEG quality for output images")
    ap.add_argument("--opencv-num-threads", type=int, default=CFG.opencv_num_threads, help="Set OpenCV threads to avoid OMP duplicate runtime issues")
    ap.add_argument("--keep-existing", action="store_true", default=CFG.keep_existing, help="Do not clear out-folder; append to it")
    ap.add_argument("--max-images", type=int, default=CFG.max_images, help="Limit number of source images per split (debugging)")
    
    # Preset support
    ap.add_argument("--preset", type=str, choices=list(CFG.presets.keys()), help="Use a predefined preset")

    args = ap.parse_args()

    # Handle preset override
    if args.preset:
        preset_config = load_augmentation_preset(args.preset)
        # Override args with preset values
        args.target_size = preset_config.target_size
        args.augs_per_image = preset_config.augs_per_image
        args.rotation_deg = preset_config.rotation_deg
        args.scale_low = preset_config.scale_low
        args.scale_high = preset_config.scale_high
        args.translation_frac = preset_config.translation_frac
        args.pad_margin = preset_config.pad_margin
        args.tag_class_id = preset_config.tag_class_id
        args.jpeg_quality = preset_config.jpeg_quality
        args.opencv_num_threads = preset_config.opencv_num_threads
        args.keep_existing = preset_config.keep_existing
        args.max_images = preset_config.max_images
        print(f"Using preset: {args.preset}")

    # Apply OpenCV threading hint (helps on Windows/Anaconda)
    try:
        cv2.setNumThreads(int(args.opencv_num_threads))
    except Exception:
        pass

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(getattr(args, "base-dir", args.base_dir))
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    print("=== Smart Augmentation v2 ===")
    print(f"Base dir        : {base_dir}")
    print(f"Target size     : {args.target_size}")
    print(f"Augs per image  : {args.augs_per_image}")
    print(f"Splits          : {args.splits}")
    print(f"In  folder      : {args.in_folder or '(auto)'}")
    print(f"Out folder      : {args.out_folder}")
    print(f"Tag class id    : {args.tag_class_id}")
    print(f"Seed            : {args.seed}")
    print(f"JPEG quality    : {args.jpeg_quality}")
    print(f"OpenCV threads  : {args.opencv_num_threads}")
    print(f"Keep existing   : {'yes' if args.keep_existing else 'no'}")
    print(f"Max images      : {args.max_images if args.max_images is not None else '(all)'}")
    print()

    totals = {}
    for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
        print(f"Augmenting {split}...")
        totals[split] = process_split(
            split,
            base_dir,
            target=args.target_size,
            augs_per=args.augs_per_image,
            in_folder=args.in_folder,
            out_folder=args.out_folder,
            jpeg_quality=args.jpeg_quality,
            tag_class_id=args.tag_class_id,
            keep_existing=args.keep_existing,
            max_images=args.max_images,
            # Pass augmentation parameters
            rotation_deg=args.rotation_deg,
            scale_low=args.scale_low,
            scale_high=args.scale_high,
            translation_frac=args.translation_frac,
            pad_margin=args.pad_margin,
        )
        print(f"[done] {split}: {totals[split]} augmented samples")

    print("=== Summary ===")
    for k, v in totals.items():
        print(f"{k:>5}: {v} outputs")
    print("Done.")

if __name__ == "__main__":
    main()
