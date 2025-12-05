#!/usr/bin/env python3
"""
panel_cropper.py
----------------
Panel-centric cropping: Use existing panel coordinates to create focused training data.

Strategy:
1. Extract panel bbox from polygon coordinates
2. Pad by 15% for context
3. Clamp to image bounds
4. Resize to 1024px (longest side)
5. Rewrite labels for the crop

This creates much more efficient training data than random tiling.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
from tqdm import tqdm
from shapely.geometry import Polygon, box

def cleanup_old_crops(crop_dir):
    """Clean up old crop directory"""
    if os.path.exists(crop_dir):
        for file in os.listdir(crop_dir):
            os.remove(os.path.join(crop_dir, file))
    else:
        os.makedirs(crop_dir, exist_ok=True)

def parse_yolo_seg_label(label_path):
    """Parse YOLO/YOLO-Seg labels. Accepts bbox-only (>=5) and optional polygon (>=7)."""
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # need at least class + bbox
            try:
                cls = int(float(parts[0]))
            except Exception:
                continue
            if cls not in {0, 1}:
                continue

            cx, cy, w, h = map(float, parts[1:5])
            pts = [float(v) for v in parts[5:]] if len(parts) > 5 else []
            objects.append({
                'class_id': cls,
                'bbox': (cx, cy, w, h),
                'polygon': pts  # may be empty for tags
            })
    return objects

def polygon_to_bbox(points, img_w, img_h):
    """Convert polygon points to absolute bbox coordinates"""
    abs_pts = [(points[i] * img_w, points[i+1] * img_h) for i in range(0, len(points), 2)]
    if not abs_pts:
        return 0, 0, 0, 0
    xs, ys = zip(*abs_pts)
    return min(xs), min(ys), max(xs), max(ys)

def _bbox_to_abs_poly(bbox_norm, img_w, img_h):
    cx, cy, w, h = bbox_norm
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def transform_labels_for_crop(all_objects, crop_x, crop_y, crop_w, crop_h, scale, x_offset, y_offset,
                              target_size, img_w, img_h):
    """
    Transforms ALL object labels for a new crop view.
    - Clips polygons/bboxes against the crop boundaries.
    - Recalculates coordinates to be relative to the new cropped image.
    - Normalizes coordinates for the final target size.
    - Returns a list of valid YOLO label strings for the new crop.
    """
    new_labels = []
    crop_rect_geom = box(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)

    for obj in all_objects:
        cls = obj['class_id']
        
        # Determine the geometry of the object (polygon for panels, bbox for tags)
        is_panel_with_poly = cls == 0 and obj.get('polygon') and len(obj['polygon']) >= 6
        if is_panel_with_poly:
            abs_pts = [(obj['polygon'][i] * img_w, obj['polygon'][i+1] * img_h)
                       for i in range(0, len(obj['polygon']), 2)]
        else:
            abs_pts = _bbox_to_abs_poly(obj['bbox'], img_w, img_h)

        if len(abs_pts) < 3:
            continue

        try:
            geom = Polygon(abs_pts).buffer(0)
        except Exception:
            continue # Skip invalid geometries

        if not geom.intersects(crop_rect_geom):
            continue

        clipped_geom = geom.intersection(crop_rect_geom)
        if clipped_geom.is_empty:
            continue
            
        # Handle cases where clipping results in multiple small polygons
        if clipped_geom.geom_type == 'MultiPolygon':
            clipped_geom = max(clipped_geom.geoms, key=lambda p: p.area)
        
        if clipped_geom.geom_type != 'Polygon' or clipped_geom.is_empty or clipped_geom.area < 1.0:
            continue

        # --- Coordinate Transformation ---
        # 1. Get clipped coordinates in original image space
        # 2. Translate to be relative to the crop's top-left corner (crop_x, crop_y)
        # 3. Scale to the new resized dimensions (new_w, new_h)
        # 4. Pad/offset to center in the final target_size canvas
        # 5. Normalize to 0-1 for YOLO format
        
        coords = list(clipped_geom.exterior.coords)
        
        # Lists to hold final normalized coordinates
        xs_norm, ys_norm = [], []

        for x_abs, y_abs in coords:
            # Step 2 & 3: Translate and scale
            x_in_resized_crop = (x_abs - crop_x) * scale
            y_in_resized_crop = (y_abs - crop_y) * scale

            # Step 4: Add padding offset
            x_in_padded_canvas = x_in_resized_crop + x_offset
            y_in_padded_canvas = y_in_resized_crop + y_offset

            # Step 5: Normalize and clamp to [0,1]
            x_norm = x_in_padded_canvas / target_size
            y_norm = y_in_padded_canvas / target_size
            # Clamp vertices defensively to avoid tiny epsilon overflow/underflow
            x_norm = 0.0 if x_norm < 0.0 else (1.0 if x_norm > 1.0 else x_norm)
            y_norm = 0.0 if y_norm < 0.0 else (1.0 if y_norm > 1.0 else y_norm)
            xs_norm.append(x_norm)
            ys_norm.append(y_norm)

        # Create bounding box from the new clipped & transformed polygon
        min_x, max_x = min(xs_norm), max(xs_norm)
        min_y, max_y = min(ys_norm), max(ys_norm)

        # Clamp coordinates to be within [0.0, 1.0] to avoid issues
        min_x, max_x = max(0.0, min_x), min(1.0, max_x)
        min_y, max_y = max(0.0, min_y), min(1.0, max_y)
        
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        bw = max_x - min_x
        bh = max_y - min_y
        
        # Final check for degenerate boxes after clipping
        if bw < 1e-4 or bh < 1e-4:
            continue

        # Format the final YOLO string
        if cls == 0: # Panel (segmentation)
            # Ensure bbox values are clamped as well for extreme edge cases
            cx = 0.0 if cx < 0.0 else (1.0 if cx > 1.0 else cx)
            cy = 0.0 if cy < 0.0 else (1.0 if cy > 1.0 else cy)
            bw = 0.0 if bw < 0.0 else (1.0 if bw > 1.0 else bw)
            bh = 0.0 if bh < 0.0 else (1.0 if bh > 1.0 else bh)
            poly_points_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in zip(xs_norm, ys_norm)])
            new_labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {poly_points_str}")
        else: # Tag (bounding box only)
            # Clamp bbox derived from clipped geometry to ensure strict bounds
            cx = 0.0 if cx < 0.0 else (1.0 if cx > 1.0 else cx)
            cy = 0.0 if cy < 0.0 else (1.0 if cy > 1.0 else cy)
            bw = 0.0 if bw < 0.0 else (1.0 if bw > 1.0 else bw)
            bh = 0.0 if bh < 0.0 else (1.0 if bh > 1.0 else bh)
            new_labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return new_labels


def create_panel_crop(image_path, label_path, target_size=1024, padding_ratio=0.15, per_tag_extras: int = 0):
    """Create a crop centered on a panel with padding"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    num_channels = img.shape[2] if len(img.shape) > 2 else 1

    objects = parse_yolo_seg_label(label_path)
    if not objects:
        return []
    
    crops = []
    panel_objects = [p for p in objects if p['class_id'] == 0]
    
    for i, panel in enumerate(panel_objects):
        # --- Robust Cropping Logic ---
        # 1. Get the bounding box of the main panel
        xmin, ymin, xmax, ymax = polygon_to_bbox(panel['polygon'], w, h)
        panel_w, panel_h = xmax - xmin, ymax - ymin

        if panel_w < 1 or panel_h < 1:
            continue

        # 2. Define the initial crop box based on the panel and padding
        pad_w = panel_w * padding_ratio
        pad_h = panel_h * padding_ratio
        
        crop_x1 = xmin - pad_w
        crop_y1 = ymin - pad_h
        crop_x2 = xmax + pad_w
        crop_y2 = ymax + pad_h

        # 3. Expand the crop box to include any tags that are clearly inside this panel.
        try:
            panel_abs_pts = [(panel['polygon'][k] * w, panel['polygon'][k + 1] * h) for k in range(0, len(panel['polygon']), 2)]
            panel_poly_geom = Polygon(panel_abs_pts).buffer(0)
            if not panel_poly_geom.is_empty:
                for obj in objects:
                    if obj['class_id'] != 1: continue
                    tag_poly_abs = _bbox_to_abs_poly(obj['bbox'], w, h)
                    tag_center = Polygon(tag_poly_abs).centroid
                    if panel_poly_geom.contains(tag_center):
                        # CORRECTED: Directly calculate bbox from absolute coordinates
                        tag_xs = [p[0] for p in tag_poly_abs]
                        tag_ys = [p[1] for p in tag_poly_abs]
                        tag_xmin, tag_ymin, tag_xmax, tag_ymax = min(tag_xs), min(tag_ys), max(tag_xs), max(tag_ys)
                        
                        crop_x1 = min(crop_x1, tag_xmin)
                        crop_y1 = min(crop_y1, tag_ymin)
                        crop_x2 = max(crop_x2, tag_xmax)
                        crop_y2 = max(crop_y2, tag_ymax)
        except Exception:
            pass 

        # 4. Clamp the final coordinates to the image boundaries
        crop_x1 = max(0, int(crop_x1))
        crop_y1 = max(0, int(crop_y1))
        crop_x2 = min(w, int(crop_x2))
        crop_y2 = min(h, int(crop_y2))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue
            
        # 5. Perform the actual crop
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop is None or crop.size == 0:
            continue

        crop_w_actual, crop_h_actual = crop_x2 - crop_x1, crop_y2 - crop_y1
        scale = target_size / max(crop_w_actual, crop_h_actual)
        new_w, new_h = int(crop_w_actual * scale), int(crop_h_actual * scale)
        
        if scale > 1.25:
            interp = cv2.INTER_LANCZOS4
        elif scale > 1.0:
            interp = cv2.INTER_CUBIC
        elif scale < 1.0:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

        try:
            sigma = 0.8 if scale >= 1.0 else 0.0
            if sigma > 0:
                blur = cv2.GaussianBlur(crop_resized, (0, 0), sigma)
                crop_resized = cv2.addWeighted(crop_resized, 1.25, blur, -0.25, 0)
        except Exception:
            pass
        
        x_offset, y_offset = (target_size - new_w) // 2, (target_size - new_h) // 2
        
        if new_w != target_size or new_h != target_size:
            padded_shape = (target_size, target_size, num_channels) if num_channels > 1 else (target_size, target_size)
            padded = np.full(padded_shape, 255, dtype=np.uint8)
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
            crop_resized = padded
        
        crop_labels = transform_labels_for_crop(objects, crop_x1, crop_y1, crop_w_actual, crop_h_actual, scale, x_offset, y_offset, target_size, w, h)
        
        if not crop_labels:
            continue

        crops.append({
            'image': crop_resized,
            'labels': crop_labels,
            'panel_index': i
        })
    
    return crops

def crop_split_images(split, base_dir="datasets/rcp_dual_seg", target_size=1024, force_cleanup=False, per_tag_extras: int = 0, padding_ratio: float = 0.15, specific_images: list = None):
    """
    Crop images in a split using panel-centric approach.
    If specific_images is provided, only those filenames will be processed.
    """
    img_dir = f"{base_dir}/images/{split}/fullsize"
    lbl_dir = f"{base_dir}/labels/{split}/fullsize"
    out_img = f"{base_dir}/images/{split}/cropped1k"
    out_lbl = f"{base_dir}/labels/{split}/cropped1k"
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Warning: fullsize directories not found for {split} split. Skipping.")
        return 0
    
    if force_cleanup:
        print(f"Cleaning up old crops in {split} split...")
        cleanup_old_crops(out_img)
        cleanup_old_crops(out_lbl)
    
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    total_crops = 0
    if specific_images:
        image_files = [f for f in specific_images if os.path.exists(os.path.join(img_dir, f))]
        print(f"Processing {len(image_files)} specific images provided via argument.")
    else:
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'png'))]

    for fn in tqdm(image_files, desc=f"Cropping {split}", unit="img"):
        img_path = f"{img_dir}/{fn}"
        lbl_path = f"{lbl_dir}/{Path(fn).stem}.txt"
        
        crops = create_panel_crop(img_path, lbl_path, target_size, padding_ratio=padding_ratio, per_tag_extras=per_tag_extras)
        
        if crops:
            for crop in crops:
                ext = Path(fn).suffix.lower()
                # If image has alpha, save as png
                if crop['image'].shape[2] == 4 and ext != ".png":
                    ext = '.png'
                
                crop_name = f"{Path(fn).stem}_panel{crop['panel_index']:03d}"
                # Save with higher JPEG quality when applicable
                if ext.lower() in ['.jpg', '.jpeg']:
                    cv2.imwrite(f"{out_img}/{crop_name}{ext}", crop['image'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    cv2.imwrite(f"{out_img}/{crop_name}{ext}", crop['image'])
                
                with open(f"{out_lbl}/{crop_name}.txt", 'w') as f:
                    f.write('\n'.join(crop['labels']))
                
                total_crops += 1
    
    print(f"  Created {total_crops} panel crops for {split} split")
    return total_crops


def _draw_polygon_mask(img, polygon_points, color=(0, 255, 0), alpha=0.3):
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if len(polygon_points) < 6:
        return img
    pts = np.array(
        [(int(polygon_points[i] * w), int(polygon_points[i + 1] * h)) for i in range(0, len(polygon_points), 2)],
        dtype=np.int32,
    )
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    cv2.polylines(img, [pts], True, color, 2)
    return img


def _draw_bbox(img, bbox, label, color=(255, 0, 0)):
    import cv2
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def crop_single_image(image_path: str, label_path: str, out_debug_dir: str, target_size: int = 1024, save_overlays: bool = True, padding_ratio: float = 0.15):
    """Crop a single fullsize image and save crops, labels, and overlays into a debug folder."""
    import cv2
    from pathlib import Path

    out_dir = Path(out_debug_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    ovl_dir = out_dir / "overlays"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        ovl_dir.mkdir(parents=True, exist_ok=True)

    crops = create_panel_crop(image_path, label_path, target_size=target_size, padding_ratio=padding_ratio)
    if not crops:
        return 0

    base_stem = Path(image_path).stem
    count = 0
    for crop in crops:
        crop_name = f"{base_stem}_panel{crop['panel_index']:03d}"
        # Save cropped image
        out_img_path = img_dir / f"{crop_name}.jpg"
        cv2.imwrite(str(out_img_path), crop['image'])
        # Save labels
        out_lbl_path = lbl_dir / f"{crop_name}.txt"
        with open(out_lbl_path, 'w') as f:
            f.write('\n'.join(crop['labels']))

        if save_overlays:
            # Render overlay from saved label lines
            annotated = crop['image'].copy()
            h, w = annotated.shape[:2]
            for line in crop['labels']:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                if cls == 0 and len(parts) > 5:
                    poly = [float(v) for v in parts[5:]]
                    annotated = _draw_polygon_mask(annotated, poly, color=(0, 255, 0), alpha=0.30)
                    annotated = _draw_bbox(annotated, (cx, cy, bw, bh), "Panel", color=(0, 255, 0))
                else:
                    annotated = _draw_bbox(annotated, (cx, cy, bw, bh), "Tag", color=(0, 200, 255))
            cv2.imwrite(str(ovl_dir / f"{crop_name}_overlay.jpg"), annotated)
        count += 1

    return count

def main():
    """Main function for panel-centric cropping"""
    parser = argparse.ArgumentParser(description="Create panel-centric crops from full-size images.")
    parser.add_argument("--force", action="store_true", help="Force cleanup of existing crops before running.")
    parser.add_argument("--base-dir", default="datasets/rcp_dual_seg", help="Dataset base directory (images/labels root)")
    parser.add_argument("--single-image", default="", help="Path to a single fullsize image to crop (debug mode)")
    parser.add_argument("--single-label", default="", help="Optional path to the label file for the single image")
    parser.add_argument("--out-debug", default="", help="Output directory for single-image crops and overlays")
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--padding-ratio", type=float, default=0.22, help="Base padding ratio around panel bbox")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits for batch mode (e.g., 'val' or 'train,val')")
    parser.add_argument("--per-tag-extras", type=int, default=0, help="Create this many extra tag-focused crops per panel (0=off)")
    parser.add_argument("--images", type=str, default="", help="Comma-separated list of specific image filenames to process within the split's fullsize dir.")
    # New: write single-image crops directly back to dataset
    parser.add_argument("--single-to-dataset", default="", help="If set, dataset base dir to overwrite images/labels/<split>/cropped1k with new crops for this single image")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to write when using --single-to-dataset")
    args = parser.parse_args()

    print("=== Panel-Centric Cropping ===")
    print("Creating focused training data using panel coordinates")
    print(f"Target size: {args.target_size}px, Padding: {args.padding_ratio*100:.0f}%")
    if args.force:
        print("FORCE mode: Cleaning up old crops.")

    # Single debug mode (optionally write back to dataset)
    if args.single_image:
        img_path = Path(args.single_image)
        if not img_path.is_file():
            print(f"ERROR: Single image not found: {img_path}")
            return
        if args.single_label:
            lbl_path = Path(args.single_label)
        else:
            # Try to infer labels: replace images/.../fullsize with labels/.../fullsize and .txt
            stem = img_path.stem
            candidate = None
            try:
                parts = list(img_path.parts)
                if "images" in parts and "fullsize" in parts:
                    parts[parts.index("images")] = "labels"
                    parts[-1] = stem + ".txt"
                    candidate = Path(*parts)
            except Exception:
                candidate = None
            lbl_path = candidate if candidate and candidate.is_file() else None
        if not lbl_path or not Path(lbl_path).is_file():
            print("ERROR: Label file for single image not found. Provide --single-label explicitly.")
            return

        # If writing back to dataset
        if args.single_to_dataset:
            base = Path(args.single_to_dataset)
            img_out_dir = base / "images" / args.split / "cropped1k"
            lbl_out_dir = base / "labels" / args.split / "cropped1k"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            crops = create_panel_crop(str(img_path), str(lbl_path), target_size=args.target_size, padding_ratio=args.padding_ratio)
            written = 0
            for crop in crops:
                stem = f"{img_path.stem}_panel{crop['panel_index']:03d}"
                # Force PNG for sharper text on problematic drawings like x40
                out_img_path = img_out_dir / f"{stem}.png"
                cv2.imwrite(str(out_img_path), crop['image'])
                with open(lbl_out_dir / f"{stem}.txt", 'w') as f:
                    f.write('\n'.join(crop['labels']))
                written += 1
            print(f"OK: Wrote {written} crops to dataset: {img_out_dir}")
        else:
            out_debug = Path(args.out_debug) if args.out_debug else Path("runs") / "crop_debug" / f"{img_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_debug.mkdir(parents=True, exist_ok=True)
            num = crop_single_image(str(img_path), str(lbl_path), str(out_debug), target_size=args.target_size, save_overlays=True, padding_ratio=args.padding_ratio)
            print(f"OK: Saved {num} crops and overlays to {out_debug}")
        return

    # Batch mode over splits
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    specific_images_list = [img.strip() for img in args.images.split(',') if img.strip()] if args.images else None
    
    info = {}
    for split in splits:
        print(f"\nProcessing {split} split under base dir: {args.base_dir} ...")
        info[split] = crop_split_images(split, base_dir=args.base_dir, target_size=args.target_size, force_cleanup=args.force, per_tag_extras=int(getattr(args, 'per-tag-extras', args.per_tag_extras)), padding_ratio=args.padding_ratio, specific_images=specific_images_list)

    Path(args.base_dir).mkdir(exist_ok=True)
    with open(str(Path(args.base_dir) / "cropping_info.json"), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'counts': info,
        'target_size': args.target_size,
            'padding_ratio': args.padding_ratio
        }, f, indent=2)

    print(f"\nCropping Summary: {info}")
    print("Panel-centric cropping complete!")

if __name__ == "__main__":
    main()
