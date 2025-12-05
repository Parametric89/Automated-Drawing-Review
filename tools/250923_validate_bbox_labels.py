#!/usr/bin/env python3
"""
validate_yolo_dir.py
Enhanced dataset label audit + optional YOLOv8 validation.

Highlights
- Deep audit of *all* lines in every label (no 3-line preview).
- Works with detection (5 tokens) or segmentation (cx cy w h + polygon) labels.
- Auto-detects format per-file (or force with --format).
- Flags and counts specific error types (empty_file, too_few_fields, out_of_range, etc.).
- Reports class histogram, avg boxes/image, missing/orphan files.
- Exports a JSON/CSV of issues for downstream tooling.
- Optional YOLO val run using a temp YAML with both 'train:' and 'val:' keys.
- Console output uses ASCII only (Windows-safe).

Exit codes
- 0: audit passed (no issues); YOLO val (if requested) attempted.
- 1: audit found issues (still may run YOLO if you asked).
- 2: fatal argument/path error.
- 3: missing Ultralytics for YOLO validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    # fallback: no progress bar
    def tqdm(x, **kw):  # type: ignore
        return x


# ----------------------------- helpers --------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def parse_allowed_classes(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None

@dataclass
class FileIssues:
    path: Path
    errors: List[str]

@dataclass
class AuditStats:
    total_images: int
    images_with_labels: int
    missing_label: int
    orphan_labels: int
    files_with_issues: int
    total_issues: int
    class_hist: Dict[int, int]
    avg_boxes_per_image: float


# ----------------------------- validation ------------------------------
def validate_detection_line(parts: List[str],
                            file: Path, line_num: int,
                            allowed_classes: Optional[List[int]],
                            errors: List[str],
                            class_hist: Counter) -> None:
    # detection: class cx cy w h (exactly 5 tokens)
    if len(parts) != 5:
        errors.append(f"{file.name}:{line_num}: tokens!=5 (got {len(parts)})")
        return
    try:
        cls = int(round(float(parts[0])))
        cx, cy, bw, bh = map(float, parts[1:5])
    except Exception:
        errors.append(f"{file.name}:{line_num}: parse_error")
        return

    # allowed classes
    if allowed_classes is not None and cls not in allowed_classes:
        errors.append(f"{file.name}:{line_num}: class_id_not_allowed:{cls}")

    # range and positivity checks
    for name, v in (("cx", cx), ("cy", cy), ("w", bw), ("h", bh)):
        if not (0.0 <= v <= 1.0):
            errors.append(f"{file.name}:{line_num}: {name}_out_of_range:{v}")
    if bw <= 0.0 or bh <= 0.0:
        errors.append(f"{file.name}:{line_num}: non_positive_bbox w={bw}, h={bh}")

    class_hist[cls] += 1


def validate_segmentation_line(parts: List[str],
                               file: Path, line_num: int,
                               allowed_classes: Optional[List[int]],
                               errors: List[str],
                               class_hist: Counter) -> None:
    # segmentation: class cx cy w h [poly...]
    if len(parts) < 5:
        errors.append(f"{file.name}:{line_num}: too_few_tokens:{len(parts)}")
        return
    try:
        cls = int(round(float(parts[0])))
        cx, cy, bw, bh = map(float, parts[1:5])
    except Exception:
        errors.append(f"{file.name}:{line_num}: parse_error_bbox")
        return

    if allowed_classes is not None and cls not in allowed_classes:
        errors.append(f"{file.name}:{line_num}: class_id_not_allowed:{cls}")

    # bbox checks
    for name, v in (("cx", cx), ("cy", cy), ("w", bw), ("h", bh)):
        if not (0.0 <= v <= 1.0):
            errors.append(f"{file.name}:{line_num}: {name}_out_of_range:{v}")
    if bw <= 0.0 or bh <= 0.0:
        errors.append(f"{file.name}:{line_num}: non_positive_bbox w={bw}, h={bh}")

    # polygon checks (if present)
    if len(parts) > 5:
        coords = parts[5:]
        if len(coords) % 2 != 0:
            errors.append(f"{file.name}:{line_num}: poly_odd_len:{len(coords)}")
        else:
            try:
                vals = list(map(float, coords))
            except Exception:
                errors.append(f"{file.name}:{line_num}: poly_parse_error")
            else:
                for i, v in enumerate(vals):
                    if not (0.0 <= v <= 1.0):
                        errors.append(f"{file.name}:{line_num}: poly_out_of_range idx={i} val={v}")

    class_hist[cls] += 1


def decide_format_for_file(lines: List[str], format_arg: str) -> str:
    if format_arg != "auto":
        return format_arg
    # auto: if any line has >5 tokens -> segmentation, else detection
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) > 5:
            return "segmentation"
    return "detection"


def audit_labels(images_dir: Path,
                 labels_dir: Path,
                 format_arg: str,
                 allowed_classes: Optional[List[int]],
                 max_images: Optional[int],
                 export_json: Optional[Path],
                 export_csv: Optional[Path]) -> Tuple[AuditStats, List[FileIssues], List[Path]]:
    if not images_dir.exists():
        print(f"[ERROR] Images dir not found: {images_dir}")
        sys.exit(2)
    if not labels_dir.exists():
        print(f"[ERROR] Labels dir not found: {labels_dir}")
        sys.exit(2)

    # gather images and labels, also detect orphans
    imgs = [p for p in images_dir.iterdir() if p.is_file() and is_image(p)]
    imgs.sort()
    if max_images is not None:
        imgs = imgs[:max_images]

    label_files = {p.stem: p for p in labels_dir.glob("*.txt")}
    missing_label = 0
    img_label_pairs: List[Tuple[Path, Path]] = []
    for ip in imgs:
        lp = label_files.get(ip.stem)
        if lp is None:
            missing_label += 1
        else:
            img_label_pairs.append((ip, lp))

    # orphans (labels without corresponding image)
    orphan_labels = 0
    img_stems = {p.stem for p in imgs}
    orphans = [lp for s, lp in label_files.items() if s not in img_stems]
    orphan_labels = len(orphans)

    # per-file validation
    issues: List[FileIssues] = []
    class_hist: Counter = Counter()
    boxes_count = 0
    files_with_issues = 0

    for _, lp in tqdm(img_label_pairs, desc="Auditing labels", unit="file"):
        file_errors: List[str] = []
        try:
            raw = lp.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            issues.append(FileIssues(lp, [f"{lp.name}: read_error: {e}"]))
            files_with_issues += 1
            continue
        text = raw.strip()
        if not text:
            issues.append(FileIssues(lp, [f"{lp.name}: empty_file"]))
            files_with_issues += 1
            continue

        lines = [ln for ln in text.splitlines() if ln.strip()]
        fmt = decide_format_for_file(lines, format_arg)
        per_file_errors: List[str] = []

        for i, ln in enumerate(lines, 1):
            parts = ln.strip().split()
            if len(parts) == 0:
                # skip (pure blank lines were filtered)
                continue

            if fmt == "detection":
                validate_detection_line(parts, lp, i, allowed_classes, per_file_errors, class_hist)
            else:
                validate_segmentation_line(parts, lp, i, allowed_classes, per_file_errors, class_hist)

            # count "boxes" as number of lines
            boxes_count += 1

        if per_file_errors:
            issues.append(FileIssues(lp, per_file_errors))
            files_with_issues += 1

    total_imgs_seen = len(imgs)
    imgs_with_labels = len(img_label_pairs)
    total_issues = sum(len(fi.errors) for fi in issues)
    avg_boxes = (boxes_count / imgs_with_labels) if imgs_with_labels else 0.0

    stats = AuditStats(
        total_images=total_imgs_seen,
        images_with_labels=imgs_with_labels,
        missing_label=missing_label,
        orphan_labels=orphan_labels,
        files_with_issues=files_with_issues,
        total_issues=total_issues,
        class_hist=dict(class_hist),
        avg_boxes_per_image=avg_boxes,
    )

    # optional exports
    if export_json:
        payload = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "format": format_arg,
            "allowed_classes": allowed_classes,
            "stats": stats.__dict__,
            "issues": [
                {"file": str(fi.path), "errors": fi.errors} for fi in issues
            ],
        }
        export_json.parent.mkdir(parents=True, exist_ok=True)
        export_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if export_csv:
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        with export_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "error"])
            for fi in issues:
                for e in fi.errors:
                    w.writerow([str(fi.path), e])

    return stats, issues, orphans


def print_audit_report(stats: AuditStats, issues: List[FileIssues], orphans: List[Path], max_show: int = 30) -> None:
    print("\n=== Dataset Audit ===")
    print(f"Total images found   : {stats.total_images}")
    print(f"Images with labels   : {stats.images_with_labels}")
    print(f"Images missing label : {stats.missing_label}")
    print(f"Orphan label files   : {stats.orphan_labels}")
    print(f"Files with issues    : {stats.files_with_issues}")
    print(f"Total issues         : {stats.total_issues}")
    print(f"Avg boxes / image    : {stats.avg_boxes_per_image:.2f}")

    if stats.class_hist:
        print("Class histogram      :", " ".join(f"{k}:{v}" for k, v in sorted(stats.class_hist.items())))

    if orphans:
        print("\nSample orphan labels (first 10):")
        for p in orphans[:10]:
            print(f"  - {p}")

    if issues:
        print("\nSample issues (first {} files):".format(min(max_show, len(issues))))
        # flatten first up to max_show files
        for fi in issues[:max_show]:
            print(f"  {fi.path}")
            for e in fi.errors[:5]:
                print(f"    - {e}")
            if len(fi.errors) > 5:
                print(f"    ... (+{len(fi.errors) - 5} more)")
        if len(issues) > max_show:
            print(f"  ... and {len(issues) - max_show} more files with issues")
    else:
        print("\nOK: All labels passed the deep checks. Ready for training.")


# ------------------------- YOLO validation -----------------------------
def write_temp_dataset_yaml(val_images: Path, names: List[str]) -> Path:
    """Make a minimal YAML with both 'train' and 'val' -> val_images."""
    img_dir = val_images.as_posix()
    yaml_text = "# Auto-generated for validation\n"
    yaml_text += f"train: {img_dir}\n"
    yaml_text += f"val: {img_dir}\n"
    yaml_text += "names:\n"
    for i, n in enumerate(names):
        yaml_text += f"  {i}: {n}\n"
    tmp = Path(tempfile.gettempdir()) / f"val_dataset_{val_images.stem}.yaml"
    tmp.write_text(yaml_text, encoding="utf-8")
    return tmp


def run_yolo_val(model_path: Path,
                 images_dir: Path,
                 names: List[str],
                 imgsz: int, batch: int, device: str,
                 save_dir: Optional[Path],
                 conf: Optional[float],
                 iou: float,
                 workers: int,
                 half: bool) -> None:
    try:
        from ultralytics import YOLO
    except Exception:
        print("[ERROR] Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(3)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(2)

    data_yaml = write_temp_dataset_yaml(images_dir, names)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== YOLO Validation ===")
    print(f"Model : {model_path}")
    print(f"Data  : {data_yaml}")
    print(f"imgsz : {imgsz}  batch: {batch}  device: {device}  half: {half}  iou: {iou}  conf: {conf}")
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        iou=iou,
        conf=conf,
        half=half,
        workers=workers,
        save_json=True,
        project=str(save_dir) if save_dir else None,
        name=None if save_dir else "val_results",
        verbose=True,
        plots=True,
        split="val",
    )

    try:
        box = metrics.box
        print("\n=== Validation Results (Boxes) ===")
        print(f"Precision : {box.precision:.3f}")
        print(f"Recall    : {box.recall:.3f}")
        print(f"mAP@50    : {box.map50:.3f}")
        print(f"mAP@50-95 : {box.map:.3f}")
        print("\nNote: Metrics are computed by Ultralytics on the provided directories. "
              "They are valid for these exact images/labels.")
    except Exception:
        print("[WARN] Could not parse metrics.box; raw metrics object follows:")
        print(metrics)


# ----------------------------- CLI ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Deep audit YOLO labels + optional YOLOv8 validation.")
    ap.add_argument("--images", required=True, help="Path to images dir (e.g., .../images/train/tiled1280)")
    ap.add_argument("--labels", required=True, help="Path to labels dir (e.g., .../labels/train/tiled1280)")
    ap.add_argument("--format", choices=["auto", "detection", "segmentation"], default="auto",
                    help="Label format. 'auto' picks per-file based on token count.")
    ap.add_argument("--allowed-classes", default="0",
                    help="Comma-separated class IDs allowed (default '0'). "
                         "Leave blank to allow any class ID.")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Audit at most N images (quick spot-check).")
    ap.add_argument("--export-json", type=str, default=None,
                    help="Write audit report (issues + stats) to JSON.")
    ap.add_argument("--export-csv", type=str, default=None,
                    help="Write audit issues to CSV.")

    ap.add_argument("--audit-only", action="store_true", help="Only audit labels, skip YOLO validation.")
    ap.add_argument("--model", default=None, help="YOLOv8 model .pt for validation (optional).")
    ap.add_argument("--names", default="panel", help="Comma-separated class names for val YAML (default: 'panel').")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save-dir", type=str, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--half", action="store_true")

    args = ap.parse_args()

    images_dir = Path(args.images).resolve()
    labels_dir = Path(args.labels).resolve()
    allowed = parse_allowed_classes(args.allowed_classes)
    export_json = Path(args.export_json).resolve() if args.export_json else None
    export_csv = Path(args.export_csv).resolve() if args.export_csv else None

    # Deep audit
    stats, issues, orphans = audit_labels(
        images_dir=images_dir,
        labels_dir=labels_dir,
        format_arg=args.format,
        allowed_classes=allowed,
        max_images=args.max_images,
        export_json=export_json,
        export_csv=export_csv,
    )
    print(f"\nImages dir : {images_dir}")
    print(f"Labels dir : {labels_dir}")
    print_audit_report(stats, issues, orphans)

    # Non-zero exit if any issues (still continue to YOLO val if requested)
    audit_rc = 0 if stats.total_issues == 0 and stats.missing_label == 0 else 1

    if args.audit_only or not args.model:
        sys.exit(audit_rc)

    # YOLO validation
    names = [s.strip() for s in args.names.split(",") if s.strip()] or ["panel"]
    model_path = Path(args.model).resolve()
    save_dir = Path(args.save_dir).resolve() if args.save_dir else None
    run_yolo_val(
        model_path=model_path,
        images_dir=images_dir,
        names=names,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_dir=save_dir,
        conf=args.conf,
        iou=args.iou,
        workers=args.workers,
        half=args.half,
    )

    # propagate audit status (0/1) as final rc
    sys.exit(audit_rc)


if __name__ == "__main__":
    main()
