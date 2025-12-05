import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm


def validate_detection_line(line: str, file: Path, line_num: int, errors: List[str]) -> None:
    """Validate YOLO detection format: class cx cy w h"""
    parts = line.strip().split()
    if not parts:
        return
    try:
        cls = int(round(float(parts[0])))
    except Exception:
        errors.append(f"{file.name}:{line_num}: invalid class id: {parts[0]}")
        return

    if len(parts) != 5:
        errors.append(f"{file.name}:{line_num}: too few tokens ({len(parts)})")
        return

    try:
        cx, cy, bw, bh = map(float, parts[1:5])
    except Exception:
        errors.append(f"{file.name}:{line_num}: bbox parse error: {' '.join(parts[1:5])}")
        return

    # Range checks
    for name, v in (('cx', cx), ('cy', cy), ('bw', bw), ('bh', bh)):
        if not (0.0 <= v <= 1.0):
            errors.append(f"{file.name}:{line_num}: {name} out of [0,1]: {v}")
    if bw <= 0 or bh <= 0:
        errors.append(f"{file.name}:{line_num}: non-positive bbox size: w={bw}, h={bh}")

    # Class validation
    if cls not in [0, 1]:  # panel=0, panel_tag=1
        errors.append(f"{file.name}:{line_num}: invalid class id: {cls} (expected 0 or 1)")


def validate_segmentation_line(line: str, file: Path, line_num: int, errors: List[str]) -> None:
    """Validate YOLO segmentation format: class cx cy w h [polygon_coords...]"""
    parts = line.strip().split()
    if not parts:
        return
    try:
        cls = int(round(float(parts[0])))
    except Exception:
        errors.append(f"{file.name}:{line_num}: invalid class id: {parts[0]}")
        return

    if len(parts) < 5:
        errors.append(f"{file.name}:{line_num}: too few tokens ({len(parts)})")
        return

    try:
        cx, cy, bw, bh = map(float, parts[1:5])
    except Exception:
        errors.append(f"{file.name}:{line_num}: bbox parse error: {' '.join(parts[1:5])}")
        return

    # Range checks for bbox
    for name, v in (('cx', cx), ('cy', cy), ('bw', bw), ('bh', bh)):
        if not (0.0 <= v <= 1.0):
            errors.append(f"{file.name}:{line_num}: {name} out of [0,1]: {v}")
    if bw <= 0 or bh <= 0:
        errors.append(f"{file.name}:{line_num}: non-positive bbox size: w={bw}, h={bh}")

    if cls == 0:
        # panel requires polygon points
        poly = parts[5:]
        if len(poly) < 6:
            errors.append(f"{file.name}:{line_num}: panel has < 3 polygon pairs ({len(poly)//2})")
            return
        if len(poly) % 2 != 0:
            errors.append(f"{file.name}:{line_num}: polygon coords not even length ({len(poly)})")
            return
        # check ranges
        try:
            coords = list(map(float, poly))
        except Exception:
            errors.append(f"{file.name}:{line_num}: polygon coord parse error")
            return
        for i, v in enumerate(coords):
            if not (0.0 <= v <= 1.0):
                errors.append(f"{file.name}:{line_num}: poly[{i}] out of [0,1]: {v}")
    else:
        # tag must be bbox-only
        if len(parts) != 5:
            errors.append(f"{file.name}:{line_num}: tag should be bbox-only (5 tokens), found {len(parts)}")


def validate_line(line: str, file: Path, line_num: int, errors: List[str], format_type: str) -> None:
    """Validate line based on format type"""
    if format_type == "detection":
        validate_detection_line(line, file, line_num, errors)
    elif format_type == "segmentation":
        validate_segmentation_line(line, file, line_num, errors)
    else:
        errors.append(f"{file.name}:{line_num}: unknown format type: {format_type}")


def main():
    ap = argparse.ArgumentParser(description="Validate YOLO labels (detection or segmentation format)")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--format", choices=["detection", "segmentation"], default="detection",
                   help="Label format to validate (default: detection)")
    args = ap.parse_args()

    lbl_dir = Path(args.labels_dir)
    if not lbl_dir.exists():
        raise SystemExit(f"Labels dir not found: {lbl_dir}")

    files = sorted(p for p in lbl_dir.glob("*.txt") if p.is_file())
    total_errors = []
    for fp in tqdm(files, desc=f"Validating {lbl_dir}", unit="file"):
        try:
            lines = fp.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            total_errors.append(f"{fp.name}: read error: {e}")
            continue
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            validate_line(line, fp, i, total_errors, args.format)

    if total_errors:
        print("Found issues:\n" + "\n".join(total_errors))
        raise SystemExit(1)
    else:
        print(f"All {args.format} labels OK.")


if __name__ == "__main__":
    main()


