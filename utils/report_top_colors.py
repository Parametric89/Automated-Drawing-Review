#!/usr/bin/env python3
"""
Report the top-N colors by area for each page/image and write an Excel file.

Features:
- Accepts a PDF/image or a folder containing PDFs/images
- Optional PDF rasterization DPI
- Per-channel color quantization to reduce noise (e.g., bin=8 → values rounded to 0,8,16,...)
- Grayscale-only mode (default): include only near-neutral tones, excluding near-white and near-black
- Outputs an Excel workbook with one sheet listing file/page and its top-N colors with percentages

Example:
  python utils/report_top_colors.py --input inference --out inference/top_colors.xlsx --topn 3 --quantize 8 --dpi 300 --gray-min 5 --gray-max 245 --gray-tol 3
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images_in_dir(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(dir_path.rglob("*")) if p.suffix.lower() in exts]


def list_pdfs_in_dir(dir_path: Path) -> List[Path]:
    return [p for p in sorted(dir_path.rglob("*.pdf"))]


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 300) -> List[Path]:
    ensure_dir(out_dir)
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF input. Please `pip install pymupdf`.") from e
    image_paths: List[Path] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{pdf_path.stem}_page_{i+1:03d}.png"
        pix.save(str(out_path))
        image_paths.append(out_path)
    doc.close()
    return image_paths


def quantize_image_bgr(img_bgr: np.ndarray, bin_size: int) -> np.ndarray:
    q = max(1, int(bin_size))
    # floor to nearest multiple of q
    quant = (img_bgr // q) * q
    return np.ascontiguousarray(quant, dtype=np.uint8)


def rgb_tuple_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def top_colors_for_image(
    img_path: Path,
    quantize: int,
    ignore_bg: int,
    gray_min: int,
    gray_max: int,
    gray_tol: int,
    grayscale_only: bool,
    down_max_side: int,
) -> Tuple[int, List[Tuple[Tuple[int, int, int], int]]]:
    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        return 0, []
    # Downscale for speed/memory if requested
    if int(down_max_side) > 0:
        h0, w0 = img.shape[:2]
        m = max(h0, w0)
        if m > int(down_max_side):
            s = float(down_max_side) / float(m)
            nw = max(1, int(round(w0 * s)))
            nh = max(1, int(round(h0 * s)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # Optional ignore of nearly-white pixels
    if ignore_bg > 0:
        # keep only pixels where any channel < ignore_bg
        mask = np.any(img < ignore_bg, axis=2)
        if not mask.any():
            total = int(img.shape[0] * img.shape[1])
            return total, []
        img = img.copy()
        img[~mask] = 255
    if quantize > 1:
        img = quantize_image_bgr(img, quantize)
    h, w = img.shape[:2]
    total = int(h * w)
    # Collapse to unique colors
    flat = img.reshape(-1, 3)
    # Get unique rows and counts
    colors_bgr, counts = np.unique(flat, axis=0, return_counts=True)
    # Convert to RGB for reporting
    colors_rgb = colors_bgr[:, ::-1]
    # Grayscale-only filter: keep colors where |R-G|<=tol and |R-B|<=tol and value within [gray_min, gray_max]
    if grayscale_only:
        r = colors_rgb[:, 0].astype(np.int16)
        g = colors_rgb[:, 1].astype(np.int16)
        b = colors_rgb[:, 2].astype(np.int16)
        is_gray = (np.abs(r - g) <= gray_tol) & (np.abs(r - b) <= gray_tol) & (r >= gray_min) & (r <= gray_max)
        if not np.any(is_gray):
            return total, []
        colors_rgb = colors_rgb[is_gray]
        counts = counts[is_gray]
    # Pair and sort by count desc
    pairs = list(zip([tuple(map(int, c)) for c in colors_rgb], [int(c) for c in counts]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return total, pairs


def run(
    input_path: Path,
    out_xlsx: Path,
    dpi: int,
    topn: int,
    quantize: int,
    ignore_bg: int,
    gray_min: int,
    gray_max: int,
    gray_tol: int,
    grayscale_only: bool,
    down_max_side: int,
) -> None:
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("This script requires pandas to write Excel. Please `pip install pandas openpyxl`.") from e

    rows = []
    sources: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            tmp_dir = out_xlsx.parent / "tmp_pdf_pages"
            images = pdf_to_images(input_path, tmp_dir, dpi=dpi)
            sources.extend(images)
        else:
            sources.append(input_path)
    elif input_path.is_dir():
        sources = list_images_in_dir(input_path)
        pdfs = list_pdfs_in_dir(input_path)
        for pdf in pdfs:
            tmp_dir = out_xlsx.parent / "tmp_pdf_pages" / pdf.stem
            images = pdf_to_images(pdf, tmp_dir, dpi=dpi)
            sources.extend(images)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    for p in tqdm(sources, desc="Analyze colors"):
        total, pairs = top_colors_for_image(
            p,
            quantize=quantize,
            ignore_bg=ignore_bg,
            gray_min=gray_min,
            gray_max=gray_max,
            gray_tol=gray_tol,
            grayscale_only=grayscale_only,
            down_max_side=down_max_side,
        )
        if total == 0:
            rows.append({"image": str(p), "total_pixels": 0})
            continue
        # Build one row per image with top-N colors
        row = {"image": str(p), "total_pixels": total}
        for i in range(min(topn, len(pairs))):
            (r, g, b), cnt = pairs[i]
            pct = (cnt / total) * 100.0
            row[f"color{i+1}_rgb"] = f"({r},{g},{b})"
            row[f"color{i+1}_hex"] = rgb_tuple_to_hex((r, g, b))
            row[f"color{i+1}_pct"] = round(pct, 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    ensure_dir(out_xlsx.parent)
    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="top_colors")
    print(f"\nWrote Excel report: {out_xlsx}")


def main():
    ap = argparse.ArgumentParser(description="Report top-N colors by area and write an Excel file")
    ap.add_argument("--input", required=True, help="Path to PDF/image or directory")
    ap.add_argument("--out", required=True, help="Output Excel path, e.g., reports/top_colors.xlsx")
    ap.add_argument("--dpi", type=int, default=300, help="PDF rasterization DPI")
    ap.add_argument("--topn", type=int, default=3, help="Number of top colors to report")
    ap.add_argument("--quantize", type=int, default=8, help="Per-channel bin size (e.g., 8 → 0,8,16,...) to reduce noise")
    ap.add_argument("--ignore-bg", type=int, default=0, help="Ignore pixels where all channels >= this (e.g., 250 to drop near-white)")
    ap.add_argument("--grayscale-only", action="store_true", default=True, help="Only count near-neutral colors (exclude colored hues)")
    ap.add_argument("--gray-min", type=int, default=5, help="Minimum gray value to include (excludes near-black)")
    ap.add_argument("--gray-max", type=int, default=245, help="Maximum gray value to include (excludes near-white)")
    ap.add_argument("--gray-tol", type=int, default=3, help="Max channel difference to consider neutral (|R-G|<=tol and |R-B|<=tol)")
    ap.add_argument("--down", type=int, default=2048, help="Downscale images to this max side before analysis (0 to disable)")
    args = ap.parse_args()

    run(
        input_path=Path(args.input),
        out_xlsx=Path(args.out),
        dpi=args.dpi,
        topn=args.topn,
        quantize=args.quantize,
        ignore_bg=args.ignore_bg,
        gray_min=args.gray_min,
        gray_max=args.gray_max,
        gray_tol=args.gray_tol,
        grayscale_only=bool(args.grayscale_only),
        down_max_side=args.down,
    )


if __name__ == "__main__":
    main()


