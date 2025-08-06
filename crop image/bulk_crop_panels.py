
"""
bulk_crop_panels.py
-------------------
Bulk‑convert PDF shop drawings to images and crop every panel detected by a
YOLO model.

Usage
-----
python bulk_crop_panels.py --pdf-dir ./input_pdfs \
                           --weights panel_yolov8.pt \
                           --out-dir ./crops \
                           --img-size 1280 --conf 0.25

Dependencies
------------
pip install ultralytics pdf2image pillow tqdm

* On Windows, install Poppler and add its /bin folder to PATH or pass
  --poppler-path "C:/path/to/poppler/bin".
* The --weights file must be a YOLOv8 model trained to detect the class
  “panel”.

Author: ChatGPT
"""

import argparse
import os
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def pdf_to_images(pdf_path: Path, dpi: int = 300, poppler_path: str = None) -> List[Image.Image]:
    """Convert each page of a PDF to a PIL Image."""
    images = convert_from_path(
        pdf_path.as_posix(),
        dpi=dpi,
        poppler_path=poppler_path
    )
    return images

def save_crop(img: Image.Image, xyxy, save_dir: Path, base_name: str, idx: int):
    x1, y1, x2, y2 = map(int, xyxy)
    crop = img.crop((x1, y1, x2, y2))
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{base_name}_crop{idx:02d}.png"
    crop.save(out_path)

def process_pdf(model: YOLO, pdf_path: Path, out_dir: Path,
                img_size: int, conf: float, poppler_path: str):
    images = pdf_to_images(pdf_path, poppler_path=poppler_path)
    for page_no, pil_img in enumerate(images, start=1):
        np_img = np.array(pil_img)[:, :, ::-1]  # PIL RGB -> BGR for YOLO
        results = model.predict(
            np_img,
            imgsz=img_size,
            conf=conf,
            verbose=False
        )
        for det_idx, box in enumerate(results[0].boxes.xyxy):
            save_crop(
                pil_img,
                box,
                out_dir / pdf_path.stem,
                f"{pdf_path.stem}_p{page_no}",
                det_idx
            )

def main():
    parser = argparse.ArgumentParser(description="Bulk crop panels from PDFs using YOLO.")
    parser.add_argument("--pdf-dir", required=True, type=Path, help="Folder with PDF files")
    parser.add_argument("--weights", required=True, type=Path, help="YOLO .pt weights file")
    parser.add_argument("--out-dir", default=Path("crops"), type=Path, help="Output folder")
    parser.add_argument("--img-size", type=int, default=1280, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--poppler-path", type=str, default=None, help="Optional path to Poppler bin")
    args = parser.parse_args()

    model = YOLO(args.weights)

    pdf_files = sorted(list(Path(args.pdf_dir).glob("*.pdf")))
    if not pdf_files:
        print("No PDF files found in", args.pdf_dir)
        return

    for pdf_path in tqdm(pdf_files, desc="PDFs", unit="file"):
        try:
            process_pdf(
                model,
                pdf_path,
                args.out_dir,
                args.img_size,
                args.conf,
                args.poppler_path
            )
        except Exception as e:
            print(f"[WARN] Failed on {pdf_path}: {e}")

if __name__ == "__main__":
    main()
