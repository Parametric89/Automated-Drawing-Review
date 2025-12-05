from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np

@dataclass
class TilingConfig:
    tile_size: int
    overlap: float
    keep_inner_frac: float
    # training
    min_panel_frac: float
    retile_if_below: bool
    pre_scale: float
    # inference post-merge
    merge_mode: str
    merge_iou: float
    # YOLO predict
    conf: float
    iou: float
    max_det: int
    # filters
    min_area_ratio: float
    min_short_side_ratio: float
    ar_min: float
    ar_max: float
    # refine
    refine_enabled: bool
    refine_imgsz: int
    refine_margin: float
    refine_iou: float

def load_tiling_config(path: Path = Path("tiling_config.yaml")) -> TilingConfig:
    d = yaml.safe_load(Path(path).read_text())
    r = d.get("refine", {})
    return TilingConfig(
        tile_size=int(d["tile_size"]),
        overlap=float(d["overlap"]),
        keep_inner_frac=float(d["keep_inner_frac"]),
        min_panel_frac=float(d["min_panel_frac"]),
        retile_if_below=bool(d["retile_if_below"]),
        pre_scale=float(d["pre_scale"]),
        merge_mode=str(d["merge_mode"]).lower(),
        merge_iou=float(d["merge_iou"]),
        conf=float(d["conf"]),
        iou=float(d["iou"]),
        max_det=int(d["max_det"]),
        min_area_ratio=float(d["min_area_ratio"]),
        min_short_side_ratio=float(d["min_short_side_ratio"]),
        ar_min=float(d["ar_min"]),
        ar_max=float(d["ar_max"]),
        refine_enabled=bool(r.get("enabled", False)),
        refine_imgsz=int(r.get("imgsz", 1536)),
        refine_margin=float(r.get("margin", 0.12)),
        refine_iou=float(r.get("iou", 0.5)),
    )

def make_tiles(h: int, w: int, tile: int, overlap: float):
    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    windows = []
    for y in ys:
        for x in xs:
            x2 = min(x + tile, w)
            y2 = min(y + tile, h)
            x = max(0, x2 - tile)
            y = max(0, y2 - tile)
            windows.append((x, y, x2, y2))
    return windows

def inner_window(x1, y1, x2, y2, overlap, keep_inner_frac):
    border = int((x2 - x1) * overlap * max(0.0, min(1.0, keep_inner_frac)))
    return (x1 + border, y1 + border, x2 - border, y2 - border)

def iqr_band(x: np.ndarray, k: float):
    if x.size == 0:
        return np.ones_like(x, dtype=bool)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (x >= lo) & (x <= hi)
