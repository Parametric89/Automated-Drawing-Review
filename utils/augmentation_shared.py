from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional, Dict, Any

@dataclass
class AugmentationConfig:
    # Core settings
    target_size: int
    augs_per_image: int
    splits: str
    out_folder: str
    
    # Geometric transformations
    rotation_deg: float
    scale_low: float
    scale_high: float
    translation_frac: float
    pad_margin: int
    
    # Class handling
    tag_class_id: int
    
    # Quality settings
    jpeg_quality: int
    opencv_num_threads: int
    
    # Processing options
    keep_existing: bool
    max_images: Optional[int]
    
    # Presets
    presets: Dict[str, Dict[str, Any]]

def load_augmentation_config(path: Path = Path("augmentation_config.yaml")) -> AugmentationConfig:
    """Load augmentation configuration from YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Augmentation config not found at {path}")
    
    d = yaml.safe_load(path.read_text())
    
    return AugmentationConfig(
        target_size=int(d["target_size"]),
        augs_per_image=int(d["augs_per_image"]),
        splits=str(d["splits"]),
        out_folder=str(d["out_folder"]),
        rotation_deg=float(d["rotation_deg"]),
        scale_low=float(d["scale_low"]),
        scale_high=float(d["scale_high"]),
        translation_frac=float(d["translation_frac"]),
        pad_margin=int(d["pad_margin"]),
        tag_class_id=int(d["tag_class_id"]),
        jpeg_quality=int(d["jpeg_quality"]),
        opencv_num_threads=int(d["opencv_num_threads"]),
        keep_existing=bool(d["keep_existing"]),
        max_images=int(d["max_images"]) if d["max_images"] is not None else None,
        presets=dict(d.get("presets", {}))
    )

def load_augmentation_preset(preset_name: str, config_path: Path = Path("augmentation_config.yaml")) -> AugmentationConfig:
    """Load a specific preset from the augmentation config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Augmentation config not found at {config_path}")
    
    d = yaml.safe_load(config_path.read_text())
    
    if preset_name not in d.get("presets", {}):
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(d.get('presets', {}).keys())}")
    
    # Start with base config
    base_config = d.copy()
    # Override with preset values
    preset_values = d["presets"][preset_name]
    base_config.update(preset_values)
    
    return AugmentationConfig(
        target_size=int(base_config["target_size"]),
        augs_per_image=int(base_config["augs_per_image"]),
        splits=str(base_config["splits"]),
        out_folder=str(base_config["out_folder"]),
        rotation_deg=float(base_config["rotation_deg"]),
        scale_low=float(base_config["scale_low"]),
        scale_high=float(base_config["scale_high"]),
        translation_frac=float(base_config["translation_frac"]),
        pad_margin=int(base_config["pad_margin"]),
        tag_class_id=int(base_config["tag_class_id"]),
        jpeg_quality=int(base_config["jpeg_quality"]),
        opencv_num_threads=int(base_config["opencv_num_threads"]),
        keep_existing=bool(base_config["keep_existing"]),
        max_images=int(base_config["max_images"]) if base_config["max_images"] is not None else None,
        presets=dict(base_config.get("presets", {}))
    )
