"""
create_speed_set.py
-------------------
Create a lightweight panel-only speed-set dataset for fast iteration (v7).

Key Features:
- Panel-only (nc:1, names: ['panel']) - drop tag class
- Hard negatives: 25-30% of train (tiles with 0 panels, panel-like clutter)
- Small-panel quota: ≥25% of masks from smallest-area quartile
- Stress split: Sheet-level hard holdout (separate from tuning)
- Keep tiles at 1536px, train with imgsz=1024

Requirements:
- Train: ~1,400 tiles (25-30% hard negatives)
- Val: 300 tiles (no hard negatives, clean for tuning)
- Stress: 10-15% sheet-level holdout (only check at end of iteration)
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

# Configuration
SOURCE_DATASET = "datasets/rcp_bbox_v6_reshuffled"
TARGET_DATASET = "datasets/rcp_bbox_v7_speed"
EXCEL_ANALYSIS = "analysis_v6_reshuffled.xlsx"

TRAIN_TARGET = 1400  # Total train tiles
HARD_NEGATIVE_RATIO_MIN = 0.25  # 25% minimum
HARD_NEGATIVE_RATIO_MAX = 0.30  # 30% maximum
HARD_NEGATIVE_RATIO = 0.275  # Middle: 27.5%
VAL_TARGET = 300
STRESS_PERCENT = 0.12  # 12% of sheets (between 10-15%)
SMALL_PANEL_QUOTA = 0.25  # ≥25% of masks from smallest quartile

# Density bin distribution (based on v6 analysis)
DENSITY_DISTRIBUTION = {
    '<=2': 0.15,    # very_sparse
    '3-5': 0.20,    # sparse
    '6-10': 0.35,   # normal (most common)
    '11-15': 0.20,  # dense
    '>15': 0.10     # very_dense
}


def load_excel_analysis(excel_path):
    """Load panel counts and density information from Excel."""
    print(f"Loading analysis from {excel_path}...")
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Analysis file not found: {excel_path}")
    
    # Read the Images sheet
    df = pd.read_excel(excel_path, sheet_name='Images')
    
    # Extract relevant columns (using actual column names from Excel)
    image_data = {}
    for _, row in df.iterrows():
        img_name = str(row.get('filename', '')).replace('.jpg', '').replace('.png', '')
        panel_count = row.get('num_panels', 0)
        
        if pd.isna(panel_count):
            panel_count = 0
        
        image_data[img_name] = {
            'panel_count': int(panel_count),
            'density_bin': categorize_density(int(panel_count))
        }
    
    print(f"  Loaded {len(image_data)} images")
    return image_data


def categorize_density(panel_count):
    """Categorize image by panel density."""
    if panel_count <= 2:
        return '<=2'
    elif panel_count <= 5:
        return '3-5'
    elif panel_count <= 10:
        return '6-10'
    elif panel_count <= 15:
        return '11-15'
    else:
        return '>15'


def extract_sheet_name(tile_filename):
    """Extract sheet name from tile filename.
    
    Examples:
    - x10_s1_r4300_c10750.jpg -> x10
    - x62_s3_r0_c4300.jpg -> x62
    """
    # Remove extension
    name = Path(tile_filename).stem
    
    # Extract sheet name (everything before first underscore after 'x')
    if '_' in name:
        parts = name.split('_')
        # First part should be like 'x10' or 'x62'
        sheet_part = parts[0]
        if sheet_part.startswith('x'):
            return sheet_part
        else:
            return parts[0]  # Fallback
    else:
        return name.split('.')[0]


def get_all_tiles(source_dataset):
    """Get all tiles from source dataset with metadata."""
    print(f"Scanning tiles from {source_dataset}...")
    
    train_dir = Path(source_dataset) / "images" / "train" / "tiled1536"
    val_dir = Path(source_dataset) / "images" / "val" / "tiled1536"
    
    tiles = {
        'train': [],
        'val': []
    }
    
    # Scan train tiles
    if train_dir.exists():
        for img_file in train_dir.glob("*.jpg"):
            tile_name = img_file.stem
            tiles['train'].append({
                'name': tile_name,
                'image_path': str(img_file),
                'label_path': str(train_dir.parent.parent / "labels" / "train" / "tiled1536" / f"{tile_name}.txt"),
                'sheet': extract_sheet_name(tile_name)
            })
    
    # Scan val tiles
    if val_dir.exists():
        for img_file in val_dir.glob("*.jpg"):
            tile_name = img_file.stem
            tiles['val'].append({
                'name': tile_name,
                'image_path': str(img_file),
                'label_path': str(val_dir.parent.parent / "labels" / "val" / "tiled1536" / f"{tile_name}.txt"),
                'sheet': extract_sheet_name(tile_name)
            })
    
    print(f"  Found {len(tiles['train'])} train tiles, {len(tiles['val'])} val tiles")
    return tiles


def identify_hard_sheets(tiles, image_data):
    """Identify hard sheets (high density, many panels)."""
    print("Identifying hard sheets...")
    
    sheet_stats = defaultdict(lambda: {'tiles': [], 'total_panels': 0, 'avg_panels': 0, 'max_density': ''})
    
    # Aggregate by sheet
    for split in ['train', 'val']:
        for tile in tiles[split]:
            sheet = tile['sheet']
            tile_name = tile['name']
            
            if tile_name in image_data:
                panel_count = image_data[tile_name]['panel_count']
                density_bin = image_data[tile_name]['density_bin']
                
                sheet_stats[sheet]['tiles'].append(tile)
                sheet_stats[sheet]['total_panels'] += panel_count
                
                # Track max density bin
                density_order = {'<=2': 0, '3-5': 1, '6-10': 2, '11-15': 3, '>15': 4}
                current_max = sheet_stats[sheet].get('max_density_order', -1)
                if density_order[density_bin] > current_max:
                    sheet_stats[sheet]['max_density'] = density_bin
                    sheet_stats[sheet]['max_density_order'] = density_order[density_bin]
    
    # Calculate averages
    for sheet, stats in sheet_stats.items():
        if len(stats['tiles']) > 0:
            stats['avg_panels'] = stats['total_panels'] / len(stats['tiles'])
    
    # Sort by difficulty (high density, many panels)
    hard_sheets = sorted(
        sheet_stats.items(),
        key=lambda x: (x[1]['max_density_order'], x[1]['avg_panels']),
        reverse=True
    )
    
    print(f"  Found {len(hard_sheets)} unique sheets")
    return hard_sheets, sheet_stats


def select_stress_sheets(hard_sheets, total_sheets):
    """Select 10-15% of hardest sheets for stress set (hard examples)."""
    if total_sheets == 0:
        print("  WARNING: No sheets found, cannot select stress sheets")
        return set()
    
    num_stress_sheets = max(1, int(total_sheets * STRESS_PERCENT))
    
    # Select hardest sheets (highest density, worst contrast)
    stress_sheets = [sheet for sheet, _ in hard_sheets[:num_stress_sheets]]
    
    print(f"  Selected {len(stress_sheets)} sheets for stress set ({len(stress_sheets)/total_sheets*100:.1f}%)")
    print(f"  Stress sheets: {stress_sheets}")
    
    return set(stress_sheets)


def assign_density_bins(tiles, image_data):
    """Assign density bins to tiles."""
    for split in ['train', 'val']:
        for tile in tiles[split]:
            tile_name = tile['name']
            if tile_name in image_data:
                tile['density_bin'] = image_data[tile_name]['density_bin']
                tile['panel_count'] = image_data[tile_name]['panel_count']
            else:
                tile['density_bin'] = '<=2'  # Default
                tile['panel_count'] = 0


def load_panel_areas(label_path):
    """Load panel areas from YOLO label file (panel-only, class 0)."""
    if not os.path.exists(label_path):
        return []
    
    areas = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id == 0:  # Panel class only
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                    # Area in normalized coordinates (will be scaled by tile size)
                    area = width_norm * height_norm
                    areas.append(area)
    
    return areas


def find_hard_negatives(all_tiles, positive_sheets, image_data):
    """Find hard negatives: tiles with 0 panels from sheets not in positive_sheets."""
    print("Finding hard negatives...")
    
    hard_negatives = []
    checked = 0
    for tile in all_tiles:
        tile_name = tile['name']
        sheet = tile['sheet']
        checked += 1
        
        # Must be from a different sheet than positive tiles
        if sheet not in positive_sheets:
            # Check if it has 0 panels
            panel_count = image_data.get(tile_name, {}).get('panel_count', 0)
            if panel_count == 0:
                # Verify label file is empty or has no panels
                label_path = Path(tile['label_path'])
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                            # Empty or only tag classes (class 1) - no panel class (class 0)
                            if not content:
                                hard_negatives.append(tile)
                            else:
                                # Check if any line has class 0 (panel)
                                has_panel = False
                                for line in content.split('\n'):
                                    if line.strip():
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            cls_id = int(parts[0])
                                            if cls_id == 0:  # Panel class
                                                has_panel = True
                                                break
                                if not has_panel:
                                    hard_negatives.append(tile)
                    except Exception as e:
                        # Skip if can't read label
                        pass
    
    print(f"  Checked {checked} tiles, found {len(hard_negatives)} potential hard negatives")
    return hard_negatives


def enforce_small_panel_quota(positive_tiles, quota=SMALL_PANEL_QUOTA):
    """Ensure >=quota% of masks come from smallest-area quartile."""
    print(f"Enforcing small-panel quota (>={quota*100:.0f}%)...")
    
    # Calculate panel areas for all tiles
    tile_areas = []
    for tile in positive_tiles:
        label_path = Path(tile['label_path'])
        areas = load_panel_areas(label_path)
        if areas:
            # Use minimum area per tile (smallest panel)
            tile['min_area'] = min(areas)
            tile['num_panels'] = len(areas)
            tile_areas.append(tile['min_area'])
        else:
            tile['min_area'] = float('inf')
            tile['num_panels'] = 0
    
    if not tile_areas:
        print("  No panels found, skipping quota enforcement")
        return positive_tiles
    
    # Find quartile threshold
    quartile_threshold = np.percentile(tile_areas, 25)  # 25th percentile = smallest quartile
    
    # Count panels in smallest quartile
    small_panel_tiles = [t for t in positive_tiles if t.get('min_area', float('inf')) <= quartile_threshold]
    total_panels = sum(t.get('num_panels', 0) for t in positive_tiles)
    small_panel_count = sum(t.get('num_panels', 0) for t in small_panel_tiles)
    
    current_ratio = small_panel_count / total_panels if total_panels > 0 else 0
    
    print(f"  Current small-panel ratio: {current_ratio:.1%} (target: ≥{quota*100:.0f}%)")
    
    if current_ratio < quota:
        # Need to add more small-panel tiles
        needed_ratio = quota - current_ratio
        needed_panels = int(total_panels * needed_ratio)
        
        # Find more small-panel tiles not yet selected
        available_small = [t for t in small_panel_tiles if t.get('num_panels', 0) > 0]
        if available_small:
            # Prioritize tiles with many small panels
            available_small.sort(key=lambda x: x.get('num_panels', 0), reverse=True)
            # Add tiles until quota is met
            added_panels = 0
            for tile in available_small:
                if added_panels >= needed_panels:
                    break
                if tile not in positive_tiles:
                    positive_tiles.append(tile)
                    added_panels += tile.get('num_panels', 0)
            
            print(f"  Added {len([t for t in available_small if t in positive_tiles])} small-panel tiles")
    
    return positive_tiles


def convert_label_to_panel_only(label_path, output_path):
    """Convert YOLO label to panel-only (remove tag class, keep only class 0)."""
    label_path_obj = Path(label_path)
    output_path_obj = Path(output_path)
    
    if not label_path_obj.exists():
        # Create empty file for hard negatives
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text('')
        return
    
    panel_lines = []
    with open(label_path_obj, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # At least class + 4 coords (bbox) or more (segmentation)
                try:
                    cls_id = int(parts[0])
                    if cls_id == 0:  # Panel class only
                        panel_lines.append(line.strip())
                except ValueError:
                    # Skip malformed lines
                    continue
    
    # Write panel-only labels
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if panel_lines:
        output_path_obj.write_text('\n'.join(panel_lines) + '\n')
    else:
        output_path_obj.write_text('')


def stratified_sample(tiles, target_count, density_distribution, exclude_sheets=None):
    """Sample tiles stratified by density bin."""
    if exclude_sheets is None:
        exclude_sheets = set()
    
    # Filter out excluded sheets
    available_tiles = [t for t in tiles if t['sheet'] not in exclude_sheets]
    
    # Group by density bin
    bins = defaultdict(list)
    for tile in available_tiles:
        bin_name = tile.get('density_bin', '<=2')
        bins[bin_name].append(tile)
    
    # Sample from each bin
    selected = []
    for bin_name, bin_tiles in bins.items():
        target_bin_count = int(target_count * density_distribution.get(bin_name, 0.2))
        target_bin_count = min(target_bin_count, len(bin_tiles))
        
        sampled = random.sample(bin_tiles, target_bin_count)
        selected.extend(sampled)
        
        print(f"  {bin_name}: {len(sampled)}/{len(bin_tiles)} tiles")
    
    # If we're short, fill from largest bins
    if len(selected) < target_count:
        remaining = target_count - len(selected)
        all_remaining = [t for t in available_tiles if t not in selected]
        if len(all_remaining) >= remaining:
            selected.extend(random.sample(all_remaining, remaining))
    
    return selected


def create_dataset_structure(target_dataset):
    """Create directory structure for new dataset."""
    print(f"Creating dataset structure: {target_dataset}")
    
    base = Path(target_dataset)
    
    for split in ['train', 'val', 'stress']:
        (base / "images" / split / "tiled1536").mkdir(parents=True, exist_ok=True)
        (base / "labels" / split / "tiled1536").mkdir(parents=True, exist_ok=True)
    
    print("  Directory structure created")


def copy_files(selected_tiles, target_dataset, split, is_hard_negative=False):
    """Copy selected tiles to target dataset, converting labels to panel-only."""
    print(f"Copying {len(selected_tiles)} tiles to {split}...")
    
    copied = 0
    for tile in selected_tiles:
        img_src = Path(tile['image_path'])
        lbl_src = Path(tile['label_path'])
        
        img_dst = Path(target_dataset) / "images" / split / "tiled1536" / img_src.name
        lbl_dst = Path(target_dataset) / "labels" / split / "tiled1536" / lbl_src.name
        
        if img_src.exists():
            shutil.copy2(img_src, img_dst)
            copied += 1
        
        # Convert label to panel-only (remove tag class)
        convert_label_to_panel_only(lbl_src, lbl_dst)
    
    print(f"  Copied {copied} images and converted labels to panel-only")
    return copied


def create_yaml(target_dataset):
    """Create dataset YAML file (panel-only, stress NOT in YAML)."""
    yaml_content = f"""path: {Path(target_dataset).absolute()}
train: images/train/tiled1536
val: images/val/tiled1536
# stress kept separate, not in YAML (only check at end of iteration)
nc: 1
names: ['panel']
"""
    
    yaml_path = Path(target_dataset) / "dataset_tiled1536.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"  Created {yaml_path} (panel-only, stress excluded)")


def validate_dataset(target_dataset, summary):
    """Validate the created dataset."""
    print("\nValidating dataset...")
    
    for split in ['train', 'val', 'stress']:
        img_dir = Path(target_dataset) / "images" / split / "tiled1536"
        lbl_dir = Path(target_dataset) / "labels" / split / "tiled1536"
        
        img_count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        
        print(f"  {split}: {img_count} images, {lbl_count} labels")
        
        if img_count != lbl_count:
            print(f"    WARNING: Mismatch in {split}!")
    
    # Validate hard negatives ratio
    train_count = summary.get('train_count', 0)
    hard_neg_count = summary.get('hard_negative_count', 0)
    if train_count > 0:
        hard_neg_ratio = hard_neg_count / train_count
        print(f"\n  Hard negatives: {hard_neg_count}/{train_count} ({hard_neg_ratio:.1%})")
        if hard_neg_ratio < HARD_NEGATIVE_RATIO_MIN or hard_neg_ratio > HARD_NEGATIVE_RATIO_MAX:
            print(f"    WARNING: Hard negative ratio outside target range ({HARD_NEGATIVE_RATIO_MIN:.0%}-{HARD_NEGATIVE_RATIO_MAX:.0%})")
        else:
            print(f"    OK: Hard negative ratio within target range")
    
    # Validate small-panel quota
    small_panel_ratio = summary.get('small_panel_ratio', 0)
    if small_panel_ratio > 0:
        print(f"\n  Small-panel quota: {small_panel_ratio:.1%} (target: ≥{SMALL_PANEL_QUOTA*100:.0f}%)")
        if small_panel_ratio < SMALL_PANEL_QUOTA:
            print(f"    WARNING: Small-panel quota below target!")
        else:
            print(f"    OK: Small-panel quota met")
    
    print("  Validation complete")


def save_summary(target_dataset, summary):
    """Save summary report."""
    summary_path = Path(target_dataset) / "speed_set_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPEED-SET DATASET SUMMARY (v7)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Source: {SOURCE_DATASET}\n")
        f.write(f"Target: {TARGET_DATASET}\n\n")
        
        f.write("SPLIT COUNTS:\n")
        f.write(f"  Train: {summary['train_count']} tiles")
        if 'hard_negative_count' in summary:
            f.write(f" ({summary['hard_negative_count']} hard negatives, {summary.get('hard_negative_ratio', 0):.1%})")
        f.write(f"\n")
        f.write(f"  Val:   {summary['val_count']} tiles\n")
        f.write(f"  Stress: {summary['stress_count']} tiles\n")
        f.write(f"  Total: {summary['total_count']} tiles\n\n")
        
        f.write("DENSITY DISTRIBUTION:\n")
        for split in ['train', 'val', 'stress']:
            f.write(f"\n{split.upper()}:\n")
            for bin_name, count in summary['density_dist'][split].items():
                f.write(f"  {bin_name}: {count} tiles\n")
        
        if 'small_panel_ratio' in summary:
            f.write(f"\nSMALL-PANEL QUOTA: {summary['small_panel_ratio']:.1%} (target: >={SMALL_PANEL_QUOTA*100:.0f}%)\n")
        
        f.write(f"\nSTRESS SHEETS: {', '.join(summary['stress_sheets'])}\n")
    
    print(f"\nSummary saved to {summary_path}")


def main():
    """Main function to create speed-set."""
    print("="*80)
    print("CREATING SPEED-SET DATASET (v7)")
    print("="*80)
    print()
    
    # Set random seed
    random.seed(42)
    
    # Step 1: Load analysis
    image_data = load_excel_analysis(EXCEL_ANALYSIS)
    
    # Step 2: Get all tiles
    tiles_dict = get_all_tiles(SOURCE_DATASET)
    
    # Step 3: Assign density bins
    all_tiles = tiles_dict['train'] + tiles_dict['val']
    assign_density_bins({'train': tiles_dict['train'], 'val': tiles_dict['val']}, image_data)
    
    # Step 4: Identify hard sheets
    hard_sheets, sheet_stats = identify_hard_sheets(tiles_dict, image_data)
    unique_sheets = len(sheet_stats)
    
    # Step 5: Select stress sheets
    stress_sheets = select_stress_sheets(hard_sheets, unique_sheets)
    
    # Step 6: Extract stress tiles (all tiles from stress sheets)
    stress_tiles = [t for t in all_tiles if t['sheet'] in stress_sheets]
    
    # Step 7: Sample positive train/val (excluding stress sheets)
    # Calculate positive tile target (train - hard negatives)
    hard_neg_target = int(TRAIN_TARGET * HARD_NEGATIVE_RATIO)
    positive_train_target = TRAIN_TARGET - hard_neg_target
    
    positive_train_tiles = stratified_sample(
        tiles_dict['train'],
        positive_train_target,
        DENSITY_DISTRIBUTION,
        exclude_sheets=stress_sheets
    )
    
    # Step 8: Enforce small-panel quota
    positive_train_tiles = enforce_small_panel_quota(positive_train_tiles)
    
    # Step 9: Find hard negatives
    # Get sheets used for positive train tiles
    positive_train_sheets = set(t['sheet'] for t in positive_train_tiles)
    all_available_tiles = [t for t in all_tiles if t['sheet'] not in stress_sheets]
    hard_negatives = find_hard_negatives(all_available_tiles, positive_train_sheets, image_data)
    
    # Sample hard negatives
    if len(hard_negatives) >= hard_neg_target:
        selected_hard_negatives = random.sample(hard_negatives, hard_neg_target)
    else:
        print(f"  WARNING: Only {len(hard_negatives)} hard negatives available, using all")
        selected_hard_negatives = hard_negatives
    
    # Combine positive and hard negative tiles
    train_tiles = positive_train_tiles + selected_hard_negatives
    random.shuffle(train_tiles)  # Shuffle to mix positives and negatives
    
    # Step 10: Sample val (no hard negatives, no sheet overlap with train)
    train_sheets = set(t['sheet'] for t in train_tiles)
    val_tiles = stratified_sample(
        tiles_dict['val'],
        VAL_TARGET,
        DENSITY_DISTRIBUTION,
        exclude_sheets=stress_sheets | train_sheets
    )
    
    # Step 11: Create dataset structure
    create_dataset_structure(TARGET_DATASET)
    
    # Step 12: Copy files
    train_count = copy_files(train_tiles, TARGET_DATASET, 'train')
    val_count = copy_files(val_tiles, TARGET_DATASET, 'val')
    stress_count = copy_files(stress_tiles, TARGET_DATASET, 'stress')
    
    # Step 13: Create YAML
    create_yaml(TARGET_DATASET)
    
    # Step 14: Calculate density distribution and metrics
    def count_by_density(tile_list):
        counts = defaultdict(int)
        for tile in tile_list:
            bin_name = tile.get('density_bin', '<=2')
            counts[bin_name] += 1
        return dict(counts)
    
    # Calculate small-panel ratio
    total_panels = sum(t.get('num_panels', 0) for t in positive_train_tiles)
    if total_panels > 0:
        tile_areas = [t.get('min_area', float('inf')) for t in positive_train_tiles if t.get('min_area', float('inf')) != float('inf')]
        if tile_areas:
            quartile_threshold = np.percentile(tile_areas, 25)
            small_panel_tiles = [t for t in positive_train_tiles if t.get('min_area', float('inf')) <= quartile_threshold]
            small_panel_count = sum(t.get('num_panels', 0) for t in small_panel_tiles)
            small_panel_ratio = small_panel_count / total_panels
        else:
            small_panel_ratio = 0
    else:
        small_panel_ratio = 0
    
    summary = {
        'train_count': train_count,
        'val_count': val_count,
        'stress_count': stress_count,
        'total_count': train_count + val_count + stress_count,
        'hard_negative_count': len(selected_hard_negatives),
        'hard_negative_ratio': len(selected_hard_negatives) / train_count if train_count > 0 else 0,
        'small_panel_ratio': small_panel_ratio,
        'density_dist': {
            'train': count_by_density(train_tiles),
            'val': count_by_density(val_tiles),
            'stress': count_by_density(stress_tiles)
        },
        'stress_sheets': sorted(list(stress_sheets))
    }
    
    # Step 15: Validate
    validate_dataset(TARGET_DATASET, summary)
    
    # Step 16: Save summary
    save_summary(TARGET_DATASET, summary)
    
    print("\n" + "="*80)
    print("SPEED-SET CREATION COMPLETE!")
    print("="*80)
    print(f"\nDataset: {TARGET_DATASET}")
    print(f"Train: {train_count} tiles ({len(selected_hard_negatives)} hard negatives, {len(selected_hard_negatives)/train_count:.1%})")
    print(f"Val: {val_count} tiles")
    print(f"Stress: {stress_count} tiles")
    print(f"\nPanel-only dataset (nc:1, names: ['panel'])")
    print(f"Ready for training with imgsz=1024!")
    print(f"\nMetrics to track:")
    print(f"  - Recall@0.4 (on val every epoch, stress only at end)")
    print(f"  - FP/image (on val every epoch, stress only at end)")
    print(f"  - Worst-bin recall (on val every epoch, stress only at end)")


if __name__ == "__main__":
    main()

