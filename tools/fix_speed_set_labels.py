"""
Fix labels in speed-set dataset by re-converting from source.
"""
from pathlib import Path
import shutil

SOURCE_DATASET = "datasets/rcp_bbox_v6_reshuffled"
TARGET_DATASET = "datasets/rcp_bbox_v7_speed"

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

def fix_labels():
    """Re-convert all labels from source dataset."""
    print("Fixing labels in speed-set dataset...")
    
    for split in ['train', 'val', 'stress']:
        print(f"\nProcessing {split}...")
        target_img_dir = Path(TARGET_DATASET) / "images" / split / "tiled1536"
        target_lbl_dir = Path(TARGET_DATASET) / "labels" / split / "tiled1536"
        
        if not target_img_dir.exists():
            print(f"  Skipping {split} - directory not found")
            continue
        
        # Get all images
        image_files = list(target_img_dir.glob("*.jpg"))
        print(f"  Found {len(image_files)} images")
        
        fixed = 0
        empty = 0
        
        for img_file in image_files:
            img_name = img_file.stem
            
            # Try to find source label
            source_lbl = None
            for source_split in ['train', 'val']:
                source_lbl_path = Path(SOURCE_DATASET) / "labels" / source_split / "tiled1536" / f"{img_name}.txt"
                if source_lbl_path.exists():
                    source_lbl = source_lbl_path
                    break
            
            target_lbl_path = target_lbl_dir / f"{img_name}.txt"
            
            if source_lbl:
                # Convert from source
                convert_label_to_panel_only(source_lbl, target_lbl_path)
                # Check if result is empty
                if target_lbl_path.exists() and target_lbl_path.stat().st_size > 0:
                    fixed += 1
                else:
                    empty += 1
            else:
                # No source label found - keep as empty (hard negative)
                target_lbl_path.parent.mkdir(parents=True, exist_ok=True)
                target_lbl_path.write_text('')
                empty += 1
        
        print(f"  Fixed: {fixed} labels with panels")
        print(f"  Empty: {empty} labels (hard negatives or missing source)")

if __name__ == "__main__":
    fix_labels()

