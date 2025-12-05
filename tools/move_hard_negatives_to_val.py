"""
Move 50-65 hard negative tiles from train to val.
"""

import random
from pathlib import Path
import shutil

TARGET_DATASET = Path("datasets/rcp_bbox_v7_speed")
MOVE_COUNT_MIN = 50
MOVE_COUNT_MAX = 65

def move_hard_negatives_to_val():
    """Move hard negatives from train to val."""
    print("="*80)
    print("MOVING HARD NEGATIVES TO VAL")
    print("="*80)
    
    train_img_dir = TARGET_DATASET / "images" / "train" / "tiled1536"
    train_lbl_dir = TARGET_DATASET / "labels" / "train" / "tiled1536"
    val_img_dir = TARGET_DATASET / "images" / "val" / "tiled1536"
    val_lbl_dir = TARGET_DATASET / "labels" / "val" / "tiled1536"
    
    # Find all hard negatives (empty labels) in train
    train_labels = list(train_lbl_dir.glob("*.txt"))
    hard_negatives = [f for f in train_labels if f.stat().st_size == 0]
    
    print(f"Found {len(hard_negatives)} hard negatives in train")
    
    if len(hard_negatives) < MOVE_COUNT_MIN:
        print(f"ERROR: Not enough hard negatives ({len(hard_negatives)}) to move {MOVE_COUNT_MIN}")
        return
    
    # Determine how many to move (between min and max, or all if less than max)
    move_count = min(MOVE_COUNT_MAX, max(MOVE_COUNT_MIN, len(hard_negatives)))
    
    # Randomly select tiles to move
    random.seed(42)
    selected = random.sample(hard_negatives, move_count)
    
    print(f"Moving {move_count} hard negatives to val...")
    
    moved_count = 0
    for lbl_file in selected:
        img_name = lbl_file.stem
        
        # Source paths
        src_img = train_img_dir / f"{img_name}.jpg"
        src_lbl = train_lbl_dir / f"{img_name}.txt"
        
        # Destination paths
        dst_img = val_img_dir / f"{img_name}.jpg"
        dst_lbl = val_lbl_dir / f"{img_name}.txt"
        
        # Move files
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))
        
        moved_count += 1
        
        if moved_count % 10 == 0:
            print(f"  Moved {moved_count}/{move_count}...")
    
    print(f"\nMoved {moved_count} hard negatives to val")
    
    # Verify final counts
    train_labels = list(train_lbl_dir.glob("*.txt"))
    train_empty = [f for f in train_labels if f.stat().st_size == 0]
    train_non_empty = [f for f in train_labels if f.stat().st_size > 0]
    
    val_labels = list(val_lbl_dir.glob("*.txt"))
    val_empty = [f for f in val_labels if f.stat().st_size == 0]
    val_non_empty = [f for f in val_labels if f.stat().st_size > 0]
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"TRAIN:")
    print(f"  Total: {len(train_labels)} tiles")
    print(f"  - With panels: {len(train_non_empty)}")
    print(f"  - Hard negatives: {len(train_empty)}")
    print(f"  - Hard negative ratio: {len(train_empty)/len(train_labels):.1%}")
    print()
    print(f"VAL:")
    print(f"  Total: {len(val_labels)} tiles")
    print(f"  - With panels: {len(val_non_empty)}")
    print(f"  - Hard negatives: {len(val_empty)}")
    print(f"  - Hard negative ratio: {len(val_empty)/len(val_labels):.1%}")

if __name__ == "__main__":
    move_hard_negatives_to_val()

