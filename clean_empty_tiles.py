#!/usr/bin/env python3
"""
Clean empty tiles - Remove tiles with no panel annotations
"""
import os
import glob
from pathlib import Path

def clean_empty_tiles():
    """Remove tiles that have no panel annotations (empty label files)"""
    print("=== Clean Empty Tiles ===")
    print("Removing tiles with no panel annotations...")
    print("💡 This improves training by removing noise")
    
    # Base directory
    base_dir = "datasets/rcp_dual_seg"
    
    # Check if tiled directories exist
    tiled_dirs = [
        f"{base_dir}/images/train/tiled1k",
        f"{base_dir}/images/val/tiled1k", 
        f"{base_dir}/images/test/tiled1k",
        f"{base_dir}/labels/train/tiled1k",
        f"{base_dir}/labels/val/tiled1k",
        f"{base_dir}/labels/test/tiled1k"
    ]
    
    missing_dirs = [d for d in tiled_dirs if not os.path.exists(d)]
    if missing_dirs:
        print("❌ Tiled directories not found:")
        for d in missing_dirs:
            print(f"   {d}")
        print("💡 Run Option 7 (Tile fullsize images) first")
        return False
    
    total_removed = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        img_dir = f"{base_dir}/images/{split}/tiled1k"
        lbl_dir = f"{base_dir}/labels/{split}/tiled1k"
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"⚠️  Skipping {split} - directories not found")
            continue
        
        # Get all label files
        label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        print(f"\n📁 Processing {split}: {len(label_files)} label files")
        
        removed_count = 0
        
        for label_file in label_files:
            # Check if label file is empty
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                
                if not content:  # Empty file
                    # Get corresponding image file
                    label_name = Path(label_file).stem
                    img_file = os.path.join(img_dir, f"{label_name}.jpg")
                    
                    # Remove both files
                    if os.path.exists(img_file):
                        os.remove(img_file)
                        print(f"   🗑️  Removed empty tile: {label_name}")
                        removed_count += 1
                    
                    os.remove(label_file)
                    
            except Exception as e:
                print(f"   ⚠️  Error processing {label_file}: {e}")
        
        print(f"   ✅ Removed {removed_count} empty tiles from {split}")
        total_removed += removed_count
    
    print(f"\n📊 Summary:")
    print(f"   Total empty tiles removed: {total_removed}")
    print(f"   💡 Training will now focus on tiles with actual panels")
    
    return True

if __name__ == "__main__":
    clean_empty_tiles() 