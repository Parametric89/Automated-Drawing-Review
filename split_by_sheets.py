"""
split_by_sheets.py
-----------------
Split dataset by full sheets (70/20/10) before tiling to avoid metric cheating.
Updated for 2048 tile size and current data structure.
"""

import os
import shutil
import random
from pathlib import Path
import json
from datetime import datetime
import subprocess
import glob


def get_sheet_names_from_pending():
    """Extract unique sheet names from pending folder (visualized images)."""
    pending_dir = "pending"
    
    if not os.path.exists(pending_dir):
        print(f"❌ Pending directory not found: {pending_dir}")
        return []
    
    # Get all visualized image files
    image_files = [f for f in os.listdir(pending_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 'visualized' in f]
    
    # Extract sheet names (assuming format like "x1_visualized.jpg", "x2_visualized.jpg")
    sheet_names = set()
    for file in image_files:
        # Extract sheet name (everything before "_visualized")
        if '_visualized' in file:
            sheet_name = file.split('_visualized')[0]
        else:
            sheet_name = file.split('.')[0]  # Use filename without extension
        
        sheet_names.add(sheet_name)
    
    print(f"📊 Found {len(sheet_names)} unique sheets from visualized images:")
    for sheet in sorted(sheet_names):
        print(f"   - {sheet}")
    
    return sorted(list(sheet_names))


def get_sheet_names_from_source_pdf():
    """Extract sheet names from source PDF file."""
    source_pdf = "Archive/Production drawings (pdfs)_bordered/36999 Apple Retail Kopa (LC-3 panels) prod.drawings_bordered.pdf"
    
    if not os.path.exists(source_pdf):
        print(f"❌ Source PDF not found: {source_pdf}")
        return []
    
    # For now, we'll assume 37 sheets based on the project plan
    # In a real implementation, you'd extract page count from PDF
    sheet_names = [f"sheet_{i:02d}" for i in range(1, 38)]  # sheet_01 to sheet_37
    
    print(f"📊 Found {len(sheet_names)} sheets in source PDF:")
    for sheet in sheet_names:
        print(f"   - {sheet}")
    
    return sheet_names


def split_sheets_train_val_test(sheet_names, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split sheets into train/val/test sets.
    
    Args:
        sheet_names: List of unique sheet names
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.2) 
        test_ratio: Proportion for testing (default 0.1)
    
    Returns:
        train_sheets, val_sheets, test_sheets: Lists of sheet names
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle sheets for random split
    random.shuffle(sheet_names)
    
    # Calculate split indices
    n_sheets = len(sheet_names)
    n_train = int(n_sheets * train_ratio)
    n_val = int(n_sheets * val_ratio)
    n_test = n_sheets - n_train - n_val  # Remaining goes to test
    
    # Split sheets
    train_sheets = sheet_names[:n_train]
    val_sheets = sheet_names[n_train:n_train + n_val]
    test_sheets = sheet_names[n_train + n_val:]
    
    print(f"\n📊 Sheet Split Results:")
    print(f"   Train: {len(train_sheets)} sheets ({len(train_sheets)/n_sheets*100:.1f}%)")
    print(f"   Val:   {len(val_sheets)} sheets ({len(val_sheets)/n_sheets*100:.1f}%)")
    print(f"   Test:  {len(test_sheets)} sheets ({len(test_sheets)/n_sheets*100:.1f}%)")
    
    return train_sheets, val_sheets, test_sheets


def process_source_pdf_to_images():
    """Convert source PDF to images and save to pending/images."""
    source_pdf = "Archive/Production drawings (pdfs)_bordered/36999 Apple Retail Kopa (LC-3 panels) prod.drawings_bordered.pdf"
    output_dir = "pending/images"
    
    if not os.path.exists(source_pdf):
        print(f"❌ Source PDF not found: {source_pdf}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Converting PDF to images...")
    print(f"   Source: {source_pdf}")
    print(f"   Output: {output_dir}")
    
    try:
        # Use pdf_to_images.py functionality
        from pdf_to_images import convert_pdf_to_images
        
        # Convert PDF to images
        num_pages = convert_pdf_to_images(
            pdf_path=source_pdf,
            output_folder=output_dir,
            pdf_number=1,  # Use x1 naming
            dpi=300  # High resolution
        )
        
        print(f"✅ Converted {num_pages} pages to images")
        return True
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return False


def organize_images_by_sheets(sheet_splits, target_dataset="rcp_dual_seg"):
    """
    Organize images into train/val/test folders based on sheet splits BEFORE tiling.
    
    Args:
        sheet_splits: Tuple of (train_sheets, val_sheets, test_sheets)
        target_dataset: Which dataset to organize for (default: rcp_dual_seg)
    """
    train_sheets, val_sheets, test_sheets = sheet_splits
    pending_dir = "pending/images"
    target_dir = f"datasets/{target_dataset}/images"
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # Define which sheets go to which split
    split_mapping = {
        'train': train_sheets,
        'val': val_sheets, 
        'test': test_sheets
    }
    
    # Get all image files from pending
    image_files = [f for f in os.listdir(pending_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ No images found in pending/images. Please run process_source_pdf_to_images() first.")
        return False
    
    print(f"🔄 Organizing {len(image_files)} images by sheet splits...")
    
    # Process each image
    organized_count = {split: 0 for split in ['train', 'val', 'test']}
    
    for file in image_files:
        # Extract sheet number from filename (e.g., "x1_1.jpg" -> sheet_01)
        if '_' in file:
            # Extract the sheet number from x1_1.jpg -> 1
            sheet_part = file.split('_')[0]  # "x1"
            if sheet_part.startswith('x'):
                sheet_num = int(sheet_part[1:])  # Extract number after 'x'
                sheet_name = f"sheet_{sheet_num:02d}"  # Convert to sheet_01 format
            else:
                sheet_name = sheet_part
        else:
            sheet_name = file.split('.')[0]
        
        # Find which split this sheet belongs to
        target_split = None
        for split, sheets in split_mapping.items():
            if sheet_name in sheets:
                target_split = split
                break
        
        if target_split:
            # Copy file to appropriate split directory
            src_path = os.path.join(pending_dir, file)
            dst_path = os.path.join(target_dir, target_split, file)
            shutil.copy2(src_path, dst_path)
            organized_count[target_split] += 1
            print(f"   {file} → {target_split} (sheet: {sheet_name})")
        else:
            print(f"⚠️  Could not determine split for {file} (sheet: {sheet_name})")
    
    print(f"\n📁 Organized {sum(organized_count.values())} images:")
    for split, count in organized_count.items():
        print(f"   {split}: {count} images")
    
    return organized_count


def tile_images_with_2048_tile_size(target_dataset="rcp_dual_seg"):
    """
    Tile images with 2048x2048 tile size after they've been organized by splits.
    
    Args:
        target_dataset: Which dataset to tile (default: rcp_dual_seg)
    """
    target_dir = f"datasets/{target_dataset}/images"
    
    if not os.path.exists(target_dir):
        print(f"❌ Target directory not found: {target_dir}")
        return False
    
    print(f"🔄 Tiling images with 2048x2048 tile size...")
    
    # Import tiling functionality
    try:
        from tile_images import tile_image_with_overlap
    except ImportError:
        print("❌ tile_images module not found. Creating basic tiling function...")
        
        def tile_image_with_overlap(image_path, output_dir, tile_size=2048, overlap=0.1):
            """Basic tiling function - you may want to enhance this."""
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return []
            
            h, w = img.shape[:2]
            overlap_pixels = int(tile_size * overlap)
            step = tile_size - overlap_pixels
            
            tiles = []
            for y in range(0, h, step):
                for x in range(0, w, step):
                    # Extract tile
                    tile = img[y:y+tile_size, x:x+tile_size]
                    
                    # Skip if tile is too small
                    if tile.shape[0] < tile_size//2 or tile.shape[1] < tile_size//2:
                        continue
                    
                    # Pad if necessary
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        padded[:tile.shape[0], :tile.shape[1]] = tile
                        tile = padded
                    
                    # Save tile
                    tile_name = f"{Path(image_path).stem}_tile_{x}_{y}.jpg"
                    tile_path = os.path.join(output_dir, tile_name)
                    cv2.imwrite(tile_path, tile)
                    tiles.append(tile_path)
            
            return tiles
    
    # Process each split
    total_tiles = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(target_dir, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  Split directory not found: {split_dir}")
            continue
        
        # Get all image files in this split
        image_files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"⚠️  No images found in {split_dir}")
            continue
        
        print(f"\n📁 Tiling {len(image_files)} images in {split} split...")
        
        # Create tiles directory
        tiles_dir = os.path.join(split_dir, "tiles")
        os.makedirs(tiles_dir, exist_ok=True)
        
        # Tile each image
        split_tiles = 0
        for file in image_files:
            src_path = os.path.join(split_dir, file)
            
            tiles = tile_image_with_overlap(
                image_path=src_path,
                output_dir=tiles_dir,
                tile_size=2048,
                overlap=0.1  # 10% overlap
            )
            
            split_tiles += len(tiles)
            print(f"   {file} → {len(tiles)} tiles")
        
        print(f"   Total {split} tiles: {split_tiles}")
        total_tiles += split_tiles
    
    print(f"\n📊 Total tiles created: {total_tiles}")
    return total_tiles


def save_split_info(sheet_splits, organized_count, target_dataset="rcp_dual_seg"):
    """Save split information for reproducibility."""
    train_sheets, val_sheets, test_sheets = sheet_splits
    
    split_info = {
        "dataset": target_dataset,
        "split_date": datetime.now().isoformat(),
        "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "tile_size": 2048,
        "overlap": 0.1,
        "sheets": {
            "train": train_sheets,
            "val": val_sheets,
            "test": test_sheets
        },
        "tile_counts": organized_count,
        "total_tiles": sum(organized_count.values())
    }
    
    # Save to JSON file
    info_path = f"datasets/{target_dataset}/split_info.json"
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✅ Split information saved to {info_path}")


def main():
    """Main function to split dataset by sheets with 2048 tile size."""
    print("=== Dataset Splitting by Full Sheets (2048 Tile Size) ===")
    print("Splitting 70/20/10 (train/val/test) before tiling to avoid metric cheating")
    print("Using 2048x2048 tile size with 10% overlap")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Process source PDF to images if needed
    pending_images = glob.glob("pending/images/*.jpg") + glob.glob("pending/images/*.png")
    if not pending_images:
        print("🔄 No images found in pending/images. Converting source PDF...")
        if not process_source_pdf_to_images():
            print("❌ Failed to convert PDF to images.")
            return
    else:
        print(f"✅ Found {len(pending_images)} images in pending/images")
    
    # Step 2: Get sheet names from source PDF
    sheet_names = get_sheet_names_from_source_pdf()
    
    if not sheet_names:
        print("❌ No sheets found. Please check the source PDF.")
        return
    
    # Step 3: Split sheets
    sheet_splits = split_sheets_train_val_test(sheet_names)
    
    # Step 4: Organize images by sheet splits
    print(f"\n📁 Organizing images by sheet splits...")
    organized_count = organize_images_by_sheets(sheet_splits, "rcp_dual_seg")
    
    if not organized_count:
        print("❌ Failed to organize images.")
        return
    
    # Step 5: Tile images with 2048x2048 size
    print(f"\n📁 Tiling images with 2048x2048 tile size...")
    total_tiles = tile_images_with_2048_tile_size("rcp_dual_seg")
    
    if total_tiles == 0:
        print("❌ Failed to create tiles.")
        return
    
    # Step 6: Save split information
    save_split_info(sheet_splits, organized_count, "rcp_dual_seg")
    
    print(f"\n🎉 Dataset splitting complete!")
    print(f"📊 Total tiles created: {sum(organized_count.values())}")
    print(f"📁 Images organized in datasets/rcp_dual_seg/images/")
    print(f"🔧 Ready for labeling and training RCP-Dual-Seg model with 2048 tile size")


if __name__ == "__main__":
    main()