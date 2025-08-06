"""
split_existing_images.py
------------------------
Split existing images from datasets/rcp_dual_seg/images/train/ into train/val/test with 70/20/10 ratio.
Also splits corresponding label files.
Moves images to fullsize folders in respective splits.
"""

import os
import shutil
import random
from pathlib import Path
import json
from datetime import datetime


def split_images_train_val_test(source_dir="datasets/rcp_dual_seg/images/train", 
                               train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split images from source directory into train/val/test with specified ratios.
    
    Args:
        source_dir: Directory containing images to split
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.2) 
        test_ratio: Proportion for testing (default 0.1)
    
    Returns:
        train_images, val_images, test_images: Lists of image filenames
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get all image files from source directory
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return [], [], []
    
    image_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {source_dir}")
        return [], [], []
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # Shuffle images for random split
    random.shuffle(image_files)
    
    # Calculate split indices
    n_images = len(image_files)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    n_test = n_images - n_train - n_val  # Remaining goes to test
    
    # Split images
    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train + n_val]
    test_images = image_files[n_train + n_val:]
    
    print(f"\nImage Split Results:")
    print(f"   Train: {len(train_images)} images ({len(train_images)/n_images*100:.1f}%)")
    print(f"   Val:   {len(val_images)} images ({len(val_images)/n_images*100:.1f}%)")
    print(f"   Test:  {len(test_images)} images ({len(test_images)/n_images*100:.1f}%)")
    
    return train_images, val_images, test_images


def move_images_and_labels_to_splits(image_splits, base_dir="datasets/rcp_dual_seg"):
    """
    Move images and corresponding labels to their respective split directories.
    
    Args:
        image_splits: Tuple of (train_images, val_images, test_images)
        base_dir: Base directory for the dataset
    """
    train_images, val_images, test_images = image_splits
    source_img_dir = os.path.join(base_dir, "images", "train")
    source_label_dir = os.path.join(base_dir, "labels", "train")
    
    # Create target directories for images and labels
    target_dirs = {
        'train': {
            'images': os.path.join(base_dir, "images", "train", "fullsize"),
            'labels': os.path.join(base_dir, "labels", "train", "fullsize")
        },
        'val': {
            'images': os.path.join(base_dir, "images", "val", "fullsize"),
            'labels': os.path.join(base_dir, "labels", "val", "fullsize")
        },
        'test': {
            'images': os.path.join(base_dir, "images", "test", "fullsize"),
            'labels': os.path.join(base_dir, "labels", "test", "fullsize")
        }
    }
    
    # Create all target directories
    for split in target_dirs.values():
        for dir_type, dir_path in split.items():
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    # Move images and labels to their respective splits
    split_mapping = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    moved_count = {split: {'images': 0, 'labels': 0} for split in ['train', 'val', 'test']}
    
    for split, images in split_mapping.items():
        target_img_dir = target_dirs[split]['images']
        target_label_dir = target_dirs[split]['labels']
        
        for image_file in images:
            # Move image
            src_img_path = os.path.join(source_img_dir, image_file)
            dst_img_path = os.path.join(target_img_dir, image_file)
            
            if os.path.exists(src_img_path):
                shutil.move(src_img_path, dst_img_path)
                moved_count[split]['images'] += 1
                print(f"   {image_file} -> {split}/fullsize/")
            else:
                print(f"Warning: Source image not found: {src_img_path}")
            
            # Move corresponding label file
            label_name = Path(image_file).stem + ".txt"
            src_label_path = os.path.join(source_label_dir, label_name)
            dst_label_path = os.path.join(target_label_dir, label_name)
            
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dst_label_path)
                moved_count[split]['labels'] += 1
                print(f"   {label_name} -> {split}/fullsize/")
            else:
                print(f"Warning: Source label not found: {src_label_path}")
    
    print(f"\nMoved files:")
    for split, counts in moved_count.items():
        print(f"   {split}: {counts['images']} images, {counts['labels']} labels")
    
    return moved_count


def save_split_info(image_splits, moved_count, base_dir="datasets/rcp_dual_seg"):
    """Save split information for reproducibility."""
    train_images, val_images, test_images = image_splits
    
    split_info = {
        "dataset": "rcp_dual_seg",
        "split_date": datetime.now().isoformat(),
        "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "source_directories": {
            "images": "datasets/rcp_dual_seg/images/train",
            "labels": "datasets/rcp_dual_seg/labels/train"
        },
        "target_directories": {
            "images": {
                "train": "datasets/rcp_dual_seg/images/train/fullsize",
                "val": "datasets/rcp_dual_seg/images/val/fullsize", 
                "test": "datasets/rcp_dual_seg/images/test/fullsize"
            },
            "labels": {
                "train": "datasets/rcp_dual_seg/labels/train/fullsize",
                "val": "datasets/rcp_dual_seg/labels/val/fullsize", 
                "test": "datasets/rcp_dual_seg/labels/test/fullsize"
            }
        },
        "images": {
            "train": train_images,
            "val": val_images,
            "test": test_images
        },
        "moved_counts": moved_count,
        "total_images": sum(counts['images'] for counts in moved_count.values()),
        "total_labels": sum(counts['labels'] for counts in moved_count.values())
    }
    
    # Save to JSON file
    info_path = os.path.join(base_dir, "split_info.json")
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split information saved to {info_path}")


def main():
    """Main function to split existing images into train/val/test."""
    print("=== Split Existing Images and Labels (70/20/10) ===")
    print("Splitting images and labels from train folder into train/val/test with fullsize folders")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Split images
    image_splits = split_images_train_val_test()
    
    if not any(image_splits):
        print("No images to split.")
        return
    
    # Step 2: Move images and labels to their respective splits
    print(f"\nMoving images and labels to split directories...")
    moved_count = move_images_and_labels_to_splits(image_splits)
    
    if not moved_count:
        print("Failed to move files.")
        return
    
    # Step 3: Save split information
    save_split_info(image_splits, moved_count)
    
    print(f"\nImage and label splitting complete!")
    total_images = sum(counts['images'] for counts in moved_count.values())
    total_labels = sum(counts['labels'] for counts in moved_count.values())
    print(f"Total images moved: {total_images}")
    print(f"Total labels moved: {total_labels}")
    print(f"Files organized in fullsize folders")
    print(f"Ready for tiling with 2048x2048 size")


if __name__ == "__main__":
    main() 