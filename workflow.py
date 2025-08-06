"""
YOLO Training Data Workflow
---------------------------
This script manages the workflow for adding new training data:
1. Review images and labels in pending folder
2. Visualize YOLO labels for verification
3. Move approved data to training dataset
4. Add borders to PDF files for Rhino import

Folder structure:
- pending/images/ - Images for review
- pending/labels/ - Labels for review
- datasets/panels/images/train/ - Final training images
- datasets/panels/labels/train/ - Final training labels
- Production drawings (pdfs)/ - PDF files to process
"""

import os
import shutil
import glob
import subprocess
from pathlib import Path
import cv2
import numpy as np
import sys
import re
import yaml
from ultralytics import YOLO

def list_pending_files():
    """List all pending images and their corresponding labels."""
    pending_images = glob.glob("pending/images/*.jpg") + glob.glob("pending/images/*.png")
    pending_labels = glob.glob("pending/labels/*.txt")
    
    print("=== Pending Files ===")
    print("Images:")
    for img in sorted(pending_images):
        print(f"  {Path(img).name}")
    
    print("\nLabels:")
    for lbl in sorted(pending_labels):
        print(f"  {Path(lbl).name}")
    
    return pending_images, pending_labels


def visualize_pending_labels(image_path, labels_path):
    """Visualize YOLO-Seg labels on a pending image for RCP-Dual-Seg model."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found: {labels_path}")
        return False
    
    # Import the visualization function from the main script
    import sys
    sys.path.append('.')
    
    try:
        from visualize_yolo_labels import draw_yolo_labels, parse_yolo_labels
    except ImportError:
        print("Error: Could not import visualize_yolo_labels module")
        return False
    
    # Parse labels using the main script's function
    labels = parse_yolo_labels(labels_path)
    if not labels:
        print("No valid labels found.")
        return False
    
    # Use the main script's visualization function
    success = draw_yolo_labels(
        image_path=image_path,
        labels=labels,
        output_path=f"pending/{Path(image_path).stem}_visualized{Path(image_path).suffix}",
        bbox_thickness=6,  # Thick bounding boxes
        polygon_thickness=2,  # Thin polygons
        show_preview=True
    )
    
    return success


def approve_and_move(image_name, label_name):
    """Move approved files to training dataset."""
    # Check both possible locations for images
    image_paths = [
        f"pending/images/{image_name}",
        f"pending/images/{image_name}.jpg",
        f"pending/images/{image_name}.png"
    ]
    
    image_path = None
    for path in image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        print(f"Error: Image file not found. Tried:")
        for path in image_paths:
            print(f"  {path}")
        return False
    
    # Check label file
    label_paths = [
        f"pending/labels/{label_name}",
        f"pending/labels/{label_name}.txt"
    ]
    
    label_path = None
    for path in label_paths:
        if os.path.exists(path):
            label_path = path
            break
    
    if not label_path:
        print(f"Error: Label file not found. Tried:")
        for path in label_paths:
            print(f"  {path}")
        return False
    
    # Find next available number in training dataset
    existing_images = glob.glob("datasets/rcp_dual_seg/images/train/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/train/*.png")
    next_num = len(existing_images) + 1
    
    # Copy files to training dataset
    new_image_name = f"x{next_num}{Path(image_path).suffix}"
    new_label_name = f"x{next_num}.txt"
    
    shutil.copy2(image_path, f"datasets/rcp_dual_seg/images/train/{new_image_name}")
    shutil.copy2(label_path, f"datasets/rcp_dual_seg/labels/train/{new_label_name}")
    
    # Remove from pending
    os.remove(image_path)
    os.remove(label_path)
    
    print(f"✅ Approved and moved:")
    print(f"  Image: {Path(image_path).name} → {new_image_name}")
    print(f"  Labels: {Path(label_path).name} → {new_label_name}")
    
    return True


def approve_all_pending():
    """Approve and move all pending files to training dataset."""
    pending_images, pending_labels = list_pending_files()
    
    if not pending_labels:
        print("No label files found in pending/labels folder.")
        return False
    
    print(f"\n=== Approving All Pending Files ===")
    print(f"Found {len(pending_labels)} label files to approve")
    
    # Confirm with user
    confirm = input("\nAre you sure you want to approve ALL pending files? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Approval cancelled.")
        return False
    
    approved_count = 0
    failed_count = 0
    
    # Process each label file (this drives the approval)
    for label_path in sorted(pending_labels):
        label_name = Path(label_path).name
        image_name = Path(label_path).stem  # Remove .txt extension
        
        print(f"\nProcessing: {label_name}")
        
        if approve_and_move(image_name, label_name):
            approved_count += 1
        else:
            failed_count += 1
    
    print(f"\n=== Approval Complete ===")
    print(f"✅ Successfully approved: {approved_count} files")
    if failed_count > 0:
        print(f"❌ Failed to approve: {failed_count} files")
    
    return approved_count > 0


def run_pdf_border_script():
    """Run the PDF border script as a separate process."""
    try:
        print("=== Running PDF Border Script ===")
        
        # Get border width from user
        try:
            border_width = float(input("Enter border width in points (default 2): ") or "2")
        except ValueError:
            border_width = 2
            print("Using default border width: 2 points")
        
        print()
        
        # Run the script with border width as argument
        result = subprocess.run(['python', 'add_pdf_borders_vector.py', str(border_width)], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running PDF border script: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: add_pdf_borders_vector.py not found")
        return False


def run_split_existing_images():
    """Run the split existing images script."""
    print("=== Split Existing Images ===")
    print("This will split images from datasets/rcp_dual_seg/images/train/ into train/val/test with 70/20/10 ratio.")
    print("Images will be moved to fullsize folders in respective splits.")
    print()
    
    try:
        # Run the split script
        cmd = ['python', 'split_existing_images.py']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running split script: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: split_existing_images.py not found")
        return False


def run_tile_fullsize_images():
    """Run the tiling script to create 2048x2048 tiles with 30% overlap."""
    print("=== Tile Images ===")
    print("Creating 2048x2048 tiles with 30% overlap for complete panel capture")
    print("Source: images in train/val/test/fullsize folders")
    print("Target: tiled1k folders")
    print("💡 Large tiles ensure complete panels without cropping")
    print()
    
    # Check if we have images to tile (now in fullsize subfolders)
    train_images = glob.glob("datasets/rcp_dual_seg/images/train/fullsize/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/train/fullsize/*.png")
    val_images = glob.glob("datasets/rcp_dual_seg/images/val/fullsize/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/val/fullsize/*.png")
    test_images = glob.glob("datasets/rcp_dual_seg/images/test/fullsize/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/test/fullsize/*.png")
    
    total_images = len(train_images) + len(val_images) + len(test_images)
    
    if total_images == 0:
        print("❌ No images found to tile!")
        print("Make sure you have images in:")
        print("  - datasets/rcp_dual_seg/images/train/fullsize/")
        print("  - datasets/rcp_dual_seg/images/val/fullsize/")
        print("  - datasets/rcp_dual_seg/images/test/fullsize/")
        print("\n💡 Run Option 6 first to split your dataset!")
        return False
    
    print(f"Found {len(train_images)} train images, {len(val_images)} val images, {len(test_images)} test images")
    print(f"Total: {total_images} images to tile")
    print("💡 Expected: Fewer but higher quality tiles with complete panels")
    print()
    
    try:
        result = subprocess.run([
            sys.executable, "tile_fullsize_images.py"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Tiling completed successfully!")
        print(result.stdout)
        
        # Show summary of created tiles
        print("\n=== Tiling Summary ===")
        for split in ['train', 'val', 'test']:
            tiled_dir = f"datasets/rcp_dual_seg/images/{split}/tiled1k"
            if os.path.exists(tiled_dir):
                tile_count = len([f for f in os.listdir(tiled_dir) if f.lower().endswith(('.jpg', '.png'))])
                print(f"  {split}: {tile_count} tiles created")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tiling failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def cleanup_old_tiles():
    """Clean up old tiles in tiled1k folders (preserves fullsize data!)."""
    print("=== Cleanup Old Tiles ===")
    print("⚠️  IMPORTANT: This will only delete tiled data, preserving all fullsize data!")
    
    base_dir = "datasets/rcp_dual_seg/images"
    splits = ['train', 'val', 'test']
    
    total_cleaned = 0
    for split in splits:
        tiled_dir = os.path.join(base_dir, split, "tiled1k")
        if os.path.exists(tiled_dir):
            # Count files before cleanup
            files = [f for f in os.listdir(tiled_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            file_count = len(files)
            
            # Remove all files
            for file in files:
                file_path = os.path.join(tiled_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            print(f"Cleaned up {file_count} tiles from {split}/tiled1k/")
            total_cleaned += file_count
        else:
            print(f"No tiled directory found for {split}")
    
    # Also clean up label tiles
    base_label_dir = "datasets/rcp_dual_seg/labels"
    for split in splits:
        tiled_label_dir = os.path.join(base_label_dir, split, "tiled1k")
        if os.path.exists(tiled_label_dir):
            # Count files before cleanup
            files = [f for f in os.listdir(tiled_label_dir) if f.lower().endswith('.txt')]
            file_count = len(files)
            
            # Remove all files
            for file in files:
                file_path = os.path.join(tiled_label_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            print(f"Cleaned up {file_count} label tiles from {split}/tiled1k/")
    
    print(f"\nTotal tiles cleaned up: {total_cleaned}")
    print("✅ Fullsize data preserved!")


def run_panel_cropping():
    """Panel-centric cropping using existing panel coordinates"""
    print("=== Panel-Centric Cropping ===")
    print("Creating focused training data using panel coordinates")
    print("💡 Much more efficient than random tiling")
    print("📏 Target size: 1024px, Padding: 15%")

    # Ask user if they want to force cleanup
    force_cleanup = input("Do you want to delete all existing crops before starting? (y/N): ").strip().lower()
    force_flag = "--force" if force_cleanup in ['y', 'yes'] else ""
    
    # Check if we have images to crop (now in fullsize subfolders)
    train_images = glob.glob("datasets/rcp_dual_seg/images/train/fullsize/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/train/fullsize/*.png")
    val_images = glob.glob("datasets/rcp_dual_seg/images/val/fullsize/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/val/fullsize/*.png")
    test_images = glob.glob("datasets/rcp_dual_seg/images/test/fullsize/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/test/fullsize/*.png")
    
    total_images = len(train_images) + len(val_images) + len(test_images)
    
    if total_images == 0:
        print("❌ No images found to crop!")
        print("Make sure you have images in:")
        print("  - datasets/rcp_dual_seg/images/train/fullsize/")
        print("  - datasets/rcp_dual_seg/images/val/fullsize/")
        print("  - datasets/rcp_dual_seg/images/test/fullsize/")
        print("\n💡 Run Option 6 first to split your dataset!")
        return False
    
    print(f"Found {len(train_images)} train images, {len(val_images)} val images, {len(test_images)} test images")
    print(f"Total: {total_images} images to crop")
    print("💡 Expected: One crop per panel with context")
    print()
    
    try:
        command = [sys.executable, "panel_cropper.py"]
        if force_flag:
            command.append(force_flag)
            
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("✅ Panel cropping completed successfully!")
        print(result.stdout)
        
        # Show summary of created crops
        print("\n=== Cropping Summary ===")
        for split in ['train', 'val', 'test']:
            cropped_dir = f"datasets/rcp_dual_seg/images/{split}/cropped1k"
            if os.path.exists(cropped_dir):
                crop_count = len([f for f in os.listdir(cropped_dir) if f.lower().endswith(('.jpg', '.png'))])
                print(f"  {split}: {crop_count} panel crops created")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Panel cropping failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def run_smart_augmentation():
    """Apply smart augmentation for better generalization"""
    print("=== Smart Augmentation ===")
    print("Applying advanced augmentation for better generalization")
    print("💡 Inward cropping, scale jitter, print artifacts")
    print("📏 Target size: 1024px, 3 augmentations per image")
    
    # Check if we have cropped images to augment
    train_crops = glob.glob("datasets/rcp_dual_seg/images/train/cropped1k/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/train/cropped1k/*.png")
    val_crops = glob.glob("datasets/rcp_dual_seg/images/val/cropped1k/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/val/cropped1k/*.png")
    test_crops = glob.glob("datasets/rcp_dual_seg/images/test/cropped1k/*.jpg") + glob.glob("datasets/rcp_dual_seg/images/test/cropped1k/*.png")
    
    total_crops = len(train_crops) + len(val_crops) + len(test_crops)
    
    if total_crops == 0:
        print("❌ No cropped images found to augment!")
        print("Make sure you have cropped images in:")
        print("  - datasets/rcp_dual_seg/images/train/cropped1k/")
        print("  - datasets/rcp_dual_seg/images/val/cropped1k/")
        print("  - datasets/rcp_dual_seg/images/test/cropped1k/")
        print("\n💡 Run Option 7 first to create panel crops!")
        return False
    
    print(f"Found {len(train_crops)} train crops, {len(val_crops)} val crops, {len(test_crops)} test crops")
    print(f"Total: {total_crops} crops to augment")
    print("💡 Expected: 3x more training data with better generalization")
    print()
    
    try:
        result = subprocess.run([
            sys.executable, "smart_augmentation.py"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Smart augmentation completed successfully!")
        print(result.stdout)
        
        # Show summary of created augmentations
        print("\n=== Augmentation Summary ===")
        for split in ['train', 'val', 'test']:
            aug_dir = f"datasets/rcp_dual_seg/images/{split}/augmented1k"
            if os.path.exists(aug_dir):
                aug_count = len([f for f in os.listdir(aug_dir) if f.lower().endswith(('.jpg', '.png'))])
                print(f"  {split}: {aug_count} augmented samples created")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Smart augmentation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    print("Ready for fresh tiling!")


def clean_empty_tiles():
    """Remove tiles that have no panel annotations (improves training)"""
    print("=== Clean Empty Tiles ===")
    print("Removing tiles with no panel annotations...")
    print("💡 This improves training by removing noise")
    
    try:
        result = subprocess.run([
            sys.executable, "clean_empty_tiles.py"
        ], check=True)
        
        print("✅ Empty tiles cleaned successfully!")
        print("💡 Dataset now contains only tiles with panel annotations")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Cleaning failed: {e}")
        return False


def verify_training_setup():
    """Verify all components are ready for training"""
    
    print("🔍 Verifying Training Setup")
    print("=" * 50)
    
    # 1. Check pre-trained model
    print("\n1. Pre-trained Model:")
    model_path = "models/standard_yolov8s_seg/best.pt"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Pre-trained model found: {model_path}")
        print(f"   Size: {size_mb:.1f} MB")
    else:
        print(f"❌ Pre-trained model not found: {model_path}")
        return False
    
    # 2. Check dataset configuration
    print("\n2. Dataset Configuration:")
    config_path = "datasets/rcp_dual_seg/dataset.yaml"
    if os.path.exists(config_path):
        print(f"✅ Dataset config found: {config_path}")
        with open(config_path, 'r') as f:
            config_content = f.read()
            print(f"   Content:\n{config_content}")
    else:
        print(f"❌ Dataset config not found: {config_path}")
        return False
    
    # 3. Check training images (augmented or cropped)
    print("\n3. Training Images:")
    splits = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    dataset_type = "unknown"
    
    # Check for cropped data first, then tiled
    data_types = ['augmented1k', 'cropped1k', 'tiled1k']
    
    for split in splits:
        found_data = False
        for data_type in data_types:
            img_dir = f"datasets/rcp_dual_seg/images/{split}/{data_type}"
            lbl_dir = f"datasets/rcp_dual_seg/labels/{split}/{data_type}"
            
            if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                images = glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png")
                labels = glob.glob(f"{lbl_dir}/*.txt")
                
                if len(images) > 0:
                    total_images += len(images)
                    total_labels += len(labels)
                    dataset_type = data_type
                    print(f"   {split}: {len(images)} images, {len(labels)} labels ({data_type})")
                    found_data = True
                    break
        
        if not found_data:
            print(f"   ❌ {split}: No training data found (checked: {', '.join(data_types)})")
            return False
    
    print(f"   Total: {total_images} images, {total_labels} labels")
    
    if total_images == 0:
        print("❌ No training images found!")
        return False
    
    # 4. Check training script
    print("\n4. Training Script:")
    script_path = "train_with_standard.py"
    if os.path.exists(script_path):
        print(f"✅ Training script found: {script_path}")
    else:
        print(f"❌ Training script not found: {script_path}")
        return False
    
    # 5. Check GPU availability
    print("\n5. GPU Check:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
            print(f"   GPU count: {gpu_count}")
        else:
            print("⚠️  No GPU detected - training will use CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    print("\n" + "=" * 50)
    print("✅ Training setup verification complete!")
    print(f"Ready to train with {total_images} {dataset_type} images")
    
    if dataset_type == "cropped1k":
        print("💡 Using panel-centric cropped data (1024x1024)")
    elif dataset_type == "tiled1k":
        print("💡 Using tiled data (2048x2048 with 30% overlap)")
    
    print("\nTo start training, run:")
    print("python train_with_standard.py")
    
    return True

def analyze_dataset_structure():
    """Analyze dataset structure and distribution"""
    print("=== Dataset Analysis ===")
    print("Analyzing RCP-Dual-Seg dataset structure and distribution...")
    print()
    
    # Import the analysis function directly
    try:
        import os
        import glob
        import yaml
        from pathlib import Path
        from collections import defaultdict
        
        base_dir = "datasets/rcp_dual_seg"
        
        # Check for different dataset configs
        dataset_configs = []
        
        # Check dataset config
        dataset_yaml = os.path.join(base_dir, "dataset.yaml")
        if os.path.exists(dataset_yaml):
            dataset_configs.append(("cropped", dataset_yaml))
        
        if not dataset_configs:
            print(f"❌ No dataset configs found!")
            print(f"Expected: {dataset_yaml}")
            return False
        
        print(f"✅ Found {len(dataset_configs)} dataset config(s)")
        
        # Analyze each dataset
        for dataset_type, config_path in dataset_configs:
            print(f"\n📊 Analyzing {dataset_type.upper()} dataset:")
            print(f"   Config: {config_path}")
            
            # Load and analyze dataset config
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                print(f"   Classes: {config.get('nc', 'N/A')}")
                print(f"   Class names: {config.get('names', 'N/A')}")
                
                # Verify 2-class structure
                if config.get('nc') != 2:
                    print(f"   ❌ Expected 2 classes, found {config.get('nc')}")
                    continue
                
                expected_names = ['panel', 'panel_tag']
                if config.get('names') != expected_names:
                    print(f"   ❌ Expected class names: {expected_names}")
                    print(f"   Found: {config.get('names')}")
                    continue
                
                print(f"   ✅ 2-class structure confirmed")
                print(f"   📋 Label format: Class 0 (panel) = polygon mask, Class 1 (panel_tag) = bbox")
                
            except Exception as e:
                print(f"   ❌ Error loading dataset config: {e}")
                continue
            
            # Analyze directory structure
            print(f"\n   📁 Directory Structure Analysis:")
            
            splits = ['train', 'val', 'test']
            data_types = ['augmented1k', 'cropped1k', 'tiled1k', 'fullsize']
            
            dataset_stats = {}
            
            for split in splits:
                dataset_stats[split] = {}
                
                for data_type in data_types:
                    img_dir = f"{base_dir}/images/{split}/{data_type}"
                    lbl_dir = f"{base_dir}/labels/{split}/{data_type}"
                    
                    # Check if this directory exists and has data
                    if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                        images = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                                glob.glob(os.path.join(img_dir, "*.jpeg")) + \
                                glob.glob(os.path.join(img_dir, "*.png"))
                        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
                        
                        if len(images) > 0 or len(labels) > 0:
                            print(f"      {split.capitalize()} - {data_type}: {len(images)} images, {len(labels)} labels")
                            
                            # Get image sizes if any exist
                            if images:
                                try:
                                    import cv2
                                    sample_img = cv2.imread(images[0])
                                    if sample_img is not None:
                                        h, w = sample_img.shape[:2]
                                        print(f"         Sample image size: {w}x{h}")
                                except:
                                    print(f"         Sample image size: Unknown")
                            
                            # Analyze label content (simplified)
                            if labels:
                                print(f"         Analyzing {len(labels)} label files...")
                                sample_files = labels[:3]  # Only sample 3 files
                                
                                class_counts = defaultdict(int)
                                empty_files = 0
                                total_annotations = 0
                                
                                for label_file in sample_files:
                                    try:
                                        with open(label_file, 'r') as f:
                                            lines = f.readlines()
                                        
                                        if not lines or all(line.strip() == '' for line in lines):
                                            empty_files += 1
                                            continue
                                        
                                        for line in lines:
                                            line = line.strip()
                                            if line:
                                                parts = line.split()
                                                if len(parts) >= 5:
                                                    class_id = int(parts[0])
                                                    class_counts[class_id] += 1
                                                    total_annotations += 1
                                    
                                    except Exception as e:
                                        print(f"            ⚠️  Error reading {Path(label_file).name}: {e}")
                                
                                print(f"         Empty files: {empty_files}")
                                print(f"         Total annotations: {total_annotations}")
                                print(f"         Class distribution: {dict(class_counts)}")
                            
                            dataset_stats[split][data_type] = {
                                'images': len(images),
                                'labels': len(labels)
                            }
            
            # Summary statistics for this dataset type
            print(f"\n   📊 {dataset_type.upper()} Dataset Summary:")
            
            total_images = 0
            total_labels = 0
            
            for split in splits:
                for data_type in data_types:
                    if data_type in dataset_stats.get(split, {}):
                        stats = dataset_stats[split][data_type]
                        total_images += stats['images']
                        total_labels += stats['labels']
            
            print(f"      Total images: {total_images}")
            print(f"      Total labels: {total_labels}")
            
            # Check for dataset type
            if total_images > 0:
                print(f"      ✅ {dataset_type.capitalize()} dataset detected")
                if dataset_type == "cropped":
                    print(f"      💡 Ready for training with cropped data")
            else:
                print(f"      ❌ No images found in {dataset_type} dataset")
        
        print(f"\n✅ Dataset analysis complete!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")
        return False

def test_gpu_memory():
    """Test GPU memory usage with 1024x1024 images and batch size 8"""
    print("=== GPU Memory Test ===")
    print("Testing if GPU can handle 1024x1024 images with batch size 8...")
    print()
    
    try:
        result = subprocess.run([
            "C:\\Users\\ersk\\AppData\\Local\\anaconda3\\Scripts\\conda.exe", 
            "run", "-n", "rhino_mcp", "python", "test_gpu_memory_1024.py"
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU memory test failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def run_transfer_learning():
    """Orchestrate the two-stage transfer learning process."""
    # Fix for OpenMP runtime error: "Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
    # This is a workaround to allow multiple OpenMP libraries.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print("🚀 Starting Two-Stage Transfer Learning 🚀")
    print("="*60)
    print("This workflow will fine-tune the standard YOLOv8s-seg model in two stages:")
    print("1. Fine-tune on the 'cropped1k' dataset (learning panel features).")
    print("2. Fine-tune the result on the 'augmented1k' dataset (for robustness).")
    print("="*60)

    # --- Setup ---
    print("\n[STEP 1/5] 셋업 확인 (Verifying Setup)")
    
    # Download standard model if it doesn't exist
    models_dir = Path("models/standard_yolov8s_seg")
    models_dir.mkdir(parents=True, exist_ok=True)
    standard_model_path = models_dir / "best.pt"

    if not standard_model_path.exists():
        print("   > Standard YOLOv8s-seg model not found. Downloading...")
        try:
            model = YOLO("yolov8s-seg.pt")
            model.save(str(standard_model_path))
            print(f"   ✅ Downloaded: {standard_model_path}")
        except Exception as e:
            print(f"   ❌ Error downloading model: {e}")
            return
    else:
        print(f"   ✅ Standard model found: {standard_model_path}")

    # --- Create dynamic YAML files ---
    print("\n[STEP 2/5] 동적 YAML 파일 생성 (Creating Dynamic YAML files)")

    base_config = {
        'path': '../datasets/rcp_dual_seg',
        'nc': 2,
        'names': ['panel', 'panel_tag']
    }

    # Config for cropped1k
    config_stage1 = base_config.copy()
    config_stage1.update({
        'train': 'images/train/cropped1k',
        'val': 'images/val/cropped1k',
        'test': 'images/test/cropped1k'
    })
    yaml_path_stage1 = Path("datasets/rcp_dual_seg/dataset_cropped1k.yaml")
    with open(yaml_path_stage1, 'w') as f:
        yaml.dump(config_stage1, f)
    print(f"   ✅ Created temporary config for Stage 1: {yaml_path_stage1}")

    # Config for augmented1k
    config_stage2 = base_config.copy()
    config_stage2.update({
        'train': 'images/train/augmented1k',
        'val': 'images/val/augmented1k',
        'test': 'images/test/augmented1k'
    })
    yaml_path_stage2 = Path("datasets/rcp_dual_seg/dataset_augmented1k.yaml")
    with open(yaml_path_stage2, 'w') as f:
        yaml.dump(config_stage2, f)
    print(f"   ✅ Created temporary config for Stage 2: {yaml_path_stage2}")

    best_model_from_stage1 = None

    try:
        # --- Stage 1: Fine-tune on cropped1k ---
        print("\n[STEP 3/5] Stage 1: cropped1k 데이터로 미세 조정 (Fine-tuning on cropped1k)")
        if not input("   > Proceed with Stage 1? (y/N): ").strip().lower() in ['y', 'yes']:
            print("   > Training cancelled.")
            return

        print("   > Loading standard model for Stage 1...")
        model_stage1 = YOLO(str(standard_model_path))
        
        print("   > Starting Stage 1 training...")
        results_stage1 = model_stage1.train(
            data=str(yaml_path_stage1),
            epochs=50, 
            imgsz=1024,
            batch=8,
            name="panel_detection_cropped1k_finetune",
            project="runs/train",
            patience=10,
            device=0,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.0005
        )
        
        print("   ✅ Stage 1 training complete!")
        best_model_from_stage1 = results_stage1.save_dir / "weights" / "best.pt"
        print(f"   > Best model from Stage 1: {best_model_from_stage1}")

        # --- Stage 2: Fine-tune on augmented1k ---
        print("\n[STEP 4/5] Stage 2: augmented1k 데이터로 미세 조정 (Fine-tuning on augmented1k)")
        if not input("   > Proceed with Stage 2? (y/N): ").strip().lower() in ['y', 'yes']:
            print("   > Training cancelled.")
            return

        print(f"   > Loading best model from Stage 1 for Stage 2...")
        model_stage2 = YOLO(str(best_model_from_stage1))

        print("   > Starting Stage 2 training...")
        results_stage2 = model_stage2.train(
            data=str(yaml_path_stage2),
            epochs=100,
            imgsz=1024,
            batch=8,
            name="panel_detection_augmented1k_finetune",
            project="runs/train",
            patience=20,
            device=0,
            pretrained=True, # It is pretrained on our cropped data
            optimizer="AdamW",
            lr0=0.0005, # Lower learning rate for the second fine-tuning stage
            weight_decay=0.0005
        )
        
        print("   ✅ Stage 2 training complete!")
        best_model_from_stage2 = results_stage2.save_dir / "weights" / "best.pt"
        print(f"   > Best model from Stage 2: {best_model_from_stage2}")

        print("\n[STEP 5/5] 완료 (Complete)")
        print("="*60)
        print("🎉 Two-Stage Transfer Learning Complete! 🎉")
        print(f"   > Final model saved at: {best_model_from_stage2}")
        print("   > This model is now ready for inference.")
        print("="*60)

    except Exception as e:
        print(f"   ❌ Training process failed: {e}")
    finally:
        # --- Cleanup ---
        print("\nCleaning up temporary YAML files...")
        if yaml_path_stage1.exists():
            os.remove(yaml_path_stage1)
            print(f"   > Removed {yaml_path_stage1}")
        if yaml_path_stage2.exists():
            os.remove(yaml_path_stage2)
            print(f"   > Removed {yaml_path_stage2}")


def run_inspect_crops():
    """Inspect cropped or augmented images and labels"""
    print("=== Inspect Data ===")
    print("Select the data type and split to inspect.")
    
    # Choose data type
    data_type = input("Enter data type (cropped1k, augmented1k): ").strip()
    if data_type not in ["cropped1k", "augmented1k"]:
        print("Invalid data type.")
        return
        
    # Choose split
    split = input("Enter dataset split (train, val, test): ").strip()
    if split not in ["train", "val", "test"]:
        print("Invalid split.")
        return
    
    try:
        subprocess.run([
            sys.executable, "inspect_crops.py", "--split", split, "--data-type", data_type
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running inspection script: {e}")
    except FileNotFoundError:
        print("Error: inspect_crops.py not found.")


def main():
    """Main workflow function."""
    while True:
        print("\n=== RCP-Dual-Seg Label Review Workflow ===")
        print("\nClasses: 0=panel (polygon), 1=panel_tag (bbox)")
        print("\nOptions:")
        print("1. List pending files")
        print("2. Visualize RCP-Dual-Seg labels")
        print("3. Approve and move to training dataset")
        print("4. Approve ALL pending files")
        print("5. Add borders to PDF files")
        print("6. Split existing images (70/20/10)")
        print("7. Tile fullsize images (2048x2048 with 30% overlap)")
        print("   💡 Alternative: Use panel-centric cropping (Option 8)")
        print("8. Panel-centric cropping (1024px with 15% padding)")
        print("9. Smart augmentation (3x data with generalization)")
        print("   💡 Training will use augmented data if available, otherwise cropped")
        print("10. Cleanup old tiles")
        print("11. Clean empty tiles (remove tiles without panels)")
        print("13. Analyze dataset structure and distribution (augmented/cropped)")
        print("14. Inspect data (cropped/augmented)")
        print("15. Test GPU memory (1024x1024 images)")
        print("16. Verify training setup")
        print("17. Start Two-Stage Transfer Learning")
        print("18. Exit")
        
        choice = input("\nEnter your choice (1-18): ").strip()
        
        if choice == "1":
            list_pending_files()
        elif choice == "2":
            pending_images, pending_labels = list_pending_files()
            
            if not pending_images:
                print("No images found in pending folder.")
                continue
            
            print("\nAvailable images for visualization:")
            for i, img in enumerate(sorted(pending_images), 1):
                print(f"  {i}. {Path(img).name}")
            
            try:
                choice_num = int(input("\nEnter image number to visualize: ").strip())
                if 1 <= choice_num <= len(pending_images):
                    selected_image = sorted(pending_images)[choice_num - 1]
                    image_name = Path(selected_image).stem
                    label_path = f"pending/labels/{image_name}.txt"
                    
                    visualize_pending_labels(selected_image, label_path)
                else:
                    print("Invalid image number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == "3":
            pending_images, pending_labels = list_pending_files()
            
            if not pending_images:
                print("No images found in pending folder.")
                continue
            
            print("\nAvailable images for approval:")
            for i, img in enumerate(sorted(pending_images), 1):
                print(f"  {i}. {Path(img).name}")
            
            try:
                choice_num = int(input("\nEnter image number to approve: ").strip())
                if 1 <= choice_num <= len(pending_images):
                    selected_image = sorted(pending_images)[choice_num - 1]
                    image_name = Path(selected_image).stem
                    label_name = f"{image_name}.txt"
                    
                    approve_and_move(image_name, label_name)
                else:
                    print("Invalid image number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == "4":
            approve_all_pending()
        elif choice == "5":
            run_pdf_border_script()
        elif choice == "6":
            run_split_existing_images()
        elif choice == "7":
            run_tile_fullsize_images()
        elif choice == "8":
            run_panel_cropping()
        elif choice == "9":
            run_smart_augmentation()
        elif choice == "10":
            cleanup_old_tiles()
        elif choice == "11":
            clean_empty_tiles()
        elif choice == "12":
            analyze_dataset_structure()
        elif choice == "13":
            analyze_dataset_structure()
        elif choice == "14":
            run_inspect_crops()
        elif choice == "15":
            test_gpu_memory()
        elif choice == "16":
            verify_training_setup()
        elif choice == "17":
            run_transfer_learning()
        elif choice == "18":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-18.")


if __name__ == "__main__":
    main()
