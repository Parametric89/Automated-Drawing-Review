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
import time
import json

def print_header(title: str) -> None:
    bar = "=" * 28
    print(f"\n{bar} {title} {bar}")

# --- simple persistent prefs ----------------------------------------------
PREFS_PATH = Path(".hub_prefs.json")

def _prefs_load() -> dict:
    try:
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _prefs_save(p: dict) -> None:
    try:
        PREFS_PATH.write_text(json.dumps(p, indent=2), encoding="utf-8")
    except Exception:
        pass

def _remember(key: str, value: str) -> None:
    p = _prefs_load(); p[key] = value; _prefs_save(p)

def _prompt_mem(key: str, prompt: str, default_value: str) -> str:
    p = _prefs_load()
    dv = str(p.get(key, default_value))
    val = input(f"{prompt} [{dv}]: ").strip()
    if not val:
        val = dv
    _remember(key, val)
    return val

def _load_training_config() -> dict:
    """Load training_config.yaml if present, else return {}."""
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path("training_config.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}

def _save_training_config(cfg: dict) -> None:
    """Save training configuration to training_config.yaml."""
    import yaml
    with open("training_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def _write_dataset_yaml(base_dir: str, folder_name: str) -> str:
    """Create a YOLO dataset YAML pointing to <base_dir>/images/*/<folder_name>.

    Returns the YAML file path.
    """
    base = Path(base_dir)
    yaml_path = base / f"dataset_{folder_name}.yaml"
    # Use paths relative to the YAML file location so training runs from any CWD
    data = {
        "train": "images/train/" + folder_name,
        "val": "images/val/" + folder_name,
        "test": "images/test/" + folder_name,
        "nc": 1,
        "names": ["panel"],
        "task": "detect",
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    _remember("last_dataset_yaml", str(yaml_path))
    return str(yaml_path)

def _write_dataset_yaml_custom(base_dir: str, train_folder: str, val_folder: str, test_folder: str) -> str:
    """Create a YOLO dataset YAML with explicit per-split folders under images/* and labels/*."""
    base = Path(base_dir)
    yaml_path = base / f"dataset_{train_folder}.yaml"
    data = {
        "train": f"images/train/{train_folder}",
        "val": f"images/val/{val_folder}",
        "test": f"images/test/{test_folder}",
        "nc": 1,
        "names": ["panel"],
        "task": "detect",
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    _remember("last_dataset_yaml", str(yaml_path))
    return str(yaml_path)

def get_conda_python() -> str:
    """Return the rhino_mcp environment's python interpreter path."""
    # Using env python directly avoids intermittent KeyboardInterrupts from `conda run` wrappers on Windows
    return r"C:\Users\ersk\AppData\Local\anaconda3\envs\rhino_mcp\python.exe"

def run_command(cmd: list) -> None:
    import subprocess
    print("$", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

def prompt_for_dataset_dir(default: str = "datasets/rcp_dual_seg_v3") -> str:
    base_dir = _prompt_mem("dataset_base", "Base dataset dir", default)
    print(f"\nWorking on dataset base: {base_dir}")
    return base_dir
def select_base_dir(default_dir: str = "datasets/rcp_dual_seg_v2") -> str:
    """Prompt for a dataset base directory and echo it back for clarity."""
    base_dir = _prompt_mem("dataset_base", "Base dataset dir", default_dir)
    print(f"\nWorking on dataset base: {base_dir}")
    return base_dir

def list_pending_files():
    """List all pending images and their corresponding labels."""
    pending_images = glob.glob("pending/images/*.jpg") + glob.glob("pending/images/*.png")
    pending_labels = glob.glob("pending/labels/*.txt")
    
    print("=== Pending Files ===")
    print("Base: pending/images, pending/labels → destination: datasets/rcp_dual_seg_v3")
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
    sys.path.append('tools')
    
    try:
        from tools.visualize_yolo_labels import draw_yolo_labels, parse_yolo_labels
    except ImportError:
        try:
             # Fallback if tools is in path
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


def visualize_all_pending_labels_batch():
    """Visualize ALL pending images using the same style as the single interactive preview.

    - Reads label files from pending/labels
    - Matches images from pending/images (jpg/jpeg/png)
    - Prompts only for output directory
    - Saves visualizations using the same styling as interactive (thick green/red, magenta polys)
    """
    import time

    pending_labels = sorted(glob.glob("pending/labels/*.txt"))
    if not pending_labels:
        print("No label files found in pending/labels folder.")
        return False

    default_out = Path("runs") / "overlays" / f"pending_{int(time.time())}"
    out_dir_inp = input(f"Output dir for visualizations [default: {default_out}]: ").strip()
    out_dir = Path(out_dir_inp) if out_dir_inp else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {out_dir}")

    # Import the shared visualization utilities
    try:
        sys.path.append('.')
        sys.path.append('tools')
        from tools.visualize_yolo_labels import draw_yolo_labels, parse_yolo_labels
    except ImportError:
        try:
             from visualize_yolo_labels import draw_yolo_labels, parse_yolo_labels
        except ImportError:
             print("Error: Could not import visualize_yolo_labels module")
             return False

    success_count = 0
    skipped_count = 0
    total = len(pending_labels)

    for idx, lbl_path in enumerate(pending_labels, 1):
        lbl_path = Path(lbl_path)
        stem = lbl_path.stem
        # Match image by common extensions
        candidates = [
            f"pending/images/{stem}.jpg",
            f"pending/images/{stem}.jpeg",
            f"pending/images/{stem}.png",
        ]
        image_path = None
        for cand in candidates:
            if os.path.exists(cand):
                image_path = cand
                break

        if image_path is None:
            print(f"[{idx}/{total}] {stem}: image not found in pending/images, skipping.")
            skipped_count += 1
            continue

        labels = parse_yolo_labels(str(lbl_path))
        if not labels:
            print(f"[{idx}/{total}] {stem}: no valid labels, skipping.")
            skipped_count += 1
            continue

        img_path_obj = Path(image_path)
        out_path = out_dir / f"{img_path_obj.stem}_yolo_visualized{img_path_obj.suffix}"

        ok = draw_yolo_labels(
            image_path=image_path,
            labels=labels,
            output_path=str(out_path),
            bbox_thickness=6,
            polygon_thickness=2,
            show_preview=False,
        )
        if ok:
            print(f"[{idx}/{total}] Saved: {out_path}")
            success_count += 1
        else:
            print(f"[{idx}/{total}] {stem}: failed to render.")
            skipped_count += 1

    print(f"\n✅ Completed. Saved {success_count} visualizations. Skipped {skipped_count}.")
    return success_count > 0
def run_overlay_labels():
    """Batch overlay labels on images in a chosen folder (Pending or ROI)."""
    print_header("Folder Batch Overlays")
    # Offer quick presets
    pending_img = Path("pending/images"); pending_lbl = Path("pending/labels")
    roi_img = Path("inference/ROI/ROI_images"); roi_lbl = Path("inference/ROI/ROI_labels")
    has_roi = roi_img.exists() and roi_lbl.exists()

    print("1) Pending (pending/images + pending/labels)")
    if has_roi:
        print("2) ROI (inference/ROI/ROI_images + inference/ROI/ROI_labels)")
    print("3) Custom folders")
    choice = input("Choose [1/2/3, default 1]: ").strip() or "1"

    if choice == "2" and has_roi:
        images_dir = str(roi_img)
        labels_dir = str(roi_lbl)
        default_out = Path("inference/ROI/overlays")
    elif choice == "3":
        images_dir = input("Images directory [pending/images]: ").strip() or str(pending_img)
        labels_dir = input("Labels directory [pending/labels]: ").strip() or str(pending_lbl)
        default_out = Path("runs") / "overlays" / f"custom_{int(time.time())}"
    else:
        images_dir = str(pending_img)
        labels_dir = str(pending_lbl)
        default_out = Path("runs") / "overlays" / f"pending_{int(time.time())}"

    out_dir_inp = input(f"Output dir for overlays [default: {default_out}]: ").strip()
    out_dir = Path(out_dir_inp) if out_dir_inp else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_command([
            get_conda_python(), "utils/overlay_labels_folder.py",
            "--images", images_dir,
            "--labels", labels_dir,
            "--out", str(out_dir),
            "--limit", "10000",
        ])
        print(f"Saved overlays to: {out_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Overlay script failed: {e}")
        return False



def approve_and_move(image_name, label_name, dest_base: str):
    """Move approved files to training dataset under dest_base."""
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
    
    # Destination base provided by caller
    print(f"Destination dataset base: {dest_base}")

    # Find next available number in training dataset (v3)
    existing_images = glob.glob(f"{dest_base}/images/train/*.jpg") + glob.glob(f"{dest_base}/images/train/*.png")
    next_num = len(existing_images) + 1
    
    # Copy files to training dataset
    new_image_name = f"x{next_num}{Path(image_path).suffix}"
    new_label_name = f"x{next_num}.txt"
    
    os.makedirs(f"{dest_base}/images/train", exist_ok=True)
    os.makedirs(f"{dest_base}/labels/train", exist_ok=True)
    shutil.copy2(image_path, f"{dest_base}/images/train/{new_image_name}")
    shutil.copy2(label_path, f"{dest_base}/labels/train/{new_label_name}")
    
    # Remove from pending
    os.remove(image_path)
    os.remove(label_path)
    
    print(f"✅ Approved and moved:")
    print(f"  Image: {Path(image_path).name} → {new_image_name}")
    print(f"  Labels: {Path(label_path).name} → {new_label_name}")
    
    return True


def approve_all_pending():
    """Approve and move all pending files to a chosen training dataset base."""
    pending_images, pending_labels = list_pending_files()
    
    if not pending_labels:
        print("No label files found in pending/labels folder.")
        return False
    
    print(f"\n=== Approving All Pending Files ===")
    print(f"Found {len(pending_labels)} label files to approve")
    dest_base = _prompt_mem("dataset_base", "Destination dataset base", "datasets/rcp_dual_seg_v3")
    os.makedirs(f"{dest_base}/images/train", exist_ok=True)
    os.makedirs(f"{dest_base}/labels/train", exist_ok=True)
    # Remember for later steps (split/tiling)
    try:
        Path(".last_dataset_base.txt").write_text(dest_base, encoding="utf-8")
    except Exception:
        pass
    
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
        
        if approve_and_move(image_name, label_name, dest_base):
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
            border_width = float(_prompt_mem("pdf_border_width", "Enter border width in points", "2"))
        except ValueError:
            border_width = 2
            print("Using default border width: 2 points")
        
        print()
        
        # Run the script with border width as argument
        result = subprocess.run(['python', 'tools/add_pdf_borders_vector.py', str(border_width)], 
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
    """Run the split existing images script, then optionally tile fullsize images."""
    print("=== Split Existing Images ===")
    print("This will split images from <base>/images/train/ into train/val/test with 70/20/10 ratio.")
    print("Images will be moved to fullsize folders in respective splits.")
    print()
    
    # Prefill with last used base if available
    try:
        last_base = Path(".last_dataset_base.txt").read_text(encoding="utf-8").strip()
    except Exception:
        last_base = "datasets/rcp_dual_seg_v3"
    base_dir = select_base_dir(last_base or "datasets/rcp_dual_seg_v3")
    print("This will operate on:")
    print(f"  {base_dir}/images/train → {base_dir}/images/*/fullsize")
    cont = _prompt_mem("split_proceed", "Proceed? (y/N)", "y").strip().lower()
    if cont not in ["y", "yes"]:
        print("Cancelled.")
        return False
    
    try:
        # Run the split script
        cmd = [sys.executable, 'tools/split_existing_images.py', '--base-dir', base_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        # Prompt to tile immediately
        do_tile = _prompt_mem("tile_now", "Tile fullsize images now? (y/N)", "y").strip().lower()
        if do_tile in ["y", "yes"]:
            try:
                # Ask for tile params (defaults)
                ts = _prompt_mem("tile_size", "Tile size", "1280")
                ov = _prompt_mem("tile_overlap", "Overlap (0.0-1.0)", "0.30")
                tcmd = [
                    get_conda_python(), "tools/tile_fullsize_images.py",
                    "--base_dir", base_dir,
                    "--tile_size", ts,
                    "--overlap", ov,
                ]
                run_command(tcmd)
                # After tiling completes, generate dataset YAML for tiled1k
                try:
                    yaml_path = _write_dataset_yaml(base_dir, "tiled1k")
                    print(f"Created dataset YAML: {yaml_path}")
                except Exception as e:
                    print(f"Warning: failed to create dataset YAML: {e}")
            except subprocess.CalledProcessError as e:
                print(f"Tiling failed: {e}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running split script: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: split_existing_images.py not found")
        return False


def run_tile_fullsize_images():
    """Run the tiling script to create tiles with configurable size and overlap."""
    print("=== Tile Images ===")
    print("Create tiles from fullsize images with configurable size and overlap")
    print("Source: images in train/val/fullsize folders (test remains untiled)")
    print("Target: tiled1k folders")
    print("💡 Large tiles ensure complete panels without cropping")
    print()
    
    base_dir = select_base_dir("datasets/rcp_dual_seg_v3")
    
    # Get tiling parameters
    tile_size = _prompt_mem("tile_size", "Tile size in pixels", "1280")
    overlap = _prompt_mem("tile_overlap", "Overlap ratio (0.0-1.0)", "0.30")
    
    try:
        tile_size = int(tile_size)
        overlap = float(overlap)
    except ValueError:
        print("❌ Invalid tile size or overlap values!")
        return False
    
    print(f"This will CREATE or OVERWRITE tiles under:")
    print(f"  {base_dir}/images/*/tiled1k/")
    print(f"  Tile size: {tile_size}px")
    print(f"  Overlap: {overlap:.1%}")
    cont = input("Proceed? (y/N): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("Cancelled.")
        return False
    
    # Check if we have images to tile (now in fullsize subfolders)
    train_images = glob.glob(f"{base_dir}/images/train/fullsize/*.jpg") + glob.glob(f"{base_dir}/images/train/fullsize/*.png")
    val_images = glob.glob(f"{base_dir}/images/val/fullsize/*.jpg") + glob.glob(f"{base_dir}/images/val/fullsize/*.png")
    test_images = glob.glob(f"{base_dir}/images/test/fullsize/*.jpg") + glob.glob(f"{base_dir}/images/test/fullsize/*.png")
    
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
            sys.executable, "tools/tile_fullsize_images.py", 
            "--base_dir", base_dir, 
            "--tile_size", str(tile_size), 
            "--overlap", str(overlap), 
            "--splits", "train,val"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Tiling completed successfully!")
        print(result.stdout)
        
        # Show summary of created tiles
        print("\n=== Tiling Summary ===")
        for split in ['train', 'val']:
            tiled_dir = f"{base_dir}/images/{split}/tiled1k"
            if os.path.exists(tiled_dir):
                tile_count = len([f for f in os.listdir(tiled_dir) if f.lower().endswith(('.jpg', '.png'))])
                print(f"  {split}: {tile_count} tiles created")
        print(f"  test: left as fullsize (not tiled)")
        
        # Auto-generate dataset YAML for tiled1k
        try:
            yaml_path = _write_dataset_yaml(base_dir, "tiled1k")
            print(f"Created dataset YAML: {yaml_path}")
        except Exception as e:
            print(f"Warning: failed to create dataset YAML: {e}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tiling failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def cleanup_old_tiles():
    """Clean up old tiles in tiled1k folders (preserves fullsize data!)."""
    print("=== Cleanup Old Tiles ===")
    print("⚠️  IMPORTANT: This will only delete tiled data, preserving all fullsize data!")
    
    base_dataset = select_base_dir("datasets/rcp_dual_seg_v3")
    print("This will DELETE:")
    print(f"  {base_dataset}/images/*/tiled1k/* and {base_dataset}/labels/*/tiled1k/*")
    cont = input("Proceed with deletion? (y/N): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("Cancelled.")
        return False

    base_dir = f"{base_dataset}/images"
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
    base_label_dir = f"{base_dataset}/labels"
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


def run_panel_cropping(specific_split: str = None, specific_images: list = None):
    """
    Run panel-centric cropping.
    Can be limited to a specific split and/or specific images.
    """
    print_header("Panel-Centric Cropping")
    base_dir = prompt_for_dataset_dir()
    if not base_dir:
        return False
    
    target_size = 1280
    try:
        size_in = input(f"Target size for longest side? [{target_size}]: ").strip()
        if size_in:
            target_size = int(size_in)
    except ValueError:
        pass
    
    print(f"Target size set to {target_size}px.")

    delete = input("Do you want to delete all existing crops before starting? (y/N): ").lower().strip() == 'y'
    
    cmd = [
        get_conda_python(), "tools/panel_cropper.py",
        "--base-dir", base_dir,
        "--target-size", str(target_size),
    ]
    if delete:
        cmd.append("--force")
        
    # Handle specific split and images
    splits_to_process = [specific_split] if specific_split else ['train', 'val', 'test']
    
    if specific_images:
        # If specific images are given, we force the split to be the one provided (or default)
        # and pass the image list.
        cmd.extend(["--splits", splits_to_process[0]])
        cmd.extend(["--images", ",".join(specific_images)])
    else:
        # Standard batch mode
        cmd.extend(["--splits", ",".join(splits_to_process)])

    try:
        run_command(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Panel cropping failed: {e}")
        return False

def run_smart_augmentation(base_dir=None, target_size=None, augs_per_image=None, splits=None, 
                          in_folder=None, out_folder=None, tag_class_id=None, seed=None, 
                          jpeg_quality=None, opencv_num_threads=None, keep_existing=False, 
                          max_images=None, preset=None):
    """Run smart augmentation on the cropped dataset"""
    print_header("Smart Augmentation")
    
    # If no parameters provided, use interactive mode
    if base_dir is None:
        base_dir = prompt_for_dataset_dir()
        if not base_dir:
            return False
        
        # Load augmentation config for defaults
        try:
            from utils.augmentation_shared import load_augmentation_config
            cfg = load_augmentation_config()
            target_size = target_size or cfg.target_size
            augs_per_image = augs_per_image or cfg.augs_per_image
            splits = splits or _prompt_mem("aug_splits", "Splits to augment (comma-separated)", cfg.splits)
        except Exception:
            # Fallback to old defaults if config not available
            target_size = target_size or 1024
            augs_per_image = augs_per_image or 3
            splits = splits or _prompt_mem("aug_splits", "Splits to augment (comma-separated)", "train,val")
        
        # Ask for preset selection
        try:
            from utils.augmentation_shared import load_augmentation_config
            cfg = load_augmentation_config()
            if cfg.presets:
                print("\nAvailable augmentation presets:")
                for i, (name, preset) in enumerate(cfg.presets.items(), 1):
                    print(f"  {i}. {name}: rot={preset.get('rotation_deg', 'N/A')}°, scale={preset.get('scale_low', 'N/A')}-{preset.get('scale_high', 'N/A')}, augs={preset.get('augs_per_image', 'N/A')}")
                print(f"  {len(cfg.presets) + 1}. Custom settings")
                
                preset_choice = input(f"Choose preset [1-{len(cfg.presets) + 1}, default {len(cfg.presets) + 1}]: ").strip()
                if preset_choice and preset_choice.isdigit():
                    choice = int(preset_choice)
                    if 1 <= choice <= len(cfg.presets):
                        preset_names = list(cfg.presets.keys())
                        preset = preset_names[choice - 1]
                        print(f"Using preset: {preset}")
                    else:
                        preset = None
                else:
                    preset = None
            else:
                preset = None
        except Exception:
            preset = None
        
        try:
            if not preset:
                size_in = input(f"Target size for longest side? [{target_size}]: ").strip()
                if size_in:
                    target_size = int(size_in)
                augs_in = input(f"Augmentations per image? [{augs_per_image}]: ").strip()
                if augs_in:
                    augs_per_image = int(augs_in)
            src_folder = in_folder or _prompt_mem("aug_src_folder", "Source folder name (train)", "tiled1k")
            out_folder = out_folder or _prompt_mem("aug_out_folder", "Output folder name", f"augmented1k_{src_folder}")
        except ValueError:
            pass
            
        print(f"Target size set to {target_size}px, {augs_per_image} augmentations per image.")

        delete = input("Do you want to delete all existing augmented data before starting? (y/N): ").lower().strip() == 'y'
        keep_existing = not delete
    else:
        # Use provided parameters
        src_folder = in_folder
        if not splits:
            splits = "train,val"
    
    cmd = [
        get_conda_python(), "tools/smart_augmentation_v2.py",
        "--base-dir", base_dir,
        "--target-size", str(target_size),
        "--augs-per-image", str(augs_per_image),
        "--splits", splits,
        "--out-folder", out_folder,
    ]
    
    # Add preset if specified
    if preset:
        cmd.extend(["--preset", preset])
    
    # Add optional parameters if provided
    if src_folder:
        cmd.extend(["--in-folder", src_folder])
    if tag_class_id is not None:
        cmd.extend(["--tag-class-id", str(tag_class_id)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if jpeg_quality is not None:
        cmd.extend(["--jpeg-quality", str(jpeg_quality)])
    if opencv_num_threads is not None:
        cmd.extend(["--opencv-num-threads", str(opencv_num_threads)])
    if keep_existing:
        cmd.append("--keep-existing")
    if max_images is not None:
        cmd.extend(["--max-images", str(max_images)])
        
    try:
        run_command(cmd)
        # After augmentation completes, also write YAML for augmented1k if present
        try:
            yaml_path = _write_dataset_yaml_custom(base_dir, out_folder, "gtcenter1k", "tiled1k")
            print(f"Created dataset YAML: {yaml_path}")
        except Exception as e:
            print(f"Warning: failed to create dataset YAML: {e}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Smart augmentation failed: {e}")
        return False
        
def configure_training_setup():
    """Interactive setup for training; saves settings for reuse."""
    print_header("Configure Training")
    existing = _load_training_config()
    
    # Updated sensible defaults for detection training
    # Prefer last auto-generated YAML if available
    prefs = _prefs_load()
    default_dataset = existing.get("dataset_yaml", prefs.get("last_dataset_yaml", "datasets/rcp_bbox_v2+3/dataset.yaml"))
    default_weights = existing.get("init_weights", "yolov8s.pt")
    default_imgsz = str(existing.get("imgsz", 1024))
    default_batch = str(existing.get("batch", 16))
    default_epochs = str(existing.get("epochs", 150))
    default_device = str(existing.get("device", 0))
    default_patience = str(existing.get("patience", 30))
    default_optimizer = existing.get("optimizer", "AdamW")
    default_lr0 = str(existing.get("lr0", 0.01))
    default_wd = str(existing.get("weight_decay", 5e-4))
    default_project = existing.get("project", "runs/train")
    default_name = existing.get("name", "v2+3_bbox_yolov8s_det")

    # Detection augmentation defaults (heavier than seg)
    default_retina_masks = str(existing.get("retina_masks", False))
    default_mask_ratio = str(existing.get("mask_ratio", 4))
    default_overlap_mask = str(existing.get("overlap_mask", True))
    default_cos_lr = str(existing.get("cos_lr", True))
    # Heavy augs for detection (mosaic helps detection a lot)
    default_mosaic = str(existing.get("mosaic", 0.7))
    default_auto_augment = existing.get("auto_augment", "randaugment")
    default_erasing = str(existing.get("erasing", 0.2))
    default_hsv_s = str(existing.get("hsv_s", 0.7))
    default_hsv_v = str(existing.get("hsv_v", 0.4))
    default_translate = str(existing.get("translate", 0.1))
    default_scale = str(existing.get("scale", 0.5))
    default_degrees = str(existing.get("degrees", 10.0))
    default_flipud = str(existing.get("flipud", 0.0))
    default_fliplr = str(existing.get("fliplr", 0.25))
    default_mixup = str(existing.get("mixup", 0.0))
    default_cutmix = str(existing.get("cutmix", 0.0))
    default_copy_paste = str(existing.get("copy_paste", 0.0))
    # Dataloader performance knobs
    default_workers = str(existing.get("workers", 8))
    default_cache = existing.get("cache", "ram")
    # Resume support
    default_resume = str(existing.get("resume", False))
    
    dataset_yaml = input(f"Dataset YAML [{default_dataset}]: ").strip() or default_dataset
    init_weights = input(f"Init weights (.pt) [{default_weights}]: ").strip() or default_weights
    imgsz = int(input(f"Image size [{default_imgsz}]: ").strip() or default_imgsz)
    batch = int(input(f"Batch size [{default_batch}]: ").strip() or default_batch)
    epochs = int(input(f"Epochs [{default_epochs}]: ").strip() or default_epochs)
    device = input(f"Device (e.g., 0) [{default_device}]: ").strip() or str(default_device)
    patience = int(input(f"Patience [{default_patience}]: ").strip() or default_patience)
    optimizer = input(f"Optimizer [{default_optimizer}]: ").strip() or default_optimizer
    lr0 = float(input(f"Learning Rate (lr0) [{default_lr0}]: ").strip() or default_lr0)
    weight_decay = float(input(f"Weight Decay [{default_wd}]: ").strip() or default_wd)
    project = input(f"Project directory [{default_project}]: ").strip() or default_project
    name = input(f"Run name [{default_name}]: ").strip() or default_name

    # Ask for advanced mask + augmentation knobs (press Enter to accept defaults)
    retina_masks = (input(f"Retina masks (True/False) [{default_retina_masks}]: ").strip() or default_retina_masks)
    mask_ratio = int(float(input(f"Mask ratio (1,2,4) [{default_mask_ratio}]: ").strip() or default_mask_ratio))
    overlap_mask = (input(f"Overlap masks (True/False) [{default_overlap_mask}]: ").strip() or default_overlap_mask)
    cos_lr = (input(f"Cosine LR schedule (True/False) [{default_cos_lr}]: ").strip() or default_cos_lr)

    mosaic = float(input(f"mosaic [{default_mosaic}]: ").strip() or default_mosaic)
    auto_augment = input(f"auto_augment (None/randaugment/augmix/autoaugment) [{default_auto_augment}]: ").strip() or default_auto_augment
    erasing = float(input(f"erasing [{default_erasing}]: ").strip() or default_erasing)
    hsv_s = float(input(f"hsv_s [{default_hsv_s}]: ").strip() or default_hsv_s)
    hsv_v = float(input(f"hsv_v [{default_hsv_v}]: ").strip() or default_hsv_v)
    translate = float(input(f"translate [{default_translate}]: ").strip() or default_translate)
    scale = float(input(f"scale [{default_scale}]: ").strip() or default_scale)
    degrees = float(input(f"degrees (rotation) [{default_degrees}]: ").strip() or default_degrees)
    flipud = float(input(f"flipud [{default_flipud}]: ").strip() or default_flipud)
    fliplr = float(input(f"fliplr [{default_fliplr}]: ").strip() or default_fliplr)
    mixup = float(input(f"mixup [{default_mixup}]: ").strip() or default_mixup)
    cutmix = float(input(f"cutmix [{default_cutmix}]: ").strip() or default_cutmix)
    copy_paste = float(input(f"copy_paste [{default_copy_paste}]: ").strip() or default_copy_paste)
    workers = int(input(f"Workers [{default_workers}]: ").strip() or default_workers)
    cache_inp = input(f"Cache (None/ram/disk) [{default_cache}]: ").strip() or str(default_cache)
    cache = None if str(cache_inp).lower() in ("none", "false", "0", "") else cache_inp
    resume_inp = input(f"Resume last run (True/False) [{default_resume}]: ").strip() or default_resume
    resume = str(resume_inp).lower() in ("true", "1", "y", "yes")

    cfg = {
        "dataset_yaml": dataset_yaml,
        "init_weights": init_weights,
        "imgsz": imgsz,
        "batch": batch,
        "epochs": epochs,
        "device": device,
        "patience": patience,
        "optimizer": optimizer,
        "lr0": lr0,
        "weight_decay": weight_decay,
        "project": project,
        "name": name,
        # Advanced knobs
        "retina_masks": str(retina_masks).lower() in ("true", "1", "y", "yes"),
        "mask_ratio": mask_ratio,
        "overlap_mask": str(overlap_mask).lower() in ("true", "1", "y", "yes"),
        "cos_lr": str(cos_lr).lower() in ("true", "1", "y", "yes"),
        "mosaic": mosaic,
        "auto_augment": None if str(auto_augment).lower() == "none" else auto_augment,
        "erasing": erasing,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "translate": translate,
        "scale": scale,
        "degrees": degrees,
        "flipud": flipud,
        "fliplr": fliplr,
        "mixup": mixup,
        "cutmix": cutmix,
        "copy_paste": copy_paste,
        # Dataloader perf
        "workers": workers,
        "cache": cache,
        "resume": resume,
    }
    # Include optional accumulate if present in existing config/prefs
    try:
        acc = existing.get("accumulate")
        if acc is not None:
            cfg["accumulate"] = int(acc)
    except Exception:
        pass
    _save_training_config(cfg)

    print("\n✅ Saved training configuration to training_config.yaml:")
    for k, v in cfg.items():
        print(f"  - {k}: {v}")
    print("\nRun Option 10 in the Hub to start training with these settings.")
    return True
        
def start_training_from_saved_config():
    """Start YOLO training using saved settings."""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    cfg = _load_training_config()
    if not cfg:
        print("❌ No training_config.yaml found. Please run Configure Training first.")
        return False

    dataset_yaml = cfg.get("dataset_yaml")
    init_weights = cfg.get("init_weights")
    if not dataset_yaml or not Path(dataset_yaml).exists():
        print(f"❌ Dataset YAML not found: {dataset_yaml}")
        return False
    if not init_weights or not (Path(init_weights).exists() or init_weights.endswith('.pt')):
        print(f"⚠️  Init weights not found on disk: {init_weights}. Will attempt to load from hub.")

    print_header("Starting Training (from saved config)")
    for k, v in cfg.items():
        print(f"  - {k}: {v}")

    confirm = input("Proceed with these settings? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Training cancelled.")
        return False

    try:
        # If resuming, auto-switch init_weights to the last checkpoint if available
        resume_flag = bool(cfg.get('resume', False))
        if resume_flag:
            proj_dir = Path(str(cfg.get('project', 'runs/train')))
            run_name = str(cfg.get('name', 'v3_yolov8s_seg'))
            last_pt = proj_dir / run_name / 'weights' / 'last.pt'
            if last_pt.exists():
                init_weights = str(last_pt)
                print(f"Resuming from checkpoint: {init_weights}")
            else:
                print(f"⚠️  Resume requested but checkpoint not found at {last_pt}. Starting fresh.")
                resume_flag = False

        # Detect task type from model name or dataset
        is_detection = any(x in init_weights.lower() for x in ['det', 'yolov8s', 'yolov8m', 'yolov8n']) and 'seg' not in init_weights.lower()
        
        model = YOLO(init_weights)
        
        # Base training parameters
        train_kwargs = {
            'data': str(dataset_yaml),
            'imgsz': int(cfg.get('imgsz', 1024)),
            'batch': int(cfg.get('batch', 16)),
            'epochs': int(cfg.get('epochs', 150)),
            'device': str(cfg.get('device', 0)),
            'patience': int(cfg.get('patience', 30)),
            'optimizer': str(cfg.get('optimizer', 'AdamW')),
            'lr0': float(cfg.get('lr0', 0.01)),
            'weight_decay': float(cfg.get('weight_decay', 5e-4)),
            'project': str(cfg.get('project', 'runs/train')),
            'name': str(cfg.get('name', 'v2+3_bbox_yolov8s_det')),
            'pretrained': True,
            # Dataloader perf
            'workers': int(cfg.get('workers', 8)),
            'cache': cfg.get('cache', None),
            'resume': resume_flag,
            # Scheduling
            'cos_lr': bool(cfg.get('cos_lr', True)),
            # Augmentations
            'mosaic': float(cfg.get('mosaic', 0.7)),
            'auto_augment': cfg.get('auto_augment', 'randaugment'),
            'erasing': float(cfg.get('erasing', 0.2)),
            'hsv_s': float(cfg.get('hsv_s', 0.7)),
            'hsv_v': float(cfg.get('hsv_v', 0.4)),
            'translate': float(cfg.get('translate', 0.1)),
            'scale': float(cfg.get('scale', 0.5)),
            'degrees': float(cfg.get('degrees', 10.0)),
            'flipud': float(cfg.get('flipud', 0.0)),
            'fliplr': float(cfg.get('fliplr', 0.25)),
            'mixup': float(cfg.get('mixup', 0.0)),
            'cutmix': float(cfg.get('cutmix', 0.0)),
            'copy_paste': float(cfg.get('copy_paste', 0.0)),
        }
        
        # Add segmentation-specific parameters only if needed
        if not is_detection:
            train_kwargs.update({
                'retina_masks': bool(cfg.get('retina_masks', True)),
                'mask_ratio': int(cfg.get('mask_ratio', 1)),
                'overlap_mask': bool(cfg.get('overlap_mask', False)),
            })
        
        results = model.train(**train_kwargs)
        print("✅ Training complete!")
        print(f"Best weights: {results.save_dir / 'weights' / 'best.pt'}")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
        
def analyze_dataset_structure():
    """Analyze dataset structure and distribution using the utility script."""
    print_header("Dataset Analysis")
    base_dir = prompt_for_dataset_dir()
    if not base_dir:
        return False
    
    try:
        run_command([get_conda_python(), "utils/analyze_dataset.py", "--base-dir", base_dir])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Dataset analysis failed: {e}")
        return False

def main():
    # This main function is now primarily for legacy use.
    # The hub.py script is the recommended new interface.
    print("This is the legacy workflow menu. It's recommended to run hub.py instead.")
    # ... (old menu logic can be kept here if needed for direct script execution)

if __name__ == "__main__":
    main()
