"""
Hub menu for the YOLO RCP Dual-Seg workflow (cleaned).

This file wraps selected utilities from workflow.py with a simplified menu:
- Top colors → ROI crop (stage outputs)
- Visualize labels
- Approve ALL pending files
- Split images (existing or pending)
- Tile fullsize images
- Verify labels
- Smart augmentation
- Analyze dataset
- Test GPU memory (prompt size/batch)
- Configure Training
- Train
- Predict (fullsize images or pdfs)
- Checklist
- Add borders to PDF files
- Panel-centric cropping
- Exit

workflow.py remains untouched as a backup/legacy menu.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess
from pathlib import Path
import argparse
import time
import json
import random

# Add src to path so we can import workflow
sys.path.append("src")
import workflow as wf


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

def menu_visualize_labels() -> bool:
    print("=== Visualize labels ===")
    print("1) Pending single-image preview (interactive)")
    print("2) Pending ALL images (save to output dir)")
    print("3) Folder batch overlays")
    print("4) Random 10 from chosen folders")
    mode = _prompt_mem("vis_mode", "Choose [1/2/3/4]", "1")

    if mode == "2":
        return wf.visualize_all_pending_labels_batch()
    if mode == "3":
        return menu_overlay_labels()
    if mode == "4":
        # Ask for arbitrary images/labels, randomly sample 10 pairs, overlay
        img_dir = _prompt_mem("rand_vis_images", "Images directory", "datasets/rcp_bbox_v4/images/train/augmented2x_gtcenter_merged")
        lbl_dir = _prompt_mem("rand_vis_labels", "Labels directory", "datasets/rcp_bbox_v4/labels/train/augmented2x_gtcenter_merged")
        try:
            img_root = Path(img_dir); lbl_root = Path(lbl_dir)
            if not img_root.exists() or not lbl_root.exists():
                print("Images or labels directory does not exist.")
                return False
            # Gather candidates with matching labels
            imgs = sorted(list(img_root.glob("*.jpg")) + list(img_root.glob("*.jpeg")) + list(img_root.glob("*.png")))
            pairs = []
            for p in imgs:
                lp = (lbl_root / p.stem).with_suffix(".txt")
                if lp.exists():
                    pairs.append((p, lp))
            if not pairs:
                print("No image/label pairs found in the provided folders.")
                return False
            k = min(10, len(pairs))
            sample = random.sample(pairs, k)
            # Prepare temp dirs
            ts = int(time.time())
            tmp_base = Path("runs/overlays") / f"random_{ts}"
            tmp_img = tmp_base / "images"
            tmp_lbl = tmp_base / "labels"
            out_dir = tmp_base / "overlays"
            tmp_img.mkdir(parents=True, exist_ok=True)
            tmp_lbl.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Copy sampled pairs
            n = 0
            for ip, lp in sample:
                tgt_i = tmp_img / ip.name
                tgt_l = tmp_lbl / f"{ip.stem}.txt"
                subprocess.run([sys.executable, "-c", "import shutil,sys;shutil.copy2(sys.argv[1], sys.argv[2])", str(ip), str(tgt_i)], check=True)
                subprocess.run([sys.executable, "-c", "import shutil,sys;shutil.copy2(sys.argv[1], sys.argv[2])", str(lp), str(tgt_l)], check=True)
                n += 1
            # Run overlay on the temp selection
            cmd = [
                sys.executable, "utils/overlay_labels_folder.py",
                "--images", str(tmp_img),
                "--labels", str(tmp_lbl),
                "--out", str(out_dir),
                "--limit", "10000",
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            print(f"Saved {n} random overlays to: {out_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print("Overlay failed for random selection.")
            print(e.stdout)
            print(e.stderr)
            return False

    pending_images, pending_labels = wf.list_pending_files()
    if not pending_images:
        print("No images found in pending folder.")
        return False

    print("\nAvailable images:")
    for i, img in enumerate(sorted(pending_images), 1):
        print(f"  {i}. {Path(img).name}")
    try:
        choice_num = int(input("\nEnter image number: ").strip())
    except ValueError:
        print("Please enter a valid number.")
        return False
    if 1 <= choice_num <= len(pending_images):
        selected_image = sorted(pending_images)[choice_num - 1]
        image_name = Path(selected_image).stem
        label_path = f"pending/labels/{image_name}.txt"
        return wf.visualize_pending_labels(selected_image, label_path)
    print("Invalid image number.")
    return False


def menu_approve_all() -> bool:
    return wf.approve_all_pending()


def menu_pdf_borders() -> bool:
    return wf.run_pdf_border_script()


def menu_split_existing() -> bool:
    print("=== Split existing images ===")
    print("1) Split existing images (70/20/10) - standard workflow")
    print("2) Split pending images into new dataset folder")
    choice = _prompt_mem("split_mode", "Choose [1/2]", "1")
    
    if choice == "1":
        return wf.run_split_existing_images()
    elif choice == "2":
        return menu_split_pending_to_dataset()
    else:
        print("Invalid choice.")
        return False


def menu_split_pending_to_dataset() -> bool:
    """Split all images in pending into a new dataset folder with train/val/test splits."""
    print("=== Split pending images into new dataset ===")
    
    # Check if pending folder exists and has images
    pending_dir = Path("pending")
    pending_img_dir = pending_dir / "images"
    pending_lbl_dir = pending_dir / "labels"
    
    if not pending_dir.exists():
        print("❌ Pending directory not found!")
        return False
    
    if not pending_img_dir.exists():
        print("❌ Pending/images directory not found!")
        return False
    
    # Find all image files in pending/images that actually exist
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        found_files = list(pending_img_dir.glob(ext))
        # Only include files that actually exist
        image_files.extend([f for f in found_files if f.exists()])
    
    if not image_files:
        print("❌ No images found in pending/images directory!")
        return False
    
    print(f"Found {len(image_files)} images in pending/images directory")
    
    # Get dataset name and base directory
    dataset_name = _prompt_mem("new_dataset_name", "New dataset name (e.g., rcp_bbox_v5)", "rcp_bbox_v5")
    base_dir = Path("datasets") / dataset_name
    
    # Check if dataset already exists
    if base_dir.exists():
        print(f"⚠️  Dataset {base_dir} already exists!")
        clean = input("Do you want to delete existing dataset and start fresh? (y/N): ").strip().lower()
        if clean in ["y", "yes"]:
            import shutil
            try:
                shutil.rmtree(base_dir)
                print(f"Deleted existing {base_dir}")
            except PermissionError:
                print(f"❌ Permission denied when deleting {base_dir}")
                print("Please manually delete the folder or close any programs that might be using it.")
                return False
            except Exception as e:
                print(f"❌ Error deleting {base_dir}: {e}")
                return False
        else:
            print("Cancelled.")
            return False
    
    print(f"This will create: {base_dir}/images/{{train,val,test}}/fullsize/")
    print(f"And move ALL images from pending/ with 70/20/10 split")
    
    cont = input("Proceed? (y/N): ").strip().lower()
    if cont not in ["y", "yes"]:
        print("Cancelled.")
        return False
    
    try:
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (base_dir / "images" / split / "fullsize").mkdir(parents=True, exist_ok=True)
            (base_dir / "labels" / split / "fullsize").mkdir(parents=True, exist_ok=True)
        
        # Shuffle images for random split
        import random
        random.shuffle(image_files)
        
        # Calculate split sizes (70/20/10)
        total = len(image_files)
        train_count = int(total * 0.7)
        val_count = int(total * 0.2)
        test_count = total - train_count - val_count
        
        print(f"Will split {total} images: {train_count} train, {val_count} val, {test_count} test")
        
        # Move images to appropriate splits
        moved_count = 0
        for i, img_path in enumerate(image_files):
            if i < train_count:
                split = 'train'
            elif i < train_count + val_count:
                split = 'val'
            else:
                split = 'test'
            
            # Check if source file still exists before moving
            if not img_path.exists():
                print(f"⚠️  Warning: {img_path} no longer exists, skipping...")
                continue
            
            # Move image
            dst_img = base_dir / "images" / split / "fullsize" / img_path.name
            try:
                img_path.rename(dst_img)
                print(f"  Moved {img_path.name} → {split}")
            except Exception as e:
                print(f"❌ Error moving {img_path.name}: {e}")
                continue
            
            # Check for corresponding label file in pending/labels
            label_path = pending_lbl_dir / img_path.with_suffix('.txt').name
            if label_path.exists():
                dst_lbl = base_dir / "labels" / split / "fullsize" / label_path.name
                try:
                    label_path.rename(dst_lbl)
                    print(f"  Moved {label_path.name} → {split}")
                except Exception as e:
                    print(f"❌ Error moving {label_path.name}: {e}")
            else:
                print(f"  No label file found for {img_path.name}")
            
            moved_count += 1
        
        # Count actual files moved to each split
        actual_train = len(list((base_dir / "images" / "train" / "fullsize").glob("*.jpg")))
        actual_val = len(list((base_dir / "images" / "val" / "fullsize").glob("*.jpg")))
        actual_test = len(list((base_dir / "images" / "test" / "fullsize").glob("*.jpg")))
        
        print(f"✅ Successfully moved {moved_count} images to {base_dir}")
        print(f"   Train: {actual_train} images")
        print(f"   Val: {actual_val} images") 
        print(f"   Test: {actual_test} images")
        
        # Create dataset YAML file
        yaml_content = f"""train: images/train/fullsize
val: images/val/fullsize
test: images/test/fullsize
nc: 1
names:
- panel
task: detect
"""
        yaml_path = base_dir / f"dataset_{dataset_name}.yaml"
        yaml_path.write_text(yaml_content)
        print(f"Created dataset YAML: {yaml_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during split: {e}")
        return False


def menu_panel_cropping() -> bool:
    """Modified to allow cropping all pending or selected pending images for debug."""
    print("=== Panel-centric cropping ===")
    
    pending_img_dir = Path("pending/images")
    image_files = sorted([f for f in pending_img_dir.glob('*.jpg')])
    
    if not image_files:
        print("No .jpg images found in pending/images to crop.")
        return False

    print("1) Crop ALL pending images (standard workflow)")
    print("2) Crop SELECTED pending images (isolated debug output)")
    choice = input("Choose [1/2, default 1]: ").strip() or "1"

    if choice == '1':
        # Standard workflow: approve, split, then crop the whole dataset
        print("\nStandard workflow requires approving and splitting files first.")
        print("This option will run the standard batch cropper on the main dataset.")
        return wf.run_panel_cropping()

    # Debug workflow for selected images
    print("\nAvailable pending images for debug cropping:")
    for i, f in enumerate(image_files, 1):
        print(f"  {i}. {f.name}")
    
    try:
        selections = input("Enter numbers to process (e.g., 1,3,4): ").strip()
        if not selections:
            print("No selection made. Aborting.")
            return False
        indices = [int(s.strip()) - 1 for s in selections.split(',')]
        images_to_process = [image_files[i] for i in indices if 0 <= i < len(image_files)]
        if not images_to_process:
            print("No valid selections made.")
            return False
    except (ValueError, IndexError):
        print("Invalid input.")
        return False

    default_out = Path("datasets/rcp_dual_seg_v3/debug_crops")
    out_dir_str = input(f"\nOutput directory for debug crops [default: {default_out}]: ").strip()
    out_dir = Path(out_dir_str) if out_dir_str else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(images_to_process)} images. Output will be saved in '{out_dir}'.")
    
    target_size = 1280
    try:
        size_in = input(f"Target size for longest side? [{target_size}]: ").strip()
        if size_in:
            target_size = int(size_in)
    except ValueError:
        pass
    print(f"Target size set to {target_size}px.")

    overall_success = True
    for img_path in images_to_process:
        lbl_path = Path("pending/labels") / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"Warning: Label file not found for {img_path.name}, skipping.")
            continue
        
        print(f"\n--- Cropping {img_path.name} ---")
        try:
            cmd = [
                sys.executable, "panel_cropper.py",
                "--single-image", str(img_path),
                "--single-label", str(lbl_path),
                "--out-debug", str(out_dir),
                "--target-size", str(target_size),
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"ERROR processing {img_path.name}:")
            print(e.stdout)
            print(e.stderr)
            overall_success = False
            
    if overall_success:
        print(f"\n✅ Successfully processed all selected images.")
        print(f"Check the subfolders in '{out_dir}' for images, labels, and overlays.")
    else:
        print(f"\n❌ One or more images failed to process.")
        
    return overall_success


def menu_augmentation() -> bool:
    """Run smart augmentation with configurable parameters"""
    print("=== Smart Augmentation ===")
    print("Configure augmentation settings")
    print()
    
    # Get basic parameters with defaults
    base_dir = _prompt_mem("aug_base_dir", "Dataset base directory", "datasets/rcp_bbox_v5_non-ROI")
    target_size = _prompt_mem("aug_target_size", "Target size (pixels)", "1024")
    augs_per_image = _prompt_mem("aug_per_image", "Augmentations per image", "3")
    splits = _prompt_mem("aug_splits", "Splits to augment (comma-separated)", "train,val")
    in_folder = _prompt_mem("aug_in_folder", "Source folder name", "tiled1k")
    out_folder = _prompt_mem("aug_out_folder", "Output folder name", "augmented1k")
    
    # Optional settings
    print("\nOptional settings (press Enter to accept defaults):")
    tag_class_id = input("  Tag class id [1]: ").strip() or "1"
    seed = input("  Seed [0]: ").strip() or "0"
    jpeg_quality = input("  JPEG quality [92]: ").strip() or "92"
    opencv_threads = input("  OpenCV threads [0]: ").strip() or "0"
    keep_existing = input("  Keep existing outputs? (y/N): ").strip().lower() == "y"
    max_images_input = input("  Max images per split (blank for all): ").strip()
    max_images = max_images_input if max_images_input else None
    
    # Validate paths exist
    if not Path(base_dir).exists():
        print(f"❌ Base directory not found: {base_dir}")
        return False
    
    print(f"\nAugmentation settings:")
    print(f"  Base directory: {base_dir}")
    print(f"  Target size: {target_size}px")
    print(f"  Augmentations per image: {augs_per_image}")
    print(f"  Splits: {splits}")
    print(f"  Source folder: {in_folder}")
    print(f"  Output folder: {out_folder}")
    print(f"  Tag class id: {tag_class_id}")
    print(f"  Seed: {seed}")
    print(f"  JPEG quality: {jpeg_quality}")
    print(f"  OpenCV threads: {opencv_threads}")
    print(f"  Keep existing: {'yes' if keep_existing else 'no'}")
    print(f"  Max images: {max_images if max_images else 'all'}")
    
    confirm = input("\nProceed with augmentation? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Augmentation cancelled.")
        return False
    
    try:
        # Convert string inputs to appropriate types
        target_size_int = int(target_size)
        augs_per_image_int = int(augs_per_image)
        tag_class_id_int = int(tag_class_id)
        seed_int = int(seed)
        jpeg_quality_int = int(jpeg_quality)
        opencv_threads_int = int(opencv_threads)
        max_images_int = int(max_images) if max_images else None
        
        # Call workflow with all parameters
        return wf.run_smart_augmentation(
            base_dir=base_dir,
            target_size=target_size_int,
            augs_per_image=augs_per_image_int,
            splits=splits,
            in_folder=in_folder,
            out_folder=out_folder,
            tag_class_id=tag_class_id_int,
            seed=seed_int,
            jpeg_quality=jpeg_quality_int,
            opencv_num_threads=opencv_threads_int,
            keep_existing=keep_existing,
            max_images=max_images_int
        )
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return False
    except Exception as e:
        print(f"❌ Augmentation failed: {e}")
        return False


def menu_tile_fullsize_images() -> bool:
    return wf.run_tile_fullsize_images()


def menu_verify_labels() -> bool:
    """Run label validation with configurable parameters"""
    print("=== Verify Labels ===")
    print("Validate YOLO labels against a trained model")
    print()
    
    # Get validation parameters with defaults
    images_dir = _prompt_mem("val_images", "Images directory", "datasets\\rcp_bbox_v5_non-ROI\\images\\train\\tiled1280")
    labels_dir = _prompt_mem("val_labels", "Labels directory", "datasets\\rcp_bbox_v5_non-ROI\\labels\\train\\tiled1280")
    
    # Ask for validation mode
    print("\nValidation options:")
    print("1. Audit only (quick label checks, no model validation)")
    print("2. Full validation (audit + model validation)")
    mode_choice = input("Choose mode (1/2): ").strip()
    
    if mode_choice == "1":
        # Audit only mode
        audit_only = True
        model_path = None
        class_names = None
        img_size = None
        batch_size = None
        device = None
        save_dir = None
    else:
        # Full validation mode
        audit_only = False
        model_path = _prompt_mem("val_model", "Model weights path", "runs\\train\\fine-tune_v4.4_yolov8m_v5\\weights\\best.pt")
        class_names = _prompt_mem("val_names", "Class names (comma-separated)", "panel")
        img_size = _prompt_mem("val_imgsz", "Image size", "1280")
        batch_size = _prompt_mem("val_batch", "Batch size", "2")
        device = _prompt_mem("val_device", "Device", "0")
        save_dir = _prompt_mem("val_save_dir", "Save directory", "runs\\val\\v6_train_tiles_check")
    
    # Validate paths exist
    if not Path(images_dir).exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not Path(labels_dir).exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
    
    if not audit_only and not Path(model_path).exists():
        print(f"❌ Model weights not found: {model_path}")
        return False
    
    print(f"\nValidation settings:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    if audit_only:
        print(f"  Mode: Audit only (quick label checks)")
    else:
        print(f"  Mode: Full validation (audit + model)")
        print(f"  Model: {model_path}")
        print(f"  Classes: {class_names}")
        print(f"  Image size: {img_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
        print(f"  Save to: {save_dir}")
    
    confirm = input("\nProceed with validation? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Validation cancelled.")
        return False
    
    try:
        # Build command
        cmd = [
            sys.executable, "tools/250923_validate_bbox_labels.py",
            "--images", images_dir,
            "--labels", labels_dir
        ]
        
        if audit_only:
            cmd.append("--audit-only")
        else:
            cmd.extend([
                "--model", model_path,
                "--names", class_names,
                "--imgsz", img_size,
                "--batch", batch_size,
                "--device", device,
                "--save_dir", save_dir
            ])
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Validation completed successfully!")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Info:")
            print(result.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Validation failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def menu_analyze_dataset() -> bool:
    return wf.analyze_dataset_structure()


def menu_overlay_labels() -> bool:
    print("=== Folder Batch Overlays ===")
    pending_img = Path("pending/images"); pending_lbl = Path("pending/labels")
    roi_img = Path("inference/ROI/ROI_images"); roi_lbl = Path("inference/ROI/ROI_labels")
    has_roi = roi_img.exists() and roi_lbl.exists()

    print("1) Pending (pending/images + pending/labels)")
    if has_roi:
        print("2) ROI (inference/ROI/ROI_images + inference/ROI/ROI_labels)")
    print("3) Custom folders")
    choice = _prompt_mem("overlay_choice", "Choose [1/2/3]", "1")

    if choice == "2" and has_roi:
        images_dir = str(roi_img)
        labels_dir = str(roi_lbl)
        default_out = Path("inference/ROI/overlays")
    elif choice == "3":
        images_dir = _prompt_mem("overlay_images", "Images directory", str(pending_img))
        labels_dir = _prompt_mem("overlay_labels", "Labels directory", str(pending_lbl))
        default_base = _prefs_load().get("overlay_out_base", "runs/overlays")
        default_out = Path(default_base) / f"custom_{int(time.time())}"
    else:
        images_dir = str(pending_img)
        labels_dir = str(pending_lbl)
        default_base = _prefs_load().get("overlay_out_base", "runs/overlays")
        default_out = Path(default_base) / f"pending_{int(time.time())}"

    out_dir_inp = input(f"Output dir for overlays [default: {default_out}]: ").strip()
    out_dir = Path(out_dir_inp) if out_dir_inp else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    _remember("overlay_out_base", str(out_dir.parent))

    try:
        cmd = [
            sys.executable, "utils/overlay_labels_folder.py",
            "--images", images_dir,
            "--labels", labels_dir,
            "--out", str(out_dir),
            "--limit", "10000",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"Saved overlays to: {out_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print("Overlay script failed.")
        print(e.stdout)
        print(e.stderr)
        return False


def menu_top_colors_roi_pipeline() -> bool:
    """Run Top-Colors → Mask → Manifest → ROI crop, then stage outputs.

    Steps:
    1) Build Excel at reports/top_colors.xlsx from an input (PDF/image/folder)
    2) Generate masks/overlays/ROI + manifest.jsonl
    3) Regenerate ROI labels from manifest (with fallback keep-nearest)
    4) Stage outputs either to pending/ (for Option 1) or a dataset base (for Option 4)
    """
    print("=== Top Colors → ROI Crop Pipeline ===")
    inp = _prompt_mem("tc_input", "Path to image/PDF or folder", "inference/fullsize").strip('"')
    excel = _prompt_mem("tc_excel", "Excel output path", "reports/top_colors.xlsx")
    out_root = _prompt_mem("tc_out_root", "Masks/ROI output root", "inference/ROI")
    manifest = _prompt_mem("tc_manifest", "Manifest path", "inference/ROI/manifest.jsonl")

    try:
        # 1) Report top colors
        cmd1 = [
            sys.executable, "utils/report_top_colors.py",
            "--input", inp,
            "--out", excel,
            "--dpi", "300",
            "--topn", "3",
        ]
        result1 = subprocess.run(cmd1, check=True, capture_output=True, text=True)
        if result1.stdout:
            print(result1.stdout)

        # 2) Apply masks + manifest
        cmd2 = [
            sys.executable, "utils/apply_top_colors_mask.py",
            "--excel", excel,
            "--out", out_root,
            "--use-topn", "2",
            "--tol-frac", "0.05",
            "--down", "2048",
            "--black-cutoff", "30",
            "--nbhd-ksize", "5",
            "--nbhd-frac", "0.58",
            "--min-comp-area-frac", "0.0004",
            "--twoof3",
            "--close-k", "2",
            "--overlay-alpha", "0.45",
            "--cap-mask-frac", "0.55",
            "--cap-strategy", "score",
            "--edge-min", "12",
            "--sat-keep-max", "55",
            "--crop-mode", "bbox",
            "--pad-frac", "0.01",
            "--visible-min-frac", "0.00",
            "--manifest", manifest,
            "--manifest-overwrite",
        ]
        result2 = subprocess.run(cmd2, check=True, capture_output=True, text=True)
        if result2.stdout:
            print(result2.stdout)

        # 3) Crop strictly from manifest and regenerate labels (with fallback)
        cmd3 = [
            sys.executable, "utils/crop_from_manifest.py",
            "--manifest", manifest,
            "--visible-min-frac", "0.00",
            "--force-keep-nearest",
        ]
        result3 = subprocess.run(cmd3, check=True, capture_output=True, text=True)
        if result3.stdout:
            print(result3.stdout)

    except subprocess.CalledProcessError as e:
        print("❌ Pipeline failed:")
        print(e.stdout)
        print(e.stderr)
        return False

    # 4) Staging choice
    print("\nStaging options:")
    print("1) Stage to pending/ for quick visualization (use Hub Option 1)")
    print("2) Stage to a NEW dataset base for splitting (use Hub Option 4)")
    print("3) Do nothing now")
    choice = _prompt_mem("tc_stage_choice", "Choose [1/2/3]", "2")

    roi_images = Path(out_root) / "ROI_images"
    roi_labels = Path(out_root) / "ROI_labels"

    def _copy_dir(src_dir: Path, dst_dir: Path, exts: tuple) -> int:
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(src_dir.glob("*")):
            if p.suffix.lower() in exts:
                tgt = dst_dir / p.name
                subprocess.run([sys.executable, "-c", "import shutil,sys;shutil.copy2(sys.argv[1], sys.argv[2])", str(p), str(tgt)], check=True)
                n += 1
        return n

    try:
        if choice == "1":
            # Stage to pending
            n_img = _copy_dir(roi_images, Path("pending/images"), (".jpg", ".jpeg", ".png"))
            n_lbl = _copy_dir(roi_labels, Path("pending/labels"), (".txt",))
            print(f"✅ Staged {n_img} images and {n_lbl} labels to pending/. Use Hub Option 1 to visualize.")
        elif choice == "2":
            # Stage to dataset base
            parent = _prompt_mem("dst_parent", "Dataset parent directory", "datasets")
            name = _prompt_mem("dst_name", "Dataset name", "rcp_dual_seg_vX")
            base = Path(parent) / name
            img_train = base / "images" / "train"
            lbl_train = base / "labels" / "train"
            img_train.mkdir(parents=True, exist_ok=True)
            lbl_train.mkdir(parents=True, exist_ok=True)
            n_img = _copy_dir(roi_images, img_train, (".jpg", ".jpeg", ".png"))
            n_lbl = _copy_dir(roi_labels, lbl_train, (".txt",))
            print(f"✅ Staged {n_img} images and {n_lbl} labels to {base}/images/train and /labels/train.")
            print("→ Use Hub Option 4 to split into train/val/test fullsize folders.")
            _remember("dataset_base", str(base))
        else:
            print("Skipped staging.")
    except subprocess.CalledProcessError as e:
        print("❌ Staging failed:")
        print(e.stdout)
        print(e.stderr)
        return False

    return True


def menu_test_gpu_memory() -> bool:
    print("=== Test GPU memory ===")
    try:
        imgsz = int(input("Image size (e.g., 1024, 1280, 1408) [1408]: ") or "1408")
    except ValueError:
        imgsz = 1408
    try:
        batch = int(input("Batch size [1]: ") or "1")
    except ValueError:
        batch = 1

    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Allocating tensor: batch={batch}, size={imgsz}x{imgsz}, device={device}")
        tensor = torch.empty((batch, 3, imgsz, imgsz), dtype=torch.float32, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated()
            print(f"✅ Allocation succeeded. Max CUDA memory allocated: {mem/1e6:.1f} MB")
        else:
            print("✅ Allocation succeeded on CPU.")
        del tensor
        return True
    except Exception as e:
        print(f"❌ Allocation failed: {e}")
        return False


def menu_configure_training() -> bool:
    return wf.configure_training_setup()


def menu_train() -> bool:
    return wf.start_training_from_saved_config()


def menu_predict() -> bool:
    print("=== Predict (fullsize images or pdfs) ===")
    inp = _prompt_mem("predict_input", "Path to image/PDF or folder", "inference").strip('"')
    if not inp:
        print("No input provided.")
        return False

    # Let user choose model weights
    print("\nAvailable model weights:")
    weights_dir = Path("runs/train")
    if weights_dir.exists():
        weight_files = []
        for run_dir in sorted(weights_dir.iterdir()):
            if run_dir.is_dir():
                best_pt = run_dir / "weights" / "best.pt"
                if best_pt.exists():
                    weight_files.append(str(best_pt))

        if weight_files:
            for i, wfpath in enumerate(weight_files, 1):
                print(f"  {i}. {wfpath}")
            print(f"  {len(weight_files) + 1}. Use default (yolov8m.pt)")

            print("\nOptions:")
            print("1) Single model inference")
            print("2) Batch inference on multiple models")
            mode_choice = input("Choose mode [1/2, default 1]: ").strip() or "1"

            # Optional extra flags for the predictor
            print("\n(Optional) Extra flags for predictor (e.g. --content-crop --iqr). Leave blank to skip.")
            extra_flags = input("Extra flags: ").strip()

            if mode_choice == "2":
                print("\nBatch inference - specify models to use:")
                print("Examples: '1-5' (models 1 through 5), '1,3,7', or '1-3,7,10-12'")
                model_selection = input("Enter model selection: ").strip()
                if not model_selection:
                    print("No selection made, using first model only.")
                    selected_weights = [weight_files[0]]
                else:
                    selected_weights = parse_model_selection(model_selection, weight_files)
                    if not selected_weights:
                        print("Invalid selection, using first model only.")
                        selected_weights = [weight_files[0]]

                return run_batch_inference(inp, selected_weights, extra_flags=extra_flags)
            else:
                try:
                    choice = int(input(f"\nChoose model [1-{len(weight_files) + 1}, default 1]: ").strip() or "1")
                    if 1 <= choice <= len(weight_files):
                        weights = weight_files[choice - 1]
                    elif choice == len(weight_files) + 1:
                        weights = "yolov8m.pt"
                    else:
                        print("Invalid choice, using first model.")
                        weights = weight_files[0]
                except ValueError:
                    print("Invalid input, using first model.")
                    weights = weight_files[0]

                # Build output folder name
                if weights != "yolov8m.pt":
                    model_name = Path(weights).parent.parent.name
                    output_folder_name = f"pred_{model_name}"
                else:
                    output_folder_name = "pred_default"

                input_path = Path(inp)
                output_folder = (input_path.parent if input_path.is_file() else input_path) / output_folder_name
                print(f"Using weights: {weights}")
                print(f"Output folder: {output_folder}")

                try:
                    cmd = [
                        sys.executable, "src/predict_panels.py",
                        "--input", inp,
                        "--weights", weights,
                        "--out", str(output_folder),
                        "--imgsz", "1024",
                        "--conf", "0.40",
                        "--iou", "0.25",
                        "--merge-iou", "0.70",
                        "--keep-inner-frac", "0.535",
                        "--merge-mode", "iou_avg",
                        "--center-merge-ratio", "0.30",
                        "--max-det", "3000",
                        "--use-training-tiler",
                    ]
                    if extra_flags:
                        cmd.extend(extra_flags.split())

                    result = subprocess.run(cmd, check=True)
                    return result.returncode == 0
                except subprocess.CalledProcessError:
                    print("Error running prediction.")
                    return False
        else:
            weights = "yolov8m.pt"
            print("No trained models found, using default yolov8m.pt")
    else:
        weights = "yolov8m.pt"
        print("No runs/train directory found, using default yolov8m.pt")

    # Fallback path if no runs/train or no models
    print(f"Using weights: {weights}")
    input_path = Path(inp)
    output_folder = (input_path.parent if input_path.is_file() else input_path) / "pred_default"
    print(f"Output folder: {output_folder}")

    try:
        cmd = [
            sys.executable, "predict_panels.py",
            "--input", inp,
            "--weights", weights,
            "--out", str(output_folder),
            "--imgsz", "1024",
            "--conf", "0.40",
            "--iou", "0.25",
            "--merge-iou", "0.70",
            "--keep-inner-frac", "0.535",
            "--merge-mode", "iou_avg",
            "--center-merge-ratio", "0.30",
            "--max-det", "3000",
            "--use-training-tiler",
        ]
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print("Error running prediction.")
        return False

def parse_model_selection(selection: str, weight_files: list) -> list:
    """Parse model selection string like '1-5', '1,3,7', or '1-3,7,10-12'"""
    selected_weights = []
    try:
        # Split by comma to handle multiple ranges/selections
        parts = selection.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle range like '1-5'
                start, end = map(int, part.split('-'))
                for i in range(start, end + 1):
                    if 1 <= i <= len(weight_files):
                        selected_weights.append(weight_files[i - 1])
            else:
                # Handle single number like '7'
                num = int(part)
                if 1 <= num <= len(weight_files):
                    selected_weights.append(weight_files[num - 1])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_weights = []
        for w in selected_weights:
            if w not in seen:
                seen.add(w)
                unique_weights.append(w)
        
        return unique_weights
    except (ValueError, IndexError):
        return []


def run_batch_inference(input_path: str, selected_weights: list, extra_flags: str = "") -> bool:
    """Run inference on multiple models with the same input via predict_panels.py"""
    input_path_obj = Path(input_path)

    print(f"\nStarting batch inference on {len(selected_weights)} models...")
    print(f"Input: {input_path}")

    success_count = 0
    for i, weights in enumerate(selected_weights, 1):
        print(f"\n--- Model {i}/{len(selected_weights)} ---")

        if weights != "yolov8m.pt":
            model_name = Path(weights).parent.parent.name
            output_folder_name = f"pred_{model_name}"
        else:
            output_folder_name = "pred_default"

        output_folder = (input_path_obj.parent if input_path_obj.is_file() else input_path_obj) / output_folder_name
        print(f"Model: {weights}")
        print(f"Output: {output_folder}")

        try:
            cmd = [
                sys.executable, "predict_panels.py",
                "--input", input_path,
                "--weights", weights,
                "--out", str(output_folder),
                "--imgsz", "1024",
                "--conf", "0.40",
                "--iou", "0.25",
                "--merge-iou", "0.70",
                "--keep-inner-frac", "0.535",
                "--merge-mode", "iou_avg",
                "--center-merge-ratio", "0.30",
                "--max-det", "3000",
                "--use-training-tiler",
            ]
            if extra_flags:
                cmd.extend(extra_flags.split())

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            print("✅ Success")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")

    print(f"\n--- Batch Inference Complete ---")
    print(f"Successful: {success_count}/{len(selected_weights)}")
    return success_count == len(selected_weights)

def menu_checklist() -> bool:
    print("=== Workflow checklist ===")
    print("1) Prepare/collect fullsize drawings and labels → pending/")
    print("2) Visualize labels (quick QA)")
    print("3) Approve ALL pending → datasets/rcp_dual_seg_vX")
    print("4) Split images (existing or pending → fullsize)")
    print("5) Panel-centric cropping (creates cropped1k)")
    print("6) Integrate hard negatives (optional) → cropped1k with empty .txt")
    print("7) Smart augmentation (creates augmented1k)")
    print("8) Analyze dataset (spot-check counts, empties)")
    print("9) Configure training → Train")
    print("10) Predict on production & iterate")
    return True


def main() -> None:
    while True:
        print("\n=== RCP-Dual-Seg Hub ===")
        print("0. Top colors → ROI crop (stage outputs)")
        print("1. Visualize labels")
        print("2. Approve ALL pending files")
        print("3. Split images (existing or pending)")
        print("4. Tile fullsize images")
        print("5. Verify labels")
        print("6. Smart augmentation")
        print("7. Analyze dataset")
        print("8. Test GPU memory")
        print("9. Configure Training")
        print("10. Train")
        print("11. Predict (fullsize images or pdfs)")
        print("12. Checklist")
        print("13. Add borders to PDF files")
        print("14. Panel-centric cropping")
        print("15. Exit")

        choice = input("Enter your choice (0-15): ").strip()

        if choice == "0":
            menu_top_colors_roi_pipeline()
        elif choice == "1":
            menu_visualize_labels()
        elif choice == "2":
            menu_approve_all()
        elif choice == "3":
            menu_split_existing()
        elif choice == "4":
            menu_tile_fullsize_images()
        elif choice == "5":
            menu_verify_labels()
        elif choice == "6":
            menu_augmentation()
        elif choice == "7":
            menu_analyze_dataset()
        elif choice == "8":
            menu_test_gpu_memory()
        elif choice == "9":
            menu_configure_training()
        elif choice == "10":
            menu_train()
        elif choice == "11":
            menu_predict()
        elif choice == "12":
            menu_checklist()
        elif choice == "13":
            menu_pdf_borders()
        elif choice == "14":
            menu_panel_cropping()
        elif choice == "15":
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")


def run_debug_cropper_from_args(img_path: str):
    """Handler for --debug_cropper CLI argument."""
    print(f"=== Running Single-Image Panel Crop (Debug Mode) ===")
    print(f"Image: {img_path}")
    # This reuses the logic from the old menu, but runs it directly
    # without further prompts, using sensible defaults.
    import time
    out_dir = Path("runs") / "crop_debug" / f"{Path(img_path).stem}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {out_dir}")

    try:
        cmd = [
            sys.executable, "tools/panel_cropper.py",
            "--single-image", img_path,
            "--out-debug", str(out_dir),
            "--target-size", "1024", # Defaulting to 1024 for debug
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ Debug crops saved under: {out_dir}")
        print("  - images/: cropped images\n  - labels/: YOLO labels\n  - overlays/: visual overlays")
        return True
    except subprocess.CalledProcessError as e:
        print("Error:", e.stdout, e.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCP Dual-Seg Workflow Hub")
    parser.add_argument(
        "--debug_cropper",
        type=str,
        metavar="<image_path>",
        help="Run the panel cropper in debug mode on a single fullsize image."
    )
    args = parser.parse_args()

    if args.debug_cropper:
        run_debug_cropper_from_args(args.debug_cropper)
    else:
        main()


