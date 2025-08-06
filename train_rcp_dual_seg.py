"""
train_rcp_dual_seg.py
--------------------
Train RCP-Dual-Seg model with Jtrain and Jcv plotting.
"""

import os
import sys
import yaml
from pathlib import Path
import subprocess
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import json


def check_yolo_installation():
    """Check if YOLO is installed and install if needed."""
    try:
        import ultralytics
        print("✅ YOLO (ultralytics) is installed")
        return True
    except ImportError:
        print("❌ YOLO (ultralytics) not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("✅ YOLO installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install YOLO")
            return False


def validate_dataset(dataset_path):
    """Validate the RCP-Dual-Seg dataset structure."""
    print(f"\n=== RCP-Dual-Seg Dataset Validation ===")
    
    # Check if dataset.yaml exists
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    if not os.path.exists(yaml_path):
        print(f"❌ dataset.yaml not found at: {yaml_path}")
        return False
    
    # Load and validate dataset.yaml
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Dataset config loaded")
        print(f"   Classes: {config.get('nc', 'N/A')}")
        print(f"   Class names: {config.get('names', 'N/A')}")
        
        # Check for 2 classes: panel and panel_tag
        if config.get('nc') != 2:
            print(f"   ❌ Expected 2 classes, found {config.get('nc')}")
            return False
        
        expected_names = {0: 'panel', 1: 'panel_tag'}
        if config.get('names') != expected_names:
            print(f"   ❌ Expected class names: {expected_names}")
            print(f"   Found: {config.get('names')}")
            return False
        
        print(f"   ✅ 2-class structure confirmed")
        print(f"   📋 Label format: Class 0 (panel) = polygon mask, Class 1 (panel_tag) = bbox")
        
        # Check paths
        base_path = config.get('path', '')
        train_img_path = os.path.join(base_path, config.get('train', ''))
        val_img_path = os.path.join(base_path, config.get('val', ''))
        test_img_path = os.path.join(base_path, config.get('test', ''))
        
        # Construct label paths (labels are in separate directories)
        train_label_path = train_img_path.replace('images/', 'labels/')
        val_label_path = val_img_path.replace('images/', 'labels/')
        test_label_path = test_img_path.replace('images/', 'labels/')
        
        print(f"   Train images: {train_img_path}")
        print(f"   Train labels: {train_label_path}")
        print(f"   Val images: {val_img_path}")
        print(f"   Val labels: {val_label_path}")
        print(f"   Test images: {test_img_path}")
        print(f"   Test labels: {test_label_path}")
        
        # Check if directories exist and count files
        total_images = 0
        total_labels = 0
        
        for split_name, img_path, label_path in [
            ('train', train_img_path, train_label_path), 
            ('val', val_img_path, val_label_path), 
            ('test', test_img_path, test_label_path)
        ]:
            if os.path.exists(img_path):
                images = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"   ✅ {split_name.capitalize()} images: {len(images)}")
                total_images += len(images)
            else:
                print(f"   ❌ {split_name.capitalize()} image directory not found: {img_path}")
                return False
            
            if os.path.exists(label_path):
                labels = [f for f in os.listdir(label_path) if f.endswith('.txt')]
                print(f"   ✅ {split_name.capitalize()} labels: {len(labels)}")
                total_labels += len(labels)
                
                # Check for matching image-label pairs
                image_stems = {Path(f).stem for f in images}
                label_stems = {Path(f).stem for f in labels}
                missing_labels = image_stems - label_stems
                missing_images = label_stems - image_stems
                
                if missing_labels:
                    print(f"      ⚠️  {len(missing_labels)} images without labels")
                if missing_images:
                    print(f"      ⚠️  {len(missing_images)} labels without images")
                if not missing_labels and not missing_images:
                    print(f"      ✅ All image-label pairs matched")
                
                # Check label format (sample a few files)
                if labels:
                    sample_label = os.path.join(label_path, labels[0])
                    try:
                        with open(sample_label, 'r') as f:
                            first_line = f.readline().strip()
                            if first_line:
                                parts = first_line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if class_id == 0 and len(parts) > 5:
                                        print(f"      ✅ Class 0 (panel) has polygon coordinates")
                                    elif class_id == 1 and len(parts) == 5:
                                        print(f"      ✅ Class 1 (panel_tag) has bbox format")
                                    else:
                                        print(f"      ⚠️  Unexpected label format in sample")
                    except Exception as e:
                        print(f"      ⚠️  Could not read sample label: {e}")
            else:
                print(f"   ❌ {split_name.capitalize()} label directory not found: {label_path}")
                return False
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total images: {total_images}")
        print(f"   Total labels: {total_labels}")
        print(f"   Average images per split: {total_images/3:.1f}")
        
        # Check if this looks like a tiled dataset
        if total_images > 1000:  # Tiled datasets typically have many more images
            print(f"   ✅ This appears to be a tiled dataset (many small images)")
            print(f"   💡 Recommended image size: 2048 (tile size)")
        else:
            print(f"   ⚠️  This appears to be a fullsize dataset")
            print(f"   💡 Recommended image size: 2048 or larger")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset config: {e}")
        return False


def train_rcp_dual_seg_model(dataset_path, model_size="n", epochs=150, batch_size=2, imgsz=2048):
    """
    Train RCP-Dual-Seg model on the dataset.
    Model: 2-class segmentation
    - Class 0: Panel polygon (mask) 
    - Class 1: Panel tag bounding box
    
    Args:
        dataset_path: Path to dataset.yaml
        model_size: YOLO model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size for training (reduced for memory)
        imgsz: Input image size (reduced for memory)
    """
    print(f"\n=== Training RCP-Dual-Seg Model ===")
    print(f"Dataset: {dataset_path}")
    print(f"Model: YOLOv8{model_size}-seg (segmentation)")
    print(f"Classes: panel (0) + panel_tag (1)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size} (reduced for 2048x2048 images)")
    print(f"Image size: {imgsz} (reduced for memory)")
    print(f"Format: Class 0 (panel) = polygon mask, Class 1 (panel_tag) = bbox")
    print(f"Note: Using conservative memory settings for segmentation")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/train/rcp_dual_seg_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # YOLO training command for segmentation with memory optimization
    cmd = [
        "yolo", "train",
        f"model=yolov8{model_size}-seg.pt",  # Segmentation model
        f"data={dataset_path}",
        f"epochs={epochs}",
        f"batch={batch_size}",  # Reduced batch size
        f"imgsz={imgsz}",  # Reduced image size
        f"project=runs/train",
        f"name=rcp_dual_seg_{timestamp}",
        "patience=20",  # Early stopping
        "save=True",
        "save_period=10",  # Save every 10 epochs
        "cache=False",  # Disable cache to save memory
        "device=0",  # Use GPU (device 0)
        "workers=0",  # Disable multiprocessing for Windows compatibility
        "amp=True",  # Use mixed precision to save memory
        "overlap_mask=True",  # Better mask quality
        "mask_ratio=4"  # Reduce mask resolution to save memory
    ]
    
    print(f"\nTraining command:")
    print(f" {' '.join(cmd)}")
    print(f"\nStarting RCP-Dual-Seg training...")
    
    try:
        # Run training with live output
        result = subprocess.run(cmd, check=True)
        print("\n✅ RCP-Dual-Seg training completed successfully!")
        print(f"Results saved to: {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return None


def plot_training_curves(results_dir):
    """
    Plot Jtrain and Jcv (training and validation curves).
    
    Args:
        results_dir: Path to training results directory
    """
    print(f"\n=== Plotting Training Curves ===")
    
    # Find results.csv file
    results_file = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    # Read training results
    import pandas as pd
    try:
        df = pd.read_csv(results_file)
        print(f"✅ Loaded training results: {len(df)} epochs")
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RCP-Dual-Seg Training Curves', fontsize=16)
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
    ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
    ax1.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='green')
    ax1.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='orange')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: mAP curves
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='blue')
    ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5-0.95', color='red')
    ax2.set_title('mAP Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Precision/Recall
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
    ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='red')
    ax3.set_title('Precision/Recall Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Learning rate
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='purple')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.legend()
    ax4.grid(True)
    
    # Save plot
    plot_file = os.path.join(results_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return plot_file


def analyze_training_results(results_dir):
    """Analyze and summarize training results."""
    print(f"\n=== Training Results Analysis ===")
    
    # Find best model
    weights_dir = os.path.join(results_dir, "weights")
    best_model = os.path.join(weights_dir, "best.pt")
    last_model = os.path.join(weights_dir, "last.pt")
    
    if os.path.exists(best_model):
        print(f"✅ Best model: {best_model}")
    if os.path.exists(last_model):
        print(f"✅ Last model: {last_model}")
    
    # Read final metrics
    results_file = os.path.join(results_dir, "results.csv")
    if os.path.exists(results_file):
        import pandas as pd
        df = pd.read_csv(results_file)
        
        # Get final metrics
        final_row = df.iloc[-1]
        print(f"\n📊 Final Training Metrics:")
        print(f"   Best mAP@0.5: {final_row['metrics/mAP50(B)']:.3f}")
        print(f"   Best mAP@0.5-0.95: {final_row['metrics/mAP50-95(B)']:.3f}")
        print(f"   Final Precision: {final_row['metrics/precision(B)']:.3f}")
        print(f"   Final Recall: {final_row['metrics/recall(B)']:.3f}")
        print(f"   Training Time: {final_row['time']:.1f} seconds")
    
    return best_model


def main():
    """Main function with interactive menu."""
    print("=== RCP-Dual-Seg Model Training ===")
    print("2-class segmentation: panel + panel_tag")
    print("Optimized for tiled dataset (2048x2048 tiles)")
    print("Format: Class 0 (panel) = polygon mask, Class 1 (panel_tag) = bbox")
    print()
    
    # Check YOLO installation
    if not check_yolo_installation():
        print("Please install YOLO manually: pip install ultralytics")
        return
    
    # Dataset path
    dataset_path = "datasets/rcp_dual_seg/dataset.yaml"
    
    while True:
        print("\n" + "="*50)
        print("RCP-Dual-Seg Training Menu:")
        print("1. Validate dataset")
        print("2. Train new model")
        print("3. Plot training curves")
        print("4. Analyze results")
        print("5. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            # Validate dataset
            validate_dataset("datasets/rcp_dual_seg")
            
        elif choice == "2":
            # Train new model
            print("\n--- RCP-Dual-Seg Training Configuration ---")
            print("💡 For tiled dataset (2048x2048 tiles), use image size 2048")
            print("💡 For memory optimization, use smaller batch size (8)")
            print("💡 Using segmentation model (polygon format)")
            print("⚠️  Note: Segmentation models require more GPU memory")
            print()
            
            # Model size
            model_size = input("Model size (n/s/m/l/x) [n]: ").strip() or "n"
            
            # Epochs
            try:
                epochs = int(input("Number of epochs [150]: ").strip() or "150")
            except ValueError:
                epochs = 150
            
            # Batch size (reduced for memory)
            try:
                batch_size = int(input("Batch size [8]: ").strip() or "8")
            except ValueError:
                batch_size = 8
            
            # Image size (reduced for memory)
            try:
                imgsz = int(input("Image size [2048]: ").strip() or "2048")
            except ValueError:
                imgsz = 2048
            
            # Validate dataset first
            if validate_dataset("datasets/rcp_dual_seg"):
                # Train model
                results_dir = train_rcp_dual_seg_model(dataset_path, model_size, epochs, batch_size, imgsz)
                if results_dir:
                    print(f"\n✅ RCP-Dual-Seg training completed!")
                    print(f"📁 Results saved to: {results_dir}")
                    
                    # Plot training curves
                    plot_training_curves(results_dir)
                    
                    # Analyze results
                    analyze_training_results(results_dir)
            
        elif choice == "3":
            # Plot training curves
            results_dir = input("Enter results directory path: ").strip()
            if results_dir and os.path.exists(results_dir):
                plot_training_curves(results_dir)
            else:
                print("❌ Invalid results directory path")
            
        elif choice == "4":
            # Analyze results
            results_dir = input("Enter results directory path: ").strip()
            if results_dir and os.path.exists(results_dir):
                analyze_training_results(results_dir)
            else:
                print("❌ Invalid results directory path")
            
        elif choice == "5":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()