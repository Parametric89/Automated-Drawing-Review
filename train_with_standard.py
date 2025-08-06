#!/usr/bin/env python3
"""
Train panel detection model using standard YOLOv8s-seg as starting weights
"""
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import yaml

def plot_training_curves(results_dir):
    """
    Plot training and validation curves.
    
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
    try:
        df = pd.read_csv(results_file)
        print(f"✅ Loaded training results: {len(df)} epochs")
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Panel Detection Training Curves (Transfer Learning)', fontsize=16)
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
    if 'val/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
    if 'train/cls_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='green')
    if 'val/cls_loss' in df.columns:
        ax1.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='orange')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: mAP curves
    ax2 = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns:
        ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='blue')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5-0.95', color='red')
    ax2.set_title('mAP Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Precision/Recall
    ax3 = axes[1, 0]
    if 'metrics/precision(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
    if 'metrics/recall(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='red')
    ax3.set_title('Precision/Recall Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Learning rate
    ax4 = axes[1, 1]
    if 'lr/pg0' in df.columns:
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
        try:
            df = pd.read_csv(results_file)
            
            # Get final metrics
            final_row = df.iloc[-1]
            print(f"\n📊 Final Training Metrics:")
            if 'metrics/mAP50(B)' in final_row:
                print(f"   Best mAP@0.5: {final_row['metrics/mAP50(B)']:.3f}")
            if 'metrics/mAP50-95(B)' in final_row:
                print(f"   Best mAP@0.5-0.95: {final_row['metrics/mAP50-95(B)']:.3f}")
            if 'metrics/precision(B)' in final_row:
                print(f"   Final Precision: {final_row['metrics/precision(B)']:.3f}")
            if 'metrics/recall(B)' in final_row:
                print(f"   Final Recall: {final_row['metrics/recall(B)']:.3f}")
            if 'time' in final_row:
                print(f"   Training Time: {final_row['time']:.1f} seconds")
        except Exception as e:
            print(f"⚠️  Could not read final metrics: {e}")
    
    return best_model

def update_dataset_yaml(dataset_type, base_path="datasets/rcp_dual_seg"):
    """Dynamically update the dataset.yaml file to point to the correct data type."""
    config_path = os.path.join(base_path, "dataset.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths to use the specified data_type
    config['path'] = os.path.abspath(base_path)
    config['train'] = os.path.join("images", "train", dataset_type)
    config['val'] = os.path.join("images", "val", dataset_type)
    config['test'] = os.path.join("images", "test", dataset_type)
    
    # Create a temporary config file for this run
    temp_config_path = os.path.join(base_path, f"temp_dataset_{dataset_type}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
        
    print(f"✅ Created temporary dataset config: {temp_config_path}")
    return temp_config_path

def train_with_standard_weights(dataset_type, weights=None):
    """Train using standard YOLOv8s-seg model as starting weights"""
    
    # Determine model to use
    if weights and os.path.exists(weights):
        print(f"🚀 Resuming training from custom weights: {weights}")
        model = YOLO(weights)
    else:
        standard_model_path = "models/standard_yolov8s_seg/best.pt"
        if os.path.exists(standard_model_path):
            print(f"Using standard YOLOv8s-seg model: {standard_model_path}")
            model = YOLO(standard_model_path)
        else:
            print("Standard model not found, downloading YOLOv8s-seg...")
            model = YOLO("yolov8s-seg.pt")
    
    # Update dataset config to point to the right data
    dataset_config = update_dataset_yaml(dataset_type)
    
    print(f"Starting transfer learning training on '{dataset_type}' data...")
    print(f"Dataset config: {dataset_config}")
    
    # Train on your dataset
    results = model.train(
        data=dataset_config,
        epochs=100,
        imgsz=1024,
        batch=8,
        name=f"panel_detection_{dataset_type}",
        project="runs/train",
        patience=20,
        save=True,
        device=0,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=2.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )
    
    print("✅ Training completed!")
    print(f"Best model saved to: {results.save_dir}")
    
    # Plot training curves and analyze results
    if results and hasattr(results, 'save_dir'):
        plot_training_curves(results.save_dir)
        analyze_training_results(results.save_dir)
    
    # Clean up temporary yaml file
    os.remove(dataset_config)
    print(f"🗑️ Removed temporary dataset config: {dataset_config}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for panel detection.")
    parser.add_argument("--dataset-type", type=str, required=True, choices=['cropped1k', 'augmented1k'],
                        help="The type of dataset to use for training ('cropped1k' or 'augmented1k').")
    parser.add_argument("--weights", type=str, default=None,
                        help="Optional path to a pre-trained model to continue training from.")
    args = parser.parse_args()

    train_with_standard_weights(args.dataset_type, args.weights)
