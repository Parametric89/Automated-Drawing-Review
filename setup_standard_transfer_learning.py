#!/usr/bin/env python3
"""
Set up transfer learning with standard YOLOv8s-seg model
"""
import os
from pathlib import Path
from ultralytics import YOLO

def setup_standard_transfer_learning():
    """Set up transfer learning with standard YOLOv8s-seg model"""
    
    print("🤖 Setting up Standard YOLOv8s-seg Transfer Learning")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path("models/standard_yolov8s_seg")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading standard YOLOv8s-seg model...")
    
    try:
        # Download the standard YOLOv8s-seg model
        model = YOLO("yolov8s-seg.pt")
        
        # Save it to our models directory
        model_path = models_dir / "best.pt"
        model.save(str(model_path))
        
        print(f"✅ Standard YOLOv8s-seg model downloaded: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

def setup_dataset_config():
    """Set up dataset configuration for transfer learning"""
    
    config_content = """# Dataset configuration for panel detection transfer learning
path: ../datasets/rcp_dual_seg
train: images/train
val: images/val
test: images/test
nc: 2
names: ['panel', 'panel_tag']
"""
    
    config_path = Path("datasets/rcp_dual_seg/dataset.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created dataset config: {config_path}")
    return config_path

def create_training_script():
    """Create training script for standard YOLOv8s-seg transfer learning"""
    
    training_script = '''#!/usr/bin/env python3
"""
Train panel detection model using standard YOLOv8s-seg as starting weights
"""
from ultralytics import YOLO
import os
from pathlib import Path

def train_with_standard_weights():
    """Train using standard YOLOv8s-seg model as starting weights"""
    
    # Check if standard model exists
    standard_model_path = "models/standard_yolov8s_seg/best.pt"
    
    if os.path.exists(standard_model_path):
        print(f"Using standard YOLOv8s-seg model: {standard_model_path}")
        model = YOLO(standard_model_path)
    else:
        print("Standard model not found, downloading YOLOv8s-seg...")
        model = YOLO("yolov8s-seg.pt")
    
    # Check dataset config
    dataset_config = "datasets/rcp_dual_seg/dataset.yaml"
    if not os.path.exists(dataset_config):
        print(f"Error: Dataset config not found: {dataset_config}")
        return None
    
    print("Starting transfer learning training...")
    print(f"Dataset config: {dataset_config}")
    
    # Train on your dataset
    results = model.train(
        data=dataset_config,
        epochs=100,
        imgsz=2048,
        batch=8,
        name="panel_detection_standard_transfer",
        project="runs/train",
        patience=20,
        save=True,
        device=0,  # Use GPU if available
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
    
    return results

if __name__ == "__main__":
    train_with_standard_weights()
'''
    
    with open("train_with_standard.py", 'w') as f:
        f.write(training_script)
    
    print("✅ Created training script: train_with_standard.py")

def main():
    """Main function"""
    
    # Download standard model
    model_path = setup_standard_transfer_learning()
    
    if model_path:
        print(f"✅ Model downloaded: {model_path}")
        
        # Set up dataset config
        config_path = setup_dataset_config()
        
        # Create training script
        create_training_script()
        
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Verify your dataset is ready in datasets/rcp_dual_seg/")
        print("2. Run training: python train_with_standard.py")
        print("3. Monitor training in runs/train/panel_detection_standard_transfer/")
        print("\nBenefits of this approach:")
        print("- Uses pre-trained YOLOv8s-seg weights")
        print("- Segmentation capabilities for precise panel boundaries")
        print("- Faster convergence than training from scratch")
        print("- Proven architecture for object detection")
        
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 