#!/usr/bin/env python3
"""
Test GPU memory usage with 1024x1024 images and batch size 8
"""
import torch
import numpy as np
from ultralytics import YOLO
import os

def test_gpu_memory():
    """Test if GPU can handle 1024x1024 images with batch size 8"""
    print("=== GPU Memory Test (1024x1024) ===")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check current memory usage
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    try:
        # Load model
        print("Loading YOLOv8s-seg model...")
        model = YOLO("yolov8s-seg.pt")
        
        # Create dummy data (8x1024x1024 images, batch size 8)
        print("Creating dummy batch (8x1024x1024 images)...")
        dummy_batch = torch.randn(8, 3, 1024, 1024).cuda()
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            results = model(dummy_batch)
        
        # Check memory usage
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        current_memory = torch.cuda.memory_allocated() / 1024**2
        
        print(f"✅ Test successful!")
        print(f"Peak memory usage: {peak_memory:.1f} MB")
        print(f"Current memory usage: {current_memory:.1f} MB")
        print(f"Memory increase: {peak_memory - initial_memory:.1f} MB")
        
        # Cleanup
        del dummy_batch, results, model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gpu_memory()
    if success:
        print("\n✅ GPU can handle 1024x1024 images with batch size 8")
        print("Ready to start training!")
    else:
        print("\n❌ GPU memory test failed")
        print("Consider reducing batch size to 4 or image size to 512") 