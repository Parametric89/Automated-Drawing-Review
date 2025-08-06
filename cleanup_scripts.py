#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files from the project
"""
import os
import shutil

def cleanup_scripts():
    """Delete unnecessary scripts and files"""
    print("=== Cleaning up unnecessary files ===")
    
    # List of files to delete
    files_to_delete = [
        # Debug/Temporary Scripts
        "debug_dataset.py",
        "test_env.py", 
        "test_gpu_memory.py",
        "test_label_format.py",
        "test_border_only.py",
        "test_border_only.pdf",
        
        # Redundant Analysis Scripts
        "analyze_dataset.py",
        "analyze_dataset_simple.py",
        
        # Roboflow Scripts (Unused)
        "download_roboflow_model.py",
        "download_roboflow_universe.py", 
        "download_roboflow_model_actual.py",
        "explore_roboflow_models.py",
        "verify_roboflow_setup.py",
        "roboflow_inference.py",
        "roboflow_API.md",
        "GET_ROBOFLOW_API_KEY.md",
        "ROBOFLOW_INTEGRATION_GUIDE.md",
        
        # Old Conversion Scripts
        "convert_old_labels.py",
        "convert_to_mixed_format.py",
        "convert_to_segmentation_format.py",
        "convert_with_dimensions.py",
        "poly2mixed.py",
        
        # Verification Scripts (Now in Workflow)
        "verify_training_setup.py",
        "verify_mixed_format.py",
        "verify_label_format.py",
        
        # Environment Setup (One-time)
        "run_with_env.py",
        "activate_env.bat",
        "setup_powershell_profile.ps1",
        "start_with_env.bat",
        
        # Data Management (One-time)
        "clear_datasets.py",
        "organize_labeled_data.py",
        "setup_project_structure.py",
        
        # PDF Processing (Alternative Versions)
        "add_pdf_borders_simple.py",
        "add_pdf_borders.py",
        
        # Other unused files
        "awake.py",
        "tile_images.py",  # Old tiling script
        "train_with_roboflow.py",  # Unused training script
    ]
    
    deleted_count = 0
    not_found_count = 0
    
    print("🗑️  Deleting unnecessary files...")
    print()
    
    for filename in files_to_delete:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"✅ Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {filename}: {e}")
        else:
            print(f"⚠️  Not found: {filename}")
            not_found_count += 1
    
    print()
    print("📊 Cleanup Summary:")
    print(f"   Files deleted: {deleted_count}")
    print(f"   Files not found: {not_found_count}")
    print(f"   Total processed: {len(files_to_delete)}")
    
    # List remaining important files
    print()
    print("✅ Important files that were kept:")
    important_files = [
        "workflow.py",
        "train_with_standard.py", 
        "tile_fullsize_images.py",
        "split_existing_images.py",
        "visualize_yolo_labels.py",
        "clean_empty_tiles.py",
        "train_rcp_dual_seg.py",
        "setup_standard_transfer_learning.py",
        "add_pdf_borders_vector.py",
        "pdf_to_images.py",
        "PROJECT_PLAN.md",
        "TERMINAL_SETUP_GUIDE.md",
        "label_pending_drawings.py",
        "split_by_sheets.py"
    ]
    
    for filename in important_files:
        if os.path.exists(filename):
            print(f"   ✅ {filename}")
        else:
            print(f"   ⚠️  {filename} (not found)")
    
    print()
    print("🎯 Project cleaned up successfully!")
    print("💡 Your workflow is now streamlined and ready for training!")

if __name__ == "__main__":
    cleanup_scripts() 