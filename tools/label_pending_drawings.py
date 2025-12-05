"""
label_pending_drawings.py
------------------------
Interactive labeling workflow for pending drawings.
Creates labels for RCP-Dual-Seg model (panel + panel_tag).
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import shutil
from datetime import datetime


class DrawingLabeler:
    def __init__(self, pending_dir="pending/images", output_dir="datasets/rcp_dual_seg"):
        self.pending_dir = pending_dir
        self.output_dir = output_dir
        self.current_image = None
        self.current_image_path = None
        self.panel_points = []
        self.tag_points = []
        self.image_name = None
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
        # Load progress
        self.progress_file = "labeling_progress.json"
        self.load_progress()
    
    def load_progress(self):
        """Load labeling progress from file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "completed": [],
                "current_image": None,
                "total_images": 0
            }
    
    def save_progress(self):
        """Save labeling progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_pending_images(self):
        """Get list of images that need labeling."""
        if not os.path.exists(self.pending_dir):
            print(f"❌ Pending directory not found: {self.pending_dir}")
            return []
        
        image_files = [f for f in os.listdir(self.pending_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Filter out already completed images
        pending = [f for f in image_files if f not in self.progress["completed"]]
        
        self.progress["total_images"] = len(image_files)
        return pending
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing polygons and rectangles."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == "panel":
                self.panel_points.append([x, y])
                # Draw point
                cv2.circle(self.current_image, (x, y), 3, (0, 255, 0), -1)
                if len(self.panel_points) > 1:
                    # Draw line to previous point
                    cv2.line(self.current_image, tuple(self.panel_points[-2]), 
                            (x, y), (0, 255, 0), 2)
                cv2.imshow('Labeling Tool', self.current_image)
            
            elif self.current_mode == "tag":
                if len(self.tag_points) < 2:
                    self.tag_points.append([x, y])
                    cv2.circle(self.current_image, (x, y), 3, (255, 0, 0), -1)
                    if len(self.tag_points) == 2:
                        # Draw rectangle
                        pt1 = tuple(self.tag_points[0])
                        pt2 = tuple(self.tag_points[1])
                        cv2.rectangle(self.current_image, pt1, pt2, (255, 0, 0), 2)
                    cv2.imshow('Labeling Tool', self.current_image)
    
    def label_image(self, image_path):
        """Label a single image with panel polygon and tag bounding box."""
        self.current_image_path = image_path
        self.image_name = os.path.basename(image_path)
        self.panel_points = []
        self.tag_points = []
        
        # Load image
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        # Create window and set mouse callback
        cv2.namedWindow('Labeling Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Labeling Tool', self.mouse_callback)
        
        # Display instructions
        instructions = self.current_image.copy()
        cv2.putText(instructions, "Press 'p' to start panel polygon", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(instructions, "Press 't' to start tag bounding box", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(instructions, "Press 'r' to reset", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(instructions, "Press 's' to save", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(instructions, "Press 'q' to quit", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Labeling Tool', instructions)
        
        self.current_mode = None
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):
                # Start panel polygon mode
                self.current_mode = "panel"
                self.panel_points = []
                print("�� Panel polygon mode - click to add points, 'd' to finish")
                cv2.imshow('Labeling Tool', self.current_image)
            
            elif key == ord('t'):
                # Start tag bounding box mode
                self.current_mode = "tag"
                self.tag_points = []
                print("��️  Tag bounding box mode - click two corners")
                cv2.imshow('Labeling Tool', self.current_image)
            
            elif key == ord('d') and self.current_mode == "panel":
                # Finish panel polygon
                if len(self.panel_points) >= 3:
                    # Close polygon
                    cv2.line(self.current_image, tuple(self.panel_points[-1]), 
                            tuple(self.panel_points[0]), (0, 255, 0), 2)
                    cv2.imshow('Labeling Tool', self.current_image)
                    print(f"✅ Panel polygon completed with {len(self.panel_points)} points")
                else:
                    print("❌ Need at least 3 points for polygon")
            
            elif key == ord('r'):
                # Reset current mode
                self.panel_points = []
                self.tag_points = []
                self.current_image = cv2.imread(image_path)
                cv2.imshow('Labeling Tool', self.current_image)
                print("🔄 Reset")
            
            elif key == ord('s'):
                # Save labels
                if self.save_labels():
                    print("✅ Labels saved successfully!")
                    break
                else:
                    print("❌ Failed to save labels")
            
            elif key == ord('q'):
                # Quit without saving
                print("❌ Quitting without saving")
                break
        
        cv2.destroyAllWindows()
        return True
    
    def save_labels(self):
        """Save YOLO-Seg labels for the current image."""
        if len(self.panel_points) < 3:
            print("❌ Need panel polygon with at least 3 points")
            return False
        
        if len(self.tag_points) < 2:
            print("❌ Need tag bounding box with 2 points")
            return False
        
        # Get image dimensions
        h, w = self.current_image.shape[:2]
        
        # Create YOLO-Seg label lines
        labels = []
        
        # Class 0: Panel polygon
        panel_x = [p[0] for p in self.panel_points]
        panel_y = [p[1] for p in self.panel_points]
        
        # Calculate bounding box
        xmin, xmax = min(panel_x), max(panel_x)
        ymin, ymax = min(panel_y), max(panel_y)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        
        # Normalize polygon points
        poly_norm = []
        for x, y in zip(panel_x, panel_y):
            poly_norm.extend([x / w, y / h])
        
        # Panel label line: "0 <cx> <cy> <width> <height> <x1> <y1> <x2> <y2> ..."
        panel_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join([f"{p:.6f}" for p in poly_norm])
        labels.append(panel_line)
        
        # Class 1: Tag bounding box
        tag_x = [p[0] for p in self.tag_points]
        tag_y = [p[1] for p in self.tag_points]
        
        # Calculate tag bounding box
        tx_min, tx_max = min(tag_x), max(tag_x)
        ty_min, ty_max = min(tag_y), max(tag_y)
        tcx = (tx_min + tx_max) / 2 / w
        tcy = (ty_min + ty_max) / 2 / h
        tbw = (tx_max - tx_min) / w
        tbh = (ty_max - ty_min) / h
        
        # Tag label line: "1 <cx> <cy> <width> <height>"
        tag_line = f"1 {tcx:.6f} {tcy:.6f} {tbw:.6f} {tbh:.6f}"
        labels.append(tag_line)
        
        # Save label file
        label_name = self.image_name.rsplit('.', 1)[0] + '.txt'
        
        # Determine split (train/val/test) - for now, put all in train
        # TODO: Implement proper split logic
        split = 'train'
        
        label_path = os.path.join(self.output_dir, 'labels', split, label_name)
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        # Copy image to appropriate split directory
        image_dst = os.path.join(self.output_dir, 'images', split, self.image_name)
        shutil.copy2(self.current_image_path, image_dst)
        
        # Update progress
        self.progress["completed"].append(self.image_name)
        self.save_progress()
        
        return True
    
    def run_labeling_session(self):
        """Run the complete labeling session."""
        print("=== RCP-Dual-Seg Labeling Tool ===")
        print("Classes: 0=panel (polygon), 1=panel_tag (bbox)")
        print()
        
        pending_images = self.get_pending_images()
        
        if not pending_images:
            print("✅ No pending images to label!")
            return
        
        print(f"�� Found {len(pending_images)} images to label")
        print(f"�� Already completed: {len(self.progress['completed'])}")
        print()
        
        for i, image_file in enumerate(pending_images):
            print(f"\n--- Image {i+1}/{len(pending_images)}: {image_file} ---")
            
            image_path = os.path.join(self.pending_dir, image_file)
            
            # Label the image
            success = self.label_image(image_path)
            
            if not success:
                print(f"❌ Failed to label {image_file}")
                continue
            
            # Ask if user wants to continue
            if i < len(pending_images) - 1:
                response = input("\nContinue to next image? (y/n): ").strip().lower()
                if response != 'y':
                    print("Labeling session paused.")
                    break
        
        print(f"\n🎉 Labeling session complete!")
        print(f"�� Total completed: {len(self.progress['completed'])}")
        print(f"📊 Remaining: {len(pending_images) - len([f for f in pending_images if f in self.progress['completed']])}")


def main():
    """Main function."""
    labeler = DrawingLabeler()
    labeler.run_labeling_session()


if __name__ == "__main__":
    main()