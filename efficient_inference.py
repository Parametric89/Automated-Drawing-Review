#!/usr/bin/env python3
"""
efficient_inference.py
---------------------
Efficient inference for panel detection on full drawings.

Strategy:
1. Tile 1536x1536 with 25% overlap (far fewer tiles than before)
2. Run detection on each tile
3. Global NMS to merge duplicates
4. Return merged results

This is much more efficient than dense tiling.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO
import torch


def tile_image_for_inference(image, tile_size=1536, overlap=0.25):
    """Tile image for efficient inference"""
    h, w = image.shape[:2]
    step = int(tile_size * (1 - overlap))
    
    tiles = []
    tile_positions = []
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Extract tile
            tile = image[y:y+tile_size, x:x+tile_size]
            
            # Skip if tile is too small
            if tile.shape[0] < tile_size//2 or tile.shape[1] < tile_size//2:
                continue
            
            # Pad if needed
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            tiles.append(tile)
            tile_positions.append((x, y))
    
    return tiles, tile_positions


def run_detection_on_tiles(model, tiles, conf_threshold=0.25):
    """Run detection on all tiles"""
    all_results = []
    
    for i, tile in enumerate(tiles):
        # Run detection
        results = model.predict(tile, conf=conf_threshold, verbose=False)
        
        # Store results with tile index
        for result in results:
            result.tile_index = i
            all_results.append(result)
    
    return all_results


def transform_detections_to_global_coords(results, tile_positions, original_size):
    """Transform tile detections back to global coordinates"""
    global_detections = []
    
    for result in results:
        tile_x, tile_y = tile_positions[result.tile_index]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # Transform masks if available
            masks = None
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                # Transform box coordinates
                x1, y1, x2, y2 = box
                global_x1 = x1 + tile_x
                global_y1 = y1 + tile_y
                global_x2 = x2 + tile_x
                global_y2 = y2 + tile_y
                
                # Clamp to original image bounds
                h, w = original_size
                global_x1 = max(0, min(w, global_x1))
                global_y1 = max(0, min(h, global_y1))
                global_x2 = max(0, min(w, global_x2))
                global_y2 = max(0, min(h, global_y2))
                
                detection = {
                    'bbox': [global_x1, global_y1, global_x2, global_y2],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'tile_index': result.tile_index
                }
                
                # Transform mask if available
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    # Transform mask coordinates (simplified)
                    # In practice, you'd need more sophisticated mask transformation
                    detection['mask'] = mask.tolist()
                
                global_detections.append(detection)
    
    return global_detections


def apply_global_nms(detections, iou_threshold=0.5):
    """Apply global NMS to merge duplicate detections"""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Apply NMS
    kept_detections = []
    
    for detection in detections:
        should_keep = True
        
        for kept in kept_detections:
            # Calculate IoU
            iou = calculate_iou(detection['bbox'], kept['bbox'])
            
            if iou > iou_threshold and detection['class_id'] == kept['class_id']:
                should_keep = False
                break
        
        if should_keep:
            kept_detections.append(detection)
    
    return kept_detections


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def visualize_results(image, detections, output_path=None):
    """Visualize detection results"""
    vis_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        conf = detection['confidence']
        cls_id = detection['class_id']
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)  # Green for panels, red for tags
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Panel {cls_id}: {conf:.2f}"
        cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


def run_efficient_inference(model_path, image_path, output_dir="results", 
                          tile_size=1536, overlap=0.25, conf_threshold=0.25):
    """Run efficient inference on a full drawing"""
    print(f"=== Efficient Inference ===")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Tile size: {tile_size}x{tile_size}, Overlap: {overlap}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Tile image
    print("Tiling image...")
    tiles, tile_positions = tile_image_for_inference(image, tile_size, overlap)
    print(f"Created {len(tiles)} tiles")
    
    # Run detection on tiles
    print("Running detection on tiles...")
    tile_results = run_detection_on_tiles(model, tiles, conf_threshold)
    print(f"Found {len(tile_results)} detection results")
    
    # Transform to global coordinates
    print("Transforming to global coordinates...")
    global_detections = transform_detections_to_global_coords(tile_results, tile_positions, (h, w))
    print(f"Transformed {len(global_detections)} detections")
    
    # Apply global NMS
    print("Applying global NMS...")
    final_detections = apply_global_nms(global_detections)
    print(f"Final detections after NMS: {len(final_detections)}")
    
    # Visualize results
    print("Visualizing results...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{Path(image_path).stem}_inference.jpg")
    vis_image = visualize_results(image, final_detections, output_path)
    
    # Save results
    results_path = os.path.join(output_dir, f"{Path(image_path).stem}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'image_path': image_path,
            'image_size': [w, h],
            'tile_size': tile_size,
            'overlap': overlap,
            'num_tiles': len(tiles),
            'num_detections': len(final_detections),
            'detections': final_detections,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"✅ Inference complete!")
    print(f"📊 Results: {len(final_detections)} panels detected")
    print(f"💾 Saved to: {output_path}")
    print(f"📄 Results: {results_path}")
    
    return final_detections


def batch_inference(model_path, image_dir, output_dir="results"):
    """Run efficient inference on multiple images"""
    print(f"=== Batch Efficient Inference ===")
    print(f"Model: {model_path}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    for ext in image_extensions:
        images.extend(Path(image_dir).glob(f"*{ext}"))
        images.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(images)} images")
    
    # Process each image
    all_results = {}
    
    for image_path in images:
        print(f"\nProcessing {image_path.name}...")
        try:
            detections = run_efficient_inference(
                model_path, str(image_path), output_dir
            )
            all_results[image_path.name] = detections
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            all_results[image_path.name] = []
    
    # Save batch results
    batch_results_path = os.path.join(output_dir, "batch_results.json")
    with open(batch_results_path, 'w') as f:
        json.dump({
            'model_path': model_path,
            'image_directory': image_dir,
            'total_images': len(images),
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Batch inference complete!")
    print(f"📊 Processed {len(images)} images")
    print(f"📄 Batch results: {batch_results_path}")
    
    return all_results


def main():
    """Main function for efficient inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Efficient inference for panel detection")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--image", help="Path to single image")
    parser.add_argument("--image-dir", help="Directory with images for batch processing")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=1536, help="Tile size for inference")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if args.image:
        run_efficient_inference(
            args.model, args.image, args.output_dir,
            args.tile_size, args.overlap, args.conf
        )
    elif args.image_dir:
        batch_inference(args.model, args.image_dir, args.output_dir)
    else:
        print("Error: Must specify either --image or --image-dir")
        return 1
    
    return 0


if __name__ == "__main__":
    main() 