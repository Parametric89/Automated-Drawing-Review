"""
visualize_yolo_labels.py
------------------------
Visualize YOLO-Seg labels by drawing bounding boxes and polygons on an image.

Usage:
    python visualize_yolo_labels.py --image path/to/image.jpg --labels-file labels.txt --output visualized_image.jpg

The labels file should contain YOLO-Seg format lines:
    class_id center_x center_y width height x1 y1 x2 y2 x3 y3 ...
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def parse_yolo_labels(file_path):
    """Parse YOLO-Seg labels from a file."""
    labels = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 9:  # Need at least: class_id, bbox(4), polygon(4)
                        print(f"Warning: Line {line_num} has insufficient data: {line}")
                        continue
                    
                    # Parse class ID and bounding box
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Parse polygon coordinates
                    polygon_coords = []
                    for i in range(5, len(parts), 2):
                        if i + 1 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i + 1])
                            polygon_coords.append((x, y))
                    
                    if len(polygon_coords) < 3:
                        print(f"Warning: Line {line_num} has insufficient polygon points")
                        continue
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': (center_x, center_y, width, height),
                        'polygon': polygon_coords,
                        'line_number': line_num
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return labels


def denormalize_coordinates(x_norm, y_norm, img_width, img_height):
    """Convert normalized coordinates back to pixel coordinates."""
    x_pixel = int(x_norm * img_width)
    y_pixel = int(y_norm * img_height)
    return x_pixel, y_pixel


def draw_yolo_labels(image_path, labels, output_path=None, 
                    bbox_thickness=3, polygon_thickness=2, show_preview=False):
    """
    Draw YOLO-Seg labels on an image with class-specific colors.
    
    Args:
        image_path (str): Path to input image
        labels (list): List of label dictionaries
        output_path (str): Path for output image (optional)
        bbox_thickness (int): Line thickness for bounding boxes
        polygon_thickness (int): Line thickness for polygons
        show_preview (bool): Whether to show preview window
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Read the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return False
    except Exception as e:
        print(f"Error reading image: {e}")
        return False
    
    # Get image dimensions
    height, width = image.shape[:2]
    print(f"Image dimensions: {width} x {height} pixels")
    print(f"Drawing {len(labels)} YOLO labels:")
    
    # Define colors for different elements (BGR format)
    bbox_color_class0 = (0, 255, 0)      # Green bounding boxes for Class 0 (panels)
    bbox_color_class1 = (0, 0, 255)      # Red bounding boxes for Class 1 (panel tags)
    polygon_color = (255, 0, 255)        # Magenta polygons for panel shapes
    text_color = (0, 0, 0)               # Black text for all classes
    
    # Class names for better labeling
    class_names = {
        0: "Panel",
        1: "Panel Tag",
        2: "Class 2",
        3: "Class 3",
        4: "Class 4",
        5: "Class 5",
        6: "Class 6",
        7: "Class 7",
        8: "Class 8",
        9: "Class 9",
    }
    
    # Prepare all drawing elements by layer
    green_bboxes = []      # Layer 1: All green bounding boxes (Class 0)
    magenta_polygons = []  # Layer 2: All magenta polygons (Class 0)
    red_bboxes = []        # Layer 3: All red bounding boxes (Class 1)
    black_texts = []       # Layer 4: All black text labels
    blue_vertices = []     # Layer 5: All blue vertices (Class 0 polygons)
    
    for i, label in enumerate(labels):
        class_id = label['class_id']
        bbox = label['bbox']
        polygon = label['polygon']
        
        class_name = class_names.get(class_id, f"Class {class_id}")
        
        # Denormalize bounding box
        center_x_norm, center_y_norm, width_norm, height_norm = bbox
        center_x, center_y = denormalize_coordinates(center_x_norm, center_y_norm, width, height)
        bbox_width = int(width_norm * width)
        bbox_height = int(height_norm * height)
        
        # Calculate bounding box corners
        x1 = center_x - bbox_width // 2
        y1 = center_y - bbox_height // 2
        x2 = center_x + bbox_width // 2
        y2 = center_y + bbox_height // 2
        
        # Denormalize polygon coordinates
        polygon_pixels = []
        for x_norm, y_norm in polygon:
            x_pixel, y_pixel = denormalize_coordinates(x_norm, y_norm, width, height)
            polygon_pixels.append((x_pixel, y_pixel))
        
        # Organize drawing elements by layer
        if class_id == 0:
            # Class 0 (Panel): Green bbox, magenta polygon, blue vertices
            green_bboxes.append(((x1, y1), (x2, y2)))
            if len(polygon_pixels) >= 3:
                magenta_polygons.append(polygon_pixels)
                blue_vertices.append(polygon_pixels)
        else:
            # Class 1 (Panel Tag): Red bbox only
            red_bboxes.append(((x1, y1), (x2, y2)))
        
        # All classes get black text
        black_texts.append((f"{class_name} ({class_id})", (x1, y1-10)))
        
        print(f"  Label {i+1}: {class_name} (Class {class_id}), bbox({x1},{y1},{x2},{y2}), {len(polygon_pixels)} polygon points")
    
    # Draw in layers (back to front)
    
    # Layer 1: All green bounding boxes (Class 0) - BEHIND
    for (x1, y1), (x2, y2) in green_bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color_class0, bbox_thickness)
    
    # Layer 2: All magenta polygons (Class 0) - ON TOP OF GREEN (dashed)
    for polygon_pixels in magenta_polygons:
        polygon_array = np.array(polygon_pixels, dtype=np.int32)
        # Draw dashed polygon by drawing line segments with gaps
        for i in range(len(polygon_pixels)):
            start_point = polygon_pixels[i]
            end_point = polygon_pixels[(i + 1) % len(polygon_pixels)]
            
            # Calculate line length and create dashed effect
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = int(np.sqrt(dx*dx + dy*dy))
            
            if length > 0:
                # Create dashed line segments
                dash_length = 10
                gap_length = 25  # 5x larger gap
                total_dash = dash_length + gap_length
                
                for j in range(0, length, total_dash):
                    # Calculate start and end of this dash segment
                    t_start = j / length
                    t_end = min((j + dash_length) / length, 1.0)
                    
                    if t_end > t_start:
                        dash_start_x = int(start_point[0] + dx * t_start)
                        dash_start_y = int(start_point[1] + dy * t_start)
                        dash_end_x = int(start_point[0] + dx * t_end)
                        dash_end_y = int(start_point[1] + dy * t_end)
                        
                        cv2.line(image, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), 
                                polygon_color, polygon_thickness)
    
    # Layer 3: All red bounding boxes (Class 1) - ON TOP OF MAGENTA
    for (x1, y1), (x2, y2) in red_bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color_class1, bbox_thickness + 3)
    
    # Layer 4: All black text labels - ON TOP OF ALL BBOXES
    for text, (x, y) in black_texts:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Layer 5: All blue vertices (Class 0) - IN FRONT OF EVERYTHING
    for polygon_pixels in blue_vertices:
        for j, (px, py) in enumerate(polygon_pixels):
            cv2.circle(image, (px, py), 6, (255, 0, 0), -1)  # Blue dots, bigger
            cv2.putText(image, f"{j+1}", (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save the output image
    if output_path is None:
        # Generate output path based on input
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_yolo_visualized{input_path.suffix}"
    
    try:
        cv2.imwrite(str(output_path), image)
        print(f"YOLO visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        return False
    
    # Show preview if requested
    if show_preview:
        try:
            # Resize for display if image is too large
            display_image = image.copy()
            max_display_size = 1200
            if max(width, height) > max_display_size:
                scale = max_display_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
                print(f"Preview resized to {new_width} x {new_height} for display")
            
            cv2.imshow('YOLO Labels Visualization (Press any key to close)', display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"Warning: Could not display preview window: {e}")
            print("This is normal if OpenCV was compiled without GUI support.")
            print("The visualization has been saved to the file.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO-Seg labels on an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--labels-file", required=True, help="File containing YOLO-Seg labels")
    parser.add_argument("--output", help="Output image path (optional)")
    parser.add_argument("--bbox-color", default="green", choices=["red", "green", "blue", "yellow", "cyan", "magenta", "white"], 
                       help="Bounding box color")
    parser.add_argument("--polygon-color", default="red", choices=["red", "green", "blue", "yellow", "cyan", "magenta", "white"], 
                       help="Polygon color")
    parser.add_argument("--bbox-thickness", type=int, default=3, help="Bounding box line thickness")
    parser.add_argument("--polygon-thickness", type=int, default=2, help="Polygon line thickness")
    parser.add_argument("--preview", action="store_true", help="Show preview window")
    
    args = parser.parse_args()
    
    # Color mapping
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255)
    }
    
    bbox_color = color_map[args.bbox_color]
    polygon_color = color_map[args.polygon_color]
    
    # Parse labels from file
    labels = parse_yolo_labels(args.labels_file)
    if not labels:
        print("No valid labels found.")
        return
    
    # Draw the labels (using class-specific colors, ignoring bbox_color and polygon_color)
    success = draw_yolo_labels(
        image_path=args.image,
        labels=labels,
        output_path=args.output,
        bbox_thickness=args.bbox_thickness,
        polygon_thickness=args.polygon_thickness,
        show_preview=args.preview
    )
    
    if success:
        print("YOLO labels visualized successfully!")
    else:
        print("Failed to visualize labels.")


if __name__ == "__main__":
    main() 