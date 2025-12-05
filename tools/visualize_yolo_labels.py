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
    """Parse YOLO(-Seg) labels from a file. Quiet by default."""
    labels = []
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    polygon_coords = []
                    if class_id == 0:
                        # Panels: polygon is optional. If present, parse pairs after bbox
                        if len(parts) >= 11:  # bbox(4) + at least 3 pairs (6)
                            for i in range(5, len(parts), 2):
                                if i + 1 < len(parts):
                                    x = float(parts[i])
                                    y = float(parts[i + 1])
                                    polygon_coords.append((x, y))
                    else:
                        polygon_coords = []
                    labels.append({
                        'class_id': class_id,
                        'bbox': (center_x, center_y, width, height),
                        'polygon': polygon_coords,
                        'line_number': line_num,
                    })
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return labels


def denormalize_coordinates(x_norm, y_norm, img_width, img_height):
    """Convert normalized coordinates back to pixel coordinates."""
    x_pixel = int(x_norm * img_width)
    y_pixel = int(y_norm * img_height)
    return x_pixel, y_pixel


def draw_yolo_labels(image_path, labels, output_path=None,
                    bbox_thickness=3, polygon_thickness=2, show_preview=False,
                    verbose: bool = False):
    """
    Draw YOLO(-Seg) labels on an image with class-specific colors.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
    except Exception:
        return False

    height, width = image.shape[:2]
    if verbose:
        print(f"Image dimensions: {width} x {height} pixels")
        print(f"Drawing {len(labels)} YOLO labels:")

    bbox_color_class0 = (0, 255, 0)
    bbox_color_class1 = (0, 0, 255)
    polygon_color = (255, 0, 255)
    text_color = (0, 0, 0)

    class_names = {0: "Panel", 1: "Panel Tag"}

    green_bboxes = []
    magenta_polygons = []
    red_bboxes = []
    black_texts = []
    blue_vertices = []

    for i, label in enumerate(labels):
        class_id = label['class_id']
        bbox = label['bbox']
        polygon = label['polygon']
        class_name = class_names.get(class_id, f"Class {class_id}")
        center_x_norm, center_y_norm, width_norm, height_norm = bbox
        center_x, center_y = denormalize_coordinates(center_x_norm, center_y_norm, width, height)
        bbox_width = int(width_norm * width)
        bbox_height = int(height_norm * height)
        x1 = center_x - bbox_width // 2
        y1 = center_y - bbox_height // 2
        x2 = center_x + bbox_width // 2
        y2 = center_y + bbox_height // 2
        polygon_pixels = []
        for x_norm, y_norm in polygon:
            x_pixel, y_pixel = denormalize_coordinates(x_norm, y_norm, width, height)
            polygon_pixels.append((x_pixel, y_pixel))
        if class_id == 0:
            green_bboxes.append(((x1, y1), (x2, y2)))
            if len(polygon_pixels) >= 3:
                magenta_polygons.append(polygon_pixels)
                blue_vertices.append(polygon_pixels)
        else:
            red_bboxes.append(((x1, y1), (x2, y2)))
        black_texts.append((f"{class_name} ({class_id})", (x1, y1-10)))
        if verbose:
            print(f"  Label {i+1}: {class_name} (Class {class_id}), bbox({x1},{y1},{x2},{y2}), {len(polygon_pixels)} polygon points")

    for (x1, y1), (x2, y2) in green_bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color_class0, bbox_thickness)
    for polygon_pixels in magenta_polygons:
        polygon_array = np.array(polygon_pixels, dtype=np.int32)
        for i in range(len(polygon_pixels)):
            start_point = polygon_pixels[i]
            end_point = polygon_pixels[(i + 1) % len(polygon_pixels)]
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = int(np.sqrt(dx*dx + dy*dy))
            if length > 0:
                dash_length = 10
                gap_length = 25
                total_dash = dash_length + gap_length
                for j in range(0, length, total_dash):
                    t_start = j / length
                    t_end = min((j + dash_length) / length, 1.0)
                    if t_end > t_start:
                        dash_start_x = int(start_point[0] + dx * t_start)
                        dash_start_y = int(start_point[1] + dy * t_start)
                        dash_end_x = int(start_point[0] + dx * t_end)
                        dash_end_y = int(start_point[1] + dy * t_end)
                        cv2.line(image, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), polygon_color, polygon_thickness)
    if red_bboxes:
        overlay = image.copy()
        tag_thickness = 1
        for (x1, y1), (x2, y2) in red_bboxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bbox_color_class1, tag_thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
    for text, (x, y) in black_texts:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    for polygon_pixels in blue_vertices:
        for j, (px, py) in enumerate(polygon_pixels):
            cv2.circle(image, (px, py), 6, (255, 0, 0), -1)
            cv2.putText(image, f"{j+1}", (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_yolo_visualized{input_path.suffix}"
    try:
        cv2.imwrite(str(output_path), image)
    except Exception:
        return False

    if show_preview:
        try:
            display_image = image.copy()
            max_display_size = 1200
            if max(width, height) > max_display_size:
                scale = max_display_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
            cv2.imshow('YOLO Labels Visualization (Press any key to close)', display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO-Seg labels on an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--labels-file", required=True, help="File containing YOLO-Seg labels")
    parser.add_argument("--output", help="Output image path (optional)")
    parser.add_argument("--bbox-color", default="green", choices=["red", "green", "blue", "yellow", "cyan", "magenta", "white"], help="Bounding box color")
    parser.add_argument("--polygon-color", default="red", choices=["red", "green", "blue", "yellow", "cyan", "magenta", "white"], help="Polygon color")
    parser.add_argument("--bbox-thickness", type=int, default=3, help="Bounding box line thickness")
    parser.add_argument("--polygon-thickness", type=int, default=2, help="Polygon line thickness")
    parser.add_argument("--preview", action="store_true", help="Show preview window")
    parser.add_argument("--verbose", action="store_true", help="Print extra details while drawing")
    args = parser.parse_args()

    labels = parse_yolo_labels(args.labels_file)
    if not labels:
        return

    success = draw_yolo_labels(
        image_path=args.image,
        labels=labels,
        output_path=args.output,
        bbox_thickness=args.bbox_thickness,
        polygon_thickness=args.polygon_thickness,
        show_preview=args.preview,
        verbose=bool(args.verbose),
    )

    if not success:
        pass


if __name__ == "__main__":
    main() 