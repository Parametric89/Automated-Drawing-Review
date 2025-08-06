"""
draw_circle.py
---------------
Draw a large, visible circle around a specified pixel coordinate on an image.

Usage:
    python draw_circle.py --image path/to/image.jpg --x 1000 --y 1500 --output marked_image.jpg

For large images (9k x 6k), the circle will be sized appropriately to be visible at full screen.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def draw_circle_on_image(image_path, x, y, output_path=None, circle_radius=None, 
                        circle_color=(0, 255, 0), thickness=10, show_preview=False):
    """
    Draw a large circle around the specified pixel coordinate.
    
    Args:
        image_path (str): Path to input image
        x (int): X coordinate (0 is left edge)
        y (int): Y coordinate (0 is top edge)
        output_path (str): Path for output image (optional)
        circle_radius (int): Radius of circle in pixels (auto-calculated if None)
        circle_color (tuple): BGR color tuple (default: green)
        thickness (int): Line thickness in pixels
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
    
    # Validate coordinates
    if x < 0 or x >= width or y < 0 or y >= height:
        print(f"Error: Coordinates ({x}, {y}) are outside image bounds ({width} x {height})")
        return False
    
    # Calculate appropriate circle radius based on image size
    if circle_radius is None:
        # For large images, make circle proportional to image size
        min_dimension = min(width, height)
        circle_radius = max(50, min_dimension // 20)  # At least 50px, or 5% of smaller dimension
    
    print(f"Drawing circle with radius {circle_radius} pixels at ({x}, {y})")
    
    # Draw the circle
    cv2.circle(image, (x, y), circle_radius, circle_color, thickness)
    
    # Also draw a small dot at the exact center for precision
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red filled circle
    
    # Save the output image
    if output_path is None:
        # Generate output path based on input
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_marked{input_path.suffix}"
    
    try:
        cv2.imwrite(str(output_path), image)
        print(f"Marked image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        return False
    
    # Show preview if requested
    if show_preview:
        # Resize for display if image is too large
        display_image = image.copy()
        max_display_size = 1200
        if max(width, height) > max_display_size:
            scale = max_display_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
            print(f"Preview resized to {new_width} x {new_height} for display")
        
        cv2.imshow('Marked Image (Press any key to close)', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Draw a large circle around a pixel coordinate on an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--x", required=True, type=int, help="X coordinate (0 is left edge)")
    parser.add_argument("--y", required=True, type=int, help="Y coordinate (0 is top edge)")
    parser.add_argument("--output", help="Output image path (optional)")
    parser.add_argument("--radius", type=int, help="Circle radius in pixels (auto-calculated if not specified)")
    parser.add_argument("--color", default="green", choices=["red", "green", "blue", "yellow", "cyan", "magenta", "white"], 
                       help="Circle color")
    parser.add_argument("--thickness", type=int, default=10, help="Line thickness in pixels")
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
    
    circle_color = color_map[args.color]
    
    # Draw the circle
    success = draw_circle_on_image(
        image_path=args.image,
        x=args.x,
        y=args.y,
        output_path=args.output,
        circle_radius=args.radius,
        circle_color=circle_color,
        thickness=args.thickness,
        show_preview=args.preview
    )
    
    if success:
        print("Circle drawn successfully!")
    else:
        print("Failed to draw circle.")


if __name__ == "__main__":
    main() 