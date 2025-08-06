#!/usr/bin/env python3
"""
smart_augmentation.py
--------------------
Smart augmentation for panel detection training.

Augmentation strategies:
1. Random inward cropping (80-90% of panel)
2. Random offset on blank canvas
3. Scale jitter (0.7-1.3)
4. Light blur/noise for print artifacts
5. Random rotation and brightness

This improves generalization for different drawing scales and conditions.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import random
import copy
from tqdm import tqdm


def cleanup_old_augmented(aug_dir):
    """Clean up old augmented directory"""
    if os.path.exists(aug_dir):
        for file in os.listdir(aug_dir):
            os.remove(os.path.join(aug_dir, file))
    else:
        os.makedirs(aug_dir, exist_ok=True)


def parse_yolo_seg_label(label_path):
    """Parse YOLO-Seg labels"""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            pts = [float(v) for v in parts[5:]]
            annotations.append({
                'class_id': cls,
                'bbox': (cx, cy, w, h),
                'polygon': pts
            })
    return annotations


def centre_inside(bbox, crop, img_w, img_h):
    """Check if the centre of a normalised YOLO bbox is inside the crop (pixel coords)."""
    cx, cy, _, _ = bbox
    cx *= img_w
    cy *= img_h
    return crop[0] <= cx <= crop[2] and crop[1] <= cy <= crop[3]


def apply_inward_cropping(image, panel_bbox, crop_ratio=0.85):
    """Apply inward cropping to keep 80-90% of panel"""
    h, w = image.shape[:2]
    cx, cy, bw, bh = panel_bbox
    
    # Convert normalized to absolute
    x_center = cx * w
    y_center = cy * h
    panel_w = bw * w
    panel_h = bh * h
    
    # Calculate crop bounds (smaller than panel)
    crop_w = panel_w * crop_ratio
    crop_h = panel_h * crop_ratio
    
    # Random offset within panel bounds
    max_offset_x = panel_w - crop_w
    max_offset_y = panel_h - crop_h
    
    if max_offset_x > 0:
        offset_x = random.uniform(0, max_offset_x)
    else:
        offset_x = 0
    
    if max_offset_y > 0:
        offset_y = random.uniform(0, max_offset_y)
    else:
        offset_y = 0
    
    # Calculate crop bounds
    crop_x1 = max(0, int(x_center - panel_w/2 + offset_x))
    crop_y1 = max(0, int(y_center - panel_h/2 + offset_y))
    crop_x2 = min(w, int(crop_x1 + crop_w))
    crop_y2 = min(h, int(crop_y1 + crop_h))
    
    # Extract crop
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def apply_scale_jitter(image, scale_range=(0.7, 1.3)):
    """Apply scale jitter to image"""
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    return resized, scale


def apply_print_artifacts(image, blur_prob=0.3, noise_prob=0.2):
    """Apply light blur and noise to mimic print artifacts"""
    # Light blur
    if random.random() < blur_prob:
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Light noise
    if random.random() < noise_prob:
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def apply_brightness_contrast(image, brightness_range=(-30, 30), contrast_range=(0.8, 1.2)):
    """Apply brightness and contrast adjustments"""
    brightness = random.uniform(brightness_range[0], brightness_range[1])
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    
    # Apply brightness and contrast
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    return image


def apply_random_rotation(image, max_angle=15):
    """Apply random rotation and return the rotation matrix for label transformation."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    
    # Calculate rotation matrix
    center_x, center_y = w / 2, h / 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated, rotation_matrix


def transform_labels_for_augmentation(annotations, initial_size, crop_bounds, initial_scale, rotation_matrix, final_scale, pad_offset, final_size):
    """Transform labels for augmented image using the same transformations."""
    transformed_labels = []
    initial_w, initial_h = initial_size
    final_w, final_h = final_size
    crop_x1, crop_y1, _, _ = crop_bounds

    for ann in annotations:
        # 1. Denormalize to original image coordinates
        denormalized_polygon = np.array(ann['polygon']).reshape(-1, 2) * [initial_w, initial_h]

        # 2. Adjust for inward crop
        adjusted_polygon = denormalized_polygon - [crop_x1, crop_y1]
        
        # 3. Apply scale jitter
        jitter_scaled_polygon = adjusted_polygon * initial_scale
        
        # 4. Apply rotation using the exact same matrix as the image
        # cv2.transform needs an extra dimension
        reshaped_polygon = jitter_scaled_polygon.reshape(-1, 1, 2).astype(np.float32)
        rotated_polygon = cv2.transform(reshaped_polygon, rotation_matrix).reshape(-1, 2)

        # 5. Apply final scaling and padding
        final_polygon_abs = rotated_polygon * final_scale + [pad_offset[0], pad_offset[1]]
        
        # 6. Normalize to final image size
        final_polygon_norm = final_polygon_abs / [final_w, final_h]

        # Clamp to [0, 1] and flatten
        final_polygon = np.clip(final_polygon_norm, 0, 1).flatten()

        # Create new label string if valid
        if len(final_polygon) >= 6:
            xs = final_polygon[0::2]
            ys = final_polygon[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w, h = xmax - xmin, ymax - ymin

            if w > 0.01 and h > 0.01:
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                label_line = f"{ann['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + \
                             " ".join(f"{p:.6f}" for p in final_polygon)
                transformed_labels.append(label_line)

    return transformed_labels


def augment_image(image_path, label_path, target_size=1024, num_augmentations=3):
    """Apply smart augmentation to an image"""
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    annotations = parse_yolo_seg_label(label_path)
    if not annotations:
        return []
    
    augmented_samples = []
    
    for aug_idx in range(num_augmentations):
        # Start with original image
        aug_image = image.copy()
        aug_annotations = copy.deepcopy(annotations)
        
        # Apply augmentations
        crop_bounds = (0, 0, image.shape[1], image.shape[0])
        
        # 1. Inward cropping (if we have panels)
        panel_annotations = [ann for ann in annotations if ann['class_id'] == 0]
        if panel_annotations:
            panel = random.choice(panel_annotations)
            crop_ratio = random.uniform(0.8, 0.9)  # 80-90% of panel
            aug_image, crop_bounds = apply_inward_cropping(aug_image, panel['bbox'], crop_ratio)
            
            # Drop objects that lie outside the crop
            aug_annotations = [
                ann for ann in aug_annotations
                if centre_inside(ann['bbox'], crop_bounds, image.shape[1], image.shape[0])
            ]
        
        # 2. Scale jitter
        aug_image, initial_scale = apply_scale_jitter(aug_image)
        
        # 3. Random rotation
        aug_image, rotation_matrix = apply_random_rotation(aug_image)
        
        # 4. Print artifacts
        aug_image = apply_print_artifacts(aug_image)
        
        # 5. Brightness/contrast
        aug_image = apply_brightness_contrast(aug_image)
        
        # 6. Resize to target size and pad
        h, w = aug_image.shape[:2]
        final_scale = target_size / max(h, w)
        new_w, new_h = int(w * final_scale), int(h * final_scale)
        
        resized_image = cv2.resize(aug_image, (new_w, new_h))
        
        padded_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        aug_image = padded_image
        
        # Transform labels
        transformed_labels = transform_labels_for_augmentation(
            annotations=aug_annotations,
            initial_size=(image.shape[1], image.shape[0]),
            crop_bounds=crop_bounds,
            initial_scale=initial_scale,
            rotation_matrix=rotation_matrix,
            final_scale=final_scale,
            pad_offset=(x_offset, y_offset),
            final_size=(target_size, target_size)
        )
        
        if transformed_labels:  # Only keep if we have valid labels
            augmented_samples.append({
                'image': aug_image,
                'labels': transformed_labels,
                'augmentation_id': aug_idx
            })
    
    return augmented_samples


def augment_split_images(split, base_dir="datasets/rcp_dual_seg", target_size=1024, num_augmentations=3):
    """Augment images in a split"""
    img_dir = f"{base_dir}/images/{split}/cropped1k"
    lbl_dir = f"{base_dir}/labels/{split}/cropped1k"
    out_img = f"{base_dir}/images/{split}/augmented1k"
    out_lbl = f"{base_dir}/labels/{split}/augmented1k"
    
    # Check if cropped directories exist
    if not os.path.exists(img_dir):
        print(f"Warning: Cropped image directory not found: {img_dir}")
        print(f"Make sure you've run panel cropping first")
        return 0
    
    if not os.path.exists(lbl_dir):
        print(f"Warning: Cropped label directory not found: {lbl_dir}")
        print(f"Make sure you've run panel cropping first")
        return 0
    
    cleanup_old_augmented(out_img)
    cleanup_old_augmented(out_lbl)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    total_augmented = 0
    
    for fn in tqdm(os.listdir(img_dir), desc=f"Augmenting {split} images"):
        if not fn.lower().endswith(('jpg', 'png')):
            continue
        
        img_path = f"{img_dir}/{fn}"
        lbl_path = f"{lbl_dir}/{Path(fn).stem}.txt"
        
        # Apply smart augmentation
        augmented_samples = augment_image(img_path, lbl_path, target_size, num_augmentations)
        
        if augmented_samples:
            for i, sample in enumerate(augmented_samples):
                # Save augmented image
                aug_name = f"{Path(fn).stem}_aug{i:02d}"
                cv2.imwrite(f"{out_img}/{aug_name}.jpg", sample['image'])
                
                # Save augmented labels
                with open(f"{out_lbl}/{aug_name}.txt", 'w') as f:
                    f.write('\n'.join(sample['labels']))
                
                total_augmented += 1
    
    return total_augmented


def main():
    """Main function for smart augmentation"""
    print("=== Smart Augmentation ===")
    print("Applying advanced augmentation for better generalization")
    print(f"Target size: 1024px, Augmentations per image: 3")
    
    splits = ['train', 'val', 'test']
    info = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        info[split] = augment_split_images(split)
    
    # Save augmentation info
    Path("datasets/rcp_dual_seg").mkdir(exist_ok=True)
    with open(f"datasets/rcp_dual_seg/augmentation_info.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'counts': info,
            'target_size': 1024,
            'num_augmentations': 3
        }, f, indent=2)
    
    print(f"\nAugmentation Summary: {info}")
    print("Smart augmentation complete!")
    print("Improved generalization for different scales and conditions")


if __name__ == "__main__":
    main()