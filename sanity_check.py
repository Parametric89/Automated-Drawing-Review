# sanity_check.py
import os
import random
import cv2
import json
from matplotlib import pyplot as plt
from pathlib import Path

# Define directories
crop_dir = "datasets/rcp_dual_seg/images/train/cropped1k"
lbl_dir  = "datasets/rcp_dual_seg/labels/train/cropped1k"

# Check if directories exist
if not os.path.exists(crop_dir) or not os.listdir(crop_dir):
    print(f"Error: Crop directory '{crop_dir}' is empty or does not exist.")
    print("Please run the panel_cropper.py script first.")
else:
    # Get a random file
    fn = random.choice(os.listdir(crop_dir))
    img_path = os.path.join(crop_dir, fn)
    label_path = os.path.join(lbl_dir, Path(fn).stem + ".txt")

    # Read and process the image
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Read and process the labels
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = list(map(float, line.strip().split()))
                # Ensure there are polygon points to draw
                if len(p) > 5:
                    # The polygon points start after the class and bbox info (5 elements)
                    pts = np.array(p[5:]).reshape(-1, 2) * [w, h]
                    cv2.polylines(img, [pts.astype(int)], True, (255, 0, 0), 2)

    # Display the image
    plt.imshow(img)
    plt.title(fn)
    plt.axis('off')
    plt.show()
