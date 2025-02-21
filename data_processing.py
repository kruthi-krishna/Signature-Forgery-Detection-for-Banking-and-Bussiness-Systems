import os
import cv2
import numpy as np

def preprocess_images(input_dir, output_dir, img_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):  # Handles subdirectories
        for file_name in files:
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
                print(f"Skipping non-image file: {file_name}")
                continue
            img_path = os.path.join(root, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                # Preserve subdirectory structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, file_name)
                cv2.imwrite(output_path, img)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Failed to process: {img_path}")

# Example usage
# preprocess_images("dataset", "processed_data")
