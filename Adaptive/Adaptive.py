#!/usr/bin/env python3
"""
adaptive_threshold.py

Applies adaptive thresholding methods to indoor, outdoor, and close-up images.
Outputs individual results and combined matplotlib visualizations.

Author: [Your Name]
Date: 2025-09-18
"""

import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(gray, blur_type='gaussian', blur_ksize=(5, 5)):
    if blur_type == 'gaussian':
        return cv2.GaussianBlur(gray, blur_ksize, 0)
    elif blur_type == 'median':
        return cv2.medianBlur(gray, blur_ksize[0])
    else:
        return gray

# -------------------------------
# Adaptive Thresholding Methods
# -------------------------------
def apply_thresholds(preprocessed, win_size=11, C=2):
    methods = {}

    # OpenCV Adaptive Mean
    mean_thresh = cv2.adaptiveThreshold(preprocessed, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, win_size, C)
    methods['Adaptive Mean'] = mean_thresh

    # OpenCV Adaptive Gaussian
    gauss_thresh = cv2.adaptiveThreshold(preprocessed, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, win_size, C)
    methods['Adaptive Gaussian'] = gauss_thresh

    # Otsu's Global Threshold
    _, otsu_thresh = cv2.threshold(preprocessed, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods["Otsu's Method"] = otsu_thresh

    # Simple Binary Threshold
    _, simple_thresh = cv2.threshold(preprocessed, 127, 255, cv2.THRESH_BINARY)
    methods["Simple Binary"] = simple_thresh
    
    # Inverted Binary Threshold
    _, inv_thresh = cv2.threshold(preprocessed, 127, 255, cv2.THRESH_BINARY_INV)
    methods["Inverted Binary"] = inv_thresh

    return methods

# -------------------------------
# Matplotlib Display and Save
# -------------------------------
def plot_results(img, gray, preprocessed, results, basename, output_dir):
    try:
        plt.figure(figsize=(16, 10))
        titles = ['Original', 'Grayscale', 'Preprocessed'] + list(results.keys())
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, preprocessed] + list(results.values())

        n = len(images)
        for i in range(n):
            plt.subplot(2, (n + 1) // 2, i + 1)
            cmap = 'gray' if len(images[i].shape) == 2 else None
            plt.imshow(images[i], cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"{basename}_summary.png")
        plt.savefig(summary_path)
        plt.close()  # Close instead of show to avoid blocking
        print(f"[INFO] Summary plot saved: {summary_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create plot: {e}")

# -------------------------------
# Main Processing Function
# -------------------------------
def process_image(image_path, output_dir, args):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return

        basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"[INFO] Processing {basename}...")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pre = preprocess_image(gray, blur_type=args.blur, blur_ksize=(args.blurksize, args.blurksize))

        results = apply_thresholds(pre, win_size=args.win, C=args.C)

        # Save each output
        cv2.imwrite(os.path.join(output_dir, f"{basename}_gray.png"), gray)
        cv2.imwrite(os.path.join(output_dir, f"{basename}_preprocessed.png"), pre)
        for name, result in results.items():
            filename = f"{basename}_{name.replace(' ', '_').lower()}.png"
            cv2.imwrite(os.path.join(output_dir, filename), result)

        # Plot and save visual summary
        plot_results(img, gray, pre, results, basename, output_dir)
        print(f"[INFO] Completed processing {basename}")
        
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")

# -------------------------------
# Main CLI Parser
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Adaptive Thresholding on Image Set")
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory to save results')
    parser.add_argument('--blur', type=str, default='gaussian',
                        choices=['gaussian', 'median', 'none'],
                        help='Preprocessing blur type')
    parser.add_argument('--blurksize', type=int, default=5,
                        help='Blur kernel size (odd number)')
    parser.add_argument('--win', type=int, default=11,
                        help='Window size for adaptive/local methods (odd)')
    parser.add_argument('--C', type=int, default=2,
                        help='Constant subtracted from mean/weighted mean')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    image_files = [f for f in os.listdir(args.input)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print(f"[ERROR] No image files found in directory: {args.input}")
        return

    for fname in image_files:
        image_path = os.path.join(args.input, fname)
        print(f"[INFO] Processing: {fname}")
        process_image(image_path, args.output, args)

    print("[DONE] All images processed.")

# -------------------------------
# Simple Main Function
# -------------------------------
def simple_main():
    # Get current working directory
    current_dir = os.getcwd()
    print(f"[DEBUG] Current working directory: {current_dir}")
    
    # Use absolute paths
    input_dir = os.path.join(current_dir, 'Adaptive', 'Images')
    output_dir = os.path.join(current_dir, 'Adaptive', 'Results')
    
    print(f"[DEBUG] Input directory path: {input_dir}")
    print(f"[DEBUG] Output directory path: {output_dir}")
    
    # Create both input and output directories
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Created directories. Please place your images in: {input_dir}")
    
    # Default parameters
    class Args:
        blur = 'gaussian'
        blurksize = 5
        win = 11
        C = 2
    
    args = Args()
    
    print(f"[INFO] Looking for images in: {input_dir}")
    
    # Check if directory exists and list files
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory does not exist: {input_dir}")
        return
    
    # List all files in directory for debugging
    all_files = os.listdir(input_dir)
    print(f"[DEBUG] All files found: {all_files}")
    
    # Find image files
    image_files = [f for f in all_files
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    
    print(f"[DEBUG] Image files found: {image_files}")
    
    if not image_files:
        print(f"[ERROR] No image files found in directory: {input_dir}")
        print(f"[INFO] Files in directory: {all_files}")
        print(f"[INFO] Supported extensions: .png, .jpg, .jpeg, .bmp, .tiff, .webp")
        print(f"[INFO] Please copy your image files to: {input_dir}")
        return
    
    # Process each image
    for fname in image_files:
        image_path = os.path.join(input_dir, fname)
        print(f"[INFO] Processing: {fname}")
        process_image(image_path, output_dir, args)
    
    print("[DONE] All images processed and saved to Adaptive/Results/")

# -------------------------------
# Run Entry Point
# -------------------------------
if __name__ == "__main__":
    simple_main()
