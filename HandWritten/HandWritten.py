import os
import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return None
    return image

def save_image(original_image, output_path):
    success = cv2.imwrite(output_path, image)
    if not success:
        print(f"Error: Failed to save image to {output_path}")
        return False
    return True

def binarize_image(original_image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image
    
def main():
    image_path = 'image/Handwritten.jpg'
    original_image = load_image(image_path)
    output_path = 'image/Handwritten_Binary.jpg'
    binary_image = binarize_image(original_image)
    
  

