import os
import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return None
    return image

def binarize_image(gray):
    # Gaussian blur before Otsu binarization for better results
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image
    
def morph_operations(binary_image, gray):
    # Use elliptical kernel for better text processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    # Basic morphological operations
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Advanced morphological operations
    gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    return eroded, dilated, opening, closing, gradient, tophat, blackhat

def save_image(image, output_path):
    success = cv2.imwrite(output_path, image)
    if not success:
        print(f"Error: Failed to save image to {output_path}")
        return False
    return True

def main():
    image_path = 'image/Handwritten.jpg'
    original_image = load_image(image_path)
    
    # Create Image directory if it doesn't exist
    os.makedirs('Image', exist_ok=True)
    
    if original_image is None:
        return
    
    # Save original grayscale
    save_image(original_image, 'Image/01_gray.jpg')
    
    # Binarize and save
    binary_image = binarize_image(original_image)
    save_image(binary_image, 'Image/02_binary.jpg')
    
    # Apply morphological operations
    eroded, dilated, opening, closing, gradient, tophat, blackhat = morph_operations(binary_image, original_image)
    
    # Save all processed images
    save_image(eroded, 'Image/03_eroded.jpg')
    save_image(dilated, 'Image/04_dilated.jpg')
    save_image(opening, 'Image/05_opening.jpg')
    save_image(closing, 'Image/06_closing.jpg')
    save_image(gradient, 'Image/07_gradient.jpg')
    save_image(tophat, 'Image/08_tophat.jpg')
    save_image(blackhat, 'Image/09_blackhat.jpg')
    
    print("All images saved to Image/ directory")
    print(f"Processed: grayscale, binary, eroded, dilated, opening, closing, gradient, tophat, blackhat")

if __name__ == "__main__":
    main() 