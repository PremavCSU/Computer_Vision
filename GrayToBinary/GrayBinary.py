import os
import cv2


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return None
    return image

def binary_image(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def display_image(image, window_name="Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, output_path):
    success = cv2.imwrite(output_path, image)
    if not success:
        print(f"Error: Failed to save image to {output_path}")
        return False
    return True

def main():
    image_path = "Image/Rose.jpg"
    output_path = "Image/Rose_binary.jpg"
    
    os.makedirs('Image', exist_ok=True)
    
    original_image = load_image(image_path)
    if original_image is None:
        return
    
    binary_img = binary_image(original_image)
    display_image(binary_img, "Binary Image")
    
    if save_image(binary_img, output_path):
        print(f"Binary image saved to {output_path}")
    else:
        print("Failed to save binary image")

if __name__ == "__main__":
    main()