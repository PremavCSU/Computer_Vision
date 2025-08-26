import os
import cv2

# Image file path
image_path = "Image/brain.jpg"

#Import the dimage
def image_load(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
  
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image
#Display the image
def image_display(image):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)  
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
#save the image to desktop
def save_image(image): 
    desktop_path = r"C:\Users\Prema\Desktop"
    filename = "brain_copy.jpg"
    save_copy_path = os.path.join(desktop_path, filename)
    cv2.imwrite(save_copy_path, image)
    print(f"Copy of image saved to {save_copy_path}")


def main():
    image = image_load(image_path)
    image_display(image)
    save_image(image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
