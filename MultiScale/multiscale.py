import os
import cv2
import matplotlib.pyplot as plt

image_path = "Image/Puppy.jpg"

#load image
def image_load(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
  
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

#discplay images
def image_display(image):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def image_save(image, output_path):
    cv2.imwrite(output_path, image)

def extract_channels(image):
  
    blue_channel = image[:, :, 0]
    green_chann = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    return blue_channel, green_chann, red_channel
    #print(blue_channel)
   # print(green_chann)
   # print(red_channel)

#merge image
def merge_channels(blue_channel, green_chann, red_channel):
    merged_image = cv2.merge((blue_channel, green_chann, red_channel))
    return merged_image

#display channels
def display_channels(blue_channel, green_chann, red_channel):

    # Display each channel using matplotlib 
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    plt.imshow(blue_channel, cmap='Blues')
    plt.title('Blue Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_chann, cmap='Greens')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(red_channel, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')


def main():
    # Step 1: Load and extract channels
    image = image_load(image_path)
    image_display(image)
    blue_channel, green_channel, red_channel = extract_channels(image)
    display_channels(blue_channel, green_channel, red_channel)

    # Step 2: Merge back to original RGB
    original_rgb = merge_channels(blue_channel, green_channel, red_channel)
    image_display(original_rgb)
    
    # Step 3: Swap red and green channels (GRB)
    swapped_rgb = merge_channels(red_channel, green_channel, blue_channel) 
    image_display(swapped_rgb)

    #display in matplatlib 
    plt.show()
    # Step 4: Save the images
    image_save(original_rgb, "Image/original_rgb.jpg")

if __name__ == "__main__":
    main()


