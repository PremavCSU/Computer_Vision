import os
import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
  
    return image


def show_image(transulate, rotated, scaled, perspective_corrected, manual_translate):
    # Show translated image
    cv2.namedWindow("Translated Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Translated Image", transulate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Show rotated image
    cv2.namedWindow("Rotated Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Show scaled image
    cv2.namedWindow("Scaled Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Scaled Image", scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #show perspective correct image
    cv2.namedWindow("Perspective Corrected Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Perspective Corrected Image", perspective_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #show manual translate image
    cv2.namedWindow("Manual Translate Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Manual Translate Image", manual_translate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def translate_image(image, tx:int, ty:int):
    m_translation = np.float32([[1, 0, tx], [0, 1, ty]])
    dims = (image.shape[1], image.shape[0])
    return cv2.warpAffine(image, m_translation, dims)

def rotate_image(image, angle:float, scale:float=1.0):
    center = (image.shape[1] // 2, image.shape[0] // 2) 
    m_rotate = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, m_rotate, (image.shape[1], image.shape[0]))

def scale_image(image,sx:float, sy:float):
    m_scale = np.float32([[sx, 0, 0], [0, sy, 0]])
    return cv2.warpAffine(image, m_scale, (image.shape[1], image.shape[0]))

def perspective_transform(image,src_pts:float, dst_pts:float):
    m_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, m_perspective, (image.shape[1], image.shape[0]))

def manual_translate(image, tx, ty):
    new_img = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_x = x + tx
            new_y = y + ty
            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                new_img[new_y, new_x] = image[y, x]
    return new_img

def inspect_pixel(image):
      # Print image dimensions
    print("Image shape (height, width, channels):", image.shape)
    
    # Print pixel matrix sample (top-left corner)
    #sample_matrix = image[0:5, 0:5]  # Top-left 5x5 block
    #print("Top-left 5x5 pixel matrix (BGR values):\n", sample_matrix)

def main():
    image_path = "Image/Banknotes.jpg"

    image = load_image(image_path)
    inspect_pixel(image)

    transulated = translate_image(image,50,30)
    rotated = rotate_image(image, 10, 1.0)
    scaled = scale_image(image, 1.5, 1.5)

        # Step 4: Apply perspective transformation
    src_pts = np.array([[100, 100], [400, 100], [100, 400], [400, 400]], dtype=np.float32)  # Example source corners
    dst_pts = np.array([[120, 80], [380, 120], [80, 420], [420, 380]], dtype=np.float32)    # Desired mapped corners
    perspective_corrected = perspective_transform(scaled, src_pts, dst_pts)
    manual_translated = manual_translate(image, 50, 30)
    show_image(transulated, rotated, scaled, perspective_corrected, manual_translated)
   

if __name__ == "__main__":
    main()
