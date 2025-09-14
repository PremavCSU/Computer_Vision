import cv2

def load_cascades():
    "load Haar cascades for face and eye detection"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade



def capture_image():
    "capture single image from webcam"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    ret, frame = cap.read()
    cap.release()
    return frame

def annotate_image(image, face_cascade, eye_cascade):
    """Draw circle around face, rectangles around eyes, and tag the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw green circle around the face
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        cv2.circle(image, center, radius, (0, 255, 0), 2)

        # Detect eyes within face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
def save_display_image(frame):
    "save image to file"
    cv2.imwrite('Image/detected_face_eyes.jpg', frame)
    cv2.imshow('Detected Face and Eyes', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    face_cascade, eye_cascade = load_cascades()
    frame = capture_image()
    annotate_image(frame, face_cascade, eye_cascade)
    save_display_image(frame)

if __name__ == "__main__":
    main()
