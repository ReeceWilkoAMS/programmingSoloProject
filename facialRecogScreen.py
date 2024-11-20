import cv2
import numpy as np
from mss import mss

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Debugging: List available monitors
with mss() as sct:
    monitors = sct.monitors
    for idx, monitor in enumerate(monitors):
        print(f"Monitor {idx}: {monitor}")

# Select the desired screen (update based on the printed monitor info)
screen_number = 2  # Change this to the correct index for your target screen
monitor = monitors[screen_number]

# Load the label-to-person mapping
label_dict = {0: "Reece", 1: "Fran", 2: "Person 3"}  # Update this with actual label-person mapping

print("Press 'q' to quit.")

while True:
    with mss() as sct:
        # Capture the screen
        screen_shot = np.array(sct.grab(monitor))

    # Convert the screenshot to BGR format (OpenCV uses BGR format)
    frame = cv2.cvtColor(screen_shot, cv2.COLOR_BGRA2BGR)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = gray[y:y + h, x:x + w]

        # Recognize the face
        label, confidence = recognizer.predict(face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the name of the person
        name = label_dict.get(label, "Unknown")
        cv2.putText(frame, f"{name}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition on Screen', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
