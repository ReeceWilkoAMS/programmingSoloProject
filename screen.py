import cv2
import os
import numpy as np
from mss import mss

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to store the images
output_dir = "test_data\\franScreen"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Press 'q' to quit.")

# Initialize MSS for screen capture
screen_capture = mss()

# List all connected monitors
monitors = screen_capture.monitors  # List of monitors
print("Available monitors:")
for idx, monitor in enumerate(monitors):
    print(f"Monitor {idx}: {monitor}")

# Select the second screen (index 2; index 0 is the "virtual" full monitor, index 1 is primary monitor)
monitor_region = monitors[2]  # Replace 2 with the desired monitor index
print(f"Using monitor region: {monitor_region}")

face_counter = 0  # To keep track of saved face images

while True:
    # Capture the screen
    screen_frame = screen_capture.grab(monitor_region)

    # Convert the captured image to a format OpenCV can work with (numpy array)
    frame = np.array(screen_frame)

    # Convert from BGRA to BGR (as OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and save the images
    for (x, y, w, h) in faces:
        # Draw rectangle on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region from the frame
        face_roi = frame[y:y + h, x:x + w]

        # Save the cropped face image to the folder
        face_filename = os.path.join(output_dir, f"face_{face_counter}.jpg")
        cv2.imwrite(face_filename, face_roi)

        # Increment the counter for unique filenames
        face_counter += 1

    # Display the captured screen frame with rectangles around detected faces
    cv2.imshow('Facial Recognition - Screen Capture', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
