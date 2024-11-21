import cv2
import numpy as np

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.read("face_recognizer.yml")

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load the label mapping
try:
    label_dict = np.load("label_map.npy", allow_pickle=True).item()
    print("Label mapping loaded successfully.")
except FileNotFoundError:
    print("Error: 'label_map.npy' not found. Train the model first.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = gray[y:y + h, x:x + w]
        
        # Recognize the face
        label, confidence = recognizer.predict(face_roi)

        CONFIDENCE_THRESHOLD = 50.0

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the name of the person
        if confidence < CONFIDENCE_THRESHOLD:
            name = label_dict.get(label, "Unknown")
        else:
            name = "Unknown"
        cv2.putText(frame, f"{name}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
