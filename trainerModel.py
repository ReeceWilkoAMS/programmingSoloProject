import os
import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the model file
model_path = "face_recognizer.yml"

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading existing model...")
    recognizer.read(model_path)
else:
    print("No existing model found. Starting fresh.")

# Initialize variables
faces = []
labels = []
label_dict = {}
label_id = 0

# Function to clean up the folder
def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete the file
            elif os.path.isdir(file_path):
                clear_directory(file_path)  # Recursively delete subdirectories
                os.rmdir(file_path)  # Remove the subdirectory
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# Create a test_data directory if it doesn't exist
test_data_dir = "test_data"
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit and train the model.")

# Face detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = gray[y:y + h, x:x + w]
        
        # Save the detected face
        person_name = "Person_" + str(label_id)
        person_path = os.path.join(test_data_dir, person_name)
        if not os.path.exists(person_path):
            os.makedirs(person_path)
        
        face_filename = os.path.join(person_path, f"face_{len(os.listdir(person_path))}.jpg")
        cv2.imwrite(face_filename, face_roi)

        # Add face and label to training data
        faces.append(face_roi)
        labels.append(label_id)

    # Display the frame
    cv2.imshow('Facial Recognition - Capture Mode', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

print("Training the model with captured faces...")

# Ensure there's data to train on
if len(faces) > 0:
    # Train the model incrementally
    recognizer.update(faces, np.array(labels))
    recognizer.save(model_path)
    print(f"Model updated and saved as '{model_path}'.")

    # Clear the test_data directory to save space
    print("Clearing the test_data directory...")
    clear_directory(test_data_dir)
    print("test_data directory cleared.")
else:
    print("No faces captured. Model not updated.")
