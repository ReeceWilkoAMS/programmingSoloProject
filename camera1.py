import cv2
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Create the directory to store the images if it doesn't exist
output_dir = "test_data\\reece"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Press 'q' to quit.")

face_counter = 0  # To keep track of the saved face images

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (Haar cascades work with grayscale images)
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

    # Display the frame with the rectangle drawn around faces
    cv2.imshow('Facial Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
