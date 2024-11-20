import os
import cv2
import numpy as np

# Path to the test data folder
test_data_dir = "test_data"

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to save the model
model_path = "face_recognizer.yml"

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading existing model...")
    recognizer.read(model_path)
else:
    print("No existing model found. Starting fresh.")
    
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

# Variables to store training data
faces = []
labels = []
label_dict = {}  # Maps labels to person names
label_id = 0     # Numeric label for each person

# Iterate over each person's folder in the test_data directory
for person_name in os.listdir(test_data_dir):
    person_path = os.path.join(test_data_dir, person_name)

    # Ensure it is a directory
    if os.path.isdir(person_path):
        print(f"Processing images for: {person_name}")

        # Assign a label ID to the person
        if person_name not in label_dict:
            label_dict[person_name] = label_id
            label_id += 1
        
        person_label = label_dict[person_name]

        # Iterate through each image in the person's folder
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)

            # Read the image in grayscale (as required by the recognizer)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Skipping invalid image: {image_path}")
                continue

            # Append the face and label to the training data
            faces.append(image)
            labels.append(person_label)

# Train the recognizer if data is available
if len(faces) > 0:
    print("Training the model...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)
    print(f"Model trained and saved as '{model_path}'.")
    
    # Clear the test_data directory to save space
    print("Clearing the test_data directory...")
    clear_directory(test_data_dir)
    print("test_data directory cleared.")

    # Print label mapping for reference
    print("Label mapping:")
    for name, label in label_dict.items():
        print(f"{label}: {name}")
else:
    print("No valid training data found. Model not updated.")
