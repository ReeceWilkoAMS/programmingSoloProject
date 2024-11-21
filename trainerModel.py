import os
import cv2
import numpy as np

# Path to the test data folder
test_data_dir = "test_data"

# Path to the label map file
label_map_path = "label_map.npy"

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

# Variables to store training data
faces = []
labels = []

# Load the label map if it exists, otherwise start with an empty dictionary
if os.path.exists(label_map_path):
    try:
        label_dict = np.load(label_map_path, allow_pickle=True).item()
        if not isinstance(label_dict, dict):
            raise ValueError("Loaded label map is not a valid dictionary.")
        label_id = max(label_dict.values()) + 1  # Start from the next available ID
    except Exception as e:
        print(f"Error loading label map: {e}. Starting fresh.")
        label_dict = {}
        label_id = 0
else:
    print("No label map found. Starting fresh.")
    label_dict = {}
    label_id = 0

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

# Train or update the recognizer
if len(faces) > 0:
    if os.path.exists(model_path):
        print("Updating the existing model...")
        recognizer.update(faces, np.array(labels))  # Update the model with new data
    else:
        print("Training the model for the first time...")
        recognizer.train(faces, np.array(labels))  # Train the model

    recognizer.save(model_path)
    print(f"Model saved as '{model_path}'.")

    # Save the updated label mapping
    np.save("label_map.npy", label_dict)

    # Clear the test_data directory to save space
    print("Clearing the test_data directory...")
    for file in os.listdir(test_data_dir):
        file_path = os.path.join(test_data_dir, file)
        if os.path.isdir(file_path):
            for sub_file in os.listdir(file_path):
                os.remove(os.path.join(file_path, sub_file))
            os.rmdir(file_path)
    print("test_data directory cleared.")

    # Print label mapping for reference
    print("Label mapping:")
    for name, label in label_dict.items():
        print(f"{label}: {name}")
else:
    print("No valid training data found. Model not updated.")
