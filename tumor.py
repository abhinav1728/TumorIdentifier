
import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from imutils import paths

# Function to extract HOG features from an image
def extract_hog_features(image_path):
    image = cv2.imread(image_path)  # Read the image
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}. Please check the path and ensure the file exists.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    gray = cv2.resize(gray, (128, 128))  # Resize the image for consistency
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)  # Extract HOG features
    return fd

# Function to prepare the dataset
def prepare_data(data_path):
    image_paths = list(paths.list_images(data_path))  # List all image paths in the directory
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in the directory: {data_path}. Please check the directory.")
    
    data = []
    labels = []
    
    for image_path in image_paths:
        features = extract_hog_features(image_path)  # Extract HOG features from each image
        data.append(features)
        
        # Assign label based on the folder name
        label = 1 if "yes_t" in image_path else 0
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Function to train the KNN classifier
def train_classifier():
    try:
        # Updated paths for the no_t and yes_t directories
        no_t_data, no_t_labels = prepare_data('no_t')  # Corrected path for no_t images
        yes_t_data, yes_t_labels = prepare_data('yes_t')  # Corrected path for yes_t images
    except ValueError as e:
        print(e)
        return None  # Return None if there was an issue loading the data
    
    data = np.vstack([no_t_data, yes_t_data])  # Combine the data from both classes
    labels = np.hstack([no_t_labels, yes_t_labels])  # Combine the labels from both classes
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Initialize KNN classifier and train it
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    print("Accuracy on test set:", knn.score(X_test, y_test))  # Evaluate on the test set
    
    return knn

# Function to predict if the image has a tumor or not
def predict_image(image_path, model):
    try:
        features = extract_hog_features(image_path)  # Extract features from the input image
        prediction = model.predict([features])  # Predict using the trained model
        return "Tumor Detected" if prediction == 1 else "No Tumor Detected"
    except ValueError as e:
        return str(e)  # Return the error message if the image can't be loaded

# Train the classifier
model = train_classifier()

if model is not None:
    # Ask the user for the image path and analyze it
    image_path = input("Please enter the path of the image you want to analyze: ")
    result = predict_image(image_path, model)  # Predict whether the image has a tumor or not
    print(result)  # Print the result

