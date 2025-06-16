
# import os
# import numpy as np
# import cv2
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from skimage.feature import hog
# from imutils import paths

# # Function to extract HOG features from an image
# def extract_hog_features(image_path):
#     image = cv2.imread(image_path)  # Read the image
    
#     # Check if the image was loaded successfully
#     if image is None:
#         raise ValueError(f"Could not load image from path: {image_path}. Please check the path and ensure the file exists.")
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
#     gray = cv2.resize(gray, (128, 128))  # Resize the image for consistency
#     fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)  # Extract HOG features
#     return fd

# # Function to prepare the dataset
# def prepare_data(data_path):
#     image_paths = list(paths.list_images(data_path))  # List all image paths in the directory
    
#     if len(image_paths) == 0:
#         raise ValueError(f"No images found in the directory: {data_path}. Please check the directory.")
    
#     data = []
#     labels = []
    
#     for image_path in image_paths:
#         features = extract_hog_features(image_path)  # Extract HOG features from each image
#         data.append(features)
        
#         # Assign label based on the folder name
#         label = 1 if "yes_t" in image_path else 0
#         labels.append(label)
    
#     return np.array(data), np.array(labels)

# # Function to train the KNN classifier
# def train_classifier():
#     try:
#         # Updated paths for the no_t and yes_t directories
#         no_t_data, no_t_labels = prepare_data('no_t')  # Corrected path for no_t images
#         yes_t_data, yes_t_labels = prepare_data('yes_t')  # Corrected path for yes_t images
#     except ValueError as e:
#         print(e)
#         return None  # Return None if there was an issue loading the data
    
#     data = np.vstack([no_t_data, yes_t_data])  # Combine the data from both classes
#     labels = np.hstack([no_t_labels, yes_t_labels])  # Combine the labels from both classes
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
#     # Initialize KNN classifier and train it
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(X_train, y_train)
    
#     print("Accuracy on test set:", knn.score(X_test, y_test))  # Evaluate on the test set
    
#     return knn

# # Function to predict if the image has a tumor or not
# def predict_image(image_path, model):
#     try:
#         features = extract_hog_features(image_path)  # Extract features from the input image
#         prediction = model.predict([features])  # Predict using the trained model
#         return "Tumor Detected" if prediction == 1 else "No Tumor Detected"
#     except ValueError as e:
#         return str(e)  # Return the error message if the image can't be loaded

# # Train the classifier
# model = train_classifier()

# if model is not None:
#     # Ask the user for the image path and analyze it
#     image_path = input("Please enter the path of the image you want to analyze: ")
#     result = predict_image(image_path, model)  # Predict whether the image has a tumor or not
#     print(result)  # Print the result





































# import os
# import numpy as np
# import cv2
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from skimage.feature import hog
# from imutils import paths

# # Function to extract HOG features from an image
# def extract_hog_features(image_path):
#     image = cv2.imread(image_path)

#     if image is None:
#         raise ValueError(f"Could not load image from path: {image_path}. Please check the path and ensure the file exists.")
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (128, 128))  # or try (256, 256) for better detail

#     # Improved HOG settings: smaller cells = finer details
#     fd, _ = hog(
#         gray, 
#         orientations=9, 
#         pixels_per_cell=(8, 8), 
#         cells_per_block=(2, 2), 
#         visualize=True,
#         block_norm='L2-Hys'
#     )
#     return fd

# # Function to prepare the dataset
# def prepare_data(data_path):
#     image_paths = list(paths.list_images(data_path))
    
#     if len(image_paths) == 0:
#         raise ValueError(f"No images found in the directory: {data_path}. Please check the directory.")
    
#     data = []
#     labels = []

#     for image_path in image_paths:
#         features = extract_hog_features(image_path)
#         data.append(features)

#         label = 1 if "yes_t" in image_path else 0
#         labels.append(label)
    
#     return np.array(data), np.array(labels)

# # Function to train the classifier
# def train_classifier():
#     try:
#         no_t_data, no_t_labels = prepare_data('no_t')
#         yes_t_data, yes_t_labels = prepare_data('yes_t')
#     except ValueError as e:
#         print(e)
#         return None, None

#     data = np.vstack([no_t_data, yes_t_data])
#     labels = np.hstack([no_t_labels, yes_t_labels])

#     # Normalize the features
#     scaler = StandardScaler()
#     data = scaler.fit_transform(data)

#     # Split into training/testing
#     X_train, X_test, y_train, y_test = train_test_split(
#         data, labels, test_size=0.2, random_state=42
#     )

#     # Use Random Forest for better accuracy
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)

#     print("✅ Accuracy on test set:", clf.score(X_test, y_test))

#     return clf, scaler

# # Prediction function
# def predict_image(image_path, model, scaler):
#     try:
#         features = extract_hog_features(image_path)
#         features = scaler.transform([features])  # Apply same normalization
#         prediction = model.predict(features)
#         return "Tumor Detected" if prediction[0] == 1 else "No Tumor Detected"
#     except ValueError as e:
#         return str(e)

# # Train and run
# model, scaler = train_classifier()

# if model is not None:
#     image_path = input("Please enter the path of the image you want to analyze: ")
#     result = predict_image(image_path, model, scaler)
#     print(result)
















# import os
# import numpy as np
# import cv2
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from skimage.feature import hog
# from imutils import paths

# # Extract HOG features from image
# def extract_hog_features(image_path):
#     image = cv2.imread(image_path)

#     if image is None:
#         raise ValueError(f"Could not load image from path: {image_path}")
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (256, 256))  # Increased size

#     # Apply CLAHE
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)

#     # Enhanced HOG settings
#     fd, _ = hog(
#         gray,
#         orientations=12,
#         pixels_per_cell=(4, 4),
#         cells_per_block=(2, 2),
#         visualize=True,
#         block_norm='L2-Hys'
#     )
#     return fd

# # Load data from directory
# def prepare_data(data_path):
#     image_paths = list(paths.list_images(data_path))
    
#     if len(image_paths) == 0:
#         raise ValueError(f"No images found in directory: {data_path}")
    
#     data = []
#     labels = []

#     for image_path in image_paths:
#         features = extract_hog_features(image_path)
#         data.append(features)

#         label = 1 if "yes_t" in image_path.lower() else 0
#         labels.append(label)
    
#     return np.array(data), np.array(labels)

# # Train model
# def train_classifier():
#     try:
#         no_t_data, no_t_labels = prepare_data('no_t')
#         yes_t_data, yes_t_labels = prepare_data('yes_t')
#     except ValueError as e:
#         print(e)
#         return None, None

#     data = np.vstack([no_t_data, yes_t_data])
#     labels = np.hstack([no_t_labels, yes_t_labels])

#     scaler = StandardScaler()
#     data = scaler.fit_transform(data)

#     X_train, X_test, y_train, y_test = train_test_split(
#         data, labels, test_size=0.2, stratify=labels, random_state=42
#     )

#     clf = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=25,
#         class_weight='balanced',
#         random_state=42
#     )
#     clf.fit(X_train, y_train)

#     accuracy = clf.score(X_test, y_test)
#     print(f"✅ Accuracy on test set: {accuracy*100:.2f}%")

#     # Optional detailed report
#     y_pred = clf.predict(X_test)
#     print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

#     return clf, scaler

# # Prediction
# def predict_image(image_path, model, scaler):
#     try:
#         features = extract_hog_features(image_path)
#         features = scaler.transform([features])
#         prediction = model.predict(features)
#         return "Tumor Detected 🧠" if prediction[0] == 1 else "No Tumor Detected ✅"
#     except ValueError as e:
#         return str(e)

# # Main
# model, scaler = train_classifier()

# if model is not None:
#     image_path = input("🔍 Enter image path to predict: ")
#     result = predict_image(image_path, model, scaler)
#     print("📌 Prediction:", result)







# IMP CODE 




import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from skimage.feature import hog
from imutils import paths
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    fd, _ = hog(gray, orientations=12, pixels_per_cell=(4, 4),
                cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return fd

def prepare_data(data_path):
    image_paths = list(paths.list_images(data_path))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory: {data_path}")
    data, labels = [], []
    for image_path in image_paths:
        features = extract_hog_features(image_path)
        data.append(features)
        label = 1 if "yes_t" in image_path.lower() else 0
        labels.append(label)
    return np.array(data), np.array(labels)

def train_and_save_model():
    print("🔁 Training model...")
    try:
        no_t_data, no_t_labels = prepare_data('no_t')
        yes_t_data, yes_t_labels = prepare_data('yes_t')
    except ValueError as e:
        print(e)
        return None, None

    data = np.vstack([no_t_data, yes_t_data])
    labels = np.hstack([no_t_labels, yes_t_labels])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, max_depth=25, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    print(f"✅ Accuracy: {clf.score(X_test, y_test) * 100:.2f}%")
    print("📊 Classification Report:\n", classification_report(y_test, clf.predict(X_test)))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("💾 Model and scaler saved.")
    return clf, scaler

def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and scaler loaded from disk.")
        return clf, scaler
    else:
        return train_and_save_model()

def predict_image(image_path, model, scaler):
    features = extract_hog_features(image_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "Tumor Detected 🧠" if prediction[0] == 1 else "No Tumor Detected ✅"

model, scaler = load_model_and_scaler()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"result": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"result": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            result = predict_image(filepath, model, scaler)
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"result": f"❌ Error: {str(e)}"}), 500
    else:
        return jsonify({"result": "❌ Invalid file type"}), 400

if __name__ == "__main__":
    app.run(debug=True)












