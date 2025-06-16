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

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Use the exact same HOG extraction as the 94% version
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    fd, _ = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        visualize=True, 
        block_norm='L2-Hys'
    )
    return fd

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    print("🔁 Training model with 94% settings...")
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

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test) * 100
    print(f"✅ Accuracy: {acc:.2f}%")
    print("📊 Classification Report:\n", classification_report(y_test, clf.predict(X_test)))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("💾 Model and scaler saved.")

    return clf, scaler

def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("📂 Loading model and scaler from disk...")
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return clf, scaler
    else:
        return train_and_save_model()

def predict_image(image_path, model, scaler):
    features = extract_hog_features(image_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "Tumor Detected 🧠" if prediction[0] == 1 else "No Tumor Detected ✅"

# Load or train
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











# import os
# import numpy as np
# import cv2
# import joblib
# from flask import Flask, request, jsonify, render_template
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix
# from skimage.feature import hog
# from imutils import paths
# from werkzeug.utils import secure_filename

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# MODEL_PATH = 'model.pkl'
# SCALER_PATH = 'scaler.pkl'

# app = Flask(__name__, static_folder='static', template_folder='templates')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_hog_features(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not load image from path: {image_path}")
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (256, 256))

#     # Apply denoising
#     gray = cv2.fastNlMeansDenoising(gray, None, h=10)

#     # Enhance contrast
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)

#     # Improved HOG parameters
#     fd, _ = hog(
#         gray,
#         orientations=12,
#         pixels_per_cell=(8, 8),
#         cells_per_block=(3, 3),
#         visualize=True,
#         block_norm='L2-Hys'
#     )
#     return fd

# def prepare_data(data_path):
#     image_paths = list(paths.list_images(data_path))
#     if len(image_paths) == 0:
#         raise ValueError(f"No images found in directory: {data_path}")
#     data, labels = [], []
#     for image_path in image_paths:
#         features = extract_hog_features(image_path)
#         data.append(features)
#         label = 1 if "yes_t" in image_path.lower() else 0
#         labels.append(label)
#     return np.array(data), np.array(labels)

# def train_and_save_model():
#     print("🔁 Training model...")
#     try:
#         no_t_data, no_t_labels = prepare_data('no_t')
#         yes_t_data, yes_t_labels = prepare_data('yes_t')
#     except ValueError as e:
#         print(e)
#         return None, None

#     # Combine datasets
#     data = np.vstack([no_t_data, yes_t_data])
#     labels = np.hstack([no_t_labels, yes_t_labels])

#     # Shuffle and scale
#     scaler = StandardScaler()
#     data = scaler.fit_transform(data)

#     X_train, X_test, y_train, y_test = train_test_split(
#         data, labels, test_size=0.2, stratify=labels, random_state=42
#     )

#     # Updated model parameters
#     clf = RandomForestClassifier(
#         n_estimators=600,
#         max_depth=40,
#         min_samples_split=3,
#         min_samples_leaf=1,
#         class_weight='balanced',
#         random_state=42
#     )
#     clf.fit(X_train, y_train)

#     # Evaluation
#     accuracy = clf.score(X_test, y_test) * 100
#     print(f"✅ Accuracy: {accuracy:.2f}%")
#     y_pred = clf.predict(X_test)
#     print("📊 Classification Report:\n", classification_report(y_test, y_pred))
#     print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#     # Save model and scaler
#     joblib.dump(clf, MODEL_PATH)
#     joblib.dump(scaler, SCALER_PATH)
#     print("💾 Model and scaler saved.")

#     return clf, scaler

# def load_model_and_scaler():
#     if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
#         clf = joblib.load(MODEL_PATH)
#         scaler = joblib.load(SCALER_PATH)
#         print("✅ Model and scaler loaded from disk.")
#         return clf, scaler
#     else:
#         return train_and_save_model()

# def predict_image(image_path, model, scaler):
#     features = extract_hog_features(image_path)
#     features_scaled = scaler.transform([features])
#     prediction = model.predict(features_scaled)
#     return "Tumor Detected 🧠" if prediction[0] == 1 else "No Tumor Detected ✅"

# # Load or train model
# model, scaler = load_model_and_scaler()

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"result": "No file part"}), 400

#     file = request.files["image"]
#     if file.filename == "":
#         return jsonify({"result": "No selected file"}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(filepath)

#         try:
#             result = predict_image(filepath, model, scaler)
#             return jsonify({"result": result})
#         except Exception as e:
#             return jsonify({"result": f"❌ Error: {str(e)}"}), 500
#     else:
#         return jsonify({"result": "❌ Invalid file type"}), 400

# if __name__ == "__main__":
#     app.run(debug=True)

# IMP CODE 




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