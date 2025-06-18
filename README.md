This is a Flask-based web application for Brain Tumor Detection using HOG (Histogram of Oriented Gradients) features and a Random Forest Classifier, trained to achieve 94% accuracy. It provides an intuitive interface for users to upload MRI images and receive instant predictions.







🚀 Features
✅ High Accuracy (~94%) on test data

🧠 HOG Feature Extraction for MRI image analysis

🌲 Random Forest Classifier for classification

🖼 Upload image via interactive web interface

🔒 Secure file handling and input validation




📂 Project Structure
php
Copy
Edit
.
├── app.py                  # Main Flask backend
├── templates/
│   └── index.html          # Frontend UI
├── static/                 # CSS, demo GIF, etc.
├── uploads/                # Uploaded MRI images
├── model.pkl               # Trained Random Forest model
├── scaler.pkl              # StandardScaler for features
├── yes_t/, no_t/           # Dataset folders
└── README.md               # This file
