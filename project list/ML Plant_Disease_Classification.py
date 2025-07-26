# ===============================================================
# 🌿 Plant Disease Classification using Traditional ML (scikit-learn)
# 📦 For Small Image Datasets (< 1000 samples)
# ===============================================================

# What This ML Template Includes:
# For small image datasets (e.g., < 1000 images total)
# Preprocessing images: grayscale, resized, flattened
# Label encoding and train-test split
# Model: SVM (SVC) — but easily swappable with:
# KNeighborsClassifier (KNN)
# RandomForestClassifier
# LogisticRegression
# NaiveBayes

# Evaluation with confusion matrix and classification report
# Optional visualization of predictions

# 📌 When to Use This Instead of DL
# Use this ML version when:
# You have very few labeled images
# You want fast training without a GPU
# You need interpretability or simple experimentation

# Let me know if you’d like:
# A Jupyter Notebook (.ipynb) version
# A version using image features like HOG instead of raw pixels
# Or a comparative analysis of ML vs DL performance on same dataset 


# 1. 📦 Setup & Imports
# ---------------------
# !pip install scikit-learn opencv-python matplotlib

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC  # You can swap with KNN, RandomForest, etc.

# 2. 🗂️ Load & Preprocess Image Data
# -----------------------------------
# Dataset folder structure: Plant/<class_name>/<image_files>
DATA_DIR = "Plant"
IMAGE_SIZE = 64  # Smaller size speeds up training
data = []
labels = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    for image_name in os.listdir(class_path):
        try:
            image_path = os.path.join(class_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            data.append(img.flatten())  # Flatten 2D image to 1D feature vector
            labels.append(class_name)
        except:
            print(f"Failed to process {image_path}")

data = np.array(data)
labels = np.array(labels)

print("Dataset shape:", data.shape)
print("Labels shape:", labels.shape)

# 3. 🧼 Encode Labels
# -------------------
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# 4. 🔀 Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# 5. 🤖 Modeling with ML Algorithm
# --------------------------------
# You can try: SVC, KNeighborsClassifier, RandomForestClassifier, etc.
model = SVC(kernel='linear', C=1.0)  # Good for small datasets
model.fit(X_train, y_train)

# 6. 🧪 Evaluation
# ----------------
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. 📊 (Optional) Visualize Some Predictions
# -------------------------------------------
plt.figure(figsize=(10, 5))
for i in range(5):
    img = X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
    plt.subplot(1, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {encoder.inverse_transform([y_pred[i]])[0]}")
    plt.axis('off')
plt.show()
