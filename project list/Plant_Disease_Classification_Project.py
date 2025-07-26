# ===============================================================
# 🪴 Plant Disease Detection - Deep Learning Template
# 🔁 Reusable Project Structure for Any Image Classification Task
# ===============================================================


# What It Includes:
# Real plant dataset structure ("Plant" folder with image subfolders)
# Environment setup (GPU config, TensorFlow, etc.)

# Standard DL pipeline:
# Image loading using image_dataset_from_directory
# Data splitting (train/val/test)
# Preprocessing (normalization, caching, shuffling)
# CNN modeling
# Training and evaluation
# Accuracy/loss plots
# Model saving

# You can reuse this file for any future image classification task by just:
# Changing the dataset path (DATA_DIR)


# Adjusting image size, batch size, or CNN layers as needed
# 📦 1. Setup & Environment
# -------------------------
# Install necessary libraries (run only once if not installed)
# !pip install tensorflow opencv-python matplotlib

# ✅ Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# (Optional) Configure GPU Memory Growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Check available GPU devices
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# 📁 2. Data Collection
# ---------------------
# Set the path to the Plant dataset directory (subfolders = classes)
DATA_DIR = "Plant"  # Change this for new projects

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20

# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Get class names from folder structure
class_names = dataset.class_names
print("Class labels:", class_names)

# 📊 3. Data Splitting (Train / Validation / Test)
# ------------------------------------------------
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size).take(val_size)
test_ds = dataset.skip(train_size + val_size)

# 🧹 4. Preprocessing
# --------------------
# Improve performance with prefetching and normalization
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(ds):
    return ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = preprocess(train_ds)
val_ds = preprocess(val_ds)
test_ds = preprocess(test_ds)

# 🧠 5. Modeling (CNN Architecture)
# ---------------------------------
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# 🛠 6. Compile the Model
# ------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 👟 7. Train the Model
# ----------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 🧪 8. Evaluate the Model
# -------------------------
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.2f}")

# 📈 9. Plot Accuracy & Loss
# ---------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss")
plt.show()

# 💾 10. Save the Model
# ----------------------
model.save("plant_disease_model.h5")
