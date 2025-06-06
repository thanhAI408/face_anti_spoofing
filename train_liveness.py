from model.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

# Initialize parameters
INIT_LR = 1e-4  # Learning rate
EPOCHS = 50
BS = 32  # Batch size
INPUT_SHAPE = (32, 32, 3)  # Input shape for model

# Load dataset
print("[INFO] loading images...")
data = []
labels = []
imagePaths = list(paths.list_images("dataset"))

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0  # Normalize images
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# Split dataset
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, 
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Initialize model
print("[INFO] compiling model...")
model = LivenessNet.build(width=32, height=32, depth=3, classes=2)
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("[INFO] training model...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), epochs=EPOCHS, verbose=1)

# Save model and label encoder
print("[INFO] saving model and label encoder...")
model.save("liveness.h5")
f = open("le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


