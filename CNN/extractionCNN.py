import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


# pre organized data MNIST CSV file
path = "data/sign_mnist_test 2.csv"

df = pd.read_csv(path)
print("Loaded Dataset:")
print(df.head)

labels = df['label'].values
images = df.drop('label', axis=1).values

#Reshape the images from flat arrays to 28x28 arrays
images = images.reshape(-1,28,28,1).astype("float32")

#normalize pixel values to [0,1]
images /= 255.0

#from tensorflow.keras import layers, models

num_classes = np.max(labels) + 1   # Expected to be 25 if motion-based 

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")  # Now this will be 25 neurons
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Split the data into training and validation sets (using a simple random split here)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_val, y_val))

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation accuracy:", val_acc)

# Save the model for later integration into a real-time app
model.save("asl_cnn_model.h5")

from tensorflow.keras.models import load_model
model = load_model("asl_cnn_model.h5")