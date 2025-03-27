


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os
import time

# Create timestamp for file naming
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Create directories for logs and models
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
log_dir = f"logs/fit-{timestamp}"

# Configuration parameters - easy to adjust
EPOCHS = 30          # Increased from 10 for better learning
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 28        # Keep original size to maintain compatibility
USE_AUGMENTATION = True

# Load and preprocess data - keeping your original structure
print("Loading dataset...")
path = "data/g_train.csv"
df = pd.read_csv(path)
print("Loaded Dataset:")
print(df.head())
print(f"Dataset shape: {df.shape}")

# Extract labels and images
labels = df['label'].values
images = df.drop('label', axis=1).values

# Show distribution of classes
unique_labels, counts = np.unique(labels, return_counts=True)
print("\nClass distribution:")
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

# Reshape the images from flat arrays to 28x28 arrays
images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32")

# Normalize pixel values to [0,1]
images /= 255.0

# Get number of classes
num_classes = np.max(labels) + 1
print(f"\nNumber of classes: {num_classes}")

# Display some sample images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Class: {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"sample_images_{timestamp}.png")
plt.show()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Create data generators with augmentation
if USE_AUGMENTATION:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Create generator for training data with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Create generator for validation data (no augmentation)
    val_datagen = ImageDataGenerator()
    
    # Configure batch-wise generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Show augmentation examples
    plt.figure(figsize=(12, 6))
    for i in range(9):
        # Get a batch with just one image
        img_batch, label_batch = train_generator.next()
        plt.subplot(3, 3, i+1)
        plt.imshow(img_batch[0].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        plt.title(f"Class: {int(label_batch[0])}")
        plt.axis('off')
    plt.suptitle("Augmented Images")
    plt.tight_layout()
    plt.savefig(f"augmented_samples_{timestamp}.png")
    plt.show()

# Define the enhanced model (keeping original structure but adding regularization)
def create_model():
    model = models.Sequential([
        # First convolutional block - same as your original
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),  # Added for training stability
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block - same as your original
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),  # Added for training stability
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block - added for better feature extraction
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation="relu"),  # Increased from 128
        layers.Dropout(0.5),                   # Added to prevent overfitting
        layers.Dense(num_classes, activation="softmax")
    ])
    
   # Replace with this:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Use an appropriate value, 0.001 is common
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Define a model variant that matches your original exactly
def create_original_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Use fixed learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
    )


    
    return model

# Create model - choose enhanced or original
print("\nCreating model...")
# model = create_original_model()  # Uncomment to use your original model
model = create_model()             # Enhanced model

# Display model summary
model.summary()

# Define callbacks for training
callbacks = [
    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    # Model checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"models/asl_model_best_{timestamp}.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Reduce learning rate
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    # Remove TensorBoard completely for now


]

# Train the model
print(f"\nTraining model for {EPOCHS} epochs...\n")

if USE_AUGMENTATION:
    # Train with data augmentation
    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
else:
    # Train without augmentation (original approach)
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'training_history_{timestamp}.png')
plt.show()

# Evaluate model on validation set
print("\nEvaluating model on validation set...\n")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation accuracy: {val_acc:.4f}")

# Get predictions for confusion matrix
y_pred = np.argmax(model.predict(X_val), axis=1)

# Create confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_val, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes), rotation=45)
plt.yticks(tick_marks, range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations to the matrix
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(f'confusion_matrix_{timestamp}.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save the model in both formats for compatibility
model.save(f"models/asl_cnn_model_{timestamp}.h5")
model.save(f"models/asl_cnn_model_{timestamp}.keras")
print(f"\nModel saved as 'models/asl_cnn_model_{timestamp}.h5' and '.keras'")

# Also save with the original filename for compatibility with your extraction code
model.save("asl_cnn_model.h5")
print("Model also saved as 'asl_cnn_model.h5' for compatibility")

print("\nTraining complete! Summary of results:")
print(f"- Final validation accuracy: {val_acc:.4f}")
print(f"- Training history saved to: training_history_{timestamp}.png")
print(f"- Confusion matrix saved to: confusion_matrix_{timestamp}.png")
print(f"- TensorBoard logs saved to: {log_dir}")

# To view TensorBoard logs, use:
print("\nTo view training details with TensorBoard, run:")
print(f"tensorboard --logdir={log_dir}")