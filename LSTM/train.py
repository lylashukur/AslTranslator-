import pandas as pd
import numpy as np
import tensorflow as tf
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Sequential = tf.keras.models.Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("asl_landmarks.csv")

# Separate features (X) and labels (y)
X = df.iloc[:, :-1].values  # Hand landmark data
y = df.iloc[:, -1].values   # ASL letter labels (0-25)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts letters into numerical labels

X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model.
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(set(y)), activation='softmax')  # Output layer for classification
])

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lstm_model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

loss, acc = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Model Accuracy: {acc * 100:.2f}%")

lstm_model.save("asl_lstm_model.h5")
print("Model saved as 'asl_lstm_model.h5'")
