import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# Load trained CNN model
model = tf.keras.models.load_model("asl_cnn_model.keras")  # or .h5 if that's what you saved

# Load class names from the training folder structure
data_dir = "data/asl_alphabet_train"  # Adjust if your dataset folder is elsewhere
class_names = sorted([folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))])

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)
print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # For now, use the whole frame (or crop later)
        hand_img = cv2.resize(frame, (64, 64))  # CNN input shape
        hand_img = hand_img.astype("float32") / 255.0
        hand_img = np.expand_dims(hand_img, axis=0)

        # Predict
        prediction = model.predict(hand_img)
        predicted_label = np.argmax(prediction)
        predicted_letter = class_names[predicted_label]

        # Display prediction
        cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("ASL Translator (CNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
