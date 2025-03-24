import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained LSTM model
model = tf.keras.models.load_model("asl_lstm_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ASL letter mapping
asl_classes = {i: chr(65 + i) for i in range(26)}  # Map 0-25 to 'A'-'Z'

# Initialize video capture
cap = cv2.VideoCapture(0)

print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert frame to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on hand
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks (x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Reshape input for LSTM model
            X_input = np.array(landmarks).reshape(1, 1, len(landmarks))

            # Predict the ASL letter
            prediction = model.predict(X_input)
            predicted_label = np.argmax(prediction)  # Get the class with highest probability
            predicted_letter = asl_classes.get(predicted_label, "?")  # Convert to ASL letter

            # Display prediction on screen
            cv2.putText(image, f"Predicted: {predicted_letter}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Show the output
    cv2.imshow("ASL Translator", image)

    # Quit when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()