import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

current_label = None

# Initialize hands
mp_hands = mp.solutions.hands # Access MediaPipe's hand tracking model.
mp_drawing = mp.solutions.drawing_utils # Draw landmarks on the video feed.
hands = mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7) # Creates an instance of the hand-tracking model.

# Video Capture
capture = cv2.VideoCapture(0)

# Store data
landmarkCoord = []
letters = []

asl_classes = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, 
    "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11,
    "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, 
    "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, 
    "Y": 24, "Z": 25 }

# Video processing loop
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        continue

    # Convert Frame to RGB & Process Hand Tracking.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 hand landmarks (x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Store x, y, z

            if current_label is not None:
                landmarkCoord.append(landmarks)
                letters.append(current_label)


        cv2.imshow("ASL Data Collection", image)

        # Key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Start recording for a letter
            current_label = input("Enter ASL letter: ").upper()
            if current_label in asl_classes:
                current_label = asl_classes[current_label]
                print(f"Recording {current_label}...")
            else:
                print("Invalid letter. Try again.")
                current_label = None
        elif key == ord('q'):  # Quit
            break

capture.release()
cv2.destroyAllWindows()

# Convert to DataFrame and save to CSV
df = pd.DataFrame(landmarkCoord)
df['label'] = letters
df.to_csv("asl_landmarks.csv", index=False)
print("Dataset saved as 'asl_landmarks.csv'!")
print(f"Collected {len(landmarkCoord)} samples.")  


