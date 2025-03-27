
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time
from collections import deque

class ASLExtractionCNN:
    def __init__(self, model_path="models/asl_cnn_model_20250327-012942.h5", confidence_threshold=0.7, history_size=5):
        """
        Initialize the ASL Extraction CNN for real-time sign language recognition.
        
        Args:
            model_path: Path to the trained model file
            confidence_threshold: Minimum confidence level to display a prediction
            history_size: Number of frames to consider for prediction smoothing
        """
        print("Initializing ASL Extraction CNN...")
        
        # Load the trained CNN model
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            self.model.summary()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get input shape expected by the model
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        print(f"Model expects input shape: {self.input_shape}")
        
        # Load class names from the training folder structure
        self.class_names = self._load_class_names()
        print(f"Loaded {len(self.class_names)} classes: {self.class_names}")
        
        # Initialize MediaPipe for hand detection and tracking
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create more detailed hand landmark drawing specs
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=2)
        
        # Initialize hand detection with improved settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,         # For video processing
            max_num_hands=1,                 # Focus on one hand for better performance
            model_complexity=1,              # Medium complexity (0=Lite, 1=Full)
            min_detection_confidence=0.5,    # Lower threshold to detect more hands
            min_tracking_confidence=0.5      # Lower for better tracking continuity
        )
        
        # Parameters
        self.confidence_threshold = confidence_threshold
        self.history_size = history_size
        self.prediction_history = deque(maxlen=history_size)  # For smoothing predictions
        self.confidence_history = deque(maxlen=history_size)  # For smoothing confidences
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)  # Store last 30 frame processing times
        self.hand_detected = False
        
        print("ASL Extraction CNN initialized and ready")

    def _load_class_names(self):
        """
        Load class names from different potential sources.
        Tries multiple methods to ensure compatibility.
        """
        # First try: Check if dataset directory exists and load class names
        data_dir = "data/asl_alphabet_train"
        if os.path.exists(data_dir):
            try:
                class_names = sorted([folder for folder in os.listdir(data_dir) 
                                    if os.path.isdir(os.path.join(data_dir, folder))])
                if class_names:
                    return class_names
            except Exception as e:
                print(f"Couldn't load class names from directory: {e}")
        
        # Second try: Based on model output size
        try:
            num_classes = self.model.output_shape[1]
            # For common ASL datasets (e.g., 24 letters A-Y, excluding J and Z which require motion)
            if num_classes == 24:
                return [chr(65 + i) for i in range(26) if i != 9 and i != 25]  # A-Y without J and Z
            elif num_classes == 26:
                return [chr(65 + i) for i in range(26)]  # A-Z
            elif num_classes == 29:
                # A-Z plus "del", "nothing", "space"
                special_classes = ["del", "nothing", "space"]
                return [chr(65 + i) for i in range(26)] + special_classes
            else:
                # Generic numerical classes
                return [str(i) for i in range(num_classes)]
        except Exception as e:
            print(f"Couldn't determine class names from model: {e}")
            
        # Fallback: Just use numbers as class identifiers
        return [str(i) for i in range(self.model.output_shape[1])]

    def preprocess_frame(self, frame):
        
        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Initialize return values
        processed_img = None
        hand_box = None
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            self.hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            
            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.landmark_drawing_spec,
                connection_drawing_spec=self.connection_drawing_spec
            )
            
            # Calculate hand bounding box with margin
            h, w, _ = frame.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Add margin (20% of hand size)
            margin_x = int((x_max - x_min) * 0.2)
            margin_y = int((y_max - y_min) * 0.2)
            
            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(w, x_max + margin_x)
            y_max = min(h, y_max + margin_y)
            
            # Store bounding box
            hand_box = (x_min, y_min, x_max, y_max)
            
            # Create square bounding box (required for aspect ratio consistency)
            box_size = max(x_max - x_min, y_max - y_min)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            
            x_min = max(0, x_center - box_size // 2)
            y_min = max(0, y_center - box_size // 2)
            x_max = min(w, x_center + box_size // 2)
            y_max = min(h, y_center + box_size // 2)
            
            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max].copy()
            
            # Process the hand image for the model
            if hand_img.size > 0:
                # Convert to grayscale (matching training data)
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                
                # Apply preprocessing to match training data characteristics
                # 1. Adjust contrast to enhance hand features
                hand_img = cv2.equalizeHist(hand_img)
                
                # 2. Apply Gaussian blur to reduce noise (similar to what model learned)
                hand_img = cv2.GaussianBlur(hand_img, (3, 3), 0)
                
                # 3. Apply thresholding to make it more like training dataset
                _, hand_img = cv2.threshold(hand_img, 0, 255, 
                                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 4. Resize to model's expected input size
                hand_img = cv2.resize(hand_img, self.input_shape)
                
                # 5. Normalize pixel values to [0,1] exactly as in training
                processed_img = hand_img.astype("float32") / 255.0
                
                # 6. Add channel and batch dimensions
                processed_img = np.expand_dims(processed_img, axis=-1)  # Add channel dimension
                processed_img = np.expand_dims(processed_img, axis=0)   # Add batch dimension
            
            return processed_img, hand_landmarks, hand_box
        self.hand_detected = False
        return None, None, None

    def predict(self, processed_img):
        """
        Make a prediction using the CNN model.
        
        Args:
            processed_img: Preprocessed image ready for model input
            
        Returns:
            pred_class: Predicted class
            pred_conf: Prediction confidence
            raw_prediction: Raw model output
        """
        if processed_img is None:
            return None, 0, None
        
        # Make prediction
        try:
            prediction = self.model.predict(processed_img, verbose=0)
            pred_idx = np.argmax(prediction[0])
            pred_conf = prediction[0][pred_idx]
            
            # Add to history for smoothing
            self.prediction_history.append(pred_idx)
            self.confidence_history.append(pred_conf)
            
            # Compute most frequent prediction in history
            from collections import Counter
            pred_counts = Counter(self.prediction_history)
            smooth_pred_idx, _ = pred_counts.most_common(1)[0]
            
            # Get average confidence for the predicted class
            avg_conf = np.mean([
                self.confidence_history[i] 
                for i, pred in enumerate(self.prediction_history) 
                if pred == smooth_pred_idx
            ])
            
            # Get class name
            if 0 <= smooth_pred_idx < len(self.class_names):
                pred_class = self.class_names[smooth_pred_idx]
            else:
                pred_class = str(smooth_pred_idx)
            
            return pred_class, avg_conf, prediction[0]
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0, None

    def run(self, camera_index=0, flip_image=True, window_name="ASL Extraction CNN"):
        """
        Run real-time ASL recognition using webcam.
        
        Args:
            camera_index: Index of the camera to use
            flip_image: Whether to flip the image horizontally (mirror mode)
            window_name: Name of the display window
        """
        print(f"Starting video capture on camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Press 'q' to quit, 's' to save a screenshot.")
        
        # Variables for FPS calculation
        prev_frame_time = 0
        fps = 0
        
        # Main loop
        try:
            while True:
                # Measure start time for FPS calculation
                current_frame_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Flip frame horizontally if requested (mirror mode)
                if flip_image:
                    frame = cv2.flip(frame, 1)
                
                # Process the frame
                start_process = time.time()
                processed_img, hand_landmarks, hand_box = self.preprocess_frame(frame)
                
                # Make prediction if hand is detected
                pred_class, confidence, raw_prediction = None, 0, None
                if processed_img is not None:
                    pred_class, confidence, raw_prediction = self.predict(processed_img)
                
                # Calculate processing time
                process_time = time.time() - start_process
                self.frame_times.append(process_time)
                avg_process_time = np.mean(self.frame_times) * 1000  # Convert to ms
                
                # Calculate FPS
                fps = 1 / (current_frame_time - prev_frame_time) if (current_frame_time - prev_frame_time) > 0 else 0
                prev_frame_time = current_frame_time
                
                # Display information on frame
                self._display_info(frame, pred_class, confidence, fps, avg_process_time, hand_box)
                
                # Show the frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"asl_screenshot_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved to {screenshot_path}")
        
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("Video capture ended")

    def _display_info(self, frame, pred_class, confidence, fps, process_time, hand_box):
        """
        Display information on the frame including prediction, confidence, and performance metrics.
        
        Args:
            frame: The current video frame
            pred_class: Predicted class
            confidence: Prediction confidence
            fps: Current frames per second
            process_time: Time taken to process frame (ms)
            hand_box: Hand bounding box (x_min, y_min, x_max, y_max)
        """
        h, w, _ = frame.shape
        
        # Draw hand bounding box if detected
        if hand_box:
            x_min, y_min, x_max, y_max = hand_box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display prediction if confidence is high enough
        if pred_class and confidence >= self.confidence_threshold:
            # Display the letter in large, clear text
            cv2.putText(frame, pred_class, (w//2 - 50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 255, 0), 8)
    # Main execution
if __name__ == "__main__":
    # Create and run the ASL Extraction CNN
    asl_cnn = ASLExtractionCNN(
        model_path="asl_cnn_model.h5",  # Use your trained model
        confidence_threshold=0.7,       # Only show predictions with confidence >= 70%
        history_size=5                  # Smooth predictions over 5 frames
    )
    
    # Run real-time recognition
    asl_cnn.run(camera_index=0, flip_image=True)