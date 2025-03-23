# Don't need modification here

import pickle
import cv2
import mediapipe as mp
import numpy as np
# import os

# Load trained models
model_hands_dict = pickle.load(open('./models/model_hands.p', 'rb'))
model_hands = model_hands_dict['model']

model_body_dict = pickle.load(open('./models/model_body.p', 'rb'))
model_body = model_body_dict['model']

# Define hand gesture labels (Mudras)
labels_dict_hands = {
    0: 'Pataka', 1: 'Tripataka', 2: 'Ardhapataka', 3: 'Katarimukha', 4: 'Mayura', 
    5: 'Ardhachandra', 6: 'Arala', 7: 'Sukatunda', 8: 'Musti', 9: 'Sikhara', 
    10: 'Kapittha', 11: 'Katamukha', 12: 'Suchi', 13: 'Chandrakala', 14: 'Padmakosa', 
    15: 'Sarpashirsa', 16: 'Mrgasirsa', 17: 'Simhamukha', 18: 'Kangula', 19: 'Alapadma', 
    20: 'Catura', 21: 'Bhramara', 22: 'Hamsasya', 23: 'Hamsapaksa', 24: 'Sandamsa', 
    25: 'Mukula', 26: 'Tamracuda', 27: 'Trisula', 28: 'Anjali', 29: 'Kapota', 
    30: 'Karkata', 31: 'Svastika', 32: 'Dola', 33: 'Puspaputa', 34: 'Utsanga', 
    35: 'Sivalinga', 36: 'Katakavardhana', 37: 'Kartarisvastika', 38: 'Sakata', 
    39: 'Sankha', 40: 'Chakra', 41: 'Pasa', 42: 'Kilaka', 43: 'Matsya', 44: 'Kurma', 
    45: 'Varaha', 46: 'Garuda', 47: 'Nagabandha', 48: 'Katva', 49: 'Bherunda'
}

# Define body pose labels
labels_dict_body = {
    0: 'Samapada', 1: 'Ekapada', 2: 'Ardha Mandali', 3: 'Alidha', 4: 'Pratyalidha',
    5: 'Swastika', 6: 'Gaja Hasta', 7: 'Nata', 8: 'Valitoruka', 9: 'Anjanasana'
}

cap = cv2.VideoCapture(0)

# Initialize Mediapipe solutions for hands and pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

previous_prediction_hands = None
previous_prediction_body = None

while True:
    data_hands = []
    data_body = []

    x_hands, y_hands = [], []
    x_body, y_body = [], []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results_hands = hands.process(frame_rgb)
    # Process body pose landmarks
    results_pose = pose.process(frame_rgb)

    detected_hand = "No Hand Detected"
    detected_body = "No Body Detected"

    # Extract hand landmarks if detected
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x_hands.append(hand_landmarks.landmark[i].x)
                y_hands.append(hand_landmarks.landmark[i].y)

            min_x, min_y = min(x_hands), min(y_hands)

            for i in range(len(hand_landmarks.landmark)):
                data_hands.append(hand_landmarks.landmark[i].x - min_x)
                data_hands.append(hand_landmarks.landmark[i].y - min_y)

    # Extract body pose landmarks if detected
    if results_pose.pose_landmarks:
        for i, lm in enumerate(results_pose.pose_landmarks.landmark):
            x_body.append(lm.x)
            y_body.append(lm.y)

        min_x_body, min_y_body = min(x_body), min(y_body)

        for i in range(len(results_pose.pose_landmarks.landmark)):
            data_body.append(x_body[i] - min_x_body)
            data_body.append(y_body[i] - min_y_body)

        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Make predictions for hands
    if len(data_hands) == 42:  # 42 features (21 landmarks * 2 coordinates)
        prediction_hands = model_hands.predict([np.asarray(data_hands)])[0]
        detected_hand = labels_dict_hands.get(int(prediction_hands), "Unknown Hand Gesture")

        if detected_hand != previous_prediction_hands:
            previous_prediction_hands = detected_hand
            print(f"Hand Gesture Detected: {detected_hand}")

    # Make predictions for body
    if len(data_body) == 66:  # 66 features (33 landmarks * 2 coordinates)
        prediction_body = model_body.predict([np.asarray(data_body)])[0]
        detected_body = labels_dict_body.get(int(prediction_body), "Unknown Body Pose")

        if detected_body != previous_prediction_body:
            previous_prediction_body = detected_body
            print(f"Body Pose Detected: {detected_body}")

    # Display predictions
    cv2.putText(frame, f"Hand: {detected_hand}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Body: {detected_body}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame with hand/pose landmarks and predictions
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
