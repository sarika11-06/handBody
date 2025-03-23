# Don't need modification here

import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands and Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3)

DATA_DIR = './data'
OUTPUT_DIR = './dataset'  # Folder to store pickle files

# Create dataset directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Categories for Hands and Body
CATEGORIES = ['hands', 'body']

# Initialize dataset storage
data_hands, labels_hands = [], []
data_body, labels_body = [], []

for category in CATEGORIES:
    category_path = os.path.join(DATA_DIR, category)

    if not os.path.exists(category_path):
        print(f"⚠️ Warning: {category_path} not found! Skipping...")
        continue

    # Fetch class labels dynamically (based on directory names)
    class_labels = {name: i for i, name in enumerate(sorted(os.listdir(category_path)))}

    for class_name, class_index in class_labels.items():
        class_path = os.path.join(category_path, class_name)

        for img_path in os.listdir(class_path):
            data_aux = []
            x_coords, y_coords = [], []  # To store landmark positions

            img = cv2.imread(os.path.join(class_path, img_path))

            if img is None:
                print(f"❌ Error loading image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process Hands or Body based on category
            if category == 'hands':
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            x_coords.append(lm.x)
                            y_coords.append(lm.y)

                    # Normalize landmarks
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_coords))
                        data_aux.append(lm.y - min(y_coords))

                    data_hands.append(data_aux)
                    labels_hands.append(class_index)

            elif category == 'body':
                results = pose.process(img_rgb)
                if results.pose_landmarks:
                    for lm in results.pose_landmarks.landmark:
                        x_coords.append(lm.x)
                        y_coords.append(lm.y)

                    # Normalize landmarks
                    for lm in results.pose_landmarks.landmark:
                        data_aux.append(lm.x - min(x_coords))
                        data_aux.append(lm.y - min(y_coords))

                    data_body.append(data_aux)
                    labels_body.append(class_index)

# Save Hand dataset in OUTPUT_DIR
with open(os.path.join(OUTPUT_DIR, 'data_hands.pickle'), 'wb') as f:
    pickle.dump({'data': data_hands, 'labels': labels_hands}, f)

with open(os.path.join(OUTPUT_DIR, 'labels_hands.pickle'), 'wb') as f:
    pickle.dump(labels_hands, f)

# Save Body dataset in OUTPUT_DIR
with open(os.path.join(OUTPUT_DIR, 'data_body.pickle'), 'wb') as f:
    pickle.dump({'data': data_body, 'labels': labels_body}, f)

with open(os.path.join(OUTPUT_DIR, 'labels_body.pickle'), 'wb') as f:
    pickle.dump(labels_body, f)

print(f"✅ Hands and Body datasets saved in {OUTPUT_DIR} successfully!")
